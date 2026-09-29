"""The variable-resolution 2.5D grid engine.

Design
------
The map is a set of concentric rings around the sensor.  Ring *l* has
cell size ``base_res * 2**l``.  Because every resolution is a power-of-two
multiple of the base resolution **and** every ring radius is a whole
multiple of the coarsest cell, the rings form a quadtree slice: a coarse
cell is exactly tiled by 2x2 finer cells and no cell ever straddles a
ring boundary.  Consequences:

*   A world point maps to exactly one cell -- no double counting, no gaps.
*   ``lookup(x, y)`` is O(1) at any resolution, which makes cross-level
    neighbour queries (needed by the planner) exact rather than
    approximate.  This is the "alignment error / data loss" problem the
    brief warns about, solved structurally instead of by clamping.

Cells are stored as a struct-of-arrays with an int64 hash key
``(level, ix, iy)``.  Only occupied cells exist, so cost scales with
observed surface area, not with map extent.

Every aggregation below is a vectorised sort+reduceat over the whole
frame -- there is no Python loop over points or cells.

2.5D, not 2D
------------
Each cell decomposes its z-column into three layers:

    ground    z <= z_min + ground_band          -> surface elevation
    obstacle  ground_band < dz < clearance_h    -> things you hit
    overhead  dz >= clearance_h                 -> things you pass under

Keeping the overhead layer separate is what lets a 2.5D map represent a
bridge, a sign gantry or a low branch, which a plain elevation map
cannot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .config import (CLASS_IDS, CLASS_KIND, CLASS_TO_COMPACT, DRIVABLE_CLASSES,
                     DYNAMIC_CLASSES, KIND_DYNAMIC, KIND_STATIC, KIND_TERRAIN,
                     KIND_UNKNOWN, N_CLASSES, GridConfig, Level)
from .preprocess import CleanFrame

# Occupancy states
OCC_UNKNOWN, OCC_FREE, OCC_OCCUPIED = 0, 1, 2

_IDX_OFFSET = 1 << 24          # keeps ix/iy non-negative inside the key
_IDX_BITS = 25


def pack_key(level: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Pack (level, ix, iy) into one int64.  Bijective for |i| < 2**24."""
    return ((level.astype(np.int64) << (2 * _IDX_BITS))
            | ((ix.astype(np.int64) + _IDX_OFFSET) << _IDX_BITS)
            | (iy.astype(np.int64) + _IDX_OFFSET))


def _agg(cell_idx: np.ndarray, vals: np.ndarray, n_cells: int
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-cell (count, min, max, sum, sumsq) with a single sort."""
    cnt = np.bincount(cell_idx, minlength=n_cells).astype(np.int32)
    mn = np.full(n_cells, np.nan, np.float32)
    mx = np.full(n_cells, np.nan, np.float32)
    sm = np.zeros(n_cells, np.float64)
    sq = np.zeros(n_cells, np.float64)
    if cell_idx.size == 0:
        return cnt, mn, mx, sm, sq
    order = np.argsort(cell_idx, kind="stable")
    ci = cell_idx[order]
    v = vals[order].astype(np.float64)
    uniq, starts = np.unique(ci, return_index=True)
    mn[uniq] = np.minimum.reduceat(v, starts).astype(np.float32)
    mx[uniq] = np.maximum.reduceat(v, starts).astype(np.float32)
    sm[uniq] = np.add.reduceat(v, starts)
    sq[uniq] = np.add.reduceat(v * v, starts)
    return cnt, mn, mx, sm, sq


@dataclass
class FoveatedGrid:
    """Sparse, variable-resolution 2.5D semantic grid for one sweep."""
    cfg: GridConfig
    name: str = ""

    # --- geometry ---------------------------------------------------
    key: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    level: np.ndarray = field(default_factory=lambda: np.empty(0, np.int8))
    ix: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    iy: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    cx: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    cy: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    res: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))

    # --- 2.5D layers ------------------------------------------------
    n_pts: np.ndarray = field(default_factory=lambda: np.empty(0, np.int32))
    z_ground: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    z_min: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    z_max: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    roughness: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    obstacle_top: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    overhead_min: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    intensity: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))

    # --- semantics --------------------------------------------------
    class_hist: np.ndarray = field(default_factory=lambda: np.empty((0, N_CLASSES), np.int32))
    class_id: np.ndarray = field(default_factory=lambda: np.empty(0, np.int16))
    class_conf: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    kind: np.ndarray = field(default_factory=lambda: np.empty(0, np.int8))

    # --- derived layers (filled by semantics.py) ---------------------
    occupancy: np.ndarray = field(default_factory=lambda: np.empty(0, np.int8))
    cost: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    drivable: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    step: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    curb: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    low_clearance: np.ndarray = field(default_factory=lambda: np.empty(0, bool))

    # --- bookkeeping -------------------------------------------------
    # Free space is stored as a polar visibility profile (n_azimuth_bins
    # floats) rather than as materialised empty cells.  Densifying free
    # space at 5 cm would cost ~500 k cells / 14 MB per sweep; the
    # profile costs 11 KB and is exact for a single sensor origin.
    free_range: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    seen_range: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))

    point_cell: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    _index: Optional[Dict[int, int]] = None
    timings: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return int(self.key.shape[0])

    @property
    def clearance(self) -> np.ndarray:
        """Vertical gap between the ground and the lowest overhead return."""
        return self.overhead_min - self.z_ground

    @property
    def obstacle_height(self) -> np.ndarray:
        return self.obstacle_top - self.z_ground

    # ------------------------------------------------------------------
    # Level / index mathematics
    # ------------------------------------------------------------------
    def ring_metric(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Distance used to pick a ring (see GridConfig.ring_metric)."""
        if self.cfg.ring_metric == "chebyshev":
            return np.maximum(np.abs(x), np.abs(y))
        return np.hypot(x, y)

    def level_of_d(self, d: np.ndarray) -> np.ndarray:
        """Vectorised ring lookup; -1 for anything outside the map."""
        radii = np.asarray(self.cfg.ring_radii, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        lv = np.searchsorted(radii, d, side="right").astype(np.int8)
        lv[d >= radii[-1]] = -1
        return lv

    def res_of_level(self, lv: np.ndarray) -> np.ndarray:
        """Resolution per level, in float64.

        float64 is not optional here.  0.05 stored as float32 is
        0.050000000745..., and dividing a 90 m coordinate by that lands
        on the wrong side of a cell edge often enough to disagree with
        any other code path that uses the exact literal.  Cell indices
        must be computed in one arithmetic, everywhere.
        """
        table = np.array([l.res for l in self.cfg.levels()], dtype=np.float64)
        return table[np.clip(lv, 0, len(table) - 1)]

    def lookup(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Map world positions to cell rows (-1 when the cell is empty).

        This is the cross-level query: the ring is chosen by range, then
        the index is computed at that ring's resolution.  Because the
        resolutions nest dyadically the answer is exact -- a query point
        one micrometre either side of a ring boundary lands in the two
        cells that genuinely tile that neighbourhood, with no overlap and
        no hole.
        """
        x = np.atleast_1d(np.asarray(x, np.float64))
        y = np.atleast_1d(np.asarray(y, np.float64))
        d = self.ring_metric(x, y)
        lv = self.level_of_d(d)
        res = self.res_of_level(lv).astype(np.float64)
        ix = np.floor(x / res).astype(np.int64)
        iy = np.floor(y / res).astype(np.int64)
        keys = pack_key(lv, ix, iy)
        out = self._lookup_keys(keys)
        out[lv < 0] = -1
        return out

    def _lookup_keys(self, keys: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(self.key, keys)
        pos = np.clip(pos, 0, max(self.n_cells - 1, 0))
        ok = (self.n_cells > 0) & (self.key[pos] == keys)
        return np.where(ok, pos, -1).astype(np.int64)

    # ------------------------------------------------------------------
    def nbytes(self) -> Dict[str, int]:
        """Actual resident bytes of the map, field by field."""
        fields = {
            "keys": self.key, "level": self.level, "ix": self.ix, "iy": self.iy,
            "n_pts": self.n_pts, "z_ground": self.z_ground,
            "z_min": self.z_min, "z_max": self.z_max,
            "roughness": self.roughness, "obstacle_top": self.obstacle_top,
            "overhead_min": self.overhead_min, "intensity": self.intensity,
            "class_id": self.class_id, "class_conf": self.class_conf,
            "kind": self.kind, "occupancy": self.occupancy,
            "cost": self.cost, "drivable": self.drivable,
            "free_profile": self.free_range, "seen_profile": self.seen_range,
        }
        out = {k: int(v.nbytes) for k, v in fields.items() if v is not None}
        out["total"] = int(sum(out.values()))
        return out


def build(frame: CleanFrame, cfg: GridConfig, keep_point_map: bool = True
          ) -> FoveatedGrid:
    """Project one cleaned sweep into a foveated 2.5D grid."""
    import time
    t_all = time.perf_counter()
    g = FoveatedGrid(cfg=cfg, name=frame.name)

    x = frame.xyz[:, 0].astype(np.float64)
    y = frame.xyz[:, 1].astype(np.float64)
    z = frame.xyz[:, 2].astype(np.float32)
    r = frame.rng.astype(np.float64)

    # ---- 1. ring assignment + cell indexing -------------------------
    t = time.perf_counter()
    d_ring = g.ring_metric(x, y)
    lv = g.level_of_d(d_ring)
    inside = lv >= 0
    if not inside.all():
        x, y, z, r = x[inside], y[inside], z[inside], r[inside]
        lv = lv[inside]
        frame_cls = frame.class_id[inside]
        frame_int = frame.intensity[inside]
        frame_az = frame.azimuth[inside]
    else:
        frame_cls, frame_int, frame_az = frame.class_id, frame.intensity, frame.azimuth

    res_pt = g.res_of_level(lv).astype(np.float64)
    ix = np.floor(x / res_pt).astype(np.int64)
    iy = np.floor(y / res_pt).astype(np.int64)
    keys = pack_key(lv, ix, iy)

    uniq_keys, inv = np.unique(keys, return_inverse=True)
    inv = inv.astype(np.int64)
    n_cells = uniq_keys.size
    g.timings["index"] = time.perf_counter() - t

    # ---- 2. decode cell geometry from the key -----------------------
    t = time.perf_counter()
    g.key = uniq_keys
    g.level = (uniq_keys >> (2 * _IDX_BITS)).astype(np.int8)
    g.ix = (((uniq_keys >> _IDX_BITS) & ((1 << _IDX_BITS) - 1)) - _IDX_OFFSET).astype(np.int32)
    g.iy = ((uniq_keys & ((1 << _IDX_BITS) - 1)) - _IDX_OFFSET).astype(np.int32)
    g.res = g.res_of_level(g.level).astype(np.float32)
    g.cx = ((g.ix + 0.5) * g.res).astype(np.float32)
    g.cy = ((g.iy + 0.5) * g.res).astype(np.float32)

    # ---- 3. full-column statistics ----------------------------------
    cnt, zmn, zmx, zsm, zsq = _agg(inv, z, n_cells)
    g.n_pts, g.z_min, g.z_max = cnt, zmn, zmx
    _, _, _, ism, _ = _agg(inv, frame_int, n_cells)
    g.intensity = (ism / np.maximum(cnt, 1)).astype(np.float32)
    g.timings["aggregate"] = time.perf_counter() - t

    # ---- 4. 2.5D layer decomposition --------------------------------
    t = time.perf_counter()
    dz = z - g.z_min[inv]
    is_ground = dz <= cfg.ground_band
    is_overhead = dz >= cfg.clearance_height
    is_obstacle = ~is_ground & ~is_overhead

    gc, _, _, gsm, gsq = _agg(inv[is_ground], z[is_ground], n_cells)
    gn = np.maximum(gc, 1)
    g.z_ground = (gsm / gn).astype(np.float32)
    var = np.maximum(gsq / gn - (gsm / gn) ** 2, 0.0)
    g.roughness = np.sqrt(var).astype(np.float32)
    g.z_ground[gc == 0] = g.z_min[gc == 0]

    _, _, obs_top, _, _ = _agg(inv[is_obstacle], z[is_obstacle], n_cells)
    g.obstacle_top = obs_top                       # NaN where no obstacle layer
    _, ovh_min, _, _, _ = _agg(inv[is_overhead], z[is_overhead], n_cells)
    g.overhead_min = ovh_min                       # NaN where nothing overhead
    g.timings["layers"] = time.perf_counter() - t

    # ---- 5. semantic histogram --------------------------------------
    t = time.perf_counter()
    lut = np.zeros(int(max(CLASS_IDS)) + 2, np.int64)
    for cid, compact in CLASS_TO_COMPACT.items():
        lut[cid] = compact
    cls_c = lut[np.clip(frame_cls, 0, len(lut) - 1)]
    hist = np.bincount(inv * N_CLASSES + cls_c,
                       minlength=n_cells * N_CLASSES).reshape(n_cells, N_CLASSES)
    g.class_hist = hist.astype(np.int32)
    dom = hist.argmax(axis=1)
    g.class_id = np.array(CLASS_IDS, np.int16)[dom]
    g.class_conf = (hist[np.arange(n_cells), dom] /
                    np.maximum(hist.sum(axis=1), 1)).astype(np.float32)
    kind_lut = np.zeros(int(max(CLASS_IDS)) + 2, np.int8)
    for cid, k in CLASS_KIND.items():
        kind_lut[cid] = k
    g.kind = kind_lut[g.class_id]
    g.timings["semantics"] = time.perf_counter() - t

    if keep_point_map:
        g.point_cell = inv
        g.meta["point_z"] = z
        g.meta["point_class"] = frame_cls
        g.meta["point_range"] = r.astype(np.float32)
        g.meta["point_az"] = frame_az
    g.meta["n_points_gridded"] = int(z.size)
    g.meta["frame_stats"] = frame.stats
    g.timings["total_build"] = time.perf_counter() - t_all
    return g


# ----------------------------------------------------------------------
# Free-space carving
# ----------------------------------------------------------------------
def _az_bin(az: np.ndarray, nb: int) -> np.ndarray:
    return np.clip(((az + np.pi) / (2 * np.pi) * nb).astype(np.int64), 0, nb - 1)


def carve_free_space(g: FoveatedGrid, frame: CleanFrame) -> None:
    """Mark cells FREE / OCCUPIED / UNKNOWN using ray evidence.

    A LiDAR return is evidence about the whole ray, not just its endpoint.
    For every azimuth bin we record

        r_block  = nearest *obstacle* return   -> everything nearer is free
        r_seen   = farthest return of any kind -> beyond it we know nothing

    Cells are then classified in polar space, which is O(cells) and needs
    no ray marching.
    """
    import time
    t = time.perf_counter()
    cfg = g.cfg
    nb = cfg.n_azimuth_bins
    az = g.meta.get("point_az")
    rng = g.meta.get("point_range")
    if az is None or rng is None:
        g.occupancy = np.full(g.n_cells, OCC_OCCUPIED, np.int8)
        return

    bin_of = _az_bin(az, nb)
    pt_cls = g.meta["point_class"]
    is_obstacle_pt = ~np.isin(pt_cls, list(DRIVABLE_CLASSES | {8}))

    r_block = np.full(nb, np.inf)
    np.minimum.at(r_block, bin_of[is_obstacle_pt], rng[is_obstacle_pt])
    r_seen = np.zeros(nb)
    np.maximum.at(r_seen, bin_of, rng)

    # Dilate by one bin: a ray that grazes a post should not leave a
    # pencil of "free" space straight through it.
    r_block = np.minimum(np.minimum(r_block, np.roll(r_block, 1)), np.roll(r_block, -1))

    g.free_range = np.where(np.isfinite(r_block), r_block, r_seen).astype(np.float32)
    g.seen_range = r_seen.astype(np.float32)

    # Existing cells all carry returns, so they are OCCUPIED by
    # definition; the interesting distinction (free vs unknown) lives in
    # the profile and is queried with is_free().
    g.occupancy = np.full(g.n_cells, OCC_OCCUPIED, np.int8)
    g.timings["freespace"] = time.perf_counter() - t


def is_free(g: FoveatedGrid, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """True where the sensor saw straight through -- known empty ground.

    Used by the planner so it can cross the gaps between LiDAR ring
    returns on a road surface without needing those cells to exist.
    """
    if g.free_range.size == 0:
        return np.zeros(np.shape(x), bool)
    x = np.atleast_1d(np.asarray(x, np.float64))
    y = np.atleast_1d(np.asarray(y, np.float64))
    r = np.hypot(x, y)
    b = _az_bin(np.arctan2(y, x), g.free_range.size)
    return (r < g.free_range[b] - g.cfg.free_margin) & (r <= g.seen_range[b])
