"""Derived map layers: terrain analysis, traversability, objects, tracks.

Everything in here consumes the raw foveated grid and produces the
layers a planner or a human actually looks at:

*   ``step`` / ``curb``          -- vertical discontinuity against neighbours
*   ``drivable`` / ``cost``      -- terrain analysis (task 1 of the brief)
*   ``low_clearance``            -- overhanging obstacle detection
*   ``instances`` / ``tracks``   -- object detection (task 2 of the brief)

Neighbour queries deliberately go through ``grid.lookup`` so they cross
ring boundaries.  A cell at the inner edge of the 20 cm ring finds its
10 cm neighbours correctly; that is the variable-resolution correctness
property being exercised, not assumed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import (CLASS_NAMES, DRIVABLE_CLASSES, DYNAMIC_CLASSES,
                     KIND_DYNAMIC, KIND_STATIC, KIND_TERRAIN, GridConfig)
from .grid import FoveatedGrid

_NEIGH = [(1, 0), (-1, 0), (0, 1), (0, -1)]
_NEIGH8 = _NEIGH + [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def neighbour_rows(g: FoveatedGrid, dx: int, dy: int) -> np.ndarray:
    """Row index of each cell's neighbour one cell away in (dx, dy).

    The offset is applied in *metres at the cell's own resolution*, then
    resolved through the global lookup, so the answer is correct even
    when the neighbour lives at a coarser or finer level.
    """
    nx = g.cx.astype(np.float64) + dx * g.res
    ny = g.cy.astype(np.float64) + dy * g.res
    return g.lookup(nx, ny)


def terrain_analysis(g: FoveatedGrid) -> None:
    """Fill step / curb / clearance / drivable / cost."""
    t = time.perf_counter()
    cfg = g.cfg
    n = g.n_cells

    # ---- vertical discontinuity against the 4-neighbourhood ----------
    step = np.zeros(n, np.float32)
    for dx, dy in _NEIGH:
        nb = neighbour_rows(g, dx, dy)
        ok = nb >= 0
        d = np.zeros(n, np.float32)
        d[ok] = np.abs(g.z_ground[ok] - g.z_ground[nb[ok]])
        step = np.maximum(step, d)
    g.step = step

    is_surface = np.isin(g.class_id, [7, 8])
    g.curb = is_surface & (step >= cfg.curb_min) & (step <= cfg.curb_max)

    # ---- overhead clearance -----------------------------------------
    clr = g.clearance
    g.low_clearance = np.isfinite(clr) & (clr < cfg.vehicle_height)

    # ---- obstacle occupancy in the cell ------------------------------
    obs_h = g.obstacle_height
    has_obstacle = np.isfinite(obs_h) & (obs_h > cfg.max_step)

    # ---- drivability --------------------------------------------------
    drivable = (
        np.isin(g.class_id, list(DRIVABLE_CLASSES))
        & ~has_obstacle
        & ~g.low_clearance
        & (step <= cfg.max_step)
        & (g.roughness <= cfg.max_roughness)
    )
    g.drivable = drivable

    # ---- continuous traversal cost (0 = ideal, >=1 = blocked) --------
    class_penalty = np.full(n, 1.0, np.float32)
    class_penalty[g.class_id == 7] = 0.0     # Road
    class_penalty[g.class_id == 8] = 0.55    # Terrain: passable, undesirable
    cost = (0.45 * np.clip(step / cfg.max_step, 0, 2)
            + 0.25 * np.clip(g.roughness / cfg.max_roughness, 0, 2)
            + class_penalty)
    cost[~drivable] = np.inf
    g.cost = cost.astype(np.float32)
    g.timings["terrain"] = time.perf_counter() - t


# ----------------------------------------------------------------------
# Object instances
# ----------------------------------------------------------------------
@dataclass
class Instance:
    iid: int
    class_id: int
    class_name: str
    kind: int
    n_cells: int
    n_points: int
    cx: float
    cy: float
    extent_x: float
    extent_y: float
    z_base: float
    z_top: float
    height: float
    range_m: float
    level: int

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (np.floating, float)) else
                    int(v) if isinstance(v, (np.integer, int)) else v)
                for k, v in self.__dict__.items()}


def extract_instances(g: FoveatedGrid, min_cells: int = 3,
                      instance_res: float = 0.75, bridge: int = 2
                      ) -> List[Instance]:
    """Segment non-terrain cells into object instances.

    Method: snap each object cell onto a coarse *instance lattice*
    (default 0.75 m), then take connected components of that lattice
    within a Chebyshev radius of ``bridge`` coarse cells, separately per
    class.  Labels are propagated back to the fine cells.

    Why not cluster the fine cells directly?  A spinning LiDAR paints an
    object as a stack of rings, and at 20 m the vertical gap between
    consecutive rings projects to well over a metre on the ground plane.
    Eight-connectivity on 10 cm cells therefore shatters one car into
    five fragments.  The coarse lattice bridges exactly those gaps while
    staying class-separated, so a pedestrian leaning on a guardrail is
    still two objects.

    Cost is O(occupied area), not O(points): clustering happens on cells,
    which is the payoff of gridding before detecting.
    """
    t = time.perf_counter()
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    obj = np.where(np.isin(g.kind, [KIND_STATIC, KIND_DYNAMIC]))[0]
    out: List[Instance] = []
    if obj.size == 0:
        g.timings["instances"] = time.perf_counter() - t
        return out

    # --- snap to the coarse lattice, keyed by (class, gx, gy) --------
    gx = np.floor(g.cx[obj] / instance_res).astype(np.int64)
    gy = np.floor(g.cy[obj] / instance_res).astype(np.int64)
    cls = g.class_id[obj].astype(np.int64)
    OFF, BITS = 1 << 20, 21
    ckey = (cls << (2 * BITS)) | ((gx + OFF) << BITS) | (gy + OFF)
    uk, inv = np.unique(ckey, return_inverse=True)
    m = uk.size
    ucls = (uk >> (2 * BITS)).astype(np.int64)
    ugx = (((uk >> BITS) & ((1 << BITS) - 1)) - OFF).astype(np.int64)
    ugy = ((uk & ((1 << BITS) - 1)) - OFF).astype(np.int64)

    # --- edges between coarse cells within the bridging radius -------
    rs, cs = [], []
    for dx in range(-bridge, bridge + 1):
        for dy in range(-bridge, bridge + 1):
            if dx == 0 and dy == 0:
                continue
            q = (ucls << (2 * BITS)) | ((ugx + dx + OFF) << BITS) | (ugy + dy + OFF)
            pos = np.searchsorted(uk, q)
            pos = np.clip(pos, 0, m - 1)
            hit = uk[pos] == q
            rs.append(np.where(hit)[0])
            cs.append(pos[hit])
    r_e = np.concatenate(rs) if rs else np.empty(0, np.int64)
    c_e = np.concatenate(cs) if cs else np.empty(0, np.int64)
    adj = coo_matrix((np.ones(r_e.size, np.int8), (r_e, c_e)), shape=(m, m))
    _, coarse_label = connected_components(adj, directed=False)

    label = coarse_label[inv]           # label per object cell
    order = np.argsort(label, kind="stable")
    lab_s = label[order]
    uniq_lab = np.unique(lab_s)
    first = np.searchsorted(lab_s, uniq_lab)
    last = np.searchsorted(lab_s, uniq_lab, side="right")

    iid = 0
    for k in range(uniq_lab.size):
        rows = obj[order[first[k]:last[k]]]
        if rows.size < min_cells:
            continue
        cid = int(g.class_id[rows[0]])
        zb = float(np.nanmin(g.z_ground[rows]))
        zt = float(np.nanmax(g.z_max[rows]))
        cx, cy = float(g.cx[rows].mean()), float(g.cy[rows].mean())
        res_m = float(g.res[rows].mean())
        out.append(Instance(
            iid=iid, class_id=cid, class_name=CLASS_NAMES.get(cid, "?"),
            kind=int(g.kind[rows[0]]), n_cells=int(rows.size),
            n_points=int(g.n_pts[rows].sum()), cx=cx, cy=cy,
            extent_x=float(g.cx[rows].max() - g.cx[rows].min() + res_m),
            extent_y=float(g.cy[rows].max() - g.cy[rows].min() + res_m),
            z_base=zb, z_top=zt, height=zt - zb,
            range_m=float(np.hypot(cx, cy)), level=int(np.median(g.level[rows])),
        ))
        iid += 1
    g.timings["instances"] = time.perf_counter() - t
    return out


# ----------------------------------------------------------------------
# Frame-to-frame tracking
# ----------------------------------------------------------------------
@dataclass
class Tracker:
    """Greedy nearest-centroid tracker over *compact dynamic* objects.

    Two deliberate restrictions, both learned from the data:

    *   **Dynamic classes only.**  A guardrail is an extended structure
        whose visible extent changes every sweep, so its centroid slides
        along the rail by metres between frames and the implied "speed"
        is an artefact of occlusion, not motion.  Centroid tracking is
        only meaningful for objects whose full extent is roughly visible.

    *   **Velocities are relative to the ego vehicle**, because these
        sweeps carry no ego pose.  With the ego moving at road speed a
        parked car reads as ~-15 m/s, which is correct as a relative
        quantity and must be labelled as one.  Exporting the scenario's
        pose stream turns these into absolute velocities and also
        unlocks multi-sweep map accumulation; see docs/ROADMAP.md.
    """
    dt: float = 0.1
    max_assoc: float = 4.0
    min_age: int = 2          # frames before a velocity is trusted
    min_cells: int = 6        # below this the centroid is too noisy to difference
    max_speed: float = 45.0   # m/s relative; beyond this it is an artefact
    smooth: float = 0.35      # low-pass on the centroid difference
    next_id: int = 0
    tracks: Dict[int, dict] = field(default_factory=dict)

    def update(self, instances: List[Instance]) -> List[dict]:
        live: List[dict] = []
        used = set()
        for inst in [i for i in instances if i.kind == KIND_DYNAMIC]:
            best, best_d = None, self.max_assoc
            for tid, tr in self.tracks.items():
                if tid in used or tr["class_id"] != inst.class_id:
                    continue
                d = float(np.hypot(tr["cx"] - inst.cx, tr["cy"] - inst.cy))
                if d < best_d:
                    best, best_d = tid, d
            if best is None:
                tid = self.next_id
                self.next_id += 1
                tr = {"tid": tid, "class_id": inst.class_id, "age": 0,
                      "vx": 0.0, "vy": 0.0}
            else:
                tid = best
                tr = self.tracks[tid]
                a = self.smooth   # a constant-velocity Kalman filter is the next step
                tr["vx"] = a * (inst.cx - tr["cx"]) / self.dt + (1 - a) * tr["vx"]
                tr["vy"] = a * (inst.cy - tr["cy"]) / self.dt + (1 - a) * tr["vy"]
                tr["age"] += 1
            used.add(tid)
            speed = float(np.hypot(tr["vx"], tr["vy"]))
            tr.update({"cx": inst.cx, "cy": inst.cy, "class_id": inst.class_id,
                       "class_name": inst.class_name, "range_m": inst.range_m,
                       "height": inst.height,
                       "extent_x": inst.extent_x, "extent_y": inst.extent_y,
                       "rel_speed": speed,
                       # Only surface a velocity once the track has been
                       # seen enough times and the value is physical.
                       "n_cells": inst.n_cells,
                       "velocity_valid": bool(tr["age"] >= self.min_age
                                              and speed <= self.max_speed
                                              and inst.n_cells >= self.min_cells)})
            self.tracks[tid] = tr
            live.append(dict(tr))
        self.tracks = {t["tid"]: t for t in live}
        return live
