"""Quantitative evidence: memory, latency and fidelity-versus-range.

The brief asks for "a significant reduction in memory usage compared to
a uniform high-resolution 3D map" and "high accuracy across varying
distances".  Those two pull against each other, so the honest way to
report them is jointly, as a Pareto trade-off.

Baselines are built with the *same* engine using degenerate one-ring
schedules, so any difference is attributable to the foveation schedule
and not to a different implementation.

Fidelity is measured by round-tripping through the map: every point is
re-read from the cell it landed in, and compared against its own ground
truth.

    class_acc  -- does the cell's dominant class still match the point's
                  label?  This is the semantic information destroyed by
                  discretisation, measured per range band.
    xy_rmse    -- horizontal quantisation error, point to cell centre.
    z_rmse     -- elevation error against the appropriate 2.5D layer.

Note on "accuracy": the labels here are simulator ground truth, so this
is *representation* fidelity, not network accuracy.  It is the part of
the end-to-end error the grid is responsible for, and it stays valid
once a learned segmenter replaces the ground-truth labels.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from .config import GridConfig, PipelineConfig
from .grid import FoveatedGrid, build, carve_free_space
from .preprocess import CleanFrame

RANGE_BINS = np.array([0, 5, 10, 15, 20, 30, 40, 60, 80, 100], float)


def uniform_config(base: GridConfig, res: float) -> GridConfig:
    """A one-ring schedule -- i.e. a classic uniform grid."""
    import dataclasses
    return dataclasses.replace(
        base, base_res=res, ring_radii=(base.r_max,), res_multipliers=(1,))


# ----------------------------------------------------------------------
def fidelity(g: FoveatedGrid) -> Dict[str, object]:
    """Round-trip every point through its cell and score the loss."""
    inv = g.point_cell
    if inv.size == 0:
        return {}
    pz = g.meta["point_z"]
    pc = g.meta["point_class"]
    pr = g.meta["point_range"]

    cell_cls = g.class_id[inv]
    correct = (cell_cls == pc)

    px = g.cx[inv]
    py = g.cy[inv]
    # reconstruct the point's own xy from range/azimuth
    az = g.meta["point_az"]
    ox, oy = pr * np.cos(az), pr * np.sin(az)
    xy_err = np.hypot(ox - px, oy - py)

    # Elevation error is measured against the *nearest stored layer*,
    # because that is what a layered 2.5D map actually reconstructs.
    # Scoring every point against a single elevation value would be
    # scoring a 2D map, not this one.
    layers = np.stack([g.z_ground[inv], g.obstacle_top[inv],
                       g.overhead_min[inv], g.z_max[inv]], axis=1)
    z_err = np.nanmin(np.abs(layers - pz[:, None]), axis=1)

    dz = pz - g.z_min[inv]
    ground = dz <= g.cfg.ground_band
    z_err_ground = np.abs(pz - g.z_ground[inv])

    idx = np.clip(np.digitize(pr, RANGE_BINS) - 1, 0, len(RANGE_BINS) - 2)
    rows: List[dict] = []
    for b in range(len(RANGE_BINS) - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "r_lo": float(RANGE_BINS[b]), "r_hi": float(RANGE_BINS[b + 1]),
            "n_points": int(m.sum()),
            "class_acc": float(correct[m].mean()),
            "xy_rmse": float(np.sqrt(np.mean(xy_err[m] ** 2))),
            "z_rmse": float(np.sqrt(np.nanmean(z_err[m] ** 2))),
            "z_rmse_ground": float(np.sqrt(np.nanmean(z_err_ground[m & ground] ** 2)))
                if (m & ground).any() else float("nan"),
            "cell_res": float(np.median(g.res[inv[m]])),
        })
    return {
        "overall": {
            "class_acc": float(correct.mean()),
            "xy_rmse": float(np.sqrt(np.mean(xy_err ** 2))),
            "z_rmse": float(np.sqrt(np.nanmean(z_err ** 2))),
            "z_rmse_ground": float(np.sqrt(np.nanmean(z_err_ground[ground] ** 2))),
        },
        "by_range": rows,
    }


# ----------------------------------------------------------------------
def memory_report(g: FoveatedGrid, frame: CleanFrame, pcfg: PipelineConfig,
                  baselines: Dict[str, FoveatedGrid]) -> Dict[str, object]:
    """Bytes for the foveated map and for every baseline representation."""
    nb = g.nbytes()
    per_cell = nb["total"] / max(g.n_cells, 1)
    cfg = g.cfg
    side = 2 * cfg.r_max

    def sparse_bytes(gg: FoveatedGrid) -> int:
        return int(gg.nbytes()["total"])

    entries: Dict[str, dict] = {}
    entries["foveated_2_5d"] = {
        "label": "Foveated 2.5D (ours)",
        "cells": int(g.n_cells),
        "bytes": int(nb["total"]),
        "kind": "sparse",
    }
    for key, gg in baselines.items():
        entries[key] = {
            "label": f"Uniform 2.5D @ {gg.cfg.base_res*100:.0f} cm (sparse)",
            "cells": int(gg.n_cells),
            "bytes": sparse_bytes(gg),
            "kind": "sparse",
        }

    # Dense allocations -- what you pay if you preallocate the map extent.
    for res, tag in ((pcfg.uniform_fine_res, "dense_fine"),
                     (pcfg.uniform_coarse_res, "dense_coarse")):
        n = int(round(side / res)) ** 2
        entries[tag] = {
            "label": f"Uniform 2.5D @ {res*100:.0f} cm (dense {side:.0f}x{side:.0f} m)",
            "cells": n, "bytes": int(n * per_cell), "kind": "dense",
        }

    # 3D voxel maps -- the thing the brief asks us to beat.
    vr = pcfg.voxel_res
    xyz_bytes_per_voxel = 6          # class(1) + occupancy(1) + intensity(4)
    nz = int(round(pcfg.voxel_z_extent / vr))
    n_dense_vox = int(round(side / vr)) ** 2 * nz
    entries["voxel_dense"] = {
        "label": f"Uniform 3D voxels @ {vr*100:.0f} cm (dense)",
        "cells": n_dense_vox,
        "bytes": int(n_dense_vox * xyz_bytes_per_voxel), "kind": "dense",
    }
    p = frame.xyz.astype(np.float64)
    vi = np.floor(p / vr).astype(np.int64)
    n_sparse_vox = int(np.unique(vi[:, 0] * 10**10 + vi[:, 1] * 10**5 + vi[:, 2]).size)
    entries["voxel_sparse"] = {
        "label": f"Uniform 3D voxels @ {vr*100:.0f} cm (sparse/occupied)",
        "cells": n_sparse_vox,
        "bytes": int(n_sparse_vox * xyz_bytes_per_voxel), "kind": "sparse",
    }
    entries["raw_points"] = {
        "label": "Raw point cloud (xyzi, float32)",
        "cells": int(len(frame)), "bytes": int(len(frame) * 16), "kind": "raw",
    }

    base = entries["foveated_2_5d"]["bytes"]
    for e in entries.values():
        e["ratio_vs_foveated"] = round(e["bytes"] / max(base, 1), 2)
    return {"per_cell_bytes": round(per_cell, 2), "entries": entries,
            "field_breakdown": nb}


# ----------------------------------------------------------------------
def evaluate_frame(frame: CleanFrame, pcfg: PipelineConfig,
                   with_baselines: bool = True) -> Dict[str, object]:
    """Build the foveated map plus baselines and score all of them."""
    cfg = pcfg.grid
    t0 = time.perf_counter()
    g = build(frame, cfg)
    carve_free_space(g, frame)
    t_build = time.perf_counter() - t0

    baselines: Dict[str, FoveatedGrid] = {}
    fid: Dict[str, object] = {"foveated": fidelity(g)}
    if with_baselines:
        for tag, res in (("uniform_fine", pcfg.uniform_fine_res),
                         ("uniform_coarse", pcfg.uniform_coarse_res)):
            gb = build(frame, uniform_config(cfg, res))
            baselines[tag] = gb
            fid[tag] = fidelity(gb)

    return {
        "grid": g,
        "baselines": baselines,
        "fidelity": fid,
        "memory": memory_report(g, frame, pcfg, baselines),
        "latency_ms": {k: round(v * 1000, 3) for k, v in g.timings.items()},
        "build_ms": round(t_build * 1000, 3),
    }


def aggregate(reports: List[Dict[str, object]]) -> Dict[str, object]:
    """Average per-frame reports into the headline numbers."""
    def mean(path) -> float:
        vals = [path(r) for r in reports]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    out: Dict[str, object] = {}
    out["n_frames"] = len(reports)
    out["mean_build_ms"] = mean(lambda r: r["build_ms"])
    out["fps"] = 1000.0 / out["mean_build_ms"] if out["mean_build_ms"] else float("nan")
    out["mean_cells"] = mean(lambda r: r["memory"]["entries"]["foveated_2_5d"]["cells"])

    mem_keys = reports[0]["memory"]["entries"].keys()
    out["memory"] = {
        k: {
            "label": reports[0]["memory"]["entries"][k]["label"],
            "bytes": mean(lambda r, k=k: r["memory"]["entries"][k]["bytes"]),
            "cells": mean(lambda r, k=k: r["memory"]["entries"][k]["cells"]),
            "ratio_vs_foveated": mean(
                lambda r, k=k: r["memory"]["entries"][k]["ratio_vs_foveated"]),
        } for k in mem_keys
    }

    out["fidelity"] = {}
    for variant in reports[0]["fidelity"]:
        rows: Dict[float, list] = {}
        for r in reports:
            for row in r["fidelity"][variant].get("by_range", []):
                rows.setdefault(row["r_lo"], []).append(row)
        by_range = []
        for r_lo in sorted(rows):
            grp = rows[r_lo]
            w = np.array([x["n_points"] for x in grp], float)
            by_range.append({
                "r_lo": r_lo, "r_hi": grp[0]["r_hi"],
                "n_points": float(w.sum()),
                "class_acc": float(np.average([x["class_acc"] for x in grp], weights=w)),
                "xy_rmse": float(np.average([x["xy_rmse"] for x in grp], weights=w)),
                "z_rmse": float(np.average([x["z_rmse"] for x in grp], weights=w)),
                "z_rmse_ground": float(np.nanmean([x["z_rmse_ground"] for x in grp])),
                "cell_res": float(np.median([x["cell_res"] for x in grp])),
            })
        out["fidelity"][variant] = {
            "overall": {
                k: mean(lambda r, v=variant, k=k: r["fidelity"][v]["overall"][k])
                for k in ("class_acc", "xy_rmse", "z_rmse", "z_rmse_ground")
            },
            "by_range": by_range,
        }
    return out


# ----------------------------------------------------------------------
def coverage_report(g: FoveatedGrid, pcfg: PipelineConfig,
                    chunk: int = 4_000_000) -> Dict[str, object]:
    """Cells needed to represent the *whole observed area*, not just returns.

    The sparse "occupied cells only" comparison flatters every baseline
    equally and understates what foveation buys, because an occupancy
    map is not allowed to forget free space: knowing a patch of road is
    empty is exactly as load-bearing for a planner as knowing a wall is
    there.  So here we count the cells required to tile everything the
    sensor actually observed -- occupied *and* carved-free -- at each
    schedule.  That is the number a real mapping stack has to allocate.
    """
    if g.free_range.size == 0:
        return {}
    cfg = g.cfg
    nb = g.free_range.size

    def observed(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.hypot(x, y)
        b = np.clip(((np.arctan2(y, x) + np.pi) / (2 * np.pi) * nb).astype(np.int64),
                    0, nb - 1)
        return r <= g.seen_range[b]

    # --- foveated: enumerate each ring at its own resolution ---------
    fov_cells = 0
    per_level = []
    radii = (0.0,) + tuple(cfg.ring_radii)
    for lv in cfg.levels():
        res = lv.res
        n_side = int(round(2 * lv.r_max / res))
        ax = (np.arange(n_side) - n_side / 2 + 0.5) * res
        X, Y = np.meshgrid(ax, ax, indexing="ij")
        d = np.maximum(np.abs(X), np.abs(Y))
        ring = (d >= lv.r_min) & (d < lv.r_max)
        X, Y = X[ring], Y[ring]
        n = int(observed(X, Y).sum())
        per_level.append({"level": lv.index, "res": res, "cells": n})
        fov_cells += n

    # --- uniform baselines ------------------------------------------
    uni = {}
    for tag, res in (("uniform_fine", pcfg.uniform_fine_res),
                     ("uniform_coarse", pcfg.uniform_coarse_res)):
        n_side = int(round(2 * cfg.r_max / res))
        total = 0
        ax = (np.arange(n_side) - n_side / 2 + 0.5) * res
        step = max(1, chunk // n_side)
        for i0 in range(0, n_side, step):
            xs = ax[i0:i0 + step]
            X, Y = np.meshgrid(xs, ax, indexing="ij")
            total += int(observed(X.ravel(), Y.ravel()).sum())
        uni[tag] = {"res": res, "cells": total}

    per_cell = g.nbytes()["total"] / max(g.n_cells, 1)
    out = {
        "per_cell_bytes": round(per_cell, 2),
        "foveated": {"cells": fov_cells, "bytes": int(fov_cells * per_cell),
                     "per_level": per_level},
    }
    for tag, v in uni.items():
        out[tag] = {"res": v["res"], "cells": v["cells"],
                    "bytes": int(v["cells"] * per_cell),
                    "ratio_vs_foveated": round(v["cells"] / max(fov_cells, 1), 2)}
    # A 3D voxel map of the same observed volume, for the brief's
    # "uniform high-resolution 3D map" comparison.
    nz = int(round(pcfg.voxel_z_extent / pcfg.voxel_res))
    out["voxel_fine"] = {
        "res": pcfg.voxel_res,
        "cells": uni["uniform_fine"]["cells"] * nz,
        "bytes": int(uni["uniform_fine"]["cells"] * nz * 6),
        "ratio_vs_foveated": round(
            uni["uniform_fine"]["cells"] * nz * 6 / max(fov_cells * per_cell, 1), 2),
    }
    return out


# ----------------------------------------------------------------------
def capacity_report(cfg: GridConfig, pcfg: PipelineConfig,
                    per_cell_bytes: float = 58.0) -> Dict[str, object]:
    """Design capacity: cost of a *fully populated* map of the same extent.

    This is the property of the data structure itself, independent of how
    much of the scene any particular sweep happened to see.  It is the
    number to quote for "how much does foveation save", with
    ``coverage_report`` as the measured in-scenario counterpart.
    """
    side = 2 * cfg.r_max
    fov = 0
    per_level = []
    for lv in cfg.levels():
        n = int(round((2 * lv.r_max / lv.res) ** 2 - (2 * lv.r_min / lv.res) ** 2))
        per_level.append({"level": lv.index, "res": lv.res, "cells": n})
        fov += n
    out = {
        "extent_m": side,
        "foveated": {"cells": fov, "bytes": int(fov * per_cell_bytes),
                     "per_level": per_level},
    }
    for tag, res in (("uniform_fine", pcfg.uniform_fine_res),
                     ("uniform_coarse", pcfg.uniform_coarse_res)):
        n = int(round(side / res)) ** 2
        out[tag] = {"res": res, "cells": n, "bytes": int(n * per_cell_bytes),
                    "ratio_vs_foveated": round(n / fov, 2)}
    nz = int(round(pcfg.voxel_z_extent / pcfg.voxel_res))
    nv = int(round(side / pcfg.voxel_res)) ** 2 * nz
    out["voxel_fine"] = {
        "res": pcfg.voxel_res, "z_extent": pcfg.voxel_z_extent,
        "cells": nv, "bytes": int(nv * 6),
        "ratio_vs_foveated": round(nv * 6 / (fov * per_cell_bytes), 2)}
    return out
