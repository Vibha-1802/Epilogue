"""Serialisation of built maps for the dashboard.

Cells go out as one packed float32 buffer in column-major order rather
than as JSON.  A sweep is ~8-20 k cells x 16 attributes; as JSON that is
several megabytes of text to parse per frame, which would make the
playback rate a property of the JSON parser instead of the mapper.  The
binary form is parsed in the browser with a single ``Float32Array``.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from .grid import FoveatedGrid

# Column order of the packed cell buffer.  The dashboard reads this list
# from the manifest instead of hard-coding offsets.
CELL_COLUMNS: List[str] = [
    "x", "y", "res", "level", "class_id", "kind", "z_ground", "z_max",
    "obstacle_h", "clearance", "cost", "drivable", "curb", "low_clearance",
    "n_pts", "intensity", "conf", "step",
]


def pack_cells(g: FoveatedGrid) -> np.ndarray:
    """Column-major float32 buffer, shape (n_columns, n_cells)."""
    n = g.n_cells
    inf = np.isfinite
    cost = np.where(inf(g.cost), g.cost, -1.0)          # -1 encodes "blocked"
    obs = np.where(inf(g.obstacle_height), g.obstacle_height, 0.0)
    clr = np.where(inf(g.clearance), g.clearance, -1.0)  # -1 encodes "open sky"
    cols = {
        "x": g.cx, "y": g.cy, "res": g.res, "level": g.level,
        "class_id": g.class_id, "kind": g.kind,
        "z_ground": g.z_ground, "z_max": g.z_max,
        "obstacle_h": obs, "clearance": clr, "cost": cost,
        "drivable": g.drivable, "curb": g.curb,
        "low_clearance": g.low_clearance, "n_pts": g.n_pts,
        "intensity": g.intensity, "conf": g.class_conf, "step": g.step,
    }
    out = np.empty((len(CELL_COLUMNS), n), np.float32)
    for i, name in enumerate(CELL_COLUMNS):
        out[i] = np.nan_to_num(np.asarray(cols[name], np.float32),
                               nan=0.0, posinf=0.0, neginf=0.0)
    return out


def free_polygon(g: FoveatedGrid, n_out: int = 720) -> List[float]:
    """Down-sampled visibility profile, ready to draw as a filled ring."""
    if g.free_range.size == 0:
        return []
    src = np.minimum(g.free_range, g.cfg.r_max)
    src = np.minimum(src, g.seen_range)
    k = max(1, src.size // n_out)
    trimmed = src[:(src.size // k) * k].reshape(-1, k)
    return [round(float(v), 2) for v in trimmed.min(axis=1)]


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return v if np.isfinite(v) else None
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))

    with open(path, "w") as fh:
        json.dump(obj, fh, default=default, separators=(",", ":"))
