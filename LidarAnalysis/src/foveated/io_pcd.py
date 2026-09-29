"""Loading of the labelled ASCII PCD frames exported from MATLAB.

The exporter writes an *organised* cloud (HEIGHT = laser rings,
WIDTH = azimuth steps) with the fields

    x y z intensity actor_id class_id material_id

Unreturned rays are written as NaN.  Parsing 6 MB of ASCII per frame is
the single most expensive step in the pipeline, so parsed frames are
cached to .npz next to the source tree.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Frame:
    """One LiDAR sweep, already flattened to valid returns only."""
    name: str
    xyz: np.ndarray        # (N,3) float32, sensor frame
    intensity: np.ndarray  # (N,)  float32
    actor_id: np.ndarray   # (N,)  int64
    class_id: np.ndarray   # (N,)  int16
    row: np.ndarray        # (N,)  int16  laser ring index
    col: np.ndarray        # (N,)  int16  azimuth index
    height: int            # rings in the original organised cloud
    width: int             # azimuth steps
    n_raw: int             # total rays incl. non-returns

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def range_xy(self) -> np.ndarray:
        return np.hypot(self.xyz[:, 0], self.xyz[:, 1])


_HEADER_KEYS = ("FIELDS", "SIZE", "TYPE", "COUNT", "WIDTH", "HEIGHT",
                "POINTS", "DATA", "VERSION", "VIEWPOINT")


def read_pcd_header(path: str) -> Dict[str, object]:
    header: Dict[str, object] = {}
    n_lines = 0
    with open(path, "r") as fh:
        for line in fh:
            n_lines += 1
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, rest = line.partition(" ")
            key = key.upper()
            if key not in _HEADER_KEYS:
                break
            header[key] = rest.strip()
            if key == "DATA":
                break
    header["_header_lines"] = n_lines
    return header


def _cache_path(path: str) -> str:
    stat = os.stat(path)
    tag = hashlib.md5(f"{stat.st_mtime_ns}:{stat.st_size}".encode()).hexdigest()[:10]
    d = os.path.join(os.path.dirname(path), ".cache")
    os.makedirs(d, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(d, f"{base}.{tag}.npz")


def load_pcd(path: str, use_cache: bool = True) -> Frame:
    """Parse one labelled PCD into a :class:`Frame` of valid returns."""
    name = os.path.splitext(os.path.basename(path))[0]
    cache = _cache_path(path)
    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        return Frame(name=name, xyz=z["xyz"], intensity=z["intensity"],
                     actor_id=z["actor_id"], class_id=z["class_id"],
                     row=z["row"], col=z["col"],
                     height=int(z["height"]), width=int(z["width"]),
                     n_raw=int(z["n_raw"]))

    header = read_pcd_header(path)
    fields = str(header.get("FIELDS", "x y z")).split()
    skip = int(header["_header_lines"])
    width = int(header.get("WIDTH", 0))
    height = int(header.get("HEIGHT", 1))

    raw = np.loadtxt(path, skiprows=skip, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    idx = {f: i for i, f in enumerate(fields)}

    n_raw = raw.shape[0]
    # Recover the organised (ring, azimuth) coordinates.  MATLAB writes the
    # cloud column-major over a HEIGHT x WIDTH grid.
    if height * width == n_raw and height > 1:
        lin = np.arange(n_raw)
        row = (lin % height).astype(np.int16)
        col = (lin // height).astype(np.int16)
    else:
        row = np.zeros(n_raw, np.int16)
        col = np.arange(n_raw, dtype=np.int64).astype(np.int16)

    xyz = raw[:, [idx["x"], idx["y"], idx["z"]]]
    valid = np.isfinite(xyz).all(axis=1)

    def col_of(field: str, default=0.0) -> np.ndarray:
        if field in idx:
            return raw[valid, idx[field]]
        return np.full(int(valid.sum()), default)

    frame = Frame(
        name=name,
        xyz=xyz[valid].astype(np.float32),
        intensity=np.nan_to_num(col_of("intensity")).astype(np.float32),
        actor_id=np.nan_to_num(col_of("actor_id")).astype(np.int64),
        class_id=np.nan_to_num(col_of("class_id")).astype(np.int16),
        row=row[valid], col=col[valid],
        height=height, width=width, n_raw=n_raw,
    )
    if use_cache:
        np.savez_compressed(
            cache, xyz=frame.xyz, intensity=frame.intensity,
            actor_id=frame.actor_id, class_id=frame.class_id,
            row=frame.row, col=frame.col,
            height=frame.height, width=frame.width, n_raw=frame.n_raw)
    return frame


def list_frames(folder: str) -> List[str]:
    files = [f for f in os.listdir(folder) if f.lower().endswith(".pcd")]
    def key(f: str):
        m = re.search(r"(\d+)", f)
        return (int(m.group(1)) if m else 0, f)
    return [os.path.join(folder, f) for f in sorted(files, key=key)]


def load_class_map(folder: str) -> Optional[dict]:
    p = os.path.join(folder, "class_map.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)
