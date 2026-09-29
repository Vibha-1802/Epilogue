"""Frame conditioning before gridding.

Two things matter here:

1.  *Ego self-hits.*  In these exports ~70 % of all valid returns are the
    sensor seeing the ego vehicle's own body (actor_id == ego, r < 2.4 m)
    and they carry ClassID 1 ("Car").  Left in, they poison both the
    near-field occupancy map and every class-balance statistic.  We drop
    them by actor id when it is available and by an ego bounding box
    otherwise, so the same code works on real logs with no actor ids.

2.  *Range gating.*  Anything beyond the outermost ring is discarded up
    front, which is itself part of the foveation budget.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GridConfig
from .io_pcd import Frame


@dataclass
class CleanFrame:
    xyz: np.ndarray
    intensity: np.ndarray
    class_id: np.ndarray
    actor_id: np.ndarray
    col: np.ndarray
    rng: np.ndarray        # horizontal range from sensor
    azimuth: np.ndarray    # radians, -pi..pi
    name: str
    stats: dict

    def __len__(self) -> int:
        return int(self.xyz.shape[0])


def clean(frame: Frame, cfg: GridConfig) -> CleanFrame:
    n0 = len(frame)
    xyz = frame.xyz
    r = np.hypot(xyz[:, 0], xyz[:, 1])

    keep = np.ones(n0, dtype=bool)

    # --- ego removal -------------------------------------------------
    by_actor = frame.actor_id == cfg.ego_actor_id
    xmin, xmax, ymin, ymax = cfg.ego_box
    by_box = ((xyz[:, 0] > xmin) & (xyz[:, 0] < xmax) &
              (xyz[:, 1] > ymin) & (xyz[:, 1] < ymax))
    ego = by_actor | by_box | (r < cfg.r_min_valid)
    keep &= ~ego

    # --- range gate --------------------------------------------------
    out_of_range = r >= cfg.r_max
    keep &= ~out_of_range

    az = np.arctan2(xyz[:, 1], xyz[:, 0])
    stats = {
        "n_raw_rays": int(frame.n_raw),
        "n_returns": int(n0),
        "n_ego_removed": int(ego.sum()),
        "n_out_of_range": int((out_of_range & ~ego).sum()),
        "n_kept": int(keep.sum()),
        "ego_fraction": float(ego.sum() / max(n0, 1)),
    }
    return CleanFrame(
        xyz=xyz[keep], intensity=frame.intensity[keep],
        class_id=frame.class_id[keep], actor_id=frame.actor_id[keep],
        col=frame.col[keep], rng=r[keep].astype(np.float32),
        azimuth=az[keep].astype(np.float32), name=frame.name, stats=stats,
    )
