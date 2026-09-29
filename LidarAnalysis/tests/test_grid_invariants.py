"""Invariants the whole variable-resolution design depends on.

These are not smoke tests.  Each one guards a failure mode that produced
a real bug during development:

*   circular ring boundaries made ``lookup`` non-injective near r = 10 m;
*   a float32 resolution table made 0.05 not equal to 0.05, so the
    planner and the map disagreed about which cell a point was in;
*   CPython's ``//`` and numpy's ``floor(x/y)`` round 4.9/0.05
    differently, which is a one-cell offset at every ring boundary.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from foveated.config import GridConfig, PipelineConfig          # noqa: E402
from foveated.grid import build, carve_free_space               # noqa: E402
from foveated.io_pcd import list_frames, load_pcd               # noqa: E402
from foveated.planner import NavLattice, plan                   # noqa: E402
from foveated.preprocess import clean                           # noqa: E402
from foveated.semantics import extract_instances, terrain_analysis  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "labeled", "road_label_test")
pytestmark = pytest.mark.skipif(not os.path.isdir(DATA), reason="dataset not present")


@pytest.fixture(scope="module")
def grids():
    cfg = GridConfig()
    out = []
    for path in list_frames(DATA)[::8]:
        c = clean(load_pcd(path), cfg)
        g = build(c, cfg)
        carve_free_space(g, c)
        terrain_analysis(g)
        out.append((g, c))
    return out


def test_schedule_is_dyadic():
    GridConfig().validate()


def test_every_point_lands_in_exactly_one_cell(grids):
    for g, c in grids:
        assert g.point_cell.size == g.meta["n_points_gridded"]
        assert int(g.n_pts.sum()) == g.meta["n_points_gridded"], "points lost or double-counted"
        assert (g.point_cell >= 0).all() and (g.point_cell < g.n_cells).all()


def test_lookup_is_bijective_on_cell_centres(grids):
    for g, _ in grids:
        idx = g.lookup(g.cx.astype(float), g.cy.astype(float))
        assert (idx == np.arange(g.n_cells)).all()


def test_lookup_agrees_with_planner_scalar_path(grids):
    rng = np.random.default_rng(0)
    for g, _ in grids:
        lat = NavLattice(g)
        r = g.cfg.r_max - 0.1
        px, py = rng.uniform(-r, r, 20000), rng.uniform(-r, r, 20000)
        vec = g.lookup(px, py)
        sca = np.array([lat._map_row(float(a), float(b)) for a, b in zip(px, py)])
        assert (vec == sca).all()


def test_cells_never_straddle_a_ring_boundary(grids):
    """Every corner of every cell must sit in that cell's own ring."""
    for g, _ in grids:
        for sx in (0.0, 1.0):
            for sy in (0.0, 1.0):
                x = (g.ix + sx) * g.res
                y = (g.iy + sy) * g.res
                # nudge inwards so a shared edge is attributed to this cell
                x = x - (sx - 0.5) * 1e-6
                y = y - (sy - 0.5) * 1e-6
                lv = g.level_of_d(g.ring_metric(x, y))
                assert (lv == g.level).all()


def test_resolution_matches_the_advertised_schedule(grids):
    want = {0: 0.05, 1: 0.10, 2: 0.20, 3: 0.40}
    for g, _ in grids:
        for lv, res in want.items():
            m = g.level == lv
            if m.any():
                assert np.allclose(g.res[m], res)


def test_ego_returns_are_removed(grids):
    for g, c in grids:
        assert c.stats["n_ego_removed"] > 0
        assert (np.hypot(c.xyz[:, 0], c.xyz[:, 1]) >= g.cfg.r_min_valid).all()


def test_plan_crosses_levels_without_leaving_the_map(grids):
    g, _ = grids[0]
    res = plan(g)
    assert res.path, "planner produced no path at all"
    xs = np.array([p[0] for p in res.path])
    ys = np.array([p[1] for p in res.path])
    assert (g.ring_metric(xs, ys) < g.cfg.r_max).all()
    # consecutive waypoints must be adjacent at the coarser of the two
    # planning resolutions -- i.e. the path never teleports across a seam
    lat = NavLattice(g)
    for (x0, y0), (x1, y1) in zip(res.path, res.path[1:]):
        step = max(lat.res_at(x0, y0), lat.res_at(x1, y1))
        assert np.hypot(x1 - x0, y1 - y0) <= step * 1.5 + 1e-6


def test_instances_are_physically_plausible(grids):
    seen = set()
    for g, _ in grids:
        for inst in extract_instances(g):
            seen.add(inst.class_name)
            assert 0.0 < inst.extent_x < 60.0
            assert 0.0 < inst.extent_y < 60.0
            assert -1.0 < inst.height < 8.0
    assert {"Car", "Pedestrian", "Guardrail"} <= seen


def test_tracker_ignores_extended_static_structures(grids):
    """A guardrail must never be handed a velocity.

    Its visible extent changes every sweep, so its centroid slides along
    the rail and centroid differencing reports tens of m/s for something
    bolted to the ground.  Only compact dynamic objects are tracked.
    """
    from foveated.semantics import Tracker
    tr = Tracker(dt=0.1)
    seen_dynamic = False
    for g, _ in grids:
        live = tr.update(extract_instances(g))
        for t in live:
            assert t["class_id"] in (1, 2, 3, 4), f"tracked a static class: {t}"
            assert t["rel_speed"] <= tr.max_speed or not t["velocity_valid"]
            seen_dynamic = True
    assert seen_dynamic, "tracker produced nothing at all"
