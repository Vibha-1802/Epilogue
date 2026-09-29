"""A* over the foveated map -- the correctness proof for the structure.

Rendering a variable-resolution map is easy; *planning* on one is where
alignment bugs surface, because the planner has to answer "what is next
to this cell" thousands of times per second across ring boundaries where
the cell size changes by 2x.  If the nesting were approximate, paths
would tunnel through obstacles at the seams or dead-end against phantom
walls.  A path that crosses every ring without either failure mode is
evidence the lookup is exact.

Two further design points worth stating:

*   **Mapping resolution is not planning resolution.**  The planner runs
    on a decimated lattice (``plan_mult``, default 4x coarser per ring:
    20 cm near-field, 1.6 m at the far edge).  Because the schedule is
    dyadic, decimation is exact -- a planning cell is a whole number of
    map cells, never a fractional overlap.

*   **Absence of a cell is information.**  A cell that does not exist but
    that the sensor saw *through* (``grid.is_free``) is known-empty
    ground and is traversable.  Without that, the gaps between LiDAR
    ring returns on an open road would look like a wall of unknown.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import DRIVABLE_CLASSES, KIND_DYNAMIC, KIND_STATIC
from .grid import _IDX_BITS, _IDX_OFFSET, FoveatedGrid, is_free

_STEPS = [(1, 0), (-1, 0), (0, 1), (0, -1),
          (1, 1), (1, -1), (-1, 1), (-1, -1)]


@dataclass
class PlanResult:
    found: bool
    path: List[Tuple[float, float]]
    cost: float
    expanded: int
    ms: float
    level_transitions: int
    goal: Tuple[float, float]
    reason: str = ""

    def to_dict(self) -> dict:
        return {"found": self.found, "path": [[float(x), float(y)] for x, y in self.path],
                "cost": float(self.cost), "expanded": int(self.expanded),
                "ms": round(self.ms, 2),
                "level_transitions": int(self.level_transitions),
                "goal": [float(self.goal[0]), float(self.goal[1])],
                "reason": self.reason}


class NavLattice:
    """Decimated, obstacle-inflated view of a foveated grid."""

    def __init__(self, g: FoveatedGrid, plan_mult: int = 4,
                 inflation: float = 1.1, influence: float = 3.0):
        from scipy.spatial import cKDTree
        self.g = g
        self.plan_mult = plan_mult
        self.inflation = inflation
        self.influence = influence

        blocked = np.isin(g.kind, [KIND_STATIC, KIND_DYNAMIC])
        pts = np.stack([g.cx[blocked], g.cy[blocked]], axis=1)
        self.obstacles = pts
        self.tree = cKDTree(pts) if pts.shape[0] else None

        # Scalar fast paths.  A* touches these hundreds of thousands of
        # times, and numpy's per-call array overhead dominates at that
        # granularity, so the hot loop uses plain Python floats.
        self._radii = tuple(float(r) for r in g.cfg.ring_radii)
        self._res = tuple(float(l.res) for l in g.cfg.levels())
        self._keys = g.key
        self._cost_memo: Dict[Tuple[int, int, int], float] = {}
        self._nb = int(g.free_range.size)
        self._free = g.free_range
        self._seen = g.seen_range

    # -- lattice geometry -------------------------------------------
    def _level_of(self, x: float, y: float) -> int:
        d = max(abs(x), abs(y)) if self.g.cfg.ring_metric == "chebyshev" \
            else (x * x + y * y) ** 0.5
        for i, r in enumerate(self._radii):
            if d < r:
                return i
        return -1

    def res_at(self, x: float, y: float) -> float:
        lv = self._level_of(x, y)
        return float("nan") if lv < 0 else self._res[lv] * self.plan_mult

    def node_of(self, x: float, y: float) -> Optional[Tuple[int, int, int]]:
        lv = self._level_of(x, y)
        if lv < 0:
            return None
        res = self._res[lv] * self.plan_mult
        # math.floor(x / res), never x // res: CPython's float floor-div
        # applies an fmod-based correction and returns 98.0 for
        # 4.9 // 0.05 where numpy's floor(4.9 / 0.05) returns 97.  Both
        # are defensible; only one of them matches grid.lookup, and a
        # planner indexing cells differently from the map is a silent
        # off-by-one cell at every ring.
        return (lv, math.floor(x / res), math.floor(y / res))

    def center(self, node: Tuple[int, int, int]) -> Tuple[float, float]:
        lv, ix, iy = node
        res = self._res[lv] * self.plan_mult
        return ((ix + 0.5) * res, (iy + 0.5) * res)

    # -- traversability ---------------------------------------------
    def clearance_of(self, x: float, y: float) -> float:
        if self.tree is None:
            return float("inf")
        return float(self.tree.query((x, y))[0])

    def _map_row(self, x: float, y: float) -> int:
        """Scalar equivalent of grid.lookup for one position."""
        lv = self._level_of(x, y)
        if lv < 0:
            return -1
        res = self._res[lv]
        key = ((lv << (2 * _IDX_BITS))
               | ((math.floor(x / res) + _IDX_OFFSET) << _IDX_BITS)
               | (math.floor(y / res) + _IDX_OFFSET))
        pos = int(np.searchsorted(self._keys, key))
        if pos < self._keys.size and int(self._keys[pos]) == key:
            return pos
        return -1

    def _is_free(self, x: float, y: float) -> bool:
        if self._nb == 0:
            return False
        r = math.hypot(x, y)
        b = int((math.atan2(y, x) + math.pi) / (2 * math.pi) * self._nb)
        b = 0 if b < 0 else (self._nb - 1 if b >= self._nb else b)
        return r < float(self._free[b]) - self.g.cfg.free_margin and \
            r <= float(self._seen[b])

    def cell_cost(self, node: Tuple[int, int, int]) -> float:
        """Per-metre multiplier, or inf when the cell is not traversable.

        Memoised: A* reaches most nodes from several neighbours, and the
        KD-tree query behind ``clearance_of`` is the single most expensive
        thing in the loop.
        """
        hit = self._cost_memo.get(node)
        if hit is not None:
            return hit
        x, y = self.center(node)
        val = self._cell_cost_xy(x, y)
        self._cost_memo[node] = val
        return val

    def _cell_cost_xy(self, x: float, y: float) -> float:
        clr = self.clearance_of(x, y)
        if clr < self.inflation:
            return float("inf")
        row = self._map_row(x, y)
        if row >= 0:
            c = float(self.g.cost[row])
            if not np.isfinite(c):
                return float("inf")
            base = c
        else:
            # No return here.  Traversable only if a ray passed through.
            if not self._is_free(x, y):
                return float("inf")
            base = 0.35          # known-free but unobserved surface
        risk = clr / self.influence
        risk = 0.0 if risk >= 1.0 else (1.0 - risk)
        return 1.0 + base + 2.0 * risk * risk


def default_goal(g: FoveatedGrid, lattice: NavLattice,
                 forward: Tuple[float, float] = (1.0, 0.0)) -> Tuple[float, float]:
    """Farthest reachable-looking drivable cell in the forward direction."""
    drivable = np.where(g.drivable)[0]
    if drivable.size == 0:
        return (10.0, 0.0)
    fx, fy = forward
    proj = g.cx[drivable] * fx + g.cy[drivable] * fy
    lat = np.abs(-g.cx[drivable] * fy + g.cy[drivable] * fx)
    ok = (lat < 4.0) & (proj > 0)
    cand = drivable[ok] if ok.any() else drivable[proj > 0]
    if cand.size == 0:
        cand = drivable
    best = cand[np.argmax(g.cx[cand] * fx + g.cy[cand] * fy)]
    return (float(g.cx[best]), float(g.cy[best]))


def plan(g: FoveatedGrid, goal: Optional[Tuple[float, float]] = None,
         start: Tuple[float, float] = (0.0, 0.0), plan_mult: int = 4,
         max_expand: int = 120_000) -> PlanResult:
    t0 = time.perf_counter()
    lat = NavLattice(g, plan_mult=plan_mult)
    if goal is None:
        goal = default_goal(g, lat)
    gx, gy = goal

    # The ego sits inside the r_min_valid hole, so seed the search at the
    # nearest traversable lattice node ahead of it rather than at (0,0).
    start_node = lat.node_of(*start)
    if start_node is None:
        return PlanResult(False, [], 0.0, 0, 0.0, 0, goal, "start outside map")
    goal_node = lat.node_of(gx, gy)
    if goal_node is None:
        return PlanResult(False, [], 0.0, 0, 0.0, 0, goal, "goal outside map")

    def h(node) -> float:
        cx, cy = lat.center(node)
        return float(np.hypot(cx - gx, cy - gy))

    open_heap: List[Tuple[float, Tuple[int, int, int]]] = []
    heapq.heappush(open_heap, (h(start_node), start_node))
    came: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    gscore: Dict[Tuple[int, int, int], float] = {start_node: 0.0}
    closed = set()
    expanded = 0

    while open_heap:
        _, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        closed.add(node)
        expanded += 1
        if node == goal_node or h(node) < 1.5 * lat.res_at(*lat.center(node)):
            goal_node = node
            break
        if expanded > max_expand:
            return PlanResult(False, [], 0.0, expanded,
                              (time.perf_counter() - t0) * 1000, 0, goal,
                              "expansion limit")
        cx, cy = lat.center(node)
        res = lat.res_at(cx, cy)
        for dx, dy in _STEPS:
            nx, ny = cx + dx * res, cy + dy * res
            nnode = lat.node_of(nx, ny)
            if nnode is None or nnode in closed:
                continue
            c = lat.cell_cost(nnode)
            if c == float("inf"):
                continue
            ncx, ncy = lat.center(nnode)
            seg = float(np.hypot(ncx - cx, ncy - cy))
            tentative = gscore[node] + seg * c
            if tentative < gscore.get(nnode, float("inf")):
                gscore[nnode] = tentative
                came[nnode] = node
                heapq.heappush(open_heap, (tentative + h(nnode), nnode))

    reason = ""
    if goal_node not in gscore:
        # No route to the requested goal.  Returning nothing would be the
        # wrong behaviour for a navigation stack: report the best
        # progress actually achievable and say why it stopped.  The
        # dashboard draws this as a dashed "best effort" path.
        if not closed:
            return PlanResult(False, [], 0.0, expanded,
                              (time.perf_counter() - t0) * 1000, 0, goal,
                              "no traversable cell at start")
        goal_node = min(closed, key=h)
        reason = "goal unreachable - showing farthest reachable point"

    path_nodes = [goal_node]
    while path_nodes[-1] in came:
        path_nodes.append(came[path_nodes[-1]])
    path_nodes.reverse()
    transitions = sum(1 for a, b in zip(path_nodes, path_nodes[1:]) if a[0] != b[0])
    path = [lat.center(n) for n in path_nodes]
    return PlanResult(not reason, path, gscore[goal_node], expanded,
                      (time.perf_counter() - t0) * 1000, transitions, goal, reason)
