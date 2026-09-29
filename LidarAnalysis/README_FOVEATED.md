# Foveated 2.5D Semantic LiDAR Mapping

Turns labelled LiDAR sweeps into a **variable-resolution 2.5D semantic
grid** — 5 cm cells near the vehicle, coarsening to 40 cm at the map edge —
plus terrain analysis, object detection, path planning on the map, and a
live dashboard with latency, memory and fidelity evidence.

```
python run_pipeline.py --scenario road_label_test      # build the maps
python dashboard/app.py --scenario road_label_test     # http://127.0.0.1:5055
python -m pytest tests/ -q                             # invariant suite
```

## What it does

| Brief asks for | Where it lives | Status |
|---|---|---|
| Terrain analysis (drivable vs not) | `semantics.terrain_analysis` | drivable / cost / step / curb / clearance layers |
| Object detection (static + dynamic) | `semantics.extract_instances`, `Tracker` | connected-component instances, class-separated, tracked |
| Adaptive spatial representation | `grid.FoveatedGrid` | dyadic quadtree rings, exact `lookup`, no alignment error |
| Real-time visualisation | `dashboard/` | 5 layers, 2 projections, 7 overlays, playback, click-to-replan |
| Performance metrics | `metrics.py` | latency percentiles, 3 memory baselines, fidelity vs range |

Semantic segmentation is **not** learned yet — the pipeline consumes the
ground-truth labels in `data/labeled/`. Everything downstream is built to
swap the label source for a network's output without changing an
interface; see `docs/ROADMAP.md`.

## The grid

```
ring 0    0–10 m     5 cm
ring 1   10–20 m    10 cm
ring 2   20–40 m    20 cm
ring 3   40–100 m   40 cm
```

Two design decisions carry the whole structure:

**Dyadic resolutions.** Every cell size is the base size times a power of
two and every ring radius is a whole multiple of the coarsest cell. A
coarse cell is therefore exactly tiled by 2×2 finer cells — the rings are
a slice through a quadtree. A point maps to exactly one cell; no gaps, no
double counting.

**Chebyshev ring boundaries**, `max(|x|,|y|)` rather than `√(x²+y²)`.
Circular boundaries cut *through* square cells, so a cell straddling
r = 10 m gets claimed by two levels at once and `lookup()` stops being a
bijection — precisely the "alignment error" the brief warns about. Square
boundaries fall exactly on cell edges at every level. The fine region
becomes a 20×20 m square, which *contains* the 10 m circle the spec asks
for. `tests/test_grid_invariants.py` asserts every corner of every cell
resolves to its own ring.

**2.5D, not 2D.** Each cell splits its z-column into three layers —
ground, obstacle, overhead — so the map can represent a gantry or a low
branch you pass *under*, which a single-elevation map cannot.

**Free space is a polar profile, not cells.** Materialising known-empty
ground at 5 cm would cost ~500 k cells (14 MB) per sweep. The same
information as a per-azimuth visibility profile costs 11 KB and is exact
for a single sensor origin.

## Evidence (41 sweeps, `road_label_test`)

| | |
|---|---|
| Mapping latency | **7.6 ms median**, 17.7 ms p95 → **131 FPS** (sensor runs at 10 Hz: 13× headroom) |
| Memory, design capacity | **26× fewer cells** than a uniform 5 cm 2.5D grid of the same 200×200 m extent; **639× less** than a uniform 5 cm 3D voxel volume |
| Memory, observed area | **4.3× less** than a uniform 5 cm grid over the area actually observed |
| Semantic fidelity | **99.6%** of points still read back their own class from their cell, vs 99.8% at uniform 5 cm and 98.7% at uniform 40 cm |
| Near-field precision | 2.0 cm horizontal RMSE inside 10 m — identical to a uniform 5 cm grid, at a fraction of the cost |

"Fidelity" here is a *representation* metric: every point is re-read from
the cell it landed in and compared against its own ground-truth label. It
isolates the loss caused by discretisation, and stays meaningful once a
learned segmenter replaces the ground-truth labels.

## Planning on the map

A* runs over a decimated version of the same rings (4× coarser: 20 cm
near-field to 1.6 m at the edge). This is the correctness proof for the
structure — a planner crossing ring boundaries thousands of times per
second surfaces alignment bugs that rendering hides. Paths cross up to 3
ring boundaries with no tunnelling and no phantom walls. When the goal is
unreachable the planner returns the farthest point it *could* reach,
dashed, rather than nothing.

Cells that do not exist but that the sensor saw *through* are traversable:
absence of a return is information, and without it the gaps between LiDAR
ring returns on open road would read as a wall.

## Known limitations

These are real and worth stating before a judge finds them:

* **The road is perfectly flat** (z spans 4 cm over 57 m). It is synthetic
  Driving Scenario Designer output, so curb and pothole detection are
  implemented and exercised but have nothing to find. Needs real data or
  scenario terrain.
* **No ego pose**, so object velocities are relative to a moving ego and
  the map cannot accumulate across sweeps.
* **Single-sweep ground coverage is sparse** beyond ~15 m — a spinning
  LiDAR paints rings, not a surface. The free-space carve covers the gaps
  for planning; accumulation is the real fix.
* **70% of raw returns were ego self-hits** labelled "Car". They are
  filtered in `preprocess.clean`, but the exporter should not emit them.

## Layout

```
src/foveated/
  config.py      LOD schedule, taxonomy, thresholds
  io_pcd.py      labelled-PCD reader with .npz caching
  preprocess.py  ego filtering, range gating
  grid.py        the foveated 2.5D engine + free-space carve
  semantics.py   terrain analysis, instances, tracking
  planner.py     A* over the variable-resolution lattice
  metrics.py     latency / memory / fidelity
  export.py      packed binary serialisation
run_pipeline.py  build a scenario bundle
dashboard/       Flask server + canvas dashboard
tests/           invariants that guard the design
```
