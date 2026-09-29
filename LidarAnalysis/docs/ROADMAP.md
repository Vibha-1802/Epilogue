# Roadmap

Ordered by *judge-visible value per hour*, not by how interesting the work
is. Each item says what it unblocks and roughly what it costs.

---

## Tier 0 — cheap fixes that remove visible weaknesses

### 1. Export the ego pose stream  ·  ~1 hour  ·  biggest single win
In MATLAB, alongside each sweep, record the ego pose:

```matlab
p = actorPoses(scnro);
ego = p([p.ActorID] == egoVehicle.ActorID);
pose = struct('t', scnro.SimulationTime, ...
              'position', ego.Position, 'yaw', ego.Yaw);
```

Write one `poses.json` next to the PCDs. This single file unblocks **four**
downstream features:

* **Absolute object velocities.** Right now a parked car reads at road
  speed because velocity is measured in a moving sensor frame. The
  dashboard has to caveat every number.
* **Multi-sweep map accumulation.** The most visible weakness today is
  that a single sweep paints the ground as rings with gaps, not as a
  surface. Accumulating 5–10 sweeps in the world frame fills the ground
  completely and makes the map look like a map.
* **A real memory argument.** Accumulated maps are where uniform 5 cm
  grids actually hurt. The current honest in-scenario figure is 4.3×;
  accumulation over a 100 m drive pushes this toward the 26× design
  capacity number, measured rather than derived.
* **Ego-motion-compensated dynamic detection.** Once static geometry is
  registered across sweeps, anything that *doesn't* register is dynamic —
  free motion segmentation, no labels required.

### 2. Fix the exporter's ego self-hits  ·  ~30 min
70% of valid returns are the sensor hitting its own vehicle, labelled
`ClassID 1 (Car)`. `preprocess.clean` filters them, but any statistic
computed from the raw PCDs before that filter is wrong. Mask the ego actor
in `exportRawLidarWithIntensity.m` at source.

### 3. Give the scenario real terrain  ·  ~1 hour
The road is flat to within 4 cm over 57 m, so the elevation layer is
nearly constant and curb/pothole detection has nothing to find — the
detectors are implemented and run, and correctly report ~zero. Add road
banking, a kerb line, or a speed bump in Driving Scenario Designer and the
elevation and curb layers immediately carry signal. **This is the single
cheapest way to make the 2.5D claim visually obvious**: right now the
oblique view's height comes almost entirely from vehicles.

---

## Tier 1 — the segmentation model

The pipeline consumes `class_id` per point. Any segmenter that produces
that array drops in with no interface change.

### 4. Get the existing PointNet++ work to inference  ·  ~1 day
`scripts/pointnet/` already has SemanticKITTI fine-tuning. Finish it to
the point where it emits per-point labels for a MATLAB sweep, then route
`preprocess.clean` to read predicted labels instead of ground-truth ones.

**Keep both paths.** The fidelity metric in `metrics.py` currently
measures representation loss against ground truth. Once predictions
exist, run it twice — on predictions and on ground truth — and the
difference cleanly separates *network* error from *grid* error. Almost
nobody does this, and it directly answers "high accuracy in object
classification across varying distances" with an attributable number.

### 5. Prefer a sparse convolutional network over PointNet++  ·  ~2 days
PointNet++'s ball-query grouping is the slowest part of any outdoor
pipeline and it scales badly past ~50 k points. For an automotive sweep,
MinkowskiEngine / SpConv / TorchSparse with a voxelised U-Net is both
faster and more accurate on road scenes.

There is an argument here that is *specific to this project*: a sparse
conv net voxelises its input anyway. **Feed it the foveated grid cells
directly instead of raw points.** Near the vehicle it sees 5 cm detail; far
away it sees 40 cm cells — which is exactly the resolution it can extract
information from at that range, given how few points land there. That
turns foveation from a post-processing step into an *inference-cost*
reduction, and makes the whole system one idea instead of two.

### 6. Train on the foveated representation  ·  ~2 days
Per-cell features are already computed: `z_ground`, `z_max`, `roughness`,
`obstacle_top`, `overhead_min`, `intensity`, `n_pts`. That is a 7-channel
2.5D image on a variable-resolution lattice. A small U-Net over the ring
pyramid — each ring is one level, already dyadically nested, so pooling
between them is exact — classifies cells directly. Orders of magnitude
cheaper than point-wise inference, and the quadtree nesting is what makes
the pooling well-defined.

---

## Tier 2 — differentiators for the demo

### 7. Information-theoretic LOD schedule  ·  ~half day
The current 10/20/40/100 m schedule is hand-picked. Derive it instead:
LiDAR point density falls as 1/r², so cell size should grow as r to keep
roughly constant points-per-cell. Measure points-per-cell per ring from
the data (the pipeline already reports it), fit the exponent, and
*derive* the schedule. Then show a sweep of schedules on the
accuracy-versus-memory plane and mark the knee.

A judge asking "why 10 metres?" is the most likely hard question, and
"because we measured the density falloff" is a much better answer than
"the brief said so".

### 8. Accuracy-vs-memory Pareto plot  ·  ~2 hours
`metrics.py` can already build any schedule via `uniform_config` and score
it. Sweep ~10 schedules, plot round-trip accuracy against bytes, and show
the foveated points dominating the uniform ones. One chart that makes the
entire argument.

### 9. Multi-resolution occupancy fusion over time  ·  ~1 day (needs #1)
Log-odds occupancy per cell, updated across sweeps with ray evidence
already computed by `carve_free_space`. Gives persistence (objects stay
mapped when briefly occluded), noise rejection, and a genuine
free/occupied/unknown map rather than a per-sweep snapshot.

### 10. Hardware-grounded latency  ·  ~half day
Current numbers are single-threaded NumPy on a laptop. Two cheap
credibility wins: report the per-stage breakdown against the 100 ms sensor
budget (the dashboard does this), and port the aggregation kernel to Numba
or a small C extension to show the same structure at 1–2 ms. The
representation is the contribution; showing it is not the bottleneck
matters.

---

## Tier 3 — polish if time allows

* **Rolling-buffer memory demo.** Show live RSS of the process against a
  simulated uniform 5 cm grid over a 100 m drive. Memory as a *time
  series* is far more convincing than a static ratio.
* **Failure-case slide.** The `frame_00010` case where the ego is boxed in
  by a truck and the planner correctly returns a dashed best-effort path
  is a better demo than any success case. Systems that fail gracefully
  read as engineered; systems that only ever succeed read as demoed.
* **ROS 2 / rosbag input path.** `io_pcd.py` is the only file that knows
  about the file format. A `sensor_msgs/PointCloud2` reader behind the
  same `Frame` dataclass makes the whole thing deployable, which is a
  different class of claim from "it runs on our export".
* **Constant-velocity Kalman filter** in `Tracker` to replace the
  exponential smoothing, with Mahalanobis gating instead of a fixed 4 m
  radius.

---

## What I would do with one day

1. Ego pose export (#1) — half a day including accumulation.
2. Terrain in the scenario (#3) — one hour, makes 2.5D visible.
3. Pareto plot (#8) — two hours, makes the argument in one chart.

That order maximises what changes on screen per hour spent.
