# Space-Based Autonomous Driving Architecture for Unstructured Roads

A complete closed-loop automated-driving stack built in MATLAB / Simulink, designed from first
principles for roads without reliable lane structure — the mixed-traffic environment of Indian
urban and highway driving.

**Status:** research prototype · simulation-based · MATLAB R2026a / Simulink 26.1

---

## The idea

Conventional driving automation asks *"which lane am I in, and what is the vehicle ahead
doing?"* On roads where markings are absent or ignored, and where cars, two-wheelers,
autorickshaws, trucks and pedestrians share the same strip of road at different speeds, that
question has no answer.

This architecture asks a different one:

> **Where is there room, for how long, and in which direction?**

Free space — resolved across **4 directional sectors** and **7 time horizons out to 5 seconds** —
replaces lane position as the primary representation. Every design decision downstream follows
from that substitution.

---

## Architecture

Twelve subsystems in a genuine closed loop. The vehicle's own commands change what it perceives
on the next cycle.

```
RoadRunner scenario
   → Vision / Radar / LiDAR
      → Perception
         → Sensor Fusion
            → Tracking
               → Trajectory Prediction
                  → Environment Model (ICE)
                     → Risk Engine
                        → Planning
                           → Collision Avoidance
                              → Decision Logic
                                 → Control
                                    → Vehicle Dynamics
                                       ↺ back to the sensors
```

Executes at 10 Hz (0.1 s fixed step). All runtime code is fixed-size and
code-generation-ready: no dynamic allocation, no variable-size signals, bounded loops only.

---

## What each stage does

**Perception.** Three sensors with deliberately different physics. The LiDAR chain is the
deepest: preprocessing → ground segmentation → ground height model → BEV occupancy clustering →
object feature extraction → detection formatting, operating on a **101 × 2251 organised cloud —
227,351 grid cells per frame**.

**Sensor Fusion.** Detections are normalised onto a common contract, gated by 2-D Mahalanobis
distance at χ²(2, 0.99) = 9.21 using the full covariance, grouped by **mutual-best matching**
(symmetric, therefore order-independent), and combined in information form. Position and
velocity are fused separately because they have different observability.

**Tracking.** 512 fixed slots. Ego motion removed by an exact constant-turn-rate arc, tracks
propagated with a constant-velocity model and DWNA process noise, associated by **optimal
Munkres assignment** over union-find components, updated in **Joseph form**, and managed by a
2-of-3 confirmation lifecycle with coast and lost tiers.

**Prediction.** Every active track propagated to **seven horizons** `[0.5 1.0 1.5 2.0 3.0 4.0
5.0] s` using a closed-form accumulation of the process noise, so both the mean and the
covariance are available at each horizon.

**Environment Model (ICE).** Interaction Constraint Estimation evaluates an **escape lattice** —
4 sectors × 7 horizons × 32 escape candidates, **896 cells per frame** — reporting the best
achievable escape margin in metres, how many escapes are clear, and a three-state certification
per cell.

**Planning.** 35 candidate manoeuvres (5 longitudinal × 7 lateral) generated as
constant-curvature arcs, then screened through five stages: dynamic feasibility → spatial
envelopes → comparative evaluation → lexicographic selection → temporal continuity.

**Collision Avoidance.** Each candidate tested against every predicted object footprint at all
seven horizons using a **separating-axis test between oriented boxes**, producing **245 signed
separations per frame** (`CandMinSeparation [35 × 7]`).

**Control & Vehicle.** Stateless feedforward laws — lateral `δ = atan(L_WB · κ)`, longitudinal
from the selected speed profile — driving a rear-axle kinematic bicycle model.

---

## Validated results

Every stage checked against an **independently written oracle**, not against itself.

| check | result |
|---|---|
| Interpreted vs compiled, every stage | **0 differing cells** |
| Fusion assembly | 0 of **269,892** cells |
| Prediction | 0 of **1,279,488** cells |
| Fused position vs independent recomputation | **1.07e-14 m** |
| Munkres optimality vs brute force, 400 cases | worst excess **0.000e+00** |
| Ego motion arc vs independent calculation | max diff **0.000e+00** |
| Fused covariance | positive definite everywhere, `P ⪯ Rᵢ` for every contributor |

### Measured characteristics

| | |
|---|---|
| Mean detections per frame | LiDAR **47.6** · Radar **1.3** · Vision **1.1** |
| Fused objects seen by exactly one sensor | **96.9 %** |
| Fused objects arriving with no usable velocity | **95.7 %** |
| Peak concurrent confirmed tracks | **107** |
| Radar longitudinal velocity accuracy | σₓ **0.31 m/s** |
| Vision lateral velocity accuracy | σᵧ **0.45 m/s** |
| Oriented-box separation tests per frame | ~**11,300** |

---

## Mixed-traffic demonstration

A 40-second closed-loop run on a divided carriageway with **seven actors** — pedestrian,
bicycle, truck, autorickshaw, lead car and two opposing-flow vehicles.

| | |
|---|---|
| **Collision-free** | **0 footprint-overlap frames across all 7 actors** |
| **Remained on the carriageway** | footprint Y ∈ [−4.40, +3.73] m within a 14 m road |
| Pedestrian blockage | negotiated with a continuous curve, cleared by **3.78 m** |
| Opposing-flow passes | **2.40 m** and **2.96 m** |
| Warnings | **none** |

At the pedestrian encounter the three sensors reported **Vision 1 · Radar 0 · LiDAR 50** — a
single frame that makes the case for the fusion architecture better than any diagram.



---

## Original technical contributions

Derived and verified for this project.

**Closed-form multi-horizon process-noise accumulation** — replaces up to 50 iterations per
track per horizon with a direct expression. Verified to **2.132e-14** against explicit repeated
propagation.

**Radar radial-velocity measurement shift** — reconciles an ego-relative range rate with an
absolute velocity state, with a proof that the rotational term is identically zero in the radial
projection. Median innovation improved from 14.85 to **0.091 m/s**; gate rate from 77 % to 15 %.

**Interaction Constraint Estimation (ICE)** — the escape-lattice formulation that turns "free
space" into an actionable measurement in metres, resolved by direction and time.

**Closed-form conditioned arc length** — `S_req = 2·asin(√(κL·|y|/2)) / κL`, an exact
replacement for an iterative solve, verified to **3.553e-15**.

**Ego speed recovered from the existing transform** — `v = (c/dt)·(Δψ/2)/sin(Δψ/2)`, accurate to
**7.105e-15 m/s**, requiring no new sensor input or signal.

**Numerically stable minimum eigenvalue** — `λ_min = det/λ_max`, with relative error
**0.000e+00** at a condition ratio of 1e14.

**Measurement-driven clustering** — four methods compared on identical data; BEV occupancy at
0.30 m cells selected on evidence, preserving the most vulnerable road users while minimising
over-merging.

---

## Engineering discipline

- **No ground truth in the runtime path.** No actor ID, class ID, material ID, target index,
  scenario trajectory or frame count — audited to zero hits in executable code.
- **LiDAR intensity is never used as a class cue**, because in simulation it is keyed to
  material and would constitute indirect ground-truth leakage.
- **Every parameter carries a declared provenance class**, distinguishing measured values from
  engineering assumptions.
- **Interpreted and compiled execution compared at every stage**, because the compiled path is
  the deployment path.
- **First-simulation-of-a-fresh-session regression rule**, since the sensor noise is unseeded.
- **Every model change is backed up, dry-run, and structurally verified** before being saved.

---





## Toolchain

MATLAB R2026a · Simulink 26.1 · Automated Driving Toolbox 26.1 · Computer Vision Toolbox ·
Lidar Toolbox · Navigation Toolbox · RoadRunner

---

## Demo video

A recorded closed-loop run of the full stack driving the mixed-traffic scenario.

**▶ [Watch the demo on YouTube](https://www.youtube.com/watch?v=vObXSjrpF-c)**

[![Demo video: closed-loop mixed-traffic run](https://img.youtube.com/vi/vObXSjrpF-c/hqdefault.jpg)](https://www.youtube.com/watch?v=vObXSjrpF-c)

---

## Repository layout

```
.
├── Mathworks/                       MATLAB / Simulink source of the closed-loop stack
│   ├── simulink/                    ADAS_Main.slx, ADAS_Main_ClosedLoop.slx, staged test models
│   ├── driving scenario/            Driving Scenario Designer .mat scenarios (V1 → V2 closed loop)
│   ├── road_runner/                 RoadRunner scenes (.rrscene) and scenarios (.rrscenario)
│   ├── graphs/                      Result figures: scenario overview, manoeuvres, separation,
│   │                                ICE free-space margin, sensor counts, BEV views, tracking
│   └── video/                       Front / LiDAR / radar / mixed-traffic demo renders and stills
│
├── LidarAnalysis/                   Foveated 2.5D semantic LiDAR mapping pipeline
│   ├── src/foveated/                Library: preprocess, grid, semantics, planner, metrics, io_pcd,
│   │                                export, config
│   ├── scripts/                     Exploration notebooks (PCD, 3D, intensity, clustering, BEV,
│   │                                foveated grid) plus generation and video helpers
│   ├── dashboard/                   Visualisation dashboard (app.py + static assets)
│   ├── scenarios/ · models/         MATLAB scenario scripts and Simulink LiDAR test models
│   ├── data/labeled/ · tests/       Labelled clouds and grid-invariant tests
│   ├── run_pipeline.py              Pipeline entry point
│   ├── exportLidarData.m · exportRawLidarWithIntensity.m · relabelRoadPCDs.m
│   └── README_FOVEATED.md · setup.md · docs/ROADMAP.md · environment.yml
│
├── vehicle_behaviour_pipeline/      Modular Python behaviour pipeline
│   ├── perception_node.py           Detection front end
│   ├── camera_radar_fusion.py · sensor_fusion_node.py
│   ├── risk_score_calculator.py     Risk engine
│   ├── object_planner.py · trajectory_planner.py
│   ├── controller_node.py · realtime_vehicle_controller.py
│   ├── main_pipeline.py             End-to-end runner
│   └── validate_fusion_model.py/.ipynb
│
├── object_risk_score/               Standalone risk-scoring study
│   ├── sensor_fusion_engine.py · main_controller.py · simulink_controls.m
│   ├── radar_detections.csv · controller_output.txt
│   └── risk_score_report.md + risk / planner / trajectory plots
│
├── reactive_planner.m               Reactive planner (modular)
├── reactive_planner_all_in_one.m    Reactive planner (single-file variant)
├── simulate_reactive_planner.m      Planner simulation harness
│
├── YOLO_Inference/                  Detection inference and annotated-video tooling
│   ├── youtube_yolo_inference.ipynb · potholes_yolo_inference.ipynb ·
│   │   potholes_2_yolo_inference.ipynb · local_yolo_test.ipynb
│   ├── generate_annotated_video.py · combine_videos.py · combine_videos_input.py
│   ├── test_local.py · verify_controls.py
│   ├── Test_Images/ · matlab_data_4/  (navigation_controls.csv, rad.json, simulink_controls.m)
│   └── yolo_small_v3.pt
├── YOLO_Models/best.pt              Trained detection weights
├── Model/                           Training / evaluation notebooks (IDD + merged segmentation,
│                                    metric evaluation, testing)
├── kaggle_idd_yolo/                 IDD inference notebook and sample image
│
├── egovehicle_data/                 Raw ego + sensor logs (.mat) with JSON/CSV extractors
│                                    (process_sensor_data.m, export_radar_to_csv.m,
│                                    extract_mat_to_json.py)
├── matlab_data_2/ · matlab_data_3/  Exported MATLAB sessions: extractor, PDF generator, Results/
│                                    (radar + camera detections, fused JSON, sensor-data report)
├── MATLAB_Analysis_Scripts/         Offline analysis: camera, LiDAR PCD, sensor data, fusion
│                                    pipeline, trajectory prediction, .mat extraction, spec PDF
│
├── web_app/                         Demo web application
│   ├── backend/main.py              API server
│   └── frontend/                    Vite + React client (src/, public/, vite.config.js)
│
├── Output_Results/                  Generated artefacts: camera frames, camera video, fusion frames
├── PROJECT-RESULTS/                 Result screenshots
├── reports/                         matlab_sensor_data_report.md
└── Notebooks_and_Docs/              Problem statement, task, setup guides, data-ingestion notes,
                                     segmentation notebooks, Master_Autonomous_Driving_Spec.pdf
```

Large binaries are excluded by [.gitignore](.gitignore): `*.pcd`, `*.mat`, `*.mp4`, `*.pt`,
`*.onnx`, datasets, virtual environments, `node_modules/`, and MATLAB / Simulink cache
(`*.asv`, `*.slxc`, `slprj/`).

---
