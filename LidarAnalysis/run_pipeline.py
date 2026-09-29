#!/usr/bin/env python3
"""Build foveated 2.5D maps for a whole scenario and write the dashboard bundle.

    python run_pipeline.py --scenario road_label_test

Produces, under outputs/<scenario>/:
    manifest.json        config, per-frame index, aggregate metrics
    cells/NNNN.bin       packed float32 cell buffer per frame
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from foveated.config import GridConfig, PipelineConfig           # noqa: E402
from foveated.export import CELL_COLUMNS, free_polygon, pack_cells, write_json  # noqa: E402
from foveated.grid import build, carve_free_space                 # noqa: E402
from foveated.io_pcd import list_frames, load_class_map, load_pcd  # noqa: E402
from foveated.metrics import (aggregate, capacity_report, coverage_report,      # noqa: E402
                              evaluate_frame)
from foveated.planner import plan                                 # noqa: E402
from foveated.preprocess import clean                             # noqa: E402
from foveated.semantics import Tracker, extract_instances, terrain_analysis  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="road_label_test")
    ap.add_argument("--data-root", default="data/labeled")
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N frames")
    ap.add_argument("--no-baselines", action="store_true",
                    help="skip the uniform-grid comparisons (much faster)")
    ap.add_argument("--no-coverage", action="store_true",
                    help="skip the observed-area memory analysis")
    ap.add_argument("--dt", type=float, default=0.1, help="frame period, seconds")
    args = ap.parse_args()

    folder = os.path.join(args.data_root, args.scenario)
    if not os.path.isdir(folder):
        print(f"no such scenario folder: {folder}", file=sys.stderr)
        return 1

    pcfg = PipelineConfig(scenario=args.scenario, data_root=args.data_root,
                          out_root=args.out_root)
    pcfg.grid.validate()
    out_dir = os.path.join(args.out_root, args.scenario)
    os.makedirs(os.path.join(out_dir, "cells"), exist_ok=True)

    paths = list_frames(folder)
    if args.limit:
        paths = paths[:args.limit]
    print(f"{len(paths)} frames in {folder}")

    # Warm the lazily-imported SciPy extension modules before timing.
    # Left cold, the first frame pays ~220 ms of import cost and the mean
    # latency becomes a statement about Python's import system rather
    # than about the mapper.
    import scipy.sparse.csgraph  # noqa: F401
    import scipy.spatial         # noqa: F401

    tracker = Tracker(dt=args.dt)
    frames_meta = []
    reports = []
    coverages = []
    t_start = time.time()

    for i, path in enumerate(paths):
        t_io = time.perf_counter()
        raw = load_pcd(path)
        io_ms = (time.perf_counter() - t_io) * 1000

        t_pre = time.perf_counter()
        cf = clean(raw, pcfg.grid)
        pre_ms = (time.perf_counter() - t_pre) * 1000

        rep = evaluate_frame(cf, pcfg, with_baselines=not args.no_baselines)
        g = rep["grid"]
        terrain_analysis(g)
        instances = extract_instances(g)
        tracks = tracker.update(instances)
        pl = plan(g)

        if not args.no_coverage:
            coverages.append(coverage_report(g, pcfg))

        buf = pack_cells(g)
        with open(os.path.join(out_dir, "cells", f"{i:05d}.bin"), "wb") as fh:
            fh.write(buf.tobytes(order="C"))

        # Re-read timings *after* terrain/instances have run; the report
        # was built before those stages existed on the grid.
        lat = {k: round(v * 1000, 3) for k, v in g.timings.items()}
        lat["io"] = round(io_ms, 3)
        lat["preprocess"] = round(pre_ms, 3)
        lat["plan"] = round(pl.ms, 3)
        # Per-frame mapping latency excludes disk I/O and the planner:
        # in a live stack the sweep arrives in memory and planning runs
        # in its own thread.
        lat["map_total"] = round(
            sum(lat.get(k, 0.0) for k in
                ("preprocess", "index", "aggregate", "layers", "semantics",
                 "freespace", "terrain", "instances")), 3)

        frames_meta.append({
            "index": i,
            "name": raw.name,
            "n_cells": int(g.n_cells),
            "cells_per_level": [int((g.level == l).sum()) for l in range(len(pcfg.grid.levels()))],
            "points": cf.stats,
            "latency_ms": lat,
            "memory": {k: {"cells": v["cells"], "bytes": v["bytes"],
                           "ratio_vs_foveated": v["ratio_vs_foveated"], "label": v["label"]}
                       for k, v in rep["memory"]["entries"].items()},
            "fidelity": rep["fidelity"]["foveated"],
            "instances": [inst.to_dict() for inst in instances],
            "tracks": tracks,
            "plan": pl.to_dict(),
            "free_profile": free_polygon(g),
            "counts": {
                "drivable": int(g.drivable.sum()),
                "curb": int(g.curb.sum()),
                "low_clearance": int(g.low_clearance.sum()),
                "dynamic_cells": int((g.kind == 3).sum()),
                "static_cells": int((g.kind == 2).sum()),
                "terrain_cells": int((g.kind == 1).sum()),
            },
        })
        reports.append(rep)
        if (i + 1) % 5 == 0 or i == len(paths) - 1:
            print(f"  [{i+1}/{len(paths)}] {raw.name}  {g.n_cells:6d} cells  "
                  f"map {lat['map_total']:6.2f} ms  {len(instances)} objects")

    agg = aggregate(reports)
    cap = capacity_report(pcfg.grid, pcfg,
                          per_cell_bytes=reports[0]["memory"]["per_cell_bytes"])

    cov_avg = {}
    if coverages and coverages[0]:
        keys = [k for k in coverages[0] if isinstance(coverages[0][k], dict)]
        for k in keys:
            cov_avg[k] = {
                "cells": float(np.mean([c[k]["cells"] for c in coverages])),
                "bytes": float(np.mean([c[k]["bytes"] for c in coverages])),
            }
            if "ratio_vs_foveated" in coverages[0][k]:
                cov_avg[k]["ratio_vs_foveated"] = round(float(
                    np.mean([c[k]["ratio_vs_foveated"] for c in coverages])), 2)

    lat_keys = ("io", "preprocess", "index", "aggregate", "layers", "semantics",
                "freespace", "terrain", "instances", "plan", "map_total")
    def lat_stats(k: str) -> dict:
        v = np.array([f["latency_ms"].get(k, 0.0) for f in frames_meta], float)
        return {"mean": round(float(v.mean()), 3),
                "median": round(float(np.median(v)), 3),
                "p95": round(float(np.percentile(v, 95)), 3),
                "max": round(float(v.max()), 3)}

    lat_avg = {k: lat_stats(k) for k in lat_keys}

    manifest = {
        "scenario": args.scenario,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dt": args.dt,
        "config": pcfg.grid.to_dict(),
        "class_map": load_class_map(folder),
        "cell_columns": CELL_COLUMNS,
        "n_frames": len(frames_meta),
        "frames": frames_meta,
        "aggregate": {
            "latency_ms": lat_avg,
            "map_fps": round(1000.0 / lat_avg["map_total"]["median"], 1),
            "map_fps_p95": round(1000.0 / lat_avg["map_total"]["p95"], 1),
            "mean_cells": agg["mean_cells"],
            "memory_occupied_only": agg["memory"],
            "memory_observed_area": cov_avg,
            "memory_design_capacity": cap,
            "fidelity": agg["fidelity"],
        },
        "wall_clock_s": round(time.time() - t_start, 2),
    }
    write_json(os.path.join(out_dir, "manifest.json"), manifest)

    size = sum(os.path.getsize(os.path.join(out_dir, "cells", f))
               for f in os.listdir(os.path.join(out_dir, "cells")))
    print(f"\nwrote {out_dir}  ({size/2**20:.1f} MiB of cell buffers) "
          f"in {manifest['wall_clock_s']} s")
    mt = lat_avg["map_total"]
    print(f"mapping latency  median {mt['median']:.2f} ms  p95 {mt['p95']:.2f} ms "
          f"-> {manifest['aggregate']['map_fps']:.0f} FPS "
          f"({manifest['aggregate']['map_fps_p95']:.0f} FPS at p95)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
