#!/usr/bin/env python3
"""Flask server for the foveated 2.5D mapping dashboard.

    python dashboard/app.py --scenario road_label_test

Serves the precomputed bundle from ``outputs/<scenario>/`` and, on
demand, re-plans a route to a goal the user clicks on the map.  Replanning
is live rather than precomputed: the grid is rebuilt from the source sweep
and A* re-run, which is the honest way to show that the map is queryable,
not a picture of one.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional

from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from foveated.config import GridConfig, PipelineConfig      # noqa: E402
from foveated.grid import FoveatedGrid, build, carve_free_space  # noqa: E402
from foveated.io_pcd import list_frames, load_pcd           # noqa: E402
from foveated.planner import plan                           # noqa: E402
from foveated.preprocess import clean                       # noqa: E402
from foveated.semantics import terrain_analysis             # noqa: E402

app = Flask(__name__, static_folder=os.path.join(ROOT, "dashboard", "static"),
            static_url_path="")

STATE: Dict[str, object] = {}
_GRID_CACHE: Dict[int, FoveatedGrid] = {}


def get_grid(index: int) -> Optional[FoveatedGrid]:
    if index in _GRID_CACHE:
        return _GRID_CACHE[index]
    paths = STATE["frame_paths"]
    if not (0 <= index < len(paths)):
        return None
    cfg: GridConfig = STATE["cfg"]
    c = clean(load_pcd(paths[index]), cfg)
    g = build(c, cfg)
    carve_free_space(g, c)
    terrain_analysis(g)
    if len(_GRID_CACHE) > 64:
        _GRID_CACHE.clear()
    _GRID_CACHE[index] = g
    return g


@app.route("/")
def index() -> Response:
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/manifest")
def manifest() -> Response:
    return send_from_directory(STATE["out_dir"], "manifest.json",
                               mimetype="application/json")


@app.route("/api/cells/<int:index>")
def cells(index: int) -> Response:
    path = os.path.join(STATE["out_dir"], "cells", f"{index:05d}.bin")
    if not os.path.exists(path):
        return Response(status=404)
    with open(path, "rb") as fh:
        data = fh.read()
    return Response(data, mimetype="application/octet-stream",
                    headers={"Cache-Control": "public, max-age=3600"})


@app.route("/api/replan", methods=["POST"])
def replan() -> Response:
    body = request.get_json(force=True)
    index = int(body.get("index", 0))
    g = get_grid(index)
    if g is None:
        return jsonify({"error": "bad frame index"}), 400
    goal = body.get("goal")
    goal_t = (float(goal[0]), float(goal[1])) if goal else None
    res = plan(g, goal=goal_t, plan_mult=int(body.get("plan_mult", 4)))
    return jsonify(res.to_dict())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="road_label_test")
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data", "labeled"))
    ap.add_argument("--out-root", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5055)
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.scenario)
    if not os.path.exists(os.path.join(out_dir, "manifest.json")):
        print(f"No bundle at {out_dir}.\n"
              f"Run:  python run_pipeline.py --scenario {args.scenario}",
              file=sys.stderr)
        return 1

    STATE["out_dir"] = out_dir
    STATE["cfg"] = PipelineConfig(scenario=args.scenario).grid
    STATE["frame_paths"] = list_frames(os.path.join(args.data_root, args.scenario))
    print(f"serving {args.scenario} on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
