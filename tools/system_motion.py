"""Step 4: a moving receiver through the real server (system run).

Called by run_experiment.py for an experiment file with ``system_cases`` (see
experiments/receiver_system.json); not meant to be run on its own.

For each case and seed, and each receiver pose the node reports with its
frames (``system_pose_sources``):

``none``   the receiver moves but reports nothing: the server solves it at its
           registered position, as it would today (S)
``truth``  the true pose
``kf``     the step 3b KF estimate from GNSS (realisation ``system_realisation``)

the orchestrator flies the scripted run through a fresh backend in real time
(run_experiment._run_system_seed), with receivers moving and each frame
carrying ``rx_motion``.  A static case runs once, with no pose.

Each run is then solved offline from its own recording with the matching
algorithm (S for ``none``, GD in the solver for ``truth``/``kf``, with the
same pose the server was sent), so the server's solves can be compared with
the offline solve of the very same frames.  Writes ``summary.csv`` and
prints the table.
"""

import csv
import datetime
import json
import math
import os
import subprocess
import sys

import gnss_pose
import offline_solve
import run_experiment

FIELDS = [
    "case", "seed", "pose", "server_solves", "server_pos_rmse_m", "server_vel_rmse_ms", "offline_solves",
    "offline_pos_rmse_m", "offline_vel_rmse_ms", "paired", "paired_pos_diff_median_m", "paired_vel_diff_median_ms",
]  # fmt: skip


def _rms(v):
    v = [x for x in v if x is not None]
    return math.sqrt(sum(x * x for x in v) / len(v)) if v else None


def _median(v):
    v = sorted(v)
    return v[len(v) // 2] if v else None


def runs_for(exp: dict) -> list[tuple[dict, dict, str]]:
    seeds = set(exp.get("system_seeds") or [r["seed"] for r in exp["runs"]])
    out = []
    for case in exp["cases"]:
        if case["id"] not in exp["system_cases"]:
            continue
        static = case["motion"].get("type", "static") == "static"
        for run in exp["runs"]:
            if run["seed"] not in seeds:
                continue
            for pose in ["none"] if static else exp.get("system_pose_sources", ["none", "truth", "kf"]):
                out.append((case, run, pose))
    return out


def run_dir(out: str, case: dict, run: dict, pose: str) -> str:
    return os.path.join(out, "cases", case["id"], f"seed-{run['seed']}", f"system_{pose}")


def _pose_file(exp: dict, case: dict, run: dict, path: str) -> str:
    ests = gnss_pose.estimates(exp, case, run["seed"], exp.get("system_realisation", 0), "kf")
    keys = ("east_m", "north_m", "up_m", "vel_east_ms", "vel_north_ms", "vel_up_ms")
    with open(path, "w", encoding="utf-8") as f:
        for e in ests:
            if e is not None:
                f.write(json.dumps({"t_s": e["t_s"], **{k: e[k] for k in keys}}) + "\n")
    return path


def _ticks(d: str) -> list[int]:
    with open(os.path.join(d, "rx_trajectory.ndjson"), encoding="utf-8") as f:
        return [json.loads(line)["t_ms"] for line in f if line.strip()]


def offline_rows(exp: dict, case: dict, run: dict, pose: str, d: str) -> list[dict]:
    """The offline solve of this system recording, with what the server was told."""
    if pose == "none" or case["motion"].get("type", "static") == "static":
        return offline_solve.solve_run(d, exp.get("solver"), "stationary")
    if pose == "truth":
        return offline_solve.solve_run(d, exp.get("solver"), "geometric_doppler_model")
    with open(os.path.join(d, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    ests = gnss_pose.estimates(exp, case, run["seed"], exp.get("system_realisation", 0), "kf")
    by_tick = gnss_pose.poses_from_estimates({**exp, "start_time_ms": 0}, nodes, ests)
    # The recording is stamped by the wall clock; tick i of the estimates is the
    # i-th recorded tick.
    poses = {t_ms: by_tick.get(i * 1000) for i, t_ms in enumerate(_ticks(d), start=1)}
    poses = {k: v for k, v in poses.items() if v}
    rows = offline_solve.solve_run(d, exp.get("solver"), "geometric_doppler_model", pose_estimate=poses)
    # A tick past the last estimate (the real-time run can record one more
    # than the experiment's duration) has no pose and would be solved as S.
    return [r for r in rows if r["t_ms"] in poses]


def server_rows(d: str) -> list[dict]:
    """Published 3-node server solves, scored by analyze_run (raw single-second solve)."""
    path = os.path.join(d, "analysis", "solves.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r.get("err_horiz_m") in (None, "") or r.get("n_nodes") in (None, "") or float(r["n_nodes"]) < 3:
            continue
        out.append({"t_ms": int(float(r["t_ms"])), "pos": float(r["err_horiz_m"]),
                    "vel": float(r["verr_horiz_ms"]) if r.get("verr_horiz_ms") not in (None, "") else None})  # fmt: skip
    return out


def compare(exp: dict, case: dict, run: dict, pose: str, d: str, t0: float) -> dict:
    srv = server_rows(d)
    off = [r for r in offline_rows(exp, case, run, pose, d) if r.get("published") and r.get("t_s", 0) >= t0]
    first = _ticks(d)[0]
    srv = [r for r in srv if (r["t_ms"] - first) / 1000.0 + 1.0 >= t0]
    by_t = {r["t_ms"]: r for r in off}
    paired = [(s, by_t[min(by_t, key=lambda t: abs(t - s["t_ms"]))]) for s in srv if by_t]
    paired = [(s, o) for s, o in paired if abs(o["t_ms"] - s["t_ms"]) <= 600]
    return {
        "case": case["id"], "seed": run["seed"], "pose": pose,
        "server_solves": len(srv), "server_pos_rmse_m": _rms([r["pos"] for r in srv]),
        "server_vel_rmse_ms": _rms([r["vel"] for r in srv]),
        "offline_solves": len(off), "offline_pos_rmse_m": _rms([r["err_horiz_m"] for r in off]),
        "offline_vel_rmse_ms": _rms([r.get("verr_horiz_ms") for r in off]),
        "paired": len(paired),
        "paired_pos_diff_median_m": _median([abs(s["pos"] - o["err_horiz_m"]) for s, o in paired]),
        "paired_vel_diff_median_ms": _median(
            [abs(s["vel"] - o["verr_horiz_ms"]) for s, o in paired if s["vel"] is not None]
        ),
    }  # fmt: skip


def report(out: str, exp: dict) -> list[dict]:
    t0 = exp.get("steady_state_from_s", 0.0)
    rows = []
    for case, run, pose in runs_for(exp):
        d = run_dir(out, case, run, pose)
        if os.path.exists(os.path.join(d, "frames.ndjson")):
            rows.append(compare(exp, case, run, pose, d, t0))
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: ("" if v is None else v) for k, v in r.items()} for r in rows)
    fmt = lambda v, p=1: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
    print(
        f"{'case':22s} {'pose':5s} | server: n  pos m  vel m/s | offline: n  pos m  vel m/s | paired n, |dpos| |dvel|"
    )
    for r in rows:
        print(f"{r['case']:22s} {r['pose']:5s} | {r['server_solves']:9d} {fmt(r['server_pos_rmse_m']):>6} "
              f"{fmt(r['server_vel_rmse_ms'], 2):>7} | {r['offline_solves']:10d} {fmt(r['offline_pos_rmse_m']):>6} "
              f"{fmt(r['offline_vel_rmse_ms'], 2):>7} | {r['paired']:4d} {fmt(r['paired_pos_diff_median_m']):>6} "
              f"{fmt(r['paired_vel_diff_median_ms'], 2):>6}")  # fmt: skip
    return rows


def run(exp: dict, args, repo: str, commit) -> None:
    if args.reanalyze:
        out = os.path.abspath(args.reanalyze)
        report(out, exp)
        return
    if not args.backend_dir:
        raise SystemExit("needs --backend-dir (the retina-server/backend folder)")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = os.path.abspath(args.out or os.path.join(repo, "runs", exp["name"], stamp))
    os.makedirs(out, exist_ok=True)
    backend_python = args.backend_python or sys.executable
    todo = runs_for(exp)
    for k, (case, run_, pose) in enumerate(todo, start=1):
        d = run_dir(out, case, run_, pose)
        if os.path.exists(os.path.join(d, "analysis", "solves.csv")):
            print(f"[{k}/{len(todo)}] {case['id']} seed {run_['seed']} pose {pose}: already done")
            continue
        os.makedirs(os.path.dirname(d), exist_ok=True)
        extra = []
        if case["motion"].get("type", "static") != "static":
            rx_pose = pose if pose in ("none", "truth") else _pose_file(exp, case, run_, d + "_rx_pose.ndjson")
            extra = ["--receiver-motion", json.dumps(case["motion"]), "--rx-pose", rx_pose]
        print(f"[{k}/{len(todo)}] {case['id']} seed {run_['seed']} pose {pose}: running the server in real time")
        run_experiment._run_system_seed(exp, run_, d, os.path.abspath(args.backend_dir), backend_python, tuple(extra))
        subprocess.run([sys.executable, os.path.join(repo, "tools", "analyze_run.py"), d, "--no-maps"], check=True,
                       stdout=subprocess.DEVNULL)  # fmt: skip
    report(out, exp)
    manifest = {
        "experiment": exp["name"], "experiment_file": exp["_path"],
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv], "runs": [[c["id"], r["seed"], p] for c, r, p in todo],
        "code": {"retina_simulation": commit(repo), "backend": commit(os.path.dirname(os.path.abspath(args.backend_dir)))},
    }  # fmt: skip
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {out}")
