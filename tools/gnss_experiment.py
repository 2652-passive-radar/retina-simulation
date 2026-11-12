"""Step 3c: GD with the receiver pose taken from GNSS instead of the truth.

Called by run_experiment.py for an experiment file with ``pose_sources`` (see
experiments/receiver_gnss.json); not meant to be run on its own.

Every step 2 recording (each motion case and seed, receiver Doppler on) is
re-recorded and fingerprint-checked as before, then solved by GD once per pose
source: ``truth`` once, and ``gnss`` / ``kf`` once per GNSS realisation, the
same GNSS draws step 3b analysed.  Solves land in each seed's
``analysis/offline_solves_gd_<source>[_r<k>].csv``; gnss_report.py turns them
into the figures and ``summary.csv``.
"""

import datetime
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import gnss_pose
import gnss_report
import motion_experiment
import motion_report
import offline_solve

_DEFAULT_MAX_JOBS = 8


def tasks_for(exp: dict, out: str) -> list[tuple]:
    tasks = []
    n_real = exp.get("realisations", 1)
    for case in exp["cases"]:
        for doppler in motion_report.dopplers_for(exp, case):
            for run in exp["runs"]:
                path = motion_report.seed_dir(out, case["id"], doppler, run["seed"])
                for source in exp["pose_sources"]:
                    for r in [None] if source == "truth" else range(n_real):
                        tasks.append((path, case, run["seed"], source, r))
    return tasks


def _solve(task: tuple, exp: dict) -> str:
    path, case, seed, source, r = task
    poses = gnss_pose.poses_for(path, exp, case, seed, r or 0, source)
    rows = offline_solve.solve_run(path, exp.get("solver"), "geometric_doppler", pose_estimate=poses)
    return offline_solve.write_rows(path, rows, gnss_pose.solve_name(source, r))


def solve_all(exp: dict, out: str, jobs: int | None) -> None:
    tasks = tasks_for(exp, out)
    slim = {k: v for k, v in exp.items() if k != "_base"}
    workers = max(1, min(len(tasks), jobs or min(os.cpu_count() or 1, _DEFAULT_MAX_JOBS)))
    print(
        f"Solving {len(tasks)} recordings x pose sources with GD on {workers} process{'es' if workers > 1 else ''}..."
    )
    t0 = time.monotonic()
    pending = list(tasks)

    def progress() -> None:
        n = len(tasks) - len(pending)
        if n % 10 == 0 or n == len(tasks):
            print(f"  {n}/{len(tasks)} ({time.monotonic() - t0:.0f} s)", flush=True)

    if workers > 1:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_solve, t, slim): t for t in tasks}
                for fut in as_completed(futures):
                    fut.result()
                    pending.remove(futures[fut])
                    progress()
        except BrokenProcessPool:
            print(
                f"  A worker process died ({len(pending)} solves left); finishing them one at a time. "
                "Use --jobs with a smaller number (or --jobs 1) to avoid this."
            )
    for t in list(pending):
        _solve(t, slim)
        pending.remove(t)
        progress()


def run(exp: dict, args, repo: str, commit) -> None:
    if args.system:
        raise SystemExit("--system is not supported for the GNSS experiment (offline only)")
    if args.reanalyze:
        out = os.path.abspath(args.reanalyze)
        summary = gnss_report.build(out, exp)
        gnss_report.print_table(summary)
        print(f"Rebuilt the analysis in {out}")
        return
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = os.path.abspath(args.out or os.path.join(repo, "runs", exp["name"], stamp))
    os.makedirs(out, exist_ok=True)

    t0 = time.monotonic()
    motion_experiment.record_all(exp, out, args.accept_changes)
    print(f"Recorded every case into {out} ({time.monotonic() - t0:.0f} s)")
    solve_all(exp, out, args.jobs)
    summary = gnss_report.build(out, exp)

    manifest = {
        "experiment": exp["name"],
        "experiment_file": exp["_path"],
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "solver": {**offline_solve.DEFAULT_SOLVER, **(exp.get("solver") or {})},
        "algorithm": "geometric_doppler",
        "pose_sources": exp["pose_sources"],
        "realisations": exp.get("realisations", 1),
        "gnss": exp["gnss"],
        "filter": exp["filter"],
        "steady_state_from_s": exp.get("steady_state_from_s", 0.0),
        "cases": exp["cases"],
        "runs": [{k: r[k] for k in ("seed", "flight", "plan")} for r in exp["runs"]],
        "code": {"retina_simulation": commit(repo)},
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    gnss_report.print_table(summary)
    print(f"\nWrote {out} ({time.monotonic() - t0:.0f} s in all)")
