"""Run the receiver-motion study: every case x receiver-Doppler setting x seed
recorded, solved offline by every algorithm, and reported.

Called by run_experiment.py for an experiment file with ``cases`` (see
experiments/receiver_motion.json); not meant to be run on its own.

Output: ``runs/<name>/<date_time>/`` with ``manifest.json``, ``summary.csv``,
``overview.png``, ``receiver_paths.png``, ``paths_map.png`` and
``cases/<case>/`` holding that case's figures and, per receiver-Doppler
setting and seed, the recording and ``analysis/offline_solves_<algorithm>.csv``.

Each recording's fingerprint is stored in the experiment file under
``fingerprints`` the first time and checked on every later run, as in step 0.
"""

import datetime
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import motion_report
import offline_solve

from retina_simulation import experiment
from retina_simulation.motion import motion_from


def _solve(task: tuple) -> tuple[str, str, int]:
    seed_path, solver_cfg, algorithm = task
    rows = offline_solve.solve_run(seed_path, solver_cfg, algorithm)
    offline_solve.write_rows(seed_path, rows, f"offline_solves_{algorithm}.csv")
    return seed_path, algorithm, sum(1 for r in rows if r.get("published"))


def _key(case_id: str, doppler: bool) -> str:
    return f"{case_id}|{motion_report.doppler_dir(doppler)}"


def record_all(exp: dict, out: str, accept_changes: bool) -> None:
    stored = exp.setdefault("fingerprints", {})
    changed = False
    for case in exp["cases"]:
        motion = motion_from(case["motion"])
        for doppler in motion_report.dopplers_for(exp, case):
            prints = stored.setdefault(_key(case["id"], doppler), {})
            for run in exp["runs"]:
                path = motion_report.seed_dir(out, case["id"], doppler, run["seed"])
                got = experiment.record_run(exp, run, path, motion, rx_doppler=doppler)
                want = prints.get(str(run["seed"]))
                if want is None:
                    prints[str(run["seed"])] = got
                    changed = True
                    continue
                diff = experiment.fingerprint_mismatch(want, got)
                if diff and not accept_changes:
                    raise SystemExit(
                        f"{case['id']} ({motion_report.doppler_dir(doppler)}), seed {run['seed']}: the simulator no "
                        "longer reproduces this recording:\n  "
                        + "\n  ".join(diff)
                        + "\nRe-run with --accept-changes to adopt the new recording as the reference."
                    )
                if diff:
                    prints[str(run["seed"])] = got
                    changed = True
    if changed:
        experiment.save(exp)
        print(f"Stored new fingerprints in {exp['_path']}")


def solve_all(exp: dict, out: str, jobs: int | None) -> None:
    tasks = [
        (motion_report.seed_dir(out, case["id"], doppler, run["seed"]), exp.get("solver"), alg)
        for case in exp["cases"]
        for doppler in motion_report.dopplers_for(exp, case)
        for run in exp["runs"]
        for alg in exp.get("algorithms", list(motion_report.ALGORITHMS))
    ]
    workers = max(1, min(len(tasks), jobs or os.cpu_count() or 1))
    print(f"Solving {len(tasks)} recordings x algorithms on {workers} processes...")
    t0 = time.monotonic()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(_solve, tasks):
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} ({time.monotonic() - t0:.0f} s)")


def print_means(summary: list[dict]) -> None:
    print()
    print(f"{'case':<22} {'Doppler':<8} {'algorithm':<11} {'pos RMSE m':>11} {'vel RMSE m/s':>13} {'avail':>6}")
    for r in summary:
        if r["seed"] != "mean":
            continue
        pos = "-" if r["pos_rmse_m"] is None else f"{r['pos_rmse_m']:.1f}"
        vel = "-" if r["vel_rmse_ms"] is None else f"{r['vel_rmse_ms']:.2f}"
        avail = "-" if r["availability"] is None else f"{r['availability']:.0%}"
        print(f"{r['case']:<22} {r['rx_doppler']:<8} {r['algorithm']:<11} {pos:>11} {vel:>13} {avail:>6}")


def run(exp: dict, args, repo: str, commit) -> None:
    if args.system:
        raise SystemExit(
            "--system is not supported for receiver-motion experiments (the server has no moving receivers)"
        )
    if args.reanalyze:
        out = os.path.abspath(args.reanalyze)
        summary = motion_report.build(out, exp)
        print_means(summary)
        print(f"Rebuilt the analysis in {out}")
        return
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = os.path.abspath(args.out or os.path.join(repo, "runs", exp["name"], stamp))
    os.makedirs(out, exist_ok=True)

    t0 = time.monotonic()
    record_all(exp, out, args.accept_changes)
    print(f"Recorded every case into {out} ({time.monotonic() - t0:.0f} s)")
    solve_all(exp, out, args.jobs)
    summary = motion_report.build(out, exp)

    manifest = {
        "experiment": exp["name"],
        "experiment_file": exp["_path"],
        "base_file": exp["_base"]["_path"] if "_base" in exp else None,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "solver": {**offline_solve.DEFAULT_SOLVER, **(exp.get("solver") or {})},
        "algorithms": exp.get("algorithms"),
        "rx_doppler": exp.get("rx_doppler"),
        "cases": exp["cases"],
        "runs": [{k: r[k] for k in ("seed", "flight", "plan")} for r in exp["runs"]],
        "fingerprints": exp.get("fingerprints"),
        "code": {
            "retina_simulation": commit(repo),
            "retina_geolocator": commit(
                os.path.dirname(os.path.dirname(offline_solve.solve_multinode.__code__.co_filename))
            ),
        },
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print_means(summary)
    print(f"\nWrote {out} ({time.monotonic() - t0:.0f} s in all)")
