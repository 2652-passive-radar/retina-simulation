"""Run a whole experiment: every seed recorded, solved, analysed and plotted.

Usage (from the retina-simulation folder, with the backend's Python, which has
retina_geolocator; matplotlib and shapely must be installed in it too)::

    python tools/run_experiment.py experiments/baseline_static.json
    python tools/run_experiment.py experiments/baseline_static.json --system --backend-dir ../../backend

Default (about a minute): records each seed without a server, solves it
offline, and writes the figures.  ``--system`` also streams every seed through
a fresh backend in real time (duration x seeds, ~26 min for the baseline),
started and stopped here so no state leaks from one seed into the next.

Output: ``runs/<experiment name>/<date_time>/`` holding ``manifest.json``,
``summary.csv``, ``rmse_vs_time.csv``, ``rmse_vs_time.png``,
``paths_map.png`` and one ``seed-<n>/`` folder per run (the recording, its
``analysis/`` and ``maps/``; under ``--system`` also ``system/``, the server
run's own recording and analysis).

The first run of an experiment file writes each flight's plan and a
fingerprint of its recording into the file.  Every later run checks the
fingerprint and stops if the simulator no longer reproduces the recording,
unless ``--accept-changes`` is given (which updates the stored fingerprints).
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor

_TOOLS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_TOOLS)
sys.path.insert(0, _TOOLS)
sys.path.insert(0, _REPO)

import experiment_report  # noqa: E402
import offline_solve  # noqa: E402

from retina_simulation import experiment  # noqa: E402

_BACKEND_URL = "http://localhost:8000"


def _git(path: str, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _commit(path: str) -> dict | None:
    head = _git(path, "rev-parse", "HEAD")
    if head is None:
        return None
    return {"path": os.path.abspath(path), "commit": head, "dirty": bool(_git(path, "status", "--porcelain"))}


def _solve_seed(args: tuple[str, dict]) -> int:
    seed_dir, solver_cfg = args
    rows = offline_solve.solve_run(seed_dir, solver_cfg)
    offline_solve.write_rows(seed_dir, rows)
    return sum(1 for r in rows if r.get("published"))


def _analyze(run_dir: str, maps: bool, offline_note: str | None) -> None:
    """analyze_run's tables, summary and delay-Doppler maps for one recording."""
    cmd = [sys.executable, os.path.join(_TOOLS, "analyze_run.py"), run_dir]
    if not maps:
        cmd.append("--no-maps")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    if maps:
        subprocess.run(
            [sys.executable, os.path.join(_TOOLS, "view_maps.py"), run_dir], check=True, stdout=subprocess.DEVNULL
        )
    if offline_note:
        with open(os.path.join(run_dir, "analysis", "summary.txt"), "a", encoding="utf-8") as f:
            f.write(offline_note)


def _wait_for_backend(proc: subprocess.Popen, timeout_s: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"backend exited during startup (code {proc.returncode})")
        try:
            with urllib.request.urlopen(_BACKEND_URL + "/api/health", timeout=3):
                return
        except OSError:
            time.sleep(2)
    raise SystemExit("backend did not answer /api/health in time")


def _backend_busy() -> bool:
    try:
        with urllib.request.urlopen(_BACKEND_URL + "/api/health", timeout=2):
            return True
    except OSError:
        return False


def _run_system_seed(exp: dict, run: dict, sys_dir: str, backend_dir: str, backend_python: str) -> None:
    env = dict(
        os.environ,
        RETINA_ENV="dev",
        AUTH_ALLOW_ANONYMOUS_ADMIN="1",
        SYNTHETIC_FLEET_ENABLED="1",
        NODE_FUZZ_MODE="off",
        PYTHONUTF8="1",
    )
    log_path = os.path.join(os.path.dirname(sys_dir), "backend.log")
    with open(log_path, "w", encoding="utf-8") as log:
        backend = subprocess.Popen(
            [backend_python, "-m", "uvicorn", "main:app", "--port", "8000"],
            cwd=backend_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_backend(backend)
            subprocess.run(
                [
                    sys.executable, "-m", "retina_simulation.orchestrator",
                    "--config", experiment.fleet_path(exp),
                    "--experiment", exp["_path"],
                    "--experiment-seed", str(run["seed"]),
                    "--duration", str(exp["duration_s"]),
                    "--record", sys_dir,
                    "--validation-url", _BACKEND_URL,
                ],
                cwd=_REPO,
                env=env,
                check=True,
            )  # fmt: skip
        finally:
            backend.terminate()
            try:
                backend.wait(timeout=30)
            except subprocess.TimeoutExpired:
                backend.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every seed of an experiment and build its figures")
    parser.add_argument("experiment", help="experiments/<name>.json")
    parser.add_argument("--out", help="Output folder (default runs/<name>/<date_time>)")
    parser.add_argument("--no-maps", action="store_true", help="Skip the delay-Doppler maps and their PNGs")
    parser.add_argument("--accept-changes", action="store_true", help="Store new fingerprints instead of stopping")
    parser.add_argument("--system", action="store_true", help="Also run every seed through a fresh backend")
    parser.add_argument("--backend-dir", help="retina-server/backend (needed with --system)")
    parser.add_argument("--backend-python", help="Python for the backend (default: this one)")
    parser.add_argument(
        "--reanalyze",
        metavar="RUN_DIR",
        help="Rebuild the analysis and figures of an existing output folder (e.g. after an analysis fix) "
        "without recording or solving anything again",
    )
    args = parser.parse_args()

    exp = experiment.load(args.experiment)
    if args.reanalyze:
        _reanalyze(exp, os.path.abspath(args.reanalyze))
        return
    if experiment.resolve_plans(exp):
        experiment.save(exp)
        print(f"Sampled and saved flight plans into {args.experiment}")

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = os.path.abspath(args.out or os.path.join(_REPO, "runs", exp["name"], stamp))
    os.makedirs(out, exist_ok=True)
    seed_dirs = {r["seed"]: os.path.join(out, f"seed-{r['seed']}") for r in exp["runs"]}

    # 1. Record every seed and check it against its stored fingerprint.
    changed = []
    for run in exp["runs"]:
        got = experiment.record_run(exp, run, seed_dirs[run["seed"]])
        if "fingerprint" not in run:
            run["fingerprint"] = got
            changed.append(run["seed"])
            continue
        diff = experiment.fingerprint_mismatch(run["fingerprint"], got)
        if diff and not args.accept_changes:
            raise SystemExit(
                f"seed {run['seed']}: the simulator no longer reproduces this run's recording:\n  "
                + "\n  ".join(diff)
                + "\nRe-run with --accept-changes to adopt the new recording as the reference."
            )
        if diff:
            run["fingerprint"] = got
            changed.append(run["seed"])
    if changed:
        experiment.save(exp)
        print(f"Stored fingerprints for seeds {changed} in {args.experiment}")
    print(f"Recorded {len(exp['runs'])} runs into {out}")

    # 2. Offline radar-only solves, one process per seed.
    t0 = time.monotonic()
    work = [(seed_dirs[r["seed"]], exp.get("solver")) for r in exp["runs"]]
    with ProcessPoolExecutor(max_workers=min(len(work), os.cpu_count() or 1)) as pool:
        published = list(pool.map(_solve_seed, work))
    print(f"Offline solves done in {time.monotonic() - t0:.0f} s")

    # 3. Per-seed tables, summary text and delay-Doppler maps, as before.
    for run, n_pub in zip(exp["runs"], published):
        note = (
            f"Offline radar-only solves (tools/offline_solve.py): {n_pub} published "
            f"of {int(exp['duration_s'] / exp.get('tick_s', 1.0))} ticks; see offline_solves.csv.\n"
        )
        _analyze(seed_dirs[run["seed"]], not args.no_maps, note)

    # 4. Optional: the same flights through the real server.
    if args.system:
        if not args.backend_dir:
            raise SystemExit("--system needs --backend-dir (the retina-server/backend folder)")
        if _backend_busy():
            raise SystemExit(f"something already answers at {_BACKEND_URL}: stop your backend first")
        backend_python = args.backend_python or sys.executable
        for run in exp["runs"]:
            sys_dir = os.path.join(seed_dirs[run["seed"]], "system")
            print(f"System run, seed {run['seed']} ({exp['duration_s']} s)...")
            _run_system_seed(exp, run, sys_dir, os.path.abspath(args.backend_dir), backend_python)
            _analyze(sys_dir, False, None)

    # 5. Figures and tables across seeds.
    summary = experiment_report.build(out, exp, seed_dirs, system=args.system)

    manifest = {
        "experiment": exp["name"],
        "experiment_file": exp["_path"],
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "system_run": args.system,
        "solver": {**offline_solve.DEFAULT_SOLVER, **(exp.get("solver") or {})},
        "offline_solve_notes": "perfect association from the recorder's labels; initial guess from a radar-only "
        "delay grid search (never truth); altitude by sweep unless solver.alt_mode says otherwise",
        "runs": [{k: r[k] for k in ("seed", "flight", "plan", "fingerprint")} for r in exp["runs"]],
        "code": {
            "retina_simulation": _commit(_REPO),
            "retina_geolocator": _commit(
                os.path.dirname(os.path.dirname(offline_solve.solve_multinode.__code__.co_filename))
            ),
            "retina_server": _commit(args.backend_dir) if args.backend_dir else None,
        },
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _print_means(summary)
    print(f"Wrote {out}")


def _print_means(summary: list[dict]) -> None:
    print()
    for row in summary:
        if row["seed"] != "mean":
            continue
        if row["pos_rmse_m"] is None:
            print(f"{row['source']:>12}: no solves")
            continue
        print(
            f"{row['source']:>12} mean over seeds: position RMSE {row['pos_rmse_m']:.1f} m, "
            f"velocity RMSE {row['vel_rmse_ms']:.2f} m/s, availability {row['availability']:.0%}"
        )


def _reanalyze(exp: dict, out: str) -> None:
    """The analysis half of a run, again, on recordings already on disk."""
    seed_dirs = {r["seed"]: os.path.join(out, f"seed-{r['seed']}") for r in exp["runs"]}
    missing = [d for d in seed_dirs.values() if not os.path.isdir(d)]
    if missing:
        raise SystemExit(f"not an output folder of {exp['name']}: missing {missing[0]}")
    system = all(os.path.isdir(os.path.join(d, "system")) for d in seed_dirs.values())
    if system:
        for d in seed_dirs.values():
            _analyze(os.path.join(d, "system"), False, None)
    summary = experiment_report.build(out, exp, seed_dirs, system=system)
    _print_means(summary)
    print(f"Rebuilt the analysis in {out}")


if __name__ == "__main__":
    main()
