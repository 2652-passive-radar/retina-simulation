"""Step 3d: how far each GNSS-model and receiver-KF parameter can move before GD degrades.

Called by run_experiment.py for an experiment file with ``sweeps`` (see
experiments/receiver_gnss_mc.json); not meant to be run on its own.

A *level* is a set of overrides of the nominal GNSS model (``gnss.<key>``) and
receiver filter (``filter.<key>``, the scales rx_filter.config_for reads).  The
GNSS overrides change both the simulated GNSS and what the filter is told
(the filter stays matched to the GNSS); the filter scales detune the filter
from a nominal GNSS.  Levels come from one-at-a-time ``sweeps`` and from the
``grid`` (two keys crossed); identical levels are solved once.

Every recording of the chosen ``case_ids`` (each seed) is re-recorded and
fingerprint-checked as in step 3c, solved by GD once with the true trajectory,
then once per level and GNSS realisation with the receiver pose from the KF:
``analysis/offline_solves_gd_mc_<level>_r<k>.csv``.  Finished solves are kept,
so an interrupted run picks up where it stopped when started again with the
same ``--out``.

Stage A (screening) runs the KF alone over more realisations per level and
records the receiver error and the filter's consistency (``screening.csv``).
gnss_mc_report.py turns both into the sensitivity table and figures.
"""

import csv
import datetime
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import gnss_mc_report
import gnss_pose
import motion_experiment
import motion_report
import offline_solve

from retina_simulation.motion import motion_from

_DEFAULT_MAX_JOBS = 8
_FILTER_SCALES = ("sigma_a_scale", "assumed_sigma_scale", "assumed_tau_scale")


def cases_of(exp: dict) -> list[dict]:
    ids = exp.get("case_ids")
    return [c for c in exp["cases"] if ids is None or c["id"] in ids]


def nominal_value(exp: dict, key: str) -> float:
    section, name = key.split(".", 1)
    if section == "gnss" and name in exp["gnss"]:
        return float(exp["gnss"][name])
    if section == "filter" and name in _FILTER_SCALES:
        return float(exp["filter"].get(name, 1.0))
    raise ValueError(f"unknown sweep key {key!r}")


def normalise(exp: dict, overrides: dict) -> dict:
    """The overrides that differ from the nominal value, keys sorted."""
    return {k: float(v) for k, v in sorted(overrides.items()) if not math.isclose(float(v), nominal_value(exp, k))}


def level_id(overrides: dict) -> str:
    if not overrides:
        return "nominal"
    return "_".join(f"{k.split('.', 1)[1]}-{v:g}" for k, v in sorted(overrides.items()))


def apply(exp: dict, overrides: dict) -> dict:
    """A copy of the experiment with the overrides in its ``gnss`` / ``filter`` blocks."""
    out = {**exp, "gnss": dict(exp["gnss"]), "filter": dict(exp["filter"])}
    for k, v in overrides.items():
        nominal_value(exp, k)
        section, name = k.split(".", 1)
        out[section][name] = v
    return out


def levels(exp: dict) -> tuple[dict[str, dict], dict[str, list[tuple[float, str]]], list[tuple[float, float, str]]]:
    """Every distinct level ``{id: overrides}``, each sweep's ``[(value, id)]``
    sorted by value with the nominal included, and the grid's ``[(x, y, id)]``."""
    found = {"nominal": {}}
    sweeps = {}
    for sw in exp.get("sweeps", []):
        pts = {nominal_value(exp, sw["key"]): "nominal"}
        for v in sw["levels"]:
            ov = normalise(exp, {sw["key"]: v})
            found[level_id(ov)] = ov
            pts[float(v)] = level_id(ov)
        sweeps[sw["id"]] = sorted(pts.items())
    grid = []
    g = exp.get("grid")
    if g:
        for x in g["x_levels"]:
            for y in g["y_levels"]:
                ov = normalise(exp, {g["x"]: x, g["y"]: y})
                found[level_id(ov)] = ov
                grid.append((float(x), float(y), level_id(ov)))
    return found, sweeps, grid


def solve_file(level: str, realisation: int | None) -> str:
    if level == "truth":
        return gnss_pose.solve_name("truth", None)
    return f"offline_solves_gd_mc_{level}_r{realisation}.csv"


def recordings(exp: dict, out: str) -> list[tuple[str, dict, int]]:
    rec = []
    for case in cases_of(exp):
        for doppler in motion_report.dopplers_for(exp, case):
            for run in exp["runs"]:
                rec.append((motion_report.seed_dir(out, case["id"], doppler, run["seed"]), case, run["seed"]))
    return rec


def tasks_for(exp: dict, out: str) -> list[tuple]:
    """Every solve not already on disk: the truth-fed solve of each recording
    first, then level by level."""
    found, _, _ = levels(exp)
    recs = recordings(exp, out)
    tasks = [(p, c, s, "truth", None, None) for p, c, s in recs]
    for lid, ov in found.items():
        for p, c, s in recs:
            for r in range(exp.get("realisations", 1)):
                tasks.append((p, c, s, lid, r, ov))
    return [t for t in tasks if not os.path.exists(os.path.join(t[0], "analysis", solve_file(t[3], t[4])))]


def _solve(task: tuple, exp: dict) -> str:
    path, case, seed, lid, r, ov = task
    poses = None if lid == "truth" else gnss_pose.poses_for(path, apply(exp, ov), case, seed, r, "kf")
    rows = offline_solve.solve_run(path, exp.get("solver"), "geometric_doppler", pose_estimate=poses)
    name = solve_file(lid, r)
    tmp = offline_solve.write_rows(path, rows, name + ".part")
    os.replace(tmp, os.path.join(path, "analysis", name))
    return name


def _pool_run(fn, tasks: list, exp: dict, workers: int, label: str) -> list:
    t0 = time.monotonic()
    results = []
    every = max(1, len(tasks) // 200)

    def progress(n: int) -> None:
        if n % every == 0 or n == len(tasks):
            el = time.monotonic() - t0
            eta = el / n * (len(tasks) - n)
            print(f"  {label} {n}/{len(tasks)} ({el / 60:.0f} min, about {eta / 60:.0f} min left)", flush=True)

    if workers > 1:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fn, t, exp) for t in tasks]
            for n, fut in enumerate(as_completed(futures), 1):
                results.append(fut.result())
                progress(n)
    else:
        for n, t in enumerate(tasks, 1):
            results.append(fn(t, exp))
            progress(n)
    return results


def _workers(jobs: int | None, n: int) -> int:
    return max(1, min(n, jobs or min(os.cpu_count() or 1, _DEFAULT_MAX_JOBS)))


def _slim(exp: dict) -> dict:
    return {k: v for k, v in exp.items() if k != "_base"}


def screen_one(task: tuple, exp: dict) -> list[dict]:
    """Stage A for one level, case and seed: the KF alone over the screening
    realisations, its horizontal error against the true trajectory after the
    warm-up, and how often the truth sits inside its own +-2 sigma."""
    lid, ov, case, seed = task
    e = apply(exp, ov)
    motion = motion_from(case["motion"])
    t0 = exp.get("steady_state_from_s", 0.0)
    rows = []
    for r in range(exp.get("screen_realisations", 8)):
        n = sp = sv = cp = cv = 0
        nis = []
        for est in gnss_pose.estimates(e, case, seed, r, "kf"):
            if est is None or est["t_s"] < t0:
                continue
            te, tn, tve, tvn = motion.state(est["t_s"])
            de, dn = est["east_m"] - te, est["north_m"] - tn
            dve, dvn = est["vel_east_ms"] - tve, est["vel_north_ms"] - tvn
            sp += de * de + dn * dn
            sv += dve * dve + dvn * dvn
            cp += (abs(de) <= 2 * est["sd_east_m"]) + (abs(dn) <= 2 * est["sd_north_m"])
            cv += (abs(dve) <= 2 * est["sd_vel_east_ms"]) + (abs(dvn) <= 2 * est["sd_vel_north_ms"])
            if est["nis"] is not None:
                nis.append(est["nis"])
            n += 1
        rows.append(
            {
                "level": lid, "case": case["id"], "seed": seed, "realisation": r, "ticks": n,
                "rx_pos_ss_m2": sp, "rx_vel_ss_ms2": sv, "pos_in_2sigma": cp, "vel_in_2sigma": cv,
                "nis_mean": sum(nis) / len(nis) if nis else "",
            }
        )  # fmt: skip
    return rows


SCREEN_FIELDS = [
    "level", "case", "seed", "realisation", "ticks", "rx_pos_ss_m2", "rx_vel_ss_ms2", "pos_in_2sigma",
    "vel_in_2sigma", "nis_mean",
]  # fmt: skip


def screen(exp: dict, out: str, jobs: int | None) -> str:
    path = os.path.join(out, "screening.csv")
    if os.path.exists(path):
        print(f"Stage A already done ({path})")
        return path
    found, _, _ = levels(exp)
    tasks = [(lid, ov, c, run["seed"]) for lid, ov in found.items() for c in cases_of(exp) for run in exp["runs"]]
    print(
        f"Stage A: the KF alone over {exp.get('screen_realisations', 8)} realisations, {len(tasks)} level x case x seed"
    )
    rows = [
        r for chunk in _pool_run(screen_one, tasks, _slim(exp), _workers(jobs, len(tasks)), "screen") for r in chunk
    ]
    with open(path + ".part", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCREEN_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(path + ".part", path)
    return path


def solve_all(exp: dict, out: str, jobs: int | None) -> None:
    tasks = tasks_for(exp, out)
    if not tasks:
        print("Stage B: every solve is already on disk")
        return
    workers = _workers(jobs, len(tasks))
    print(f"Stage B: {len(tasks)} GD solves left on {workers} process{'es' if workers > 1 else ''}")
    _pool_run(_solve, tasks, _slim(exp), workers, "solve")


def run(exp: dict, args, repo: str, commit) -> None:
    if args.system:
        raise SystemExit("--system is not supported for the GNSS Monte Carlo (offline only)")
    exp = {**exp, "cases": cases_of(exp)}
    if args.reanalyze:
        out = os.path.abspath(args.reanalyze)
        gnss_mc_report.build(out, exp)
        print(f"Rebuilt the analysis in {out}")
        return
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = os.path.abspath(args.out or os.path.join(repo, "runs", exp["name"], stamp))
    os.makedirs(out, exist_ok=True)
    found, _, _ = levels(exp)
    print(f"{len(found)} levels x {len(recordings(exp, out))} recordings x {exp.get('realisations', 1)} realisations")

    t0 = time.monotonic()
    if all(os.path.exists(os.path.join(p, "nodes.json")) for p, _, _ in recordings(exp, out)):
        print(f"Recordings already in {out}; not recording again")
    else:
        motion_experiment.record_all(exp, out, args.accept_changes)
        print(f"Recorded every case into {out} ({time.monotonic() - t0:.0f} s)")
    manifest = {
        "experiment": exp["name"],
        "experiment_file": exp["_path"],
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "solver": {**offline_solve.DEFAULT_SOLVER, **(exp.get("solver") or {})},
        "algorithm": "geometric_doppler",
        "pose_source": "kf",
        "realisations": exp.get("realisations", 1),
        "screen_realisations": exp.get("screen_realisations", 8),
        "gnss_nominal": exp["gnss"],
        "filter_nominal": exp["filter"],
        "sweeps": exp.get("sweeps", []),
        "grid": exp.get("grid"),
        "thresholds_pct": exp.get("thresholds_pct"),
        "steady_state_from_s": exp.get("steady_state_from_s", 0.0),
        "cases": exp["cases"],
        "runs": [{k: r[k] for k in ("seed", "flight", "plan")} for r in exp["runs"]],
        "code": {"retina_simulation": commit(repo)},
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    screen(exp, out, args.jobs)
    solve_all(exp, out, args.jobs)
    gnss_mc_report.build(out, exp)
    print(f"\nWrote {out} ({(time.monotonic() - t0) / 3600:.1f} h in all)")
