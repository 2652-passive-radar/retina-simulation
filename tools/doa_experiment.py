"""Direction-finding study, step 1a: system A against system B.

Called by run_experiment.py for an experiment file with a ``doa`` block (see
experiments/doa_static.json); not meant to be run on its own.

* System A, today's multistatic radar: each tick's delays and Dopplers from
  all three towers, solved by the existing offline solve (needs
  retina_geolocator; skipped with a note if it is missing).
* System B, direction finding: one tower's delay (``doa.node``) plus the
  bearing measured by the 3-element array, solved in closed form
  (doa_solve.py) at an assumed altitude.

Steps: sample the flight plans if the file has none (beam turned by
``node_overrides``), record every seed and check its fingerprint, solve A,
then evaluate B at every level:

* ``nominal``: every error at its default (``doa.errors``), and the same at
  the true altitude, a perfect array, and the other layout;
* ``bearing_noise``: the true bearing plus Gaussian noise of each size, which
  maps a bearing accuracy to a position accuracy (the requirement curve);
* one sweep per error term with every other term off ("alone"), so each
  term's own cost is visible.  At the nominal levels thermal noise dominates
  and would hide the others, which is why the sweeps do not start from
  nominal.

Each level runs ``doa.realisations`` bearing realisations per seed; a
realisation uses the same underlying random draws at every level.
doa_report.py turns ``results.json`` into the tables and figures.
"""

import datetime
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import doa_report
import doa_solve

from retina_simulation import array as arr
from retina_simulation import experiment

_DEFAULT_MAX_JOBS = 8
_dets_cache: dict = {}


def levels(exp: dict) -> list[dict]:
    d = exp["doa"]
    nominal, none = d["errors"], arr.ArrayErrors.none().to_dict()
    layout = d["array"]["layout"]
    other = "line" if layout == "triangle" else "triangle"
    alt = d["assumed_alt_km"]

    def lv(group, param=None, value=None, errors=None, gaussian=None, alt_km=alt, lay=layout):
        return {"group": group, "param": param, "value": value, "errors": errors, "gaussian": gaussian,
                "alt_km": alt_km, "layout": lay}  # fmt: skip

    out = [
        lv("nominal", errors=nominal),
        lv("nominal_true_alt", errors=nominal, alt_km=None),
        lv("perfect", errors=none),
        lv("perfect_true_alt", errors=none, alt_km=None),
        lv(f"nominal_{other}", errors=nominal, lay=other),
    ]
    out += [lv("bearing_noise", "bearing_noise_deg", s, gaussian=s) for s in d["bearing_noise_deg"]]
    for param, values in d["sweeps"].items():
        for v in values:
            errs = dict(none, **{param: v})
            if param == "snr_offset_db":
                errs["thermal"] = True
            out.append(lv("alone", param, v, errors=errs))
    return out


def _dets(seed_dir: str, node_id: str) -> tuple[dict, list[dict]]:
    key = (seed_dir, node_id)
    if key not in _dets_cache:
        _dets_cache[key] = (doa_solve.node_config(seed_dir, node_id), doa_solve.detections(seed_dir, node_id))
    return _dets_cache[key]


def _run_level(task: tuple) -> dict:
    """Every run (seed x realisation) of one level: per-run error sums, and
    for the nominal level every solve (for the figures)."""
    level, seed_dirs, d, n_ticks = task
    cfg = arr.ArrayConfig(**dict(d["array"], layout=level["layout"]))
    err = arr.ArrayErrors(**level["errors"]) if level["errors"] else arr.ArrayErrors.none()
    runs, solves = [], []
    for seed, seed_dir in seed_dirs.items():
        node, dets = _dets(seed_dir, d["node"])
        for r in range(d["realisations"]):
            b = doa_solve.measure(dets, node, cfg, err, (seed, r), level["gaussian"])
            rows = doa_solve.solve(dets, b, node, level["layout"], level["alt_km"])
            e2 = [x["err_horiz_m"] ** 2 for x in rows if "err_horiz_m" in x]
            be = [x["bearing_err_deg"] for x in rows]
            runs.append({"seed": seed, "r": r, "sse": sum(e2), "n": len(e2), "ticks": n_ticks, "bse": sum(v * v for v in be),
                         "nb": len(be), "wrong": sum(abs(v) > doa_solve.WRONG_LOBE_DEG for v in be)})  # fmt: skip
            if level["group"] == "nominal":
                solves += [dict(x, seed=seed, r=r) for x in rows]
    return dict(level, runs=runs, solves=solves)


def system_a(exp: dict, seed_dirs: dict[int, str], jobs: int) -> dict | None:
    """The existing multistatic offline solve of every seed, or None without
    retina_geolocator."""
    try:
        import offline_solve  # noqa: F401  (only to check retina_geolocator is installed)
    except SystemExit:
        return None
    n_ticks = int(exp["duration_s"] / exp.get("tick_s", 1.0))
    work = [(seed_dirs[r["seed"]], exp.get("solver")) for r in exp["runs"]]
    with ProcessPoolExecutor(max_workers=max(1, min(len(work), jobs))) as pool:
        all_rows = list(pool.map(_solve_a, work))
    runs, solves = [], []
    for run, rows in zip(exp["runs"], all_rows):
        node = doa_solve.node_config(seed_dirs[run["seed"]], exp["doa"]["node"])
        pub = [x for x in rows if x.get("published") and x.get("err_horiz_m") is not None]
        runs.append({"seed": run["seed"], "sse": sum(float(x["err_horiz_m"]) ** 2 for x in pub), "n": len(pub),
                     "ticks": n_ticks})  # fmt: skip
        for x in pub:
            e, n = doa_solve._en_m(node["rx_lat"], node["rx_lon"], x["true_lat"], x["true_lon"])
            solves.append({"seed": run["seed"], "range_km": round(((e * e + n * n) ** 0.5) / 1000.0, 3),
                           "err_horiz_m": float(x["err_horiz_m"])})  # fmt: skip
    return {"runs": runs, "solves": solves}


def _solve_a(args: tuple[str, dict]) -> list[dict]:
    import offline_solve

    seed_dir, solver_cfg = args
    rows = offline_solve.solve_run(seed_dir, solver_cfg)
    offline_solve.write_rows(seed_dir, rows)
    return rows


def run(exp: dict, args, repo: str, commit) -> None:
    if args.reanalyze:
        out = os.path.abspath(args.reanalyze)
        with open(os.path.join(out, "results.json"), encoding="utf-8") as f:
            results = json.load(f)
        doa_report.build(out, exp, results)
        print(f"Rebuilt the report in {out}")
        return
    import run_experiment

    if experiment.resolve_plans(exp):
        experiment.save(exp)
        print(f"Sampled and saved flight plans into {args.experiment}")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = os.path.abspath(args.out or os.path.join(repo, "runs", exp["name"], stamp))
    os.makedirs(out, exist_ok=True)
    seed_dirs = {r["seed"]: os.path.join(out, f"seed-{r['seed']}") for r in exp["runs"]}
    run_experiment.record_all(exp, seed_dirs, args.accept_changes, args.experiment)
    print(f"Recorded {len(exp['runs'])} runs into {out}")
    jobs = args.jobs or min(os.cpu_count() or 1, _DEFAULT_MAX_JOBS)

    t0 = time.monotonic()
    a = system_a(exp, seed_dirs, jobs)
    if a is None:
        print("System A skipped: retina_geolocator is not installed in this Python (use the backend's Python).")
    else:
        print(f"System A (multistatic) solved in {time.monotonic() - t0:.0f} s")

    t0 = time.monotonic()
    n_ticks = int(exp["duration_s"] / exp.get("tick_s", 1.0))
    tasks = [(lv, seed_dirs, exp["doa"], n_ticks) for lv in levels(exp)]
    print(f"System B (direction finding): {len(tasks)} levels x {len(seed_dirs) * exp['doa']['realisations']} runs...")
    if jobs == 1:
        done = [_run_level(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            done = list(pool.map(_run_level, tasks))
    print(f"System B done in {time.monotonic() - t0:.0f} s")

    results = {"system_a": a, "levels": done}
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f)
    summary = doa_report.build(out, exp, results)

    manifest = {
        "experiment": exp["name"],
        "experiment_file": exp["_path"],
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "doa": exp["doa"],
        "node_overrides": exp.get("node_overrides"),
        "system_a_run": a is not None,
        "runs": [{k: r[k] for k in ("seed", "flight", "plan", "fingerprint")} for r in exp["runs"]],
        "code": {"retina_simulation": commit(repo)},
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    for row in summary:
        print(f"{row['system']:>34}: RMSE {doa_report.fmt_m(row['pos_rmse_m'])}, median "
              f"{doa_report.fmt_m(row.get('pos_median_m'))}, availability {row['availability']:.0%}")  # fmt: skip
    print(f"Wrote {out}")
