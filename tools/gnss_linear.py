"""Step 3d: GD's response to a receiver pose error, linearised per tick.

Tried as a shortcut for the Monte Carlo and rejected (experiments/NOTEBOOK.md,
step 3d, "Condensing the run"): GD's response is sub-linear at the receiver
errors that matter (Huber loss) and some ticks flip altitude layer.  Kept for
the record and for reruns of that check (``"method": "linear"``).

GD solves every tick on its own (a fresh initial guess, no memory across
ticks), and a receiver pose error of metres and m/s moves its answer by a
small amount compared with the geometry (tens of km) and the measurement noise
(2 Hz).  So for each tick k of a recording, to first order,

    a_k(delta) = a_k(0) + J_k delta

where ``a`` is the aircraft's horizontal error [east m, north m, east m/s,
north m/s], ``delta`` the receiver pose error GD was given [east, north, up m;
east, north, up m/s], and ``J_k`` (4 x 6) the tick's sensitivity.  ``J_k`` is
measured by central differences: GD solved with the true trajectory shifted by
+-h along each of the six inputs (12 solves per recording), column
j = (a(+h e_j) - a(-h e_j)) / 2h.  ``a_k(0)`` is the truth-fed solve.

Any level and GNSS realisation then costs one KF run and a 4 x 6 product per
tick instead of a GD solve, so a level can have many more realisations.

A tick enters only if the truth-fed solve and all 12 perturbed solves
produced an aircraft estimate (the full method likewise pairs only ticks that
both solves produced).
"""

import csv
import json
import os

import gnss_mc
import gnss_pose
import numpy as np
import offline_solve

from retina_simulation.motion import motion_from

INPUTS = ("east_m", "north_m", "up_m", "vel_east_ms", "vel_north_ms", "vel_up_ms")
OUT_KEYS = ("err_east_m", "err_north_m", "verr_east_ms", "verr_north_ms")


def steps(exp: dict) -> list[float]:
    lin = exp.get("linear", {})
    hp, hv = lin.get("h_pos_m", 2.0), lin.get("h_vel_ms", 0.5)
    return [hp, hp, hp, hv, hv, hv]


def jac_file(j: int, sign: int) -> str:
    return f"offline_solves_gd_jac_{INPUTS[j]}_{'p' if sign > 0 else 'm'}.csv"


def shifted_truth(exp: dict, case: dict, j: int, h: float) -> list[dict]:
    """The true receiver trajectory at every tick, input j shifted by h, as KF-style estimates."""
    motion = motion_from(case["motion"])
    ests = []
    for t in gnss_pose.tick_times(exp):
        e, n, ve, vn = motion.state(t)
        est = dict(zip(INPUTS, (e, n, 0.0, ve, vn, 0.0), strict=True), t_s=t)
        est[INPUTS[j]] += h
        ests.append(est)
    return ests


def jac_tasks(exp: dict, out: str) -> list[tuple]:
    tasks = []
    for path, case, _seed in gnss_mc.recordings(exp, out):
        for j in range(len(INPUTS)):
            for sign in (+1, -1):
                if not os.path.exists(os.path.join(path, "analysis", jac_file(j, sign))):
                    tasks.append((path, case, j, sign))
    return tasks


def solve_jac(task: tuple, exp: dict) -> str:
    path, case, j, sign = task
    with open(os.path.join(path, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    poses = gnss_pose.poses_from_estimates(exp, nodes, shifted_truth(exp, case, j, sign * steps(exp)[j]))
    rows = offline_solve.solve_run(path, exp.get("solver"), "geometric_doppler", pose_estimate=poses)
    name = jac_file(j, sign)
    tmp = offline_solve.write_rows(path, rows, name + ".part")
    os.replace(tmp, os.path.join(path, "analysis", name))
    return name


def _errors(path: str) -> dict[float, tuple[np.ndarray, str]]:
    """{t_s: (aircraft error [east m, north m, east m/s, north m/s], altitude layer)}."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if all(r.get(k) not in (None, "") for k in OUT_KEYS):
                out[float(r["t_s"])] = (np.array([float(r[k]) for k in OUT_KEYS]), r.get("est_alt_m", ""))
    return out


def jacobian(path: str, exp: dict) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """{t_s: (a(0), J)} for the ticks every solve of the recording produced.

    A tick whose winning altitude layer changes under a +-h shift sits near a
    tie between two layers: there the answer jumps rather than moves, and a
    central difference across the jump is meaningless.  Such a column is taken
    one-sided, from the shift that kept the truth-fed solve's layer (0 if both
    changed it), so the tick follows its truth-fed branch; ``layer_flips``
    counts them."""
    d = os.path.join(path, "analysis")
    base = _errors(os.path.join(d, gnss_mc.solve_file("truth", None)))
    pert = {(j, s): _errors(os.path.join(d, jac_file(j, s))) for j in range(len(INPUTS)) for s in (+1, -1)}
    h = steps(exp)
    out = {}
    for t, (a0, layer) in base.items():
        if not all(t in p for p in pert.values()):
            continue
        cols = []
        for j in range(len(INPUTS)):
            (ap, lp), (am, lm) = pert[(j, 1)][t], pert[(j, -1)][t]
            if lp == layer and lm == layer:
                cols.append((ap - am) / (2 * h[j]))
            elif lp == layer:
                cols.append((ap - a0) / h[j])
            elif lm == layer:
                cols.append((a0 - am) / h[j])
            else:
                cols.append(np.zeros(len(OUT_KEYS)))
        out[t] = (a0, np.column_stack(cols))
    return out


def layer_flips(path: str) -> tuple[int, int]:
    """(ticks whose layer any +-h shift changes, ticks with every solve) in one recording."""
    d = os.path.join(path, "analysis")
    base = _errors(os.path.join(d, gnss_mc.solve_file("truth", None)))
    pert = [_errors(os.path.join(d, jac_file(j, s))) for j in range(len(INPUTS)) for s in (+1, -1)]
    ts = [t for t in base if all(t in p for p in pert)]
    return sum(any(p[t][1] != base[t][1] for p in pert) for t in ts), len(ts)


RUN_FIELDS = [
    "level", "case", "seed", "realisation", "ticks", "pos_kf_ss", "pos_truth_ss", "vel_kf_ss", "vel_truth_ss",
    "rx_pos_ss_m2", "rx_vel_ss_ms2", "pos_in_2sigma", "vel_in_2sigma", "nis_sum", "nis_n",
]  # fmt: skip


def predict(exp: dict, case: dict, seed: int, r: int, jac: dict, lid: str) -> dict:
    """One run (level, case, seed, realisation): the KF, then GD's aircraft error
    to first order, summed over the paired ticks after the warm-up; with the
    receiver error and the KF's consistency over the same ticks."""
    motion = motion_from(case["motion"])
    t0 = exp.get("steady_state_from_s", 0.0)
    s = dict.fromkeys(RUN_FIELDS[4:], 0.0)
    for est in gnss_pose.estimates(exp, case, seed, r, "kf"):
        if est is None or est["t_s"] < t0 or est["t_s"] not in jac:
            continue
        te, tn, tve, tvn = motion.state(est["t_s"])
        truth = (te, tn, 0.0, tve, tvn, 0.0)
        delta = np.array([est[k] - v for k, v in zip(INPUTS, truth, strict=True)])
        a0, J = jac[est["t_s"]]
        a = a0 + J @ delta
        s["ticks"] += 1
        s["pos_kf_ss"] += a[0] ** 2 + a[1] ** 2
        s["pos_truth_ss"] += a0[0] ** 2 + a0[1] ** 2
        s["vel_kf_ss"] += a[2] ** 2 + a[3] ** 2
        s["vel_truth_ss"] += a0[2] ** 2 + a0[3] ** 2
        s["rx_pos_ss_m2"] += delta[0] ** 2 + delta[1] ** 2
        s["rx_vel_ss_ms2"] += delta[3] ** 2 + delta[4] ** 2
        s["pos_in_2sigma"] += (abs(delta[0]) <= 2 * est["sd_east_m"]) + (abs(delta[1]) <= 2 * est["sd_north_m"])
        s["vel_in_2sigma"] += (abs(delta[3]) <= 2 * est["sd_vel_east_ms"]) + (
            abs(delta[4]) <= 2 * est["sd_vel_north_ms"]
        )
        if est["nis"] is not None:
            s["nis_sum"] += est["nis"]
            s["nis_n"] += 1
    return {"level": lid, "case": case["id"], "seed": seed, "realisation": r, **s}


_JAC_CACHE: dict = {}


def predict_task(task: tuple, exp: dict) -> list[dict]:
    """Every realisation of one level, case and seed."""
    lid, ov, path, case, seed = task
    if path not in _JAC_CACHE:
        _JAC_CACHE[path] = jacobian(path, exp)
    e = gnss_mc.apply(exp, ov)
    n = exp.get("linear", {}).get("realisations", 50)
    return [predict(e, case, seed, r, _JAC_CACHE[path], lid) for r in range(n)]


def predict_all(exp: dict, out: str, jobs: int | None) -> str:
    path = os.path.join(out, "linear_runs.csv")
    if os.path.exists(path):
        print(f"Linear predictions already done ({path})")
        return path
    found, _, _ = gnss_mc.levels(exp)
    recs = gnss_mc.recordings(exp, out)
    tasks = [(lid, ov, p, c, s) for p, c, s in recs for lid, ov in found.items()]
    n = exp.get("linear", {}).get("realisations", 50)
    print(f"Linear predictions: {len(found)} levels x {len(recs)} recordings x {n} realisations")
    chunks = gnss_mc._pool_run(predict_task, tasks, gnss_mc._slim(exp), gnss_mc._workers(jobs, len(tasks)), "predict")
    with open(path + ".part", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        w.writeheader()
        for chunk in chunks:
            w.writerows(chunk)
    os.replace(path + ".part", path)
    return path


def solve_jacobians(exp: dict, out: str, jobs: int | None) -> None:
    tasks = jac_tasks(exp, out)
    truth = [t for t in gnss_mc.tasks_for(exp, out) if t[3] == "truth"]
    if truth:
        print(f"Truth-fed solves: {len(truth)}")
        gnss_mc._pool_run(gnss_mc._solve, truth, gnss_mc._slim(exp), gnss_mc._workers(jobs, len(truth)), "truth")
    if not tasks:
        print("Sensitivity solves already on disk")
        return
    print(f"Sensitivity solves: {len(tasks)} (12 per recording)")
    gnss_mc._pool_run(solve_jac, tasks, gnss_mc._slim(exp), gnss_mc._workers(jobs, len(tasks)), "jacobian")


def spot_levels(exp: dict, rows: list[dict], threshold: float = 10.0) -> list[dict]:
    """Each +threshold% crossing of the table (worst case), as a level to solve in full in its binding case."""
    keys = {s["id"]: s["key"] for s in exp.get("sweeps", [])}
    spots = []
    for r in rows:
        v = r["worst_value"]
        if r["threshold_pct"] != threshold or v is None or v == r["nominal"] or not r["binding_case"]:
            continue
        ov = gnss_mc.normalise(exp, {keys[r["sweep"]]: v})
        spot = {"sweep": r["sweep"], "direction": r["direction"], "output": r["output"], "value": v,
                "case": r["binding_case"], "overrides": ov, "level": "spot_" + gnss_mc.level_id(ov)}  # fmt: skip
        spots.append(spot)
    return spots


def spot_check(exp: dict, out: str, rows: list[dict], jobs: int | None) -> list[dict]:
    """Full GD solves at each threshold in its binding case (5 seeds x the full
    method's realisations), against the linearised prediction from the same draws."""
    import gnss_mc_report

    spots = spot_levels(exp, rows)
    recs = {c["id"]: [(p, c, s) for p, c, s in gnss_mc.recordings(exp, out) if c["id"] == cid]
            for cid in {sp["case"] for sp in spots} for c in exp["cases"] if c["id"] == cid}  # fmt: skip
    n_real = exp.get("realisations", 4)
    tasks, seen = [], set()
    for sp in spots:
        for p, c, s in recs[sp["case"]]:
            for r in range(n_real):
                key = (p, sp["level"], r)
                if key not in seen and not os.path.exists(
                    os.path.join(p, "analysis", gnss_mc.solve_file(sp["level"], r))
                ):
                    seen.add(key)
                    tasks.append((p, c, s, sp["level"], r, sp["overrides"]))
    if tasks:
        print(f"Spot checks: {len(tasks)} full GD solves at the +10% thresholds")
        gnss_mc._pool_run(gnss_mc._solve, tasks, gnss_mc._slim(exp), gnss_mc._workers(jobs, len(tasks)), "spot")
    t0 = exp.get("steady_state_from_s", 0.0)
    idx = None
    result = []
    for sp in spots:
        full, lin = [], []
        for p, c, s in recs[sp["case"]]:
            truth = {
                x["t_s"]: x for x in gnss_mc_report.load(os.path.join(p, "analysis", gnss_mc.solve_file("truth", None)))
            }
            jac = jacobian(p, exp)
            e = gnss_mc.apply(exp, sp["overrides"])
            for r in range(n_real):
                full.append(
                    gnss_mc_report.run_sums(os.path.join(p, "analysis", gnss_mc.solve_file(sp["level"], r)), truth, t0)
                )
                d = predict(e, c, s, r, jac, sp["level"])
                lin.append([d[k] for k in RUN_FIELDS[5:11]] + [d["ticks"]])
        full = np.array(full)
        idx = gnss_mc_report.boot_indices({sp["case"]: list(range(len(full)))})
        boot = gnss_mc_report.boot_increase({sp["case"]: full}, idx, sp["output"], sp["case"])
        lo, hi = np.percentile(boot, gnss_mc_report.CI)
        result.append({**{k: sp[k] for k in ("sweep", "direction", "output", "value", "case")}, "threshold_pct": 10,
                       "full_pct": gnss_mc_report.increase(full, sp["output"]), "full_lo": float(lo),
                       "full_hi": float(hi), "linear_pct": gnss_mc_report.increase(np.array(lin), sp["output"])})  # fmt: skip
    path = os.path.join(out, "spot_checks.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=gnss_mc_report.SPOT_FIELDS)
        w.writeheader()
        w.writerows(result)
    return result
