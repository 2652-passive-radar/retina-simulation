"""Step 3b: the receiver Kalman filter against the true trajectory, for every
motion case, fed by the step 3a GNSS error model.  No radar is involved.

For each case of ``experiments/receiver_filter.json`` (the step 1 motions),
each seed and each GNSS realisation, the fixes are the true trajectory plus
modelled GNSS error, and three methods estimate the receiver's position and
velocity at every radar tick:

``kf``     Kalman filter with the GNSS drift (Gauss-Markov) in its state
``white``  the same filter treating all GNSS error as white noise
``raw``    the latest fix, velocity by differencing it with the one before

The filters' acceleration noise and initial velocity uncertainty come from the
case's platform mode (none, walk, vehicle); their GNSS error parameters match
the model's.

Writes to ``runs/receiver-filter/<date_time>/`` (or ``--out``):

``rmse.png``         position and velocity RMSE (3D: east, north and up
                     together) of the three methods, per case and over all
                     cases
``coverage.png``     how often the true position and velocity are inside each
                     filter's +/-2 sigma (95% = honest), per case and overall
``summary.csv``      the numbers: 3D and horizontal RMSE per case and method,
                     the GNSS fixes' own error, mean NEES / NIS
``manifest.json``    settings and code version

Usage::

    python tools/rx_filter_report.py experiments/receiver_filter.json [--out DIR]
"""

import argparse
import csv
import datetime
import json
import math
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rx_filter  # noqa: E402
from experiment_report import INK, INK_2, SURFACE, _save, _style  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from retina_simulation import experiment, gnss  # noqa: E402
from retina_simulation.motion import motion_from  # noqa: E402

VAR_LABEL = {
    "kf": "KF with GNSS drift",
    "white": "KF, white noise only",
    "raw": "raw fixes (differenced velocity)",
}
VAR_COLOUR = {"kf": "#2a78d6", "white": "#eb6834", "raw": "#1baf7a"}
AXES = rx_filter.AXES


def _truth(motion, t: float) -> dict:
    e, n, ve, vn = motion.state(t)
    return {"east": (e, ve), "north": (n, vn), "up": (0.0, 0.0)}


def filter_config(exp: dict, case: dict) -> rx_filter.FilterConfig:
    mode = exp["filter"]["modes"][case.get("speed", "none")]
    g = exp["gnss"]
    return rx_filter.FilterConfig(
        sigma_a=mode["sigma_a"],
        sigma_v0=mode["sigma_v0"],
        sigma_white=g["sigma_white_m"],
        sigma_corr=g["sigma_corr_m"],
        tau_s=g["tau_s"],
    )


def run_case(exp: dict, case: dict) -> dict:
    """Per method, a list of runs, each a list of per-tick records (None
    before the first fix).  A record holds, per axis, the estimate, the truth
    and (filters) the standard deviation, for position and velocity."""
    model = gnss.model_from(exp["gnss"])
    motion = motion_from(case["motion"])
    cfg = filter_config(exp, case)
    tick = exp.get("tick_s", 1.0)
    ticks = [i * tick for i in range(1, int(round(exp["duration_s"] / tick)) + 1)]
    out: dict[str, list] = {v: [] for v in exp["filter"]["variants"]}
    out["fix_error"] = []
    for run in exp["runs"]:
        for r in range(exp.get("realisations", 1)):
            fixes = gnss.fixes(model, motion, exp["duration_s"], f"seed-{run['seed']}", r)
            out["fix_error"].append(
                [math.sqrt(sum((f[f"{a}_m"] - f[f"true_{a}_m"]) ** 2 for a in AXES)) for f in fixes]
            )
            for variant in exp["filter"]["variants"]:
                recs = []
                for t, est in zip(ticks, rx_filter.run(fixes, ticks, cfg, variant)):
                    if est is None:
                        recs.append(None)
                        continue
                    tr = _truth(motion, t)
                    rec = {"t": t, "nis": est.get("nis")}
                    for a in AXES:
                        rec[f"p_{a}"], rec[f"v_{a}"] = est[f"{a}_m"], est[f"vel_{a}_ms"]
                        rec[f"tp_{a}"], rec[f"tv_{a}"] = tr[a]
                        rec[f"sp_{a}"], rec[f"sv_{a}"] = est.get(f"sd_{a}_m"), est.get(f"sd_vel_{a}_ms")
                    if "cov_pv" in est:
                        rec["nees"] = rx_filter.nees(
                            est, {f"{a}_m": tr[a][0] for a in AXES} | {f"vel_{a}_ms": tr[a][1] for a in AXES}
                        )
                    recs.append(rec)
                out[variant].append(recs)
    return out


def _steady(exp: dict, runs: list) -> list[dict]:
    t0 = exp.get("steady_state_from_s", 0.0)
    return [r for run in runs for r in run if r is not None and r["t"] >= t0]


def _rms(vals) -> float | None:
    vals = list(vals)
    return round(math.sqrt(sum(v * v for v in vals) / len(vals)), 4) if vals else None


def _err(r: dict, kind: str, axes=AXES) -> float:
    k, tk = ("p", "tp") if kind == "pos" else ("v", "tv")
    return math.sqrt(sum((r[f"{k}_{a}"] - r[f"{tk}_{a}"]) ** 2 for a in axes))


def stats(recs: list[dict], fix: list[float]) -> dict:
    nees = [r["nees"] for r in recs if r.get("nees") is not None]
    nis = [r["nis"] for r in recs if r.get("nis") is not None]
    return {
        "pos_rmse_3d_m": _rms(_err(r, "pos") for r in recs),
        "vel_rmse_3d_ms": _rms(_err(r, "vel") for r in recs),
        "pos_rmse_h_m": _rms(_err(r, "pos", ("east", "north")) for r in recs),
        "vel_rmse_h_ms": _rms(_err(r, "vel", ("east", "north")) for r in recs),
        "gnss_fix_rmse_3d_m": _rms(fix),
        "nees_mean": round(float(np.mean(nees)), 3) if nees else None,
        "nis_mean": round(float(np.mean(nis)), 3) if nis else None,
        "pos_inside_2sd": _inside(recs, "p", "tp", "sp"),
        "vel_inside_2sd": _inside(recs, "v", "tv", "sv"),
    }


def _inside(recs: list[dict], k: str, tk: str, sk: str) -> float | None:
    """Share of (tick, axis) samples whose true value is within the filter's +/-2 sigma."""
    hits = [abs(r[f"{k}_{a}"] - r[f"{tk}_{a}"]) <= 2 * r[f"{sk}_{a}"] for r in recs for a in AXES if r[f"{sk}_{a}"]]
    return round(float(np.mean(hits)), 4) if hits else None


def plot_case(path: str, title: str, res: dict, variants) -> None:
    """One run (the first seed, realisation 0): position error and velocity,
    each axis, all methods, both filters' +/-2 sigma bands."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), sharex=True, facecolor=SURFACE)
    for col, a in enumerate(AXES):
        for row, (k, tk, sk, label) in enumerate(
            (("p", "tp", "sp", "position error (m)"), ("v", "tv", "sv", "velocity (m/s)"))
        ):
            ax = axes[row][col]
            _style(ax)
            for v in ("white", "kf"):
                recs = [r for r in res.get(v, [[]])[0] if r is not None and r["t"] >= 3]
                if not recs:
                    continue
                t = np.array([r["t"] for r in recs])
                centre = np.array([r[f"{k}_{a}"] - (r[f"{tk}_{a}"] if k == "p" else 0.0) for r in recs])
                sd = np.array([r[f"{sk}_{a}"] for r in recs])
                ax.fill_between(t, centre - 2 * sd, centre + 2 * sd, color=VAR_COLOUR[v], alpha=0.13, linewidth=0,
                                label=f"{VAR_LABEL[v]} +/-2 sigma")  # fmt: skip
            for v in variants:
                recs = [r for r in res[v][0] if r is not None and r["t"] >= 3]
                t = [r["t"] for r in recs]
                y = [r[f"{k}_{a}"] - (r[f"{tk}_{a}"] if k == "p" else 0.0) for r in recs]
                if v == "raw":
                    ax.scatter(t, y, s=6, color=VAR_COLOUR[v], alpha=0.7, label=VAR_LABEL[v], zorder=2)
                else:
                    ax.plot(t, y, color=VAR_COLOUR[v], linewidth=1.3, label=VAR_LABEL[v], zorder=3)
            recs = [r for r in res[variants[0]][0] if r is not None and r["t"] >= 3]
            truth = [0.0 if k == "p" else r[f"{tk}_{a}"] for r in recs]
            ax.plot([r["t"] for r in recs], truth, color=INK, linewidth=2, label="truth", zorder=4)
            ax.set_title(f"{a} {label}", color=INK, fontsize=10.5, loc="left")
            if row == 1:
                ax.set_xlabel("Time (s)", color=INK, fontsize=10)
    handles, labels = axes[1][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=6, fontsize=9, frameon=False,
               labelcolor=INK_2)  # fmt: skip
    fig.suptitle(title, color=INK, fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, path)


def plot_rmse(path: str, title: str, rows: list[dict], variants) -> None:
    cases = list(dict.fromkeys(r["case"] for r in rows))
    by = {(r["case"], r["variant"]): r for r in rows}
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), facecolor=SURFACE)
    x = np.arange(len(cases))
    w = 0.8 / len(variants)
    for ax, (key, label) in zip(
        axes, (("pos_rmse_3d_m", "Position RMSE (m)"), ("vel_rmse_3d_ms", "Velocity RMSE (m/s)"))
    ):
        _style(ax)
        ax.grid(axis="x", visible=False)
        for k, v in enumerate(variants):
            vals = [by[(c, v)][key] for c in cases]
            xs = x + (k - (len(variants) - 1) / 2) * w
            ax.bar(xs, vals, width=w - 0.02, color=VAR_COLOUR[v], label=VAR_LABEL[v])
            for xx, val in zip(xs, vals):
                ax.text(xx, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
        ax.axvline(len(cases) - 1.5, color=INK_2, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cases, fontsize=9)
        ax.set_ylabel(label, color=INK, fontsize=10)
        ax.set_ylim(0, max(by[(c, v)][key] for c in cases for v in variants) * 1.15)
    axes[0].legend(loc="upper left", fontsize=9, frameon=False, labelcolor=INK_2, ncol=3)
    fig.suptitle(title, color=INK, fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, path)


def plot_coverage(path: str, title: str, rows: list[dict]) -> None:
    filters = ("kf", "white")
    cases = list(dict.fromkeys(r["case"] for r in rows))
    by = {(r["case"], r["variant"]): r for r in rows}
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), facecolor=SURFACE)
    x = np.arange(len(cases))
    w = 0.8 / len(filters)
    for ax, (key, label) in zip(
        axes,
        (
            ("pos_inside_2sd", "Position: truth inside +/-2 sigma (%)"),
            ("vel_inside_2sd", "Velocity: truth inside +/-2 sigma (%)"),
        ),
    ):
        _style(ax)
        ax.grid(axis="x", visible=False)
        for k, v in enumerate(filters):
            vals = [100 * by[(c, v)][key] for c in cases]
            xs = x + (k - (len(filters) - 1) / 2) * w
            ax.bar(xs, vals, width=w - 0.02, color=VAR_COLOUR[v], label=VAR_LABEL[v])
            for xx, val in zip(xs, vals):
                ax.text(xx, val, f"{val:.0f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.axhline(95, color=INK, linestyle=(0, (5, 3)), linewidth=1.3, label="95%: honest uncertainty")
        ax.axvline(len(cases) - 1.5, color=INK_2, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cases, fontsize=9)
        ax.set_ylabel(label, color=INK, fontsize=10)
        ax.set_ylim(50, 104)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=3, fontsize=9, frameon=False,
               labelcolor=INK_2)  # fmt: skip
    fig.suptitle(title, color=INK, fontsize=12.5, x=0.01, ha="left")
    fig.text(0.01, 0.005, "Below 95%: the filter is overconfident (its real error is larger than it claims). "
             "Above: it is cautious. East, north and up pooled.", color=INK_2, fontsize=8.5)  # fmt: skip
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    _save(fig, path)


def _commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", _REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Receiver Kalman filter vs truth for every motion case (step 3b)")
    p.add_argument("experiment", nargs="?", default=os.path.join(_REPO, "experiments", "receiver_filter.json"))
    p.add_argument("--out", help="output folder (default runs/receiver-filter/<date_time>)")
    args = p.parse_args()
    exp = experiment.load(args.experiment)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = os.path.abspath(args.out or os.path.join(_REPO, "runs", exp["name"], stamp))
    os.makedirs(out, exist_ok=True)
    variants = exp["filter"]["variants"]
    n_runs = len(exp["runs"]) * exp.get("realisations", 1)

    rows, pooled, pooled_fix = [], {v: [] for v in variants}, []
    for case in exp["cases"]:
        res = run_case(exp, case)
        fix = [v for run in res["fix_error"] for v in run]
        pooled_fix += fix
        for v in variants:
            recs = _steady(exp, res[v])
            pooled[v] += recs
            rows.append({"case": case["id"], "variant": v, "runs": n_runs} | stats(recs, fix))
        print(f"  {case['id']}")
    for v in variants:
        rows.append(
            {"case": "all cases", "variant": v, "runs": n_runs * len(exp["cases"])} | stats(pooled[v], pooled_fix)
        )

    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot_rmse(os.path.join(out, "rmse.png"),
              f"Receiver position and velocity RMSE (3D), {n_runs} runs per case, after the first "
              f"{exp.get('steady_state_from_s', 0):g} s", rows, variants)  # fmt: skip
    plot_coverage(os.path.join(out, "coverage.png"),
                  f"How often the truth is inside each Kalman filter's +/-2 sigma, {n_runs} runs per case, after the "
                  f"first {exp.get('steady_state_from_s', 0):g} s", rows)  # fmt: skip
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": exp["name"],
                "experiment_file": exp["_path"],
                "created": datetime.datetime.now().isoformat(timespec="seconds"),
                "gnss": exp["gnss"],
                "filter": exp["filter"],
                "realisations": exp.get("realisations", 1),
                "seeds": [r["seed"] for r in exp["runs"]],
                "steady_state_from_s": exp.get("steady_state_from_s", 0.0),
                "code": _commit(),
            },
            f,
            indent=2,
        )
    print()
    print(f"{'case':<22} {'method':<6} {'pos 3D m':>9} {'vel 3D m/s':>11} {'NEES':>6}")
    for r in rows:
        nees = "-" if r["nees_mean"] is None else f"{r['nees_mean']:.2f}"
        print(f"{r['case']:<22} {r['variant']:<6} {r['pos_rmse_3d_m']:>9.2f} {r['vel_rmse_3d_ms']:>11.3f} {nees:>6}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
