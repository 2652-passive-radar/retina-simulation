"""Step 3b: the receiver Kalman filter against the true trajectory, for every
motion case, fed by the step 3a GNSS error model.  No radar is involved.

For each case of ``experiments/receiver_filter.json`` (the step 1 motions),
each seed and each GNSS realisation, the fixes are the true trajectory plus
modelled GNSS error; each filter variant (``rx_filter.VARIANTS``) estimates
the receiver's position and velocity at every radar tick, and the estimate is
scored against the truth.  The filter's acceleration noise and initial
velocity uncertainty come from the case's platform mode (none, walk,
vehicle); its GNSS error parameters match the model's.

Writes to ``runs/receiver-filter/<date_time>/`` (or ``--out``):

``summary.csv``            per case x variant: horizontal and vertical position
                           and velocity RMSE (after ``steady_state_from_s``),
                           the GNSS fixes' own position error, mean NEES / NIS
``overview.png``           horizontal position and velocity RMSE per case and
                           variant
``time_series_<case>.png`` one run's east position and velocity error over time,
                           every variant, with the filter's own +/-2 sigma
``consistency.png``        mean NEES over time per case against its 95% band:
                           is the filter's uncertainty honest?
``manifest.json``          settings and code version

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
from experiment_report import GRID, INK, INK_2, SURFACE, _save, _style  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from retina_simulation import experiment, gnss  # noqa: E402
from retina_simulation.motion import motion_from  # noqa: E402

VAR_LABEL = {
    "kf": "KF with GNSS drift state",
    "white": "KF, GNSS error treated as white",
    "raw": "raw fixes, differenced velocity",
}
VAR_COLOUR = {"kf": "#2a78d6", "white": "#eb6834", "raw": "#1baf7a"}
SHOWN_CASES = ("linear-east-vehicle", "oscillate-vehicle", "circle-vehicle")
DOF = 6
# Receiver velocity error that shifts a 521 MHz channel's Doppler by the 2 Hz
# measurement noise in the worst geometry (|u_RP + u_TR| = 2): 2 dv / lambda.
_LAMBDA_521_M = 299792458.0 / 521e6
VEL_FOR_2HZ = 2.0 * _LAMBDA_521_M / 2.0


def _chi2_quantile(p: float, k: float) -> float:
    """Wilson-Hilferty approximation of the chi-square quantile (no scipy)."""
    z = math.sqrt(2) * _erfinv(2 * p - 1)
    return k * (1 - 2 / (9 * k) + z * math.sqrt(2 / (9 * k))) ** 3


def _erfinv(y: float) -> float:
    a = 0.147
    ln = math.log(1 - y * y)
    t = 2 / (math.pi * a) + ln / 2
    return math.copysign(math.sqrt(math.sqrt(t * t - ln / a) - t), y)


def _truth(motion, t: float) -> dict:
    e, n, ve, vn = motion.state(t)
    return {"east_m": e, "north_m": n, "up_m": 0.0, "vel_east_ms": ve, "vel_north_ms": vn, "vel_up_ms": 0.0}


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
    """Every run of one case: per variant, a list of runs, each a list of
    per-tick records (None before the first fix)."""
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
                [math.hypot(f["east_m"] - f["true_east_m"], f["north_m"] - f["true_north_m"]) for f in fixes]
            )
            for variant in exp["filter"]["variants"]:
                recs = []
                for t, est in zip(ticks, rx_filter.run(fixes, ticks, cfg, variant)):
                    if est is None:
                        recs.append(None)
                        continue
                    tr = _truth(motion, t)
                    rec = {
                        "t": t,
                        "pos_h": math.hypot(est["east_m"] - tr["east_m"], est["north_m"] - tr["north_m"]),
                        "vel_h": math.hypot(
                            est["vel_east_ms"] - tr["vel_east_ms"], est["vel_north_ms"] - tr["vel_north_ms"]
                        ),
                        "pos_up": est["up_m"] - tr["up_m"],
                        "vel_up": est["vel_up_ms"] - tr["vel_up_ms"],
                        "e_east": est["east_m"] - tr["east_m"],
                        "ev_east": est["vel_east_ms"] - tr["vel_east_ms"],
                        "sd_east": est.get("sd_east_m"),
                        "sdv_east": est.get("sd_vel_east_ms"),
                        "nis": est.get("nis"),
                        "nees": rx_filter.nees(est, tr) if "cov_pv" in est else None,
                    }
                    recs.append(rec)
                out[variant].append(recs)
    return out


def _rms(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(math.sqrt(sum(v * v for v in vals) / len(vals)), 4) if vals else None


def _mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def summarise(exp: dict, case: dict, res: dict) -> list[dict]:
    t0 = exp.get("steady_state_from_s", 0.0)
    rows = []
    fix = [v for run in res["fix_error"] for v in run]
    for variant in exp["filter"]["variants"]:
        recs = [r for run in res[variant] for r in run if r is not None and r["t"] >= t0]
        rows.append(
            {
                "case": case["id"],
                "speed": case.get("speed", ""),
                "variant": variant,
                "runs": len(res[variant]),
                "pos_rmse_h_m": _rms(r["pos_h"] for r in recs),
                "vel_rmse_h_ms": _rms(r["vel_h"] for r in recs),
                "pos_rmse_up_m": _rms(r["pos_up"] for r in recs),
                "vel_rmse_up_ms": _rms(r["vel_up"] for r in recs),
                "gnss_fix_rmse_h_m": _rms(fix),
                "nees_mean": _mean(r["nees"] for r in recs),
                "nis_mean": _mean(r["nis"] for r in recs),
            }
        )
    return rows


def plot_overview(path: str, title: str, summary: list[dict], cases: list[dict], variants) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=SURFACE)
    ids = [c["id"] for c in cases]
    x = np.arange(len(ids))
    w = 0.8 / len(variants)
    by = {(r["case"], r["variant"]): r for r in summary}
    for ax, (key, label) in zip(
        axes, (("pos_rmse_h_m", "Horizontal position RMSE (m)"), ("vel_rmse_h_ms", "Horizontal velocity RMSE (m/s)"))
    ):
        _style(ax)
        ax.grid(axis="x", visible=False)
        for k, v in enumerate(variants):
            vals = [by[(c, v)][key] or 0 for c in ids]
            ax.bar(x + (k - (len(variants) - 1) / 2) * w, vals, width=w - 0.03, color=VAR_COLOUR[v], label=VAR_LABEL[v])
        if key == "pos_rmse_h_m":
            fix = by[(ids[0], variants[0])]["gnss_fix_rmse_h_m"]
            ax.axhline(
                fix, color=INK, linestyle=(0, (5, 3)), linewidth=1.3, label=f"GNSS fixes themselves ({fix:.2f} m)"
            )
        else:
            ax.axhline(VEL_FOR_2HZ, color=INK, linestyle=(0, (5, 3)), linewidth=1.3,
                       label=f"{VEL_FOR_2HZ:.2f} m/s: worst-case 2 Hz Doppler error at 521 MHz")  # fmt: skip
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=35, ha="right", fontsize=8.5)
        ax.set_ylabel(label, color=INK, fontsize=9.5)
        ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK_2)
    fig.suptitle(title, color=INK, fontsize=12, x=0.02, ha="left")
    fig.text(
        0.02,
        0.005,
        "Pooled over 5 seeds x GNSS realisations, every radar tick after the warm-up. "
        "Velocity line: receiver velocity error that would equal the radar's own Doppler noise in the worst geometry.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    _save(fig, path)


def plot_time_series(path: str, title: str, res: dict, variants) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, facecolor=SURFACE)
    for ax, (key, sd, label) in zip(
        axes, (("e_east", "sd_east", "East position error (m)"), ("ev_east", "sdv_east", "East velocity error (m/s)"))
    ):
        _style(ax)
        for v in variants:
            recs = [r for r in res[v][0] if r is not None]
            t = [r["t"] for r in recs]
            ax.plot(t, [r[key] for r in recs], color=VAR_COLOUR[v], linewidth=1.4 if v == "kf" else 1.0,
                    label=VAR_LABEL[v], zorder=3 if v == "kf" else 2)  # fmt: skip
            if v == "kf":
                s = np.array([r[sd] for r in recs])
                ax.fill_between(t, -2 * s, 2 * s, color=VAR_COLOUR[v], alpha=0.12, linewidth=0, label="KF +/-2 sigma")
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_ylabel(label, color=INK, fontsize=9.5)
    axes[1].set_ylim(-3, 3)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_2, ncol=2)
    axes[-1].set_xlabel("Time (s)", color=INK, fontsize=10)
    fig.suptitle(title, color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path)


def plot_consistency(path: str, title: str, results: dict, cases: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.4), facecolor=SURFACE, sharey=True)
    for ax, v in zip(axes, ("kf", "white")):
        _style(ax)
        n_runs = None
        for k, c in enumerate(cases):
            runs = results[c["id"]].get(v)
            if not runs:
                continue
            n_runs = len(runs)
            n = min(len(r) for r in runs)
            t = [runs[0][i]["t"] for i in range(n) if runs[0][i] is not None]
            m = [np.mean([r[i]["nees"] for r in runs]) for i in range(n) if runs[0][i] is not None]
            ax.plot(t, m, linewidth=1.1, color=plt.cm.tab10(k % 10), label=c["id"])
        if n_runs:
            lo = _chi2_quantile(0.025, n_runs * DOF) / n_runs
            hi = _chi2_quantile(0.975, n_runs * DOF) / n_runs
            ax.axhspan(lo, hi, color=GRID, alpha=0.8, zorder=0, label=f"95% band for {n_runs} runs")
            ax.axhline(DOF, color=INK, linewidth=1, linestyle=(0, (5, 3)))
        ax.set_yscale("log")
        ax.set_title(VAR_LABEL[v], color=INK, fontsize=10.5, loc="left")
        ax.set_xlabel("Time (s)", color=INK, fontsize=10)
    axes[0].set_ylabel(f"Mean NEES of position + velocity ({DOF} dof)", color=INK, fontsize=9.5)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False, labelcolor=INK_2)
    fig.suptitle(title, color=INK, fontsize=12, x=0.02, ha="left")
    fig.text(
        0.02,
        0.005,
        f"Inside the band (around {DOF}): the filter's stated uncertainty matches its actual error. Above it: "
        "overconfident. Below: too cautious.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
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

    results, summary = {}, []
    for case in exp["cases"]:
        results[case["id"]] = run_case(exp, case)
        summary += summarise(exp, case, results[case["id"]])
        print(f"  {case['id']}")

    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    plot_overview(os.path.join(out, "overview.png"), f"{exp['name']}: receiver position and velocity from GNSS",
                  summary, exp["cases"], variants)  # fmt: skip
    for cid in SHOWN_CASES:
        if cid in results:
            plot_time_series(os.path.join(out, f"time_series_{cid}.png"),
                             f"{cid}: one run (seed {exp['runs'][0]['seed']}, realisation 0)", results[cid], variants)  # fmt: skip
    plot_consistency(os.path.join(out, "consistency.png"), f"{exp['name']}: is the filter's uncertainty honest?",
                     results, exp["cases"])  # fmt: skip
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
    print(f"{'case':<22} {'variant':<6} {'pos m':>7} {'vel m/s':>8} {'NEES':>6} {'NIS':>6}")
    for r in summary:
        nees = "-" if r["nees_mean"] is None else f"{r['nees_mean']:.2f}"
        nis = "-" if r["nis_mean"] is None else f"{r['nis_mean']:.2f}"
        print(
            f"{r['case']:<22} {r['variant']:<6} {r['pos_rmse_h_m']:>7.2f} {r['vel_rmse_h_ms']:>8.3f} {nees:>6} {nis:>6}"
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
