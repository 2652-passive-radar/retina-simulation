"""Figures and tables for step 3c (experiments/receiver_gnss.json): GD's aircraft
position and velocity error with the receiver pose from the truth, from raw GNSS
fixes, and from the receiver Kalman filter.

Reads ``cases/<case>/doppler-on/seed-<n>/analysis/offline_solves_gd_<source>[_r<k>].csv``
under one output folder and writes beside ``cases/``:

``rmse.png``     horizontal position and velocity RMSE per motion case and over
                 all cases, one bar per pose source, pooled over every run (seed
                 x GNSS realisation); whiskers span the per-run RMSEs
``summary.csv``  per case x pose source (and over all cases): aircraft position
                 and velocity RMSE after the warm-up and over the whole run, the
                 spread over runs, the receiver pose error the source gave GD,
                 and the RMS error of GD's Doppler correction

Errors count every solve the solver produced, as in steps 1 and 2.  Solves in
the first ``steady_state_from_s`` seconds are left out of the headline numbers
(the filter's warm-up), for every source alike.
"""

import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from experiment_report import INK, INK_2, SURFACE, _read_csv, _save, _style  # noqa: E402
from gnss_pose import solve_name  # noqa: E402
from motion_report import dopplers_for, seed_dir  # noqa: E402

SOURCE_LABEL = {
    "truth": "true trajectory",
    "gnss": "GNSS fixes (differenced velocity)",
    "kf": "GNSS + Kalman filter",
}
SOURCE_COLOUR = {"truth": "#52514e", "gnss": "#1baf7a", "kf": "#2a78d6"}
CHANNELS = ("chan", "chek", "cisc")

FIELDS = [
    "case", "source", "runs", "solves", "pos_rmse_m", "vel_rmse_ms", "pos_rmse_all_m", "vel_rmse_all_ms",
    "pos_rmse_run_min_m", "pos_rmse_run_max_m", "vel_rmse_run_min_ms", "vel_rmse_run_max_ms",
    "rx_pos_rmse_m", "rx_vel_rmse_ms", "rxdop_corr_err_rms_hz",
]  # fmt: skip


def _f(v):
    return None if v in (None, "") else float(v)


def _rms(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return math.sqrt(sum(v * v for v in vals) / len(vals)) if vals else None


def load(path: str) -> list[dict]:
    out = []
    for r in _read_csv(path):
        if r.get("err_horiz_m") in (None, ""):
            continue
        corr = [
            _f(r.get(f"rxdop_corr_hz_{c}")) - _f(r.get(f"rxdop_true_hz_{c}"))
            for c in CHANNELS
            if _f(r.get(f"rxdop_corr_hz_{c}")) is not None and _f(r.get(f"rxdop_true_hz_{c}")) is not None
        ]
        out.append(
            {
                "t_s": float(r["t_s"]),
                "pos": float(r["err_horiz_m"]),
                "vel": _f(r.get("verr_horiz_ms")),
                "rx_pos": _f(r.get("rx_pos_err_m")),
                "rx_vel": _f(r.get("rx_vel_err_ms")),
                "corr_err": corr,
            }
        )
    return out


def runs_for(out: str, exp: dict, case: dict, source: str) -> list[list[dict]]:
    """Every run of one case and pose source: a list of solved rows per (seed, realisation)."""
    runs = []
    reals = [None] if source == "truth" else range(exp.get("realisations", 1))
    for doppler in dopplers_for(exp, case):
        for run in exp["runs"]:
            d = os.path.join(seed_dir(out, case["id"], doppler, run["seed"]), "analysis")
            for r in reals:
                runs.append(load(os.path.join(d, solve_name(source, r))))
    return runs


def stats(case_id: str, source: str, runs: list[list[dict]], t0: float) -> dict:
    steady = [[x for x in run if x["t_s"] >= t0] for run in runs]
    rows = [x for run in steady for x in run]
    allrows = [x for run in runs for x in run]
    per_pos = [_rms(x["pos"] for x in run) for run in steady if run]
    per_vel = [_rms(x["vel"] for x in run) for run in steady if run]
    return {
        "case": case_id,
        "source": source,
        "runs": len(runs),
        "solves": len(rows),
        "pos_rmse_m": _rms(x["pos"] for x in rows),
        "vel_rmse_ms": _rms(x["vel"] for x in rows),
        "pos_rmse_all_m": _rms(x["pos"] for x in allrows),
        "vel_rmse_all_ms": _rms(x["vel"] for x in allrows),
        "pos_rmse_run_min_m": min(per_pos, default=None),
        "pos_rmse_run_max_m": max(per_pos, default=None),
        "vel_rmse_run_min_ms": min(per_vel, default=None),
        "vel_rmse_run_max_ms": max(per_vel, default=None),
        "rx_pos_rmse_m": _rms(x["rx_pos"] for x in rows),
        "rx_vel_rmse_ms": _rms(x["rx_vel"] for x in rows),
        "rxdop_corr_err_rms_hz": _rms(e for x in rows for e in x["corr_err"]),
    }


def plot_rmse(path: str, title: str, summary: list[dict], sources) -> None:
    cases = list(dict.fromkeys(r["case"] for r in summary))
    by = {(r["case"], r["source"]): r for r in summary}
    fig, axes = plt.subplots(2, 1, figsize=(16, 9.5), facecolor=SURFACE)
    x = np.arange(len(cases))
    w = 0.8 / len(sources)
    panels = (
        ("pos_rmse_m", "pos_rmse_run_min_m", "pos_rmse_run_max_m", "Aircraft position RMSE (m)", "%.1f"),
        ("vel_rmse_ms", "vel_rmse_run_min_ms", "vel_rmse_run_max_ms", "Aircraft velocity RMSE (m/s)", "%.2f"),
    )
    for ax, (key, lo_key, hi_key, label, fmt) in zip(axes, panels):
        _style(ax)
        ax.grid(axis="x", visible=False)
        top = 0.0
        for k, s in enumerate(sources):
            vals = np.array([by[(c, s)][key] for c in cases], dtype=float)
            lo = np.array([by[(c, s)][lo_key] for c in cases], dtype=float)
            hi = np.array([by[(c, s)][hi_key] for c in cases], dtype=float)
            xs = x + (k - (len(sources) - 1) / 2) * w
            ax.bar(xs, vals, width=w - 0.02, color=SOURCE_COLOUR[s], label=SOURCE_LABEL[s])
            ax.errorbar(xs, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
            for xx, val, h in zip(xs, vals, hi):
                ax.text(xx, h, fmt % val, ha="center", va="bottom", fontsize=7, color=INK_2)
            top = max(top, float(np.nanmax(hi)))
        ax.axvline(len(cases) - 1.5, color=INK_2, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cases, fontsize=8.5, rotation=20, ha="right")
        ax.set_ylabel(label, color=INK, fontsize=10)
        ax.set_ylim(0, top * 1.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=3, fontsize=9.5, frameon=False,
               labelcolor=INK_2)  # fmt: skip
    fig.suptitle(title, color=INK, fontsize=12.5, x=0.01, ha="left")
    fig.text(
        0.01,
        0.005,
        "Bars: RMSE pooled over every run of the case (5 seeds; x GNSS realisations for the estimated poses). "
        "Whiskers: lowest and highest single-run RMSE.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    _save(fig, path)


def build(out: str, exp: dict) -> list[dict]:
    t0 = exp.get("steady_state_from_s", 0.0)
    sources = exp["pose_sources"]
    summary, pooled = [], {s: [] for s in sources}
    for case in exp["cases"]:
        for s in sources:
            runs = runs_for(out, exp, case, s)
            pooled[s] += runs
            summary.append(stats(case["id"], s, runs, t0))
    for s in sources:
        summary.append(stats("all cases", s, pooled[s], t0))

    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for r in summary:
            wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    plot_rmse(
        os.path.join(out, "rmse.png"),
        f"GD aircraft error by receiver trajectory source (from {t0:g} s; receiver Doppler on)",
        summary,
        sources,
    )
    return summary


def print_table(summary: list[dict]) -> None:
    def fmt(v, spec):
        return "-" if v is None else format(v, spec)

    print()
    print(f"{'case':<22} {'source':<6} {'pos m':>7} {'vel m/s':>8} {'rx pos m':>9} {'rx vel m/s':>11} {'corr Hz':>8}")
    for r in summary:
        print(
            f"{r['case']:<22} {r['source']:<6} {fmt(r['pos_rmse_m'], '7.1f')} {fmt(r['vel_rmse_ms'], '8.2f')} "
            f"{fmt(r['rx_pos_rmse_m'], '9.2f')} {fmt(r['rx_vel_rmse_ms'], '11.3f')} "
            f"{fmt(r['rxdop_corr_err_rms_hz'], '8.3f')}"
        )
