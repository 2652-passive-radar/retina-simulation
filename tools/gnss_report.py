"""Figures and tables for step 3c (experiments/receiver_gnss.json): GD's aircraft
position and velocity error with the receiver pose from the truth, from raw GNSS
fixes, and from the receiver Kalman filter.

Reads ``cases/<case>/doppler-on/seed-<n>/analysis/offline_solves_gd_<source>[_r<k>].csv``
under one output folder and writes beside ``cases/``:

``rmse.png``     horizontal position and velocity RMSE per motion case and over
                 all cases, one bar per pose source, pooled over every run (seed
                 x GNSS realisation); whiskers span the per-run RMSEs
``velocity_tolerance.png``  how much receiver velocity error GD tolerates:
                 every GNSS- and KF-based solve binned by the receiver velocity
                 error GD was given, aircraft velocity RMSE per bin against the
                 truth-fed RMSE; and GD's Doppler correction error per bin
                 against the 2 Hz Doppler measurement noise
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


def paired_rows(out: str, exp: dict, t0: float) -> dict[str, dict[str, np.ndarray]]:
    """For each estimated pose source, every solve after the warm-up paired with
    the truth-fed solve of the same recording and tick: the receiver velocity
    error, the aircraft velocity error with the estimate and with the truth, and
    the RMS of GD's Doppler correction error over the channels."""
    out_by = {}
    for source in exp["pose_sources"]:
        if source == "truth":
            continue
        rx, est, tru, corr = [], [], [], []
        for case in exp["cases"]:
            for doppler in dopplers_for(exp, case):
                for run in exp["runs"]:
                    d = os.path.join(seed_dir(out, case["id"], doppler, run["seed"]), "analysis")
                    truth = {x["t_s"]: x["vel"] for x in load(os.path.join(d, solve_name("truth", None)))}
                    for r in range(exp.get("realisations", 1)):
                        for x in load(os.path.join(d, solve_name(source, r))):
                            if x["t_s"] < t0 or x["rx_vel"] is None or x["vel"] is None:
                                continue
                            if truth.get(x["t_s"]) is None:
                                continue
                            rx.append(x["rx_vel"])
                            est.append(x["vel"])
                            tru.append(truth[x["t_s"]])
                            corr.append(_rms(x["corr_err"]) if x["corr_err"] else np.nan)
        out_by[source] = {
            k: np.array(v, dtype=float) for k, v in (("rx", rx), ("est", est), ("tru", tru), ("corr", corr))
        }
    return out_by


def tolerance_curve(rx, est, tru, edges, min_n=100, n_boot=300, seed=0):
    """Per receiver-velocity-error bin: centre, % increase of aircraft velocity RMSE
    over the truth-fed RMSE at the same ticks, and a 90 % bootstrap interval."""
    rng = np.random.default_rng(seed)
    xs, inc, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = np.flatnonzero((rx >= a) & (rx < b))
        if len(m) < min_n:
            continue
        e2, t2 = est[m] ** 2, tru[m] ** 2
        xs.append(float(rx[m].mean()))
        inc.append(100 * (math.sqrt(e2.mean() / t2.mean()) - 1))
        k = rng.integers(0, len(m), size=(n_boot, len(m)))
        boot = 100 * (np.sqrt(e2[k].mean(axis=1) / t2[k].mean(axis=1)) - 1)
        lo.append(float(np.percentile(boot, 5)))
        hi.append(float(np.percentile(boot, 95)))
    return np.array(xs), np.array(inc), np.array(lo), np.array(hi)


def first_crossing(xs, ys, level):
    """Receiver error where the curve first rises through ``level`` (linear interpolation)."""
    for i in range(1, len(xs)):
        if ys[i - 1] < level <= ys[i]:
            return float(xs[i - 1] + (level - ys[i - 1]) * (xs[i] - xs[i - 1]) / (ys[i] - ys[i - 1]))
    return None


def plot_velocity_tolerance(path: str, paired: dict, threshold_pct: float = 10.0, noise_hz: float = 2.0) -> dict:
    """How much receiver velocity error GD tolerates, as % increase of the aircraft
    velocity RMSE over the truth-fed RMSE at the same ticks."""
    edges = np.arange(0.0, 2.81, 0.1)
    rx = np.concatenate([v["rx"] for v in paired.values()])
    est = np.concatenate([v["est"] for v in paired.values()])
    tru = np.concatenate([v["tru"] for v in paired.values()])
    corr = np.concatenate([v["corr"] for v in paired.values()])
    xs, inc, lo, hi = tolerance_curve(rx, est, tru, edges)
    ok = np.isfinite(corr)
    hz_per_ms = float(np.sum(rx[ok] * corr[ok]) / np.sum(rx[ok] ** 2))
    limit = first_crossing(xs, inc, threshold_pct)
    x_max = float(edges[-1])

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(12, 8.2), sharex=True, facecolor=SURFACE,
                                 gridspec_kw={"height_ratios": [2.4, 1]})  # fmt: skip
    _style(ax)
    _style(bx)
    if limit is not None:
        ax.axvspan(0, limit, color="#dff2e7", zorder=0)
        ax.text(limit / 2, 0.97, "tolerated", transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=11, color="#1b7a52", fontweight="bold")  # fmt: skip
        ax.axvline(limit, color="#1b7a52", linewidth=1.5)
        ax.text(limit, 0.86, f"  +{threshold_pct:g}% at {limit:.2f} m/s", transform=ax.get_xaxis_transform(),
                fontsize=10, color="#1b7a52", fontweight="bold")  # fmt: skip
    ax.fill_between(xs, lo, hi, color=INK, alpha=0.12, linewidth=0, label="90% interval (bootstrap)")
    ax.plot(xs, inc, "-o", color=INK, linewidth=2.2, markersize=4.5, label="raw GNSS and KF solves pooled")
    ax.axhline(0, color=INK_2, linewidth=0.9)
    ax.axhline(threshold_pct, color="#c0392b", linewidth=1.2, linestyle=(0, (5, 3)))
    ax.text(x_max, threshold_pct, f"+{threshold_pct:g}% threshold ", ha="right", va="bottom", fontsize=9,
            color="#c0392b")  # fmt: skip
    x_noise = noise_hz / hz_per_ms
    ax.axvline(x_noise, color=INK_2, linewidth=1, linestyle=":")
    ax.text(x_noise, 0.02, f" correction error = radar's {noise_hz:g} Hz Doppler noise", rotation=90,
            transform=ax.get_xaxis_transform(), va="bottom", ha="right", fontsize=8.5, color=INK_2)  # fmt: skip
    ax.set_ylabel("Aircraft velocity RMSE increase\nover true-trajectory GD (%)", color=INK, fontsize=10)
    ax.set_ylim(min(-5.0, float(np.min(lo)) - 2), max(threshold_pct * 2.5, float(np.max(hi)) + 5))
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.8), fontsize=9, frameon=False, labelcolor=INK_2)
    top = ax.secondary_xaxis("top", functions=(lambda v: v * hz_per_ms, lambda v: v / hz_per_ms))
    top.set_xlabel(f"GD's Doppler correction error per channel (Hz, RMS; about {hz_per_ms:.2f} Hz per m/s)",
                   color=INK_2, fontsize=9)  # fmt: skip
    top.tick_params(colors=INK_2, labelsize=8.5)

    width = 0.05
    for source, v in paired.items():
        h, _ = np.histogram(v["rx"], bins=np.arange(0, x_max + width, width))
        frac = 100 * h / max(len(v["rx"]), 1)
        bx.step(np.arange(0, x_max, width) + width / 2, frac, where="mid", color=SOURCE_COLOUR[source], linewidth=1.8,
                label=SOURCE_LABEL[source])  # fmt: skip
        p95 = float(np.percentile(v["rx"], 95))
        bx.axvline(p95, color=SOURCE_COLOUR[source], linewidth=1.2, linestyle=(0, (4, 2)))
        bx.text(p95, 0.95, f" 95% below {p95:.2f}", transform=bx.get_xaxis_transform(), va="top", fontsize=8.5,
                color=SOURCE_COLOUR[source])  # fmt: skip
    if limit is not None:
        bx.axvspan(0, limit, color="#dff2e7", zorder=0)
    bx.set_ylabel("Share of solves (%)", color=INK, fontsize=10)
    bx.set_xlabel("Receiver velocity error given to GD (m/s, horizontal)", color=INK, fontsize=10)
    bx.set_xlim(0, x_max)
    bx.set_ylim(0, None)
    bx.legend(loc="upper right", fontsize=9, frameon=False, labelcolor=INK_2)
    bx.set_title("Where each receiver-trajectory source actually operates", color=INK, fontsize=10, loc="left")
    fig.suptitle("How much receiver velocity error GD tolerates before aircraft velocity accuracy degrades",
                 color=INK, fontsize=12.5, x=0.01, ha="left")  # fmt: skip
    fig.text(
        0.01,
        0.005,
        "Each solve compared with the true-trajectory solve of the same recording and second; binned by receiver "
        "velocity error (0.1 m/s bins, at least 100 solves); every motion case, after the warm-up.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    _save(fig, path)
    return {"tolerance_ms": limit, "hz_per_ms": hz_per_ms, "noise_crossing_ms": x_noise}


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
    if "truth" in sources and len(sources) > 1:
        tol = plot_velocity_tolerance(os.path.join(out, "velocity_tolerance.png"), paired_rows(out, exp, t0))
        if tol["tolerance_ms"] is not None:
            print(f"Receiver velocity tolerance (+10% aircraft velocity RMSE): {tol['tolerance_ms']:.2f} m/s")

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
