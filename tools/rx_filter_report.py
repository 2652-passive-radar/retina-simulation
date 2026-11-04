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

``summary.csv``              per case x variant: horizontal and vertical position
                             and velocity RMSE (after ``steady_state_from_s``),
                             the error the filter claims, how often the truth is
                             inside its +/-2 sigma, the GNSS fixes' own error,
                             mean NEES / NIS
``accuracy.png``             how accurate: position and velocity RMSE per kind of
                             motion and method, values written on the bars
``velocity_tracking.png``    what it looks like: true east velocity against each
                             method's estimate for straight driving, circling
                             and oscillating, one run
``honesty.png``              does the filter know how wrong it is: the error it
                             claims against the error it makes, and how often
                             the truth falls inside its +/-2 sigma (95% ideal)
``error_vs_acceleration.png`` why it lags: velocity error against the receiver's
                             true acceleration, with the filter's sigma_a marked
``manifest.json``            settings and code version

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
from experiment_report import GRID, INK, INK_2, MUTED, SURFACE, _save, _style  # noqa: E402

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
SHOWN_CASES = ("linear-east-vehicle", "circle-vehicle", "oscillate-vehicle")
# Cases that give identical numbers (same GNSS draws through a linear filter
# on unaccelerated motion) are shown once.
GROUPS = (
    ("Static / straight walk", ("static", "linear-east-walk", "linear-north-walk")),
    ("Straight drive", ("linear-east-vehicle", "linear-north-vehicle")),
    ("Circle walk", ("circle-walk",)),
    ("Circle drive", ("circle-vehicle",)),
    ("Oscillate walk", ("oscillate-walk",)),
    ("Oscillate drive", ("oscillate-vehicle",)),
)
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
    h = 1e-3
    _, _, ve1, vn1 = motion.state(t + h)
    _, _, ve0, vn0 = motion.state(t - h)
    acc = math.hypot((ve1 - ve0) / (2 * h), (vn1 - vn0) / (2 * h))
    return {"east_m": e, "north_m": n, "up_m": 0.0, "vel_east_ms": ve, "vel_north_ms": vn, "vel_up_ms": 0.0, "acc": acc}


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
                        "e_north": est["north_m"] - tr["north_m"],
                        "ev_east": est["vel_east_ms"] - tr["vel_east_ms"],
                        "ev_north": est["vel_north_ms"] - tr["vel_north_ms"],
                        "sd_east": est.get("sd_east_m"),
                        "sd_north": est.get("sd_north_m"),
                        "sdv_east": est.get("sd_vel_east_ms"),
                        "sdv_north": est.get("sd_vel_north_ms"),
                        "v_east_true": tr["vel_east_ms"],
                        "v_east_est": est["vel_east_ms"],
                        "acc": tr["acc"],
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


def _stats(recs: list[dict], fix: list[float]) -> dict:
    has_sd = bool(recs) and recs[0]["sd_east"] is not None
    out = {
        "pos_rmse_h_m": _rms(r["pos_h"] for r in recs),
        "vel_rmse_h_ms": _rms(r["vel_h"] for r in recs),
        "pos_rmse_up_m": _rms(r["pos_up"] for r in recs),
        "vel_rmse_up_ms": _rms(r["vel_up"] for r in recs),
        "pos_sd_claimed_h_m": None,
        "vel_sd_claimed_h_ms": None,
        "pos_inside_2sd": None,
        "vel_inside_2sd": None,
        "gnss_fix_rmse_h_m": _rms(fix),
        "nees_mean": _mean(r["nees"] for r in recs),
        "nis_mean": _mean(r["nis"] for r in recs),
    }
    if has_sd:
        out["pos_sd_claimed_h_m"] = round(math.sqrt(np.mean([r["sd_east"] ** 2 + r["sd_north"] ** 2 for r in recs])), 4)
        out["vel_sd_claimed_h_ms"] = round(
            math.sqrt(np.mean([r["sdv_east"] ** 2 + r["sdv_north"] ** 2 for r in recs])), 4
        )
        out["pos_inside_2sd"] = round(
            float(np.mean([abs(r[f"e_{a}"]) <= 2 * r[f"sd_{a}"] for r in recs for a in ("east", "north")])), 3
        )
        out["vel_inside_2sd"] = round(
            float(np.mean([abs(r[f"ev_{a}"]) <= 2 * r[f"sdv_{a}"] for r in recs for a in ("east", "north")])), 3
        )
    return out


def _steady(exp: dict, runs: list) -> list[dict]:
    t0 = exp.get("steady_state_from_s", 0.0)
    return [r for run in runs for r in run if r is not None and r["t"] >= t0]


def summarise(exp: dict, case: dict, res: dict) -> list[dict]:
    fix = [v for run in res["fix_error"] for v in run]
    return [
        {"case": case["id"], "speed": case.get("speed", ""), "variant": v, "runs": len(res[v])}
        | _stats(_steady(exp, res[v]), fix)
        for v in exp["filter"]["variants"]
    ]


def group_stats(exp: dict, results: dict) -> dict:
    """{(group, variant): stats} with the group's cases pooled."""
    out = {}
    for name, ids in GROUPS:
        ids = [i for i in ids if i in results]
        if not ids:
            continue
        fix = [v for i in ids for run in results[i]["fix_error"] for v in run]
        for v in exp["filter"]["variants"]:
            out[(name, v)] = _stats([r for i in ids for r in _steady(exp, results[i][v])], fix)
    return out


def plot_accuracy(path: str, title: str, gstats: dict, variants) -> None:
    groups = [g for g, _ in GROUPS if (g, variants[0]) in gstats]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), facecolor=SURFACE)
    y = np.arange(len(groups))[::-1]
    h = 0.8 / len(variants)
    panels = (
        ("pos_rmse_h_m", "Horizontal position error, RMS (m)", "m", "%.2f"),
        ("vel_rmse_h_ms", "Horizontal velocity error, RMS (m/s)", "m/s", "%.2f"),
    )
    for ax, (key, label, _unit, fmt) in zip(axes, panels):
        _style(ax)
        ax.grid(axis="y", visible=False)
        for k, v in enumerate(variants):
            vals = [gstats[(g, v)][key] for g in groups]
            ys = y + ((len(variants) - 1) / 2 - k) * h
            ax.barh(ys, vals, height=h - 0.03, color=VAR_COLOUR[v], label=VAR_LABEL[v])
            for yy, val in zip(ys, vals):
                ax.text(val, yy, "  " + fmt % val, va="center", fontsize=8, color=INK_2)
        if key == "pos_rmse_h_m":
            fix = gstats[(groups[0], variants[0])]["gnss_fix_rmse_h_m"]
            ax.axvline(fix, color=INK, linestyle=(0, (5, 3)), linewidth=1.2)
            ax.text(fix, y[0] + 0.62, f" GNSS fixes alone: {fix:.2f} m", fontsize=8.5, color=INK, va="bottom")
        else:
            ax.axvline(VEL_FOR_2HZ, color=INK, linestyle=(0, (5, 3)), linewidth=1.2)
            ax.text(
                VEL_FOR_2HZ, y[0] + 0.62, f" GD tolerance ~{VEL_FOR_2HZ:.2f} m/s", fontsize=8.5, color=INK, va="bottom"
            )
        ax.set_yticks(y)
        ax.set_yticklabels(groups, fontsize=9.5)
        ax.set_xlabel(label, color=INK, fontsize=10)
        ax.set_xlim(0, max(gstats[(g, v)][key] for g in groups for v in variants) * 1.22)
        ax.set_ylim(-0.6, len(groups) - 0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.945), ncol=3, fontsize=9, frameon=False,
               labelcolor=INK_2)  # fmt: skip
    fig.suptitle(title, color=INK, fontsize=12.5, x=0.02, ha="left")
    fig.text(
        0.02,
        0.005,
        "Lower is better. RMS over 20 runs per motion (5 seeds x 4 GNSS realisations), every second after the first 30 s. "
        "GD tolerance: receiver velocity error that shifts a 521 MHz channel's Doppler by the radar's own 2 Hz noise "
        "(worst geometry).",
        color=INK_2,
        fontsize=8.3,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    _save(fig, path)


def plot_tracking(path: str, title: str, results: dict, variants, window_s: float = 150.0,
                  zoom=(15.0, 45.0)) -> None:  # fmt: skip
    shown = [c for c in SHOWN_CASES if c in results]
    names = {"linear-east-vehicle": "Straight drive (east, 15 m/s)", "circle-vehicle": "Circle drive (15 m/s)",
             "oscillate-vehicle": "Oscillate drive (east-west, peak 15 m/s)"}  # fmt: skip
    fig, axes = plt.subplots(len(shown), 2, figsize=(15, 3.4 * len(shown) + 1.0), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.6, 1]}, squeeze=False)  # fmt: skip
    for row, cid in enumerate(shown):
        for col, (t0, t1) in enumerate(((0.0, window_s), zoom)):
            ax = axes[row][col]
            _style(ax)
            truth = [r for r in results[cid][variants[0]][0] if r is not None and t0 <= r["t"] <= t1]
            t = [r["t"] for r in truth]
            for v in variants[::-1]:
                recs = [r for r in results[cid][v][0] if r is not None and t0 <= r["t"] <= t1]
                if v == "raw":
                    ax.scatter([r["t"] for r in recs], [r["v_east_est"] for r in recs], s=9, color=VAR_COLOUR[v],
                               alpha=0.8, label=VAR_LABEL[v], zorder=2)  # fmt: skip
                else:
                    ax.plot([r["t"] for r in recs], [r["v_east_est"] for r in recs], color=VAR_COLOUR[v],
                            linewidth=2 if v == "kf" else 1.4, label=VAR_LABEL[v], zorder=4 if v == "kf" else 3)  # fmt: skip
            kf = [r for r in results[cid].get("kf", [[]])[0] if r is not None and max(t0, 3.0) <= r["t"] <= t1]
            if kf:
                ax.fill_between([r["t"] for r in kf], [r["v_east_est"] - 2 * r["sdv_east"] for r in kf],
                                [r["v_east_est"] + 2 * r["sdv_east"] for r in kf], color=VAR_COLOUR["kf"],
                                alpha=0.13, linewidth=0, label="KF's own +/-2 sigma")  # fmt: skip
            ax.plot(t, [r["v_east_true"] for r in truth], color=INK, linewidth=2.4, label="true velocity", zorder=5)
            lo, hi = min(r["v_east_true"] for r in truth), max(r["v_east_true"] for r in truth)
            pad = max(2.5, 0.15 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(t0, t1)
            if col == 0:
                ax.set_title(names.get(cid, cid), color=INK, fontsize=10.5, loc="left")
                ax.set_ylabel("East velocity (m/s)", color=INK, fontsize=9.5)
                ax.axvspan(*zoom, color=GRID, alpha=0.5, zorder=0)
            else:
                ax.set_title(f"zoom: {zoom[0]:g}-{zoom[1]:g} s", color=INK_2, fontsize=9.5, loc="left")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.02, 0.955), ncol=5, fontsize=9, frameon=False,
               labelcolor=INK_2)  # fmt: skip
    for ax in axes[-1]:
        ax.set_xlabel("Time (s)", color=INK, fontsize=10)
    fig.suptitle(title, color=INK, fontsize=12.5, x=0.02, ha="left")
    fig.text(
        0.02,
        0.005,
        "One run each (seed 101, GNSS realisation 0). Good estimates sit on the black line. Straight: all methods "
        "on it, raw noisiest. Circle and oscillate: the KF trails the truth where the velocity changes fastest "
        "(shaded window, zoomed on the right).",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    _save(fig, path)


def plot_honesty(path: str, title: str, gstats: dict, variants) -> None:
    filt = [v for v in variants if v != "raw"]
    groups = [g for g, _ in GROUPS if (g, filt[0]) in gstats]
    markers = ("o", "s", "^", "D", "v", "P")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1.15]})
    rows = (
        ("vel_sd_claimed_h_ms", "vel_rmse_h_ms", "vel_inside_2sd", "Velocity", "m/s"),
        ("pos_sd_claimed_h_m", "pos_rmse_h_m", "pos_inside_2sd", "Position", "m"),
    )
    for r, (claim, actual, inside, what, unit) in enumerate(rows):
        ax = axes[r][0]
        _style(ax)
        top = max(max(gstats[(g, v)][claim], gstats[(g, v)][actual]) for g in groups for v in filt)
        lim = top * 1.15
        xs = np.linspace(0, lim, 50)
        ax.fill_between(xs, xs, lim, color="#f6e1d9", alpha=0.6, linewidth=0, zorder=0)
        ax.plot(xs, xs, color=INK, linewidth=1.2, linestyle=(0, (5, 3)))
        for v in filt:
            for g, m in zip(groups, markers):
                st = gstats[(g, v)]
                ax.scatter(st[claim], st[actual], s=80, marker=m, color=VAR_COLOUR[v], edgecolors=SURFACE,
                           linewidths=1, zorder=3)  # fmt: skip
        ax.text(lim * 0.04, lim * 0.96, "overconfident: real error\nlarger than claimed", fontsize=8.5, color=INK_2,
                va="top")  # fmt: skip
        ax.text(lim * 0.96, lim * 0.04, "too cautious: claims more\nerror than it makes", fontsize=8.5, color=INK_2,
                ha="right")  # fmt: skip
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{what} error the filter claims, RMS ({unit})", color=INK, fontsize=10)
        ax.set_ylabel(f"{what} error it actually makes, RMS ({unit})", color=INK, fontsize=10)
        ax.set_title(f"{what}: on the dashed line = honest", color=INK, fontsize=10.5, loc="left")

        ax = axes[r][1]
        _style(ax)
        ax.grid(axis="y", visible=False)
        y = np.arange(len(groups))[::-1]
        h = 0.8 / len(filt)
        for k, v in enumerate(filt):
            vals = [100 * gstats[(g, v)][inside] for g in groups]
            ys = y + ((len(filt) - 1) / 2 - k) * h
            ax.barh(ys, vals, height=h - 0.04, color=VAR_COLOUR[v])
            for yy, val in zip(ys, vals):
                ax.text(val, yy, f"  {val:.0f}%", va="center", fontsize=8, color=INK_2)
        ax.axvline(95, color=INK, linestyle=(0, (5, 3)), linewidth=1.2)
        ax.text(95, y[0] + 0.55, " 95% = honest", fontsize=8.5, color=INK, va="bottom")
        ax.set_yticks(y)
        ax.set_yticklabels(groups, fontsize=9.5)
        ax.set_xlim(0, 112)
        ax.set_ylim(-0.6, len(groups) - 0.2)
        ax.set_xlabel(
            f"How often the true {what.lower()} is inside the filter's +/-2 sigma (%)", color=INK, fontsize=10
        )
        ax.set_title("Below 95%: overconfident. Above: too cautious", color=INK, fontsize=10.5, loc="left")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=VAR_COLOUR[v], markersize=9) for v in filt]
    handles += [plt.Line2D([], [], marker=m, linestyle="", color=MUTED, markersize=8) for m in markers[: len(groups)]]
    fig.legend(handles, [VAR_LABEL[v] for v in filt] + groups, loc="upper left", bbox_to_anchor=(0.02, 0.96),
               ncol=4, fontsize=9, frameon=False, labelcolor=INK_2)  # fmt: skip
    fig.suptitle(title, color=INK, fontsize=12.5, x=0.02, ha="left")
    fig.text(
        0.02,
        0.005,
        "Left: each point is one kind of motion. Right: per axis, every second after 30 s, 20 runs. Honest uncertainty "
        "can be passed downstream (to GD, the tracker) and checked; an overconfident filter hides its own errors.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    _save(fig, path)


def plot_error_vs_acceleration(path: str, title: str, exp: dict, results: dict, variants) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), facecolor=SURFACE)
    edges = np.array([0, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    for ax, speed, title_s in zip(axes, ("walk", "vehicle"), ("Walking cases", "Vehicle cases")):
        _style(ax)
        ids = [c["id"] for c in exp["cases"] if c.get("speed") == speed]
        sa = exp["filter"]["modes"][speed]["sigma_a"]
        for v in variants:
            recs = [r for i in ids for r in _steady(exp, results[i][v])]
            acc = np.array([r["acc"] for r in recs])
            err = np.array([r["vel_h"] for r in recs])
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (acc >= lo) & (acc < hi)
                if m.sum() >= 50:
                    xs.append(float(acc[m].mean()))
                    ys.append(math.sqrt(float(np.mean(err[m] ** 2))))
            ax.plot(xs, ys, "-o", color=VAR_COLOUR[v], linewidth=2, markersize=5, label=VAR_LABEL[v])
        ax.axvline(sa, color=INK, linestyle=(0, (5, 3)), linewidth=1.2)
        ax.text(sa, 0.97, f" filter assumes\n sigma_a = {sa:g} m/s^2", transform=ax.get_xaxis_transform(),
                fontsize=8.5, color=INK, va="top")  # fmt: skip
        ax.axhline(VEL_FOR_2HZ, color=MUTED, linestyle=(0, (2, 2)), linewidth=1)
        ax.set_title(title_s, color=INK, fontsize=10.5, loc="left")
        ax.set_xlabel("Receiver's true acceleration at the time (m/s^2)", color=INK, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Horizontal velocity error, RMS (m/s)", color=INK, fontsize=10)
    axes[0].legend(loc="center right", fontsize=8.5, frameon=False, labelcolor=INK_2)
    fig.suptitle(title, color=INK, fontsize=12.5, x=0.02, ha="left")
    fig.text(
        0.02,
        0.005,
        f"All cases of that speed pooled, binned by acceleration (bins with >= 50 samples). Dotted: GD tolerance "
        f"~{VEL_FOR_2HZ:.2f} m/s. Filtered velocity lags the truth by an amount that grows steadily with acceleration; "
        "raw differencing barely lags but is noisy.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
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
    gstats = group_stats(exp, results)
    plot_accuracy(os.path.join(out, "accuracy.png"), "How accurately is the receiver's motion estimated from GNSS?",
                  gstats, variants)  # fmt: skip
    plot_tracking(os.path.join(out, "velocity_tracking.png"), "What the estimates look like: east velocity, true vs estimated",
                  results, variants)  # fmt: skip
    plot_honesty(os.path.join(out, "honesty.png"), "Does the filter know how wrong it is?", gstats, variants)
    plot_error_vs_acceleration(os.path.join(out, "error_vs_acceleration.png"),
                               "Why some cases are worse: velocity error grows with the receiver's acceleration",
                               exp, results, variants)  # fmt: skip
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
