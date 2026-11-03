"""Figures that show the GNSS error model (retina_simulation/gnss.py) behaves
as specified.  Step 3a of the receiver-motion study.

Writes to ``runs/gnss-model/<date_time>/`` (or ``--out``):

``gnss_autocorrelation.png``   autocorrelation of the correlated part, theory
                               exp(-|lag|/T) with the measured one on top (as in
                               the paper), and of the total error; thin lines are
                               what single 300 s runs measure
``gnss_error_time_series.png`` five 300 s runs per axis: the correlated part
                               (thick) and the total error (thin), over the band
                               90% of runs stay in
``gnss_time_constant.png``     the correlated part for several T, five runs each,
                               same size
``gnss_rate_check.png``        measured size of the correlated part against T,
                               at 1 Hz and 10 Hz: it stays at sigma_corr
``gnss_horizontal_error.png``  the horizontal error of five runs, east vs north,
                               over every fix of many runs, with the circles 39%
                               and 95% of fixes fall in
``stats.csv``                  set and measured values

Usage::

    python tools/gnss_figures.py [--sigma-white 0.5] [--sigma-corr 1.5] [--tau 60] [--rate 1] [--out DIR]
"""

import argparse
import csv
import datetime
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from experiment_report import GRID, INK, INK_2, MUTED, SURFACE, _save, _style  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from retina_simulation import gnss  # noqa: E402
from retina_simulation.motion import Motion  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
KEY = "gnss-figures"
RUN_S = 300.0
N_SHOWN = 5
N_BAND = 300
BAND = "#d9d8d3"


def _runs(model: gnss.GnssModel, n: int) -> list[list[dict]]:
    """``n`` independent 300 s runs of a static receiver (realisations 0..n-1)."""
    return [gnss.fixes(model, Motion(), RUN_S, KEY, k) for k in range(n)]


def _band(runs: list[list[dict]], field) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """5th and 95th percentile across runs at each fix index (runs share no stamps,
    so the index stands in for time; the phase is under one update interval)."""
    n = min(len(r) for r in runs)
    vals = np.array([[field(f) for f in r[:n]] for r in runs])
    t = np.mean([[f["t_s"] for f in r[:n]] for r in runs], axis=0)
    return t, np.percentile(vals, 5, axis=0), np.percentile(vals, 95, axis=0)


def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    d = x - x.mean()
    n = len(d)
    f = np.fft.rfft(d, 2 * n)
    r = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    return r / r[0]


def _long_errors(model: gnss.GnssModel, duration_s: float, realisation: int = 0) -> dict:
    times = gnss.stamps(model, duration_s, KEY, realisation)
    err = gnss.errors(model, times, KEY, realisation)
    return {a: {k: np.array(v) for k, v in err[a].items()} for a in gnss.AXES}


def plot_autocorrelation(path: str, model: gnss.GnssModel) -> dict:
    tau, rate = model.tau_s, model.rate_hz
    dur = max(400 * tau, 20000.0)
    err = _long_errors(model, dur)
    max_lag = int(round(3 * tau * rate))
    lags = np.arange(-max_lag, max_lag + 1) / rate
    nu = np.mean([_acf(err[a]["corr"], max_lag) for a in gnss.AXES], axis=0)
    tot = np.mean([_acf(err[a]["corr"] + err[a]["white"], max_lag) for a in gnss.AXES], axis=0)
    share = model.sigma_corr_m**2 / (model.sigma_corr_m**2 + model.sigma_white_m**2)
    theory_nu = np.exp(-np.abs(lags) / tau)
    theory_tot = np.where(lags == 0, 1.0, share * theory_nu)
    short = []
    for k in range(20):
        e = _long_errors(model, RUN_S, k + 1)["east"]
        n = min(max_lag, len(e["corr"]) - 1)
        short.append((_acf(e["corr"], n), _acf(e["corr"] + e["white"], n)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)
    panels = (
        (axes[0], "Correlated part only (Gauss-Markov)", theory_nu, nu, "theory  exp(-|lag| / T)", 0),
        (axes[1], "Total error (correlated + white)", theory_tot, tot, "theory  0.90 exp(-|lag| / T), 1 at 0", 1),
    )
    for ax, title, theory, measured, tlabel, which in panels:
        _style(ax)
        for j, pair in enumerate(short):
            one = pair[which]
            x = np.arange(-len(one) + 1, len(one)) / rate
            ax.plot(x, np.concatenate([one[:0:-1], one]), color=BAND, linewidth=0.9, zorder=1,
                    label=f"single {RUN_S:g} s runs (20)" if j == 0 else None)  # fmt: skip
        sym = np.concatenate([measured[:0:-1], measured])
        ax.plot(lags, theory, color=SERIES[0], linewidth=2.2, label=tlabel.replace("0.90", f"{share:.2f}"))
        step = max(1, len(lags) // 60)
        ax.scatter(lags[::step], sym[::step], s=16, color=SERIES[1], zorder=4, label="measured (simulated)")
        ax.axhline(1 / math.e, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        for x in (-tau, tau):
            ax.axvline(x, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xticks([-3 * tau, -2 * tau, -tau, 0, tau, 2 * tau, 3 * tau])
        ax.set_xticklabels(["-3T", "-2T", "-T", "0", "T", "2T", "3T"])
        ax.set_yticks([-0.5, 0, 1 / math.e, share, 1])
        ax.set_yticklabels(["-0.5", "0", "1/e", f"{share:.2f}", "1"])
        ax.set_ylim(-0.6, 1.05)
        ax.set_xlabel(f"Lag (T = {tau:g} s)", color=INK, fontsize=10)
        ax.set_ylabel("Autocorrelation", color=INK, fontsize=10)
        ax.set_title(title, color=INK, fontsize=10.5, loc="left")
        ax.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=INK_2)
    fig.suptitle("GNSS error autocorrelation", color=INK, fontsize=12, x=0.02, ha="left")
    fig.text(
        0.02,
        0.005,
        f"sigma_corr {model.sigma_corr_m:g} m, sigma_white {model.sigma_white_m:g} m, {rate:g} Hz; measured over "
        f"{dur:,.0f} s, mean of the 3 axes. The white part adds a spike at lag 0 and scales the rest by "
        f"sigma_corr^2 / (sigma_corr^2 + sigma_white^2) = {share:.2f}.\nA single {RUN_S:g} s run spans only "
        f"{RUN_S / tau:g} T: what it measures scatters widely and sits below the theory (removing its own mean "
        "takes out part of the slow wander), dipping negative at long lags.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    _save(fig, path)
    k = int(round(tau * rate))
    return {"acf_at_T_measured": round(float(nu[k]), 3), "acf_at_T_theory": round(math.exp(-1), 3)}


def plot_time_series(path: str, model: gnss.GnssModel) -> None:
    runs = _runs(model, N_BAND)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, facecolor=SURFACE)
    for ax, axis in zip(axes, gnss.AXES):
        _style(ax)
        t, lo, hi = _band(runs, lambda f, a=axis: f[f"{a}_m"] - f[f"true_{a}_m"])
        ax.fill_between(t, lo, hi, color=BAND, alpha=0.7, linewidth=0, label=f"90% of {N_BAND} runs")
        for k, rows in enumerate(runs[:N_SHOWN]):
            tt = [r["t_s"] for r in rows]
            ax.plot(
                tt, [r[f"{axis}_m"] - r[f"true_{axis}_m"] for r in rows], color=SERIES[k], linewidth=0.7, alpha=0.45
            )
            ax.plot(tt, [r[f"corr_{axis}_m"] for r in rows], color=SERIES[k], linewidth=2, label=f"run {k + 1}")
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_ylabel(f"{axis} error (m)", color=INK, fontsize=9.5)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK_2, ncol=6)
    axes[-1].set_xlabel("Time (s)", color=INK, fontsize=10)
    axes[-1].set_xlim(0, RUN_S)
    fig.suptitle(
        f"GNSS position error, {N_SHOWN} independent {RUN_S:g} s runs (sigma_corr {model.sigma_corr_m:g} m, "
        f"T {model.tau_s:g} s, sigma_white {model.sigma_white_m:g} m, {model.rate_hz:g} Hz)",
        color=INK,
        fontsize=11.5,
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.005,
        "Thick: each run's correlated part. Thin, same colour: its total error (correlated + white). "
        f"Grey band: where 90% of {N_BAND} runs are at each moment.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _save(fig, path)


def plot_time_constants(path: str, model: gnss.GnssModel, taus=(1.0, 10.0, 60.0, 300.0)) -> None:
    fig, axes = plt.subplots(len(taus), 1, figsize=(11, 2.2 * len(taus) + 0.9), sharex=True, sharey=True)
    fig.set_facecolor(SURFACE)
    lim = 3.5 * model.sigma_corr_m
    for ax, tau in zip(axes, taus):
        _style(ax)
        m = gnss.GnssModel(sigma_corr_m=model.sigma_corr_m, tau_s=tau, rate_hz=model.rate_hz)
        series = []
        for k in range(N_BAND):
            times = gnss.stamps(m, RUN_S, KEY, k)
            series.append((times, gnss.errors(m, times, KEY, k)["east"]["corr"]))
        n = min(len(v) for _, v in series)
        vals = np.array([v[:n] for _, v in series])
        t = np.mean([tt[:n] for tt, _ in series], axis=0)
        ax.fill_between(
            t, np.percentile(vals, 5, axis=0), np.percentile(vals, 95, axis=0), color=BAND, alpha=0.7, linewidth=0
        )
        for k, (times, nu) in enumerate(series[:N_SHOWN]):
            ax.plot(times, nu, color=SERIES[k], linewidth=1.4)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        for s in (-model.sigma_corr_m, model.sigma_corr_m):
            ax.axhline(s, color=INK, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("east (m)", color=INK, fontsize=9)
        ax.set_title(f"T = {tau:g} s", color=INK, fontsize=10, loc="left")
    axes[-1].set_xlabel("Time (s)", color=INK, fontsize=10)
    axes[-1].set_xlim(0, RUN_S)
    fig.suptitle(
        f"Correlated part for different time constants: {N_SHOWN} runs each, same size "
        f"(sigma_corr {model.sigma_corr_m:g} m, dashed)",
        color=INK,
        fontsize=11.5,
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.005,
        f"Run k uses the same random draws in every panel. Grey band: where 90% of {N_BAND} runs are at each moment; "
        "it is the same width for every T.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _save(fig, path)


def plot_rate_check(path: str, model: gnss.GnssModel, taus=(1.0, 10.0, 60.0, 300.0)) -> list[dict]:
    stats = []
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    _style(ax)
    for rate, colour, marker in ((1.0, SERIES[0], "o"), (10.0, SERIES[1], "s")):
        ys = []
        for tau in taus:
            m = gnss.GnssModel(sigma_corr_m=model.sigma_corr_m, tau_s=tau, rate_hz=rate)
            err = _long_errors(m, max(200 * tau, 4000.0))
            s = float(np.std(np.concatenate([err[a]["corr"] for a in gnss.AXES])))
            ys.append(s)
            stats.append(
                {
                    "rate_hz": rate,
                    "tau_s": tau,
                    "sigma_corr_set_m": model.sigma_corr_m,
                    "sigma_corr_measured_m": round(s, 3),
                }
            )
        ax.plot(taus, ys, marker=marker, color=colour, linewidth=2, markersize=7, label=f"{rate:g} Hz")
    ax.axhline(
        model.sigma_corr_m, color=INK, linewidth=1.2, linestyle=(0, (5, 3)), label=f"set: {model.sigma_corr_m:g} m"
    )
    ax.set_xscale("log")
    ax.set_xticks(taus)
    ax.set_xticklabels([f"{t:g}" for t in taus])
    ax.set_ylim(0, 2 * model.sigma_corr_m)
    ax.set_xlabel("Time constant T (s)", color=INK, fontsize=10)
    ax.set_ylabel("Measured std of the correlated part (m)", color=INK, fontsize=10)
    ax.legend(loc="upper left", fontsize=9, frameon=False, labelcolor=INK_2)
    fig.suptitle(
        "Size of the correlated error does not depend on T or the update rate",
        color=INK,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.005,
        "Each point: 200 T (at least 4,000 s) of simulated error, 3 axes pooled.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    _save(fig, path)
    return stats


def plot_horizontal(path: str, model: gnss.GnssModel) -> None:
    runs = _runs(model, N_BAND)
    fig, ax = plt.subplots(figsize=(8, 7.6), facecolor=SURFACE)
    _style(ax)
    allp = np.array([[f["east_m"] - f["true_east_m"], f["north_m"] - f["true_north_m"]] for r in runs for f in r])
    ax.scatter(
        allp[:, 0], allp[:, 1], s=2, color=BAND, alpha=0.5, linewidths=0, zorder=1, label=f"every fix, {N_BAND} runs"
    )
    for k, rows in enumerate(runs[:N_SHOWN]):
        e = [r["east_m"] - r["true_east_m"] for r in rows]
        n = [r["north_m"] - r["true_north_m"] for r in rows]
        ce, cn = [r["corr_east_m"] for r in rows], [r["corr_north_m"] for r in rows]
        ax.scatter(e, n, s=5, color=SERIES[k], alpha=0.35, linewidths=0, zorder=2)
        ax.plot(ce, cn, color=SERIES[k], linewidth=2, zorder=3, label=f"run {k + 1}")
        ax.scatter([ce[0]], [cn[0]], s=45, color=SERIES[k], edgecolors=SURFACE, linewidths=1.2, zorder=4)
        ax.scatter([ce[-1]], [cn[-1]], s=65, marker="s", color=SERIES[k], edgecolors=SURFACE, linewidths=1.2, zorder=4)
    sig = math.sqrt(model.sigma_corr_m**2 + model.sigma_white_m**2)
    ring = np.linspace(0, 2 * math.pi, 200)
    for r, style, label in (
        (sig, (0, (5, 3)), f"1 sigma per axis, {sig:.2f} m: holds 39% in 2D"),
        (sig * math.sqrt(-2 * math.log(0.05)), "-", f"{sig * math.sqrt(-2 * math.log(0.05)):.2f} m: holds 95%"),
    ):
        ax.plot(r * np.cos(ring), r * np.sin(ring), color=INK, linewidth=1.1, linestyle=style, zorder=3, label=label)
    ax.scatter([0], [0], marker="+", s=140, color=INK, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East error (m)", color=INK, fontsize=10)
    ax.set_ylabel("North error (m)", color=INK, fontsize=10)
    ax.legend(
        loc="upper left", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK_2, markerscale=3
    )
    fig.suptitle(
        f"Horizontal GNSS error: {N_SHOWN} runs of {RUN_S:g} s (dot: start, square: end)",
        color=INK,
        fontsize=12,
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.005,
        "Line: each run's correlated part (the wander); dots, same colour: its fixes.\nEach run covers a different "
        "part of the cloud; over many runs the fixes fill it evenly around the true position (+).",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    _save(fig, path)


def main() -> None:
    d = gnss.GnssModel()
    p = argparse.ArgumentParser(description="Figures of the GNSS error model (step 3a)")
    p.add_argument("--sigma-white", type=float, default=d.sigma_white_m, help="white noise sigma, m")
    p.add_argument("--sigma-corr", type=float, default=d.sigma_corr_m, help="correlated part sigma (steady state), m")
    p.add_argument("--tau", type=float, default=d.tau_s, help="time constant T, s")
    p.add_argument("--rate", type=float, default=d.rate_hz, help="update rate, Hz")
    p.add_argument("--out", help="output folder (default runs/gnss-model/<date_time>)")
    args = p.parse_args()
    model = gnss.GnssModel(
        sigma_white_m=args.sigma_white, sigma_corr_m=args.sigma_corr, tau_s=args.tau, rate_hz=args.rate
    )
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = os.path.abspath(args.out or os.path.join(_REPO, "runs", "gnss-model", stamp))
    os.makedirs(out, exist_ok=True)

    acf = plot_autocorrelation(os.path.join(out, "gnss_autocorrelation.png"), model)
    plot_time_series(os.path.join(out, "gnss_error_time_series.png"), model)
    plot_time_constants(os.path.join(out, "gnss_time_constant.png"), model)
    stats = plot_rate_check(os.path.join(out, "gnss_rate_check.png"), model)
    plot_horizontal(os.path.join(out, "gnss_horizontal_error.png"), model)

    err = _long_errors(model, 20000.0)
    white = float(np.std(np.concatenate([err[a]["white"] for a in gnss.AXES])))
    with open(os.path.join(out, "stats.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "set", "measured"])
        w.writerow(["sigma_white_m", model.sigma_white_m, round(white, 3)])
        w.writerow(["acf_at_lag_T", acf["acf_at_T_theory"], acf["acf_at_T_measured"]])
        for s in stats:
            w.writerow(
                [
                    f"sigma_corr_m (T {s['tau_s']:g} s, {s['rate_hz']:g} Hz)",
                    s["sigma_corr_set_m"],
                    s["sigma_corr_measured_m"],
                ]
            )
    print(f"Wrote {out}")
    print(f"  autocorrelation at lag T: measured {acf['acf_at_T_measured']}, theory {acf['acf_at_T_theory']}")
    for s in stats:
        print(f"  T {s['tau_s']:>5g} s, {s['rate_hz']:>4g} Hz: sigma_corr measured {s['sigma_corr_measured_m']} m")


if __name__ == "__main__":
    main()
