"""Figures that show the GNSS error model (retina_simulation/gnss.py) behaves
as specified.  Step 3a of the receiver-motion study.

Writes to ``runs/gnss-model/<date_time>/`` (or ``--out``):

``gnss_autocorrelation.png``   autocorrelation of the correlated part, theory
                               exp(-|lag|/T) with the measured one on top (as in
                               the paper), and of the total error
``gnss_error_time_series.png`` one 300 s run: total error per axis, its
                               correlated and white parts
``gnss_time_constant.png``     the correlated part for several T, same size
``gnss_rate_check.png``        measured size of the correlated part against T,
                               at 1 Hz and 10 Hz: it stays at sigma_corr
``gnss_horizontal_error.png``  the horizontal error of one run, east vs north,
                               coloured by time: the wander
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

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
KEY = "gnss-figures"
RUN_S = 300.0


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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=SURFACE)
    panels = (
        (axes[0], "Correlated part only (Gauss-Markov)", theory_nu, nu, "theory  exp(-|lag| / T)"),
        (axes[1], "Total error (correlated + white)", theory_tot, tot, "theory  0.90 exp(-|lag| / T), 1 at 0"),
    )
    for ax, title, theory, measured, tlabel in panels:
        _style(ax)
        sym = np.concatenate([measured[:0:-1], measured])
        ax.plot(lags, theory, color=SERIES[0], linewidth=2.2, label=tlabel.replace("0.90", f"{share:.2f}"))
        step = max(1, len(lags) // 60)
        ax.scatter(lags[::step], sym[::step], s=16, color=SERIES[1], zorder=4, label="measured (simulated)")
        ax.axhline(1 / math.e, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        for x in (-tau, tau):
            ax.axvline(x, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xticks([-3 * tau, -2 * tau, -tau, 0, tau, 2 * tau, 3 * tau])
        ax.set_xticklabels(["-3T", "-2T", "-T", "0", "T", "2T", "3T"])
        ax.set_yticks([0, 1 / math.e, share, 1])
        ax.set_yticklabels(["0", "1/e", f"{share:.2f}", "1"])
        ax.set_ylim(-0.05, 1.05)
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
        f"sigma_corr^2 / (sigma_corr^2 + sigma_white^2) = {share:.2f}.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    _save(fig, path)
    k = int(round(tau * rate))
    return {"acf_at_T_measured": round(float(nu[k]), 3), "acf_at_T_theory": round(math.exp(-1), 3)}


def plot_time_series(path: str, model: gnss.GnssModel) -> None:
    rows = gnss.fixes(model, Motion(), RUN_S, KEY)
    t = [r["t_s"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, facecolor=SURFACE)
    for ax, axis in zip(axes, gnss.AXES):
        _style(ax)
        total = [r[f"{axis}_m"] - r[f"true_{axis}_m"] for r in rows]
        ax.plot(t, total, color=SERIES[0], linewidth=1.2, label="total error")
        ax.plot(t, [r[f"corr_{axis}_m"] for r in rows], color=SERIES[1], linewidth=2, label="correlated part")
        ax.scatter(t, [r[f"white_{axis}_m"] for r in rows], s=5, color=MUTED, label="white part", zorder=2)
        ax.axhline(0, color=GRID, linewidth=1)
        ax.set_ylabel(f"{axis} error (m)", color=INK, fontsize=9.5)
    axes[0].legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=INK_2, ncol=3)
    axes[-1].set_xlabel("Time (s)", color=INK, fontsize=10)
    axes[-1].set_xlim(0, RUN_S)
    fig.suptitle(
        f"GNSS position error over one {RUN_S:g} s run (sigma_corr {model.sigma_corr_m:g} m, T {model.tau_s:g} s, "
        f"sigma_white {model.sigma_white_m:g} m, {model.rate_hz:g} Hz)",
        color=INK,
        fontsize=11.5,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path)


def plot_time_constants(path: str, model: gnss.GnssModel, taus=(1.0, 10.0, 60.0, 300.0)) -> None:
    fig, axes = plt.subplots(len(taus), 1, figsize=(11, 2.1 * len(taus) + 0.8), sharex=True, sharey=True)
    fig.set_facecolor(SURFACE)
    lim = 3.5 * model.sigma_corr_m
    for ax, tau, colour in zip(axes, taus, SERIES):
        _style(ax)
        m = gnss.GnssModel(sigma_corr_m=model.sigma_corr_m, tau_s=tau, rate_hz=model.rate_hz)
        times = gnss.stamps(m, RUN_S, KEY)
        nu = gnss.errors(m, times, KEY)["east"]["corr"]
        ax.plot(times, nu, color=colour, linewidth=1.6)
        ax.axhline(0, color=GRID, linewidth=1)
        for s in (-model.sigma_corr_m, model.sigma_corr_m):
            ax.axhline(s, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("east (m)", color=INK, fontsize=9)
        ax.set_title(f"T = {tau:g} s", color=INK, fontsize=10, loc="left")
    axes[-1].set_xlabel("Time (s)", color=INK, fontsize=10)
    axes[-1].set_xlim(0, RUN_S)
    fig.suptitle(
        f"Correlated part for different time constants, same size (sigma_corr {model.sigma_corr_m:g} m, dashed) "
        "and the same random draws",
        color=INK,
        fontsize=11.5,
        x=0.02,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
    rows = gnss.fixes(model, Motion(), RUN_S, KEY)
    e = np.array([r["east_m"] - r["true_east_m"] for r in rows])
    n = np.array([r["north_m"] - r["true_north_m"] for r in rows])
    t = np.array([r["t_s"] for r in rows])
    fig, ax = plt.subplots(figsize=(7.5, 7), facecolor=SURFACE)
    _style(ax)
    ax.plot(e, n, color=GRID, linewidth=1, zorder=1)
    sc = ax.scatter(e, n, c=t, cmap="viridis", s=14, zorder=2)
    total = math.sqrt(model.sigma_corr_m**2 + model.sigma_white_m**2)
    ring = np.linspace(0, 2 * math.pi, 200)
    ax.plot(
        total * np.cos(ring),
        total * np.sin(ring),
        color=INK,
        linewidth=1.1,
        linestyle=(0, (5, 3)),
        label=f"1 sigma ({total:.2f} m)",
    )
    ax.scatter([0], [0], marker="+", s=120, color=INK, zorder=3, label="true position")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("East error (m)", color=INK, fontsize=10)
    ax.set_ylabel("North error (m)", color=INK, fontsize=10)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, labelcolor=INK_2)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("time (s)", color=INK_2, fontsize=9)
    fig.suptitle(
        f"Horizontal GNSS error over one {RUN_S:g} s run: it wanders", color=INK, fontsize=12, x=0.02, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
