"""Figures and tables for the receiver-motion study (experiments/receiver_motion.json).

Reads ``cases/<case>/doppler-<off|on>/seed-<n>/analysis/offline_solves_<algorithm>.csv``
under one output folder and writes, beside ``cases/``:

``summary.csv``          one row per case x receiver-Doppler setting x algorithm
                         x seed, plus a mean row per case/setting/algorithm
``overview.png``         position and velocity RMSE of every moving case, S beside
                         G, against the static baseline; receiver Doppler off
                         (top) and on (bottom)
``receiver_paths.png``   the receiver trajectories, walking and vehicle
``paths_map.png``        the flights (as in step 0)

and in each case folder ``rmse_vs_time.png`` (S, G and the static baseline,
mean over seeds with the seed range shaded) and ``error_vs_offset.png`` (every
solve's position error against how far the receiver was from its registered
position).

Errors here count every solve the solver produced, including ones the
publication gates would reject: under receiver motion the stationary
algorithm's gates can stop publishing, and availability alone would then hide
how wrong it was.  ``summary.csv`` carries the published-only numbers too.
"""

import csv
import json
import math
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from experiment_report import (  # noqa: E402
    GRID,
    INK,
    INK_2,
    MUTED,
    SURFACE,
    _p95,
    _read_csv,
    _rmse,
    _round,
    _save,
    _style,
    _truth_path,
    plot_map,
    windowed,
)

ALGORITHMS = ("stationary", "geometric")
ALG_LABEL = {"stationary": "S: stationary", "geometric": "G: geometric compensation"}
ALG_COLOUR = {"stationary": "#eb6834", "geometric": "#2a78d6"}
BASELINE_COLOUR = MUTED
PROFILE_COLOUR = {"linear-east": "#1baf7a", "linear-north": "#eda100", "oscillate": "#e87ba4", "circle": "#6250d6"}

SUMMARY_FIELDS = [
    "case", "speed", "rx_doppler", "algorithm", "seed", "ticks", "solved", "published", "availability",
    "pos_rmse_m", "pos_median_m", "pos_p95_m", "vel_rmse_ms", "vel_median_ms",
    "pos_rmse_published_m", "vel_rmse_published_ms", "rejected_rms_delay", "rejected_rms_doppler",
    "end_offset_m", "pos_err_last30s_m",
]  # fmt: skip


def doppler_dir(on: bool) -> str:
    return "doppler-on" if on else "doppler-off"


def dopplers_for(exp: dict, case: dict) -> list[bool]:
    """A static receiver has no Doppler of its own: one setting is enough."""
    return [False] if case["motion"].get("type", "static") == "static" else list(exp.get("rx_doppler", [False]))


def seed_dir(out: str, case_id: str, doppler: bool, seed: int) -> str:
    return os.path.join(out, "cases", case_id, doppler_dir(doppler), f"seed-{seed}")


def _f(v):
    return None if v in (None, "") else float(v)


def load(seed_path: str, algorithm: str) -> list[dict]:
    rows = _read_csv(os.path.join(seed_path, "analysis", f"offline_solves_{algorithm}.csv"))
    return [
        {
            "t_s": float(r["t_s"]),
            "solved": r.get("err_horiz_m") not in (None, ""),
            "published": r["published"] == "True",
            "reason": r["reason"],
            "pos": _f(r.get("err_horiz_m")),
            "vel": _f(r.get("verr_horiz_ms")),
            "offset": _f(r.get("rx_offset_m")) or 0.0,
            "bearing": _f(r.get("rx_offset_bearing_deg")),
        }
        for r in rows
    ]


def summary_row(case: dict, doppler: bool, algorithm: str, seed, rows: list[dict]) -> dict:
    solved = [r for r in rows if r["solved"]]
    pub = [r for r in solved if r["published"]]
    pos = [r["pos"] for r in solved]
    vel = [r["vel"] for r in solved if r["vel"] is not None]
    t_end = max((r["t_s"] for r in rows), default=0)
    last = [r["pos"] for r in solved if r["t_s"] > t_end - 30]
    return {
        "case": case["id"],
        "speed": case.get("speed", ""),
        "rx_doppler": "on" if doppler else "off",
        "algorithm": algorithm,
        "seed": seed,
        "ticks": len(rows),
        "solved": len(solved),
        "published": len(pub),
        "availability": _round(len(pub) / len(rows) if rows else None, 3),
        "pos_rmse_m": _round(_rmse(pos), 1),
        "pos_median_m": _round(statistics.median(pos) if pos else None, 1),
        "pos_p95_m": _round(_p95(pos), 1),
        "vel_rmse_ms": _round(_rmse(vel)),
        "vel_median_ms": _round(statistics.median(vel) if vel else None),
        "pos_rmse_published_m": _round(_rmse([r["pos"] for r in pub]), 1),
        "vel_rmse_published_ms": _round(_rmse([r["vel"] for r in pub if r["vel"] is not None])),
        "rejected_rms_delay": sum(1 for r in rows if r["reason"] == "rejected_rms_delay"),
        "rejected_rms_doppler": sum(1 for r in rows if r["reason"] == "rejected_rms_doppler"),
        "end_offset_m": _round(max((r["offset"] for r in rows if r["t_s"] == t_end), default=0.0), 1),
        "pos_err_last30s_m": _round(statistics.fmean(last) if last else None, 1),
    }


def mean_row(rows: list[dict]) -> dict:
    out = {k: rows[0][k] for k in ("case", "speed", "rx_doppler", "algorithm")}
    out["seed"] = "mean"
    for k in SUMMARY_FIELDS[5:]:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = round(statistics.fmean(vals), 3) if vals else None
    return out


def _mean_band(series: list[list[float | None]]) -> tuple[list, list, list]:
    mean, lo, hi = [], [], []
    for vals in zip(*series):
        v = [x for x in vals if x is not None]
        mean.append(statistics.fmean(v) if v else None)
        lo.append(min(v) if v else None)
        hi.append(max(v) if v else None)
    return mean, lo, hi


def _nan(xs):
    return [math.nan if x is None else x for x in xs]


def plot_rmse_vs_time(path, title, curves, window_s, t_max):
    """curves: list of (label, colour, dashed, {seed: rows})."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, facecolor=SURFACE)
    t = list(range(1, t_max + 1))
    for ax, (key, label) in zip(
        axes, (("pos", "Horizontal position RMSE (m)"), ("vel", "Horizontal velocity RMSE (m/s)"))
    ):
        _style(ax)
        for name, colour, dashed, per_seed in curves:
            series = [windowed([r for r in rows if r["solved"]], key, window_s, t_max) for rows in per_seed.values()]
            mean, lo, hi = _mean_band(series)
            if not dashed:
                ax.fill_between(t, _nan(lo), _nan(hi), color=colour, alpha=0.12, linewidth=0)
            ax.plot(
                t,
                _nan(mean),
                color=colour,
                linewidth=2.0,
                linestyle=(0, (5, 3)) if dashed else "-",
                zorder=4 if dashed else 3,
                label=name,
            )
        ax.set_ylabel(label, color=INK, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK_2)
    axes[-1].set_xlabel("Time since start of run (s)", color=INK, fontsize=10)
    axes[-1].set_xlim(0, t_max)
    fig.suptitle(title, color=INK, fontsize=12, x=0.06, y=0.975, ha="left")
    fig.text(
        0.06,
        0.935,
        f"Mean over the 5 seeds of a {window_s:g} s sliding-window RMSE; shading spans the seeds. "
        "Every solve counts, including ones the gates would reject.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.08, hspace=0.12)
    _save(fig, path)


def plot_error_vs_offset(path, title, panels, by_bearing: bool = False):
    """panels: list of (panel title, {algorithm: [rows across seeds]}).

    ``by_bearing`` puts the direction of the offset on the x axis instead of
    its size: on the circle the size is constant and only the direction moves."""
    xkey = "bearing" if by_bearing else "offset"
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels) + 0.6, 4.8), facecolor=SURFACE, squeeze=False)
    for ax, (ptitle, by_alg) in zip(axes[0], panels):
        _style(ax)
        for alg in ALGORITHMS:
            rows = [r for r in by_alg.get(alg, []) if r["solved"]]
            pub = [r for r in rows if r["published"]]
            rej = [r for r in rows if not r["published"]]
            ax.scatter(
                [r[xkey] for r in pub],
                [r["pos"] for r in pub],
                s=9,
                color=ALG_COLOUR[alg],
                alpha=0.55,
                linewidths=0,
                label=ALG_LABEL[alg],
            )
            if rej:
                ax.scatter(
                    [r[xkey] for r in rej],
                    [r["pos"] for r in rej],
                    s=16,
                    facecolors="none",
                    edgecolors=ALG_COLOUR[alg],
                    linewidths=0.9,
                    label=f"{ALG_LABEL[alg].split(':')[0]}: rejected by gates",
                )
        ax.set_title(ptitle, color=INK, fontsize=10, loc="left")
        if by_bearing:
            ax.set_xlabel("Direction of the receiver offset (deg from north)", color=INK, fontsize=9.5)
            ax.set_xlim(0, 360)
            ax.set_xticks(range(0, 361, 45))
        else:
            ax.set_xlabel("Receiver offset from registered position (m)", color=INK, fontsize=9.5)
            ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK_2)
    axes[0][0].set_ylabel("Horizontal position error (m)", color=INK, fontsize=9.5)
    fig.suptitle(title, color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path)


def plot_overview(path, title, means: dict, baseline: dict, cases: list[dict]):
    """means[(case, doppler, alg)] = mean summary row."""
    moving = [c for c in cases if c["motion"].get("type", "static") != "static"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), facecolor=SURFACE)
    x = list(range(len(moving)))
    w = 0.38
    for row, doppler in enumerate((False, True)):
        for col, (key, label, log) in enumerate(
            (("pos_rmse_m", "Position RMSE (m, log scale)", True), ("vel_rmse_ms", "Velocity RMSE (m/s)", False))
        ):
            ax = axes[row][col]
            _style(ax)
            ax.grid(axis="x", visible=False)
            for k, alg in enumerate(ALGORITHMS):
                vals = [(means.get((c["id"], doppler, alg)) or {}).get(key) for c in moving]
                ax.bar(
                    [i + (k - 0.5) * w for i in x],
                    [v or 0 for v in vals],
                    width=w - 0.04,
                    color=ALG_COLOUR[alg],
                    label=ALG_LABEL[alg],
                )
            if baseline.get(key):
                ax.axhline(
                    baseline[key],
                    color=BASELINE_COLOUR,
                    linestyle=(0, (5, 3)),
                    linewidth=1.6,
                    label=f"static baseline ({baseline[key]:.1f})",
                )
            if log:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            else:
                ax.set_ylim(bottom=0)
            ax.set_xticks(x)
            ax.set_xticklabels([c["id"] for c in moving], rotation=35, ha="right", fontsize=8.5)
            ax.set_ylabel(label, color=INK, fontsize=9.5)
            ax.set_title(
                f"Receiver Doppler {'on (physical)' if doppler else 'off (geometry only)'}",
                color=INK,
                fontsize=10,
                loc="left",
            )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.955),
        ncol=3,
        fontsize=9,
        frameon=False,
        labelcolor=INK_2,
    )
    fig.suptitle(title, color=INK, fontsize=12, x=0.02, ha="left")
    fig.text(
        0.02,
        0.005,
        "Mean over 5 seeds; every solve counts, including ones the gates would reject.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    _save(fig, path)


def plot_receiver_paths(path, cases: list[dict], duration_s: float):
    from retina_simulation.motion import motion_from

    groups = [("walk", "Walking, 1.5 m/s"), ("vehicle", "Vehicle, 15 m/s")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor=SURFACE)
    for ax, (speed, label) in zip(axes, groups):
        _style(ax)
        for c in cases:
            if c.get("speed") != speed:
                continue
            m = motion_from(c["motion"])
            pts = [m.state(t)[:2] for t in [i * duration_s / 300 for i in range(301)]]
            profile = c["id"].rsplit("-", 1)[0]
            colour = PROFILE_COLOUR.get(profile, INK)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colour, linewidth=2.2, label=profile)
            ax.scatter([pts[0][0]], [pts[0][1]], s=36, color=colour, edgecolors=SURFACE, linewidths=1.2, zorder=5)
            j = len(pts) // 3
            ax.annotate(
                "",
                xy=pts[j + 1],
                xytext=pts[j],
                arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 2, "mutation_scale": 15},
            )
        ax.scatter([0], [0], marker="s", s=60, color="#e34948", edgecolors=SURFACE, zorder=6)
        ax.annotate(
            "registered position (UBC)", (0, 0), xytext=(8, -14), textcoords="offset points", fontsize=8.5, color=INK
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(label, color=INK, fontsize=10.5, loc="left")
        ax.set_xlabel("East (m)", color=INK_2, fontsize=9)
        ax.set_ylabel("North (m)", color=INK_2, fontsize=9)
        ax.legend(loc="best", fontsize=8.5, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK_2)
    fig.suptitle(
        "Receiver platform trajectories over the 300 s run (dots: start)", color=INK, fontsize=12, x=0.02, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path)


def build(out: str, exp: dict) -> list[dict]:
    window_s = exp.get("analysis", {}).get("rmse_window_s", 15)
    t_max = int(round(exp["duration_s"] / exp.get("tick_s", 1.0)))
    seeds = [r["seed"] for r in exp["runs"]]
    algorithms = exp.get("algorithms", list(ALGORITHMS))
    summary, means = [], {}
    data: dict[tuple, dict[int, list[dict]]] = {}
    for case in exp["cases"]:
        for doppler in dopplers_for(exp, case):
            for alg in algorithms:
                per_seed = {s: load(seed_dir(out, case["id"], doppler, s), alg) for s in seeds}
                data[(case["id"], doppler, alg)] = per_seed
                rows = [summary_row(case, doppler, alg, s, per_seed[s]) for s in seeds]
                m = mean_row(rows)
                means[(case["id"], doppler, alg)] = m
                summary += rows + [m]

    static = next(c for c in exp["cases"] if c["motion"].get("type", "static") == "static")
    baseline_rows = data[(static["id"], False, "stationary")]
    baseline = means[(static["id"], False, "stationary")]

    for case in exp["cases"]:
        if case is static:
            continue
        case_dir = os.path.join(out, "cases", case["id"])
        panels = []
        for doppler in dopplers_for(exp, case):
            curves = [("static baseline", BASELINE_COLOUR, True, baseline_rows)]
            curves += [(ALG_LABEL[a], ALG_COLOUR[a], False, data[(case["id"], doppler, a)]) for a in algorithms]
            tag = "receiver Doppler on" if doppler else "receiver Doppler off"
            plot_rmse_vs_time(
                os.path.join(case_dir, f"rmse_vs_time_{doppler_dir(doppler)}.png"),
                f"{case['id']}, {tag}",
                curves,
                window_s,
                t_max,
            )
            panels.append(
                (tag, {a: [r for rows in data[(case["id"], doppler, a)].values() for r in rows] for a in algorithms})
            )
        circle = case["motion"].get("type") == "circle"
        what = "direction of the receiver offset" if circle else "receiver offset"
        plot_error_vs_offset(
            os.path.join(case_dir, "error_vs_offset.png"),
            f"{case['id']}: position error against {what}",
            panels,
            by_bearing=circle,
        )

    plot_overview(
        os.path.join(out, "overview.png"),
        f"{exp['name']}: S vs G for every receiver motion",
        means,
        baseline,
        exp["cases"],
    )
    plot_receiver_paths(os.path.join(out, "receiver_paths.png"), exp["cases"], exp["duration_s"])
    first = seed_dir(out, static["id"], False, seeds[0])
    nodes_path = os.path.join(first, "nodes.json")
    with open(nodes_path, encoding="utf-8") as f:
        nodes = json.load(f)
    paths = [(r, _truth_path(seed_dir(out, static["id"], False, r["seed"]))) for r in exp["runs"]]
    plot_map(os.path.join(out, "paths_map.png"), f"{exp['name']}: flight paths", nodes, paths)

    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary)
    return summary
