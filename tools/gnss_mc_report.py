"""Sensitivity table and figures for step 3d (experiments/receiver_gnss_mc.json).

Reads, under one output folder, each recording's truth-fed GD solve and its
per-level KF-fed solves (gnss_mc.py), plus ``screening.csv``, and writes:

``levels.csv``             per level and case (and over all cases): the paired %
                           increase of aircraft position and velocity RMSE over
                           the truth-fed solve, with a 90 % interval, the
                           receiver error GD was given, and the filter's
                           consistency (stage A)
``sensitivity_table.csv``  per sweep, direction (the parameter raised or
                           lowered from nominal), output (position, velocity)
                           and threshold: the value where the % increase first
                           crosses the threshold, pooled over the cases and in
                           the case that crosses first, with 90 % intervals
``sensitivity_table.md``   the same, readable
``sensitivity_curves.png`` % increase against each swept parameter
``consistency.png``        stage A: the filter's +-2 sigma coverage per level
``grid.png``               the two-key grid (stage C)

The % increase of a set of solves is 100 (sqrt(sum e_kf^2 / sum e_truth^2) - 1),
each KF-fed solve paired with the truth-fed solve of the same recording and
second, after the warm-up.  Intervals resample whole runs (case, seed and
realisation), stratified by case, with the same draw for every level, so
neighbouring levels keep their common GNSS draws.  Thresholds are interpolated
linearly in log(value) between levels (linearly in value where a sweep
includes zero); a crossing is searched for outward from the nominal value.
"""

import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import gnss_mc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from experiment_report import INK, INK_2, SURFACE, _save, _style  # noqa: E402
from gnss_report import load  # noqa: E402

POOLED = "all cases"
OUTPUTS = (("vel", "aircraft velocity"), ("pos", "aircraft position"))
COLOUR = {"vel": "#2a78d6", "pos": "#c2701b"}
N_BOOT = 1000
CI = (5, 95)


def run_sums(est_path: str, truth: dict, t0: float) -> np.ndarray | None:
    """[sum e_kf^2, sum e_truth^2] for position then velocity, plus the
    receiver error sums and the count, over the paired solves of one run."""
    if not os.path.exists(est_path):
        return None
    s = np.zeros(7)
    for x in load(est_path):
        tr = truth.get(x["t_s"])
        if x["t_s"] < t0 or tr is None or x["vel"] is None or tr["vel"] is None:
            continue
        s += [x["pos"] ** 2, tr["pos"] ** 2, x["vel"] ** 2, tr["vel"] ** 2, (x["rx_pos"] or 0.0) ** 2,
              (x["rx_vel"] or 0.0) ** 2, 1]  # fmt: skip
    return s


def collect(out: str, exp: dict) -> tuple[dict, dict]:
    """{level: {case: array (runs, 7)}} for the levels whose every run is on disk,
    and {case: [run keys]} in a fixed order."""
    found, _, _ = gnss_mc.levels(exp)
    t0 = exp.get("steady_state_from_s", 0.0)
    nreal = exp.get("realisations", 1)
    keys = {}
    data = {lid: {} for lid in found}
    for path, case, seed in gnss_mc.recordings(exp, out):
        tp = os.path.join(path, "analysis", gnss_mc.solve_file("truth", None))
        if not os.path.exists(tp):
            continue
        truth = {x["t_s"]: x for x in load(tp)}
        for r in range(nreal):
            keys.setdefault(case["id"], []).append((seed, r))
            for lid in found:
                s = run_sums(os.path.join(path, "analysis", gnss_mc.solve_file(lid, r)), truth, t0)
                data[lid].setdefault(case["id"], []).append(s)
    complete = {}
    for lid, by_case in data.items():
        if by_case and all(s is not None for runs in by_case.values() for s in runs):
            complete[lid] = {c: np.array(runs) for c, runs in by_case.items()}
    return complete, keys


def increase(sums: np.ndarray, output: str) -> float:
    """% increase over truth-fed from summed runs (rows) or one summed row."""
    s = sums.sum(axis=0) if sums.ndim == 2 else sums
    a, b = (0, 1) if output == "pos" else (2, 3)
    return 100.0 * (math.sqrt(s[a] / s[b]) - 1.0) if s[b] > 0 else float("nan")


def boot_indices(keys: dict, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {c: rng.integers(0, len(k), size=(n_boot, len(k))) for c, k in keys.items()}


def boot_increase(by_case: dict, idx: dict, output: str, case: str | None = None) -> np.ndarray:
    """The bootstrap distribution of one level's % increase, in one case or pooled."""
    cases = [case] if case else list(by_case)
    tot = sum(by_case[c][idx[c]].sum(axis=1) for c in cases)
    a, b = (0, 1) if output == "pos" else (2, 3)
    return 100.0 * (np.sqrt(tot[:, a] / tot[:, b]) - 1.0)


def crossing(xs, ys, nominal: float, level: float, direction: int, log: bool) -> float | None:
    """Walking from the nominal value in ``direction`` (+1 up, -1 down), the
    value where ``ys`` first reaches ``level``; the nominal itself if it is
    already there; None if it never does within the levels."""
    xs, ys = list(xs), list(ys)
    i = xs.index(nominal)
    if ys[i] >= level:
        return nominal
    j = i + direction
    while 0 <= j < len(xs):
        if ys[j] >= level:
            x0, x1, y0, y1 = xs[j - direction], xs[j], ys[j - direction], ys[j]
            f = (level - y0) / (y1 - y0)
            if log:
                return float(math.exp(math.log(x0) + f * (math.log(x1) - math.log(x0))))
            return float(x0 + f * (x1 - x0))
        j += direction
    return None


def _ci(vals: list[float | None]) -> tuple[float | None, float | None, float]:
    """5th and 95th percentiles of bootstrap thresholds, None marking 'never
    crosses' (treated as beyond the tested range), and the share that never cross."""
    miss = sum(v is None for v in vals) / len(vals)
    arr = np.array([math.inf if v is None else v for v in vals])
    lo, hi = np.percentile(arr, CI)
    return (None if math.isinf(lo) else float(lo)), (None if math.isinf(hi) else float(hi)), miss


def _sweep_curve(sweep_pts, complete):
    pts = [(v, lid) for v, lid in sweep_pts if lid in complete]
    return [v for v, _ in pts], [lid for _, lid in pts]


def table(exp: dict, complete: dict, keys: dict) -> list[dict]:
    _, sweeps, _ = gnss_mc.levels(exp)
    idx = boot_indices(keys)
    thresholds = exp.get("thresholds_pct", [5, 10, 20])
    cases = list(keys)
    rows = []
    for sw in exp.get("sweeps", []):
        nominal = gnss_mc.nominal_value(exp, sw["key"])
        xs, lids = _sweep_curve(sweeps[sw["id"]], complete)
        if nominal not in xs or len(xs) < 2:
            continue
        log = min(xs) > 0
        for direction, name in ((+1, "raised"), (-1, "lowered")):
            if not any((x - nominal) * direction > 0 for x in xs):
                continue
            for output, _ in OUTPUTS:
                point = {c: [increase(complete[lid][c], output) for lid in lids] for c in cases}
                point[POOLED] = [increase(np.concatenate(list(complete[lid].values())), output) for lid in lids]
                boot = {c: np.array([boot_increase(complete[lid], idx, output, c) for lid in lids]) for c in cases}
                boot[POOLED] = np.array([boot_increase(complete[lid], idx, output) for lid in lids])
                for thr in thresholds:
                    cross = {c: crossing(xs, point[c], nominal, thr, direction, log) for c in point}
                    dist = {c: abs(v - nominal) if v is not None else math.inf for c, v in cross.items() if c != POOLED}
                    binding = min(dist, key=dist.get)
                    row = {
                        "sweep": sw["id"], "parameter": sw["label"], "unit": sw.get("unit", ""), "nominal": nominal,
                        "direction": name, "output": output, "threshold_pct": thr,
                        "binding_case": binding if math.isfinite(dist[binding]) else "",
                        "tested_to": max(xs) if direction > 0 else min(xs),
                    }  # fmt: skip
                    for tag, c in (("pooled", POOLED), ("worst", binding)):
                        bs = [crossing(xs, boot[c][:, b], nominal, thr, direction, log) for b in range(N_BOOT)]
                        lo, hi, miss = _ci(bs)
                        row.update(
                            {f"{tag}_value": cross[c], f"{tag}_lo": lo, f"{tag}_hi": hi, f"{tag}_never_share": miss}
                        )
                    rows.append(row)
    return rows


TABLE_FIELDS = [
    "sweep", "parameter", "unit", "nominal", "direction", "output", "threshold_pct", "tested_to", "binding_case",
    "worst_value", "worst_lo", "worst_hi", "worst_never_share", "pooled_value", "pooled_lo", "pooled_hi",
    "pooled_never_share",
]  # fmt: skip


def _fmt(v, unit: str = "") -> str:
    if v is None:
        return "—"
    return f"{v:.3g}{(' ' + unit) if unit and unit != '×' else ''}" if unit != "×" else f"×{v:.3g}"


def _cell(row: dict, tag: str) -> str:
    v, lo, hi, unit = row[f"{tag}_value"], row[f"{tag}_lo"], row[f"{tag}_hi"], row["unit"]
    beyond = ("> " if row["direction"] == "raised" else "< ") + _fmt(row["tested_to"], unit)
    if v is None:
        return f"{beyond} (never crosses)"
    if v == row["nominal"]:
        return "at nominal already"
    lo_s = _fmt(lo, unit) if lo is not None else beyond
    hi_s = _fmt(hi, unit) if hi is not None else beyond
    return f"**{_fmt(v, unit)}** [{lo_s} – {hi_s}]"


def write_markdown(path: str, rows: list[dict], exp: dict) -> None:
    thresholds = exp.get("thresholds_pct", [5, 10, 20])
    lines = [
        "# Step 3d sensitivity table",
        "",
        "Value of each parameter at which GD's aircraft RMSE, with the receiver pose from the KF, first rises the given "
        "% above GD with the true trajectory (same recordings and seconds, after the warm-up). Every other parameter "
        "at nominal. **Worst case**: the motion case that crosses first, with its 90% interval; pooled: all cases "
        "together. Intervals resample whole runs.",
        "",
    ]
    for output, name in OUTPUTS:
        lines += [f"## {name.capitalize()}", ""]
        head = "| Parameter | Nominal | Direction | " + " | ".join(f"+{t:g}% (worst case)" for t in thresholds)
        head += f" | Binding case (+{thresholds[len(thresholds) // 2]:g}%) | Pooled (+{thresholds[len(thresholds) // 2]:g}%) |"
        lines += [head, "|" + "---|" * (head.count("|") - 1)]
        mid = thresholds[len(thresholds) // 2]
        groups = {}
        for r in rows:
            if r["output"] == output:
                groups.setdefault((r["sweep"], r["direction"]), {})[r["threshold_pct"]] = r
        for (_, direction), by_t in groups.items():
            r0 = next(iter(by_t.values()))
            cells = [_cell(by_t[t], "worst") if t in by_t else "" for t in thresholds]
            m = by_t.get(mid, r0)
            lines.append(
                f"| {r0['parameter']} | {_fmt(r0['nominal'], r0['unit'])} | {direction} | " + " | ".join(cells)
                + f" | {m['binding_case'] or '—'} | {_cell(m, 'pooled')} |"
            )  # fmt: skip
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def read_screening(path: str) -> dict:
    """{level: {case: [ticks, pos ss, vel ss, pos in 2 sigma, vel in 2 sigma, nis sum, nis n]}}."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            acc = out.setdefault(r["level"], {}).setdefault(r["case"], np.zeros(7))
            nis = r["nis_mean"]
            acc += [float(r["ticks"]), float(r["rx_pos_ss_m2"]), float(r["rx_vel_ss_ms2"]), float(r["pos_in_2sigma"]),
                    float(r["vel_in_2sigma"]), float(nis) if nis else 0.0, 1.0 if nis else 0.0]  # fmt: skip
    return out


def consistency(acc: np.ndarray) -> dict:
    n = acc[0]
    return {
        "kf_rx_pos_rmse_m": math.sqrt(acc[1] / n) if n else None,
        "kf_rx_vel_rmse_ms": math.sqrt(acc[2] / n) if n else None,
        "kf_pos_in_2sigma_pct": 100 * acc[3] / (2 * n) if n else None,
        "kf_vel_in_2sigma_pct": 100 * acc[4] / (2 * n) if n else None,
        "kf_nis_mean": acc[5] / acc[6] if acc[6] else None,
    }


LEVEL_FIELDS = [
    "level", "case", "runs", "solves", "pos_increase_pct", "pos_lo", "pos_hi", "vel_increase_pct", "vel_lo", "vel_hi",
    "rx_pos_rmse_m", "rx_vel_rmse_ms", "kf_rx_pos_rmse_m", "kf_rx_vel_rmse_ms", "kf_pos_in_2sigma_pct",
    "kf_vel_in_2sigma_pct", "kf_nis_mean",
]  # fmt: skip


def level_rows(complete: dict, keys: dict, screening: dict) -> list[dict]:
    idx = boot_indices(keys)
    rows = []
    for lid, by_case in complete.items():
        for c in [*by_case, POOLED]:
            sums = np.concatenate(list(by_case.values())) if c == POOLED else by_case[c]
            tot = sums.sum(axis=0)
            row = {"level": lid, "case": c, "runs": len(sums), "solves": int(tot[6])}
            for output in ("pos", "vel"):
                b = boot_increase(by_case, idx, output, None if c == POOLED else c)
                row[f"{output}_increase_pct"] = increase(sums, output)
                row[f"{output}_lo"], row[f"{output}_hi"] = (float(v) for v in np.percentile(b, CI))
            row["rx_pos_rmse_m"] = math.sqrt(tot[4] / tot[6]) if tot[6] else None
            row["rx_vel_rmse_ms"] = math.sqrt(tot[5] / tot[6]) if tot[6] else None
            scr = screening.get(lid, {})
            acc = sum(scr.values()) if c == POOLED and scr else scr.get(c)
            if acc is not None:
                row.update(consistency(acc))
            rows.append(row)
    return rows


def _xscale(ax, xs):
    pos = [x for x in xs if x > 0]
    if min(xs) > 0:
        ax.set_xscale("log")
    elif pos:
        ax.set_xscale("symlog", linthresh=min(pos))


def plot_curves(path: str, exp: dict, by_level: dict) -> None:
    _, sweeps, _ = gnss_mc.levels(exp)
    sw_list = [s for s in exp.get("sweeps", []) if s["id"] in sweeps]
    cols = 3
    nrows = math.ceil(len(sw_list) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(16, 4.3 * nrows), facecolor=SURFACE, squeeze=False)
    thresholds = exp.get("thresholds_pct", [5, 10, 20])
    for ax, sw in zip(axes.flat, sw_list):
        _style(ax)
        pts = [(v, lid) for v, lid in sweeps[sw["id"]] if (lid, POOLED) in by_level]
        if len(pts) < 2:
            ax.set_visible(False)
            continue
        xs = [v for v, _ in pts]
        for t in thresholds:
            ax.axhline(t, color="#c0392b", linewidth=1.2 if t == 10 else 0.7, linestyle=(0, (5, 3)), alpha=0.8)
        for output, name in OUTPUTS:
            ys = [by_level[(lid, POOLED)][f"{output}_increase_pct"] for _, lid in pts]
            lo = [by_level[(lid, POOLED)][f"{output}_lo"] for _, lid in pts]
            hi = [by_level[(lid, POOLED)][f"{output}_hi"] for _, lid in pts]
            worst = [
                max(r[f"{output}_increase_pct"] for (lv, c), r in by_level.items() if lv == lid and c != POOLED)
                for _, lid in pts
            ]
            ax.fill_between(xs, lo, hi, color=COLOUR[output], alpha=0.15, linewidth=0)
            ax.plot(xs, ys, "-o", color=COLOUR[output], markersize=3.5, linewidth=1.8, label=f"{name}, all cases")
            ax.plot(xs, worst, "--", color=COLOUR[output], linewidth=1.1, label=f"{name}, worst case")
        ax.axvline(gnss_mc.nominal_value(exp, sw["key"]), color=INK_2, linewidth=1, linestyle=":")
        _xscale(ax, xs)
        ax.set_title(sw["label"], color=INK, fontsize=10, loc="left")
        ax.set_xlabel(sw.get("unit", ""), color=INK_2, fontsize=9)
        ax.set_ylabel("RMSE increase over\ntrue trajectory (%)", color=INK_2, fontsize=9)
        top = max(30.0, *(by_level[(lid, POOLED)]["vel_hi"] for _, lid in pts))
        ax.set_ylim(-5, min(top, 60.0))
    for ax in list(axes.flat)[len(sw_list) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=4, fontsize=9, frameon=False, labelcolor=INK_2)
    fig.suptitle("GD's aircraft RMSE increase as each GNSS / KF parameter moves from nominal (dotted line)",
                 color=INK, fontsize=12.5, x=0.01, ha="left")  # fmt: skip
    fig.text(0.01, 0.005, f"Red dashed lines: +{', +'.join(f'{t:g}' for t in thresholds)}% thresholds. Band: 90% "
             "interval of the all-cases curve (runs resampled). Worst case: the highest motion case at each level. "
             "Y axis clipped at 60%.", color=INK_2, fontsize=8.5)  # fmt: skip
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    _save(fig, path)


def plot_consistency(path: str, exp: dict, by_level: dict) -> None:
    _, sweeps, _ = gnss_mc.levels(exp)
    sw_list = [s for s in exp.get("sweeps", []) if s["id"] in sweeps]
    cols = 3
    nrows = math.ceil(len(sw_list) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(16, 3.8 * nrows), facecolor=SURFACE, squeeze=False)
    for ax, sw in zip(axes.flat, sw_list):
        _style(ax)
        pts = [(v, by_level.get((lid, POOLED), {})) for v, lid in sweeps[sw["id"]]]
        pts = [(v, r) for v, r in pts if r.get("kf_vel_in_2sigma_pct") is not None]
        if len(pts) < 2:
            ax.set_visible(False)
            continue
        xs = [v for v, _ in pts]
        ax.axhline(95.4, color=INK_2, linewidth=1, linestyle=(0, (4, 2)))
        for key, name, colour in (("kf_pos_in_2sigma_pct", "position", COLOUR["pos"]),
                                  ("kf_vel_in_2sigma_pct", "velocity", COLOUR["vel"])):  # fmt: skip
            ax.plot(xs, [r[key] for _, r in pts], "-o", color=colour, markersize=3.5, linewidth=1.8, label=name)
        ax.axvline(gnss_mc.nominal_value(exp, sw["key"]), color=INK_2, linewidth=1, linestyle=":")
        _xscale(ax, xs)
        ax.set_ylim(0, 102)
        ax.set_title(sw["label"], color=INK, fontsize=10, loc="left")
        ax.set_xlabel(sw.get("unit", ""), color=INK_2, fontsize=9)
        ax.set_ylabel("Truth within KF ±2σ (%)", color=INK_2, fontsize=9)
    for ax in list(axes.flat)[len(sw_list) :]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, fontsize=9, frameon=False, labelcolor=INK_2)
    fig.suptitle("Is the receiver KF honest about its own error? (dashed: 95.4% expected; dotted: nominal)",
                 color=INK, fontsize=12.5, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, path)


def plot_grid(path: str, exp: dict, by_level: dict) -> None:
    g = exp.get("grid")
    _, _, grid = gnss_mc.levels(exp)
    if not g or not all((lid, POOLED) in by_level for _, _, lid in grid):
        return
    xl, yl = list(g["x_levels"]), list(g["y_levels"])
    panels = (("vel", "all"), ("vel", "worst"), ("pos", "all"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor=SURFACE)
    for ax, (output, which) in zip(axes, panels):
        z = np.full((len(yl), len(xl)), np.nan)
        for x, y, lid in grid:
            if which == "all":
                v = by_level[(lid, POOLED)][f"{output}_increase_pct"]
            else:
                v = max(r[f"{output}_increase_pct"] for (lv, c), r in by_level.items() if lv == lid and c != POOLED)
            z[yl.index(y), xl.index(x)] = v
        ax.imshow(z, origin="lower", cmap="YlOrRd", vmin=0, vmax=30, aspect="auto")
        for i in range(len(yl)):
            for j in range(len(xl)):
                v = z[i, j]
                ax.text(j, i, f"{v:+.1f}%", ha="center", va="center", fontsize=9,
                        color="white" if v > 18 else INK, fontweight="bold" if v >= 10 else "normal")  # fmt: skip
        ax.set_xticks(range(len(xl)), [f"{v:g}" for v in xl])
        ax.set_yticks(range(len(yl)), [f"{v:g}" for v in yl])
        ax.set_xlabel(g.get("x_label", g["x"]), color=INK, fontsize=10)
        ax.set_ylabel(g.get("y_label", g["y"]), color=INK, fontsize=10)
        name = dict(OUTPUTS)[output]
        ax.set_title(f"{name}, {'all cases' if which == 'all' else 'worst case'}", color=INK, fontsize=10, loc="left")
        ax.tick_params(colors=INK_2)
    fig.suptitle("Correlated GNSS error: size against time constant (RMSE increase over true trajectory; bold ≥ 10%)",
                 color=INK, fontsize=12.5, x=0.01, ha="left")  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, path)


def _write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def build(out: str, exp: dict) -> list[dict]:
    complete, keys = collect(out, exp)
    if not complete:
        print("No complete level on disk yet")
        return []
    screening = read_screening(os.path.join(out, "screening.csv"))
    lrows = level_rows(complete, keys, screening)
    _write_csv(os.path.join(out, "levels.csv"), lrows, LEVEL_FIELDS)
    by_level = {(r["level"], r["case"]): r for r in lrows}
    rows = table(exp, complete, keys)
    _write_csv(os.path.join(out, "sensitivity_table.csv"), rows, TABLE_FIELDS)
    write_markdown(os.path.join(out, "sensitivity_table.md"), rows, exp)
    plot_curves(os.path.join(out, "sensitivity_curves.png"), exp, by_level)
    plot_consistency(os.path.join(out, "consistency.png"), exp, by_level)
    plot_grid(os.path.join(out, "grid.png"), exp, by_level)
    found, _, _ = gnss_mc.levels(exp)
    print(f"{len(complete)}/{len(found)} levels complete")
    return rows
