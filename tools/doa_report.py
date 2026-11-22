"""Tables and figures of the direction-finding study (doa_experiment.py).

Written into the run folder:

``summary.csv``         system A, and system B nominal and its variants, each
                        second and with each real-time averaging window:
                        position RMSE (90% bootstrap interval), median, 95th
                        percentile, availability, wrong-lobe rate, bearing error
``sweep.csv``           system B at every sweep level and averaging window
``requirements.csv``    for each error term and averaging window, the value at
                        which system B (that term alone) reaches system A's
                        RMSE and each of ``doa.targets_m``
``paths_map.png``       the flights, towers and the beam
``error_vs_range.png``  error against distance: A, B each second, B averaged
``requirement.png``     B's position RMSE against bearing accuracy
``sensitivity.png``     B's position RMSE against each error term alone
``bearing_error.png``   how B's bearings are distributed (nominal array)

Intervals: 90%, from resampling whole runs (seed x realisation for B, seed
for A), the same resampling for every level.
"""

import csv
import json
import math
import os

import experiment_report as er
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

A_STEP0_RMSE_M = 81.8  # step 0 baseline (148 deg beam, other flights): used only when A was not run
LABELS = {
    "bearing_noise_deg": "bearing error, 1σ (°)",
    "snr_offset_db": "SNR offset (dB), thermal noise only",
    "phase_fixed_deg": "fixed phase error per element, 1σ (°)",
    "device_offset_deg": "RSPduo B phase offset, 1σ (°)",
    "phase_random_deg": "random phase error per detection, 1σ (°)",
    "coupling_db": "coupling between elements (dB)",
    "multipath_db": "multipath level in the cell (dB)",
    "heading_deg": "heading error, 1σ (°)",
    "tilt_deg": "tilt, 1σ (°)",
}
HIGHER_IS_BETTER = {"snr_offset_db"}
GROUP_NAMES = {
    "nominal": "B: nominal",
    "calibrated": "B: fixed errors calibrated out",
    "nominal_true_alt": "B: nominal, true altitude",
    "perfect": "B: perfect array",
    "perfect_true_alt": "B: perfect array, true altitude",
    "nominal_line": "B: nominal, line array",
    "nominal_triangle": "B: nominal, triangle array",
}
WINDOW_COLOURS = [er.SERIES[0], er.SERIES[3], er.SERIES[2], er.SERIES[4]]


def fmt_m(v) -> str:
    if v is None:
        return "-"
    return f"{v / 1000:.2f} km" if v >= 1000 else f"{v:.0f} m"


def window_name(w) -> str:
    return "each second" if int(w) == 0 else f"real time, {int(w)} s"


def _pool(runs: list[dict], w: str = "0") -> dict:
    sse, n = sum(r["win"][w]["sse"] for r in runs), sum(r["win"][w]["n"] for r in runs)
    out = {"pos_rmse_m": math.sqrt(sse / n) if n else None, "availability": n / sum(r["ticks"] for r in runs)}
    nb = sum(r.get("nb", 0) for r in runs)
    if nb:
        out["wrong_lobe_rate"] = sum(r["wrong"] for r in runs) / nb
        out["bearing_rms_deg"] = math.sqrt(sum(r["bse"] for r in runs) / nb)
    return out


def _boot_idx(n_runs: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_runs, (n_boot, n_runs))


def _rmse_boot(runs: list[dict], idx: np.ndarray, w: str = "0") -> np.ndarray:
    sse = np.array([r["win"][w]["sse"] for r in runs])[idx].sum(axis=1)
    n = np.array([r["win"][w]["n"] for r in runs])[idx].sum(axis=1)
    return np.sqrt(sse / np.maximum(n, 1))


def _interval(samples: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(samples, 5)), float(np.percentile(samples, 95))


def crossing(values: list[float], rmse: list[float], target: float, param: str):
    """(value, note): where the RMSE curve first reaches ``target`` walking
    from the best tested value; None with a note when it never does or the
    best value is already above it."""
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=param in HIGHER_IS_BETTER)
    v, y = [values[i] for i in order], [rmse[i] for i in order]
    if y[0] >= target:
        return None, "above target even at the best tested value"
    for i in range(1, len(v)):
        if y[i] >= target:
            a, b = v[i - 1], v[i]
            f = (math.log(target) - math.log(y[i - 1])) / (math.log(y[i]) - math.log(y[i - 1]))
            if a > 0 and b > 0 and not param.endswith("_db"):
                return math.exp(math.log(a) + f * (math.log(b) - math.log(a))), ""
            return a + f * (b - a), ""
    return None, "below target over the whole tested range"


def build(out: str, exp: dict, results: dict) -> list[dict]:
    d = exp["doa"]
    n_boot = d.get("bootstrap", 1000)
    levels = results["levels"]
    wins = [str(w) for w in [0, *d.get("tracking_windows_s", [])]]
    by_group = {lv["group"]: lv for lv in levels if lv["param"] is None}
    idx = _boot_idx(len(levels[0]["runs"]), n_boot, 7)

    # --- summary -----------------------------------------------------------
    summary = []
    a = results.get("system_a")
    a_rmse, a_note = A_STEP0_RMSE_M, "system A not run here: reference is the step 0 baseline RMSE"
    if a:
        errs = np.array([s["err_horiz_m"] for s in a["solves"]])
        lo, hi = _interval(_rmse_boot(a["runs"], _boot_idx(len(a["runs"]), n_boot, 11)))
        p = _pool(a["runs"])
        summary.append({"system": "A: multistatic (3 towers)", "averaging": "each second", **p, "ci_lo_m": lo,
                        "ci_hi_m": hi, "pos_median_m": float(np.median(errs)),
                        "pos_p95_m": float(np.percentile(errs, 95))})  # fmt: skip
        a_rmse, a_note = p["pos_rmse_m"], "system A on these flights"
    for g, name in GROUP_NAMES.items():
        if g not in by_group:
            continue
        lv = by_group[g]
        for w in wins:
            lo, hi = _interval(_rmse_boot(lv["runs"], idx, w))
            row = {"system": name, "averaging": window_name(w), **_pool(lv["runs"], w), "ci_lo_m": lo, "ci_hi_m": hi}
            if g == "nominal":
                errs = np.array([s["err_horiz_m"] for s in lv["solves"][w]])
                row.update(pos_median_m=float(np.median(errs)), pos_p95_m=float(np.percentile(errs, 95)))
                if w == "0":
                    be = np.abs([s["bearing_err_deg"] for s in lv["solves"]["0"]])
                    row.update(
                        bearing_median_abs_deg=float(np.median(be)), bearing_p95_abs_deg=float(np.percentile(be, 95))
                    )
            if w != "0":
                row.pop("bearing_rms_deg", None)
                row.pop("wrong_lobe_rate", None)
            summary.append(row)
    fields = ["system", "averaging", "pos_rmse_m", "ci_lo_m", "ci_hi_m", "pos_median_m", "pos_p95_m", "availability",
              "wrong_lobe_rate", "bearing_rms_deg", "bearing_median_abs_deg", "bearing_p95_abs_deg"]  # fmt: skip
    _write_csv(os.path.join(out, "summary.csv"), fields, summary)

    # --- sweeps and requirements --------------------------------------------
    curves: dict[tuple, dict] = {}
    sweep_rows = []
    for lv in levels:
        if lv["param"] is None:
            continue
        for w in wins:
            boot = _rmse_boot(lv["runs"], idx, w)
            p = _pool(lv["runs"], w)
            c = curves.setdefault((lv["param"], w), {"values": [], "rmse": [], "boot": []})
            c["values"].append(lv["value"])
            c["rmse"].append(p["pos_rmse_m"])
            c["boot"].append(boot)
            lo, hi = _interval(boot)
            sweep_rows.append({"param": lv["param"], "value": lv["value"], "averaging": window_name(w), **p,
                               "ci_lo_m": lo, "ci_hi_m": hi})  # fmt: skip
    _write_csv(
        os.path.join(out, "sweep.csv"),
        ["param", "value", "averaging", "pos_rmse_m", "ci_lo_m", "ci_hi_m", "availability", "wrong_lobe_rate",
         "bearing_rms_deg"],
        sweep_rows,
    )  # fmt: skip
    targets = [("system A", a_rmse)] + [(f"{t} m", float(t)) for t in d["targets_m"]]
    req_rows = []
    for (param, w), c in curves.items():
        boot = np.array(c["boot"])
        nominal = d["errors"].get(param) if param != "bearing_noise_deg" else None
        for name, t in targets:
            val, note = crossing(c["values"], c["rmse"], t, param)
            reps = [crossing(c["values"], list(boot[:, k]), t, param)[0] for k in range(boot.shape[1])]
            finite = np.array([r for r in reps if r is not None])
            lo, hi = _interval(finite) if len(finite) >= 0.9 * len(reps) else (None, None)
            req_rows.append({"param": param, "averaging": window_name(w), "nominal": nominal, "target": name,
                             "target_m": round(t, 1), "threshold": val, "ci_lo": lo, "ci_hi": hi,
                             "note": note})  # fmt: skip
    _write_csv(
        os.path.join(out, "requirements.csv"),
        ["param", "averaging", "nominal", "target", "target_m", "threshold", "ci_lo", "ci_hi", "note"],
        req_rows,
    )

    # --- figures -------------------------------------------------------------
    with open(os.path.join(out, f"seed-{exp['runs'][0]['seed']}", "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    paths = [(r, er._truth_path(os.path.join(out, f"seed-{r['seed']}"))) for r in exp["runs"]]
    er.plot_map(os.path.join(out, "paths_map.png"), f"{exp['name']}: flight paths", nodes, paths)
    nominal = by_group["nominal"]
    _plot_error_vs_range(os.path.join(out, "error_vs_range.png"), nominal["solves"], wins, a)
    _plot_requirement(os.path.join(out, "requirement.png"), curves, wins, a_rmse, a_note, d["targets_m"],
                      [_pool(nominal["runs"], w) for w in wins])  # fmt: skip
    _plot_sensitivity(os.path.join(out, "sensitivity.png"), curves, wins, d["errors"], a_rmse, a_note)
    _plot_bearing(os.path.join(out, "bearing_error.png"), nominal["solves"]["0"])
    return summary


def _write_csv(path: str, fields: list[str], rows: list[dict]) -> None:
    def r(v):
        return round(v, 4) if isinstance(v, float) else v

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows({k: r(v) for k, v in row.items()} for row in rows)


def _edges(ranges: np.ndarray) -> np.ndarray:
    step = 1.0 if ranges.max() < 15 else 3.0
    return np.arange(math.floor(ranges.min()), math.ceil(ranges.max()) + step, step)


def _binned(ranges: np.ndarray, errs: np.ndarray, edges: np.ndarray, min_n: int = 20) -> tuple[list, list]:
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ranges >= lo) & (ranges < hi)
        if m.sum() >= min_n:
            xs.append((lo + hi) / 2)
            ys.append(math.sqrt(float(np.mean(errs[m] ** 2))))
    return xs, ys


def _plot_error_vs_range(path: str, b_solves: dict, wins: list[str], a: dict | None) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=er.SURFACE)
    er._style(ax)
    b0 = b_solves["0"]
    br, be = np.array([s["range_km"] for s in b0]), np.array([s["err_horiz_m"] for s in b0])
    edges = _edges(br)
    ax.scatter(br, be, s=4, color=er.SERIES[0], alpha=0.06, linewidths=0, zorder=2)
    for k, w in enumerate(wins):
        s = b_solves[w]
        xs, ys = _binned(np.array([x["range_km"] for x in s]), np.array([x["err_horiz_m"] for x in s]), edges)
        ax.plot(xs, ys, color=WINDOW_COLOURS[k], lw=2.4, marker="o", ms=4, zorder=4, label=f"B, {window_name(w)}")
    if a:
        ar, ae = np.array([s["range_km"] for s in a["solves"]]), np.array([s["err_horiz_m"] for s in a["solves"]])
        ax.scatter(ar, ae, s=7, color=er.SERIES[1], alpha=0.4, linewidths=0, zorder=3)
        xs, ys = _binned(ar, ae, edges, min_n=10)
        ax.plot(xs, ys, color=er.SERIES[1], lw=2.4, marker="o", ms=4, zorder=4, label="A: multistatic, each second")
    ax.set_yscale("log")
    ax.set_xlabel("aircraft distance from the receiver (km)", color=er.INK_2, fontsize=9.5)
    ax.set_ylabel("horizontal position error (m)", color=er.INK_2, fontsize=9.5)
    ax.legend(
        loc="lower right", fontsize=8.5, frameon=True, facecolor=er.SURFACE, edgecolor=er.GRID, labelcolor=er.INK_2
    )
    ax.set_title("Position error against distance (lines: RMSE per distance bin; dots: B's single solves)",
                 color=er.INK, fontsize=11, loc="left")  # fmt: skip
    fig.tight_layout()
    er._save(fig, path)


def _plot_requirement(path, curves, wins, a_rmse, a_note, targets, nominal_pools) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.4), facecolor=er.SURFACE)
    er._style(ax)
    for k, w in enumerate(wins):
        c = curves[("bearing_noise_deg", w)]
        v, y = np.array(c["values"]), np.array(c["rmse"])
        boot = np.array(c["boot"])
        keep = v > 0
        ax.fill_between(v[keep], np.percentile(boot, 5, axis=1)[keep], np.percentile(boot, 95, axis=1)[keep],
                        color=WINDOW_COLOURS[k], alpha=0.12, linewidth=0)  # fmt: skip
        ax.plot(v[keep], y[keep], color=WINDOW_COLOURS[k], lw=2.2, marker="o", ms=3.5,
                label=f"B, true bearing + Gaussian error, {window_name(w)}")  # fmt: skip
        ax.scatter([nominal_pools[0]["bearing_rms_deg"]], [nominal_pools[k]["pos_rmse_m"]], s=55, color=WINDOW_COLOURS[k], edgecolors=er.INK, linewidths=0.8,
                       zorder=6, label=f"B with the nominal array, {window_name(w)}")  # fmt: skip
    ax.axhline(a_rmse, color=er.SERIES[1], lw=1.8, ls=(0, (5, 3)), label=f"A: multistatic, {a_rmse:.0f} m ({a_note})")
    for t in targets:
        ax.axhline(t, color=er.MUTED, lw=0.9, ls=":")
        ax.annotate(f"{t} m", (0.06, t), xytext=(2, 3), textcoords="offset points", fontsize=8, color=er.INK_2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bearing error per detection, 1σ (°)", color=er.INK_2, fontsize=9.5)
    ax.set_ylabel("system B position RMSE (m)", color=er.INK_2, fontsize=9.5)
    ax.legend(
        loc="lower right", fontsize=7.8, frameon=True, facecolor=er.SURFACE, edgecolor=er.GRID, labelcolor=er.INK_2
    )
    ax.set_title("How accurate must the bearing be?", color=er.INK, fontsize=11.5, loc="left")
    fig.tight_layout()
    er._save(fig, path)


def _plot_sensitivity(path, curves, wins, nominal, a_rmse, a_note) -> None:
    params = [p for p in LABELS if (p, "0") in curves and p != "bearing_noise_deg"]
    cols = 4
    rows = math.ceil(len(params) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.6 * rows + 0.6), facecolor=er.SURFACE, squeeze=False)
    for ax, param in zip(axes.flat, params):
        er._style(ax)
        for k, w in enumerate(wins):
            c = curves[(param, w)]
            v, boot = np.array(c["values"], float), np.array(c["boot"])
            ax.fill_between(v, np.percentile(boot, 5, axis=1), np.percentile(boot, 95, axis=1),
                            color=WINDOW_COLOURS[k], alpha=0.12, linewidth=0)  # fmt: skip
            ax.plot(v, c["rmse"], color=WINDOW_COLOURS[k], lw=2.0, marker="o", ms=3, label=f"B, {window_name(w)}")
        ax.axhline(a_rmse, color=er.SERIES[1], lw=1.6, ls=(0, (5, 3)), label=f"A: {a_rmse:.0f} m")
        if nominal.get(param) is not None:
            ax.axvline(nominal[param], color=er.MUTED, lw=1.0, ls=":", label="nominal value")
        if not param.endswith("_db"):
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(LABELS[param], color=er.INK_2, fontsize=9)
        ax.set_ylabel("position RMSE (m)", color=er.INK_2, fontsize=9)
    for ax in list(axes.flat)[len(params) :]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper left", fontsize=7.5, frameon=True, facecolor=er.SURFACE, edgecolor=er.GRID,
                        labelcolor=er.INK_2)  # fmt: skip
    fig.suptitle("System B: position RMSE with one array error at a time (all others off)", x=0.01, ha="left",
                 color=er.INK, fontsize=12.5)  # fmt: skip
    fig.text(0.01, 0.005, f"Bands: 90% interval over runs. A reference: {a_note}.", color=er.INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    er._save(fig, path)


def _plot_bearing(path: str, solves: list[dict]) -> None:
    be = np.array([s["bearing_err_deg"] for s in solves])
    fig, ax = plt.subplots(figsize=(8.5, 4.4), facecolor=er.SURFACE)
    er._style(ax)
    lim = max(10.0, float(np.percentile(np.abs(be), 99)))
    ax.hist(np.clip(be, -lim, lim), bins=np.linspace(-lim, lim, 81), color=er.SERIES[0], alpha=0.8)
    med, p95 = float(np.median(np.abs(be))), float(np.percentile(np.abs(be), 95))
    for x in (-p95, -med, med, p95):
        ax.axvline(x, color=er.INK_2, lw=1.0, ls=":" if abs(x) == p95 else "--")
    ax.set_xlabel("bearing error per detection, nominal array (°)", color=er.INK_2, fontsize=9.5)
    ax.set_ylabel("detections", color=er.INK_2, fontsize=9.5)
    ax.set_title(f"Bearing error: half within ±{med:.1f}°, 95% within ±{p95:.1f}° (dashed, dotted)", color=er.INK,
                 fontsize=11.5, loc="left")  # fmt: skip
    fig.tight_layout()
    er._save(fig, path)
