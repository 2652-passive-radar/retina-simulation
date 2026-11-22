"""Tables and figures of the direction-finding study (doa_experiment.py).

Written into the run folder:

``summary.csv``         system A, system B nominal and its variants: position
                        RMSE (90% bootstrap interval), median, 95th
                        percentile, availability, wrong-lobe rate, bearing RMS
``sweep.csv``           system B at every sweep level (each error term alone,
                        and Gaussian bearing noise)
``requirements.csv``    for each error term, the value at which system B
                        alone reaches system A's RMSE and 100 / 200 / 500 m
``paths_map.png``       the flights, towers and the south-west beam
``error_vs_range.png``  each solve's error against range, A and B nominal
``requirement.png``     B's position RMSE against bearing accuracy
``sensitivity.png``     B's position RMSE against each error term alone

Intervals: 90%, from resampling whole runs (seed x realisation for B, seed
for A), the same resampling for every level.
"""

import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import experiment_report as er  # noqa: E402
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
SYSTEM_NAMES = {
    "system_a": "A: multistatic (3 towers)",
    "nominal": "B: direction finding, nominal",
    "nominal_true_alt": "B: nominal, true altitude",
    "perfect": "B: perfect array",
    "perfect_true_alt": "B: perfect array, true altitude",
    "nominal_line": "B: nominal, line array",
    "nominal_triangle": "B: nominal, triangle array",
}


def fmt_m(v) -> str:
    if v is None:
        return "-"
    return f"{v / 1000:.2f} km" if v >= 1000 else f"{v:.0f} m"


def _pool(runs: list[dict], idx=None) -> dict:
    rs = runs if idx is None else [runs[i] for i in idx]
    sse, n = sum(r["sse"] for r in rs), sum(r["n"] for r in rs)
    out = {"pos_rmse_m": math.sqrt(sse / n) if n else None, "availability": n / sum(r["ticks"] for r in rs)}
    nb = sum(r.get("nb", 0) for r in rs)
    if nb:
        out["wrong_lobe_rate"] = sum(r["wrong"] for r in rs) / nb
        out["bearing_rms_deg"] = math.sqrt(sum(r["bse"] for r in rs) / nb)
    return out


def _boot_idx(n_runs: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_runs, (n_boot, n_runs))


def _rmse_boot(runs: list[dict], idx: np.ndarray) -> np.ndarray:
    sse = np.array([r["sse"] for r in runs])[idx].sum(axis=1)
    n = np.array([r["n"] for r in runs])[idx].sum(axis=1)
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
    by_group = {lv["group"]: lv for lv in levels if lv["param"] is None}
    n_runs = len(levels[0]["runs"])
    idx = _boot_idx(n_runs, n_boot, 7)

    # --- summary -----------------------------------------------------------
    summary = []
    a = results.get("system_a")
    a_rmse, a_note = A_STEP0_RMSE_M, "system A not run here: reference is the step 0 baseline RMSE"
    if a:
        a_idx = _boot_idx(len(a["runs"]), n_boot, 11)
        p = _pool(a["runs"])
        errs = np.array([s["err_horiz_m"] for s in a["solves"]])
        lo, hi = _interval(_rmse_boot(a["runs"], a_idx))
        summary.append({"system": SYSTEM_NAMES["system_a"], **p, "ci_lo_m": lo, "ci_hi_m": hi,
                        "pos_median_m": float(np.median(errs)), "pos_p95_m": float(np.percentile(errs, 95))})  # fmt: skip
        a_rmse, a_note = p["pos_rmse_m"], "system A on these flights"
    nominal_errs = np.array([s["err_horiz_m"] for s in by_group["nominal"]["solves"] if "err_horiz_m" in s])
    for g in ("nominal", "nominal_true_alt", "perfect", "perfect_true_alt", "nominal_line", "nominal_triangle"):
        if g not in by_group:
            continue
        lv = by_group[g]
        lo, hi = _interval(_rmse_boot(lv["runs"], idx))
        row = {"system": SYSTEM_NAMES[g], **_pool(lv["runs"]), "ci_lo_m": lo, "ci_hi_m": hi}
        if g == "nominal":
            row.update(pos_median_m=float(np.median(nominal_errs)), pos_p95_m=float(np.percentile(nominal_errs, 95)))
        summary.append(row)
    fields = ["system", "pos_rmse_m", "ci_lo_m", "ci_hi_m", "pos_median_m", "pos_p95_m", "availability",
              "wrong_lobe_rate", "bearing_rms_deg"]  # fmt: skip
    _write_csv(os.path.join(out, "summary.csv"), fields, summary)

    # --- sweeps and requirements --------------------------------------------
    curves: dict[str, dict] = {}
    sweep_rows = []
    for lv in levels:
        if lv["param"] is None:
            continue
        boot = _rmse_boot(lv["runs"], idx)
        p = _pool(lv["runs"])
        c = curves.setdefault(lv["param"], {"values": [], "rmse": [], "boot": [], "bearing": []})
        c["values"].append(lv["value"])
        c["rmse"].append(p["pos_rmse_m"])
        c["boot"].append(boot)
        c["bearing"].append(p.get("bearing_rms_deg"))
        lo, hi = _interval(boot)
        sweep_rows.append({"param": lv["param"], "value": lv["value"], **p, "ci_lo_m": lo, "ci_hi_m": hi})
    _write_csv(
        os.path.join(out, "sweep.csv"),
        ["param", "value", "pos_rmse_m", "ci_lo_m", "ci_hi_m", "availability", "wrong_lobe_rate", "bearing_rms_deg"],
        sweep_rows,
    )
    targets = [("system A", a_rmse)] + [(f"{t} m", float(t)) for t in d["targets_m"]]
    req_rows = []
    for param, c in curves.items():
        boot = np.array(c["boot"])
        nominal = d["errors"].get(param) if param != "bearing_noise_deg" else None
        for name, t in targets:
            val, note = crossing(c["values"], c["rmse"], t, param)
            reps = [crossing(c["values"], list(boot[:, k]), t, param)[0] for k in range(boot.shape[1])]
            finite = np.array([r for r in reps if r is not None])
            lo, hi = _interval(finite) if len(finite) >= 0.9 * len(reps) else (None, None)
            req_rows.append({"param": param, "nominal": nominal, "target": name, "target_m": round(t, 1),
                             "threshold": val, "ci_lo": lo, "ci_hi": hi, "note": note})  # fmt: skip
    _write_csv(
        os.path.join(out, "requirements.csv"),
        ["param", "nominal", "target", "target_m", "threshold", "ci_lo", "ci_hi", "note"],
        req_rows,
    )

    # --- figures -------------------------------------------------------------
    first_seed = os.path.join(out, f"seed-{exp['runs'][0]['seed']}")
    with open(os.path.join(first_seed, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    paths = [(r, er._truth_path(os.path.join(out, f"seed-{r['seed']}"))) for r in exp["runs"]]
    er.plot_map(os.path.join(out, "paths_map.png"), f"{exp['name']}: flight paths", nodes, paths)
    _plot_error_vs_range(os.path.join(out, "error_vs_range.png"), by_group["nominal"]["solves"], a)
    nominal_pool = _pool(by_group["nominal"]["runs"])
    _plot_requirement(os.path.join(out, "requirement.png"), curves["bearing_noise_deg"], a_rmse, a_note,
                      d["targets_m"], nominal_pool)  # fmt: skip
    _plot_sensitivity(os.path.join(out, "sensitivity.png"), curves, d["errors"], a_rmse, a_note)
    return summary


def _write_csv(path: str, fields: list[str], rows: list[dict]) -> None:
    def r(v):
        return round(v, 4) if isinstance(v, float) else v

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows({k: r(v) for k, v in row.items()} for row in rows)


def _binned(ranges: np.ndarray, errs: np.ndarray, edges: np.ndarray) -> tuple[list, list]:
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ranges >= lo) & (ranges < hi)
        if m.sum() >= 20:
            xs.append((lo + hi) / 2)
            ys.append(math.sqrt(float(np.mean(errs[m] ** 2))))
    return xs, ys


def _plot_error_vs_range(path: str, b_solves: list[dict], a: dict | None) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=er.SURFACE)
    er._style(ax)
    edges = np.arange(8, 38, 3.0)
    b = [s for s in b_solves if "err_horiz_m" in s]
    br, be = np.array([s["range_km"] for s in b]), np.array([s["err_horiz_m"] for s in b])
    ax.scatter(br, be, s=4, color=er.SERIES[0], alpha=0.08, linewidths=0, zorder=2)
    xs, ys = _binned(br, be, edges)
    ax.plot(
        xs, ys, color=er.SERIES[0], lw=2.4, marker="o", ms=4, zorder=4, label="B: direction finding (RMSE per 3 km)"
    )
    if a:
        ar, ae = np.array([s["range_km"] for s in a["solves"]]), np.array([s["err_horiz_m"] for s in a["solves"]])
        ax.scatter(ar, ae, s=6, color=er.SERIES[1], alpha=0.35, linewidths=0, zorder=3)
        xs, ys = _binned(ar, ae, edges)
        ax.plot(xs, ys, color=er.SERIES[1], lw=2.4, marker="o", ms=4, zorder=4, label="A: multistatic (RMSE per 3 km)")
    ax.set_yscale("log")
    ax.set_ylim(5, 30000)
    ax.set_xlabel("aircraft distance from the receiver (km)", color=er.INK_2, fontsize=9.5)
    ax.set_ylabel("horizontal position error (m)", color=er.INK_2, fontsize=9.5)
    ax.legend(
        loc="lower right", fontsize=8.5, frameon=True, facecolor=er.SURFACE, edgecolor=er.GRID, labelcolor=er.INK_2
    )
    ax.set_title("Position error against range (dots: each solve; B: all bearing realisations)", color=er.INK,
                 fontsize=11.5, loc="left")  # fmt: skip
    fig.tight_layout()
    er._save(fig, path)


def _plot_requirement(path: str, c: dict, a_rmse: float, a_note: str, targets: list, nominal: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), facecolor=er.SURFACE)
    er._style(ax)
    v = np.array(c["values"])
    boot = np.array(c["boot"])
    lo, hi = np.percentile(boot, 5, axis=1), np.percentile(boot, 95, axis=1)
    keep = v > 0
    ax.fill_between(v[keep], lo[keep], hi[keep], color=er.SERIES[0], alpha=0.15, linewidth=0)
    ax.plot(v[keep], np.array(c["rmse"])[keep], color=er.SERIES[0], lw=2.4, marker="o", ms=4,
            label="B with a perfect bearing plus Gaussian noise")  # fmt: skip
    ax.axhline(a_rmse, color=er.SERIES[1], lw=1.8, ls=(0, (5, 3)), label=f"A: multistatic, {a_rmse:.0f} m ({a_note})")
    for t in targets:
        ax.axhline(t, color=er.MUTED, lw=0.9, ls=":")
        ax.annotate(f"{t} m", (v[keep][0], t), xytext=(2, 3), textcoords="offset points", fontsize=8, color=er.INK_2)
    if nominal.get("bearing_rms_deg"):
        ax.scatter([nominal["bearing_rms_deg"]], [nominal["pos_rmse_m"]], s=60, color=er.SERIES[2], zorder=6,
                   label="B with the nominal array model")  # fmt: skip
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("bearing error, 1σ (°)", color=er.INK_2, fontsize=9.5)
    ax.set_ylabel("system B position RMSE (m)", color=er.INK_2, fontsize=9.5)
    ax.legend(
        loc="upper left", fontsize=8.5, frameon=True, facecolor=er.SURFACE, edgecolor=er.GRID, labelcolor=er.INK_2
    )
    ax.set_title("How accurate must the bearing be?", color=er.INK, fontsize=11.5, loc="left")
    fig.tight_layout()
    er._save(fig, path)


def _plot_sensitivity(path: str, curves: dict, nominal: dict, a_rmse: float, a_note: str) -> None:
    params = [p for p in LABELS if p in curves and p != "bearing_noise_deg"]
    cols = 4
    rows = math.ceil(len(params) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.6 * rows + 0.6), facecolor=er.SURFACE, squeeze=False)
    for ax, param in zip(axes.flat, params):
        er._style(ax)
        c = curves[param]
        v = np.array(c["values"], float)
        boot = np.array(c["boot"])
        ax.fill_between(v, np.percentile(boot, 5, axis=1), np.percentile(boot, 95, axis=1), color=er.SERIES[0],
                        alpha=0.15, linewidth=0)  # fmt: skip
        ax.plot(v, c["rmse"], color=er.SERIES[0], lw=2.2, marker="o", ms=3.5, label="B, this error alone")
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
    axes.flat[0].legend(loc="upper left", fontsize=8, frameon=True, facecolor=er.SURFACE, edgecolor=er.GRID,
                        labelcolor=er.INK_2)  # fmt: skip
    fig.suptitle("System B: position RMSE with one array error at a time (all others off, altitude assumed 2 km)",
                 x=0.01, ha="left", color=er.INK, fontsize=12.5)  # fmt: skip
    fig.text(0.01, 0.005, f"Bands: 90% interval over runs. A reference: {a_note}.", color=er.INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    er._save(fig, path)
