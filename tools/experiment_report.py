"""Figures and tables for one experiment run (all its seeds together).

Reads each seed's ``analysis/offline_solves.csv`` (and, for a system run,
``analysis/solves.csv``) under an experiment output folder and writes, next
to the seed folders:

``summary.csv``         one row per seed and source, plus a mean row:
                        availability, and RMSE / mean / median / 95th
                        percentile of horizontal position and velocity error
``rmse_vs_time.csv``    the windowed RMSE series behind the plot
``rmse_vs_time.png``    (offline) horizontal position and velocity RMSE
                        against time, one line per seed
``rmse_vs_time_system.png``  the same for the server's solves, when present
``paths_map.png``       the flights over the UBC site, its illuminators and
                        the surveillance beam

Windowed RMSE at time t is the RMSE of the published solves within
``rmse_window_s / 2`` either side of t; a point with fewer than 3 solves in
its window is left blank rather than drawn from one or two.
"""

import csv
import json
import math
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as PolyPatch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "retina_simulation", "data")

# Categorical slots 1-5 of the reference palette, in fixed order: a seed keeps
# its colour in every figure.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
LAND = "#efede6"
WATER = "#dfe9f3"
MIN_WINDOW_SOLVES = 3

SUMMARY_FIELDS = [
    "source", "seed", "flight", "ticks", "solves_published", "availability",
    "pos_rmse_m", "pos_mean_m", "pos_median_m", "pos_p95_m",
    "vel_rmse_ms", "vel_mean_ms", "vel_median_ms", "vel_p95_ms",
    "adsb_pos_rmse_m", "adsb_vel_rmse_ms",
]  # fmt: skip


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(v) -> float | None:
    return None if v in (None, "") else float(v)


def _rmse(values: list[float]) -> float | None:
    return math.sqrt(statistics.fmean(v * v for v in values)) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def _round(v, nd=2):
    return None if v is None else round(v, nd)


def load_offline(seed_dir: str) -> tuple[list[dict], int]:
    """Published offline solves: t_s, err_horiz_m, verr_horiz_ms, adsb errors."""
    path = os.path.join(seed_dir, "analysis", "offline_solves.csv")
    rows = _read_csv(path)
    return [
        {
            "t_s": float(r["t_s"]),
            "pos": _f(r["err_horiz_m"]),
            "vel": _f(r["verr_horiz_ms"]),
            "adsb_pos": _f(r["adsb_err_horiz_m"]),
            "adsb_vel": _f(r["adsb_verr_horiz_ms"]),
        }
        for r in rows
        if r["published"] == "True"
    ], len(rows)


def load_system(sys_dir: str, ticks: int) -> tuple[list[dict], int] | None:
    """Published server solves from analyze_run's solves.csv (a seed's
    ``system/`` recording), with velocity error added from its true and
    estimated velocity columns, timed from the recording's first tick."""
    path = os.path.join(sys_dir, "analysis", "solves.csv")
    if not os.path.exists(path):
        return None
    with open(os.path.join(sys_dir, "truth.ndjson"), encoding="utf-8") as f:
        t0_ms = json.loads(f.readline())["t_ms"] - 1000
    out = []
    for r in _read_csv(path):
        if r.get("err_horiz_m") in (None, ""):
            continue
        vel = None
        if r.get("est_vel_east_ms") not in (None, "") and r.get("true_vel_east_ms") not in (None, ""):
            vel = math.hypot(
                float(r["est_vel_east_ms"]) - float(r["true_vel_east_ms"]),
                float(r["est_vel_north_ms"]) - float(r["true_vel_north_ms"]),
            )
        out.append({"t_s": (int(float(r["t_ms"])) - t0_ms) / 1000.0, "pos": float(r["err_horiz_m"]), "vel": vel})
    return out, ticks


def windowed(rows: list[dict], key: str, window_s: float, t_max: int) -> list[float | None]:
    half = window_s / 2.0
    series = []
    for t in range(1, t_max + 1):
        vals = [r[key] for r in rows if r[key] is not None and abs(r["t_s"] - t) <= half]
        series.append(_rmse(vals) if len(vals) >= MIN_WINDOW_SOLVES else None)
    return series


def summary_row(source: str, run: dict, rows: list[dict], ticks: int) -> dict:
    pos = [r["pos"] for r in rows if r["pos"] is not None]
    vel = [r["vel"] for r in rows if r.get("vel") is not None]
    adsb_pos = [r["adsb_pos"] for r in rows if r.get("adsb_pos") is not None]
    adsb_vel = [r["adsb_vel"] for r in rows if r.get("adsb_vel") is not None]
    return {
        "source": source,
        "seed": run["seed"],
        "flight": run["flight"],
        "ticks": ticks,
        "solves_published": len(rows),
        "availability": _round(len(rows) / ticks if ticks else None, 3),
        "pos_rmse_m": _round(_rmse(pos), 1),
        "pos_mean_m": _round(statistics.fmean(pos) if pos else None, 1),
        "pos_median_m": _round(statistics.median(pos) if pos else None, 1),
        "pos_p95_m": _round(_p95(pos), 1),
        "vel_rmse_ms": _round(_rmse(vel)),
        "vel_mean_ms": _round(statistics.fmean(vel) if vel else None),
        "vel_median_ms": _round(statistics.median(vel) if vel else None),
        "vel_p95_ms": _round(_p95(vel)),
        "adsb_pos_rmse_m": _round(_rmse(adsb_pos), 1),
        "adsb_vel_rmse_ms": _round(_rmse(adsb_vel)),
    }


def mean_row(source: str, rows: list[dict]) -> dict:
    out = {"source": source, "seed": "mean", "flight": ""}
    for k in SUMMARY_FIELDS[3:]:
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = round(statistics.fmean(vals), 3) if vals else None
    return out


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_rmse(path: str, title: str, per_seed: list[tuple[dict, list[dict], dict]], window_s: float, t_max: int):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, facecolor=SURFACE)
    panels = (
        ("pos", "pos_rmse_m", "Horizontal position RMSE (m)"),
        ("vel", "vel_rmse_ms", "Horizontal velocity RMSE (m/s)"),
    )
    t_axis = list(range(1, t_max + 1))
    for ax, (key, sum_key, label) in zip(axes, panels):
        _style(ax)
        for i, (run, rows, summ) in enumerate(per_seed):
            colour = SERIES[i % len(SERIES)]
            pts = [r for r in rows if r[key] is not None]
            ax.scatter([r["t_s"] for r in pts], [r[key] for r in pts], s=6, color=colour, alpha=0.18, linewidths=0)
            unit = "m" if key == "pos" else "m/s"
            overall = summ[sum_key]
            name = (
                f"seed {run['seed']} ({run['flight']}), overall {overall:.1f} {unit}"
                if overall
                else f"seed {run['seed']}"
            )
            ax.plot(t_axis, windowed(rows, key, window_s, t_max), color=colour, linewidth=2.0, label=name)
        ax.set_ylabel(label, color=INK, fontsize=10)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor=INK_2)
    axes[-1].set_xlabel("Time since start of run (s)", color=INK, fontsize=10)
    axes[-1].set_xlim(0, t_max)
    fig.suptitle(title, color=INK, fontsize=12, x=0.06, y=0.975, ha="left")
    fig.text(
        0.06,
        0.935,
        f"Lines: RMSE over a {window_s:g} s sliding window (blank where fewer than {MIN_WINDOW_SOLVES} solves). "
        "Dots: each solve's error.",
        color=INK_2,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.08, hspace=0.12)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _land_patches(ax, bbox) -> None:
    """Natural Earth land (and lakes) clipped to bbox, drawn behind everything."""
    from shapely.geometry import box, shape

    clip = box(*bbox)

    def draw(path, colour):
        with open(path, encoding="utf-8") as f:
            features = json.load(f)["features"]
        for feat in features:
            geom = shape(feat["geometry"])
            if not geom.intersects(clip):
                continue
            part = geom.intersection(clip)
            for poly in getattr(part, "geoms", [part]):
                if poly.geom_type == "Polygon" and not poly.is_empty:
                    ax.add_patch(
                        PolyPatch(list(poly.exterior.coords), closed=True, facecolor=colour, edgecolor="none", zorder=0)
                    )

    draw(os.path.join(_DATA, "ne_10m_land.geojson"), LAND)
    draw(os.path.join(_DATA, "ne_50m_lakes.geojson"), WATER)


def _beam_wedge(lat, lon, az, width, r_km, n=40):
    pts = [(lon, lat)]
    for k in range(n + 1):
        b = math.radians(az - width / 2 + width * k / n)
        dlat = math.degrees(r_km * math.cos(b) / 6371.0)
        dlon = math.degrees(r_km * math.sin(b) / (6371.0 * math.cos(math.radians(lat))))
        pts.append((lon + dlon, lat + dlat))
    return pts


def plot_map(path: str, title: str, nodes: list[dict], paths: list[tuple[dict, list[tuple[float, float]]]]):
    rx = nodes[0]
    lats = [rx["rx_lat"]] + [n["tx_lat"] for n in nodes] + [p[0] for _, pts in paths for p in pts]
    lons = [rx["rx_lon"]] + [n["tx_lon"] for n in nodes] + [p[1] for _, pts in paths for p in pts]
    pad = 0.08
    bbox = (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)
    aspect = 1.0 / math.cos(math.radians((bbox[1] + bbox[3]) / 2))
    width_in = 8.0
    height_in = width_in * (bbox[3] - bbox[1]) * aspect / (bbox[2] - bbox[0]) + 0.5
    fig, ax = plt.subplots(figsize=(width_in, height_in), facecolor=SURFACE)
    ax.set_facecolor(WATER)
    _land_patches(ax, bbox)
    beam = _beam_wedge(rx["rx_lat"], rx["rx_lon"], rx["beam_azimuth_deg"], rx["beam_width_deg"], 40.0)
    ax.add_patch(
        PolyPatch(
            beam,
            closed=True,
            facecolor="#2a78d6",
            alpha=0.06,
            edgecolor="#2a78d6",
            linewidth=0.8,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
    )
    for n in nodes:
        ax.plot(
            [rx["rx_lon"], n["tx_lon"]],
            [rx["rx_lat"], n["tx_lat"]],
            color=MUTED,
            linewidth=1.0,
            linestyle=(0, (3, 3)),
            zorder=2,
        )
        ax.scatter(
            [n["tx_lon"]], [n["tx_lat"]], marker="^", s=90, color=INK, edgecolors=SURFACE, linewidths=1.5, zorder=5
        )
        ax.annotate(
            f"{n.get('tx_callsign', n['node_id'])}\n{n['fc_hz'] / 1e6:.1f} MHz",
            (n["tx_lon"], n["tx_lat"]),
            xytext=(7, 4),
            textcoords="offset points",
            fontsize=8.5,
            color=INK,
        )
    for i, (run, pts) in enumerate(paths):
        colour = SERIES[i % len(SERIES)]
        la, lo = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(
            lo,
            la,
            color=colour,
            linewidth=2.2,
            zorder=4,
            solid_capstyle="round",
            label=f"seed {run['seed']} ({run['flight']})",
        )
        ax.scatter([lo[0]], [la[0]], s=40, color=colour, edgecolors=SURFACE, linewidths=1.5, zorder=6)
        j = max(1, len(pts) // 2)
        ax.annotate(
            "",
            xy=(lo[j], la[j]),
            xytext=(lo[j - 1], la[j - 1]),
            arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 2, "mutation_scale": 16},
            zorder=6,
        )
        ax.annotate(
            str(run["seed"]),
            (lo[0], la[0]),
            xytext=(-6, 6),
            textcoords="offset points",
            fontsize=8.5,
            color=INK_2,
            ha="right",
        )
    ax.scatter(
        [rx["rx_lon"]], [rx["rx_lat"]], marker="s", s=80, color="#e34948", edgecolors=SURFACE, linewidths=1.5, zorder=7
    )
    ax.annotate(
        "UBC receiver", (rx["rx_lon"], rx["rx_lat"]), xytext=(8, -12), textcoords="offset points", fontsize=9, color=INK
    )
    ax.scatter([-123.1839], [49.1947], marker="x", s=40, color=INK_2, zorder=5)
    ax.annotate("YVR", (-123.1839, 49.1947), xytext=(6, -10), textcoords="offset points", fontsize=8.5, color=INK_2)
    # 10 km scale bar in the lower-left corner.
    y0, x0 = bbox[1] + 0.03, bbox[0] + 0.03
    dlon = math.degrees(10.0 / (6371.0 * math.cos(math.radians(y0))))
    ax.plot([x0, x0 + dlon], [y0, y0], color=INK, linewidth=2.5, zorder=8)
    ax.annotate(
        "10 km", (x0 + dlon / 2, y0), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8.5, color=INK
    )
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_aspect(aspect)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_xlabel("Longitude (deg)", color=INK_2, fontsize=9)
    ax.set_ylabel("Latitude (deg)", color=INK_2, fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower right",
        fontsize=8.5,
        frameon=True,
        facecolor=SURFACE,
        edgecolor=GRID,
        labelcolor=INK_2,
    )
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    fig.text(0.03, 0.012, "Triangles: illuminators. Dashed lines: baselines to UBC. Shaded wedge: surveillance beam\n"
             f"({rx['beam_azimuth_deg']:g} deg, {rx['beam_width_deg']:g} deg wide, drawn to 40 km). "
             "Dots mark each flight's start. Coastline: Natural Earth.",
             color=INK_2, fontsize=7.5)  # fmt: skip
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _truth_path(seed_dir: str) -> list[tuple[float, float]]:
    pts = []
    with open(os.path.join(seed_dir, "truth.ndjson"), encoding="utf-8") as f:
        for line in f:
            tick = json.loads(line)
            if tick["aircraft"]:
                pts.append((tick["aircraft"][0]["lat"], tick["aircraft"][0]["lon"]))
    return pts


def build(out_dir: str, exp: dict, seed_dirs: dict[int, str], system: bool = False) -> list[dict]:
    """Write every figure and table for the experiment; returns the summary rows."""
    window_s = exp.get("analysis", {}).get("rmse_window_s", 15)
    t_max = int(round(exp["duration_s"] / exp.get("tick_s", 1.0)))
    summary, long_rows = [], []
    sources = [("offline", "Offline radar-only solves")]
    if system:
        sources.append(("system", "Server (system) radar-only solves"))
    for source, label in sources:
        per_seed, rows_for_mean = [], []
        for run in exp["runs"]:
            d = seed_dirs[run["seed"]]
            if source == "offline":
                rows, ticks = load_offline(d)
            else:
                loaded = load_system(os.path.join(d, "system"), t_max)
                if loaded is None:
                    continue
                rows, ticks = loaded
            summ = summary_row(source, run, rows, ticks)
            per_seed.append((run, rows, summ))
            rows_for_mean.append(summ)
            pos_w = windowed(rows, "pos", window_s, t_max)
            vel_w = windowed(rows, "vel", window_s, t_max)
            for t, p, v in zip(range(1, t_max + 1), pos_w, vel_w):
                long_rows.append(
                    {"source": source, "seed": run["seed"], "flight": run["flight"], "t_s": t,
                     "pos_rmse_m": _round(p, 1), "vel_rmse_ms": _round(v)}
                )  # fmt: skip
        if not per_seed:
            continue
        summary += rows_for_mean + [mean_row(source, rows_for_mean)]
        name = "rmse_vs_time.png" if source == "offline" else "rmse_vs_time_system.png"
        plot_rmse(os.path.join(out_dir, name), f"{exp['name']}: {label}, receiver static", per_seed, window_s, t_max)

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(out_dir, "rmse_vs_time.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "seed", "flight", "t_s", "pos_rmse_m", "vel_rmse_ms"])
        w.writeheader()
        w.writerows(long_rows)
    with open(os.path.join(next(iter(seed_dirs.values())), "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    paths = [(run, _truth_path(seed_dirs[run["seed"]])) for run in exp["runs"]]
    plot_map(os.path.join(out_dir, "paths_map.png"), f"{exp['name']}: flight paths", nodes, paths)
    return summary
