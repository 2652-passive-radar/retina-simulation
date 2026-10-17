"""Turn a recorded run into the tables and maps the project uses.

Reads ``orchestrator --record DIR`` output and writes ``DIR/analysis/``:

``detections.csv``  (ML) every detection: measured delay (µs) and Doppler
                    (Hz), SNR, the label (aircraft id, or ``clutter``), the
                    true delay/Doppler and the residual (measured - true)
``solves.csv``      (moving receiver) every published radar-only solve, one
                    per key and measurement time: the solver's raw state
                    guess (position, velocity, before the server's track
                    smoother), the aircraft it was (label), that aircraft's
                    true state at the same moment, the localisation error
                    vector (east, north, up, metres) and velocity error, and
                    the smoothed map position with its error
``aircraft.csv``    (moving receiver) each aircraft's ADS-B state, every second
``summary.txt``     the headline numbers, also printed

and the labelled delay-Doppler maps into ``DIR/maps/`` (see render_maps.py).

Usage::

    python tools/analyze_run.py runs/ubc-1 [--no-maps] [--every 1]
"""

import argparse
import bisect
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_maps import write_maps  # noqa: E402

_R_EARTH_M = 6_371_000.0
_TRUTH_MAX_GAP_MS = 2_500
_NEAREST_MATCH_M = 5_000.0


def _read_ndjson(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _en_offset_m(lat0, lon0, lat, lon) -> tuple[float, float]:
    north = math.radians(lat - lat0) * _R_EARTH_M
    east = math.radians(lon - lon0) * _R_EARTH_M * math.cos(math.radians(lat0))
    return east, north


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


class Truth:
    """Each aircraft's state over time, looked up by sim id or ADS-B hex."""

    _INTERP = ("lat", "lon", "alt_m", "vel_east_ms", "vel_north_ms", "vel_up_ms")

    def __init__(self, ticks: list[dict]):
        self.ticks = ticks
        self.times = [t["t_ms"] for t in ticks]
        self.series: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for tick in ticks:
            for ac in tick["aircraft"]:
                self.series[ac["id"]].append((tick["t_ms"], ac))
                if ac.get("adsb_hex"):
                    self.series[ac["adsb_hex"].lower()] = self.series[ac["id"]]

    def at(self, key: str, t_ms: int) -> dict | None:
        series = self.series.get(key) or self.series.get(str(key).lower())
        if not series:
            return None
        times = [t for t, _ in series]
        i = bisect.bisect_left(times, t_ms)
        if i in (0, len(series)):
            t, st = series[min(i, len(series) - 1)]
            return st if abs(t - t_ms) <= _TRUTH_MAX_GAP_MS else None
        (ta, a), (tb, b) = series[i - 1], series[i]
        w = (t_ms - ta) / (tb - ta) if tb != ta else 0.0
        return {**b, **{k: a[k] + w * (b[k] - a[k]) for k in self._INTERP}}

    def nearest(self, lat: float, lon: float, t_ms: int) -> dict | None:
        j = bisect.bisect_left(self.times, t_ms)
        near = [k for k in (j - 1, j) if 0 <= k < len(self.times)]
        if not near:
            return None
        i = min(near, key=lambda k: abs(self.times[k] - t_ms))
        best, best_d = None, _NEAREST_MATCH_M
        for ac in self.ticks[i]["aircraft"]:
            e, n = _en_offset_m(ac["lat"], ac["lon"], lat, lon)
            if math.hypot(e, n) <= best_d:
                best, best_d = ac, math.hypot(e, n)
        return self.at(best["id"], t_ms) if best else None


def detection_rows(frames: list[dict]) -> list[dict]:
    rows = []
    for fr in frames:
        for d in fr["detections"]:
            rows.append(
                {
                    "t_ms": fr["t_ms"],
                    "node": fr["node_id"],
                    "tower": fr["tx_callsign"],
                    "delay_us": d["delay_us"],
                    "doppler_hz": d["doppler_hz"],
                    "snr_db": d["snr_db"],
                    "label": "clutter" if d["is_clutter"] else d["object_id"],
                    "delay_true_us": d["delay_true"],
                    "doppler_true_hz": d["doppler_true"],
                    "delay_resid_us": d["delay_resid_us"],
                    "doppler_resid_hz": d["doppler_resid_hz"],
                }
            )
    return rows


def is_radar_only_published(s: dict) -> bool:
    published = s.get("published") or s.get("outcome") == "published"
    return bool(published) and not s.get("known_lane") and not s.get("adsb_hex")


def one_per_epoch(solves: list[dict]) -> list[dict]:
    """The published radar-only solves, one per (key, measurement time).

    The server can publish the same key at the same measurement time twice:
    a 2-node solve, then a re-solve once a third node joins.  The re-solve's
    map position is held at the first one (the smoother takes no new epoch),
    so both records carry the same lat/lon and only raw_lat/raw_lon tells
    them apart.  The last published record is the server's final answer for
    that epoch, and the only one kept.
    """
    latest: dict[tuple, dict] = {}
    for s in solves:
        if not is_radar_only_published(s):
            continue
        epoch = (s.get("solve_key"), s.get("measurement_ts_ms") or s.get("ts_ms"))
        if epoch not in latest or (s.get("ts_ms") or 0) >= (latest[epoch].get("ts_ms") or 0):
            latest[epoch] = s
    return sorted(latest.values(), key=lambda s: s.get("measurement_ts_ms") or s.get("ts_ms") or 0)


def solve_rows(solves: list[dict], truth: Truth) -> list[dict]:
    """One row per published radar-only solve, scored on the solver's raw
    position (raw_lat/raw_lon, before the server's track smoother) so it is
    comparable with an offline solve; the smoothed map position and its error
    are kept beside it."""
    rows = []
    for s in one_per_epoch(solves):
        map_lat, map_lon = s.get("lat"), s.get("lon")
        lat = s["raw_lat"] if s.get("raw_lat") is not None else map_lat
        lon = s["raw_lon"] if s.get("raw_lon") is not None else map_lon
        if lat is None or lon is None:
            continue
        t_ms = s.get("measurement_ts_ms") or s["ts_ms"]
        st = (truth.at(s["gt_hex"], t_ms) if s.get("gt_hex") else None) or truth.nearest(lat, lon, t_ms)
        row = {
            "t_ms": t_ms,
            "aircraft": st["id"] if st else None,
            "n_nodes": s.get("n_nodes"),
            "solve_key": s.get("solve_key"),
            "est_lat": lat,
            "est_lon": lon,
            "est_alt_m": s.get("alt_m"),
            "alt_pinned": s.get("altitude_mode") == "pinned",
            "est_vel_east_ms": s.get("vel_east"),
            "est_vel_north_ms": s.get("vel_north"),
            "map_lat": map_lat,
            "map_lon": map_lon,
        }
        if st:
            e, n = _en_offset_m(st["lat"], st["lon"], lat, lon)
            row.update(
                {
                    "true_lat": round(st["lat"], 6),
                    "true_lon": round(st["lon"], 6),
                    "true_alt_m": round(st["alt_m"], 1),
                    "true_vel_east_ms": round(st["vel_east_ms"], 2),
                    "true_vel_north_ms": round(st["vel_north_ms"], 2),
                    "true_vel_up_ms": round(st["vel_up_ms"], 2),
                    "err_east_m": round(e, 1),
                    "err_north_m": round(n, 1),
                    "err_up_m": round(s["alt_m"] - st["alt_m"], 1) if s.get("alt_m") is not None else None,
                    "err_horiz_m": round(math.hypot(e, n), 1),
                }
            )
            if s.get("vel_east") is not None and s.get("vel_north") is not None:
                row["verr_horiz_ms"] = round(
                    math.hypot(s["vel_east"] - st["vel_east_ms"], s["vel_north"] - st["vel_north_ms"]), 2
                )
            if map_lat is not None and map_lon is not None:
                me, mn = _en_offset_m(st["lat"], st["lon"], map_lat, map_lon)
                row["map_err_horiz_m"] = round(math.hypot(me, mn), 1)
        rows.append(row)
    return rows


def aircraft_rows(ticks: list[dict]) -> list[dict]:
    keep = ("lat", "lon", "alt_m", "vel_east_ms", "vel_north_ms", "vel_up_ms", "heading_deg", "speed_ms")
    return [
        {
            "t_ms": t["t_ms"],
            "aircraft": ac["id"],
            "adsb_hex": ac["adsb_hex"],
            "callsign": ac["adsb_callsign"],
            **{k: ac[k] for k in keep},
        }
        for t in ticks
        for ac in t["aircraft"]
        if ac["has_adsb"]
    ]


_DETECTION_FIELDS = [
    "t_ms", "node", "tower", "delay_us", "doppler_hz", "snr_db", "label",
    "delay_true_us", "doppler_true_hz", "delay_resid_us", "doppler_resid_hz",
]  # fmt: skip
_SOLVE_FIELDS = [
    "t_ms", "aircraft", "n_nodes", "solve_key",
    "est_lat", "est_lon", "est_alt_m", "alt_pinned", "est_vel_east_ms", "est_vel_north_ms",
    "true_lat", "true_lon", "true_alt_m", "true_vel_east_ms", "true_vel_north_ms", "true_vel_up_ms",
    "err_east_m", "err_north_m", "err_up_m", "err_horiz_m", "verr_horiz_ms",
    "map_lat", "map_lon", "map_err_horiz_m",
]  # fmt: skip
_AIRCRAFT_FIELDS = [
    "t_ms", "aircraft", "adsb_hex", "callsign", "lat", "lon", "alt_m",
    "vel_east_ms", "vel_north_ms", "vel_up_ms", "heading_deg", "speed_ms",
]  # fmt: skip


def summary_text(nodes, dets, solves_all, solves, ac_rows, ticks, n_maps) -> str:
    lines = []
    span_s = (ticks[-1]["t_ms"] - ticks[0]["t_ms"]) / 1000 if len(ticks) > 1 else 0
    aircraft = sorted({r["aircraft"] for r in ac_rows})
    lines.append(f"Run: {span_s:.0f} s, {len(nodes)} nodes, {len(aircraft)} aircraft ({', '.join(aircraft)})")
    clutter = sum(1 for d in dets if d["label"] == "clutter")
    lines.append(f"Detections: {len(dets)} ({clutter} clutter)")
    lines.append("Residuals (measured - true), standard deviation per node:")
    for n in nodes:
        mine = [d for d in dets if d["node"] == n["node_id"] and d["label"] != "clutter"]
        dr = [d["delay_resid_us"] for d in mine]
        fr = [d["doppler_resid_hz"] for d in mine]
        if dr:
            lines.append(
                f"  {n['node_id']} ({n['tx_callsign']}, {n['fc_hz'] / 1e6:.1f} MHz): {len(mine)} aircraft detections, "
                f"delay {statistics.pstdev(dr):.3f} us, Doppler {statistics.pstdev(fr):.2f} Hz"
            )
        else:
            lines.append(f"  {n['node_id']} ({n['tx_callsign']}): no aircraft detections")
    radar_only = [s for s in solves_all if not s.get("known_lane") and not s.get("adsb_hex")]
    unpublished = Counter(s.get("outcome") for s in radar_only if not is_radar_only_published(s))
    reasons = ", ".join(f"{k} {v}" for k, v in unpublished.most_common())
    if solves_all:
        lines.append(
            f"Radar-only solves: {len(solves)} published, {sum(unpublished.values())} not published ({reasons})"
        )
    else:
        lines.append("Server solves: none recorded (no server in this run)")
    if unpublished.get("dark_follow_shadow"):
        lines.append("  dark_follow_shadow solves are the server's diagnostic shadow lane: recorded, never published.")
    err = [r["err_horiz_m"] for r in solves if r.get("err_horiz_m") is not None]
    if err:
        lines.append(
            "Localisation error (horizontal, raw solve, one per epoch): "
            f"median {_pct(err, 0.5):.0f} m, 90th percentile {_pct(err, 0.9):.0f} m"
        )
        for n in sorted({r["n_nodes"] for r in solves if r.get("err_horiz_m") is not None}, reverse=True):
            e_n = [r["err_horiz_m"] for r in solves if r.get("n_nodes") == n and r.get("err_horiz_m") is not None]
            lines.append(f"  {n} nodes: {len(e_n)} solves, median {_pct(e_n, 0.5):.0f} m, max {max(e_n):.0f} m")
    if any(r.get("alt_pinned") for r in solves):
        lines.append("  Altitude is pinned in radar-only solves, so err_up_m is not a solved quantity.")
    known = [s for s in solves_all if s.get("known_lane") or s.get("adsb_hex")]
    if known:
        lines.append(
            f"Note: {len(known)} ADS-B-assisted solves were left out (run with --adsb-truth-only to avoid them)."
        )
    if n_maps is not None:
        lines.append(f"Delay-Doppler maps: {n_maps}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tables and maps from a recorded run")
    parser.add_argument("run_dir")
    parser.add_argument("--no-maps", action="store_true", help="Skip the delay-Doppler maps")
    parser.add_argument("--every", type=int, default=1, help="Map every Nth frame")
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    frames = _read_ndjson(os.path.join(args.run_dir, "frames.ndjson"))
    ticks = _read_ndjson(os.path.join(args.run_dir, "truth.ndjson"))
    solves_all = _read_ndjson(os.path.join(args.run_dir, "solves.ndjson"))
    truth = Truth(ticks)

    dets = detection_rows(frames)
    solves = solve_rows(solves_all, truth)
    ac_rows = aircraft_rows(ticks)

    out = os.path.join(args.run_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    _write_csv(os.path.join(out, "detections.csv"), dets, _DETECTION_FIELDS)
    _write_csv(os.path.join(out, "solves.csv"), solves, _SOLVE_FIELDS)
    _write_csv(os.path.join(out, "aircraft.csv"), ac_rows, _AIRCRAFT_FIELDS)
    n_maps = None if args.no_maps else write_maps(args.run_dir, args.every)[0]

    text = summary_text(nodes, dets, solves_all, solves, ac_rows, ticks, n_maps)
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text, end="")
    print(f"wrote {out}{os.sep}" + ("" if args.no_maps else f" and {os.path.join(args.run_dir, 'maps')}{os.sep}"))


if __name__ == "__main__":
    main()
