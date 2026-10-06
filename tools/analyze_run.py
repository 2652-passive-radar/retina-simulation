"""Turn a recorded run into error analytics tables.

Reads a run directory written by ``orchestrator --record DIR`` (nodes.json,
frames.ndjson, truth.ndjson) plus, if present, ``solves.ndjson`` from
``tools/record_solves.py``, and writes ``DIR/analysis/``:

``detections.csv``  one row per detection: measured and true delay/Doppler,
                    residuals, SNR, and the label (object id or clutter)
``per_node.csv``    what each node saw, with its tower: detection counts,
                    clutter share, distinct targets, residual statistics
``per_target.csv``  each aircraft: which nodes detected it and how often
``solves.csv``      each server solve: the state guess, the truth at the
                    measurement time, and the localisation error vector
                    (east/north/up metres) and velocity error
``summary.json``    the headline numbers, also printed

Delays are microseconds of bistatic differential delay, as the sim sends
them.  Truth comes from the sim's own aircraft state, so it includes the dark
aircraft the ADS-B push leaves out.

Usage::

    python tools/analyze_run.py runs/ubc-1
"""

import argparse
import bisect
import csv
import json
import math
import os
import statistics
from collections import defaultdict

_R_EARTH_M = 6_371_000.0
_TRUTH_MAX_GAP_MS = 2_500
_NEAREST_MATCH_M = 5_000.0


def _read_ndjson(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _R_EARTH_M * math.asin(math.sqrt(a))


def _en_offset_m(lat_ref, lon_ref, lat, lon) -> tuple[float, float]:
    north = math.radians(lat - lat_ref) * _R_EARTH_M
    east = math.radians(lon - lon_ref) * _R_EARTH_M * math.cos(math.radians(lat_ref))
    return east, north


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "std": None, "p50": None, "p90": None}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "std": round(statistics.pstdev(values), 4),
        "p50": round(ordered[len(ordered) // 2], 4),
        "p90": round(ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))], 4),
    }


class TruthIndex:
    """Aircraft state over time, looked up by sim object id or ADS-B hex."""

    _FIELDS = ("lat", "lon", "alt_m", "vel_east_ms", "vel_north_ms", "vel_up_ms")

    def __init__(self, ticks: list[dict]):
        self.times = [t["t_ms"] for t in ticks]
        self.ticks = ticks
        self.series: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        self.meta: dict[str, dict] = {}
        for tick in ticks:
            for ac in tick["aircraft"]:
                self.series[ac["id"]].append((tick["t_ms"], ac))
                self.meta[ac["id"]] = ac
                if ac.get("adsb_hex"):
                    self.series.setdefault(ac["adsb_hex"].lower(), self.series[ac["id"]])

    def state(self, key: str, t_ms: int) -> dict | None:
        series = self.series.get(key) or self.series.get(str(key).lower())
        if not series:
            return None
        times = [t for t, _ in series]
        i = bisect.bisect_left(times, t_ms)
        if i == 0:
            t0, s0 = series[0]
            return s0 if abs(t0 - t_ms) <= _TRUTH_MAX_GAP_MS else None
        if i == len(series):
            t1, s1 = series[-1]
            return s1 if abs(t_ms - t1) <= _TRUTH_MAX_GAP_MS else None
        (ta, a), (tb, b) = series[i - 1], series[i]
        if tb - ta > 2 * _TRUTH_MAX_GAP_MS:
            return None
        w = (t_ms - ta) / (tb - ta) if tb != ta else 0.0
        out = dict(b)
        for k in self._FIELDS:
            out[k] = a[k] + w * (b[k] - a[k])
        return out

    def nearest(self, lat: float, lon: float, t_ms: int) -> tuple[str, dict, float] | None:
        if not self.times:
            return None
        j = bisect.bisect_left(self.times, t_ms)
        i = min((k for k in (j - 1, j) if 0 <= k < len(self.times)), key=lambda k: abs(self.times[k] - t_ms))
        if abs(self.times[i] - t_ms) > _TRUTH_MAX_GAP_MS:
            return None
        best = None
        for ac in self.ticks[i]["aircraft"]:
            d = _haversine_m(lat, lon, ac["lat"], ac["lon"])
            if best is None or d < best[2]:
                best = (ac["id"], ac, d)
        if best and best[2] <= _NEAREST_MATCH_M:
            return best[0], self.state(best[0], t_ms) or best[1], best[2]
        return None


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_detections(nodes: dict, frames: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    det_rows = []
    per_node = defaultdict(
        lambda: {"frames": 0, "sent": 0, "dets": 0, "clutter": 0, "targets": set(), "dr": [], "fr": [], "snr": []}
    )
    per_target = defaultdict(lambda: defaultdict(int))
    target_span: dict[str, list[int]] = {}
    for fr in frames:
        nid = fr["node_id"]
        agg = per_node[nid]
        agg["frames"] += 1
        agg["sent"] += int(fr["sent"])
        for d in fr["detections"]:
            agg["dets"] += 1
            agg["snr"].append(d["snr_db"])
            if d["is_clutter"]:
                agg["clutter"] += 1
            else:
                agg["targets"].add(d["object_id"])
                agg["dr"].append(d["delay_resid_us"])
                agg["fr"].append(d["doppler_resid_hz"])
                per_target[d["object_id"]][nid] += 1
                span = target_span.setdefault(d["object_id"], [fr["t_ms"], fr["t_ms"]])
                span[1] = fr["t_ms"]
            det_rows.append(
                {
                    "t_ms": fr["t_ms"],
                    "node_id": nid,
                    "tx_callsign": fr["tx_callsign"],
                    "delay_us": d["delay_us"],
                    "doppler_hz": d["doppler_hz"],
                    "snr_db": d["snr_db"],
                    "is_clutter": d["is_clutter"],
                    "object_id": d["object_id"],
                    "adsb_hex": d["adsb_hex"],
                    "adsb_tagged": d.get("adsb") is not None,
                    "delay_true_us": d["delay_true"],
                    "doppler_true_hz": d["doppler_true"],
                    "delay_resid_us": d["delay_resid_us"],
                    "doppler_resid_hz": d["doppler_resid_hz"],
                }
            )

    node_rows = []
    for nid, agg in sorted(per_node.items()):
        cfg = nodes.get(nid, {})
        dr, fr_ = _stats(agg["dr"]), _stats(agg["fr"])
        baseline = (
            _haversine_m(cfg["rx_lat"], cfg["rx_lon"], cfg["tx_lat"], cfg["tx_lon"]) / 1000 if cfg else float("nan")
        )
        node_rows.append(
            {
                "node_id": nid,
                "tx_callsign": cfg.get("tx_callsign"),
                "fc_mhz": round(cfg.get("fc_hz", 0) / 1e6, 2),
                "baseline_km": round(baseline, 1),
                "beam_azimuth_deg": cfg.get("beam_azimuth_deg"),
                "frames": agg["frames"],
                "frames_sent": agg["sent"],
                "detections": agg["dets"],
                "target_detections": agg["dets"] - agg["clutter"],
                "clutter_detections": agg["clutter"],
                "clutter_frac": round(agg["clutter"] / agg["dets"], 3) if agg["dets"] else None,
                "distinct_targets": len(agg["targets"]),
                "dets_per_frame": round(agg["dets"] / agg["frames"], 2) if agg["frames"] else None,
                "snr_mean_db": round(statistics.fmean(agg["snr"]), 2) if agg["snr"] else None,
                "delay_resid_mean_us": dr["mean"],
                "delay_resid_std_us": dr["std"],
                "doppler_resid_mean_hz": fr_["mean"],
                "doppler_resid_std_hz": fr_["std"],
            }
        )

    node_ids = sorted(per_node)
    target_rows = []
    for oid, counts in sorted(per_target.items()):
        row = {"object_id": oid, "n_nodes_seen": len(counts), "first_t_ms": target_span[oid][0]}
        row["last_t_ms"] = target_span[oid][1]
        for nid in node_ids:
            row[f"dets_{nid}"] = counts.get(nid, 0)
        target_rows.append(row)
    return det_rows, node_rows, target_rows


def analyze_solves(solves: list[dict], truth: TruthIndex) -> list[dict]:
    rows = []
    for s in solves:
        lat = s["lat"] if s.get("lat") is not None else s.get("raw_lat")
        lon = s["lon"] if s.get("lon") is not None else s.get("raw_lon")
        if lat is None or lon is None:
            continue
        t_ms = s.get("measurement_ts_ms") or s.get("ts_ms")
        key, st, how = s.get("gt_hex"), None, None
        if key:
            st = truth.state(key, t_ms)
            how = "server"
        if st is None:
            near = truth.nearest(lat, lon, t_ms)
            if near:
                key, st, _ = near
                how = "nearest"
        row = {
            "ts_ms": s.get("ts_ms"),
            "measurement_ts_ms": t_ms,
            "outcome": s.get("outcome"),
            "lane": "known" if s.get("known_lane") else "dark",
            "published": bool(s.get("published") or s.get("outcome") == "published"),
            "n_nodes": s.get("n_nodes"),
            "altitude_mode": s.get("altitude_mode"),
            "rms_delay_us": s.get("rms_delay"),
            "rms_doppler_hz": s.get("rms_doppler"),
            "est_lat": lat,
            "est_lon": lon,
            "est_alt_m": s.get("alt_m"),
            "est_vel_east_ms": s.get("vel_east"),
            "est_vel_north_ms": s.get("vel_north"),
            "guess_lat": s.get("guess_lat"),
            "guess_lon": s.get("guess_lon"),
            "guess_alt_m": round(s["guess_alt_km"] * 1000, 1) if s.get("guess_alt_km") is not None else None,
            "truth_id": key if st else None,
            "truth_match": how if st else None,
            "server_gt_error_km": s.get("gt_error_km"),
        }
        if st:
            e, n = _en_offset_m(st["lat"], st["lon"], lat, lon)
            up = (s["alt_m"] - st["alt_m"]) if s.get("alt_m") is not None else None
            row.update(
                {
                    "true_lat": round(st["lat"], 6),
                    "true_lon": round(st["lon"], 6),
                    "true_alt_m": round(st["alt_m"], 1),
                    "true_vel_east_ms": round(st["vel_east_ms"], 2),
                    "true_vel_north_ms": round(st["vel_north_ms"], 2),
                    "true_vel_up_ms": round(st["vel_up_ms"], 2),
                    "true_has_adsb": st.get("has_adsb"),
                    "true_object_type": st.get("object_type"),
                    "err_east_m": round(e, 1),
                    "err_north_m": round(n, 1),
                    "err_up_m": round(up, 1) if up is not None else None,
                    "err_horiz_m": round(math.hypot(e, n), 1),
                    "err_3d_m": round(math.sqrt(e * e + n * n + up * up), 1) if up is not None else None,
                    "vel_err_east_ms": round(s["vel_east"] - st["vel_east_ms"], 2)
                    if s.get("vel_east") is not None
                    else None,
                    "vel_err_north_ms": round(s["vel_north"] - st["vel_north_ms"], 2)
                    if s.get("vel_north") is not None
                    else None,
                }
            )
        rows.append(row)
    return rows


def summarize(node_rows, target_rows, solve_rows, frames, truth: TruthIndex) -> dict:
    by_outcome = defaultdict(list)
    by_lane = defaultdict(list)
    by_n = defaultdict(list)
    for r in solve_rows:
        if r.get("err_horiz_m") is None:
            continue
        by_outcome[r["outcome"]].append(r["err_horiz_m"])
        if r["published"]:
            by_lane[r["lane"]].append(r["err_horiz_m"])
            by_n[str(r["n_nodes"])].append(r["err_horiz_m"])
    aircraft_ids = {ac["id"] for t in truth.ticks for ac in t["aircraft"]}
    seen = {r["object_id"] for r in target_rows}
    return {
        "frames": len(frames),
        "nodes": len(node_rows),
        "aircraft_in_world": len(aircraft_ids),
        "aircraft_detected": len(seen),
        "aircraft_seen_by_3plus_nodes": sum(1 for r in target_rows if r["n_nodes_seen"] >= 3),
        "solves": len(solve_rows),
        "solves_with_truth": sum(1 for r in solve_rows if r.get("err_horiz_m") is not None),
        "solve_outcomes": {k: len(v) for k, v in sorted(by_outcome.items())},
        "horiz_error_m_by_outcome": {k: _stats(v) for k, v in sorted(by_outcome.items())},
        "published_horiz_error_m_by_lane": {k: _stats(v) for k, v in sorted(by_lane.items())},
        "published_horiz_error_m_by_n_nodes": {k: _stats(v) for k, v in sorted(by_n.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Error analytics for a recorded simulation run")
    parser.add_argument("run_dir")
    args = parser.parse_args()

    with open(os.path.join(args.run_dir, "nodes.json"), encoding="utf-8") as f:
        nodes = {n["node_id"]: n for n in json.load(f)}
    frames = _read_ndjson(os.path.join(args.run_dir, "frames.ndjson"))
    truth = TruthIndex(_read_ndjson(os.path.join(args.run_dir, "truth.ndjson")))
    solves = _read_ndjson(os.path.join(args.run_dir, "solves.ndjson"))

    det_rows, node_rows, target_rows = analyze_detections(nodes, frames)
    solve_rows = analyze_solves(solves, truth)
    summary = summarize(node_rows, target_rows, solve_rows, frames, truth)

    out = os.path.join(args.run_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    _write_csv(os.path.join(out, "detections.csv"), det_rows)
    _write_csv(os.path.join(out, "per_node.csv"), node_rows)
    _write_csv(os.path.join(out, "per_target.csv"), target_rows)
    _write_csv(os.path.join(out, "solves.csv"), solve_rows)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "per_node": node_rows}, f, indent=2)

    print(
        f"{'node':<12}{'tower':<11}{'MHz':>7}{'frames':>8}{'dets':>7}{'clutter':>9}{'targets':>9}"
        f"{'dly σ µs':>10}{'dop σ Hz':>10}"
    )
    for r in node_rows:
        print(
            f"{r['node_id']:<12}{r['tx_callsign'] or '':<11}{r['fc_mhz']:>7}{r['frames']:>8}{r['detections']:>7}"
            f"{r['clutter_detections']:>9}{r['distinct_targets']:>9}"
            f"{r['delay_resid_std_us'] or 0:>10.3f}{r['doppler_resid_std_hz'] or 0:>10.2f}"
        )
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
