"""Radar-only solves of a recorded run, without the server.

Replays ``frames.ndjson`` through ``retina_geolocator.multinode_solver`` the
way the backend's solver worker does for a three-node candidate, and scores
every solve against the simulator's truth.  Deterministic: the same recording
always gives the same solves, so two algorithms can be compared on identical
input.

What it shares with the server
    * the geolocator's ``solve_multinode`` (whatever version is installed)
    * altitude by sweep: a pinned solve per layer, plus the initial guess's
      altitude, keeping the lowest ``rms_delay``
    * the publication gates on ``rms_delay`` and ``rms_doppler``

What it does differently, and why it is optimistic
    * association is perfect: the recorder's labels say which detection is the
      aircraft, so clutter never enters a solve
    * no tracker, no epoch alignment (all nodes are framed on the same tick),
      no consensus, trimming or displacement gates
    * the initial guess comes from the radar alone: a grid search over the
      nodes' shared beam for the position and altitude layer whose bistatic
      delays best match the measured ones.  Truth is never used for it.

Writes ``RUN/analysis/offline_solves.csv`` with one row per tick.

Usage::

    python tools/offline_solve.py runs/x/seed-101 [--experiment experiments/baseline_static.json]
"""

import argparse
import csv
import json
import math
import os

import numpy as np

try:
    from retina_geolocator.multinode_solver import solve_multinode, solve_multinode_multistart
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise SystemExit(
        "retina_geolocator is not installed in this Python. Run this with the backend's "
        "Python (retina-server/backend/.venv), which has it."
    ) from exc

_R_EARTH_KM = 6371.0
_C_KM_US = 0.299792458
_KT_TO_MS = 0.514444
_GRID_HALF_KM = 60.0
_GRID_STEP_KM = 0.5

DEFAULT_SOLVER = {
    "alt_mode": "sweep",
    "alt_layers_km": [1.5, 3.0, 5.0, 7.0, 9.0, 11.0],
    "rms_delay_max_us": 3.0,
    "rms_doppler_max_hz_n3": 60.0,
    "rms_doppler_max_hz": 200.0,
    "min_nodes": 3,
}

FIELDS = [
    "t_ms", "t_s", "n_nodes", "published", "reason",
    "est_lat", "est_lon", "est_alt_m", "est_vel_east_ms", "est_vel_north_ms", "rms_delay_us", "rms_doppler_hz",
    "true_lat", "true_lon", "true_alt_m", "true_vel_east_ms", "true_vel_north_ms",
    "err_east_m", "err_north_m", "err_horiz_m", "verr_east_ms", "verr_north_ms", "verr_horiz_ms",
    "adsb_err_horiz_m", "adsb_verr_horiz_ms", "guess_err_horiz_m",
]  # fmt: skip


def _read_ndjson(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _en_m(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    """East/north offset of (lat, lon) from (lat0, lon0), metres."""
    north = math.radians(lat - lat0) * _R_EARTH_KM * 1000.0
    east = math.radians(lon - lon0) * _R_EARTH_KM * 1000.0 * math.cos(math.radians(lat0))
    return east, north


class Geometry:
    """Flat local frame at the first receiver, for the initial-guess grid."""

    def __init__(self, nodes: list[dict]):
        self.lat0, self.lon0 = nodes[0]["rx_lat"], nodes[0]["rx_lon"]
        self.coslat = math.cos(math.radians(self.lat0))
        axis = np.arange(-_GRID_HALF_KM, _GRID_HALF_KM + 1e-9, _GRID_STEP_KM)
        self.e, self.n = np.meshgrid(axis, axis)
        self.nodes = {}
        for cfg in nodes:
            rx = self.enu(cfg["rx_lat"], cfg["rx_lon"], cfg["rx_alt_ft"] * 0.3048e-3)
            tx = self.enu(cfg["tx_lat"], cfg["tx_lon"], cfg["tx_alt_ft"] * 0.3048e-3)
            de, dn = self.e - rx[0], self.n - rx[1]
            bearing = np.degrees(np.arctan2(de, dn)) % 360
            off = np.abs((bearing - cfg.get("beam_azimuth_deg", 0.0) + 180) % 360 - 180)
            in_beam = (off <= cfg.get("beam_width_deg", 360) / 2) & (np.hypot(de, dn) <= cfg.get("max_range_km", 60))
            self.nodes[cfg["node_id"]] = (rx, tx, math.dist(rx, tx), in_beam)

    def enu(self, lat: float, lon: float, alt_km: float) -> tuple[float, float, float]:
        return (
            math.radians(lon - self.lon0) * _R_EARTH_KM * self.coslat,
            math.radians(lat - self.lat0) * _R_EARTH_KM,
            alt_km,
        )

    def lla(self, e_km: float, n_km: float) -> tuple[float, float]:
        return (
            self.lat0 + math.degrees(n_km / _R_EARTH_KM),
            self.lon0 + math.degrees(e_km / (_R_EARTH_KM * self.coslat)),
        )

    def initial_guess(self, meas: list[dict], layers_km: list[float]) -> dict:
        """Grid point and altitude layer whose delays best fit the measured ones."""
        best = None
        mask = np.logical_and.reduce([self.nodes[m["node_id"]][3] for m in meas])
        for alt in layers_km:
            cost = np.zeros_like(self.e)
            for m in meas:
                rx, tx, base, _ = self.nodes[m["node_id"]]
                d_tx = np.sqrt((self.e - tx[0]) ** 2 + (self.n - tx[1]) ** 2 + (alt - tx[2]) ** 2)
                d_rx = np.sqrt((self.e - rx[0]) ** 2 + (self.n - rx[1]) ** 2 + (alt - rx[2]) ** 2)
                cost += ((d_tx + d_rx - base) / _C_KM_US - m["delay_us"]) ** 2
            cost = np.where(mask, cost, np.inf)
            i = np.unravel_index(int(np.argmin(cost)), cost.shape)
            if best is None or cost[i] < best[0]:
                best = (cost[i], float(self.e[i]), float(self.n[i]), alt)
        lat, lon = self.lla(best[1], best[2])
        return {"lat": lat, "lon": lon, "alt_km": best[3]}


def _solve_sweep(guess: dict, meas: list[dict], node_cfgs: dict, layers: list[float]) -> dict | None:
    """Pinned solve per layer, lowest rms_delay wins (the server's 'sweep')."""
    best = None
    for alt in layers:
        s_in = {"initial_guess": dict(guess, alt_km=alt), "measurements": meas, "n_nodes": len(meas)}
        res = solve_multinode(s_in, node_cfgs)
        if res and res.get("success") and (best is None or res["rms_delay"] < best["rms_delay"]):
            best = res
    return best


def _solve_free(guess: dict, meas: list[dict], node_cfgs: dict, layers: list[float]) -> dict | None:
    """Altitude as an unknown, started from the layer nearest the guess (the
    server's 'free' mode with its default of one start)."""
    start = min(layers, key=lambda a: abs(a - guess["alt_km"]))
    s_in = {"initial_guess": guess, "measurements": meas, "n_nodes": len(meas)}
    res = solve_multinode_multistart(s_in, node_cfgs, [start], free_altitude=True)
    return res if res and res.get("success") else None


def solve_epoch(meas: list[dict], node_cfgs: dict, geo: Geometry, cfg: dict) -> tuple[dict | None, dict, str]:
    """(result, initial guess, reason) for one tick's measurements."""
    guess = geo.initial_guess(meas, cfg["alt_layers_km"])
    layers = sorted(set(cfg["alt_layers_km"]) | {round(guess["alt_km"], 3)})
    if cfg["alt_mode"] == "sweep":
        best = _solve_sweep(guess, meas, node_cfgs, layers)
    elif cfg["alt_mode"] == "free":
        best = _solve_free(guess, meas, node_cfgs, layers)
    else:
        raise ValueError(f"unknown alt_mode {cfg['alt_mode']!r}")
    if best is None:
        return None, guess, "solve_failed"
    if best["rms_delay"] > cfg["rms_delay_max_us"]:
        return best, guess, "rejected_rms_delay"
    dop_max = cfg["rms_doppler_max_hz_n3"] if len(meas) == 3 else cfg["rms_doppler_max_hz"]
    if best["rms_doppler"] > dop_max:
        return best, guess, "rejected_rms_doppler"
    return best, guess, "published"


def solve_run(run_dir: str, solver_cfg: dict | None = None) -> list[dict]:
    cfg = {**DEFAULT_SOLVER, **(solver_cfg or {})}
    with open(os.path.join(run_dir, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    node_cfgs = {n["node_id"]: n for n in nodes}
    geo = Geometry(nodes)
    truth = {t["t_ms"]: t["aircraft"][0] for t in _read_ndjson(os.path.join(run_dir, "truth.ndjson")) if t["aircraft"]}
    by_t: dict[int, list[dict]] = {}
    for fr in _read_ndjson(os.path.join(run_dir, "frames.ndjson")):
        by_t.setdefault(fr["t_ms"], []).append(fr)
    t_first = min(by_t)

    rows = []
    for t_ms in sorted(by_t):
        st = truth.get(t_ms)
        meas, adsb = [], None
        for fr in by_t[t_ms]:
            for d in fr["detections"]:
                if st and d["object_id"] == st["id"]:
                    meas.append(
                        {
                            "node_id": fr["node_id"],
                            "delay_us": d["delay_us"],
                            "doppler_hz": d["doppler_hz"],
                            "snr": d["snr_db"],
                        }
                    )
                    adsb = adsb or d.get("adsb")
        row = {"t_ms": t_ms, "t_s": round((t_ms - t_first) / 1000.0 + 1.0, 3), "n_nodes": len(meas)}
        if st:
            row.update(
                {
                    "true_lat": st["lat"],
                    "true_lon": st["lon"],
                    "true_alt_m": st["alt_m"],
                    "true_vel_east_ms": st["vel_east_ms"],
                    "true_vel_north_ms": st["vel_north_ms"],
                }
            )
        if not st or len(meas) < cfg["min_nodes"]:
            row.update(published=False, reason=f"n_nodes<{cfg['min_nodes']}")
            rows.append(row)
            continue
        res, guess, reason = solve_epoch(meas, node_cfgs, geo, cfg)
        row.update(published=reason == "published", reason=reason)
        ge, gn = _en_m(st["lat"], st["lon"], guess["lat"], guess["lon"])
        row["guess_err_horiz_m"] = round(math.hypot(ge, gn), 1)
        if res:
            e, n = _en_m(st["lat"], st["lon"], res["lat"], res["lon"])
            ve, vn = res["vel_east"] - st["vel_east_ms"], res["vel_north"] - st["vel_north_ms"]
            row.update(
                {
                    "est_lat": round(res["lat"], 6),
                    "est_lon": round(res["lon"], 6),
                    "est_alt_m": round(res["alt_m"], 1),
                    "est_vel_east_ms": round(res["vel_east"], 2),
                    "est_vel_north_ms": round(res["vel_north"], 2),
                    "rms_delay_us": round(res["rms_delay"], 4),
                    "rms_doppler_hz": round(res["rms_doppler"], 3),
                    "err_east_m": round(e, 1),
                    "err_north_m": round(n, 1),
                    "err_horiz_m": round(math.hypot(e, n), 1),
                    "verr_east_ms": round(ve, 2),
                    "verr_north_ms": round(vn, 2),
                    "verr_horiz_ms": round(math.hypot(ve, vn), 2),
                }
            )
            if adsb:
                ae, an = _en_m(adsb["lat"], adsb["lon"], res["lat"], res["lon"])
                gs = adsb["gs"] * _KT_TO_MS
                tr = math.radians(adsb["track"])
                row["adsb_err_horiz_m"] = round(math.hypot(ae, an), 1)
                row["adsb_verr_horiz_ms"] = round(
                    math.hypot(res["vel_east"] - gs * math.sin(tr), res["vel_north"] - gs * math.cos(tr)), 2
                )
        rows.append(row)
    return rows


def write_rows(run_dir: str, rows: list[dict]) -> str:
    out = os.path.join(run_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "offline_solves.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline radar-only solves of a recorded run")
    parser.add_argument("run_dir")
    parser.add_argument("--experiment", help="Experiment file whose 'solver' block to use")
    args = parser.parse_args()
    solver_cfg = None
    if args.experiment:
        with open(args.experiment, encoding="utf-8") as f:
            solver_cfg = json.load(f).get("solver")
    rows = solve_run(args.run_dir, solver_cfg)
    path = write_rows(args.run_dir, rows)
    pub = [r for r in rows if r.get("published")]
    print(f"{len(pub)}/{len(rows)} ticks published; wrote {path}")


if __name__ == "__main__":
    main()
