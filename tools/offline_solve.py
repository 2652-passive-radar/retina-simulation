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

Writes ``RUN/analysis/offline_solves.csv`` (``--receiver geometric``:
``offline_solves_geometric.csv``; ``--receiver geometric_doppler``:
``offline_solves_geometric_doppler.csv``) with one row per tick.

Usage::

    python tools/offline_solve.py runs/x/seed-101 [--experiment experiments/baseline_static.json]
    python tools/offline_solve.py runs/x/seed-101 --receiver geometric
    python tools/offline_solve.py runs/x/seed-101 --receiver geometric_doppler
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
_C_M_S = 299792458.0
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
    "rx_doppler_passes": 1,
}

RECEIVER_MODES = ("stationary", "geometric", "geometric_doppler")

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


_WGS84_A_M = 6378137.0
_WGS84_E2 = 6.69437999014e-3


def _ecef_m(lat: float, lon: float, alt_m: float) -> tuple[float, float, float]:
    phi, lam = math.radians(lat), math.radians(lon)
    s = math.sin(phi)
    n = _WGS84_A_M / math.sqrt(1.0 - _WGS84_E2 * s * s)
    return (
        (n + alt_m) * math.cos(phi) * math.cos(lam),
        (n + alt_m) * math.cos(phi) * math.sin(lam),
        (n * (1.0 - _WGS84_E2) + alt_m) * s,
    )


def _bistatic_range_m(p, rx, tx) -> float:
    return math.dist(p, tx) + math.dist(p, rx) - math.dist(rx, tx)


def _short(node_id: str) -> str:
    return node_id.rsplit("-", 1)[-1]


def receiver_poses(run_dir: str) -> dict[int, dict]:
    """Each tick's receiver pose per node, from rx_trajectory.ndjson ({} if absent)."""
    path = os.path.join(run_dir, "rx_trajectory.ndjson")
    if not os.path.exists(path):
        return {}
    return {r["t_ms"]: r["rx"] for r in _read_ndjson(path)}


def _at_pose(nodes: list[dict], pose: dict) -> list[dict]:
    """Node configs with each receiver moved to its pose this tick."""
    return [
        dict(n, rx_lat=pose[n["node_id"]]["lat"], rx_lon=pose[n["node_id"]]["lon"],
             rx_alt_ft=pose[n["node_id"]]["alt_m"] / 0.3048)
        if n["node_id"] in pose else n
        for n in nodes
    ]  # fmt: skip


def _enu_to_ecef_vec(lat: float, lon: float, e: float, n: float, u: float) -> tuple[float, float, float]:
    phi, lam = math.radians(lat), math.radians(lon)
    sp, cp, sl, cl = math.sin(phi), math.cos(phi), math.sin(lam), math.cos(lam)
    return (-sl * e - sp * cl * n + cp * cl * u, cl * e - sp * sl * n + cp * sl * u, cp * n + sp * u)


def _unit(a, b) -> tuple[float, float, float]:
    """Unit vector from a to b."""
    d = math.dist(a, b)
    return tuple((bi - ai) / d for ai, bi in zip(a, b))


def receiver_doppler_hz(p_ecef, rx_ecef, tx_ecef, rx_vel_ecef, fc_hz: float) -> float:
    """The Doppler a receiver moving at ``rx_vel_ecef`` (m/s) adds to its own
    measurement, Hz: (fc/c) (u_RP + u_TR) . v_R, u_RP pointing from the
    receiver to the aircraft and u_TR from the tower to the receiver.  The
    simulator adds exactly this (world._receiver_doppler)."""
    u_rp, u_tr = _unit(rx_ecef, p_ecef), _unit(tx_ecef, rx_ecef)
    return fc_hz / _C_M_S * sum((a + b) * v for a, b, v in zip(u_rp, u_tr, rx_vel_ecef))


def _pose_ecef(pose: dict) -> tuple[tuple, tuple]:
    pos = _ecef_m(pose["lat"], pose["lon"], pose["alt_m"])
    vel = _enu_to_ecef_vec(
        pose["lat"], pose["lon"], pose["vel_east_ms"], pose["vel_north_ms"], pose.get("vel_up_ms", 0.0)
    )
    return pos, vel


def _doppler_corrections(p_ecef, nodes: list[dict], pose: dict) -> dict[str, float]:
    """Each node's receiver Doppler for an aircraft at ``p_ecef``, Hz."""
    out = {}
    for n in nodes:
        rx, vel = _pose_ecef(pose[n["node_id"]])
        tx = _ecef_m(n["tx_lat"], n["tx_lon"], n["tx_alt_ft"] * 0.3048)
        out[n["node_id"]] = receiver_doppler_hz(p_ecef, rx, tx, vel, n["fc_hz"])
    return out


def _solve_doppler_compensated(meas: list[dict], moved: list[dict], pose: dict, cfg: dict):
    """GD: solve as G, remove each channel's receiver Doppler evaluated at that
    solve, and solve again (``rx_doppler_passes`` times).  Delay needs no
    correction: in one ~1 s capture the receiver moves ~15 m at vehicle speed,
    against ~150 m of delay resolution.  Returns the last pass's result and the
    corrections it used."""
    cfgs, geo = {n["node_id"]: n for n in moved}, Geometry(moved)
    res, guess, reason = solve_epoch(meas, cfgs, geo, cfg)
    corr: dict[str, float] = {}
    for _ in range(cfg["rx_doppler_passes"]):
        at = res or {"lat": guess["lat"], "lon": guess["lon"], "alt_m": guess["alt_km"] * 1000.0}
        corr = _doppler_corrections(_ecef_m(at["lat"], at["lon"], at["alt_m"]), moved, pose)
        fixed = [dict(m, doppler_hz=m["doppler_hz"] - corr[m["node_id"]]) for m in meas]
        res, _, reason = solve_epoch(fixed, cfgs, geo, cfg)
    return res, guess, reason, corr


def solve_run(
    run_dir: str, solver_cfg: dict | None = None, receiver: str = "stationary", pose_estimate: dict | None = None
) -> list[dict]:
    """Solve every tick of a recording.

    ``receiver`` says what the solver is told about where the receivers are:
    ``stationary`` (S), their registered position from nodes.json, as the
    server does today; ``geometric`` (G), their true position that tick from
    rx_trajectory.ndjson, i.e. geometric compensation; ``geometric_doppler``
    (GD), G with each channel's receiver Doppler, computed from the receiver's
    true velocity, removed from the measured Doppler.  On a recording without
    receiver motion all three are identical.

    Each row also carries the receiver's offset from its registered position
    and, per node, ``bias_us_<node>``: how far the true bistatic delay that
    tick differs from the delay the same aircraft would give a receiver at its
    registered position, which is the error S builds its solve on, and
    ``rxdop_true_hz_<node>``: the receiver Doppler in that node's measurement,
    from the true aircraft position.  GD rows add ``rxdop_corr_hz_<node>``,
    the correction GD subtracted.

    ``pose_estimate`` ({t_ms: {node_id: pose}}, the rx_trajectory format)
    replaces the true receiver pose given to G and GD, e.g. a GNSS-based
    estimate.  The receiver offset, bias and true receiver Doppler columns
    still describe the truth, and each row adds ``rx_pos_err_m`` and
    ``rx_vel_err_ms``, the estimate's horizontal error.
    """
    if receiver not in RECEIVER_MODES:
        raise ValueError(f"unknown receiver mode {receiver!r}")
    cfg = {**DEFAULT_SOLVER, **(solver_cfg or {})}
    with open(os.path.join(run_dir, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    node_cfgs = {n["node_id"]: n for n in nodes}
    geo = Geometry(nodes)
    poses = receiver_poses(run_dir)
    tx_ecef = {n["node_id"]: _ecef_m(n["tx_lat"], n["tx_lon"], n["tx_alt_ft"] * 0.3048) for n in nodes}
    rx0_ecef = {n["node_id"]: _ecef_m(n["rx_lat"], n["rx_lon"], n["rx_alt_ft"] * 0.3048) for n in nodes}
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
        pose = poses.get(t_ms) or {}
        used = pose if pose_estimate is None else pose_estimate.get(t_ms) or {}
        if pose_estimate is not None and pose and used:
            est, tru = next(iter(used.values())), next(iter(pose.values()))
            row["rx_pos_err_m"] = round(
                math.hypot(est["offset_east_m"] - tru["offset_east_m"], est["offset_north_m"] - tru["offset_north_m"]),
                3,
            )
            row["rx_vel_err_ms"] = round(
                math.hypot(est["vel_east_ms"] - tru["vel_east_ms"], est["vel_north_ms"] - tru["vel_north_ms"]), 4
            )
        if pose:
            first = next(iter(pose.values()))
            oe, on = first.get("offset_east_m", 0.0), first.get("offset_north_m", 0.0)
            row["rx_offset_m"] = round(math.hypot(oe, on), 2)
            row["rx_offset_bearing_deg"] = round(math.degrees(math.atan2(oe, on)) % 360, 1) if (oe or on) else None
            row["rx_speed_ms"] = round(math.hypot(first["vel_east_ms"], first["vel_north_ms"]), 3)
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
            p = _ecef_m(st["lat"], st["lon"], st["alt_m"])
            for nid, rx0 in rx0_ecef.items():
                rx_t = _ecef_m(pose[nid]["lat"], pose[nid]["lon"], pose[nid]["alt_m"]) if nid in pose else rx0
                bias_m = _bistatic_range_m(p, rx_t, tx_ecef[nid]) - _bistatic_range_m(p, rx0, tx_ecef[nid])
                row[f"bias_us_{_short(nid)}"] = round(bias_m / (_C_KM_US * 1000.0), 4)
            if pose:
                for nid, f in _doppler_corrections(p, nodes, pose).items():
                    row[f"rxdop_true_hz_{_short(nid)}"] = round(f, 3)
        if not st or len(meas) < cfg["min_nodes"]:
            row.update(published=False, reason=f"n_nodes<{cfg['min_nodes']}")
            rows.append(row)
            continue
        if receiver == "geometric_doppler" and used:
            res, guess, reason, corr = _solve_doppler_compensated(meas, _at_pose(nodes, used), used, cfg)
            for nid, f in corr.items():
                row[f"rxdop_corr_hz_{_short(nid)}"] = round(f, 3)
        elif receiver != "stationary" and used:
            moved = _at_pose(nodes, used)
            res, guess, reason = solve_epoch(meas, {n["node_id"]: n for n in moved}, Geometry(moved), cfg)
        else:
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


def write_rows(run_dir: str, rows: list[dict], name: str = "offline_solves.csv") -> str:
    out = os.path.join(run_dir, "analysis")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, name)
    extra = sorted({k for r in rows for k in r} - set(FIELDS))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS + extra)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline radar-only solves of a recorded run")
    parser.add_argument("run_dir")
    parser.add_argument("--experiment", help="Experiment file whose 'solver' block to use")
    parser.add_argument(
        "--receiver",
        choices=RECEIVER_MODES,
        default="stationary",
        help="Receiver position given to the solver: registered (stationary), true each tick (geometric), "
        "or true each tick with its own Doppler removed (geometric_doppler)",
    )
    args = parser.parse_args()
    solver_cfg = None
    if args.experiment:
        with open(args.experiment, encoding="utf-8") as f:
            solver_cfg = json.load(f).get("solver")
    rows = solve_run(args.run_dir, solver_cfg, args.receiver)
    name = "offline_solves.csv" if args.receiver == "stationary" else f"offline_solves_{args.receiver}.csv"
    path = write_rows(args.run_dir, rows, name)
    pub = [r for r in rows if r.get("published")]
    print(f"{len(pub)}/{len(rows)} ticks published; wrote {path}")


if __name__ == "__main__":
    main()
