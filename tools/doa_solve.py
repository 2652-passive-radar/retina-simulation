"""System B of the direction-finding study: one tower's delay plus the array's
bearing, solved in closed form.

For each tick where the array's channel (``doa.node``, CHAN-DT by default)
detected the aircraft:

1. ``measure``: the array snapshot of the detection (``retina_simulation.array``,
   true aircraft direction, the run's errors) and its Bartlett bearing
   (``doa.py``, front-half search).  Or, for the requirement curve, the true
   bearing plus Gaussian noise of a given size.
2. ``locate``: the aircraft on the bistatic ellipse along that bearing,
       R = c * delay + L,   r_R = (R^2 - L^2) / (2 (R - d . t)),
   with L the receiver-tower distance, t the tower's position and d the unit
   vector along the bearing, all east/north/up of the receiver.  d needs the
   aircraft's elevation, which needs its distance and altitude, so the formula
   is iterated a few times at an assumed altitude (one delay and one bearing
   do not measure altitude).  For a line array the bearing is a cone angle;
   the same iteration turns it into an azimuth.

Everything is exact geometry on WGS84 (the simulator's own conversions), so a
noise-free bearing and delay at the true altitude land on the aircraft.
Velocity is not solved: one Doppler gives one component of it.
"""

import json
import math
import os

import doa as doa_mod
import numpy as np

from retina_simulation import array as arr
from retina_simulation.world import C_KM_US, _enu_to_lla, _lla_to_enu

WRONG_LOBE_DEG = 20.0


def _read_ndjson(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def node_config(run_dir: str, node_id: str) -> dict:
    with open(os.path.join(run_dir, "nodes.json"), encoding="utf-8") as f:
        return next(n for n in json.load(f) if n["node_id"] == node_id)


def detections(run_dir: str, node_id: str) -> list[dict]:
    """The aircraft's detections on one node, with the truth that tick."""
    node = node_config(run_dir, node_id)
    rx_alt_km = node["rx_alt_ft"] * 0.3048e-3
    truth = {t["t_ms"]: t["aircraft"][0] for t in _read_ndjson(os.path.join(run_dir, "truth.ndjson")) if t["aircraft"]}
    out = []
    for fr in _read_ndjson(os.path.join(run_dir, "frames.ndjson")):
        st = truth.get(fr["t_ms"])
        if fr["node_id"] != node_id or not st:
            continue
        for d in fr["detections"]:
            if d["object_id"] != st["id"]:
                continue
            e, n, u = _lla_to_enu(st["lat"], st["lon"], st["alt_m"] / 1000.0, node["rx_lat"], node["rx_lon"], rx_alt_km)
            rng = math.sqrt(e * e + n * n + u * u)
            out.append(
                {
                    "t_ms": fr["t_ms"],
                    "delay_us": d["delay_us"],
                    "delay_true_us": d["delay_true"],
                    "snr_db": d["snr_db"],
                    "true_lat": st["lat"],
                    "true_lon": st["lon"],
                    "true_alt_m": st["alt_m"],
                    "u": (e / rng, n / rng, u / rng),
                    "az_true_deg": math.degrees(math.atan2(e, n)) % 360.0,
                    "el_true_deg": math.degrees(math.asin(u / rng)),
                    "range_km": math.hypot(e, n),
                }
            )
    return out


def wavelength_m(node: dict) -> float:
    return arr.C_M_S / node["fc_hz"]


def measure(
    dets: list[dict],
    node: dict,
    cfg: arr.ArrayConfig,
    err: arr.ArrayErrors,
    run_key: tuple,
    gaussian_deg: float | None = None,
) -> np.ndarray:
    """Bearing (compass deg) of each detection: from the array model, or the
    true bearing plus N(0, gaussian_deg) noise when that is given."""
    if not dets:
        return np.zeros(0)
    if gaussian_deg is not None:
        z = arr.keyed_rng("doa-gauss", *run_key).standard_normal(len(dets))
        return (np.array([d["az_true_deg"] for d in dets]) + gaussian_deg * z) % 360.0
    lam, boresight = wavelength_m(node), node["beam_azimuth_deg"]
    u = np.array([d["u"] for d in dets])
    snr = np.array([d["snr_db"] for d in dets])
    x = arr.snapshots(u, snr, cfg, err, boresight, lam, run_key)
    est, _ = doa_mod.doa(x, arr.positions(cfg, boresight, lam), lam, doa_mod.front_half(boresight))
    return est


def locate(node: dict, delay_us: float, bearing_deg: float, alt_km: float, cone_axis_deg: float | None = None):
    """(lat, lon) on the bistatic ellipse along the bearing at the assumed
    altitude, or None when the bearing never meets the ellipse.
    ``cone_axis_deg``: for a line array, its boresight (the bearing is then a
    cone angle about the line's axis)."""
    rx_alt_km = node["rx_alt_ft"] * 0.3048e-3
    t = np.array(
        _lla_to_enu(
            node["tx_lat"], node["tx_lon"], node["tx_alt_ft"] * 0.3048e-3, node["rx_lat"], node["rx_lon"], rx_alt_km
        )
    )
    big_l = float(np.linalg.norm(t))
    big_r = delay_us * C_KM_US + big_l
    up, r_h, az = alt_km - rx_alt_km, None, bearing_deg
    lat = lon = None
    for _ in range(8):
        el = math.atan2(up, r_h) if r_h else 0.0
        if cone_axis_deg is not None:
            s = math.sin(math.radians(bearing_deg - cone_axis_deg)) / math.cos(el)
            az = cone_axis_deg + math.degrees(math.asin(max(-1.0, min(1.0, s))))
        d = np.array(
            [
                math.cos(el) * math.sin(math.radians(az)),
                math.cos(el) * math.cos(math.radians(az)),
                math.sin(el),
            ]
        )
        den = 2.0 * (big_r - float(d @ t))
        if den <= 0:
            return None
        r = (big_r * big_r - big_l * big_l) / den
        if r <= 0:
            return None
        p = r * d
        lat, lon, alt = _enu_to_lla(*p, node["rx_lat"], node["rx_lon"], rx_alt_km)
        up += alt_km - alt
        r_h = r * math.cos(el)
    return lat, lon


def _en_m(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    north = math.radians(lat - lat0) * 6_371_000.0
    east = math.radians(lon - lon0) * 6_371_000.0 * math.cos(math.radians(lat0))
    return east, north


def solve(
    dets: list[dict], bearings: np.ndarray, node: dict, layout: str = "triangle", alt_km: float | None = 2.0
) -> list[dict]:
    """One row per detection: the solve and its errors.  ``alt_km`` None uses
    the true altitude (the best case)."""
    cone = node["beam_azimuth_deg"] if layout == "line" else None
    rows = []
    for d, b in zip(dets, bearings):
        alt = d["true_alt_m"] / 1000.0 if alt_km is None else alt_km
        berr = (float(b) - d["az_true_deg"] + 180.0) % 360.0 - 180.0
        row = {"t_ms": d["t_ms"], "range_km": round(d["range_km"], 3), "snr_db": d["snr_db"], "bearing_err_deg": berr}
        fix = locate(node, d["delay_us"], float(b), alt, cone)
        if fix is not None:
            e, n = _en_m(d["true_lat"], d["true_lon"], *fix)
            az = math.radians(d["az_true_deg"])
            row.update(
                err_horiz_m=math.hypot(e, n),
                err_cross_m=e * math.cos(az) - n * math.sin(az),
                err_along_m=e * math.sin(az) + n * math.cos(az),
            )
        rows.append(row)
    return rows


def metrics(rows: list[dict], n_ticks: int) -> dict:
    """Position RMSE, median and 95th percentile over solved ticks, share of
    ticks solved, share of bearings on a wrong lobe, bearing RMS error."""
    err = np.array([r["err_horiz_m"] for r in rows if "err_horiz_m" in r])
    be = np.array([r["bearing_err_deg"] for r in rows])
    return {
        "solves": int(len(err)),
        "pos_rmse_m": float(np.sqrt(np.mean(err**2))) if len(err) else None,
        "pos_median_m": float(np.median(err)) if len(err) else None,
        "pos_p95_m": float(np.percentile(err, 95)) if len(err) else None,
        "availability": len(err) / n_ticks if n_ticks else None,
        "wrong_lobe_rate": float(np.mean(np.abs(be) > WRONG_LOBE_DEG)) if len(be) else None,
        "bearing_rms_deg": float(np.sqrt(np.mean(be**2))) if len(be) else None,
    }
