"""Receiver Doppler compensation (GD) in tools/offline_solve.py.

Needs retina_geolocator, which this repo's CI does not install, so the module
skips there; it runs wherever the backend's environment is used.
"""

import json
import math
import os
import sys

import pytest

pytest.importorskip("retina_geolocator")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "tools"))

import offline_solve  # noqa: E402

from retina_simulation import experiment  # noqa: E402
from retina_simulation.motion import Motion  # noqa: E402
from retina_simulation.world import _lla_to_enu, _receiver_doppler  # noqa: E402

_STUDY = os.path.join(_HERE, "..", "experiments", "receiver_doppler.json")
_VEHICLE_EAST = Motion("linear", speed_ms=15.0, heading_deg=90.0)


def _rmse(values):
    return math.sqrt(sum(v * v for v in values) / len(values))


def _record(tmp_path, motion, name):
    exp = experiment.load(_STUDY)
    d = str(tmp_path / name)
    experiment.record_run(exp, exp["runs"][1], d, motion, rx_doppler=True)
    return d


def _noise_free(run_dir):
    """The recording with every aircraft detection replaced by its true delay and Doppler."""
    path = os.path.join(run_dir, "frames.ndjson")
    with open(path, encoding="utf-8") as f:
        frames = [json.loads(line) for line in f]
    for fr in frames:
        for d in fr["detections"]:
            if not d["is_clutter"]:
                d["delay_us"], d["doppler_hz"] = d["delay_true"], d["doppler_true"]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(fr) + "\n" for fr in frames)
    return run_dir


def _solved(rows):
    return [r for r in rows if "err_horiz_m" in r]


@pytest.mark.parametrize(
    "lat,lon,alt_m,vel",
    [(49.10, -123.05, 1800.0, (15.0, 0.0, 0.0)), (49.35, -123.40, 9000.0, (-3.0, 11.0, 0.5))],
)
def test_correction_is_the_doppler_the_simulator_adds(lat, lon, alt_m, vel):
    rx = (49.266028, -123.251806, 97.0)
    tx = (49.345, -122.972778, 617.0)
    fc = 521e6
    target = _lla_to_enu(lat, lon, alt_m / 1000, rx[0], rx[1], rx[2] / 1000)
    tower = _lla_to_enu(tx[0], tx[1], tx[2] / 1000, rx[0], rx[1], rx[2] / 1000)
    want = _receiver_doppler(target, tower, tuple(v / 1000 for v in vel), fc)
    vel_ecef = offline_solve._enu_to_ecef_vec(rx[0], rx[1], *vel)
    got = offline_solve.receiver_doppler_hz(
        offline_solve._ecef_m(lat, lon, alt_m), offline_solve._ecef_m(*rx), offline_solve._ecef_m(*tx), vel_ecef, fc
    )
    assert abs(want) > 1.0
    assert got == pytest.approx(want, rel=1e-6)


def test_static_receiver_gd_equals_g(tmp_path):
    d = _record(tmp_path, Motion(), "static")
    g, gd = offline_solve.solve_run(d, receiver="geometric"), offline_solve.solve_run(d, receiver="geometric_doppler")
    keys = ("published", "reason", "est_lat", "est_lon", "est_alt_m", "est_vel_east_ms", "est_vel_north_ms")
    assert [{k: r.get(k) for k in keys} for r in g] == [{k: r.get(k) for k in keys} for r in gd]
    assert all(r[f"rxdop_corr_hz_{c}"] == 0.0 for r in _solved(gd) for c in ("chan", "chek", "cisc"))


def test_gd_removes_the_receiver_doppler_noise_free(tmp_path):
    """Noise-free measurements from a receiver driving east at 15 m/s: G's
    velocity carries the receiver's Doppler, GD's does not, and both place
    the aircraft identically."""
    d = _noise_free(_record(tmp_path, _VEHICLE_EAST, "vehicle"))
    g = _solved(offline_solve.solve_run(d, receiver="geometric"))
    gd = _solved(offline_solve.solve_run(d, receiver="geometric_doppler"))
    assert _rmse([r["verr_horiz_ms"] for r in g]) > 3.0
    assert _rmse([r["verr_horiz_ms"] for r in gd]) < 0.5
    assert _rmse([r["err_horiz_m"] for r in gd]) == pytest.approx(_rmse([r["err_horiz_m"] for r in g]), abs=1.0)


def test_correction_matches_the_true_receiver_doppler(tmp_path):
    d = _record(tmp_path, _VEHICLE_EAST, "vehicle")
    rows = _solved(offline_solve.solve_run(d, receiver="geometric_doppler"))
    errs = [r[f"rxdop_corr_hz_{c}"] - r[f"rxdop_true_hz_{c}"] for r in rows for c in ("chan", "chek", "cisc")]
    assert _rmse(errs) < 0.2
    assert max(abs(r[f"rxdop_true_hz_{c}"]) for r in rows for c in ("chan", "chek", "cisc")) > 10.0


def test_one_correction_pass_is_enough(tmp_path):
    d = _record(tmp_path, _VEHICLE_EAST, "vehicle")
    one = _solved(offline_solve.solve_run(d, {"rx_doppler_passes": 1}, "geometric_doppler"))
    two = _solved(offline_solve.solve_run(d, {"rx_doppler_passes": 2}, "geometric_doppler"))
    dv = [
        math.hypot(a["est_vel_east_ms"] - b["est_vel_east_ms"], a["est_vel_north_ms"] - b["est_vel_north_ms"])
        for a, b in zip(one, two)
    ]
    assert [r["t_ms"] for r in one] == [r["t_ms"] for r in two]
    assert _rmse(dv) < 0.1


def test_gd_is_deterministic(tmp_path):
    d = _record(tmp_path, _VEHICLE_EAST, "vehicle")
    assert offline_solve.solve_run(d, receiver="geometric_doppler") == offline_solve.solve_run(
        d, receiver="geometric_doppler"
    )
