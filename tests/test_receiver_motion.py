"""Receiver platform motion (retina_simulation/motion.py) and what it does to
the simulated measurements.

The Doppler test is the one the rest of the study leans on: the measured
Doppler of a moving receiver must equal -(1/lambda) times the rate of change
of the bistatic range, with the aircraft and the receiver both moving.
"""

import math
import os

import pytest

from retina_simulation import experiment
from retina_simulation.motion import Motion, motion_from
from retina_simulation.world import C_KM_S, _bistatic_doppler, _receiver_doppler

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY = os.path.join(_HERE, "..", "experiments", "receiver_motion.json")

_PROFILES = [
    Motion("linear", speed_ms=15.0, heading_deg=90.0),
    Motion("linear", speed_ms=1.5, heading_deg=0.0),
    Motion("oscillate", speed_ms=15.0, heading_deg=90.0, period_s=100.0),
    Motion("circle", speed_ms=15.0, lap_s=300.0, start_bearing_deg=0.0),
]


@pytest.mark.parametrize("m", _PROFILES, ids=lambda m: m.kind)
def test_velocity_is_the_derivative_of_the_offset(m):
    h = 1e-3
    for t in (0.0, 37.0, 150.0, 299.0):
        e0, n0, _, _ = m.state(t - h)
        e1, n1, _, _ = m.state(t + h)
        _, _, ve, vn = m.state(t)
        assert (e1 - e0) / (2 * h) == pytest.approx(ve, abs=1e-4)
        assert (n1 - n0) / (2 * h) == pytest.approx(vn, abs=1e-4)


@pytest.mark.parametrize("m", _PROFILES, ids=lambda m: m.kind)
def test_every_profile_moves_at_its_speed_at_peak(m):
    speeds = [math.hypot(*m.state(t)[2:]) for t in range(301)]
    assert max(speeds) == pytest.approx(m.speed_ms, rel=1e-3)


def test_linear_covers_speed_times_time():
    e, n, _, _ = Motion("linear", speed_ms=15.0, heading_deg=90.0).state(300.0)
    assert (e, n) == pytest.approx((4500.0, 0.0), abs=1e-6)


def test_circle_stays_at_its_radius_and_closes_in_one_lap():
    m = Motion("circle", speed_ms=15.0, lap_s=300.0)
    assert m.radius_m == pytest.approx(716.2, abs=0.1)
    assert all(math.hypot(*m.state(t)[:2]) == pytest.approx(m.radius_m) for t in range(0, 301, 10))
    assert m.state(300.0)[:2] == pytest.approx(m.state(0.0)[:2], abs=1e-6)


def test_oscillate_amplitude_and_zero_speed_at_turnarounds():
    m = Motion("oscillate", speed_ms=15.0, heading_deg=90.0, period_s=100.0)
    assert m.amplitude_m == pytest.approx(238.7, abs=0.1)
    e, _, ve, _ = m.state(25.0)
    assert (e, ve) == pytest.approx((m.amplitude_m, 0.0), abs=1e-9)


def test_unknown_motion_is_refused():
    with pytest.raises(ValueError):
        motion_from({"type": "spiral"})


def _bistatic_range(p, r, t):
    return math.dist(p, t) + math.dist(p, r) - math.dist(r, t)


def test_receiver_doppler_matches_the_rate_of_change_of_bistatic_range():
    """Fixed frame, everything in km and km/s; the world's two Doppler terms
    take positions relative to the receiver, as generate_detections_for_node
    passes them."""
    f = 521e6
    p0, vp = (8.0, -12.0, 2.0), (0.05, 0.08, 0.0)
    r0, vr = (0.3, -0.2, 0.1), (0.012, -0.009, 0.0)
    tx = (16.0, 9.0, 0.6)
    rel_p = tuple(p0[i] - r0[i] for i in range(3))
    rel_t = tuple(tx[i] - r0[i] for i in range(3))
    doppler = _bistatic_doppler(rel_p, vp, rel_t, (0.0, 0.0, 0.0), f) + _receiver_doppler(rel_p, rel_t, vr, f)
    h = 1e-4
    rb = [
        _bistatic_range([p0[i] + vp[i] * dt for i in range(3)], [r0[i] + vr[i] * dt for i in range(3)], tx)
        for dt in (-h, h)
    ]
    expected = -(f / C_KM_S) * (rb[1] - rb[0]) / (2 * h)
    assert doppler == pytest.approx(expected, rel=1e-6)
    assert abs(_receiver_doppler(rel_p, rel_t, vr, f)) > 1.0


def test_study_inherits_the_baseline_flights():
    study = experiment.load(_STUDY)
    base = experiment.load(os.path.join(_HERE, "..", "experiments", "baseline_static.json"))
    assert [r["plan"] for r in study["runs"]] == [r["plan"] for r in base["runs"]]
    assert study["fleet"] == base["fleet"] and experiment.fleet_path(study) == experiment.fleet_path(base)
    assert {c["id"] for c in study["cases"]} >= {"static", "linear-east-vehicle", "circle-walk"}


def test_static_case_reproduces_step_0_exactly(tmp_path):
    study = experiment.load(_STUDY)
    static = motion_from(next(c["motion"] for c in study["cases"] if c["id"] == "static"))
    for run in study["runs"][:2]:
        got = experiment.record_run(study, run, str(tmp_path / str(run["seed"])), static)
        assert experiment.fingerprint_mismatch(run["fingerprint"], got) == []


def test_moving_receiver_changes_the_recording_and_logs_its_pose(tmp_path):
    study = experiment.load(_STUDY)
    run = study["runs"][0]
    m = Motion("linear", speed_ms=15.0, heading_deg=90.0)
    fp = experiment.record_run(study, run, str(tmp_path), m)
    assert experiment.fingerprint_mismatch(run["fingerprint"], fp)
    import json

    with open(tmp_path / "rx_trajectory.ndjson", encoding="utf-8") as f:
        last = json.loads(f.readlines()[-1])["rx"]["synth-ubc-cisc"]
    assert last["offset_east_m"] == pytest.approx(4500.0, abs=0.01)
    assert last["vel_east_ms"] == pytest.approx(15.0)


def test_rx_doppler_switch_only_changes_doppler(tmp_path):
    study = experiment.load(_STUDY)
    run = study["runs"][0]
    m = Motion("linear", speed_ms=15.0, heading_deg=90.0)
    on = experiment.record_run(study, run, str(tmp_path / "on"), m, rx_doppler=True)
    off = experiment.record_run(study, run, str(tmp_path / "off"), m, rx_doppler=False)
    assert on["sum_delay_us"] == off["sum_delay_us"]
    assert on["sum_doppler_hz"] != off["sum_doppler_hz"]
