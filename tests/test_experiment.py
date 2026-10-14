"""Reproducible experiments: seeded worlds, WGS84 geometry, scripted flights,
fast recording, and the frozen baseline experiment.

The last test is the one that matters for the study: every run of
experiments/baseline_static.json must reproduce the recording its stored
fingerprint describes, so a change to the simulator that would silently move
the baseline fails here first.
"""

import json
import math
import os
import random

import pytest

from retina_simulation import experiment
from retina_simulation.scripted import (
    Coverage,
    FlightPlan,
    aircraft_for,
    path_covered,
    propagate,
    sample_plan,
)
from retina_simulation.world import (
    SimulationWorld,
    _advance_latlon,
    _enu_to_lla,
    _lla_to_enu,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASELINE = os.path.join(_HERE, "..", "experiments", "baseline_static.json")


def _baseline():
    return experiment.load(_BASELINE)


def _nodes():
    return [experiment.node_config(c) for c in experiment.fleet_nodes(_baseline())]


def _world(seed, plans):
    world = SimulationWorld(center_lat=49.266, center_lon=-123.25)
    for n in _nodes():
        world.add_node(n)
    world.min_aircraft = world.max_aircraft = 0
    world.rng_seed = seed
    world.aircraft = [aircraft_for(p, i) for i, p in enumerate(plans, start=1)]
    return world


def _frames(world, ticks=20):
    out = []
    for _ in range(ticks):
        world.step(1.0)
        for nid in sorted(world.nodes):
            labels = []
            frame = world.generate_detections_for_node(nid, 0, labels=labels)
            out.append((nid, frame, labels))
    return out


_PLAN_A = FlightPlan(lat=49.15, lon=-123.05, alt_m=2000.0, speed_ms=90.0, heading_deg=200.0)
_PLAN_B = FlightPlan(lat=49.10, lon=-122.95, alt_m=2500.0, speed_ms=100.0, heading_deg=250.0)


# ── seeded randomness ────────────────────────────────────────────────────────


def test_same_seed_same_frames():
    assert _frames(_world("s1", [_PLAN_A])) == _frames(_world("s1", [_PLAN_A]))


def test_different_seed_different_frames():
    assert _frames(_world("s1", [_PLAN_A])) != _frames(_world("s2", [_PLAN_A]))


def test_one_aircraft_noise_does_not_depend_on_another():
    """Keyed streams: adding a second aircraft leaves the first one's
    detections exactly as they were."""

    def first_aircraft(frames):
        return [
            (nid, d, dp, lab["delay_true"])
            for nid, fr, labels in frames
            for d, dp, lab in zip(fr["delay"], fr["doppler"], labels)
            if lab["object_id"] == "flight-1"
        ]

    alone = first_aircraft(_frames(_world("s1", [_PLAN_A])))
    paired = first_aircraft(_frames(_world("s1", [_PLAN_A, _PLAN_B])))
    assert alone and alone == paired


def test_unseeded_world_draws_from_global_random():
    random.seed(11)
    a = _frames(_world(None, [_PLAN_A]), ticks=5)
    random.seed(11)
    b = _frames(_world(None, [_PLAN_A]), ticks=5)
    assert a == b


# ── WGS84 geometry ───────────────────────────────────────────────────────────


def test_enu_round_trip():
    enu = _lla_to_enu(49.1, -123.0, 2.0, 49.266028, -123.251806, 0.097)
    lat, lon, alt = _enu_to_lla(*enu, 49.266028, -123.251806, 0.097)
    assert lat == pytest.approx(49.1, abs=1e-9)
    assert lon == pytest.approx(-123.0, abs=1e-9)
    assert alt == pytest.approx(2.0, abs=1e-6)


def test_enu_is_on_the_ellipsoid_not_a_sphere():
    """One degree of latitude at 49 N is 111.2 km on WGS84 (meridian radius
    ~6372.5 km); the old spherical frame made it 111.19 km at any latitude and
    put east-west distances 0.3% short."""
    _, north, _ = _lla_to_enu(50.0, -123.0, 0.0, 49.0, -123.0, 0.0)
    east, _, _ = _lla_to_enu(49.0, -122.0, 0.0, 49.0, -123.0, 0.0)
    assert north == pytest.approx(111.25, abs=0.05)
    assert east == pytest.approx(73.17, abs=0.05)


def test_distance_flown_matches_velocity():
    lat, lon = _advance_latlon(49.1, -123.0, 2.0, 0.06, 0.08, 10.0)
    e, n, _ = _lla_to_enu(lat, lon, 2.0, 49.1, -123.0, 2.0)
    assert e == pytest.approx(0.6, rel=1e-3)
    assert n == pytest.approx(0.8, rel=1e-3)


# ── scripted flights ─────────────────────────────────────────────────────────


def test_turn_rate_turns_the_aircraft():
    world = _world(
        "s1", [FlightPlan(lat=49.15, lon=-123.05, alt_m=2000.0, speed_ms=90, heading_deg=0, turn_rate_deg_s=1.0)]
    )
    for _ in range(30):
        world.step(1.0)
    assert world.aircraft[0].heading_deg == pytest.approx(30.0)


def test_world_flies_the_plan_propagate_predicts():
    plan = FlightPlan(lat=49.15, lon=-123.05, alt_m=2000.0, speed_ms=90, heading_deg=120, turn_rate_deg_s=0.5)
    world = _world("s1", [plan])
    for _ in range(300):
        world.step(1.0)
    lat, lon, _ = propagate(plan, 300)[-1]
    e, n, _ = _lla_to_enu(world.aircraft[0].lat, world.aircraft[0].lon, 2.0, lat, lon, 2.0)
    assert math.hypot(e, n) < 0.1  # only the per-tick velocity jitter differs


@pytest.mark.parametrize("kind", ["straight", "turn"])
def test_sampled_plan_is_covered_and_repeatable(kind):
    nodes = _nodes()
    plan = sample_plan(7, kind, nodes, 300)
    assert plan == sample_plan(7, kind, nodes, 300)
    assert path_covered(propagate(plan, 300), nodes, Coverage())
    assert (plan.turn_rate_deg_s == 0.0) == (kind == "straight")


def test_unknown_flight_kind_is_refused():
    with pytest.raises(ValueError):
        sample_plan(7, "loop", _nodes(), 300)


# ── fingerprints and the frozen baseline ─────────────────────────────────────


def test_fingerprint_tolerates_rounding_but_not_a_different_run():
    fp = {"frames": 900, "detections": 1980, "sum_delay_us": 93848.6, "end_lat": 49.056746}
    assert experiment.fingerprint_mismatch(fp, dict(fp, sum_delay_us=93848.62)) == []
    assert experiment.fingerprint_mismatch(fp, dict(fp, detections=1981))
    assert experiment.fingerprint_mismatch(fp, dict(fp, sum_delay_us=93850.0))


def test_baseline_experiment_reproduces_its_recordings(tmp_path):
    exp = _baseline()
    assert [r["seed"] for r in exp["runs"]] == [101, 202, 303, 404, 505]
    assert [r["flight"] for r in exp["runs"]] == ["straight", "straight", "turn", "turn", "turn"]
    nodes = _nodes()
    for run in exp["runs"]:
        plan = FlightPlan(**run["plan"])
        assert path_covered(propagate(plan, exp["duration_s"]), nodes, experiment.coverage(exp))
        got = experiment.record_run(exp, run, str(tmp_path / str(run["seed"])))
        assert experiment.fingerprint_mismatch(run["fingerprint"], got) == [], run["seed"]


def test_recording_carries_receiver_pose(tmp_path):
    exp = _baseline()
    experiment.record_run(exp, exp["runs"][0], str(tmp_path))
    with open(tmp_path / "rx_trajectory.ndjson", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == exp["duration_s"]
    pose = lines[0]["rx"]["synth-ubc-cisc"]
    assert (pose["vel_east_ms"], pose["vel_north_ms"]) == (0.0, 0.0)
    assert lines[0]["t_ms"] == exp["start_time_ms"] + 1000
