"""Opt-in radar-equation SNR model, helicopter hover / piston orbit behaviours, and the default-run guard."""

import hashlib
import json
import math
import random

import pytest

from retina_simulation import micro_doppler
from retina_simulation.world import (
    SNR_CAP_DB,
    SNR_REF_DB,
    NodeConfig,
    SimulatedAircraft,
    SimulationWorld,
    _bistatic_delay,
    _lla_to_enu,
    radar_snr_db,
)


def _default_run_digest() -> str:
    random.seed(2024)
    world = SimulationWorld()
    world.frac_drone = 0.15
    world.frac_dark = 0.25
    world.frac_anomalous = 0.1
    world.add_node(NodeConfig(node_id="n1", max_bistatic_range_km=200.0, beam_width_deg=360.0))
    out = []
    for i in range(240):
        world.step(0.5, mode="adsb")
        out.append(world.generate_all_frames(i * 500))
        out.append(world.generate_detections_for_node("n1", i * 500))
    return hashlib.sha256(json.dumps(out, sort_keys=True, default=str).encode()).hexdigest()


def test_default_seeded_run_is_byte_identical_to_before_snr_model_and_flight_behaviours():
    assert _default_run_digest() == "f7c268430b435725afa32763cbf0cc9037cff552f7c6e6b62dcfa65d51d2e23a"


# ── radar-equation SNR ───────────────────────────────────────────────────────

_RX_LAT, _RX_LON = 34.85, -82.39
_KM_PER_DEG_LON = 111.32 * math.cos(math.radians(_RX_LAT))


def _node(baseline_km=12.0):
    return NodeConfig(
        node_id="snr-node",
        rx_lat=_RX_LAT,
        rx_lon=_RX_LON,
        rx_alt_ft=0.0,
        tx_lat=_RX_LAT,
        tx_lon=_RX_LON + baseline_km / _KM_PER_DEG_LON,
        tx_alt_ft=0.0,
        beam_azimuth_deg=0.0,
        beam_width_deg=360.0,
        max_range_km=100.0,
        doppler_min=-1000.0,
        doppler_max=1000.0,
    )


def _equidistant_aircraft(airframe, range_km, baseline_km=12.0, heading_deg=90.0, speed_km_s=0.14):
    """An aircraft range_km from both TX and RX: on the baseline's perpendicular bisector, level with them."""
    north_km = math.sqrt(range_km**2 - (baseline_km / 2) ** 2)
    heading = math.radians(heading_deg)
    return SimulatedAircraft(
        object_id="ac-1",
        lat=_RX_LAT + north_km / 111.32,
        lon=_RX_LON + (baseline_km / 2) / _KM_PER_DEG_LON,
        alt_km=0.0,
        vel_east=speed_km_s * math.sin(heading),
        vel_north=speed_km_s * math.cos(heading),
        vel_up=0.0,
        heading_deg=heading_deg,
        speed_km_s=speed_km_s,
        airframe=airframe,
        rotor_phases=[0.3, 1.1][: len(micro_doppler.AIRFRAMES[airframe])],
    )


@pytest.fixture
def no_noise(monkeypatch):
    monkeypatch.setattr(random, "gauss", lambda mu, sigma: 0.0)


def test_reference_point_reads_the_reference_snr():
    assert radar_snr_db(10.0, 30.0, 30.0) == pytest.approx(SNR_REF_DB)


def test_bigger_rcs_gives_higher_snr_at_the_same_geometry():
    assert radar_snr_db(20.0, 25.0, 18.0) > radar_snr_db(2.0, 25.0, 18.0) > radar_snr_db(0.02, 25.0, 18.0)
    assert radar_snr_db(20.0, 25.0, 18.0) - radar_snr_db(2.0, 25.0, 18.0) == pytest.approx(10.0)


def test_doubling_both_ranges_lowers_snr_by_about_12_db():
    assert radar_snr_db(5.0, 20.0, 15.0) - radar_snr_db(5.0, 40.0, 30.0) == pytest.approx(12.04, abs=0.01)


def test_radar_snr_orders_airframes_by_cross_section(no_noise):
    world = SimulationWorld()
    world.snr_model = "radar"
    snr = {}
    for airframe in ("jet", "turboprop", "helicopter", "piston", "drone"):
        ac = _equidistant_aircraft(airframe, 20.0)
        tx = _lla_to_enu(_node().tx_lat, _node().tx_lon, 0.0, _RX_LAT, _RX_LON, 0.0)
        snr[airframe] = world._body_snr(ac, _lla_to_enu(ac.lat, ac.lon, 0.0, _RX_LAT, _RX_LON, 0.0), tx)
    assert snr["jet"] > snr["turboprop"] > snr["helicopter"] > snr["piston"] > snr["drone"]


def test_radar_snr_is_capped_and_floored(no_noise):
    world = SimulationWorld()
    world.snr_model = "radar"
    tx = (12.0, 0.0, 0.0)
    ac = _equidistant_aircraft("jet", 6.0, heading_deg=0.0)
    on_top = (0.0, 0.01, 0.0)
    assert world._body_snr(ac, on_top, (0.0, 0.02, 0.0)) == SNR_CAP_DB
    ac.airframe = "drone"
    assert world._body_snr(ac, (400.0, 0.0, 0.0), tx) == 4.0


def test_legacy_snr_is_unchanged():
    world = SimulationWorld()
    assert world.snr_model == "legacy"
    ac = _equidistant_aircraft("drone", 20.0)
    target = _lla_to_enu(ac.lat, ac.lon, 0.0, _RX_LAT, _RX_LON, 0.0)
    random.seed(9)
    got = world._body_snr(ac, target, (12.0, 0.0, 0.0))
    random.seed(9)
    dist = math.sqrt(sum(x * x for x in target))
    assert got == max(25 - 10 * math.log10(max(dist, 1)) + random.gauss(0, 2), 4.0)


def _lines_per_frame(model, airframe, range_km, frames=100):
    random.seed(1)
    world = SimulationWorld()
    world.add_node(_node())
    world.snr_model = model
    world.micro_doppler_enabled = True
    ac = _equidistant_aircraft(airframe, range_km)
    world.aircraft = [ac]
    node = world.nodes["snr-node"]
    target = _lla_to_enu(ac.lat, ac.lon, ac.alt_km, _RX_LAT, _RX_LON, 0.0)
    tx = _lla_to_enu(node.tx_lat, node.tx_lon, 0.0, _RX_LAT, _RX_LON, 0.0)
    body_delay = _bistatic_delay(target, tx, (0.0, 0.0, 0.0))
    total = 0
    for i in range(frames):
        world._time = 12.0 + 0.5 * i
        frame = world.generate_detections_for_node("snr-node", i)
        total += sum(1 for d in frame["delay"] if abs(d - body_delay) < 0.6) - 1
    return total / frames


def test_radar_snr_lets_a_turboprop_show_comb_lines_at_moderate_range():
    assert _lines_per_frame("radar", "turboprop", 20.0) >= 3.0
    assert _lines_per_frame("legacy", "turboprop", 20.0) < 0.5


# ── helicopter hover and piston orbit ────────────────────────────────────────


def _lone_aircraft(airframe, seed):
    """A spawned aircraft of this airframe flying a long straight leg, alone in the world."""
    random.seed(seed)
    world = SimulationWorld()
    world.airframe_mix = {"jet": 0.5, airframe: 0.5}
    ac = world._spawn_aircraft()
    while ac.airframe != airframe:
        ac = world._spawn_aircraft()
    ac.waypoints = [(ac.lat, ac.lon), (ac.lat + 4.0, ac.lon)]
    ac.waypoint_idx = 1
    ac.heading_deg = 0.0
    ac.vel_up = 0.0
    world.aircraft = [ac]
    return world, ac


def _fly(world, ac, seconds, dt=0.5):
    for _ in range(int(seconds / dt)):
        world._time += dt
        ac.sim_now_s = world._time
        world._update_aircraft(ac, dt)
        world._enforce_separation(dt)
        yield world._time


def test_helicopter_hovers_then_recovers_its_cruise_speed():
    world, ac = _lone_aircraft("helicopter", seed=4)
    base = ac.base_speed_km_s
    slow_since = None
    hovered_for = 0.0
    recovered = False
    for t in _fly(world, ac, 1200):
        if ac.speed_km_s < 0.003:
            slow_since = t if slow_since is None else slow_since
            hovered_for = max(hovered_for, t - slow_since)
            assert math.hypot(ac.vel_east, ac.vel_north) < 0.004
        else:
            slow_since = None
            if hovered_for >= 20.0 and ac.speed_km_s >= 0.9 * base:
                recovered = True
                break
    assert hovered_for >= 20.0
    assert recovered


def test_hovering_helicopter_does_not_reach_its_waypoint_and_resumes_after():
    world, ac = _lone_aircraft("helicopter", seed=4)
    ac.waypoints = [(ac.lat, ac.lon), (ac.lat + 5.1 / 111.32, ac.lon), (ac.lat + 4.0, ac.lon)]
    ac.maneuver_next_at = world._time
    ac.maneuver_phase = "cruise"
    hovered = False
    for _ in _fly(world, ac, 90):
        if ac.maneuver_phase != "cruise":
            hovered = True
            assert ac.waypoint_idx == 1
    assert hovered
    assert ac.maneuver_phase == "cruise"
    assert ac.waypoint_idx == 2
    assert ac.speed_km_s == pytest.approx(ac.base_speed_km_s)


def test_piston_flies_orbits_at_a_constant_three_to_six_degrees_per_second():
    world, ac = _lone_aircraft("piston", seed=6)
    dt = 0.5
    prev = ac.heading_deg
    run = 0.0
    longest = 0.0
    for _ in _fly(world, ac, 1500, dt):
        rate = abs((ac.heading_deg - prev + 180) % 360 - 180) / dt
        prev = ac.heading_deg
        run = run + dt if 2.9 <= rate <= 6.1 else 0.0
        longest = max(longest, run)
    assert longest >= 60.0


def test_orbit_hands_back_to_waypoint_navigation():
    world, ac = _lone_aircraft("piston", seed=6)
    ac.maneuver_next_at = world._time
    for _ in _fly(world, ac, 400):
        pass
    assert ac.maneuver_phase == "cruise"
    # The leg runs due north, so a resumed route means heading is back near 0/360.
    assert abs((ac.heading_deg + 180) % 360 - 180) < 20.0


def test_other_airframes_never_get_a_maneuver_schedule():
    random.seed(8)
    world = SimulationWorld()
    world.airframe_mix = {"jet": 0.3, "turboprop": 0.3, "piston": 0.2, "helicopter": 0.2}
    world.frac_drone = 0.2
    world.frac_anomalous = 0.1
    spawned = [world._spawn_aircraft() for _ in range(300)]
    for ac in spawned:
        expected = {"helicopter": "hover", "piston": "orbit"}.get(ac.airframe)
        assert ac.maneuver_kind == expected
