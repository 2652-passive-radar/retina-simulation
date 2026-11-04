"""Rotor/propeller micro-Doppler: the comb physics, the world integration, and the CLI mix parser."""

import json
import math
import random

import numpy as np
import pytest

from retina_simulation import micro_doppler as md
from retina_simulation.micro_doppler import AIRFRAMES, Rotor, comb_lines
from retina_simulation.orchestrator import build_ground_truth_payload, parse_airframe_mix
from retina_simulation.world import NodeConfig, SimulatedAircraft, SimulationWorld

FC = 500e6
WAVELENGTH = md.C_M_S / FC
# A loud body, so the (physically weak) blade lines clear the detection floor and the geometry is what is tested.
LOUD = 70.0
U_TX, U_RX = md._unit_vectors(60.0)
B = U_TX + U_RX


def _lines(rotors, b=B, f_body=0.0, cpi=2.0, lo=-2000.0, hi=2000.0, snr=LOUD, max_lines=300, **kw):
    return comb_lines(rotors, b, 90.0, FC, f_body, cpi, lo, hi, snr, max_lines=max_lines, **kw)


def test_even_blade_count_spaces_lines_at_n_times_rev():
    freqs = np.sort([f for f, _ in _lines((Rotor(2, 5.0, 6.8, "up", 0.3),))])
    assert abs(np.median(np.diff(freqs)) - 2 * 6.8) < 1.0


def test_odd_blade_count_spaces_lines_at_n_times_rev():
    # A hub-to-tip wire is not symmetric under a half turn, so odd counts keep N*f
    # spacing; the 2*N*f of a centred rod does not apply to a blade rooted at the hub.
    freqs = np.sort([f for f, _ in _lines((Rotor(3, 5.0, 6.8, "up", 0.3),))])
    assert abs(np.median(np.diff(freqs)) - 3 * 6.8) < 1.0


def test_spread_is_bounded_by_tip_speed():
    rotor = Rotor(2, 5.0, 6.8, "up", 0.3)
    f_body = 120.0
    lines = _lines((rotor,), f_body=f_body)
    bound = 2 * math.pi * rotor.rev_per_s * rotor.radius_m * np.linalg.norm(B) / WAVELENGTH + 3 * 0.5
    assert lines
    assert max(abs(f - f_body) for f, _ in lines) <= bound


def test_forward_scatter_shows_no_lines():
    rotors = AIRFRAMES["helicopter"]
    assert _lines(rotors, b=U_TX + (-U_TX)) == []


def test_monostatic_like_geometry_shows_lines():
    assert _lines(AIRFRAMES["helicopter"], b=2 * U_TX)


def test_lines_stay_inside_the_doppler_window():
    lines = _lines(AIRFRAMES["helicopter"], f_body=50.0, lo=-300.0, hi=300.0)
    assert lines
    assert all(-300.0 <= f <= 300.0 for f, _ in lines)


def test_narrow_window_yields_fewer_lines():
    wide = _lines(AIRFRAMES["helicopter"], lo=-1000.0, hi=1000.0, cpi=0.5)
    narrow = _lines(AIRFRAMES["helicopter"], lo=-300.0, hi=300.0, cpi=0.5)
    assert 0 < len(narrow) < len(wide)


def test_lines_exclude_the_body_skirt():
    f_body = 37.3
    lines = _lines((Rotor(2, 5.0, 6.8, "up", 0.3),), f_body=f_body)
    assert all(abs(f - f_body) > 4 * 0.5 for f, _ in lines)


def test_max_lines_caps_and_orders_strongest_first():
    lines = _lines(AIRFRAMES["helicopter"], max_lines=5, cpi=0.5)
    assert len(lines) == 5
    snrs = [s for _, s in lines]
    assert snrs == sorted(snrs, reverse=True)


def test_jet_returns_nothing():
    assert _lines("jet") == []
    assert _lines(()) == []


def test_drone_at_default_amplitudes_shows_at_most_a_couple_of_lines():
    assert len(comb_lines("drone", B, 90.0, FC, 0.0, 0.5, -1000.0, 1000.0, 30.0)) <= 2


def test_line_snr_is_body_snr_plus_relative_level():
    low = dict(_lines((Rotor(2, 5.0, 6.8, "up", 0.3),), snr=60.0, max_lines=300))
    high = dict(_lines((Rotor(2, 5.0, 6.8, "up", 0.3),), snr=70.0, max_lines=300))
    common = set(low) & set(high)
    assert common
    assert all(abs((high[f] - low[f]) - 10.0) < 1e-6 for f in common)


def test_body_doppler_sign_follows_the_world_convention():
    t, s = md.slow_time((), B, 90.0, FC, 150.0, 1.0)
    freqs, power_db = md.line_spectrum(t, s, 8192.0)
    assert freqs[int(np.argmax(power_db))] == pytest.approx(150.0, abs=1.0)


def test_phases_advance_with_time_so_the_comb_evolves():
    rotors = (Rotor(2, 5.0, 6.8, "up", 0.3),)
    now = _lines(rotors, phases=[0.0], cpi=0.5)
    later = _lines(rotors, phases=md.advance_phases(rotors, [0.0], 0.0123), cpi=0.5)
    again = _lines(rotors, phases=md.advance_phases(rotors, [0.0], 0.0), cpi=0.5)
    assert again == now
    assert later != now


def test_spectrogram_shapes():
    t, s = md.slow_time(AIRFRAMES["helicopter"], B, 90.0, FC, 0.0, 0.5)
    times, freqs, power = md.spectrogram(t, s, 8192.0, 0.1)
    assert power.shape == (len(times), len(freqs))
    assert power.max() == pytest.approx(0.0)


def test_cli_prints_a_line_table(capsys):
    md.main(["--airframe", "turboprop", "--cpi", "0.5"])
    out = capsys.readouterr().out
    assert "offset_hz" in out and "blade-passing 102.0 Hz" in out


# ── World integration ────────────────────────────────────────────────────────

_RX_LAT, _RX_LON = 34.85, -82.39


def _node():
    return NodeConfig(
        node_id="md-node",
        rx_lat=_RX_LAT,
        rx_lon=_RX_LON,
        rx_alt_ft=0.0,
        tx_lat=_RX_LAT,
        tx_lon=_RX_LON + 0.4,
        tx_alt_ft=0.0,
        beam_azimuth_deg=0.0,
        beam_width_deg=360.0,
        max_range_km=50.0,
        doppler_min=-1000.0,
        doppler_max=1000.0,
    )


def _helicopter():
    return SimulatedAircraft(
        object_id="heli-1",
        lat=_RX_LAT + 0.01,
        lon=_RX_LON + 0.1,
        alt_km=0.5,
        vel_east=0.0,
        vel_north=0.04,
        vel_up=0.0,
        heading_deg=0.0,
        speed_km_s=0.04,
        airframe="helicopter",
        rotor_phases=[0.3, 1.1],
        has_adsb=True,
        adsb_hex="abc123",
    )


def _seeded_frames(configure=None, n=30):
    random.seed(11)
    world = SimulationWorld()
    world.frac_drone = 0.2
    world.frac_dark = 0.3
    world.add_node(NodeConfig(node_id="n1", max_bistatic_range_km=200.0, beam_width_deg=360.0))
    if configure:
        configure(world)
    frames = []
    for i in range(n):
        world.step(0.5, mode="adsb")
        frames.append(world.generate_all_frames(i * 500))
    return world, json.dumps(frames, sort_keys=True)


def test_default_world_is_deterministic_and_all_jet_or_drone():
    world, first = _seeded_frames()
    _, second = _seeded_frames()
    assert first == second
    assert world.micro_doppler_enabled is False
    assert world.aircraft
    assert {ac.airframe for ac in world.aircraft} <= {"jet", "drone"}
    assert all(ac.airframe == "drone" for ac in world.aircraft if ac.object_type == "drone")
    assert all(ac.rotor_phases == [] and ac.rpm_scale == 1.0 for ac in world.aircraft)


def test_enabling_micro_doppler_with_the_default_all_jet_mix_changes_nothing():
    _, baseline = _seeded_frames()
    _, enabled = _seeded_frames(lambda w: setattr(w, "micro_doppler_enabled", True))
    assert enabled == baseline


def test_mixed_airframes_spawn_with_phases_and_realistic_kinematics():
    random.seed(3)
    world = SimulationWorld()
    world.add_node(_node())
    world.airframe_mix = {"jet": 0.2, "turboprop": 0.2, "piston": 0.2, "helicopter": 0.4}
    spawned = [world._spawn_aircraft() for _ in range(200)]
    kinds = {ac.airframe for ac in spawned}
    assert kinds == {"jet", "turboprop", "piston", "helicopter"}
    for ac in spawned:
        rotors = AIRFRAMES[ac.airframe]
        assert len(ac.rotor_phases) == len(rotors)
        assert (0.97 <= ac.rpm_scale <= 1.03) if rotors else ac.rpm_scale == 1.0
        if ac.airframe == "helicopter":
            assert 0.2 <= ac.alt_km <= 1.5 and 0.030 <= ac.speed_km_s <= 0.070
        if ac.airframe == "turboprop":
            assert 3.0 <= ac.alt_km <= 7.5 and 0.120 <= ac.speed_km_s <= 0.170
        if ac.airframe == "piston":
            assert 0.5 <= ac.alt_km <= 3.0 and 0.040 <= ac.speed_km_s <= 0.070


def _frame_with_helicopter(enabled, seed=5):
    random.seed(seed)
    world = SimulationWorld()
    world.add_node(_node())
    world.aircraft = [_helicopter()]
    world.micro_doppler_enabled = enabled
    world._time = 12.0
    return world.generate_detections_for_node("md-node", 1000)


def _at_body_delay(frame):
    """Indices of detections sharing the body's delay (the body is always the first one appended)."""
    return [i for i, d in enumerate(frame["delay"]) if abs(d - frame["delay"][0]) < 0.6]


@pytest.fixture
def loud_helicopter(monkeypatch):
    # Placeholder amplitudes put every line below the detection floor at the
    # simulator's SNRs, so the world tests boost them to exercise the plumbing.
    monkeypatch.setitem(AIRFRAMES, "helicopter", (Rotor(2, 5.0, 6.8, "up", 10.0),))


def test_disabled_world_emits_no_comb_detections(loud_helicopter):
    for seed in range(10):
        random.seed(seed)
        world = SimulationWorld()
        world.add_node(_node())
        jet = _helicopter()
        jet.airframe = "jet"
        world.aircraft = [jet]
        as_jet = world.generate_detections_for_node("md-node", 1000)
        assert _frame_with_helicopter(False, seed) == as_jet


def test_enabled_world_adds_lines_at_the_body_delay_without_adsb(loud_helicopter):
    seen = 0
    for seed in range(20):
        frame = _frame_with_helicopter(True, seed)
        if "adsb" not in frame:
            continue
        group = _at_body_delay(frame)
        lines = group[1:]
        seen += len(lines)
        assert len(lines) <= 12
        assert all(-1000.0 <= frame["doppler"][i] <= 1000.0 for i in lines)
        assert all(frame["adsb"][i] is None for i in lines)
        assert frame["adsb"][0]["hex"] == "abc123"
        assert len(frame["adsb"]) == len(frame["delay"]) == len(frame["doppler"]) == len(frame["snr"])
    assert seen > 0


def test_body_detection_is_unchanged_by_lines(loud_helicopter):
    off = _frame_with_helicopter(False, 4)
    on = _frame_with_helicopter(True, 4)
    assert "adsb" in off
    assert [off[k][0] for k in ("delay", "doppler", "snr")] == [on[k][0] for k in ("delay", "doppler", "snr")]


def test_training_batch_ground_truth_stays_aligned(loud_helicopter):
    random.seed(2)
    world = SimulationWorld(center_lat=_RX_LAT, center_lon=_RX_LON)
    world.add_node(_node())
    world.min_aircraft = world.max_aircraft = 0
    world.aircraft = [_helicopter()]
    world.micro_doppler_enabled = True
    records = world.generate_training_batch(10, dt=0.1)
    assert records
    flagged = 0
    for rec in records:
        assert len(rec["ground_truth"]) == len(rec["delay"]) == len(rec["adsb"])
        flagged += sum(1 for g in rec["ground_truth"] if g.get("micro_doppler"))
        assert all(g["object_id"] == "heli-1" for g in rec["ground_truth"] if g.get("micro_doppler"))
    assert flagged > 0


# ── Orchestrator ─────────────────────────────────────────────────────────────


def test_airframe_mix_is_parsed_and_normalised():
    mix = parse_airframe_mix("jet:0.5,turboprop:0.2,piston:0.2,helicopter:0.1")
    assert set(mix) == {"jet", "turboprop", "piston", "helicopter"}
    assert sum(mix.values()) == pytest.approx(1.0)
    assert parse_airframe_mix("jet:2,helicopter:2") == {"jet": 0.5, "helicopter": 0.5}


@pytest.mark.parametrize("spec", ["jet:0.5,blimp:0.5", "", "jet:0", "jet:-1"])
def test_invalid_airframe_mix_is_rejected(spec):
    with pytest.raises(ValueError):
        parse_airframe_mix(spec)


def test_ground_truth_payload_carries_airframe():
    summary = [{"id": "obj-1", "lat": 1.0, "lon": 2.0, "alt_km": 3.0, "airframe": "helicopter"}]
    assert build_ground_truth_payload(summary)[0]["airframe"] == "helicopter"
    assert build_ground_truth_payload([{"id": "x", "lat": 0, "lon": 0, "alt_km": 0}])[0]["airframe"] == "jet"
