"""Per-detection truth labels and the run recorder.

The labels ride beside the frame, never in it: the wire frame must be
byte-for-byte what it was without recording, and each label must sit at the
same index as the detection it describes.
"""

import json
import math
import random

from retina_simulation.recorder import RunRecorder
from retina_simulation.world import NodeConfig, SimulatedAircraft, SimulationWorld, _bearing_deg

_RX_LAT, _RX_LON = 49.259323, -123.247566
_TX_LAT, _TX_LON = 49.345, -122.972778


def _world_with_targets(n=6):
    world = SimulationWorld(center_lat=_RX_LAT, center_lon=_RX_LON)
    world.add_node(
        NodeConfig(
            node_id="ubc-test",
            rx_lat=_RX_LAT,
            rx_lon=_RX_LON,
            rx_alt_ft=318.0,
            tx_lat=_TX_LAT,
            tx_lon=_TX_LON,
            tx_alt_ft=2024.0,
            fc_hz=521_000_000,
            beam_azimuth_deg=147.0,
            beam_width_deg=60.0,
            max_range_km=60.0,
        )
    )
    for i in range(n):
        br = math.radians(130.0 + 6.0 * i)
        dist = 8.0 + 3.0 * i
        world.aircraft.append(
            SimulatedAircraft(
                object_id=f"obj-{i:05d}",
                lat=_RX_LAT + math.degrees(dist * math.cos(br) / 6371.0),
                lon=_RX_LON + math.degrees(dist * math.sin(br) / (6371.0 * math.cos(math.radians(_RX_LAT)))),
                alt_km=3.0,
                vel_east=0.2,
                vel_north=0.0,
                vel_up=0.0,
                heading_deg=90.0,
                speed_km_s=0.2,
                has_adsb=i % 2 == 0,
                adsb_hex=f"c0{i:04x}" if i % 2 == 0 else None,
            )
        )
    return world


def test_labels_align_with_frame_arrays():
    random.seed(7)
    world = _world_with_targets()
    labels = []
    frame = world.generate_detections_for_node("ubc-test", 1_000, labels=labels)
    assert len(labels) == len(frame["delay"]) == len(frame["doppler"]) == len(frame["snr"])
    assert any(not lab["is_clutter"] for lab in labels)
    for lab, delay, doppler in zip(labels, frame["delay"], frame["doppler"]):
        if lab["is_clutter"]:
            assert lab["object_id"] is None and lab["delay_true"] is None
        else:
            assert lab["object_id"].startswith("obj-")
            assert abs(delay - lab["delay_true"]) < 2.0
            assert abs(doppler - lab["doppler_true"]) < 20.0


def test_recording_does_not_change_the_wire_frame():
    random.seed(11)
    plain = _world_with_targets().generate_detections_for_node("ubc-test", 1_000)
    random.seed(11)
    recorded = _world_with_targets().generate_detections_for_node("ubc-test", 1_000, labels=[])
    assert plain == recorded


def test_recorder_writes_residuals_and_truth(tmp_path):
    random.seed(3)
    world = _world_with_targets()
    node = {"node_id": "ubc-test", "tx_callsign": "CHAN-DT", "fc_hz": 521_000_000}
    rec = RunRecorder(str(tmp_path), [node])
    labels = []
    frame = world.generate_detections_for_node("ubc-test", 2_000, labels=labels)
    rec.frame("ubc-test", frame, labels, sent=True)
    rec.truth(2_000, world.aircraft)
    rec.close()

    line = json.loads((tmp_path / "frames.ndjson").read_text().splitlines()[0])
    assert line["tx_callsign"] == "CHAN-DT" and line["sent"] is True
    for det in line["detections"]:
        if det["is_clutter"]:
            assert det["delay_resid_us"] is None
        else:
            assert det["delay_resid_us"] == round(det["delay_us"] - det["delay_true"], 4)
    truth = json.loads((tmp_path / "truth.ndjson").read_text().splitlines()[0])
    assert {a["id"] for a in truth["aircraft"]} == {ac.object_id for ac in world.aircraft}
    assert truth["aircraft"][0]["vel_east_ms"] == 200.0
    assert json.loads((tmp_path / "nodes.json").read_text()) == [node]


def test_ubc_receiver_beam_reaches_yvr():
    assert abs(_bearing_deg(_RX_LAT, _RX_LON, 49.1947, -123.1839) - 147.2) < 0.5


def test_recorder_keeps_each_solve_once(tmp_path):
    rec = RunRecorder(str(tmp_path), [])
    a = {"ts_ms": 2, "solve_key": "mn-dark-1", "outcome": "published", "solver_hex": "mn1"}
    b = {"ts_ms": 1, "solve_key": "mn-dark-2", "outcome": "n2_unconfirmed", "solver_hex": None}
    assert rec.solves([a, b]) == 2
    assert rec.solves([a, b, {**a, "ts_ms": 3}]) == 1
    rec.close()
    lines = [json.loads(line) for line in (tmp_path / "solves.ndjson").read_text().splitlines()]
    assert [r["ts_ms"] for r in lines] == [1, 2, 3]
