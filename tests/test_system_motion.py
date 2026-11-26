"""Step 4 system run: what a moving node reports with its frames, and which runs are flown."""

import math
import os
import sys
from types import SimpleNamespace

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "tools"))

from retina_simulation import experiment  # noqa: E402
from retina_simulation.motion import motion_from, place_receivers  # noqa: E402
from retina_simulation.orchestrator import FleetOrchestrator  # noqa: E402

_STUDY = os.path.join(_HERE, "..", "experiments", "receiver_system.json")


def _world_and_origins():
    exp = experiment.load(_STUDY)
    world = experiment.build_world(exp, exp["runs"][0], experiment.fleet_nodes(exp))
    origins = {nid: (n.rx_lat, n.rx_lon, n.rx_alt_ft * 0.3048 / 1000.0) for nid, n in world.nodes.items()}
    return exp, world, origins


def test_truth_pose_is_the_moved_receiver():
    _, world, origins = _world_and_origins()
    motion = motion_from({"type": "circle", "speed_ms": 15.0, "lap_s": 300.0})
    place_receivers(world, origins, motion, 40.0)
    stub = SimpleNamespace(world=world, rx_pose_estimates=None, receiver_origins=origins)
    nid = next(iter(world.nodes))
    pose = FleetOrchestrator._rx_motion(stub, nid, 40.0)
    assert pose["lat"] == pytest.approx(world.nodes[nid].rx_lat, abs=1e-8)
    assert math.hypot(pose["vel_east_ms"], pose["vel_north_ms"]) == pytest.approx(15.0, abs=1e-3)


def test_estimated_pose_is_the_offset_from_each_registered_position():
    _, world, origins = _world_and_origins()
    est = {"t_s": 5.0, "east_m": 100.0, "north_m": -50.0, "up_m": 0.0, "vel_east_ms": 3.0, "vel_north_ms": 1.0,
           "vel_up_ms": 0.0}  # fmt: skip
    stub = SimpleNamespace(world=world, rx_pose_estimates={5.0: est}, receiver_origins=origins)
    for nid, (lat0, lon0, _) in origins.items():
        pose = FleetOrchestrator._rx_motion(stub, nid, 5.0)
        north = math.radians(pose["lat"] - lat0) * 6371e3
        east = math.radians(pose["lon"] - lon0) * 6371e3 * math.cos(math.radians(lat0))
        assert (east, north) == pytest.approx((100.0, -50.0), abs=1.0)
        assert pose["vel_east_ms"] == 3.0
    assert FleetOrchestrator._rx_motion(stub, nid, 6.0) is None


def test_runs_static_once_and_moving_cases_per_pose_source():
    pytest.importorskip("retina_geolocator")
    import system_motion

    exp = experiment.load(_STUDY)
    runs = system_motion.runs_for(exp)
    assert [(c["id"], p) for c, _, p in runs if c["id"] == "static"] == [("static", "none")]
    moving = {(c["id"], p) for c, _, p in runs if c["id"] != "static"}
    assert moving == {(c, p) for c in exp["system_cases"][1:] for p in ("none", "truth", "kf")}
    assert {r["seed"] for _, r, _ in runs} == set(exp["system_seeds"])
