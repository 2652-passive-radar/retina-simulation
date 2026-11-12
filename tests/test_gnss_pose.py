"""Receiver poses from GNSS for GD (tools/gnss_pose.py, step 3c).

The solve tests need retina_geolocator, which this repo's CI does not install,
so they skip there; they run wherever the backend's environment is used.
"""

import json
import math
import os
import sys

import pytest

pytest.importorskip("numpy")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "tools"))

import gnss_pose  # noqa: E402
import rx_filter  # noqa: E402

from retina_simulation import experiment, gnss  # noqa: E402
from retina_simulation.motion import motion_from  # noqa: E402

_STUDY = os.path.join(_HERE, "..", "experiments", "receiver_gnss.json")


def _exp(**gnss_overrides):
    exp = experiment.load(_STUDY)
    exp["gnss"] = {**exp["gnss"], **gnss_overrides}
    return exp


def _case(exp, case_id):
    return next(c for c in exp["cases"] if c["id"] == case_id)


def _nodes(exp):
    return experiment.fleet_nodes(exp)


def test_quiet_gnss_poses_follow_the_true_trajectory():
    exp = _exp(sigma_white_m=0.001, sigma_corr_m=0.001)
    case = _case(exp, "circle-vehicle")
    motion = motion_from(case["motion"])
    for source in ("gnss", "kf"):
        poses = gnss_pose.poses_from_estimates(exp, _nodes(exp), gnss_pose.estimates(exp, case, 101, 0, source))
        for i, t in enumerate(gnss_pose.tick_times(exp)):
            if t < 30:
                continue
            pose = next(iter(poses[exp["start_time_ms"] + int(round(t * 1000))].values()))
            e, n, ve, vn = motion.state(t)
            assert math.hypot(pose["offset_east_m"] - e, pose["offset_north_m"] - n) < 0.5
            assert math.hypot(pose["vel_east_ms"] - ve, pose["vel_north_ms"] - vn) < 0.6


def test_pose_positions_are_the_offsets_on_the_ellipsoid():
    exp = _exp()
    case = _case(exp, "linear-east-vehicle")
    nodes = _nodes(exp)
    ests = gnss_pose.estimates(exp, case, 202, 1, "kf")
    poses = gnss_pose.poses_from_estimates(exp, nodes, ests)
    t_ms = exp["start_time_ms"] + 200_000
    est = ests[199]
    for n in nodes:
        p = poses[t_ms][n["node_id"]]
        north_m = math.radians(p["lat"] - n["rx_lat"]) * 6371000.0
        east_m = math.radians(p["lon"] - n["rx_lon"]) * 6371000.0 * math.cos(math.radians(n["rx_lat"]))
        assert east_m == pytest.approx(est["east_m"], abs=15.0)
        assert north_m == pytest.approx(est["north_m"], abs=15.0)
        assert p["vel_east_ms"] == est["vel_east_ms"]


def test_same_gnss_draws_as_step_3b():
    exp = _exp()
    case = _case(exp, "oscillate-vehicle")
    fixes = gnss.fixes(gnss.model_from(exp["gnss"]), motion_from(case["motion"]), exp["duration_s"], "seed-303", 2)
    direct = rx_filter.run(fixes, gnss_pose.tick_times(exp), rx_filter.config_for(exp, case), "kf")
    keys = ("east_m", "north_m", "up_m", "vel_east_ms", "vel_north_ms", "vel_up_ms")
    got = gnss_pose.estimates(exp, case, 303, 2, "kf")
    assert [{k: e[k] for k in keys} for e in got] == [{k: e[k] for k in keys} for e in direct]


def test_realisations_differ_and_repeat():
    exp = _exp()
    case = _case(exp, "linear-north-walk")
    a = gnss_pose.estimates(exp, case, 404, 0, "gnss")
    assert a == gnss_pose.estimates(exp, case, 404, 0, "gnss")
    assert a != gnss_pose.estimates(exp, case, 404, 1, "gnss")


def test_truth_source_has_no_estimate():
    exp = _exp()
    assert gnss_pose.poses_for("unused", exp, _case(exp, "static"), 101, 0, "truth") is None
    with pytest.raises(ValueError):
        gnss_pose.estimates(exp, _case(exp, "static"), 101, 0, "truth")


def test_truth_override_reproduces_gd_and_estimate_reports_its_error(tmp_path):
    pytest.importorskip("retina_geolocator")
    import offline_solve

    exp = _exp()
    case = _case(exp, "linear-east-vehicle")
    run = exp["runs"][1]
    d = str(tmp_path / "rec")
    experiment.record_run(exp, run, d, motion_from(case["motion"]), rx_doppler=True)
    truth = offline_solve.receiver_poses(d)
    keys = ("est_lat", "est_lon", "est_vel_east_ms", "est_vel_north_ms", "reason")
    a = offline_solve.solve_run(d, None, "geometric_doppler")
    b = offline_solve.solve_run(d, None, "geometric_doppler", pose_estimate=truth)
    assert [{k: r.get(k) for k in keys} for r in a] == [{k: r.get(k) for k in keys} for r in b]
    assert all(r.get("rx_pos_err_m", 0.0) == 0.0 for r in b)

    with open(os.path.join(d, "nodes.json"), encoding="utf-8") as f:
        assert len(json.load(f)) == 3
    kf = offline_solve.solve_run(
        d, None, "geometric_doppler", pose_estimate=gnss_pose.poses_for(d, exp, case, run["seed"], 0, "kf")
    )
    errs = [r["rx_vel_err_ms"] for r in kf if r.get("t_s", 0) >= 30 and "rx_vel_err_ms" in r]
    assert errs and 0.05 < math.sqrt(sum(e * e for e in errs) / len(errs)) < 1.5
