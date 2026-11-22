"""Direction-finding study (step 1a): the array model, the Bartlett estimator,
the closed-form single-tower solve, and the south-west experiment file.

None of this needs retina_geolocator (system A is the existing offline solve,
tested in test_offline_solve.py)."""

import math
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "tools"))

import doa  # noqa: E402
import doa_solve  # noqa: E402

from retina_simulation import array as arr  # noqa: E402
from retina_simulation import experiment  # noqa: E402

_DOA = os.path.join(_HERE, "..", "experiments", "doa_static.json")
_BASELINE = os.path.join(_HERE, "..", "experiments", "baseline_static.json")
LAM = arr.C_M_S / 521e6
BORE = 225.0


def _bearings(az, el, err, snr_db=200.0, cfg=None, key=(1, 0)):
    cfg = cfg or arr.ArrayConfig()
    u = arr.unit_vectors(np.atleast_1d(az), np.atleast_1d(el))
    snr = np.full(len(u), snr_db)
    x = arr.snapshots(u, snr, cfg, err, BORE, LAM, key)
    est, _ = doa.doa(x, arr.positions(cfg, BORE, LAM), LAM, doa.front_half(BORE))
    return (est - np.atleast_1d(az) + 180.0) % 360.0 - 180.0


def test_noise_free_triangle_finds_the_true_azimuth_at_any_elevation():
    az = np.linspace(BORE - 80, BORE + 80, 33)
    for el in (0.0, 10.0, 30.0):
        assert np.max(np.abs(_bearings(az, np.full_like(az, el), arr.ArrayErrors.none()))) < 0.02


def test_thermal_spread_matches_the_bound():
    """sigma_az = 1 / (2 pi (s/lambda) sqrt(SNR)) for the triangle, any azimuth."""
    rng = np.random.default_rng(3)
    az = BORE + rng.uniform(-60, 60, 4000)
    err = arr.ArrayErrors.none()
    err.thermal = True
    e = _bearings(az, np.zeros_like(az), err, snr_db=20.0)
    bound = math.degrees(1.0 / (2 * math.pi * 0.5 * math.sqrt(100.0)))
    assert abs(np.sqrt(np.mean(e**2)) / bound - 1.0) < 0.1


def test_front_half_search_never_answers_behind_the_array():
    e = _bearings(np.array([BORE + 150.0, BORE - 170.0]), np.zeros(2), arr.ArrayErrors.none())
    est = (np.array([BORE + 150.0, BORE - 170.0]) + e - BORE + 180) % 360 - 180
    assert np.all(np.abs(est) <= 90.0 + 1e-9)


def test_rspduo_offset_biases_only_across_element_0():
    """Elements 1 and 2 shifted together: no bias along element 0 (boresight),
    about 0.37 deg per degree of offset at 90 deg from it."""
    err = arr.ArrayErrors.none()
    err.device_offset_deg = 5.0
    # the run's device draw is a standard normal; scale the expected bias by it
    z = arr.keyed_rng("doa-run", 1, 0)
    z.standard_normal(3)
    offset = 5.0 * z.standard_normal()
    bias = _bearings(np.array([BORE, BORE + 89.0]), np.zeros(2), err)
    assert abs(bias[0]) < 0.01
    assert abs(abs(bias[1]) / abs(offset) - 0.37) < 0.05


def test_draws_are_keyed_by_run():
    err = arr.ArrayErrors()
    az = BORE + np.linspace(-50, 50, 20)
    a = _bearings(az, np.zeros(20), err, snr_db=12.0, key=(101, 0))
    b = _bearings(az, np.zeros(20), err, snr_db=12.0, key=(101, 0))
    c = _bearings(az, np.zeros(20), err, snr_db=12.0, key=(101, 1))
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_beam_override_reaches_the_nodes_but_not_the_baseline():
    assert {n["beam_azimuth_deg"] for n in experiment.fleet_nodes(experiment.load(_DOA))} == {225.0}
    assert {n["beam_azimuth_deg"] for n in experiment.fleet_nodes(experiment.load(_BASELINE))} == {148.0}


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    exp = experiment.load(_DOA)
    out = str(tmp_path_factory.mktemp("doa") / "seed-101")
    experiment.record_run(exp, exp["runs"][0], out)
    return exp, out


def test_closed_form_solve_lands_on_the_aircraft(recording):
    """True delay and bearing at the true altitude: exact geometry, < 1 m."""
    exp, out = recording
    node = doa_solve.node_config(out, exp["doa"]["node"])
    dets = doa_solve.detections(out, exp["doa"]["node"])
    worst = 0.0
    for d in dets:
        lat, lon = doa_solve.locate(node, d["delay_true_us"], d["az_true_deg"], d["true_alt_m"] / 1000.0)
        worst = max(worst, math.hypot(*doa_solve._en_m(d["true_lat"], d["true_lon"], lat, lon)))
    assert dets and worst < 1.0


def test_perfect_array_leaves_only_delay_and_altitude_error(recording):
    exp, out = recording
    node = doa_solve.node_config(out, exp["doa"]["node"])
    dets = doa_solve.detections(out, exp["doa"]["node"])
    b = doa_solve.measure(dets, node, arr.ArrayConfig(), arr.ArrayErrors.none(), (101, 0))
    m_true = doa_solve.metrics(doa_solve.solve(dets, b, node, alt_km=None), 300)
    m_assumed = doa_solve.metrics(doa_solve.solve(dets, b, node, alt_km=2.0), 300)
    assert m_true["bearing_rms_deg"] < 0.02
    assert m_true["pos_rmse_m"] < 60.0
    assert m_true["pos_rmse_m"] < m_assumed["pos_rmse_m"] < 150.0


def test_doa_experiment_reproduces_its_recordings(tmp_path):
    exp = experiment.load(_DOA)
    assert [r["seed"] for r in exp["runs"]] == [101, 202, 303, 404, 505]
    for run in exp["runs"]:
        got = experiment.record_run(exp, run, str(tmp_path / str(run["seed"])))
        assert experiment.fingerprint_mismatch(run["fingerprint"], got) == [], run["seed"]
