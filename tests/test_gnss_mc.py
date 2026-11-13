"""Step 3d Monte Carlo levels and thresholds (tools/gnss_mc.py, gnss_mc_report.py).

Needs retina_geolocator (gnss_mc imports offline_solve), so it skips in this repo's CI.
"""

import math
import os
import sys

import pytest

pytest.importorskip("numpy")
pytest.importorskip("matplotlib")
pytest.importorskip("retina_geolocator")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "tools"))

import gnss_mc  # noqa: E402
import gnss_mc_report  # noqa: E402
import numpy as np  # noqa: E402
import rx_filter  # noqa: E402

from retina_simulation import experiment  # noqa: E402

_STUDY = os.path.join(_HERE, "..", "experiments", "receiver_gnss_mc.json")


def _exp():
    return {**(e := experiment.load(_STUDY)), "cases": gnss_mc.cases_of(e)}


def test_levels_are_distinct_and_include_the_nominal():
    exp = _exp()
    found, sweeps, grid = gnss_mc.levels(exp)
    assert found["nominal"] == {}
    for sw in exp["sweeps"]:
        values = [v for v, _ in sweeps[sw["id"]]]
        assert values == sorted(values)
        assert (gnss_mc.nominal_value(exp, sw["key"]), "nominal") in sweeps[sw["id"]]
    assert ("sigma_corr_m-6" in found) and (6.0, 60.0, "sigma_corr_m-6") in grid
    assert (1.5, 300.0, "tau_s-300") in grid
    assert len(found) == len(set(found))


def test_overrides_reach_the_gnss_and_a_matched_filter():
    exp = _exp()
    case = exp["cases"][0]
    e = gnss_mc.apply(exp, {"gnss.sigma_corr_m": 6.0, "gnss.tau_s": 300.0})
    assert e["gnss"]["sigma_corr_m"] == 6.0 and exp["gnss"]["sigma_corr_m"] == 1.5
    cfg = rx_filter.config_for(e, case)
    assert (cfg.sigma_corr, cfg.tau_s) == (6.0, 300.0)


def test_filter_scales_detune_only_the_filter():
    exp = _exp()
    case = next(c for c in exp["cases"] if c.get("speed") == "vehicle")
    base = rx_filter.config_for(exp, case)
    e = gnss_mc.apply(
        exp, {"filter.sigma_a_scale": 4.0, "filter.assumed_sigma_scale": 0.5, "filter.assumed_tau_scale": 10.0}
    )
    cfg = rx_filter.config_for(e, case)
    assert cfg.sigma_a == pytest.approx(4 * base.sigma_a)
    assert (cfg.sigma_white, cfg.sigma_corr) == pytest.approx((0.5 * base.sigma_white, 0.5 * base.sigma_corr))
    assert cfg.tau_s == pytest.approx(10 * base.tau_s)
    assert e["gnss"] == exp["gnss"]


def test_unknown_key_is_refused():
    with pytest.raises(ValueError):
        gnss_mc.nominal_value(_exp(), "gnss.nonsense")
    with pytest.raises(ValueError):
        gnss_mc.nominal_value(_exp(), "filter.sigma_v0")


def test_crossing_interpolates_outward_from_nominal():
    xs = [0.1, 0.5, 1.0, 2.0, 4.0]
    ys = [30.0, 8.0, 0.0, 6.0, 14.0]
    up = gnss_mc_report.crossing(xs, ys, 1.0, 10.0, +1, log=True)
    assert up == pytest.approx(math.exp(math.log(2) + 0.5 * math.log(2)))
    down = gnss_mc_report.crossing(xs, ys, 1.0, 10.0, -1, log=False)
    assert down == pytest.approx(0.5 - (2 / 22) * 0.4)
    assert gnss_mc_report.crossing(xs, ys, 1.0, 20.0, +1, log=True) is None
    assert gnss_mc_report.crossing(xs, [0, 0, 12.0, 0, 0], 1.0, 10.0, +1, log=True) == 1.0


def test_increase_and_bootstrap_agree_on_identical_runs():
    run = np.array([4.0, 1.0, 1.21, 1.0, 0.0, 0.0, 10.0])
    by_case = {"a": np.tile(run, (5, 1)), "b": np.tile(run, (3, 1))}
    assert gnss_mc_report.increase(by_case["a"], "pos") == pytest.approx(100.0)
    assert gnss_mc_report.increase(by_case["a"], "vel") == pytest.approx(10.0)
    idx = gnss_mc_report.boot_indices({"a": list(range(5)), "b": list(range(3))}, n_boot=50)
    b = gnss_mc_report.boot_increase(by_case, idx, "vel")
    assert b.shape == (50,) and np.allclose(b, 10.0)


def test_finished_solves_are_not_repeated(tmp_path):
    exp = _exp()
    out = str(tmp_path)
    tasks = gnss_mc.tasks_for(exp, out)
    found, _, _ = gnss_mc.levels(exp)
    n_rec = len(gnss_mc.recordings(exp, out))
    assert len(tasks) == n_rec * (1 + len(found) * exp["realisations"])
    path = tasks[0][0]
    os.makedirs(os.path.join(path, "analysis"))
    open(os.path.join(path, "analysis", gnss_mc.solve_file("truth", None)), "w").close()
    assert len(gnss_mc.tasks_for(exp, out)) == len(tasks) - 1
