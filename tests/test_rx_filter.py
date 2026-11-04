"""The receiver Kalman filter (tools/rx_filter.py).

The key test is the matched one: when the truth really is what the filter
assumes (white acceleration of the filter's sigma_a, Gauss-Markov plus white
GNSS error of its parameters), the filter's stated uncertainty must match its
actual error (mean NEES = 6) and it must beat the filter that treats the GNSS
error as white.
"""

import math
import os
import random
import sys

import pytest

np = pytest.importorskip("numpy")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "tools"))

import rx_filter  # noqa: E402

from retina_simulation import gnss  # noqa: E402
from retina_simulation.motion import Motion  # noqa: E402


def _matched_run(cfg, model, seed, duration=300.0):
    """Truth that obeys the filter's own motion model, and fixes from the GNSS model."""
    rng = random.Random(f"matched:{seed}")
    stamps = gnss.stamps(model, duration, f"m{seed}")
    err = gnss.errors(model, stamps, f"m{seed}")
    fixes, truth = [], []
    state = {a: [0.0, rng.gauss(0.0, cfg.sigma_v0)] for a in gnss.AXES}
    prev = stamps[0]
    for i, t in enumerate(stamps):
        dt = t - prev
        for a in gnss.AXES:
            acc = rng.gauss(0.0, cfg.sigma_a)
            x, v = state[a]
            state[a] = [x + v * dt + acc * dt * dt / 2, v + acc * dt]
        prev = t
        fix = {"t_s": t, "t_avail_s": t}
        tr = {}
        for a in gnss.AXES:
            fix[f"{a}_m"] = state[a][0] + err[a]["corr"][i] + err[a]["white"][i]
            tr[f"{a}_m"], tr[f"vel_{a}_ms"] = state[a]
        fixes.append(fix)
        truth.append(tr)
    return stamps, fixes, truth


def _matched(variant, runs=40):
    model = gnss.GnssModel(sigma_white_m=0.5, sigma_corr_m=1.5, tau_s=60.0)
    cfg = rx_filter.FilterConfig(sigma_a=0.5, sigma_v0=4.0, sigma_white=0.5, sigma_corr=1.5, tau_s=60.0)
    nees, verr = [], []
    for seed in range(runs):
        stamps, fixes, truth = _matched_run(cfg, model, seed)
        for est, tr in zip(rx_filter.run(fixes, stamps, cfg, variant), truth):
            if est is None or est["t_s"] < 30:
                continue
            nees.append(rx_filter.nees(est, tr))
            verr.append(math.hypot(est["vel_east_ms"] - tr["vel_east_ms"], est["vel_north_ms"] - tr["vel_north_ms"]))
    return float(np.mean(nees)), math.sqrt(float(np.mean(np.square(verr))))


def test_matched_truth_gives_honest_uncertainty_and_beats_the_white_filter():
    nees_kf, v_kf = _matched("kf")
    nees_white, v_white = _matched("white")
    assert nees_kf == pytest.approx(6.0, rel=0.1)
    assert nees_white > 7.5
    assert v_kf < v_white


def test_starting_covariance_matches_a_normal_update_from_a_diffuse_start():
    cfg = rx_filter.FilterConfig(sigma_a=0.5, sigma_v0=20.0, sigma_white=0.5, sigma_corr=1.5, tau_s=60.0)
    f = rx_filter.AxisFilter(cfg, 12.0, 0.0)
    big = 1e6
    x = np.zeros(3)
    p = np.diag([big**2, cfg.sigma_v0**2, cfg.sigma_corr**2])
    h = np.array([1.0, 0.0, 1.0])
    s = h @ p @ h + cfg.sigma_white**2
    k = p @ h / s
    x = x + k * (12.0 - h @ x)
    p = (np.eye(3) - np.outer(k, h)) @ p
    assert np.allclose(f.x, x, atol=1e-6)
    assert np.allclose(f.P, p, atol=1e-4)


def test_converges_on_a_straight_line_with_almost_no_noise():
    motion = Motion("linear", speed_ms=15.0, heading_deg=90.0)
    model = gnss.GnssModel(sigma_white_m=0.01, sigma_corr_m=0.01, tau_s=60.0)
    cfg = rx_filter.FilterConfig(sigma_a=0.01, sigma_v0=20.0, sigma_white=0.01, sigma_corr=0.01, tau_s=60.0)
    fixes = gnss.fixes(model, motion, 60.0, "k")
    est = rx_filter.run(fixes, [60.0], cfg)[0]
    assert est["vel_east_ms"] == pytest.approx(15.0, abs=0.05)
    assert est["vel_north_ms"] == pytest.approx(0.0, abs=0.05)
    assert est["east_m"] == pytest.approx(900.0, abs=0.5)


def test_uses_only_fixes_that_have_arrived():
    fixes = [
        {"t_s": 0.0, "t_avail_s": 0.5, "east_m": 0.0, "north_m": 0.0, "up_m": 0.0},
        {"t_s": 1.0, "t_avail_s": 1.5, "east_m": 1000.0, "north_m": 0.0, "up_m": 0.0},
    ]
    cfg = rx_filter.FilterConfig()
    for variant in rx_filter.VARIANTS:
        before, after = rx_filter.run(fixes, [0.2, 1.2], cfg, variant)
        assert before is None
        assert after["east_m"] < 1.0
    third = rx_filter.run(fixes, [1.6], cfg, "kf")[0]
    assert third["east_m"] > 100.0


def test_raw_variant_differences_the_last_two_fixes():
    fixes = [
        {"t_s": 0.0, "t_avail_s": 0.0, "east_m": 0.0, "north_m": 0.0, "up_m": 0.0},
        {"t_s": 1.0, "t_avail_s": 1.0, "east_m": 3.0, "north_m": -1.0, "up_m": 0.0},
    ]
    est = rx_filter.run(fixes, [1.5], rx_filter.FilterConfig(), "raw")[0]
    assert (est["vel_east_ms"], est["vel_north_ms"]) == pytest.approx((3.0, -1.0))
    assert (est["east_m"], est["north_m"]) == pytest.approx((4.5, -1.5))


def test_prediction_to_the_query_time_follows_the_velocity():
    motion = Motion("linear", speed_ms=15.0, heading_deg=0.0)
    model = gnss.GnssModel(sigma_white_m=0.01, sigma_corr_m=0.01, rate_hz=1.0)
    cfg = rx_filter.FilterConfig(sigma_white=0.01, sigma_corr=0.01)
    fixes = gnss.fixes(model, motion, 100.0, "k")
    last = fixes[-1]["t_s"]
    est = rx_filter.run(fixes, [last + 0.9], cfg)[0]
    assert est["north_m"] == pytest.approx(15.0 * (last + 0.9), abs=0.3)
