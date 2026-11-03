"""The GNSS error model (retina_simulation/gnss.py)."""

import math
import statistics

import pytest

from retina_simulation import gnss
from retina_simulation.motion import Motion

_LONG_S = 40000.0


@pytest.mark.parametrize("rate_hz", [1.0, 10.0])
@pytest.mark.parametrize("tau_s", [5.0, 20.0])
def test_correlated_error_keeps_its_size_at_any_rate_and_time_constant(rate_hz, tau_s):
    model = gnss.GnssModel(sigma_corr_m=1.5, tau_s=tau_s, rate_hz=rate_hz)
    times = gnss.stamps(model, _LONG_S, "test")
    err = gnss.errors(model, times, "test")
    nu = [v for axis in gnss.AXES for v in err[axis]["corr"]]
    assert statistics.pstdev(nu) == pytest.approx(1.5, rel=0.05)


def test_autocorrelation_follows_exp_minus_lag_over_t():
    model = gnss.GnssModel(sigma_corr_m=1.5, tau_s=10.0)
    times = gnss.stamps(model, _LONG_S, "test")
    acf = gnss.autocorrelation(gnss.errors(model, times, "test")["north"]["corr"], 30)
    for lag in (1, 5, 10, 20, 30):
        assert acf[lag] == pytest.approx(math.exp(-lag / 10.0), abs=0.05)


def test_zero_time_constant_is_white_noise():
    model = gnss.GnssModel(sigma_corr_m=1.5, tau_s=0.0)
    times = gnss.stamps(model, 20000.0, "test")
    acf = gnss.autocorrelation(gnss.errors(model, times, "test")["up"]["corr"], 3)
    assert all(abs(v) < 0.03 for v in acf[1:])


def test_starts_stationary():
    model = gnss.GnssModel(sigma_corr_m=1.5, tau_s=60.0)
    first = [gnss.errors(model, [0.0, 1.0], "test", r)["east"]["corr"][0] for r in range(3000)]
    assert statistics.pstdev(first) == pytest.approx(1.5, rel=0.05)


def test_white_noise_has_its_sigma():
    model = gnss.GnssModel(sigma_white_m=0.5)
    times = gnss.stamps(model, 20000.0, "test")
    assert statistics.pstdev(gnss.errors(model, times, "test")["east"]["white"]) == pytest.approx(0.5, rel=0.03)


def test_axes_are_independent():
    model = gnss.GnssModel(tau_s=5.0)
    times = gnss.stamps(model, _LONG_S, "test")
    err = gnss.errors(model, times, "test")
    for a, b in (("east", "north"), ("east", "up"), ("north", "up")):
        assert abs(statistics.correlation(err[a]["corr"], err[b]["corr"])) < 0.05
        assert abs(statistics.correlation(err[a]["white"], err[b]["white"])) < 0.05


def test_same_key_same_fixes_other_realisation_differs():
    model, motion = gnss.GnssModel(), Motion("linear", speed_ms=15.0, heading_deg=90.0)
    a = gnss.fixes(model, motion, 300.0, "seed-101", 0)
    assert a == gnss.fixes(model, motion, 300.0, "seed-101", 0)
    assert a != gnss.fixes(model, motion, 300.0, "seed-101", 1)


def test_stamps_follow_the_rate_from_a_random_phase():
    model = gnss.GnssModel(rate_hz=5.0)
    t = gnss.stamps(model, 300.0, "test")
    assert 0.0 <= t[0] < 0.2
    assert all(b - a == pytest.approx(0.2) for a, b in zip(t, t[1:]))
    assert t[-1] <= 300.0 < t[-1] + 0.2


def test_time_offset_reports_an_earlier_position_and_latency_only_the_arrival():
    quiet = {"sigma_white_m": 0.0, "sigma_corr_m": 0.0}
    motion = Motion("linear", speed_ms=15.0, heading_deg=90.0)
    late = gnss.fixes(gnss.GnssModel(**quiet, time_offset_s=0.2), motion, 10.0, "k")
    for f in late:
        assert f["east_m"] - f["true_east_m"] == pytest.approx(-3.0, abs=1e-3)
    slow = gnss.fixes(gnss.GnssModel(**quiet, latency_s=0.5), motion, 10.0, "k")
    for f in slow:
        assert f["east_m"] == pytest.approx(f["true_east_m"], abs=1e-3)
        assert f["t_avail_s"] == pytest.approx(f["t_s"] + 0.5)


def test_fix_is_truth_plus_both_error_parts():
    motion = Motion("circle", speed_ms=15.0, lap_s=300.0)
    for f in gnss.fixes(gnss.GnssModel(), motion, 60.0, "k"):
        for axis in gnss.AXES:
            want = f[f"true_{axis}_m"] + f[f"corr_{axis}_m"] + f[f"white_{axis}_m"]
            assert f[f"{axis}_m"] == pytest.approx(want, abs=2e-4)
