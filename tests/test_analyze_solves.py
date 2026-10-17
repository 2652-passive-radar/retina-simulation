"""tools/analyze_run.py's scoring of server solves.

The two records in the first test are the shape found in a real system run
(2026-10-17): one key, one measurement time, published as a 2-node solve and
again ~1 s later as a 3-node re-solve whose map position the smoother held at
the first one.  Only the raw position tells them apart.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "tools"))

from analyze_run import Truth, one_per_epoch, solve_rows  # noqa: E402

_TRUTH = Truth(
    [
        {
            "t_ms": t,
            "aircraft": [
                {
                    "id": "flight-101",
                    "adsb_hex": "e00065",
                    "lat": 49.155158,
                    "lon": -123.097374,
                    "alt_m": 1500.0,
                    "vel_east_ms": -50.0,
                    "vel_north_ms": -60.0,
                    "vel_up_ms": 0.0,
                }
            ],
        }
        for t in (1000, 2000, 3000)
    ]
)


def _rec(**kw):
    base = {
        "outcome": "published",
        "solve_key": "mn-dark-1",
        "measurement_ts_ms": 2000,
        "gt_hex": "e00065",
        "alt_m": 1500.0,
        "vel_east": -50.0,
        "vel_north": -60.0,
    }
    return {**base, **kw}


_TWO = _rec(ts_ms=2100, n_nodes=2, lat=49.169856, lon=-123.159828, raw_lat=49.169856, raw_lon=-123.159828)
_THREE = _rec(ts_ms=2900, n_nodes=3, lat=49.169856, lon=-123.159828, raw_lat=49.1555, raw_lon=-123.0980)


def test_same_epoch_published_twice_counts_once_as_the_later_solve():
    kept = one_per_epoch([_TWO, _THREE])
    assert kept == [_THREE]


def test_scored_on_the_raw_solve_not_the_held_map_position():
    (row,) = solve_rows([_TWO, _THREE], _TRUTH)
    assert row["n_nodes"] == 3
    assert row["err_horiz_m"] < 100.0
    assert row["map_err_horiz_m"] > 4000.0


def test_different_epochs_and_keys_are_all_kept():
    later = _rec(ts_ms=3100, n_nodes=3, measurement_ts_ms=3000, lat=49.15, lon=-123.1, raw_lat=49.15, raw_lon=-123.1)
    other_key = dict(_THREE, solve_key="mn-dark-2")
    assert len(one_per_epoch([_TWO, _THREE, later, other_key])) == 3


def test_unpublished_and_adsb_assisted_records_are_left_out():
    shadow = _rec(ts_ms=2200, n_nodes=2, outcome="dark_follow_shadow", lat=None, lon=None, raw_lat=49.1, raw_lon=-123.1)
    assisted = _rec(ts_ms=2300, n_nodes=3, adsb_hex="e00065", lat=49.1, lon=-123.1)
    assert one_per_epoch([shadow, assisted]) == []


def test_velocity_error_is_scored():
    (row,) = solve_rows([dict(_THREE, vel_east=-47.0, vel_north=-56.0)], _TRUTH)
    assert row["verr_horiz_ms"] == 5.0


def test_scored_on_this_seconds_solve_not_a_multi_epoch_fit():
    fitted = _rec(
        ts_ms=2900,
        n_nodes=2,
        lat=49.2,
        lon=-123.3,
        raw_lat=49.2,
        raw_lon=-123.3,
        pos_source="cv_fit",
        solve_raw_lat=49.1555,
        solve_raw_lon=-123.0980,
        vel_source="cv_fit",
        vel_east=40.0,
        vel_north=40.0,
        solver_vel_east=-50.0,
        solver_vel_north=-60.0,
    )
    (row,) = solve_rows([fitted], _TRUTH)
    assert row["err_horiz_m"] < 100.0
    assert row["verr_horiz_ms"] == 0.0
    assert row["pub_verr_horiz_ms"] > 100.0
    assert (row["pos_source"], row["vel_source"]) == ("cv_fit", "cv_fit")


def test_adopted_node_is_flagged():
    widened = _rec(ts_ms=2900, n_nodes=3, n_nodes_pre_adopt=2, epoch_skew_s=5.0, lat=49.15, lon=-123.1)
    (row,) = solve_rows([widened], _TRUTH)
    assert row["adopted"] is True
    assert row["epoch_skew_s"] == 5.0
