"""tools/offline_solve.py against a recorded baseline run.

Needs retina_geolocator, which this repo's CI does not install, so the module
skips there; it runs wherever the backend's environment is used.
"""

import json
import math
import os
import sys

import pytest

pytest.importorskip("retina_geolocator")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "tools"))

import offline_solve  # noqa: E402

from retina_simulation import experiment  # noqa: E402

_BASELINE = os.path.join(_HERE, "..", "experiments", "baseline_static.json")


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    exp = experiment.load(_BASELINE)
    d = str(tmp_path_factory.mktemp("seed-202"))
    experiment.record_run(exp, exp["runs"][1], d)
    return d


def _true_measurements(run_dir):
    with open(os.path.join(run_dir, "truth.ndjson"), encoding="utf-8") as f:
        truth = {t["t_ms"]: t["aircraft"][0] for t in map(json.loads, f)}
    by_t = {}
    with open(os.path.join(run_dir, "frames.ndjson"), encoding="utf-8") as f:
        for fr in map(json.loads, f):
            for d in fr["detections"]:
                if not d["is_clutter"]:
                    by_t.setdefault(fr["t_ms"], []).append(
                        {
                            "node_id": fr["node_id"],
                            "delay_us": d["delay_true"],
                            "doppler_hz": d["doppler_true"],
                            "snr": d["snr_db"],
                        }
                    )
    return truth, {t: m for t, m in by_t.items() if len(m) == 3}


def test_initial_guess_is_near_the_aircraft_from_radar_alone(run_dir):
    with open(os.path.join(run_dir, "nodes.json"), encoding="utf-8") as f:
        geo = offline_solve.Geometry(json.load(f))
    truth, epochs = _true_measurements(run_dir)
    errs = []
    for t, meas in list(epochs.items())[:20]:
        guess = geo.initial_guess(meas, offline_solve.DEFAULT_SOLVER["alt_layers_km"])
        e, n = offline_solve._en_m(truth[t]["lat"], truth[t]["lon"], guess["lat"], guess["lon"])
        errs.append(math.hypot(e, n))
    assert max(errs) < 2000.0


def test_every_three_node_tick_is_solved_and_scored(run_dir):
    rows = offline_solve.solve_run(run_dir)
    assert len(rows) == 300
    three = [r for r in rows if r["n_nodes"] == 3]
    assert three and all("err_horiz_m" in r for r in three if r["reason"] != "solve_failed")
    assert all(not r["published"] for r in rows if r["n_nodes"] < 3)
    published = [r for r in rows if r["published"]]
    assert published
    rmse = math.sqrt(sum(r["err_horiz_m"] ** 2 for r in published) / len(published))
    assert rmse < 300.0


def test_offline_solve_is_deterministic(run_dir):
    assert offline_solve.solve_run(run_dir) == offline_solve.solve_run(run_dir)
