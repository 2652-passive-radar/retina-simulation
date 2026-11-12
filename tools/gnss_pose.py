"""Receiver poses estimated from GNSS, in the format offline_solve uses (step 3c).

For one recording (a motion case and a seed), the receiver's GNSS fixes come
from the step 3a error model with the same keys step 3b used (``seed-<n>``,
realisation r), so 3b and 3c see identical GNSS errors.  Each radar tick then
gets a pose from one source:

``truth``  the recording's own rx_trajectory (no estimate; offline_solve's default)
``gnss``   the latest GNSS fix, velocity by differencing it with the one before
           (rx_filter's ``raw`` variant)
``kf``     the receiver Kalman filter with the GNSS drift in its state (step 3b)

A pose is, per node, the estimated position (lat, lon, alt_m) and velocity
(east, north, up, m/s), plus the estimated offset from the registered
position, the same fields rx_trajectory.ndjson carries.  Every node of a site
rides one platform, so each node's pose is its registered position moved by
the one estimated offset.
"""

import json
import os
import sys

import rx_filter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from retina_simulation import gnss  # noqa: E402
from retina_simulation.motion import motion_from  # noqa: E402
from retina_simulation.world import _enu_to_lla  # noqa: E402

SOURCES = ("truth", "gnss", "kf")
_VARIANT = {"gnss": "raw", "kf": "kf"}


def solve_name(source: str, realisation: int | None) -> str:
    """The solve file of one pose source (and GNSS realisation) in a seed's analysis folder."""
    return f"offline_solves_gd_{source}" + ("" if realisation is None else f"_r{realisation}") + ".csv"


def gnss_key(seed: int) -> str:
    return f"seed-{seed}"


def tick_times(exp: dict) -> list[float]:
    """Radar ticks in seconds from the start of the run (record_run's clock)."""
    tick = exp.get("tick_s", 1.0)
    return [i * tick for i in range(1, int(round(exp["duration_s"] / tick)) + 1)]


def estimates(exp: dict, case: dict, seed: int, realisation: int, source: str) -> list[dict | None]:
    """The source's receiver estimate at every radar tick (rx_filter.run output)."""
    if source not in _VARIANT:
        raise ValueError(f"no estimate for pose source {source!r}")
    model = gnss.model_from(exp["gnss"])
    motion = motion_from(case["motion"])
    fixes = gnss.fixes(model, motion, exp["duration_s"], gnss_key(seed), realisation)
    return rx_filter.run(fixes, tick_times(exp), rx_filter.config_for(exp, case), _VARIANT[source])


def poses_from_estimates(exp: dict, nodes: list[dict], ests: list[dict | None]) -> dict[int, dict]:
    """{t_ms: {node_id: pose}} from per-tick estimates (east/north/up metres and
    m/s relative to the registered position).  A tick with no estimate yet (no
    fix has arrived) gets no pose, and offline_solve then solves it as S would."""
    out = {}
    for t, est in zip(tick_times(exp), ests):
        if est is None:
            continue
        t_ms = exp["start_time_ms"] + int(round(t * 1000))
        pose = {}
        for n in nodes:
            alt0_km = n["rx_alt_ft"] * 0.3048 / 1000.0
            lat, lon, alt_km = _enu_to_lla(
                est["east_m"] / 1000.0, est["north_m"] / 1000.0, est["up_m"] / 1000.0, n["rx_lat"], n["rx_lon"], alt0_km
            )
            pose[n["node_id"]] = {
                "lat": lat,
                "lon": lon,
                "alt_m": alt_km * 1000.0,
                "vel_east_ms": est["vel_east_ms"],
                "vel_north_ms": est["vel_north_ms"],
                "vel_up_ms": est["vel_up_ms"],
                "offset_east_m": est["east_m"],
                "offset_north_m": est["north_m"],
            }
        out[t_ms] = pose
    return out


def poses_for(run_dir: str, exp: dict, case: dict, seed: int, realisation: int, source: str) -> dict | None:
    """Pose estimates for one recording, or None for ``truth`` (offline_solve's default)."""
    if source == "truth":
        return None
    with open(os.path.join(run_dir, "nodes.json"), encoding="utf-8") as f:
        nodes = json.load(f)
    return poses_from_estimates(exp, nodes, estimates(exp, case, seed, realisation, source))
