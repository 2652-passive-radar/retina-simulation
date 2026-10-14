"""Reproducible single-flight experiments.

An experiment file (see ``experiments/baseline_static.json``) names a fleet,
a duration and a list of runs, each a seed and a flight kind.  The first time
it is resolved, every run's flight plan is sampled from its seed and written
back into the file together with a fingerprint of the recording it produced;
from then on the plan is read, never re-sampled, so a later change to the
sampler cannot move the aircraft, and the fingerprint says whether the
simulator still produces the same detections.

``record_run`` writes a run without a server and without waiting for real
time: the world steps in fixed ticks, every node is framed every tick, and
frames are stamped from the experiment's fixed start time.  The files are the
ones ``orchestrator --record`` writes (see recorder.py), plus
``rx_trajectory.ndjson``.
"""

import json
import math
import os

from retina_simulation.recorder import RunRecorder
from retina_simulation.scripted import Coverage, FlightPlan, FlightRanges, aircraft_for, sample_plan
from retina_simulation.world import NodeConfig, SimulationWorld

_FINGERPRINT_TOL = 0.1


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        exp = json.load(f)
    exp["_path"] = os.path.abspath(path)
    return exp


def save(exp: dict) -> None:
    body = {k: v for k, v in exp.items() if not k.startswith("_")}
    with open(exp["_path"], "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)
        f.write("\n")


def fleet_path(exp: dict) -> str:
    return os.path.normpath(os.path.join(os.path.dirname(exp["_path"]), exp["fleet"]))


def fleet_nodes(exp: dict) -> list[dict]:
    with open(fleet_path(exp), encoding="utf-8") as f:
        return json.load(f)["nodes"]


def node_config(cfg: dict) -> NodeConfig:
    """A fleet-file node as the world's NodeConfig (the orchestrator's mapping)."""
    return NodeConfig(
        node_id=cfg["node_id"],
        rx_lat=cfg["rx_lat"],
        rx_lon=cfg["rx_lon"],
        rx_alt_ft=cfg["rx_alt_ft"],
        tx_lat=cfg["tx_lat"],
        tx_lon=cfg["tx_lon"],
        tx_alt_ft=cfg["tx_alt_ft"],
        fc_hz=cfg["fc_hz"],
        fs_hz=cfg.get("fs_hz", 2_000_000),
        beam_width_deg=cfg.get("beam_width_deg", 42),
        max_range_km=cfg.get("max_range_km", 50),
        beam_azimuth_deg=cfg.get("beam_azimuth_deg"),
        max_bistatic_range_km=cfg.get("max_bistatic_range_km"),
    )


def coverage(exp: dict) -> Coverage:
    return Coverage(**exp.get("coverage", {}))


def flight_ranges(exp: dict) -> FlightRanges:
    return FlightRanges(**{k: tuple(v) for k, v in exp.get("flight_ranges", {}).items()})


def resolve_plans(exp: dict) -> bool:
    """Sample a plan for every run that has none; True if any was added."""
    if exp.get("receiver_motion", {"type": "static"}).get("type") != "static":
        raise NotImplementedError("only a static receiver is implemented")
    nodes = [node_config(c) for c in fleet_nodes(exp)]
    added = False
    for run in exp["runs"]:
        if "plan" not in run:
            plan = sample_plan(run["seed"], run["flight"], nodes, exp["duration_s"], coverage(exp), flight_ranges(exp))
            run["plan"] = plan.to_dict()
            added = True
    return added


def build_world(exp: dict, run: dict, nodes_raw: list[dict]) -> SimulationWorld:
    """The world for one run: the fleet's nodes, the run's one aircraft, no
    spawning, and every random draw keyed by the run's seed."""
    nodes = [node_config(c) for c in nodes_raw]
    world = SimulationWorld(center_lat=nodes[0].rx_lat, center_lon=nodes[0].rx_lon)
    for n in nodes:
        world.add_node(n)
    world.min_aircraft = world.max_aircraft = 0
    world.rng_seed = f"run:{run['seed']}"
    world.aircraft = [aircraft_for(FlightPlan(**run["plan"]), run["seed"])]
    return world


def receiver_pose(world: SimulationWorld) -> dict:
    """Every node's receiver position and velocity (static until motion lands)."""
    return {
        nid: {
            "lat": n.rx_lat,
            "lon": n.rx_lon,
            "alt_m": round(n.rx_alt_ft * 0.3048, 2),
            "vel_east_ms": 0.0,
            "vel_north_ms": 0.0,
            "vel_up_ms": 0.0,
        }
        for nid, n in world.nodes.items()
    }


def record_run(exp: dict, run: dict, out_dir: str) -> dict:
    """Record one run as fast as it computes; returns its fingerprint."""
    nodes_raw = fleet_nodes(exp)
    world = build_world(exp, run, nodes_raw)
    rec = RunRecorder(out_dir, nodes_raw)
    tick_s = exp.get("tick_s", 1.0)
    try:
        for i in range(1, int(round(exp["duration_s"] / tick_s)) + 1):
            world.step(tick_s)
            t_ms = exp["start_time_ms"] + int(round(i * tick_s * 1000))
            rec.truth(t_ms, world.aircraft)
            rec.receiver(t_ms, receiver_pose(world))
            for nid in world.nodes:
                labels: list[dict] = []
                frame = world.generate_detections_for_node(nid, t_ms, labels=labels)
                rec.frame(nid, frame, labels, sent=bool(frame.get("delay")))
    finally:
        rec.close()
    return fingerprint(out_dir)


def fingerprint(run_dir: str) -> dict:
    """A short, platform-tolerant summary of a recording, to tell whether two
    recordings of the same run are the same."""
    n_frames = n_det = n_target = 0
    sum_delay = sum_doppler = 0.0
    with open(os.path.join(run_dir, "frames.ndjson"), encoding="utf-8") as f:
        for line in f:
            fr = json.loads(line)
            n_frames += 1
            for d in fr["detections"]:
                n_det += 1
                if not d["is_clutter"]:
                    n_target += 1
                sum_delay += d["delay_us"]
                sum_doppler += d["doppler_hz"]
    last = None
    with open(os.path.join(run_dir, "truth.ndjson"), encoding="utf-8") as f:
        for line in f:
            last = json.loads(line)
    ac = last["aircraft"][0] if last and last["aircraft"] else {}
    return {
        "frames": n_frames,
        "detections": n_det,
        "target_detections": n_target,
        "sum_delay_us": round(sum_delay, 2),
        "sum_doppler_hz": round(sum_doppler, 2),
        "end_lat": ac.get("lat"),
        "end_lon": ac.get("lon"),
    }


def fingerprint_mismatch(expected: dict, got: dict) -> list[str]:
    """What differs, beyond rounding; empty when the recordings agree."""
    out = []
    for k, v in expected.items():
        g = got.get(k)
        if isinstance(v, int) and not isinstance(v, bool):
            if v != g:
                out.append(f"{k}: expected {v}, got {g}")
        elif isinstance(v, float):
            tol = 1e-4 if k.startswith("end_") else _FINGERPRINT_TOL
            if g is None or not math.isclose(v, g, abs_tol=tol):
                out.append(f"{k}: expected {v}, got {g}")
    return out
