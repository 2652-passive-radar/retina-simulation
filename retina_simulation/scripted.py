"""Scripted single flights for controlled experiments.

A ``FlightPlan`` is a start point, altitude, speed, heading and a constant turn
rate.  ``sample_plan`` draws one from a seed and redraws until the whole flight
stays inside every node's coverage with a margin, so a 300 s run never loses
the aircraft to geometry.  ``SimulationWorld`` flies it with the same
kinematics as any other aircraft (``turn_rate_deg_s`` replaces waypoint
steering), so the check here and the flown path agree to the per-tick jitter.
"""

import math
import random
from dataclasses import asdict, dataclass

from retina_simulation.world import (
    R_EARTH,
    NodeConfig,
    SimulatedAircraft,
    _advance_latlon,
    _bearing_deg,
    _haversine_km,
    _lla_to_enu,
)


@dataclass
class FlightPlan:
    lat: float
    lon: float
    alt_m: float
    speed_ms: float
    heading_deg: float
    turn_rate_deg_s: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(v, 6) for k, v in asdict(self).items()}


@dataclass
class Coverage:
    """Where a scripted flight is allowed to be, relative to every node."""

    beam_margin_deg: float = 10.0
    min_range_km: float = 3.0
    max_range_km: float = 35.0
    max_bistatic_angle_deg: float = 150.0


@dataclass
class FlightRanges:
    speed_ms: tuple[float, float] = (70.0, 110.0)
    alt_m: tuple[float, float] = (1000.0, 3000.0)
    turn_rate_deg_s: tuple[float, float] = (0.3, 1.0)


def propagate(plan: FlightPlan, duration_s: float, dt: float = 1.0) -> list[tuple[float, float, float]]:
    """Noise-free (lat, lon, alt_km) every dt from t=0 to duration_s, flown as
    ``SimulationWorld._update_aircraft`` flies it: move, then turn."""
    lat, lon, alt_km = plan.lat, plan.lon, plan.alt_m / 1000.0
    heading = plan.heading_deg
    speed = plan.speed_ms / 1000.0
    out = [(lat, lon, alt_km)]
    for _ in range(int(round(duration_s / dt))):
        ve = speed * math.sin(math.radians(heading))
        vn = speed * math.cos(math.radians(heading))
        lat, lon = _advance_latlon(lat, lon, alt_km, ve, vn, dt)
        heading = (heading + plan.turn_rate_deg_s * dt) % 360
        out.append((lat, lon, alt_km))
    return out


def bistatic_angle_deg(node: NodeConfig, lat: float, lon: float, alt_km: float) -> float:
    """Angle at the target between the directions to TX and to RX."""
    rx_alt_km = node.rx_alt_ft * 0.3048 / 1000.0
    tgt = _lla_to_enu(lat, lon, alt_km, node.rx_lat, node.rx_lon, rx_alt_km)
    tx = _lla_to_enu(node.tx_lat, node.tx_lon, node.tx_alt_ft * 0.3048 / 1000.0, node.rx_lat, node.rx_lon, rx_alt_km)
    to_rx = [-c for c in tgt]
    to_tx = [tx[i] - tgt[i] for i in range(3)]
    dot = sum(a * b for a, b in zip(to_rx, to_tx))
    norm = math.sqrt(sum(a * a for a in to_rx)) * math.sqrt(sum(b * b for b in to_tx))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / norm))))


def point_covered(node: NodeConfig, lat: float, lon: float, alt_km: float, cov: Coverage) -> bool:
    rng_km = _haversine_km(node.rx_lat, node.rx_lon, lat, lon)
    if not cov.min_range_km <= rng_km <= min(cov.max_range_km, node.max_range_km):
        return False
    off = abs((_bearing_deg(node.rx_lat, node.rx_lon, lat, lon) - node.beam_azimuth_deg + 180) % 360 - 180)
    if off > node.beam_width_deg / 2 - cov.beam_margin_deg:
        return False
    return bistatic_angle_deg(node, lat, lon, alt_km) <= cov.max_bistatic_angle_deg


def path_covered(path, nodes: list[NodeConfig], cov: Coverage) -> bool:
    return all(point_covered(n, *p, cov) for p in path for n in nodes)


def sample_plan(
    seed: int,
    kind: str,
    nodes: list[NodeConfig],
    duration_s: float,
    cov: Coverage | None = None,
    ranges: FlightRanges | None = None,
    max_tries: int = 50_000,
) -> FlightPlan:
    """A flight of ``kind`` ("straight" or "turn") that stays covered throughout."""
    if kind not in ("straight", "turn"):
        raise ValueError(f"unknown flight kind {kind!r}")
    cov = cov or Coverage()
    ranges = ranges or FlightRanges()
    rng = random.Random(f"flight:{seed}:{kind}")
    rx = nodes[0]
    half = min(n.beam_width_deg for n in nodes) / 2 - cov.beam_margin_deg
    for _ in range(max_tries):
        bearing = rx.beam_azimuth_deg + rng.uniform(-half, half)
        dist = rng.uniform(cov.min_range_km, cov.max_range_km)
        lat = rx.rx_lat + math.degrees(dist * math.cos(math.radians(bearing)) / R_EARTH)
        lon = rx.rx_lon + math.degrees(
            dist * math.sin(math.radians(bearing)) / (R_EARTH * math.cos(math.radians(rx.rx_lat)))
        )
        turn = 0.0
        if kind == "turn":
            turn = rng.uniform(*ranges.turn_rate_deg_s) * rng.choice((-1.0, 1.0))
        plan = FlightPlan(
            lat=lat,
            lon=lon,
            alt_m=rng.uniform(*ranges.alt_m),
            speed_ms=rng.uniform(*ranges.speed_ms),
            heading_deg=rng.uniform(0.0, 360.0),
            turn_rate_deg_s=turn,
        )
        if path_covered(propagate(plan, duration_s), nodes, cov):
            return plan
    raise RuntimeError(f"no covered {kind} flight found for seed {seed} in {max_tries} tries")


def aircraft_for(plan: FlightPlan, seed: int) -> SimulatedAircraft:
    """The plan as a world aircraft with ADS-B, never retired by the world."""
    speed = plan.speed_ms / 1000.0
    return SimulatedAircraft(
        object_id=f"flight-{seed}",
        lat=plan.lat,
        lon=plan.lon,
        alt_km=plan.alt_m / 1000.0,
        vel_east=speed * math.sin(math.radians(plan.heading_deg)),
        vel_north=speed * math.cos(math.radians(plan.heading_deg)),
        vel_up=0.0,
        heading_deg=plan.heading_deg,
        speed_km_s=speed,
        base_speed_km_s=speed,
        has_adsb=True,
        adsb_hex=f"e{seed:05x}",
        adsb_callsign=f"EXP{seed}",
        lifetime_s=1e9,
        turn_rate_deg_s=plan.turn_rate_deg_s,
    )
