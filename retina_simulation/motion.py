"""Receiver platform motion for the moving-receiver study.

One platform carries the surveillance antenna and every reference antenna,
so all nodes of a site share one trajectory.  A motion gives, at time t
(seconds from the start of the run), the platform's horizontal offset from
its registered position R0 and its velocity, both in a fixed local
east/north frame at R0 (metres and m/s).  Height is held at R0's: a ground
vehicle.

Profiles (see experiments/NOTEBOOK.md, step 1, for why each one):

``static``     offset 0
``linear``     constant speed along ``heading_deg``
``oscillate``  back and forth along ``heading_deg``; amplitude speed*T/(2 pi)
               so the peak speed is ``speed_ms``
``circle``     constant speed on a circle centred on R0, one lap every
               ``lap_s``; radius speed*lap/(2 pi); starts at
               ``start_bearing_deg`` from R0 and runs clockwise
"""

import math
from dataclasses import dataclass

from retina_simulation.world import SimulationWorld, _enu_to_lla


@dataclass(frozen=True)
class Motion:
    kind: str = "static"
    speed_ms: float = 0.0
    heading_deg: float = 0.0
    period_s: float = 100.0
    lap_s: float = 300.0
    start_bearing_deg: float = 0.0

    def __post_init__(self):
        if self.kind not in ("static", "linear", "oscillate", "circle"):
            raise ValueError(f"unknown receiver motion {self.kind!r}")

    @property
    def amplitude_m(self) -> float:
        return self.speed_ms * self.period_s / (2 * math.pi)

    @property
    def radius_m(self) -> float:
        return self.speed_ms * self.lap_s / (2 * math.pi)

    def state(self, t: float) -> tuple[float, float, float, float]:
        """(offset east m, offset north m, velocity east m/s, velocity north m/s) at t."""
        h = math.radians(self.heading_deg)
        ue, un = math.sin(h), math.cos(h)
        if self.kind == "static":
            return 0.0, 0.0, 0.0, 0.0
        if self.kind == "linear":
            d = self.speed_ms * t
            return d * ue, d * un, self.speed_ms * ue, self.speed_ms * un
        if self.kind == "oscillate":
            w = 2 * math.pi / self.period_s
            d = self.amplitude_m * math.sin(w * t)
            v = self.amplitude_m * w * math.cos(w * t)
            return d * ue, d * un, v * ue, v * un
        r, w = self.radius_m, 2 * math.pi / self.lap_s
        phi = math.radians(self.start_bearing_deg) + w * t
        return r * math.sin(phi), r * math.cos(phi), r * w * math.cos(phi), -r * w * math.sin(phi)


def motion_from(cfg: dict | None) -> Motion:
    cfg = dict(cfg or {"type": "static"})
    return Motion(kind=cfg.pop("type", "static"), **cfg)


def place_receivers(world: SimulationWorld, origins: dict[str, tuple[float, float, float]], motion: Motion, t: float):
    """Move every node's receiver to R0 + offset(t) and give it velocity(t).

    ``origins`` is each node's registered (lat, lon, alt_km).  Returns the
    offset and velocity used, for the recording."""
    de, dn, ve, vn = motion.state(t)
    for nid, (lat0, lon0, alt0_km) in origins.items():
        node = world.nodes[nid]
        node.rx_lat, node.rx_lon, _ = _enu_to_lla(de / 1000.0, dn / 1000.0, 0.0, lat0, lon0, alt0_km)
        node.rx_vel_enu = (ve / 1000.0, vn / 1000.0, 0.0)
    return de, dn, ve, vn
