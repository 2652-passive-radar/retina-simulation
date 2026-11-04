"""Receiver position and velocity from GNSS position fixes (step 3b).

Per axis (east, north, up), a linear Kalman filter on the state [x, v, nu]:
position, velocity, and the time-correlated part of the GNSS error, which the
filter models as the same first-order Gauss-Markov process the simulator uses
(retina_simulation/gnss.py).

    x_k = F_k x_{k-1} + w_k,   w_k ~ N(0, Q_k)
    z_k = H x_k + eps_k,       eps_k ~ N(0, R),  H = [1 0 1],  R = sigma_white^2

    F_k = [[1, dt, 0], [0, 1, 0], [0, 0, a_k]],   a_k = exp(-dt / T)
    Q_k = [[sa^2 dt^4/4, sa^2 dt^3/2, 0], [sa^2 dt^3/2, sa^2 dt^2, 0], [0, 0, sigma_corr^2 (1 - a_k^2)]]

Acceleration is white noise held constant over each step (sigma_a, "sa"),
fixed per platform mode.  The filter starts at the first fix:
x_0 = [z_0, 0, 0] with

    P_0 = [[sigma_corr^2 + sigma_white^2, 0, -sigma_corr^2],
           [0, sigma_v0^2, 0],
           [-sigma_corr^2, 0, sigma_corr^2]]

(x_0 = z_0 means the position error is -(nu + eps), hence the correlation),
and updates (Joseph form) on every later fix.  At each query time it predicts,
without updating, from the last fix that has *arrived* by then
(``t_avail_s``), so a known latency is respected.

Two baselines for comparison:

``white``  the same filter without nu: state [x, v], R = sigma_white^2 +
           sigma_corr^2, i.e. all GNSS error treated as white
``raw``    no filter: the latest arrived fix, velocity from it and the one
           before, extrapolated to the query time
"""

import math
from dataclasses import dataclass

import numpy as np

AXES = ("east", "north", "up")
VARIANTS = ("kf", "white", "raw")


@dataclass(frozen=True)
class FilterConfig:
    sigma_a: float = 0.5
    sigma_v0: float = 20.0
    sigma_white: float = 0.5
    sigma_corr: float = 1.5
    tau_s: float = 60.0


def _a(cfg: FilterConfig, dt: float) -> float:
    return math.exp(-dt / cfg.tau_s) if cfg.tau_s > 0 else 0.0


def _fq(cfg: FilterConfig, dt: float, with_nu: bool) -> tuple[np.ndarray, np.ndarray]:
    s2 = cfg.sigma_a**2
    q = np.array([[s2 * dt**4 / 4, s2 * dt**3 / 2], [s2 * dt**3 / 2, s2 * dt**2]])
    f = np.array([[1.0, dt], [0.0, 1.0]])
    if not with_nu:
        return f, q
    a = _a(cfg, dt)
    f3, q3 = np.zeros((3, 3)), np.zeros((3, 3))
    f3[:2, :2], q3[:2, :2] = f, q
    f3[2, 2] = a
    q3[2, 2] = cfg.sigma_corr**2 * (1.0 - a * a)
    return f3, q3


class AxisFilter:
    """One axis.  ``with_nu`` False is the white-noise baseline."""

    def __init__(self, cfg: FilterConfig, z0: float, t0: float, with_nu: bool = True):
        self.cfg, self.with_nu, self.t = cfg, with_nu, t0
        sc2, sw2 = cfg.sigma_corr**2, cfg.sigma_white**2
        if with_nu:
            self.x = np.array([z0, 0.0, 0.0])
            self.P = np.array([[sc2 + sw2, 0.0, -sc2], [0.0, cfg.sigma_v0**2, 0.0], [-sc2, 0.0, sc2]])
            self.H = np.array([1.0, 0.0, 1.0])
            self.R = sw2
        else:
            self.x = np.array([z0, 0.0])
            self.P = np.diag([sc2 + sw2, cfg.sigma_v0**2])
            self.H = np.array([1.0, 0.0])
            self.R = sc2 + sw2

    def predicted(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """State and covariance at ``t`` >= the last update, without changing the filter."""
        dt = t - self.t
        if dt <= 0:
            return self.x.copy(), self.P.copy()
        f, q = _fq(self.cfg, dt, self.with_nu)
        return f @ self.x, f @ self.P @ f.T + q

    def update(self, z: float, t: float) -> float:
        """Predict to ``t`` and fold in the fix; returns the normalised innovation squared."""
        x, p = self.predicted(t)
        s = float(self.H @ p @ self.H) + self.R
        y = z - float(self.H @ x)
        k = p @ self.H / s
        ikh = np.eye(len(x)) - np.outer(k, self.H)
        self.x = x + k * y
        self.P = ikh @ p @ ikh.T + self.R * np.outer(k, k)
        self.t = t
        return y * y / s


def run(fixes: list[dict], query_s: list[float], cfg: FilterConfig, variant: str = "kf") -> list[dict | None]:
    """The receiver estimate at each query time, from the fixes that had
    arrived by then.  None before the first fix arrives.

    Each estimate: ``t_s``, per axis ``<axis>_m`` and ``vel_<axis>_ms`` and
    their standard deviations ``sd_<axis>_m`` / ``sd_vel_<axis>_ms``, the
    position/velocity covariance ``cov_pv`` (6x6, east/north/up positions then
    velocities), and the mean NIS of the updates since the last query."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown filter variant {variant!r}")
    order = sorted(fixes, key=lambda f: (f["t_avail_s"], f["t_s"]))
    filters: dict[str, AxisFilter] = {}
    seen: list[dict] = []
    out: list[dict | None] = []
    i = 0
    for tq in query_s:
        nis = []
        while i < len(order) and order[i]["t_avail_s"] <= tq + 1e-9:
            fix = order[i]
            i += 1
            seen.append(fix)
            if variant == "raw":
                continue
            for axis in AXES:
                if axis not in filters:
                    filters[axis] = AxisFilter(cfg, fix[f"{axis}_m"], fix["t_s"], variant == "kf")
                else:
                    nis.append(filters[axis].update(fix[f"{axis}_m"], fix["t_s"]))
        if not seen:
            out.append(None)
            continue
        est = {"t_s": tq, "nis": float(np.mean(nis)) if nis else None}
        if variant == "raw":
            last = seen[-1]
            prev = seen[-2] if len(seen) > 1 else None
            for axis in AXES:
                v = (last[f"{axis}_m"] - prev[f"{axis}_m"]) / (last["t_s"] - prev["t_s"]) if prev else 0.0
                est[f"{axis}_m"] = last[f"{axis}_m"] + v * (tq - last["t_s"])
                est[f"vel_{axis}_ms"] = v
            out.append(est)
            continue
        cov = np.zeros((6, 6))
        for j, axis in enumerate(AXES):
            x, p = filters[axis].predicted(tq)
            est[f"{axis}_m"], est[f"vel_{axis}_ms"] = float(x[0]), float(x[1])
            est[f"sd_{axis}_m"], est[f"sd_vel_{axis}_ms"] = math.sqrt(p[0, 0]), math.sqrt(p[1, 1])
            cov[np.ix_([j, j + 3], [j, j + 3])] = p[:2, :2]
        est["cov_pv"] = cov
        out.append(est)
    return out


def nees(est: dict, truth: dict) -> float:
    """Normalised estimation error squared of [position, velocity] (6 dof)."""
    e = np.array(
        [truth[f"{a}_m"] - est[f"{a}_m"] for a in AXES] + [truth[f"vel_{a}_ms"] - est[f"vel_{a}_ms"] for a in AXES]
    )
    return float(e @ np.linalg.solve(est["cov_pv"], e))
