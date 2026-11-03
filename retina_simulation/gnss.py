"""GNSS position fixes of the moving receiver, with a realistic error model.

The receiver's GNSS gives position only (no velocity).  Each fix, per axis
(east, north, up), is

    p_meas(t_i) = p_true(t_i - time_offset) + nu_i + eps_i

``nu`` is a first-order Gauss-Markov (AR(1)) process, the time-correlated
part of GNSS error (multipath, atmosphere, the receiver's own smoothing):

    nu_i = a_i * nu_{i-1} + w_i,   a_i = exp(-(t_i - t_{i-1}) / T),
    w_i ~ N(0, sigma_corr^2 * (1 - a_i^2)),   nu_0 ~ N(0, sigma_corr^2)

``sigma_corr`` is the process's steady-state standard deviation, so the size
of the error does not change with T or with the update rate; ``w``'s variance
is derived from it.  ``eps`` is white noise, N(0, sigma_white^2).  The three
axes are independent and share the same parameters.

Timing:

``time_offset_s``  unknown: a fix describes the receiver ``time_offset_s``
                   earlier than its timestamp says, and nothing corrects it
``latency_s``      known: a fix becomes available ``latency_s`` after its
                   timestamp; a filter can predict past it

Fixes are stamped t_i = phase + i / rate, phase drawn uniformly in one update
interval, so the GNSS clock does not line up with the radar's ticks.  The
antenna sits on the surveillance antenna (no lever arm).

Positions are metres east/north/up of the receiver's registered position, in
the same fixed local frame as motion.py.  Every random draw comes from a
stream keyed by (key, realisation, axis), independent of the radar's noise.
"""

import json
import math
import random
from dataclasses import asdict, dataclass

from retina_simulation.motion import Motion

AXES = ("east", "north", "up")


@dataclass(frozen=True)
class GnssModel:
    sigma_white_m: float = 0.5
    sigma_corr_m: float = 1.5
    tau_s: float = 60.0
    rate_hz: float = 1.0
    time_offset_s: float = 0.0
    latency_s: float = 0.0

    def __post_init__(self):
        if self.rate_hz <= 0 or self.tau_s < 0 or self.sigma_white_m < 0 or self.sigma_corr_m < 0:
            raise ValueError(f"invalid GNSS model {self}")

    def coefficient(self, dt: float) -> float:
        """The AR(1) coefficient a for a step of ``dt`` seconds (0 when T is 0: white)."""
        return math.exp(-dt / self.tau_s) if self.tau_s > 0 else 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def model_from(cfg: dict | None) -> GnssModel:
    return GnssModel(**(cfg or {}))


def _rng(key: str, realisation: int, *rest) -> random.Random:
    return random.Random(":".join(str(k) for k in (key, "gnss", realisation, *rest)))


def stamps(model: GnssModel, duration_s: float, key: str = "", realisation: int = 0) -> list[float]:
    """Fix timestamps covering [0, duration_s], at a random phase."""
    step = 1.0 / model.rate_hz
    phase = _rng(key, realisation, "phase").uniform(0.0, step)
    n = int(math.floor((duration_s - phase) / step)) + 1
    return [phase + i * step for i in range(max(n, 0))]


def correlated_error(model: GnssModel, times: list[float], rng: random.Random) -> list[float]:
    """One axis of the Gauss-Markov error at ``times``, started stationary."""
    out = []
    nu = rng.gauss(0.0, model.sigma_corr_m)
    prev = None
    for t in times:
        if prev is not None:
            a = model.coefficient(t - prev)
            nu = a * nu + rng.gauss(0.0, model.sigma_corr_m * math.sqrt(1.0 - a * a))
        out.append(nu)
        prev = t
    return out


def errors(model: GnssModel, times: list[float], key: str = "", realisation: int = 0) -> dict[str, dict[str, list]]:
    """Per axis, the correlated and white parts of the error at ``times``."""
    out = {}
    for axis in AXES:
        nu = correlated_error(model, times, _rng(key, realisation, axis, "corr"))
        white = _rng(key, realisation, axis, "white")
        out[axis] = {"corr": nu, "white": [white.gauss(0.0, model.sigma_white_m) for _ in times]}
    return out


def _true_enu(motion: Motion, t: float) -> tuple[float, float, float]:
    e, n, _, _ = motion.state(t)
    return e, n, 0.0


def fixes(model: GnssModel, motion: Motion, duration_s: float, key: str = "", realisation: int = 0) -> list[dict]:
    """Every fix of one run: measured position, the true position at the
    timestamp, and the two error parts per axis."""
    times = stamps(model, duration_s, key, realisation)
    err = errors(model, times, key, realisation)
    out = []
    for i, t in enumerate(times):
        seen = _true_enu(motion, t - model.time_offset_s)
        truth = _true_enu(motion, t)
        fix = {"t_s": round(t, 6), "t_avail_s": round(t + model.latency_s, 6)}
        for k, axis in enumerate(AXES):
            corr, white = err[axis]["corr"][i], err[axis]["white"][i]
            fix[f"{axis}_m"] = round(seen[k] + corr + white, 4)
            fix[f"true_{axis}_m"] = round(truth[k], 4)
            fix[f"corr_{axis}_m"] = round(corr, 4)
            fix[f"white_{axis}_m"] = round(white, 4)
        out.append(fix)
    return out


def write_fixes(path: str, model: GnssModel, rows: list[dict]) -> None:
    """``gnss.ndjson``: the model on the first line, then one fix per line."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"model": model.to_dict()}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def autocorrelation(x: list[float], max_lag: int) -> list[float]:
    """Sample autocorrelation of ``x`` for lags 0..max_lag (mean removed)."""
    n = len(x)
    mean = sum(x) / n
    d = [v - mean for v in x]
    var = sum(v * v for v in d) / n
    return [sum(d[i] * d[i + k] for i in range(n - k)) / (n * var) for k in range(max_lag + 1)]
