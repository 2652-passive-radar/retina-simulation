"""The receiver's 3-element antenna array, for the direction-finding study.

The array lies flat at the receiver and faces the surveillance beam's centre
(its *boresight*).  Two layouts:

* ``triangle`` (default): an equilateral triangle with side ``spacing_wl``
  wavelengths; element 0 points ``element0_deg`` from boresight, elements 1
  and 2 follow at +120 and +240 deg.
* ``line``: three elements across the boresight, ``spacing_wl`` apart, left
  to right as seen looking along boresight.

``snapshots`` builds what the array would hand the direction finder for each
target detection: the complex value of the target's cell on each element after
the cross-ambiguity (one snapshot per detection).  The error model, each term
switchable and scaled by ``ArrayErrors``:

* thermal noise at the detection's own SNR (plus ``snr_offset_db``);
* a fixed phase error per element (per run) and a fixed offset shared by the
  two channels on the second RSPduo (per run);
* a random phase error per element and detection;
* coupling between elements, not compensated (per run);
* one coherent multipath copy of the echo from a random direction (per
  detection);
* a heading error and a tilt of the whole array (per run).

Gain errors are left out: with one source in the cell they do not move the
bearing (see experiments/NOTEBOOK.md, direction-finding study).

Randomness: every draw comes from a generator keyed by the run (seed and
realisation), separate from the recording's own streams, so the recordings are
untouched.  Draws are standard normals or uniforms scaled by the error sizes
at use, so two error levels see the same underlying draws (a paired
comparison).
"""

import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np

C_M_S = 299_792_458.0


@dataclass
class ArrayConfig:
    layout: str = "triangle"
    spacing_wl: float = 0.5
    element0_deg: float = 0.0
    # The element on the RSPduo that also carries the reference.  None: element
    # 0 for the triangle, the centre element (1) for the line, the wiring for
    # which an offset between the two RSPduos biases the bearing least.
    device_a_element: int | None = None

    def device_a(self) -> int:
        if self.device_a_element is not None:
            return self.device_a_element
        return 1 if self.layout == "line" else 0


@dataclass
class ArrayErrors:
    thermal: bool = True
    snr_offset_db: float = 0.0
    phase_fixed_deg: float = 5.0
    device_offset_deg: float = 0.0
    phase_random_deg: float = 2.0
    coupling_db: float | None = -20.0
    multipath_db: float | None = -15.0
    heading_deg: float = 1.0
    tilt_deg: float = 1.0

    @classmethod
    def none(cls) -> "ArrayErrors":
        """Every error off: a perfect array (noise-free snapshots)."""
        return cls(False, 0.0, 0.0, 0.0, 0.0, None, None, 0.0, 0.0)

    def to_dict(self) -> dict:
        return asdict(self)


def element_offsets(cfg: ArrayConfig, wavelength_m: float) -> np.ndarray:
    """Element positions in the array's own frame (right, forward), metres."""
    s = cfg.spacing_wl * wavelength_m
    if cfg.layout == "triangle":
        r = s / math.sqrt(3.0)
        a = np.radians(cfg.element0_deg + np.array([0.0, 120.0, 240.0]))
        return np.stack([r * np.sin(a), r * np.cos(a)], axis=1)
    if cfg.layout == "line":
        return np.array([[-s, 0.0], [0.0, 0.0], [s, 0.0]])
    raise ValueError(f"unknown array layout {cfg.layout!r}")


def positions(
    cfg: ArrayConfig, heading_deg: float, wavelength_m: float, tilt_deg: tuple[float, float] = (0.0, 0.0)
) -> np.ndarray:
    """Element positions east/north/up of the array centre, metres, for an
    array whose boresight points ``heading_deg`` (compass) and that is tilted
    by ``tilt_deg`` about the east and north axes."""
    xy = element_offsets(cfg, wavelength_m)
    h = math.radians(heading_deg)
    east = xy[:, 0] * math.cos(h) + xy[:, 1] * math.sin(h)
    north = -xy[:, 0] * math.sin(h) + xy[:, 1] * math.cos(h)
    p = np.stack([east, north, np.zeros(len(xy))], axis=1)
    tx, ty = np.radians(tilt_deg)
    rot_e = np.array([[1, 0, 0], [0, math.cos(tx), -math.sin(tx)], [0, math.sin(tx), math.cos(tx)]])
    rot_n = np.array([[math.cos(ty), 0, math.sin(ty)], [0, 1, 0], [-math.sin(ty), 0, math.cos(ty)]])
    return p @ (rot_n @ rot_e).T


def unit_vectors(az_deg, el_deg) -> np.ndarray:
    """East/north/up unit vectors for compass azimuths and elevations (deg)."""
    az, el = np.radians(np.asarray(az_deg, float)), np.radians(np.asarray(el_deg, float))
    return np.stack(np.broadcast_arrays(np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)), axis=-1)


def steering(pos: np.ndarray, u: np.ndarray, wavelength_m: float) -> np.ndarray:
    """Phase each element sees for arrivals from unit vectors ``u`` (..., 3):
    exp(j 2 pi p.u / lambda), shape (..., n_elements)."""
    return np.exp(2j * math.pi * (u @ pos.T) / wavelength_m)


def keyed_rng(*key) -> np.random.Generator:
    digest = hashlib.sha256(repr(key).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def coupling_matrix(cfg: ArrayConfig, wavelength_m: float, level_db: float | None, phase_rad: float) -> np.ndarray:
    """I plus coupling c between every pair, c = level at one spacing (with the
    run's phase), falling off as 1/distance for pairs further apart."""
    xy = element_offsets(cfg, wavelength_m)
    n = len(xy)
    if level_db is None:
        return np.eye(n, dtype=complex)
    s = cfg.spacing_wl * wavelength_m
    c = 10 ** (level_db / 20.0) * np.exp(1j * phase_rad)
    out = np.eye(n, dtype=complex)
    for i in range(n):
        for j in range(n):
            if i != j:
                out[i, j] = c * s / float(np.hypot(*(xy[i] - xy[j])))
    return out


def snapshots(
    u_target: np.ndarray,
    snr_db: np.ndarray,
    cfg: ArrayConfig,
    err: ArrayErrors,
    boresight_deg: float,
    wavelength_m: float,
    run_key: tuple,
) -> np.ndarray:
    """One snapshot per detection, shape (n_detections, 3).

    ``u_target`` (n, 3): true unit vectors from the receiver to the aircraft
    (east/north/up).  ``boresight_deg`` is where the array is meant to point;
    it actually points there plus the run's heading error, tilted by the run's
    tilt.  ``run_key`` identifies the run (e.g. ``(seed, realisation)``): the
    same key and detections always give the same snapshots."""
    n_det, n_el = len(u_target), 3
    fixed = keyed_rng("doa-run", *run_key)
    z_phase = fixed.standard_normal(n_el)
    z_device = fixed.standard_normal()
    coupling_phase = fixed.uniform(0, 2 * math.pi)
    z_heading = fixed.standard_normal()
    z_tilt = fixed.standard_normal(2) / math.sqrt(2.0)

    det = keyed_rng("doa-det", *run_key)
    z_random = det.standard_normal((n_det, n_el))
    mp_az = det.uniform(0, 360, n_det)
    mp_phase = det.uniform(0, 2 * math.pi, n_det)
    sig_phase = det.uniform(0, 2 * math.pi, n_det)
    noise = (det.standard_normal((n_det, n_el)) + 1j * det.standard_normal((n_det, n_el))) / math.sqrt(2.0)

    pos = positions(cfg, boresight_deg + err.heading_deg * z_heading, wavelength_m, tuple(err.tilt_deg * z_tilt))
    x = steering(pos, u_target, wavelength_m)
    if err.multipath_db is not None:
        rho = 10 ** (err.multipath_db / 20.0)
        x = x + rho * np.exp(1j * mp_phase)[:, None] * steering(pos, unit_vectors(mp_az, 0.0), wavelength_m)
    phase = np.radians(err.phase_fixed_deg) * z_phase[None, :] + np.radians(err.phase_random_deg) * z_random
    device_b = np.ones(n_el)
    device_b[cfg.device_a()] = 0.0
    phase = phase + np.radians(err.device_offset_deg) * z_device * device_b[None, :]
    x = x * np.exp(1j * phase)
    x = x @ coupling_matrix(cfg, wavelength_m, err.coupling_db, coupling_phase).T
    x = x * np.exp(1j * sig_phase)[:, None]
    if err.thermal:
        amp = np.sqrt(10 ** ((np.asarray(snr_db, float) + err.snr_offset_db) / 10.0))
        x = amp[:, None] * x + noise
    return x
