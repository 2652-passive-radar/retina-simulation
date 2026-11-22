"""Direction of arrival from array snapshots: the Bartlett (delay-and-sum) scan.

One interface for simulated and, later, real snapshots::

    az, spectrum = doa(snapshots, positions, wavelength_m, az_grid_deg)

``snapshots`` (n_detections, n_elements) complex: one snapshot per detection,
the target cell's value on each element.  ``positions`` (n_elements, 3): the
elements east/north/up of the array centre, metres, as installed (nominal
heading, flat).  ``az_grid_deg``: compass azimuths to scan, e.g.
``front_half(boresight)``.

The scan assumes the arrival is horizontal.  For the flat triangle that costs
nothing (elevation scales the phases without turning them); for the line the
result is a cone angle, which ``doa_solve`` turns back into an azimuth with
the aircraft's elevation.  With one source and one snapshot the Bartlett peak
is also the maximum-likelihood estimate (and MUSIC would give the same peak).
"""

import numpy as np


def front_half(boresight_deg: float, step_deg: float = 0.25) -> np.ndarray:
    """Azimuths within 90 deg of boresight: the search that keeps the
    triangle's back lobe (and the line's mirror) out of the answer."""
    return boresight_deg + np.arange(-90.0, 90.0 + step_deg / 2, step_deg)


def doa(
    snapshots: np.ndarray, positions: np.ndarray, wavelength_m: float, az_grid_deg: np.ndarray, el_deg: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """(azimuth estimate per detection, compass deg; scan power per detection
    and grid azimuth, normalised so a perfect match is 1)."""
    x = np.atleast_2d(snapshots)
    az = np.radians(az_grid_deg)
    el = np.radians(el_deg)
    u = np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.full_like(az, np.sin(el))], axis=1)
    a = np.exp(2j * np.pi * (u @ positions.T) / wavelength_m)
    power = np.abs(x @ a.conj().T) ** 2
    norm = np.sum(np.abs(x) ** 2, axis=1, keepdims=True) * a.shape[1]
    spectrum = power / np.where(norm > 0, norm, 1.0)
    k = np.argmax(spectrum, axis=1)
    rows = np.arange(len(x))
    inner = (k > 0) & (k < len(az_grid_deg) - 1)
    kk = np.clip(k, 1, len(az_grid_deg) - 2)
    p0, p1, p2 = spectrum[rows, kk - 1], spectrum[rows, kk], spectrum[rows, kk + 1]
    den = p0 - 2 * p1 + p2
    shift = np.where(inner & (den != 0), 0.5 * (p0 - p2) / np.where(den != 0, den, 1.0), 0.0)
    step = az_grid_deg[1] - az_grid_deg[0]
    est = np.asarray(az_grid_deg)[k] + shift * step
    return est % 360.0, spectrum
