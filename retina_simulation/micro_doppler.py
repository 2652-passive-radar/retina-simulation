"""Rotor and propeller micro-Doppler for the detection-list simulator.

RETINA nodes illuminate with ~500 MHz DTV (wavelength ~0.6 m) and integrate over a
CPI of ~0.5-1 s, which is many rotor periods.  Over that long a window a rotating
blade does not smear into a flash: the periodic modulation resolves into a comb of
spectral lines at the target's delay, spaced by the blade-passing rate and spread
up to about +/- v_tip * |b| / lambda around the body Doppler (the "HERM lines" of
helicopter rotor modulation).  Jet fans modulate in the kHz, far outside any
node's Doppler window, so a jet shows nothing.

The simulator works on detection lists, not IQ, so this module synthesises the
slow-time signal of the rotating blades, takes its spectrum, and hands back the
strongest lines for the caller to emit as extra detections.

Blade model: a thin straight wire from hub to tip (Chen; Costa et al. 2025).  For
the bistatic vector b = u_tx + u_rx (unit vectors from the target toward the
transmitter and receiver, ENU) a scatterer s metres along a blade with direction e
contributes exp(j*2*pi*s*(e . b)/lambda), the same sign convention as
world._bistatic_doppler (positive Doppler for motion toward TX/RX).  Integrating
uniformly over s in [0, L] gives exp(j*pi*L*c/lambda) * sinc(L*c/lambda) with
c = e . b.

Hub offsets from the aircraft centre are ignored: a constant offset d adds the
constant phase 2*pi*(d . b)/lambda to the whole rotor, which moves no line.
"""

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

C_M_S = 299_792_458.0

# A Hann window's main lobe spans +/-2 bins, so anything closer than that to the
# body peak is the body's own skirt rather than a blade line.
_HANN_HALF_MAINLOBE_BINS = 2
_BODY_PEAK_SEARCH_BINS = 2
DETECTION_FLOOR_DB = 6.0


@dataclass(frozen=True)
class Rotor:
    """One rotating blade set, in the aircraft body frame."""

    n_blades: int
    radius_m: float
    rev_per_s: float
    axis: str  # "up" | "forward" | "right" (body frame)
    # Summed blade return relative to the body return.  A GUESS: it depends on
    # blade chord, material and aspect, none of which we model.  Placeholder
    # until calibrated against a real node recording.
    amplitude: float
    offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)  # documented but unused, see module docstring
    spin: int = 1


# Common-variant values, to verify.  Amplitudes are an interim calibration, not
# RCS ratios: each is set so the rotor's strongest line sits about 15-17 dB below
# the body line at 500 MHz (tail rotor ~21, drone props ~27).  A long blade
# spreads its energy over dozens of lines, so a physically "small" amplitude left
# every line 40-50 dB down, undetectable, whereas Berndt et al. (IET RSN 2024)
# detect propeller modulation at 50-100% of the fuselage's detection range.
# Replace these with values fitted to a real node recording of a helicopter or
# turboprop.
AIRFRAMES: dict[str, tuple[Rotor, ...]] = {
    # Fan modulation is in the kHz, outside any node's Doppler window.
    "jet": (),
    # Dash 8-400-like: two six-blade props.
    "turboprop": (
        Rotor(6, 2.05, 17.0, "forward", 4.96),
        Rotor(6, 2.05, 17.0, "forward", 4.96),
    ),
    # C172-like: two-blade fixed-pitch prop.
    "piston": (Rotor(2, 0.95, 45.0, "forward", 1.92),),
    # R44-like: main rotor plus tail rotor.
    "helicopter": (
        Rotor(2, 5.0, 6.8, "up", 12.89),
        Rotor(2, 0.8, 35.0, "right", 0.86),
    ),
    # Quadcopter: counter-rotating pairs.
    "drone": (
        Rotor(2, 0.12, 90.0, "up", 0.37, spin=1),
        Rotor(2, 0.12, 90.0, "up", 0.37, spin=-1),
        Rotor(2, 0.12, 90.0, "up", 0.37, spin=1),
        Rotor(2, 0.12, 90.0, "up", 0.37, spin=-1),
    ),
}


def _rotors(airframe: "str | Sequence[Rotor]") -> tuple[Rotor, ...]:
    if isinstance(airframe, str):
        return AIRFRAMES[airframe]
    return tuple(airframe)


def _plane_basis(axis: str, heading_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis (p1, p2) of the plane perpendicular to a body-frame rotor axis, in ENU."""
    h = math.radians(heading_deg)
    forward = np.array([math.sin(h), math.cos(h), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    if axis == "up":
        return forward, right
    if axis == "forward":
        return right, up
    if axis == "right":
        return forward, up
    raise ValueError(f"unknown rotor axis {axis!r}")


def advance_phases(
    airframe: "str | Sequence[Rotor]",
    phases: Sequence[float],
    time_s: float,
    rpm_scale: float = 1.0,
) -> list[float]:
    """Per-rotor phase at world time `time_s`, so the comb evolves from frame to frame."""
    return [
        ph + rotor.spin * 2.0 * math.pi * rotor.rev_per_s * rpm_scale * time_s
        for rotor, ph in zip(_rotors(airframe), phases)
    ]


def slow_time(
    airframe: "str | Sequence[Rotor]",
    b_vec: Sequence[float],
    heading_deg: float,
    fc_hz: float,
    f_body_hz: float,
    cpi_s: float,
    fs_hz: float = 8192.0,
    phases: Sequence[float] | None = None,
    rpm_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseband slow-time signal of the body plus rotating blades over one CPI."""
    rotors = _rotors(airframe)
    if phases is None:
        phases = [0.0] * len(rotors)
    wavelength = C_M_S / fc_hz
    b = np.asarray(b_vec, dtype=float)
    n = max(2, int(round(cpi_s * fs_hz)))
    t = np.arange(n) / fs_hz
    blades = np.zeros(n, dtype=complex)
    for rotor, phi0 in zip(rotors, phases):
        p1, p2 = _plane_basis(rotor.axis, heading_deg)
        b1 = float(p1 @ b)
        b2 = float(p2 @ b)
        theta0 = rotor.spin * 2.0 * math.pi * rotor.rev_per_s * rpm_scale * t + phi0
        total = np.zeros(n, dtype=complex)
        for k in range(rotor.n_blades):
            theta = theta0 + 2.0 * math.pi * k / rotor.n_blades
            x = rotor.radius_m * (np.cos(theta) * b1 + np.sin(theta) * b2) / wavelength
            total += np.exp(1j * math.pi * x) * np.sinc(x)
        blades += rotor.amplitude / rotor.n_blades * total
    return t, np.exp(2j * math.pi * f_body_hz * t) * (1.0 + blades)


def line_spectrum(t: np.ndarray, s: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed spectrum, as (freqs_hz, power in dB relative to the body-line peak).

    The body line is taken as the strongest bin of the whole spectrum.  The body
    carries the largest single line at every geometry we simulate, and this keeps
    the function independent of the caller's body Doppler.
    """
    n = len(s)
    window = np.hanning(n)
    spec = np.fft.fftshift(np.fft.fft(s * window))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs_hz))
    power = np.abs(spec) ** 2
    return freqs, 10.0 * np.log10(np.maximum(power, 1e-30) / power.max())


def _spectrum_rel_body(s: np.ndarray, fs_hz: float, f_body_hz: float) -> tuple[np.ndarray, np.ndarray, int]:
    """(freqs, power, body-peak index) with power normalised to the body line."""
    n = len(s)
    spec = np.fft.fftshift(np.fft.fft(s * np.hanning(n)))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs_hz))
    power = np.abs(spec) ** 2
    centre = int(np.argmin(np.abs(freqs - f_body_hz)))
    lo = max(0, centre - _BODY_PEAK_SEARCH_BINS)
    hi = min(n, centre + _BODY_PEAK_SEARCH_BINS + 1)
    peak = lo + int(np.argmax(power[lo:hi]))
    return freqs, power / power[peak], peak


def comb_lines(
    airframe: "str | Sequence[Rotor]",
    b_vec: Sequence[float],
    heading_deg: float,
    fc_hz: float,
    f_body_hz: float,
    cpi_s: float,
    doppler_min: float,
    doppler_max: float,
    body_snr_db: float,
    min_line_snr_db: float = DETECTION_FLOOR_DB,
    max_lines: int = 12,
    phases: Sequence[float] | None = None,
    fs_hz: float = 8192.0,
    rpm_scale: float = 1.0,
) -> list[tuple[float, float]]:
    """Blade lines as (doppler_hz, snr_db), strongest first, inside the node's Doppler window."""
    rotors = _rotors(airframe)
    if not rotors:
        return []
    _, s = slow_time(airframe, b_vec, heading_deg, fc_hz, f_body_hz, cpi_s, fs_hz, phases, rpm_scale)
    freqs, power, peak = _spectrum_rel_body(s, fs_hz, f_body_hz)
    bin_hz = fs_hz / len(s)

    # The body's Hann skirt is not a blade line.  When the body is loud (high SNR)
    # its sidelobes would clear the SNR floor, so a line must also stand above the
    # pure tone's own leakage at that bin.
    _, tone = slow_time((), b_vec, heading_deg, fc_hz, f_body_hz, cpi_s, fs_hz)
    _, tone_power, _ = _spectrum_rel_body(tone, fs_hz, f_body_hz)

    exclusion_hz = (_BODY_PEAK_SEARCH_BINS + _HANN_HALF_MAINLOBE_BINS) * bin_hz
    interior = np.zeros(len(power), dtype=bool)
    interior[1:-1] = (power[1:-1] > power[:-2]) & (power[1:-1] >= power[2:])
    rel_db = 10.0 * np.log10(np.maximum(power, 1e-30))
    snr = body_snr_db + rel_db
    keep = (
        interior
        & (freqs >= doppler_min)
        & (freqs <= doppler_max)
        & (np.abs(freqs - freqs[peak]) > exclusion_hz)
        & (snr >= min_line_snr_db)
        & (power >= 4.0 * tone_power)
    )
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(-power[idx], kind="stable")][:max_lines]
    return [(float(freqs[i]), float(snr[i])) for i in idx]


def spectrogram(
    t: np.ndarray, s: np.ndarray, fs_hz: float, window_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short-time FFT (50% overlap, Hann): (frame centre times, freqs, power dB re global peak)."""
    n_win = max(2, int(round(window_s * fs_hz)))
    hop = max(1, n_win // 2)
    window = np.hanning(n_win)
    starts = range(0, len(s) - n_win + 1, hop)
    frames = np.array([np.fft.fftshift(np.fft.fft(s[i : i + n_win] * window)) for i in starts])
    freqs = np.fft.fftshift(np.fft.fftfreq(n_win, d=1.0 / fs_hz))
    times = np.array([t[i] + 0.5 * n_win / fs_hz for i in starts])
    power = np.abs(frames) ** 2
    return times, freqs, 10.0 * np.log10(np.maximum(power, 1e-30) / power.max())


def _unit_vectors(bistatic_angle_deg: float, elevation_deg: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """TX and RX directions symmetric about north, `bistatic_angle_deg` apart, below the target.

    Real nodes sit on the ground under the aircraft, so both unit vectors point downward.
    """
    half = math.radians(bistatic_angle_deg) / 2.0
    el = math.radians(elevation_deg)
    horiz = math.cos(el)
    u_tx = np.array([-math.sin(half) * horiz, math.cos(half) * horiz, -math.sin(el)])
    u_rx = np.array([math.sin(half) * horiz, math.cos(half) * horiz, -math.sin(el)])
    return u_tx, u_rx


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print the micro-Doppler line table for an airframe")
    parser.add_argument("--airframe", choices=sorted(AIRFRAMES), default="helicopter")
    parser.add_argument("--fc-hz", type=float, default=500e6)
    parser.add_argument("--cpi", type=float, default=0.5, help="Integration time in seconds")
    parser.add_argument("--bistatic-angle-deg", type=float, default=60.0)
    parser.add_argument(
        "--doppler-window", type=float, default=1000.0, help="Half-width of the node Doppler window, Hz"
    )
    parser.add_argument("--heading-deg", type=float, default=90.0)
    parser.add_argument("--body-snr-db", type=float, default=20.0)
    parser.add_argument("--f-body-hz", type=float, default=0.0)
    parser.add_argument("--max-lines", type=int, default=12)
    parser.add_argument(
        "--min-line-snr-db",
        type=float,
        default=-200.0,
        help="Detection floor for a line; the default lists the strongest lines whatever their SNR",
    )
    args = parser.parse_args(argv)

    u_tx, u_rx = _unit_vectors(args.bistatic_angle_deg)
    b = u_tx + u_rx
    b_norm = float(np.linalg.norm(b))
    wavelength = C_M_S / args.fc_hz
    lines = comb_lines(
        args.airframe,
        b,
        args.heading_deg,
        args.fc_hz,
        args.f_body_hz,
        args.cpi,
        args.f_body_hz - args.doppler_window,
        args.f_body_hz + args.doppler_window,
        args.body_snr_db,
        min_line_snr_db=args.min_line_snr_db,
        max_lines=args.max_lines,
    )
    print(
        f"airframe={args.airframe} fc={args.fc_hz / 1e6:.0f} MHz lambda={wavelength:.3f} m cpi={args.cpi} s "
        f"|b|={b_norm:.2f} bin={1.0 / args.cpi:.2f} Hz window=+/-{args.doppler_window:.0f} Hz"
    )
    if not lines:
        print("no lines (nothing rotating inside the window)")
    else:
        print(f"{'offset_hz':>10} {'rel_db':>8} {'snr_db':>8}")
        for f, snr in lines:
            print(f"{f - args.f_body_hz:>10.1f} {snr - args.body_snr_db:>8.1f} {snr:>8.1f}")
        detectable = sum(1 for _, snr in lines if snr >= DETECTION_FLOOR_DB)
        print(
            f"{detectable} of {len(lines)} clear the {DETECTION_FLOOR_DB:.0f} dB detection floor at body SNR {args.body_snr_db:.0f} dB"
        )
    for i, rotor in enumerate(AIRFRAMES[args.airframe]):
        v_tip = 2.0 * math.pi * rotor.rev_per_s * rotor.radius_m
        print(
            f"rotor {i}: {rotor.n_blades} blades, {rotor.rev_per_s:g} rev/s -> blade-passing "
            f"{rotor.n_blades * rotor.rev_per_s:.1f} Hz, v_tip={v_tip:.0f} m/s, "
            f"max spread +/-{v_tip * b_norm / wavelength:.0f} Hz"
        )


if __name__ == "__main__":
    main()
