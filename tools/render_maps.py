"""Labelled delay-Doppler maps from a recorded run, for ML training.

The simulator emits detection lists, not maps.  This draws each node frame
(one per node per second) onto blah2's own grid: 301 Doppler bins (±300 Hz
at 2 Hz, the 1/CPI resolution) by 411 delay bins (-10..400 samples at
2 MHz, 0.5 µs each), from blah2-arm/config/config.yml and
src/process/ambiguity/Ambiguity.cpp.

Each detection is a Gaussian peak at its TRUE delay/Doppler (clutter at its
measured one), its height the detection's SNR.  The delay width follows the
illuminator: FM's ~150 kHz bandwidth smears a peak over ~13 delay bins, a TV
channel over about one.

What is stored, per frame, in ``maps/maps_NNN.npz`` (500 frames a file):

``signal``    float16 [301, 411]: target power over the noise floor, linear,
              noise NOT included.  Noise is random and does not compress;
              storing it made each file ~110 MB.  Add it at load time with
              ``with_noise`` (fresh noise per epoch is also better training).
``instance``  uint8 [301, 411]: 0 background, k = the frame's k-th target
              (1-based) over its half-power region.  The labelling target.

and in ``maps/index.json``, per frame: time, node, tower, and the target
list (object id or clutter, true and measured delay/Doppler, bins, SNR).

Only targets the sim reported are drawn: an aircraft its SNR model dropped
leaves no energy, where a real map would hold its sub-threshold peak.

Usage::

    python tools/render_maps.py runs/ubc-1 [--every 1]

(``tools/analyze_run.py`` runs this too.)  Load a frame::

    import json, numpy as np
    from render_maps import with_noise
    z = np.load("runs/ubc-1/maps/maps_000.npz")
    map_db = with_noise(z["signal"][0])          # what a radar would show, dB
    labels = z["instance"][0]
"""

import argparse
import json
import os

import numpy as np

# blah2 defaults: fs 2 MHz, delay -10..400 samples, Doppler ±300 Hz, CPI 0.5 s.
FS = 2_000_000
DELAY_MIN, DELAY_MAX = -10, 400
DOPPLER_MIN, DOPPLER_MAX, DOPPLER_RES = -300.0, 300.0, 2.0
SHAPE = (int((DOPPLER_MAX - DOPPLER_MIN) / DOPPLER_RES) + 1, DELAY_MAX - DELAY_MIN + 1)

_FM_BANDWIDTH_HZ = 150e3
_TV_BANDWIDTH_HZ = 6e6
_FM_BAND_MAX_HZ = 200e6
_SHARD = 500


def with_noise(signal: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """A stored ``signal`` map as a radar would show it: dB over the noise
    floor, with exponentially distributed noise power (|ambiguity|²)."""
    rng = rng or np.random.default_rng()
    return 10 * np.log10(signal.astype(np.float32) + rng.exponential(1.0, size=signal.shape))


def delay_bin(delay_us: float) -> float:
    return delay_us * FS / 1e6 - DELAY_MIN


def doppler_bin(doppler_hz: float) -> float:
    return (doppler_hz - DOPPLER_MIN) / DOPPLER_RES


def render_frame(frame: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    fc = frame.get("fc_hz") or 0.0
    bandwidth = _FM_BANDWIDTH_HZ if fc and fc < _FM_BAND_MAX_HZ else _TV_BANDWIDTH_HZ
    delay_fwhm = max(1.0, FS / bandwidth)
    sig_d, sig_f = delay_fwhm / 2.3548, 1.0 / 2.3548
    rows = np.arange(SHAPE[0])[:, None]
    cols = np.arange(SHAPE[1])[None, :]

    signal = np.zeros(SHAPE, dtype=np.float32)
    instance = np.zeros(SHAPE, dtype=np.uint8)
    targets = []
    for k, d in enumerate(frame["detections"][:255], start=1):
        clutter = d["is_clutter"]
        delay = d["delay_us"] if clutter else d["delay_true"]
        doppler = d["doppler_hz"] if clutter else d["doppler_true"]
        db, fb = delay_bin(delay), doppler_bin(doppler)
        in_grid = bool(0 <= db < SHAPE[1] and 0 <= fb < SHAPE[0])
        targets.append(
            {
                "k": k,
                "label": "clutter" if clutter else d["object_id"],
                "delay_true_us": d["delay_true"],
                "doppler_true_hz": d["doppler_true"],
                "delay_meas_us": d["delay_us"],
                "doppler_meas_hz": d["doppler_hz"],
                "snr_db": d["snr_db"],
                "delay_bin": round(db, 2),
                "doppler_bin": round(fb, 2),
                "in_grid": in_grid,
            }
        )
        if not in_grid:
            continue
        shape = np.exp(-0.5 * (((cols - db) / sig_d) ** 2 + ((rows - fb) / sig_f) ** 2))
        peak = 10 ** (d["snr_db"] / 10.0) * shape
        # Half of the sampled maximum: a one-bin peak between pixels still
        # gets at least one labelled pixel.
        mine = (shape >= 0.5 * shape.max()) & (peak > signal)
        instance[mine] = k
        signal += peak
    return signal.astype(np.float16), instance, targets


def write_maps(run_dir: str, every: int = 1) -> tuple[int, str]:
    out_dir = os.path.join(run_dir, "maps")
    os.makedirs(out_dir, exist_ok=True)
    index = {
        "grid": {
            "shape": list(SHAPE),
            "axes": ["doppler", "delay"],
            "delay_us": [DELAY_MIN / FS * 1e6, DELAY_MAX / FS * 1e6],
            "doppler_hz": [DOPPLER_MIN, DOPPLER_MAX],
            "signal": "linear power over the noise floor; add noise with with_noise()",
        },
        "frames": [],
    }
    signals, instances = [], []
    shard = 0

    def flush():
        nonlocal shard
        if signals:
            name = f"maps_{shard:03d}.npz"
            np.savez_compressed(os.path.join(out_dir, name), signal=np.stack(signals), instance=np.stack(instances))
            signals.clear()
            instances.clear()
            shard += 1

    with open(os.path.join(run_dir, "frames.ndjson"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % every or not line.strip():
                continue
            frame = json.loads(line)
            if not frame["detections"]:
                continue
            signal, instance, targets = render_frame(frame)
            n = len(index["frames"])
            index["frames"].append(
                {
                    "file": f"maps_{n // _SHARD:03d}.npz",
                    "row": n % _SHARD,
                    "t_ms": frame["t_ms"],
                    "node_id": frame["node_id"],
                    "tower": frame.get("tx_callsign"),
                    "targets": targets,
                }
            )
            signals.append(signal)
            instances.append(instance)
            if len(signals) == _SHARD:
                flush()
    flush()
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)
    return len(index["frames"]), out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a recorded run as labelled delay-Doppler maps")
    parser.add_argument("run_dir")
    parser.add_argument("--every", type=int, default=1, help="Keep every Nth frame")
    args = parser.parse_args()
    n, out_dir = write_maps(args.run_dir, args.every)
    print(f"wrote {n} maps to {out_dir}")


if __name__ == "__main__":
    main()
