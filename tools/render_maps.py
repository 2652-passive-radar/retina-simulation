"""Render recorded frames as labelled delay-Doppler maps for ML training.

The simulator emits detection lists, not maps.  This draws each frame onto
blah2's own grid so a model trained here sees the shape it will see on a node:

* delay bins ``delayMin..delayMax`` samples at ``fs`` (default -10..400 at
  2 MHz, so 411 bins of 0.5 µs), Doppler ``dopplerMin..dopplerMax`` at
  ``1/cpi`` (default ±300 Hz at 2 Hz, so 301 bins).  These defaults are
  blah2-arm/config/config.yml, and the bin arithmetic follows
  src/process/ambiguity/Ambiguity.cpp.
* values are dB over the noise floor, from exponentially distributed noise
  power, which is what a CFAR on |ambiguity|² sees.

Each target is drawn at its TRUE delay/Doppler (where the energy is; the
sim's measured value is that plus estimation noise) as a Gaussian peak whose
height is the detection's SNR.  The delay width follows the illuminator's
bandwidth: FM's ~150 kHz smears a peak over ~13 bins at 2 MHz, while a DTV
channel wider than fs resolves to about one.  Clutter detections are drawn
at their own position.

Labels per frame:

``instance``  uint16 map: 0 background, k = 1-based index into the frame's
              target list for the pixel's strongest target
``overlap``   bool map: pixels inside the half-power region of 2+ targets
``targets``   the frame's targets (object id, true and measured delay and
              Doppler, bin coordinates, SNR, widths, clutter flag), from
              which any per-target mask can be regenerated exactly
``n_overlapping_pairs``  pairs of targets whose half-power regions touch

Limitation: only targets the sim actually reported are drawn.  An aircraft
in the beam that the sim's SNR model missed contributes nothing, where a
real map would still hold its sub-threshold energy.

Usage::

    python tools/render_maps.py runs/ubc-1 [--every 1] [--shard 500]

writes ``runs/ubc-1/maps/maps_000.npz ...`` and ``maps/index.json``.
"""

import argparse
import json
import os

import numpy as np

_FM_BANDWIDTH_HZ = 150e3
_DTV_BANDWIDTH_HZ = 6e6
_FM_BAND_MAX_HZ = 200e6


class Grid:
    def __init__(self, fs, delay_min, delay_max, doppler_min, doppler_max, cpi):
        self.fs = fs
        self.delay_min = delay_min
        self.delays_us = np.arange(delay_min, delay_max + 1) / fs * 1e6
        res = 1.0 / cpi
        mid = (doppler_min + doppler_max) / 2.0
        n_half = int(np.floor((doppler_max - mid) / res + 1e-9))
        self.dopplers_hz = mid + res * np.arange(-n_half, n_half + 1)
        self.doppler_res = res
        self.shape = (len(self.dopplers_hz), len(self.delays_us))

    def delay_bin(self, delay_us: float) -> float:
        return delay_us * self.fs / 1e6 - self.delay_min

    def doppler_bin(self, doppler_hz: float) -> float:
        return (doppler_hz - self.dopplers_hz[0]) / self.doppler_res


def _delay_fwhm_bins(fc_hz: float, fs: float) -> float:
    bandwidth = _FM_BANDWIDTH_HZ if fc_hz and fc_hz < _FM_BAND_MAX_HZ else _DTV_BANDWIDTH_HZ
    return max(1.0, fs / bandwidth)


def render_frame(frame: dict, grid: Grid, rng: np.random.Generator, doppler_fwhm_bins: float = 1.0):
    fc = frame.get("fc_hz") or 0.0
    delay_fwhm = _delay_fwhm_bins(fc, grid.fs)
    sig_d = delay_fwhm / 2.3548
    sig_f = doppler_fwhm_bins / 2.3548
    rows = np.arange(grid.shape[0])[:, None]
    cols = np.arange(grid.shape[1])[None, :]

    power = rng.exponential(1.0, size=grid.shape)
    best = np.zeros(grid.shape)
    instance = np.zeros(grid.shape, dtype=np.uint16)
    half_count = np.zeros(grid.shape, dtype=np.uint8)
    targets = []
    for k, d in enumerate(frame["detections"], start=1):
        if d["is_clutter"]:
            delay_us, doppler_hz = d["delay_us"], d["doppler_hz"]
        else:
            delay_us, doppler_hz = d["delay_true"], d["doppler_true"]
        db, fb = grid.delay_bin(delay_us), grid.doppler_bin(doppler_hz)
        in_grid = bool(0 <= db < grid.shape[1] and 0 <= fb < grid.shape[0])
        targets.append(
            {
                "index": k,
                "object_id": d["object_id"],
                "adsb_hex": d["adsb_hex"],
                "is_clutter": d["is_clutter"],
                "snr_db": d["snr_db"],
                "delay_true_us": d["delay_true"],
                "doppler_true_hz": d["doppler_true"],
                "delay_meas_us": d["delay_us"],
                "doppler_meas_hz": d["doppler_hz"],
                "delay_bin": round(db, 3),
                "doppler_bin": round(fb, 3),
                "delay_fwhm_bins": round(delay_fwhm, 3),
                "doppler_fwhm_bins": doppler_fwhm_bins,
                "in_grid": in_grid,
            }
        )
        if not in_grid:
            continue
        shape = np.exp(-0.5 * (((cols - db) / sig_d) ** 2 + ((rows - fb) / sig_f) ** 2))
        peak = 10 ** (d["snr_db"] / 10.0)
        power += peak * shape
        # Half of the sampled maximum, not of 1.0: a one-bin-wide peak that
        # falls between pixels would otherwise leave no pixel in its mask.
        half = shape >= 0.5 * shape.max()
        stronger = peak * shape > best
        best = np.where(stronger, peak * shape, best)
        instance = np.where(stronger & half, k, instance)
        half_count += half.astype(np.uint8)

    overlap = half_count >= 2
    n_pairs = 0
    for i, a in enumerate(targets):
        for b in targets[i + 1 :]:
            if not (a["in_grid"] and b["in_grid"]):
                continue
            if (
                abs(a["delay_bin"] - b["delay_bin"]) <= (a["delay_fwhm_bins"] + b["delay_fwhm_bins"]) / 2
                and abs(a["doppler_bin"] - b["doppler_bin"]) <= doppler_fwhm_bins
            ):
                n_pairs += 1
    map_db = (10 * np.log10(power)).astype(np.float16)
    return map_db, instance, overlap, targets, n_pairs


def _iter_frames(path: str, every: int):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip() and i % every == 0:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render recorded frames as labelled delay-Doppler maps")
    parser.add_argument("run_dir")
    parser.add_argument("--every", type=int, default=1, help="Keep every Nth frame")
    parser.add_argument("--shard", type=int, default=500, help="Frames per .npz file")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many (0 = all)")
    parser.add_argument("--include-empty", action="store_true", help="Also render frames with no detections")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fs", type=float, default=2_000_000)
    parser.add_argument("--delay-min", type=int, default=-10)
    parser.add_argument("--delay-max", type=int, default=400)
    parser.add_argument("--doppler-min", type=float, default=-300)
    parser.add_argument("--doppler-max", type=float, default=300)
    parser.add_argument("--cpi", type=float, default=0.5)
    args = parser.parse_args()

    grid = Grid(args.fs, args.delay_min, args.delay_max, args.doppler_min, args.doppler_max, args.cpi)
    rng = np.random.default_rng(args.seed)
    out_dir = os.path.join(args.run_dir, "maps")
    os.makedirs(out_dir, exist_ok=True)

    index = {
        "grid": {
            "shape": list(grid.shape),
            "axes": ["doppler", "delay"],
            "delays_us": [float(grid.delays_us[0]), float(grid.delays_us[-1])],
            "delay_bin_us": 1e6 / args.fs,
            "dopplers_hz": [float(grid.dopplers_hz[0]), float(grid.dopplers_hz[-1])],
            "doppler_bin_hz": grid.doppler_res,
            "values": "dB over noise floor",
        },
        "shards": [],
        "frames": [],
    }
    buf = {"map": [], "instance": [], "overlap": []}
    shard_no = 0

    def flush():
        nonlocal shard_no
        if not buf["map"]:
            return
        name = f"maps_{shard_no:03d}.npz"
        np.savez_compressed(
            os.path.join(out_dir, name),
            map_db=np.stack(buf["map"]),
            instance=np.stack(buf["instance"]),
            overlap=np.stack(buf["overlap"]),
        )
        index["shards"].append({"file": name, "n": len(buf["map"])})
        for k in buf:
            buf[k].clear()
        shard_no += 1

    n = 0
    for frame in _iter_frames(os.path.join(args.run_dir, "frames.ndjson"), args.every):
        if not frame["detections"] and not args.include_empty:
            continue
        map_db, instance, overlap, targets, n_pairs = render_frame(frame, grid, rng)
        buf["map"].append(map_db)
        buf["instance"].append(instance)
        buf["overlap"].append(overlap)
        index["frames"].append(
            {
                "shard": shard_no,
                "row": len(buf["map"]) - 1,
                "t_ms": frame["t_ms"],
                "node_id": frame["node_id"],
                "tx_callsign": frame.get("tx_callsign"),
                "fc_hz": frame.get("fc_hz"),
                "n_targets": sum(1 for t in targets if not t["is_clutter"]),
                "n_clutter": sum(1 for t in targets if t["is_clutter"]),
                "n_overlapping_pairs": n_pairs,
                "targets": targets,
            }
        )
        n += 1
        if len(buf["map"]) >= args.shard:
            flush()
        if args.max_frames and n >= args.max_frames:
            break
    flush()
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f)
    n_ov = sum(1 for fr in index["frames"] if fr["n_overlapping_pairs"])
    print(f"rendered {n} frames on a {grid.shape[0]}x{grid.shape[1]} grid into {len(index['shards'])} shard(s)")
    print(f"frames with overlapping targets: {n_ov}")
    print(f"wrote {out_dir}/")


if __name__ == "__main__":
    main()
