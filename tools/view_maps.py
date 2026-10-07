"""Save a few rendered delay-Doppler maps as PNG images to look at.

Each PNG is one node's map at one moment: on the left the map as a radar
would show it (noise added, brighter = stronger, Doppler +300 Hz at the top,
delay -5 µs at the left edge to 200 µs at the right), on the right its labels
(one colour per aircraft, grey for clutter).  The console prints which colour
is which aircraft.

Usage::

    python tools/view_maps.py runs/ubc-1                 # first 10 maps with aircraft in them
    python tools/view_maps.py runs/ubc-1 --count 30 --node synth-ubc-cisc

writes ``runs/ubc-1/maps/png/<time>_<node>.png``.
"""

import argparse
import json
import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_maps import with_noise  # noqa: E402

_COLOURS = [
    ("blue", (42, 120, 214)),
    ("orange", (235, 104, 52)),
    ("green", (27, 175, 122)),
    ("yellow", (237, 161, 0)),
    ("pink", (232, 123, 164)),
    ("violet", (74, 58, 167)),
]
_CLUTTER = (120, 120, 120)
_LOW, _HIGH = np.array([10, 18, 32.0]), np.array([225, 238, 255.0])


def _png(rgb: np.ndarray, path: str) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


def picture(
    signal: np.ndarray, instance: np.ndarray, targets: list[dict], colours: dict, rng
) -> tuple[np.ndarray, dict]:
    map_db = with_noise(signal, rng)
    t = np.clip((map_db + 5) / 25, 0, 1)[..., None]
    left = (_LOW + t * (_HIGH - _LOW)).astype(np.uint8)
    right = np.full(instance.shape + (3,), 24, np.uint8)
    key = {}
    for tg in targets:
        if tg["label"] == "clutter":
            colour = _CLUTTER
        else:
            name, colour = colours[tg["label"]]
            key[tg["label"]] = name
        right[instance == tg["k"]] = colour
    gap = np.full((instance.shape[0], 6, 3), 255, np.uint8)
    img = np.concatenate([left, gap, right], axis=1)[::-1]  # Doppler +300 Hz at the top
    return np.repeat(np.repeat(img, 2, axis=0), 2, axis=1), key


def main() -> None:
    parser = argparse.ArgumentParser(description="Save delay-Doppler maps as PNG images")
    parser.add_argument("run_dir")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--node", default=None, help="Only this node's maps")
    parser.add_argument("--start", type=int, default=0, help="Skip this many matching maps first")
    args = parser.parse_args()

    maps_dir = os.path.join(args.run_dir, "maps")
    with open(os.path.join(maps_dir, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    picks = [
        fr
        for fr in index["frames"]
        if any(t["label"] != "clutter" and t["in_grid"] for t in fr["targets"])
        and (args.node is None or fr["node_id"] == args.node)
    ][args.start : args.start + args.count]
    if not picks:
        print("no maps match")
        return

    out_dir = os.path.join(maps_dir, "png")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    # One colour per aircraft for the whole run, so images compare.
    ids = sorted({t["label"] for fr in index["frames"] for t in fr["targets"] if t["label"] != "clutter"})
    colours = {aid: _COLOURS[i % len(_COLOURS)] for i, aid in enumerate(ids)}
    loaded = {}
    for fr in picks:
        if fr["file"] not in loaded:
            with np.load(os.path.join(maps_dir, fr["file"])) as z:
                loaded[fr["file"]] = (z["signal"], z["instance"])
        signal, instance = loaded[fr["file"]]
        img, key = picture(signal[fr["row"]], instance[fr["row"]], fr["targets"], colours, rng)
        fname = f"{fr['t_ms']}_{fr['node_id']}.png"
        _png(img, os.path.join(out_dir, fname))
        legend = ", ".join(f"{colour} = {aid}" for aid, colour in key.items())
        print(f"{fname}  ({fr['tower']})  {legend}")
    print(f"wrote {len(picks)} images to {out_dir}")


if __name__ == "__main__":
    main()
