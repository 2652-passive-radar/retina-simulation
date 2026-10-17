"""Look inside the server's worst solves of a recorded --system run.

For every published solve whose horizontal position or velocity error is
above a threshold, prints what the server recorded about it (nodes, residuals,
altitude, where position and velocity came from, how far apart in time its
measurements were, whether a node was adopted, its own contamination verdict).
With --frames it also prints what each node actually detected over the solve's
time span: the aircraft's own detection, if the node made one, and the clutter
in the same frame.

Usage::

    python tools/inspect_solves.py runs/baseline-static/<run>/seed-303/system
    python tools/inspect_solves.py <system dir> --pos 300 --vel 30 --frames
"""

import argparse
import csv
import json
import os

_FIELDS = [
    "n_nodes", "contributing_node_ids", "rms_delay", "rms_doppler", "alt_m", "guess_alt_km", "alt_source",
    "pos_source", "fit_vs_solve_km", "vel_source", "vel_untrusted", "vz_saturated", "solver_vel_east",
    "solver_vel_north", "epoch_aligned", "epoch_skew_s", "displacement_km", "kf_action", "contaminated",
    "foreign_node_ids", "n_nodes_pre_adopt", "adopt_meta", "gt_error_km",
]  # fmt: skip


def _read_ndjson(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _show_frame(nid: str, fr: dict, t_ms: int) -> None:
    ac = [x for x in fr["detections"] if not x["is_clutter"]]
    clutter = [x for x in fr["detections"] if x["is_clutter"]]
    dt = (fr["t_ms"] - t_ms) / 1000
    if ac:
        a = ac[0]
        print(
            f"   {nid} (frame {dt:+.1f} s): aircraft at delay {a['delay_us']} us, Doppler {a['doppler_hz']} Hz "
            f"(true {a['delay_true']} / {a['doppler_true']})"
        )
    else:
        print(f"   {nid} (frame {dt:+.1f} s): aircraft NOT detected")
    for c in clutter:
        print(f"      clutter: delay {c['delay_us']} us, Doppler {c['doppler_hz']} Hz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the server's worst solves of a --system recording")
    parser.add_argument("system_dir", help="a seed's system/ folder")
    parser.add_argument("--pos", type=float, default=300.0, help="position error threshold, m")
    parser.add_argument("--vel", type=float, default=30.0, help="velocity error threshold, m/s")
    parser.add_argument("--frames", action="store_true", help="also print every node's detections over each solve")
    args = parser.parse_args()
    d = args.system_dir

    truth = _read_ndjson(os.path.join(d, "truth.ndjson"))
    t0 = truth[0]["t_ms"] - 1000
    frames = _read_ndjson(os.path.join(d, "frames.ndjson"))
    records = {
        (r.get("solve_key"), r.get("measurement_ts_ms")): r for r in _read_ndjson(os.path.join(d, "solves.ndjson"))
    }
    with open(os.path.join(d, "analysis", "solves.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    bad = [
        r
        for r in rows
        if (r["err_horiz_m"] and float(r["err_horiz_m"]) > args.pos)
        or (r.get("verr_horiz_ms") and float(r["verr_horiz_ms"]) > args.vel)
    ]
    print(f"{len(bad)} of {len(rows)} published solves above {args.pos:g} m or {args.vel:g} m/s\n")
    for r in bad:
        t_ms = int(float(r["t_ms"]))
        rec = records.get((r["solve_key"], t_ms), {})
        print(f"== t = {(t_ms - t0) / 1000:.1f} s  key {r['solve_key']}")
        print(f"   error: position {r['err_horiz_m']} m, velocity {r.get('verr_horiz_ms')} m/s")
        print(f"   true alt {r['true_alt_m']} m, solved alt {r['est_alt_m']} m")
        for k in _FIELDS:
            if rec.get(k) not in (None, [], {}):
                print(f"   {k}: {rec[k]}")
        if args.frames:
            nodes = rec.get("contributing_node_ids") or sorted({fr["node_id"] for fr in frames})
            skew_ms = int((rec.get("epoch_skew_s") or 0) * 1000)
            for nid in nodes:
                span = [fr for fr in frames if fr["node_id"] == nid and t_ms - skew_ms - 1500 <= fr["t_ms"] <= t_ms]
                for fr in sorted(span, key=lambda fr: fr["t_ms"]):
                    _show_frame(nid, fr, t_ms)
        print()


if __name__ == "__main__":
    main()
