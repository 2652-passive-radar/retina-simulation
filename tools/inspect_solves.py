"""Look inside the server's worst solves of a recorded --system run.

For every published solve whose horizontal position or velocity error is
above a threshold, prints what the server recorded about it (nodes, residuals,
altitude, where position and velocity came from, its own contamination
verdict) beside what each contributing node actually detected at that moment:
the aircraft's own detection, if the node made one, and the clutter in the
same frame.  A solve built on a clutter detection, or on a node that missed
the aircraft that second, shows up as a node whose aircraft detection is
missing or does not match.

Usage::

    python tools/inspect_solves.py runs/baseline-static/<run>/seed-303/system
    python tools/inspect_solves.py <system dir> --pos 300 --vel 30
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the server's worst solves of a --system recording")
    parser.add_argument("system_dir", help="a seed's system/ folder")
    parser.add_argument("--pos", type=float, default=300.0, help="position error threshold, m")
    parser.add_argument("--vel", type=float, default=30.0, help="velocity error threshold, m/s")
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
        nodes = rec.get("contributing_node_ids") or []
        for nid in nodes:
            near = [fr for fr in frames if fr["node_id"] == nid and abs(fr["t_ms"] - t_ms) <= 1500]
            fr = min(near, key=lambda fr: abs(fr["t_ms"] - t_ms)) if near else None
            if fr is None:
                print(f"   {nid}: no frame within 1.5 s")
                continue
            ac = [x for x in fr["detections"] if not x["is_clutter"]]
            clutter = [x for x in fr["detections"] if x["is_clutter"]]
            dt = (fr["t_ms"] - t_ms) / 1000
            if ac:
                a = ac[0]
                print(
                    f"   {nid} (frame {dt:+.1f} s): aircraft detected at delay {a['delay_us']} us, "
                    f"Doppler {a['doppler_hz']} Hz (true {a['delay_true']} / {a['doppler_true']})"
                )
            else:
                print(f"   {nid} (frame {dt:+.1f} s): aircraft NOT detected this second")
            for c in clutter:
                print(f"      clutter: delay {c['delay_us']} us, Doppler {c['doppler_hz']} Hz")
        print()


if __name__ == "__main__":
    main()
