"""Record the server's per-solve history to disk while a simulation runs.

The backend keeps every solver outcome (published or gate-rejected) for about
30 minutes and serves it at ``/api/test/mlat-history``.  This polls that
window and appends each record it has not seen yet to ``solves.ndjson``, so a
run of any length is kept.  Run it beside ``orchestrator --record DIR`` with
the same DIR and ``tools/analyze_run.py`` joins the two.

A record carries the solver's state guess (lat, lon, alt_m, vel_east,
vel_north), its delay/Doppler fit residuals (rms_delay µs, rms_doppler Hz),
the node count, the outcome, and the server's own nearest-truth match
(gt_hex, gt_error_km).

The endpoint answers an administrator or the radar key.  A local dev backend
started with AUTH_ALLOW_ANONYMOUS_ADMIN=1 needs neither; otherwise pass the
key with --radar-key or RADAR_API_KEY.

Usage::

    python tools/record_solves.py --out runs/ubc-1 [--url http://localhost:8000]
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def _fetch(url: str, radar_key: str | None) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 retina-sim-recorder"}
    if radar_key:
        headers["X-API-Key"] = radar_key
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _record_id(rec: dict) -> tuple:
    return (rec.get("ts_ms"), rec.get("solve_key"), rec.get("outcome"), rec.get("solver_hex"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Append the server's solve history to solves.ndjson")
    parser.add_argument("--out", required=True, help="Run directory (same as orchestrator --record)")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--interval", type=float, default=20.0, help="Seconds between polls")
    parser.add_argument("--duration", type=float, default=0.0, help="Stop after this many seconds (0 = until Ctrl+C)")
    parser.add_argument("--radar-key", default=os.getenv("RADAR_API_KEY"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "solves.ndjson")
    seen: set[tuple] = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    seen.add(_record_id(json.loads(line)))

    endpoint = f"{args.url.rstrip('/')}/api/test/mlat-history?all=1&limit=5000"
    start = time.monotonic()
    with open(path, "a", encoding="utf-8", buffering=1) as out:
        while True:
            try:
                doc = _fetch(endpoint, args.radar_key)
                if "error" in doc:
                    print(f"server: {doc['error']}")
                new = 0
                for rec in sorted(doc.get("records", []), key=lambda r: r.get("ts_ms") or 0):
                    rid = _record_id(rec)
                    if rid in seen:
                        continue
                    seen.add(rid)
                    out.write(json.dumps(rec) + "\n")
                    new += 1
                print(f"{time.strftime('%H:%M:%S')} +{new} solves ({len(seen)} total)", flush=True)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                print(f"{time.strftime('%H:%M:%S')} poll failed: {exc}", flush=True)
            if args.duration and time.monotonic() - start >= args.duration:
                break
            try:
                time.sleep(args.interval)
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
