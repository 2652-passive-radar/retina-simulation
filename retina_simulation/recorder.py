"""Run recorder: writes what the fleet sent, and the truth behind it, to disk.

Enabled with ``orchestrator --record DIR``.  Three files, all append-only so a
run killed mid-way still leaves everything up to that point readable:

``nodes.json``
    The node configs the run used (RX, TX, frequency, beam), written once.
``frames.ndjson``
    One line per node frame the world produced, sent or not.  Each detection
    carries its measured delay (µs), Doppler (Hz) and SNR beside its truth
    label: the object that made the echo, or ``is_clutter``, plus the
    noise-free delay/Doppler and the residual (measured minus true).
``truth.ndjson``
    One line per world tick: every aircraft's absolute state, including the
    dark ones the ADS-B push leaves out.

Nothing here touches the wire.  The labels are produced alongside the frame
by ``SimulationWorld.generate_detections_for_node(labels=...)`` and only ever
reach this file.
"""

import json
import os


def _detections(frame: dict, labels: list[dict]) -> list[dict]:
    adsb = frame.get("adsb") or [None] * len(frame["delay"])
    out = []
    for delay, doppler, snr, label, adsb_row in zip(frame["delay"], frame["doppler"], frame["snr"], labels, adsb):
        det = {"delay_us": delay, "doppler_hz": doppler, "snr_db": snr, **label}
        if label["delay_true"] is not None:
            det["delay_resid_us"] = round(delay - label["delay_true"], 4)
            det["doppler_resid_hz"] = round(doppler - label["doppler_true"], 4)
        else:
            det["delay_resid_us"] = None
            det["doppler_resid_hz"] = None
        det["adsb"] = adsb_row
        out.append(det)
    return out


def aircraft_state(ac) -> dict:
    """Absolute state of one simulated aircraft, in SI-ish units."""
    return {
        "id": ac.object_id,
        "adsb_hex": ac.adsb_hex,
        "adsb_callsign": ac.adsb_callsign,
        "lat": round(ac.lat, 6),
        "lon": round(ac.lon, 6),
        "alt_m": round(ac.alt_km * 1000.0, 1),
        "vel_east_ms": round(ac.vel_east * 1000.0, 2),
        "vel_north_ms": round(ac.vel_north * 1000.0, 2),
        "vel_up_ms": round(ac.vel_up * 1000.0, 2),
        "heading_deg": round(ac.heading_deg, 1),
        "speed_ms": round(ac.speed_km_s * 1000.0, 1),
        "object_type": ac.object_type,
        "has_adsb": ac.has_adsb,
        "adsb_silent": ac.adsb_silent,
        "is_anomalous": ac.is_anomalous,
        "anomaly_event": ac.anomaly_event,
        "source": ac.source,
    }


class RunRecorder:
    def __init__(self, out_dir: str, node_configs: list[dict]):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self._nodes = {n["node_id"]: n for n in node_configs}
        with open(os.path.join(out_dir, "nodes.json"), "w", encoding="utf-8") as f:
            json.dump(node_configs, f, indent=2)
        # Held open for the whole run and closed by close(); line-buffered so
        # a killed run still leaves every complete line on disk.
        self._frames = open(os.path.join(out_dir, "frames.ndjson"), "a", buffering=1, encoding="utf-8")  # noqa: SIM115
        self._truth = open(os.path.join(out_dir, "truth.ndjson"), "a", buffering=1, encoding="utf-8")  # noqa: SIM115
        self.frames_written = 0

    def frame(self, node_id: str, frame: dict, labels: list[dict], sent: bool) -> None:
        node = self._nodes.get(node_id, {})
        rec = {
            "t_ms": frame["timestamp"],
            "node_id": node_id,
            "tx_callsign": node.get("tx_callsign"),
            "fc_hz": node.get("fc_hz"),
            "sent": sent,
            "detections": _detections(frame, labels),
        }
        self._frames.write(json.dumps(rec) + "\n")
        self.frames_written += 1

    def truth(self, timestamp_ms: int, aircraft) -> None:
        rec = {"t_ms": timestamp_ms, "aircraft": [aircraft_state(ac) for ac in aircraft]}
        self._truth.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        self._frames.close()
        self._truth.close()
