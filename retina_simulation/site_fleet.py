"""Build a fleet config from hand-placed receivers and named real towers.

The generator places receivers itself and draws illuminators from built-in US
tower tables.  A site fleet instead takes both from you: every receiver is a
position you choose, and its illuminator is named by callsign out of a
towers.retina.fm search result saved as JSON.  The output is the
``fleet_config.json`` shape ``orchestrator --config`` already reads, so the
rest of the simulation runs unchanged.

Site file::

    {
      "name": "ubc",
      "metro": "yvr",
      "defaults": {"rx_alt_m": 90, "beam_width_deg": 60, "max_range_km": 60},
      "cells": [{"ring_id": "synth-ubc", "core_lat": 49.1947, "core_lon": -123.1839,
                 "radius_km": 60, "arrival_bearings_deg": [263, 83]}],
      "nodes": [
        {"node_id": "synth-ubc-1", "rx_lat": 49.266028, "rx_lon": -123.251806,
         "tower": "CBUT-DT", "beam_azimuth_deg": 150}
      ]
    }

``tower`` matches a tower's ``callsign`` or any of its ``shared_callsigns``.
Omit ``beam_azimuth_deg`` to aim broadside to the RX→TX baseline.  A cell's
``ring_id`` must prefix the node ids it serves: the orchestrator drops cells
no surviving node id starts with.

Usage::

    python -m retina_simulation.site_fleet sites/ubc/site.json sites/ubc/towers.json -o sites/ubc/fleet.json
"""

import argparse
import json
import sys

_M_TO_FT = 3.28084
# The backend recognises a simulated node by this id prefix (is_synthetic_node
# in retina-server); without it a node is treated as unregistered hardware and
# dropped from the public node list and the map.
SYNTHETIC_PREFIX = "synth-"


def _tower_index(towers_doc: dict) -> dict[str, dict]:
    index = {}
    for t in towers_doc.get("towers", []):
        for name in [t.get("callsign"), *(t.get("shared_callsigns") or [])]:
            if name:
                index.setdefault(name.strip().upper(), t)
    return index


def _tower_alt_m(tower: dict) -> float:
    if tower.get("altitude_m") is not None:
        return float(tower["altitude_m"])
    return float(tower.get("elevation_m") or 0.0) + float(tower.get("antenna_height_m") or 0.0)


def build_site_fleet(site: dict, towers_doc: dict) -> dict:
    """Return a fleet config dict for ``orchestrator --config``.

    Raises ValueError naming every node whose tower is not in the search
    result, rather than stopping at the first.
    """
    index = _tower_index(towers_doc)
    defaults = site.get("defaults", {})
    nodes = []
    missing = []
    for spec in site["nodes"]:
        merged = {**defaults, **spec}
        tower = index.get(str(merged["tower"]).strip().upper())
        if tower is None:
            missing.append(f"{merged['node_id']}: {merged['tower']}")
            continue
        node = {
            "node_id": merged["node_id"],
            "rx_lat": float(merged["rx_lat"]),
            "rx_lon": float(merged["rx_lon"]),
            "rx_alt_ft": round(float(merged.get("rx_alt_m", 0.0)) * _M_TO_FT, 1),
            "tx_lat": float(tower["latitude"]),
            "tx_lon": float(tower["longitude"]),
            "tx_alt_ft": round(_tower_alt_m(tower) * _M_TO_FT, 1),
            "fc_hz": round(float(tower["frequency_mhz"]) * 1_000_000),
            "fs_hz": float(merged.get("fs_hz", 2_000_000.0)),
            "beam_width_deg": float(merged.get("beam_width_deg", 42.0)),
            "max_range_km": float(merged.get("max_range_km", 50.0)),
            "region": merged.get("region", site.get("region", "ca")),
            "tx_callsign": tower["callsign"],
        }
        if merged.get("beam_azimuth_deg") is not None:
            node["beam_azimuth_deg"] = float(merged["beam_azimuth_deg"])
        if merged.get("max_bistatic_range_km") is not None:
            node["max_bistatic_range_km"] = float(merged["max_bistatic_range_km"])
        nodes.append(node)
    if missing:
        raise ValueError("towers not in the search result: " + ", ".join(missing))
    return {
        "fleet": {"site": site.get("name"), "metro": site.get("metro")},
        "nodes": nodes,
        "cells": site.get("cells", []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("site", help="Site file: receivers and the tower each one uses")
    parser.add_argument("towers", help="towers.retina.fm /api/towers response saved as JSON")
    parser.add_argument("-o", "--output", default="fleet_config.json")
    args = parser.parse_args(argv)

    with open(args.site, encoding="utf-8") as f:
        site = json.load(f)
    with open(args.towers, encoding="utf-8") as f:
        towers_doc = json.load(f)
    try:
        fleet = build_site_fleet(site, towers_doc)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(fleet, f, indent=2)
    unprefixed = [n["node_id"] for n in fleet["nodes"] if not n["node_id"].startswith(SYNTHETIC_PREFIX)]
    if unprefixed:
        print(
            f"warning: {', '.join(unprefixed)} do not start with {SYNTHETIC_PREFIX!r}; the server will take them "
            "for unregistered hardware and keep them off the map",
            file=sys.stderr,
        )
    for n in fleet["nodes"]:
        print(f"{n['node_id']}: {n['tx_callsign']} {n['fc_hz'] / 1e6:.1f} MHz @ ({n['tx_lat']:.4f}, {n['tx_lon']:.4f})")
    print(f"wrote {args.output} ({len(fleet['nodes'])} nodes, {len(fleet['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
