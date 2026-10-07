"""Site fleets: hand-placed receivers, towers named by callsign."""

import json
from pathlib import Path

import pytest

from retina_simulation.site_fleet import SYNTHETIC_PREFIX, build_site_fleet

_TOWERS = {
    "towers": [
        {
            "callsign": "CHAN-DT",
            "shared_callsigns": ["CHAN-DT-AX1"],
            "frequency_mhz": 521.0,
            "latitude": 49.345,
            "longitude": -122.972778,
            "antenna_height_m": 33.9,
            "elevation_m": 583.0,
            "altitude_m": 616.9,
        },
        {
            "callsign": "CISF-FM",
            "frequency_mhz": 107.7,
            "latitude": 49.183889,
            "longitude": -122.844722,
            "antenna_height_m": 117.0,
            "elevation_m": 77.0,
        },
    ]
}


def _site(*nodes, **defaults):
    return {
        "name": "t",
        "defaults": {"rx_lat": 49.26, "rx_lon": -123.25, "rx_alt_m": 100.0, **defaults},
        "cells": [{"ring_id": "t", "core_lat": 49.19, "core_lon": -123.18}],
        "nodes": list(nodes),
    }


def test_node_takes_its_named_tower():
    fleet = build_site_fleet(_site({"node_id": "t-1", "tower": "CHAN-DT"}), _TOWERS)
    (node,) = fleet["nodes"]
    assert node["tx_callsign"] == "CHAN-DT"
    assert node["fc_hz"] == 521_000_000
    assert (node["tx_lat"], node["tx_lon"]) == (49.345, -122.972778)
    assert node["tx_alt_ft"] == round(616.9 * 3.28084, 1)
    assert node["rx_alt_ft"] == round(100.0 * 3.28084, 1)
    assert fleet["cells"][0]["ring_id"] == "t"


def test_shared_callsign_and_case_resolve_to_the_station():
    fleet = build_site_fleet(_site({"node_id": "t-1", "tower": "chan-dt-ax1"}), _TOWERS)
    assert fleet["nodes"][0]["tx_callsign"] == "CHAN-DT"


def test_altitude_falls_back_to_ground_plus_mast():
    fleet = build_site_fleet(_site({"node_id": "t-1", "tower": "CISF-FM"}), _TOWERS)
    assert fleet["nodes"][0]["tx_alt_ft"] == round(194.0 * 3.28084, 1)


def test_beam_azimuth_is_omitted_unless_set():
    fleet = build_site_fleet(
        _site({"node_id": "t-1", "tower": "CHAN-DT"}, {"node_id": "t-2", "tower": "CISF-FM", "beam_azimuth_deg": 147}),
        _TOWERS,
    )
    assert "beam_azimuth_deg" not in fleet["nodes"][0]
    assert fleet["nodes"][1]["beam_azimuth_deg"] == 147.0


def test_every_unknown_tower_is_named():
    with pytest.raises(ValueError, match="t-1: NOPE.*t-2: ALSO"):
        build_site_fleet(_site({"node_id": "t-1", "tower": "NOPE"}, {"node_id": "t-2", "tower": "ALSO"}), _TOWERS)


def test_checked_in_ubc_site_builds():
    root = Path(__file__).resolve().parent.parent / "sites" / "ubc"
    site = json.loads((root / "site.json").read_text(encoding="utf-8"))
    towers = json.loads((root / "towers.json").read_text(encoding="utf-8"))
    fleet = build_site_fleet(site, towers)
    assert fleet == json.loads((root / "fleet.json").read_text(encoding="utf-8"))
    assert all(n["node_id"].startswith(fleet["cells"][0]["ring_id"]) for n in fleet["nodes"])
    assert all(n["node_id"].startswith(SYNTHETIC_PREFIX) for n in fleet["nodes"])
