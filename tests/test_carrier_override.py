"""Carrier selection: 500 MHz default, and a single-carrier override for generated fleets.

The tower tables hold the real US frequencies (e.g. 195 MHz for WSB-TV) and must
stay untouched; the override exists so a run can mimic the Vancouver ~500 MHz DTV
setup without rewriting them.
"""

import pytest

from retina_simulation.generator import _TOWERS_US, generate_fleet
from retina_simulation.world import NodeConfig

_FLEET = dict(n_nodes=16, n_cluster=16, n_clusters=1, use_tower_api=False, seed=42)


def _freqs(**kw):
    return {n["fc_hz"] for n in generate_fleet(**{**_FLEET, **kw})}


def test_node_config_default_carrier_is_500_mhz():
    assert NodeConfig().fc_hz == 500_000_000.0


@pytest.mark.parametrize("layout", ["ring", "dual", "scatter"])
def test_override_puts_every_node_on_that_frequency(layout):
    assert _freqs(metro="gvl", layout=layout, fc_hz_override=503_000_000.0) == {503_000_000.0}


def test_override_applies_to_metro_nodes_without_jitter():
    nodes = generate_fleet(n_nodes=40, regions=["us"], use_tower_api=False, seed=1, fc_hz_override=500e6)
    assert nodes and {n["fc_hz"] for n in nodes} == {500e6}


def test_without_override_tower_frequencies_are_unchanged():
    freqs = _freqs(metro="gvl", layout="ring")
    table = {t[3] for t in _TOWERS_US}
    assert freqs <= table
    assert 500_000_000.0 not in freqs


def test_without_override_atlanta_keeps_195_mhz_wsb():
    assert 195_000_000 in _freqs(metro="atl", layout="ring", n_nodes=30, n_cluster=30)
