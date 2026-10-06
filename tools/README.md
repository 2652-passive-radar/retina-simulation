# Recording a simulated run for error analytics

Four steps: run the server, run a site fleet with `--record`, record the
server's solves next to it, then analyse and render maps. Everything lands in
one run directory.

| File | Written by | Holds |
|---|---|---|
| `nodes.json` | orchestrator | the node configs used (RX, tower, frequency, beam) |
| `frames.ndjson` | orchestrator | every node frame: measured delay (µs) / Doppler (Hz) / SNR per detection, with its truth label (object id or clutter), noise-free delay/Doppler and the residual |
| `truth.ndjson` | orchestrator | every aircraft's absolute state each tick (lat, lon, alt, ENU velocity, ADS-B status, dark/anomalous), dark ones included |
| `solves.ndjson` | `record_solves.py` | every server solve: state guess, fit residuals, node count, outcome |
| `analysis/*.csv` | `analyze_run.py` | per detection, per node/tower, per target, and per solve with the localisation error vector (east/north/up m) |
| `maps/*.npz` | `render_maps.py` | labelled delay-Doppler maps on blah2's 301 × 411 grid |

## Choosing nodes and towers

A site file places receivers and names each one's illuminator by callsign out
of a towers.retina.fm search saved as JSON (`sites/ubc/` is the worked
example). To change towers, edit `site.json` and rebuild:

```
python -m retina_simulation.site_fleet sites/ubc/site.json sites/ubc/towers.json -o sites/ubc/fleet.json
```

For a new area, save `https://towers.retina.fm/api/towers?lat=..&lon=..&radius_km=80&limit=50&source=auto`
as `towers.json`. A cell's `ring_id` must prefix its node ids.

## Windows (PowerShell), from the retina-server checkout

Window 1, backend (as usual):

```
cd C:\Users\chase\retina-server\backend
$env:RETINA_ENV = "dev"; $env:AUTH_ALLOW_ANONYMOUS_ADMIN = "1"; $env:SYNTHETIC_FLEET_ENABLED = "1"
.\.venv\Scripts\uvicorn main:app --reload
```

Window 2, the UBC fleet, recording for 10 minutes:

```
cd C:\Users\chase\retina-server\libs\retina-simulation
$env:PYTHONUTF8 = "1"
..\..\backend\.venv\Scripts\python.exe -m retina_simulation.orchestrator --config sites\ubc\fleet.json --metro yvr --mode adsb --interval 0.5 --min-aircraft 25 --max-aircraft 35 --seed 42 --duration 600 --record runs\ubc-1
```

Window 3, the server's solves, started straight after the fleet connects:

```
cd C:\Users\chase\retina-server\libs\retina-simulation
..\..\backend\.venv\Scripts\python.exe tools\record_solves.py --out runs\ubc-1 --duration 630
```

Then:

```
..\..\backend\.venv\Scripts\python.exe tools\analyze_run.py runs\ubc-1
..\..\backend\.venv\Scripts\python.exe tools\render_maps.py runs\ubc-1 --every 2
```

Maps are about 230 kB a frame compressed (noise does not compress), so a
10-minute 5-node run is about 700 MB at `--every 1`. `--every` and
`--max-frames` trim it.

## What the numbers are, and are not

* Residuals are the sim's measurement noise model (measured minus the
  noise-free value), not a radar's: they check what the solver is fed.
* The localisation error is the solve minus truth interpolated to the
  measurement time. `err_up_m` is only meaningful where `altitude_mode` is
  not `pinned` (dark-lane solves pin altitude).
* Maps draw only the targets the sim reported. An aircraft in the beam that
  the sim's SNR model dropped is absent, where a real map would still hold
  its sub-threshold energy. Targets beyond blah2's 205 µs delay window are
  listed with `in_grid: false` and not drawn.
* All five UBC receivers share one rooftop placeholder and one beam, so they
  see the same aircraft. Spread them out (real sites) for geometry diversity.
