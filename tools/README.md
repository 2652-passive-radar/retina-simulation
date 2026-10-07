# Recording a simulated run

Two commands: one records a run, one turns it into tables and maps.

## The scene (`sites/ubc/`)

* **3 receivers** on one UBC rooftop (49°15'57.7"N 123°15'06.5"W), each on a
  different real tower: CHAN-DT (TV, 521 MHz, ENE), CHEK-DT (TV, 485 MHz, S)
  and CISC-FM (FM, 107.5 MHz, NW). Beams aim at YVR, 120° wide.
* **3 aircraft**, all with ADS-B, flying in and out of YVR within 40 km, so
  at least one is in view about 97% of the time.
* `--adsb-truth-only` keeps the ADS-B from the server: it solves every aircraft
  from radar alone, and the ADS-B is kept as the truth to score it against.

To change towers or the receiver position, edit `sites/ubc/site.json` (any
callsign in `towers.json`) and rebuild:

```
python -m retina_simulation.site_fleet sites/ubc/site.json sites/ubc/towers.json -o sites/ubc/fleet.json
```

Node ids must start with `synth-`, or the server treats them as hardware and
keeps them off the map.

## Windows (PowerShell)

Window 1, backend:

```
cd C:\Users\chase\retina-server\backend
$env:RETINA_ENV = "dev"; $env:AUTH_ALLOW_ANONYMOUS_ADMIN = "1"; $env:SYNTHETIC_FLEET_ENABLED = "1"
$env:NODE_FUZZ_MODE = "off"
.\.venv\Scripts\uvicorn main:app --reload
```

Window 2, the run (10 minutes; it stops by itself and also records the
server's solves):

```
cd C:\Users\chase\retina-server\libs\retina-simulation
$env:PYTHONUTF8 = "1"
..\..\backend\.venv\Scripts\python.exe -m retina_simulation.orchestrator --config sites\ubc\fleet.json --metro yvr --adsb-truth-only --metro-traffic-frac 1.0 --min-aircraft 3 --max-aircraft 3 --seed 42 --duration 600 --record runs\ubc-1
```

Optional window 3, the map: `npm run dev -w dashboard` from
`C:\Users\chase\retina-server`, open http://localhost:5174/map and press
**Raw**. It then shows only the ADS-B truth dots, each node's delay arcs and
the radar-only solves.

After the run:

```
..\..\backend\.venv\Scripts\python.exe tools\analyze_run.py runs\ubc-1
..\..\backend\.venv\Scripts\python.exe tools\view_maps.py runs\ubc-1
```

## What you get (`runs\ubc-1\analysis\` and `runs\ubc-1\maps\`)

| File | For | Holds |
|---|---|---|
| `detections.csv` | ML | every detection: delay (µs), Doppler (Hz), SNR, label (aircraft id or `clutter`), true delay/Doppler, residuals (measured − true) |
| `maps\maps_*.npz` + `index.json` | ML | one delay-Doppler map per node per second, with a label mask (which pixels are which aircraft) |
| `solves.csv` | moving receiver | each radar-only solve: state guess (position, velocity), which aircraft it was, that aircraft's ADS-B state, error vector (east/north/up m) |
| `aircraft.csv` | moving receiver | each aircraft's ADS-B state every second |
| `summary.txt` | everyone | headline numbers |

The raw recording (`frames.ndjson`, `truth.ndjson`, `solves.ndjson`,
`nodes.json`) stays in the run folder in case the tables need rebuilding.

## Maps

There is one map per node per second. Each is blah2's grid: 301 Doppler bins
(±300 Hz) × 411 delay bins (−5 to 200 µs). The files store only the aircraft
and clutter peaks, not the noise, which keeps them small. Add noise when you
load one:

```python
import numpy as np, sys
sys.path.insert(0, "tools")
from render_maps import with_noise
z = np.load("runs/ubc-1/maps/maps_000.npz")
map_db = with_noise(z["signal"][0])   # what the radar would show, in dB
labels = z["instance"][0]             # 0 = background, k = k-th target in index.json
```

`tools\view_maps.py` saves PNGs to `maps\png\` to look at. Each image shows
the map on the left and its labels on the right; the console says which
colour is which aircraft. `--node synth-ubc-cisc` picks one node and
`--count 30` takes more.

## Limits

* Residuals come from the sim's noise model, not a real radar.
* Radar-only solves pin altitude, so `err_up_m` is not a solved quantity
  (`alt_pinned` is true).
* A map holds only the peaks the sim reported. An aircraft its detection
  model missed leaves nothing, where a real map would still show a weak peak.
* All receivers share one rooftop and one beam. Real, spread-out sites would
  give better geometry.
