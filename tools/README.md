# Recording a simulated run

Two commands: one records a run, one turns it into tables and maps.

For the controlled experiments of the receiver-motion study (one scripted
aircraft, fixed seeds, the same flights every time), see
[Experiments](#experiments-one-command-reproducible) at the end. The lab
notebook is `experiments/NOTEBOOK.md`.

## The scene (`sites/ubc/`)

* **3 receivers** on one UBC rooftop (49°15'57.7"N 123°15'06.5"W), each on a
  different real tower: CHAN-DT (DTV, 521 MHz, ENE, the strongest), CHEK-DT
  (DTV, 485 MHz, S) and CISC-FM (FM, 107.5 MHz, NW). The towers sit about
  110–140° apart around UBC; every other DTV station in `towers.json` shares
  CHAN-DT's direction. Beams aim at YVR, 120° wide.
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

## Experiments: one command, reproducible

An experiment file (`experiments/baseline_static.json`) fixes everything about
a set of runs: the fleet, the duration, one scripted aircraft per run, and the
seed behind every random draw. Its flight plans and a fingerprint of each
recording are stored in the file, so every run of it replays exactly the same
flights with exactly the same detections. If the simulator ever stops
reproducing them, the run stops and says which seed changed.

Run everything with the backend's Python, which already has
`retina_geolocator`. It also needs matplotlib, once. The backend's
environment is made by uv and has no pip, so install with uv:

```
cd "$root\backend"
uv pip install --python .venv matplotlib
```

(Without uv: `.\.venv\Scripts\python.exe -m ensurepip` first, then
`.\.venv\Scripts\python.exe -m pip install matplotlib`.) A later `uv sync`
removes it again, since it is not in the backend's lock file; reinstall if the
experiment reports `No module named 'matplotlib'`.

where `$root` is your `retina-server` folder, e.g.
`$root = "C:\Users\ericf\OneDrive\Documents\Year 5 Term 1\ENPH 479\2652-Passive-Radar\retina-server"`.

Offline (about a minute; no server, no other terminals):

```
cd "$root\libs\retina-simulation"
..\..\backend\.venv\Scripts\python.exe tools\run_experiment.py experiments\baseline_static.json
```

Through the real server as well (adds the run duration per seed, ~26 min for
the baseline). Stop your own backend first; this starts a fresh one per seed:

```
..\..\backend\.venv\Scripts\python.exe tools\run_experiment.py experiments\baseline_static.json --system --backend-dir ..\..\backend
```

Output goes to `runs\baseline-static\<date_time>\`:

| File | Holds |
|---|---|
| `rmse_vs_time.png` | horizontal position and velocity RMSE against time (15 s sliding window), one line per seed |
| `rmse_vs_time_system.png` | the same for the server's solves (`--system` only) |
| `paths_map.png` | the 5 flights over UBC, the three illuminators and the beam |
| `summary.csv` | per seed and mean: availability, RMSE / mean / median / 95th percentile of position and velocity error |
| `rmse_vs_time.csv` | the numbers behind the plot |
| `manifest.json` | seeds, flight plans, solver settings, and the git commits that produced the run |
| `seed-<n>\` | that run's recording, `analysis\` (including `offline_solves.csv`) and `maps\`, as for any run; under `--system` also `system\` |

The offline solve (`tools/offline_solve.py`) runs the same geolocator the
server uses on the recorded detections, with the server's altitude sweep and
publication gates, but with perfect association and no tracker, so it is the
optimistic, exactly repeatable number; the `--system` run is what the full
pipeline does. Only seconds where all three channels detect the aircraft are
solved; the rest count against availability.
