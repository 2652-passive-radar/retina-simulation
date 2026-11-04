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
| `error_vs_time_system.png` | each of the server's 3-node solves' position and velocity error against time, one line per seed (`--system` only; the server solves every ~15 s, too sparse for a sliding-window RMSE) |
| `paths_map.png` | the 5 flights over UBC, the three illuminators and the beam |
| `summary.csv` | per seed and mean: availability, RMSE / mean / median / 95th percentile of position and velocity error |
| `rmse_vs_time.csv`, `errors.csv` | the numbers behind the plots: windowed RMSE (offline), and every solve's error (all sources) |
| `manifest.json` | seeds, flight plans, solver settings, and the git commits that produced the run |
| `seed-<n>\` | that run's recording, `analysis\` (including `offline_solves.csv`) and `maps\`, as for any run; under `--system` also `system\` |

The offline solve (`tools/offline_solve.py`) runs the same geolocator the
server uses on the recorded detections, with the server's altitude sweep and
publication gates, but with perfect association and no tracker, so it is the
optimistic, exactly repeatable number; the `--system` run is what the full
pipeline does. Only seconds where all three channels detect the aircraft are
solved; the rest count against availability.

How the server's solves are scored (`--system`):

* **Like with like.** `summary.csv` reports the server's 3-node solves as
  `system`, the same rule as offline, and its 2-node solves separately as
  `system_2node`. With every receiver on one rooftop, two delay ellipses and a
  pinned altitude often cross in two places inside the beam, and the server
  sometimes publishes the wrong one, kilometres away: a real weakness of the
  geometry, but a different question from the 3-node accuracy.
* **One solve per epoch.** The server can publish one key at one measurement
  time twice: a 2-node solve, then a 3-node re-solve when the third node joins.
  Only the later one is kept.
* **Raw solve position.** The server smooths what it puts on the map, and holds
  the map position still for a re-solve of the same epoch, so the published
  `lat`/`lon` can be the earlier 2-node answer. The score uses the solver's
  own `raw_lat`/`raw_lon`, like the offline solve; the smoothed map position
  and its error are kept beside it in each `solves.csv`.
* **Availability means something different.** The server does not re-solve an
  aircraft every second: it keeps a solved aircraft on the map between solves.
  Its solves-per-second is therefore far below offline's by design.
  `dark_follow_shadow` records are a diagnostic lane that is never published.

To rebuild the analysis of an existing output folder after a change to the
analysis (no recording, solving or server needed):

```
..\..\backend\.venv\Scripts\python.exe tools\run_experiment.py experiments\baseline_static.json --reanalyze runs\baseline-static\<date_time>
```

## Step 1: receiver motion (offline)

`experiments/receiver_motion.json` reuses the step 0 flights (it names
`baseline_static.json` as its `base`) and moves the receiver platform: static,
straight east and north, back and forth, and round a circle, each at walking
(1.5 m/s) and vehicle (15 m/s) speed, with the moving receiver's own Doppler
left out and included. Every recording is solved twice: by the stationary
algorithm (S, receiver assumed at its registered position, as the server does
today) and with geometric compensation (G, the receiver's true position each
second). Plans and recordings are fingerprinted exactly as in step 0.

```
cd "$root\libs\retina-simulation"
..\..\backend\.venv\Scripts\python.exe tools\run_experiment.py experiments\receiver_motion.json --out C:\radar-runs\receiver-motion
```

(`--out` keeps the ~85 recordings out of OneDrive; leave it off to use
`runs\receiver-motion\<date_time>\`.) About 10 minutes on 4 cores. `--jobs N`
sets the number of processes (default: up to 8); if a worker process crashes,
which happened once on Windows with 16, the run finishes the remaining solves
one at a time, and `--jobs 1` avoids multiprocessing altogether. `--reanalyze <folder>` rebuilds the figures and
tables of an existing run.

| File | Holds |
|---|---|
| `overview.png` | position and velocity RMSE of every moving case, S beside G, against the static baseline; receiver Doppler off and on |
| `summary.csv` | per case x receiver Doppler x algorithm x seed, plus means: RMSE / median / p95 (every solve, and published only), availability, gate rejections, final offset |
| `receiver_paths.png` | the receiver trajectories |
| `paths_map.png` | the flights, as in step 0 |
| `cases\<case>\rmse_vs_time_doppler-<off\|on>.png` | S, G and the static baseline over time, mean over seeds with the seed range shaded |
| `cases\<case>\error_vs_offset.png` | every solve's position error against the receiver's offset (circle: against its direction) |
| `cases\<case>\doppler-<off\|on>\seed-<n>\` | the recording, with `analysis\offline_solves_stationary.csv` and `offline_solves_geometric.csv` |

One recording can also be solved by hand:
`tools\offline_solve.py <seed folder> --receiver geometric`.

## Step 2: receiver Doppler compensation (offline)

`experiments/receiver_doppler.json` reuses step 1 (it names
`receiver_motion.json` as its `base`), keeps only the physical recordings
(receiver Doppler on) and adds a third algorithm, GD: geometric compensation
with each channel's receiver Doppler, (fc/c)(u_RP + u_TR) . v_R, removed from
the measured Doppler before the solve. The correction is computed from the
receiver's true velocity at G's solved position, then the epoch is solved
again. The recordings are the step 1 ones, checked against the stored
fingerprints.

```
cd "$root\libs\retina-simulation"
..\..\backend\.venv\Scripts\python.exe tools\run_experiment.py experiments\receiver_doppler.json --out C:\radar-runs\receiver-doppler
```

Same options as step 1 (`--jobs`, `--reanalyze <folder>`); leave `--out` off
for `runs\receiver-doppler\<date_time>\`. The output has the step 1 layout
(`overview.png`, `summary.csv`, `receiver_paths.png`, `paths_map.png`,
`cases\<case>\rmse_vs_time_doppler-on.png`, `cases\<case>\error_vs_offset.png`,
with every recording under `cases\<case>\doppler-on\seed-<n>\`) with GD added
everywhere, plus:

| File | Holds |
|---|---|
| `velocity_vs_speed.png` | velocity RMSE against the receiver's speed at the time, every moving case pooled |
| `doppler_correction.png` | each channel's receiver Doppler, and how far GD's correction was from it |
| `summary.csv` columns `rxdop_rms_hz`, `rxdop_corr_err_rms_hz` | the same two numbers per case and seed |

Each solve file also carries, per channel, `rxdop_true_hz_<node>` (the receiver
Doppler in that measurement) and, for GD, `rxdop_corr_hz_<node>` (what GD
subtracted). One recording by hand:
`tools\offline_solve.py <seed folder> --receiver geometric_doppler`.

## Step 3a: GNSS error model

`retina_simulation/gnss.py` makes the receiver's GNSS position fixes (position
only, no velocity): per axis, the true position plus a first-order
Gauss-Markov (AR(1)) error and white noise, at a set update rate, with an
uncorrected time offset and a known latency. Its parameters are
`sigma_white_m` (0.5), `sigma_corr_m` (1.5, the correlated part's steady-state
size), `tau_s` (60), `rate_hz` (1), `time_offset_s` (0) and `latency_s` (0).

Figures that check the model behaves as specified (no server, no recordings,
about 10 s):

```
cd "$root\libs\retina-simulation"
..\..\backend\.venv\Scripts\python.exe tools\gnss_figures.py --out "C:\radar-runs\gnss-model\$(Get-Date -Format yyyy-MM-dd_HHmmss)"
```

`--sigma-white`, `--sigma-corr`, `--tau` and `--rate` change the model. Output:
`gnss_autocorrelation.png` (theory against simulated, as in the paper),
`gnss_error_time_series.png`, `gnss_time_constant.png`, `gnss_rate_check.png`,
`gnss_horizontal_error.png` and `stats.csv` (set against measured values).

## Step 3b: receiver Kalman filter

`tools/rx_filter.py`: per axis a linear Kalman filter on [position, velocity,
GNSS drift], fed by GNSS position fixes (step 3a), starting at the first fix.
`tools/rx_filter_report.py` runs it on every step 1 motion case (5 seeds x 4
GNSS realisations each, about 25 s, no radar) against the true trajectory,
with two baselines: the same filter treating GNSS error as white, and raw fixes
with differenced velocity.

```
cd "$root\libs\retina-simulation"
..\..\backend\.venv\Scripts\python.exe tools\rx_filter_report.py experiments\receiver_filter.json --out "C:\radar-runs\receiver-filter\$(Get-Date -Format yyyy-MM-dd_HHmmss)"
```

Settings (GNSS model, the filter's sigma_a and initial velocity uncertainty
per platform mode, realisations) are in `experiments/receiver_filter.json`.
Output: `overview.png` (position and velocity RMSE per case and variant),
`time_series_<case>.png` (one run's east error over time with the filter's
+/-2 sigma), `consistency.png` (mean NEES over time against its 95% band),
`summary.csv`, `manifest.json`.
