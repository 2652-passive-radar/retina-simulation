# Receiver-motion study: lab notebook

A running log of what was built for the moving-receiver study, why, and what
was found. Newest entries at the bottom of each step.

## Step 0: stationary baseline

### Goal
- Baseline localisation and velocity error of the current (stationary) system:
  one aircraft, the UBC site, receiver not moving.
- 5 runs of 300 s, seeds 101, 202 (straight legs) and 303, 404, 505 (gentle turns).
- Every future run of the experiment must replay exactly these 5 flights.
- Outputs: horizontal position RMSE and horizontal velocity RMSE against time
  (one line per seed), per-seed and mean summary numbers, a map of the 5 paths,
  and the usual per-run analysis folders.

### Starting point (what the code did before this work)
- `--seed` on the orchestrator only seeds `generate_fleet`, which is skipped when
  a `--config` fleet file is given. With `sites/ubc/fleet.json` nothing was
  seeded: aircraft routes, measurement noise, missed detections and clutter all
  drew from Python's unseeded global `random`. Every earlier UBC run was
  different, even with `--seed 42`.
- Traffic came from the YVR arrival/departure model: aircraft spawn, fly a
  route, retire and respawn, so one aircraft in coverage for 300 s could not be
  guaranteed.
- The world advances in fixed steps (`world.step(tick_dt)`, tick 1 s); wall
  clock only stamps frames. So a recording does not need real time, only the
  server does.
- The server's radar-only solve, for a 3-node candidate (default settings):
  - altitude by sweep: pinned solves at 1.5, 3, 5, 7, 9, 11 km (plus the
    initial guess's altitude), keep the lowest `rms_delay`
    (`retina-server/backend/services/tasks/solver.py:_solve_best_altitude`)
  - publish only if `rms_delay <= 3.0 us` and `rms_doppler <= 60 Hz` (n=3) or
    `200 Hz` (other n)
  - the geometry is rebuilt from each node's config on every call to
    `retina_geolocator.multinode_solver.solve_multinode` (`NodeSetup`)

### 1. Reproducible randomness (`retina_simulation/world.py`)
- Added `SimulationWorld.rng_seed`. `None` (the default) keeps the old
  behaviour exactly: every draw still comes from the global `random`, in the
  same order. All 222 existing tests pass unchanged.
- With a seed set, each kind of randomness gets its own stream, created fresh
  from a key:
  - aircraft motion jitter: `(seed, "motion", aircraft id, tick)`
  - a detection's SNR, miss decision and delay/Doppler noise:
    `(seed, "detection", node id, tick, aircraft id)`
  - a frame's clutter: `(seed, "clutter", node id, tick)`
- Why keyed streams rather than one seeded stream: in step 1 the receiver will
  move, which changes distances, SNR and miss decisions. With one shared stream
  every changed decision would shift all later draws, so the static and moving
  runs would get different noise. Keyed streams give each detection the same
  noise in both runs: a paired comparison.
- Added a world tick counter (`_tick`, +1 per `step`), used as part of the key.

### 2. Scripted flights (`retina_simulation/scripted.py`, `world.py`)
- `SimulatedAircraft.turn_rate_deg_s`: a constant turn (deg/s, positive =
  clockwise) for an aircraft with no waypoints. 0 = straight.
- `FlightPlan`: start lat/lon, altitude, speed, heading, turn rate.
- Coverage rule, checked at every second of the 300 s flight, for every node:
  - bearing from UBC within the beam (148 deg +/- 60) minus a 10 deg margin
  - 3 to 35 km from UBC
  - bistatic angle (at the aircraft, between the Tx and Rx directions)
    at most 150 deg, which keeps it out of the forward-scatter zone along each
    Tx-UBC baseline, where geometry is poor
- `sample_plan(seed, kind)` draws start point (inside the beam), heading,
  speed 70-110 m/s, altitude 1-3 km and, for "turn", 0.3-1.0 deg/s either way;
  it redraws until the whole flight passes the coverage rule.
- The flight's sampler has its own stream (`flight:<seed>:<kind>`), separate
  from the run's noise streams.

### 3. Fast recording (`retina_simulation/experiment.py`, `recorder.py`)
- `experiments/baseline_static.json`: fleet, duration (300 s), tick (1 s), a
  fixed start time (2026-01-01 00:00 UTC) so timestamps are identical every
  run, the coverage and flight ranges, solver settings, and the 5 runs.
- The first time the file is resolved, each run's sampled plan is written into
  it. After that the plan is read, never re-sampled.
- `record_run` steps the world 300 times without a server or real-time waits,
  frames all 3 nodes every tick, and writes the usual `nodes.json`,
  `frames.ndjson`, `truth.ndjson` plus a new `rx_trajectory.ndjson`
  (receiver pose per tick; constant for now, ready for step 1).
- One seed records in ~0.3 s. Two recordings of the same seed are identical.
- A fingerprint per run (frame and detection counts, sums of delay and Doppler,
  final aircraft position) is stored in the experiment file; a later run that
  produces something different is reported, with a tolerance so last-digit
  floating-point differences between computers do not trip it.
- Observation: the simulator's miss model detects the aircraft in ~74% of
  frames per channel, so all 3 channels see it together in ~40% of seconds.

### 4. Offline solve (`tools/offline_solve.py`)
- For each second with the aircraft detected by all 3 channels: radar-only
  initial guess, then `retina_geolocator.solve_multinode`, then the server's
  publication gates (rms_delay <= 3.0 us, rms_doppler <= 60 Hz for 3 nodes).
- Initial guess: grid search (0.5 km spacing, +/-60 km, inside the shared beam)
  for the point and altitude layer whose bistatic delays best fit the measured
  ones. Truth is never used for it; the recorder's labels are used only to pick
  which detection is the aircraft (perfect association).
- Altitude: "sweep" as the server's default (pinned solve at 1.5, 3, 5, 7, 9,
  11 km and the guess's layer, lowest rms_delay wins). "free" also implemented.
- Errors: horizontal position (east, north, total) and horizontal velocity
  against the simulator's exact truth, plus the same against the ADS-B report.
- Takes ~70 s for all 5 seeds (sweep: 7 solves per second).

### Finding: the simulator's Earth model added ~65 m of error
- First results: horizontal RMSE 110-170 m. Altitude snaps to the 1.5 or 3 km
  layer while the aircraft fly at 1.1-2.7 km, which accounts for part of it.
- Test: solve noise-free (true delay/Doppler) with the true altitude given.
  Expected ~0 m; got median 65 m, max 101 m.
- Cause: the simulator converted lat/lon to local km with a spherical
  flat-Earth formula (R = 6371 km); the geolocator uses the WGS84 ellipsoid.
  At 20-30 km range the two disagree by tens of metres, so the simulated echoes
  were not where the solver's Earth puts the aircraft.
- Confirmed by swapping in WGS84: noise-free error 0.0 m.
- Decision (with you): use WGS84 everywhere in the simulator.

### 5. WGS84 geometry in the simulator (`world.py`)
- `_lla_to_enu` / `_enu_to_lla` now go through WGS84 ECEF, the same maths as
  the geolocator (checked: identical to the millimetre).
- Aircraft movement (`_advance_latlon`) uses WGS84's meridian and
  prime-vertical radii, so the distance flown each second matches the velocity
  the Doppler is computed from. Used by scripted, synthetic and live aircraft.
- After the change, noise-free solves: median 0.04 m, max 0.07 m; velocity
  0.18 m/s. All 222 existing tests pass.
- Note for the wider team: this shifts every simulated detection slightly
  (up to a few tenths of a microsecond at long range). It is a correction.

### Decisions taken with you
- Geometry: WGS84 everywhere.
- Altitude mode for the baseline: sweep (the server default).
- Only seconds with all 3 channels are solved; the rest count against
  availability.

### 6. Figures and tables (`tools/experiment_report.py`)
- Windowed RMSE: at each second t, the RMSE of the published solves within
  +/-7.5 s of t (15 s window, set in the experiment file). Left blank where
  the window holds fewer than 3 solves, rather than drawing a "RMSE" from one
  or two points. Each solve's own error is drawn as a faint dot behind.
- `rmse_vs_time.png`: two panels on one time axis, horizontal position RMSE (m)
  and horizontal velocity RMSE (m/s); one line per seed, labelled with the
  seed, flight type and that seed's overall RMSE. Velocity is horizontal only:
  radar-only solves pin altitude.
- `summary.csv`: per seed and a mean row: ticks, solves published,
  availability, and RMSE / mean / median / 95th percentile for position and
  velocity, plus RMSE scored against the ADS-B reports instead of exact truth.
- `paths_map.png`: Natural Earth coastline (bundled, so it renders offline and
  identically each time), UBC, the three illuminators with frequency, dashed
  baselines, the surveillance beam wedge, YVR, a 10 km scale bar, and the 5
  flights with start dots, direction arrows and seed labels.
- Colours: the same 5 fixed colours for the same seeds in every figure;
  palette checked for colour-blind separation (passes). Three of the colours
  are low-contrast on white, so every line also has a legend entry, the map
  has seed labels, and the numbers are in the CSVs.

### 7. One-command runner (`tools/run_experiment.py`)
- Steps: resolve plans (first run only, written into the experiment file) ->
  record each seed -> check its fingerprint -> offline solves (one process per
  seed) -> per-seed `analyze_run` tables, summary and delay-Doppler maps (as
  before) -> figures and tables across seeds -> `manifest.json`.
- `--system` also streams every seed through the real backend: it starts a
  fresh backend per seed (dev settings, NODE_FUZZ_MODE=off), runs the
  orchestrator with `--experiment`/`--experiment-seed` (same flight, same
  seeded noise, ADS-B withheld), stops the backend, and analyses the server's
  solves into `seed-<n>/system/`. It refuses to start if a backend is already
  running on port 8000.
- `manifest.json` records the command, Python version, solver settings, plans,
  fingerprints and the git commit (and whether the tree was dirty) of
  retina-simulation, retina-geolocator and, with `--system`, retina-server.
- Orchestrator: new `--experiment FILE --experiment-seed N` (implies
  `--adsb-truth-only`); the recording then also carries the receiver pose.
- Runtime here: ~1 minute for everything offline (solves 21 s, the rest is the
  delay-Doppler maps; `--no-maps` skips those).

### 8. Tests
- `tests/test_experiment.py` (14 tests): same seed gives identical frames,
  different seed different ones; adding a second aircraft leaves the first
  one's noise untouched (keyed streams); unseeded worlds still follow the
  global `random`; WGS84 round trip, distances on the ellipsoid, distance flown
  matches velocity; constant turn; the world flies what `propagate` predicts
  (within 0.1 km over 300 s); sampled plans are covered and repeatable;
  fingerprint tolerance; **the frozen baseline file reproduces all 5 stored
  fingerprints** (so a simulator change that would move the baseline fails CI);
  receiver pose is recorded.
- `tests/test_offline_solve.py` (3 tests, skipped where retina_geolocator is
  not installed, as in this repo's CI): radar-only initial guess within 2 km,
  every 3-channel tick solved and scored, deterministic.
- 240 tests pass; ruff check/format clean; no new dead code.

### Result: step 0 baseline (offline, receiver static)

| Seed | Flight | Solves (of 300 s) | Position RMSE | Position median / p95 | Velocity RMSE |
|---|---|---|---|---|---|
| 101 | straight | 118 (39%) | 69.6 m | 49 / 160 m | 3.16 m/s |
| 202 | straight | 109 (36%) | 72.9 m | 49 / 168 m | 3.50 m/s |
| 303 | turn | 135 (45%) | 63.9 m | 37 / 142 m | 4.05 m/s |
| 404 | turn | 132 (44%) | 116.9 m | 102 / 196 m | 3.79 m/s |
| 505 | turn | 118 (39%) | 85.9 m | 61 / 141 m | 3.31 m/s |
| **mean** | | **122 (41%)** | **81.8 m** | 60 / 161 m | **3.56 m/s** |

- Two identical runs give byte-identical recordings and an identical
  `summary.csv`.
- Scoring against ADS-B instead of exact truth changes RMSE by at most 0.1 m
  (position) and 0.1 m/s (velocity).
- Seed 404 is the worst: it starts close to UBC, west of YVR, where the error
  is ~150-180 m for the first minute and settles to ~100 m.
- Most of the remaining position error is altitude: the sweep pins altitude to
  the 1.5 or 3 km layer while the aircraft fly at 1.1-2.7 km. With the
  receivers all on one rooftop, altitude is weakly observable ("free" altitude
  gave similar numbers).
- Turns vs straight legs: no clear difference in position error; velocity RMSE
  is slightly higher on two of the three turning flights (303, 404).

### Open items
- The `--system` path is untested end to end: the backend cannot run in the
  environment this was built in. Its pieces were checked (the orchestrator
  loads the experiment, flies the scripted aircraft, withholds ADS-B and
  starts recording). First real run should be on your machine.
- Ground truth for the offline solve: positions are exact, but each solve is
  independent (no tracking), so velocity comes only from the 3 Doppler
  measurements of that second.

## Step 0b: first system (server) run, 2026-10-17

### Result as first reported
- Run on your machine with `--system`. Offline: position RMSE 81.8 m,
  velocity 3.56 m/s, availability 41%. System: **3,197 m, 39.3 m/s, 6%**.

### Diagnosis
- Published server solves over the 5 seeds, split by node count:
  - 2-node: 17 solves, median 3,392 m, max 19,216 m
  - 3-node: 66 solves, median 115 m, max 4,827 m
- The worst 2-node solves sit 12-19 km **west** of the aircraft: the second
  intersection of two delay ellipses at a pinned altitude. Seen from UBC it is
  still inside the 120 deg beam, so the server's 2-node beam check cannot
  reject it. A property of the single-rooftop geometry.
- Some "3-node" outliers were the same solve twice. The raw records showed one
  key and one measurement time published as `n_nodes 2`, then ~1 s later as
  `n_nodes 3` with an identical `lat`/`lon`. Cause, from the server code: the
  3-node re-solve of the same epoch has no new time step, so the track
  smoother holds the map position at the previous (2-node) answer
  (`track_filter._hold_position`); the 3-node answer is only in
  `raw_lat`/`raw_lon`.
- The analysis was also scoring the smoothed map position rather than the raw
  solve, which is not comparable with the offline solve.
- 6% is the solve rate, not a failure: the server attempts ~35-66 solves per
  300 s run by design and keeps a solved aircraft on the map in between.
  `dark_follow_shadow` (the largest "not published" group) is a diagnostic
  shadow lane that never publishes.

### Fixes (analysis only, nothing in the server)
- `tools/analyze_run.py`: one solve per (key, measurement time), keeping the
  last published record; scored on `raw_lat`/`raw_lon`; smoothed map position
  and its error kept as extra columns; velocity error added; the summary text
  breaks errors down by node count and explains `dark_follow_shadow`.
- `tools/experiment_report.py`: `summary.csv` now has `system` (3-node, the
  like-for-like comparison) and `system_2node` rows; new
  `error_vs_time_system.png` (each solve's error, joined per seed) because the
  server's ~1 solve per 15 s is too sparse for a sliding-window RMSE;
  `errors.csv` holds every solve's error. Offline keeps the sliding-window
  RMSE plot (dense enough; the per-solve version is unreadable at ~600
  solves).
- `tools/run_experiment.py --reanalyze RUN_DIR`: rebuilds the analysis of an
  existing output folder without recording, solving or a server.
- `tests/test_analyze_solves.py`: the duplicate pair from the real run (one
  kept, scored on its raw position), keys and epochs kept apart, shadow and
  ADS-B-assisted records left out, velocity error.
- Checked end to end on a constructed system folder (offline solves re-shaped
  as server records, with 2-node mirrors and held duplicates): duplicates
  collapse, 2-node solves land in their own row, plots render.

### Step 0c: why some server solves are far off (seed 303)
- `tools/inspect_solves.py` prints a bad solve's own server record and, with
  `--frames`, what every node detected over the solve's time span.
- Seed 303, t = 66.4 s, 3-node solve 858 m off: it began as a 2-node solve and
  the server **adopted** CHEK-DT's measurement to widen it to 3 nodes
  (`adopt_meta.outcome = widened`), from measurements **5.0 s apart**
  (`epoch_skew_s = 5.014`), dead-reckoned onto one epoch. On a turning aircraft
  they no longer agree: rms_delay 0.92 us (a normal 3-node solve is 0.02-0.2),
  vertical velocity saturated, velocity flagged untrusted. 0.92 us still passes
  the 3 us gate, so it published. A pipeline effect the offline solve, which
  never mixes seconds, cannot reproduce.
- 3-node solves at 218 s and 248 s: 42-61 m in position but 40-70 m/s in
  velocity. Their published velocity is a multi-epoch constant-velocity fit
  (`vel_source = cv_fit`), which a turning flight violates. The record keeps
  that second's own solve velocity as `solver_vel_east/north`.
- 2-node solves: grid altitude guesses of 5 and 7 km for an aircraft at
  1.55 km, plus the mirror intersection; position from a constant-velocity
  fit (`pos_source = cv_fit`).
- Analysis change: score each record's single-second solve, like offline:
  `solve_raw_lat/lon` (when the server swapped in a fit) else `raw_lat/lon`;
  `solver_vel_east/north` else `vel_east/north`. The published position and
  velocity, with their own errors, and `pos_source`, `vel_source`,
  `rms_delay_us`, `epoch_skew_s` and an `adopted` flag are kept in
  `solves.csv`.
