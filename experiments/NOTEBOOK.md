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

## Step 1: receiver motion, offline (2026-10-18)

### Built
- `retina_simulation/motion.py`: a `Motion` (static, linear, oscillate,
  circle) gives the platform's east/north offset from its registered position
  R0 and its velocity at any time; `place_receivers` moves every node's
  receiver to R0 + offset (WGS84) and sets its velocity. One platform, all
  three channels on one trajectory; height held; beam fixed in the world.
- `world.py`: `NodeConfig.rx_vel_enu`; the moving receiver's Doppler term
  `(1/lambda)(u_RP + u_TR).v_R` added in `generate_detections_for_node`, behind
  `SimulationWorld.rx_doppler` (on = physical; off isolates geometry). Delay,
  SNR and the beam check follow the receiver automatically because they are
  computed from the node's current position.
- `experiment.py`: an experiment file can name a `base` and inherit its fleet,
  flights and settings (only its own keys are saved back); `record_run` takes a
  motion and the Doppler switch and logs the pose and offset every tick.
- `experiments/receiver_motion.json`: 9 cases (static + linear east/north,
  oscillate, circle x walk 1.5 m/s / vehicle 15 m/s), receiver Doppler off/on,
  algorithms S and G; 17 recordings x 5 seeds, fingerprinted.
- `tools/offline_solve.py --receiver stationary|geometric`: G solves each tick
  with node configs (and the initial-guess grid) moved to that tick's receiver
  pose. Each row also carries the receiver offset and, per channel,
  `bias_us_<node>`: the true delay change caused by the receiver being off its
  registered position.
- `tools/motion_experiment.py` (run through `run_experiment.py`) and
  `tools/motion_report.py`: record, check fingerprints, solve in parallel,
  report.

### Verified
- `tests/test_receiver_motion.py` (17): each profile's velocity is the
  derivative of its offset; peak speed; circle radius and closure; oscillate
  amplitude; **the receiver Doppler term matches -(1/lambda) dRb/dt by finite
  difference with aircraft and receiver both moving**; the study inherits the
  baseline flights; **static reproduces the step 0 fingerprints**; the receiver
  Doppler switch changes Doppler and nothing else.
- `tests/test_offline_solve.py`: noise-free, receiver 4.5 km east: G recovers
  the aircraft to < 1 m, S is > 500 m off; offset and per-channel bias reported.
- Noise-free, true altitude: G's worst error 0.07 m on linear-east-vehicle,
  circle-vehicle and linear-north-walk.
- 266 tests pass.

### Results (offline, 5 seeds; every solve counted, including rejected ones)

| Case | S position RMSE | G position RMSE | S velocity RMSE (Doppler off / on) | G velocity RMSE (off / on) | S availability |
|---|---|---|---|---|---|
| static | 82 m | 82 m | 3.6 | 3.6 | 41% |
| linear-east-walk | 469 m | 82 m | 5.0 / 5.4 | 3.6 / 3.9 | 41% |
| linear-east-vehicle | 4,455 m | 82 m | 18.0 / 31.1 | 3.8 / 13.2 | 39% |
| linear-north-walk | 220 m | 81 m | 3.3 / 3.5 | 3.6 / 3.8 | 41% |
| linear-north-vehicle | 1,965 m | 83 m | 4.2 / 11.1 | 3.6 / 11.3 | 28% |
| oscillate-walk | 91 m | 82 m | 3.6 / 3.7 | 3.6 / 3.7 | 41% |
| oscillate-vehicle | 268 m | 82 m | 3.9 / 9.3 | 3.6 / 9.1 | 41% |
| circle-walk | 124 m | 82 m | 3.7 / 3.9 | 3.6 / 3.7 | 41% |
| circle-vehicle | 849 m | 81 m | 4.6 / 12.6 | 3.6 / 11.9 | 41% |

(position RMSE is the same with the receiver Doppler off and on to < 1%)

### Findings
1. **Geometric compensation removes the geometric error completely.** G's
   position RMSE is 81-83 m in every case, the static baseline's 81.8 m.
2. **The stationary algorithm degrades with displacement, about 0.7-1.9 m of
   position error per metre moved.** Over the last 30 s: walking north 356 m
   at 450 m offset (0.8 m/m, as first-order theory predicted); walking east
   824 m at 450 m (1.8 m/m); vehicle east 8.3 km at 4.5 km; vehicle north
   3.2 km at 4.5 km.
3. **Direction matters more than first-order theory said, because of the
   altitude sweep.** East motion biases the delays so that S picks the 3 and
   5 km altitude layers (aircraft at 1.1-2.7 km) in the last minute; north
   motion leaves it at 1.5 km. The wrong altitude roughly doubles the
   horizontal error. The circle shows the same thing directly: at a constant
   716 m offset, S's error swings between ~500 m and ~1,300 m with direction,
   peaking with the offset towards ~110 deg.
4. **The publication gates barely catch it.** Driving 4.5 km east, S is
   ~8 km off at the end yet only ~7 solves per seed are rejected; S keeps
   publishing in 39% of seconds. Only the north-driving vehicle trips the
   delay gate often (35 per seed, availability 41% -> 28%).
5. **Errors follow the offset, not the history.** Oscillating, S's error
   returns towards the baseline as the receiver passes home (walk 91 m RMSE
   with +/-24 m excursions).
6. **The receiver's own Doppler leaves position alone but spoils velocity.**
   With it on, G's velocity RMSE rises from 3.6 m/s to 9-13 m/s at vehicle
   speed (3.7-3.9 m/s walking). That is what receiver Doppler compensation,
   the next step, has to remove.

### Open items
- Through the server: needs a per-frame receiver pose on the server side.
- Next step: receiver Doppler compensation in the solver
  (`f_meas = f_static + (1/lambda)(u_RP + u_TR).v_R`).

### Step 1 fix: Windows worker crash (2026-10-18)
- First run on your machine (16 logical cores) died in the solve stage:
  `BrokenProcessPool` after a worker exited abruptly, the others then failing
  with `WinError 6 (handle is invalid)`. Step 0 (5 workers) never hit it.
- Likely cause: 16 heavy workers (numpy, scipy, matplotlib each), each also
  starting a maths thread per core.
- Fix in `tools/motion_experiment.py`: at most 8 workers by default; one
  BLAS/OpenMP thread per worker; if a worker still dies, the remaining solves
  run one at a time instead of aborting; `--jobs 1` runs without
  multiprocessing at all.
- Checked: a deliberately crashing worker is recovered from and gives the same
  summary as a single-process run.

## Step 2: receiver Doppler compensation, offline (2026-11-02)

### Decided with you
- GD = G + receiver Doppler compensation, done in the offline solve
  (option a): no change to `retina-geolocator`, so G and GD share the exact
  solver and differ only by the correction. Putting the receiver velocity
  into the geolocator's own Doppler model (option b) waits for the server
  step; it should then reproduce these results to ~0.2 Hz.
- Receiver trajectory known exactly here. GNSS errors and a Kalman filter that
  estimates the receiver's velocity from GNSS move to step 3.
- Receiver Doppler on only (the physical case), for every algorithm. Altitude
  by sweep, as in steps 0 and 1.

### Built
- `tools/offline_solve.py --receiver geometric_doppler`, per epoch:
  - solve as G (receiver at its true pose);
  - for each channel, `df = (fc/c)(u_RP + u_TR) . v_R` at G's solved
    position, from the receiver's true velocity (`rx_trajectory.ndjson`);
    the same term the simulator adds (`world._receiver_doppler`);
  - subtract `df` from the measured Doppler and solve again (full sweep).
  - Delay is not corrected: in one ~1 s capture the receiver moves ~15 m at
    vehicle speed, against ~150 m of delay resolution.
  - Every row now carries `rxdop_true_hz_<node>` (the receiver Doppler in the
    measurement, at the true aircraft position); GD rows add
    `rxdop_corr_hz_<node>` (what was subtracted).
  - `rx_doppler_passes` (solver setting, default 1) repeats the correction.
- `experiments/receiver_doppler.json`: `base` = `receiver_motion.json`,
  receiver Doppler on, algorithms S, G, GD. Reuses the step 1 recordings;
  only the static case gained a new fingerprint (`static|doppler-on`), and it
  is identical to the step 1 `static|doppler-off` one.
- `tools/motion_report.py`: GD everywhere (aqua), the overview as one row per
  Doppler setting, and two new figures: `velocity_vs_speed.png`,
  `doppler_correction.png`. Step 1's report rebuilds unchanged.

### Verified
- `tests/test_receiver_doppler.py` (7):
  - the correction equals the simulator's receiver Doppler term (to 1e-6);
  - static receiver: GD gives exactly G's solves, correction 0;
  - noise-free, 15 m/s east: G velocity RMSE > 3 m/s, GD < 0.5 m/s, same
    position;
  - correction within 0.2 Hz RMS of the true receiver Doppler;
  - one pass vs two: velocity differs by < 0.1 m/s;
  - deterministic.

### Results (offline, 5 seeds, receiver Doppler on; every solve counted)

| Case | S pos | G pos | GD pos | S vel | G vel | GD vel | Target: G vel, Doppler off (step 1) |
|---|---|---|---|---|---|---|---|
| static | 82 m | 82 m | 82 m | 3.56 | 3.56 | 3.56 | 3.56 |
| linear-east-walk | 469 m | 82 m | 82 m | 5.36 | 3.90 | 3.58 | 3.58 |
| linear-east-vehicle | 4,467 m | 82 m | 82 m | 31.07 | 13.17 | 3.80 | 3.80 |
| linear-north-walk | 220 m | 81 m | 81 m | 3.50 | 3.76 | 3.57 | 3.57 |
| linear-north-vehicle | 1,965 m | 83 m | 83 m | 11.12 | 11.28 | 3.63 | 3.63 |
| oscillate-walk | 91 m | 82 m | 82 m | 3.73 | 3.69 | 3.56 | 3.56 |
| oscillate-vehicle | 268 m | 82 m | 82 m | 9.35 | 9.12 | 3.57 | 3.57 |
| circle-walk | 124 m | 82 m | 82 m | 3.85 | 3.73 | 3.55 | 3.55 |
| circle-vehicle | 849 m | 81 m | 81 m | 12.63 | 11.95 | 3.57 | 3.57 |

(velocity RMSE in m/s; availability unchanged from step 1: 41%, S 39% east
vehicle and 28% north vehicle)

| Case | RMS receiver Doppler (Hz) | RMS correction error (Hz) |
|---|---|---|
| walking cases | 0.8-1.7 | 0.005-0.008 |
| linear-east-vehicle | 11.1 | 0.081 |
| linear-north-vehicle | 16.3 | 0.072 |
| oscillate-vehicle | 7.7 | 0.052 |
| circle-vehicle | 14.1 | 0.067 |

### Findings
1. **GD removes the receiver Doppler completely.** In every case GD's velocity
   RMSE equals G's with the receiver Doppler left out of the simulation
   (step 1), to 0.01 m/s, and sits at the static baseline (3.56-3.80 m/s).
   At vehicle speed that is 9-13 m/s -> 3.6-3.8 m/s.
2. **Position is untouched.** GD's position RMSE equals G's to 0.1 m: the
   sweep picks altitude by delay residual, and the corrected Doppler only
   moves the velocity.
3. **Velocity error no longer depends on receiver speed.** G's grows almost
   linearly with the receiver's speed (3.5 m/s at rest to ~12.5 m/s at
   15 m/s, `velocity_vs_speed.png`); GD stays at 3.3-4.3 m/s across the whole
   range.
4. **The one-pass correction is far more accurate than it needs to be.**
   RMS error 0.005-0.08 Hz, 25-400x below the 2 Hz Doppler noise, because
   G's ~80 m position error barely tilts the unit vectors.
5. **The FM channel's receiver Doppler is not smaller in Hz, against my
   earlier prediction.** Its ~5x longer wavelength is offset by geometry:
   the tower is north-west, so driving east (or circling) moves the receiver
   nearly straight away from it and towards the aircraft, and
   `|u_RP + u_TR|` is close to 2. Its *correction error* is ~5x smaller, as
   the wavelength predicts (same angular error, 5x fewer Hz per m/s).
6. **No Doppler-gate rejections** for any algorithm; delay-gate rejections are
   S's only, as in step 1.

### Next (not started; your go-ahead first)
- Step 3: GNSS errors on the pose given to G and GD (position and velocity
  noise, rate, latency, dropouts, lever arm), and a Kalman filter that
  estimates the receiver's position and velocity from GNSS, compared against
  raw GNSS and against the known trajectory of this step.

## Step 3a: GNSS error model (2026-11-03)

### Decided with you
- The receiver's GNSS gives position only; the Kalman filter (3b) will
  estimate velocity from it.
- Error per axis = time-correlated part + white noise:
  `p_meas(t_i) = p_true(t_i - time_offset) + nu_i + eps_i`, with `nu` a
  first-order Gauss-Markov (AR(1)) process as in the paper:
  `nu_i = a_i nu_{i-1} + w_i`, `a_i = exp(-(t_i - t_{i-1}) / T)`.
- Two corrections to the first write-up:
  - each axis draws its own `w` and `eps` (independent axes);
  - the size is set by the steady-state `sigma_corr`, and
    `w ~ N(0, sigma_corr^2 (1 - a_i^2))` is derived from it. Otherwise
    changing T or the update rate would also change how big the error is.
- Same parameters on all three axes (no vertical scale), no lever arm.
- Timing: an unknown `time_offset_s` (the fix describes an earlier moment and
  nothing corrects it) and a known `latency_s` (the fix arrives late; a filter
  can predict past it), kept separate.
- Defaults, placeholders for the Monte Carlo to vary: sigma_white 0.5 m,
  sigma_corr 1.5 m, T 60 s, 1 Hz, no offset, no latency.

### Built
- `retina_simulation/gnss.py`: `GnssModel`; `stamps` (fixes at the update rate
  from a random phase, so the GNSS clock is not aligned with the radar's
  ticks); `errors` (both error parts per axis; `nu` starts stationary,
  `nu_0 ~ N(0, sigma_corr^2)`); `fixes` (measured position, true position at
  the stamp, and both error parts, in metres east/north/up of the registered
  position); `write_fixes` (`gnss.ndjson`). Pure Python; every draw from its
  own seeded stream keyed by (key, realisation, axis), independent of the
  radar noise, so the step 2 recordings are reused unchanged.
- `tools/gnss_figures.py`: the figures below, to a timestamped folder.

### Verified
- `tests/test_gnss.py` (13): the correlated part's size is sigma_corr at
  T 5 and 20 s and at 1 and 10 Hz; autocorrelation follows exp(-lag/T);
  T = 0 is white noise; it starts stationary; white noise has its sigma;
  axes are uncorrelated; same key gives the same fixes and another
  realisation does not; stamps follow the rate from a random phase; a time
  offset reports the position `offset x velocity` earlier while latency only
  delays arrival; each fix = truth + both parts.
- Figures (defaults): measured autocorrelation at lag T 0.351 against
  1/e = 0.368; measured sigma_corr 1.47-1.56 m against 1.5 m for T 1-300 s at
  1 and 10 Hz; white sigma 0.5 m.

### What the figures show
- `gnss_autocorrelation.png`: the paper's figure, theory against simulation.
  The total error's autocorrelation is the same curve scaled by
  sigma_corr^2 / (sigma_corr^2 + sigma_white^2) = 0.90, plus a spike at lag 0
  from the white part.
- `gnss_error_time_series.png`: the error is a slow wander of a few metres
  with small white noise on top, as real GNSS is.
- `gnss_time_constant.png`: T = 1 s looks like white noise, T = 300 s like a
  drifting offset; all have the same size.
- `gnss_horizontal_error.png`: over 300 s the horizontal error drifts ~5 m
  from one side of the true position to the other; averaging over a few
  seconds would not remove it.

### Next (not started)
- 3b: Kalman filter estimating receiver position and velocity from these
  fixes, with `nu` in its model.

## Step 3b: receiver Kalman filter (2026-11-04)

### Decided with you
- GNSS gives position only; the filter estimates velocity.
- Per axis (east, north, up, same parameters), a linear Kalman filter on
  `x = [x, v, nu]`: position, velocity and the GNSS error's Gauss-Markov part.
  No EKF: GNSS position in local metres is linear in the state, and so is the
  motion model; turns are absorbed as random acceleration.
  - `F = [[1, dt, 0], [0, 1, 0], [0, 0, a]]`, `a = exp(-dt/T)`
  - `Q = [[sa^2 dt^4/4, sa^2 dt^3/2, 0], [sa^2 dt^3/2, sa^2 dt^2, 0],
    [0, 0, sigma_corr^2 (1 - a^2)]]` (white acceleration held over each step)
  - `z = H x + eps`, `H = [1 0 1]`, `R = sigma_white^2`
  - start `x0 = [z0, 0, 0]`, `P0 = [[sigma_corr^2 + sigma_white^2, 0,
    -sigma_corr^2], [0, sigma_v0^2, 0], [-sigma_corr^2, 0, sigma_corr^2]]`:
    starting at the fix makes the position error -(nu + eps), hence the
    off-diagonal term
  - Joseph-form update at every later fix; at each radar tick, predict only,
    from the last fix that has arrived (known latency respected).
- Fixed platform modes: sigma_a 0.1 m/s^2 and sigma_v0 2 m/s for static and
  walking, 0.5 m/s^2 and 20 m/s for vehicles. Adaptive variants (mode
  scheduling, zero-velocity updates, innovation-based Q) later.
- The filter's GNSS parameters match the model's (sigma_white 0.5 m,
  sigma_corr 1.5 m, T 60 s, 1 Hz, no offset or latency).

### Built
- `tools/rx_filter.py`: `AxisFilter`, `run` (estimates at query times, causal),
  `nees`; variants `kf`, `white` (no drift state, R = sigma_white^2 +
  sigma_corr^2) and `raw` (latest fix, velocity by differencing, extrapolated).
  Lives with the solver tools (numpy); the GNSS model stays in the simulator.
- `experiments/receiver_filter.json`, `tools/rx_filter_report.py`: every
  step 1 motion case x 5 seeds x 4 GNSS realisations = 20 runs per case;
  scored from 30 s on.

### Verified
- `tests/test_rx_filter.py` (6). The decisive one: on truth that obeys the
  filter's own model (white acceleration of its sigma_a, the 3a GNSS error),
  **mean NEES 6.07 for 6 dof**: the filter's stated uncertainty matches its
  error. The white-noise filter scores 11.5 (overconfident ~2x) and has higher
  velocity error (0.89 vs 0.75 m/s). Also: P0 equals a normal first update
  from a diffuse start; converges on a noise-free straight line; uses no fix
  before it arrives; raw differencing; prediction to the query time.
- With nearly noise-free fixes and sigma_a 0.5 the velocity error is ~0.08 m/s
  (filter's own sigma 0.10): sigma_a sets how little it smooths.

### Results (20 runs per case, from 30 s; GNSS fixes themselves 2.01 m RMS horizontal)

| Case | KF pos | KF vel | White-filter vel | Raw vel | KF NEES (6 = honest) | White NEES |
|---|---|---|---|---|---|---|
| static, linear walk (both directions) | 1.99 m | 0.17 m/s | 0.12 | 1.07 | 3.6 | 13.7 |
| linear vehicle (both directions) | 2.07 m | 0.45 | 0.25 | 1.07 | 3.4 | 6.5 |
| oscillate-walk | 2.03 m | 0.33 | 0.40 | 1.07 | 5.6 | 16.3 |
| oscillate-vehicle | 2.26 m | 1.21 | 1.77 | 1.29 | 6.7 | 11.4 |
| circle-walk | 1.99 m | 0.22 | 0.22 | 1.07 | 4.1 | 14.2 |
| circle-vehicle | 2.09 m | 0.70 | 0.86 | 1.12 | 4.2 | 7.6 |

(horizontal RMSE; vertical: KF 1.6 m, 0.12-0.32 m/s. Reference: 0.58 m/s of
receiver velocity error shifts a 521 MHz channel's Doppler by the radar's own
2 Hz noise in the worst geometry.)

### Findings
1. **Position cannot beat the GNSS drift.** Every variant sits at the fixes'
   own ~2 m: the slow drift goes straight into the position estimate, as
   predicted. For G this costs ~2-4 m of aircraft error, small next to 80 m.
2. **Velocity is where filtering pays**: 1.07 m/s raw -> 0.17 m/s (walking) and
   0.45 m/s (straight driving), under the 0.58 m/s reference.
3. **Turns cost lag.** sigma_a 0.5 is below oscillate-vehicle's 0.94 m/s^2
   peaks: its velocity error is a +/-2 m/s sine (1.21 m/s RMS), barely better
   than raw, and the filter is overconfident at each turnaround (NEES ~10).
   circle-vehicle (0.31 m/s^2, constant) lags less (0.70 m/s). Motion model,
   not GNSS, limits these cases.
4. **The drift state makes the uncertainty honest.** Treating the GNSS error as
   white is overconfident by ~2x everywhere (NEES 6.5-16); with the drift
   state the filter is consistent on truth that matches its model, and on our
   straight cases slightly cautious (NEES 3.4-3.6), because their true
   acceleration is zero, below the assumed sigma_a.
5. **The white filter's lower velocity error on straight lines is an
   artefact of over-smoothing**: its large R smooths harder, which helps when
   nothing accelerates and hurts in every turn (oscillate 1.77, circle 0.86
   m/s). It also claims accuracy it does not have.
6. Linear and static cases give identical numbers per speed class: the same
   GNSS draws through a linear filter on linear motion. A check, not a
   coincidence.

### Figures redrawn (2026-11-04)
- The first figures (bar chart per case, error-over-time, mean NEES over time)
  were hard to read. Replaced by one question per figure: `accuracy.png`,
  `velocity_tracking.png` (true vs estimated velocity, zoomed),
  `honesty.png` (claimed vs actual error, and % of time the truth is inside
  the filter's +/-2 sigma), `error_vs_acceleration.png`. NEES/NIS stay in
  `summary.csv`.
- What they add:
  - **The white filter's overconfidence is in position, not velocity.** Its
    position +/-2 sigma holds the truth only 77-96% of the time (KF 96-98%);
    its velocity claims are cautious (100% on straight paths) because its large
    R makes it smooth hard. Treating the drift as white noise makes it believe
    it has averaged the drift out of the position.
  - **Velocity error grows steadily with acceleration** (vehicle cases: KF
    0.45 m/s at rest, ~1.0 at 0.5 m/s^2, ~1.6 at 0.9 m/s^2), and crosses the
    0.58 m/s GD tolerance at ~0.15 m/s^2, well below sigma_a: the lag is
    proportional to acceleration, not a threshold effect. Raw differencing
    barely lags, so above ~0.75 m/s^2 it is no worse than the filter.
  - Both filters are overconfident in velocity only when oscillating (81-90%
    inside), the case whose acceleration exceeds sigma_a.

### Next (not started)
- 3c: feed the KF pose (and raw GNSS) to G and GD on the step 2 recordings.
- Worth trying for the manoeuvring cases: sigma_a ~1 m/s^2 for vehicles, a
  Singer (Gauss-Markov acceleration) or constant-acceleration model, or the
  adaptive modes discussed.

### Figures simplified again (2026-11-04)
- At your request, two figures only: `case_<case>.png` (one run per case,
  position error and velocity per axis, all methods, both filters' +/-2
  sigma) and `rmse.png` (3D RMSE per case and over all cases).
- 3D RMSE over all cases (20 runs each, from 30 s): position KF-drift 2.61 m,
  KF-white 2.71 m, raw 2.89 m; velocity 0.59, 0.70, 1.35 m/s.
- KF-white is better only for velocity on unaccelerated motion (static and
  straight: 0.15 vs 0.21 m/s walking, 0.30 vs 0.56 m/s driving); worse in
  every accelerating case (circle drive 0.88 vs 0.77, oscillate drive 1.78
  vs 1.26 m/s, position 3.48 vs 2.79 m) and overall.

### Final figures for 3b (2026-11-04)
- `rmse.png` (3D RMSE, three methods, every case and all cases) and
  `coverage.png` (share of time the truth is inside each filter's +/-2 sigma).
- Coverage, all cases: KF with drift 97% position / 98% velocity; white-noise
  KF 85% / 97%. The white KF is overconfident in position in every case
  (76-95%); both are overconfident in velocity only when oscillating
  (drift KF 93% / 87%, white 90% / 82%), the case whose acceleration exceeds
  sigma_a.
- Decision: proceed with the KF with the GNSS drift state.

## Step 3c: GD with the receiver pose from GNSS (2026-11-12)

### Decided with you
- GD only (the algorithm we will implement), to isolate the effect of how the
  receiver trajectory is obtained.
- Three receiver-trajectory sources: the truth (step 2), raw GNSS fixes
  (velocity by differencing consecutive fixes), and the drift-state Kalman
  filter from 3b. The white-noise KF is left out.
- GNSS at the 3a defaults; 4 realisations per seed, same keys as 3b, so 3b and
  3c see identical GNSS errors.

### Built
- `offline_solve.solve_run(..., pose_estimate=...)`: replaces the true receiver
  pose given to G/GD; the true-pose columns (offset, bias, true receiver
  Doppler) are kept, and each row adds `rx_pos_err_m` / `rx_vel_err_ms`.
- `tools/gnss_pose.py`: per radar tick, the GNSS-based pose in the
  rx_trajectory format (east/north/up offset to lat/lon/alt on WGS84, velocity).
- `experiments/receiver_gnss.json` (base `receiver_filter.json`),
  `tools/gnss_experiment.py`, `tools/gnss_report.py`: 9 cases x 5 seeds, truth
  once and gnss / kf x 4 realisations: 405 GD solves (~39 min on 4 cores).
  Recordings fingerprint-checked against step 2.

### Verified
- `tests/test_gnss_pose.py` (6): near-noiseless GNSS gives poses on the true
  trajectory; pose positions are the offsets on the ellipsoid; identical GNSS
  draws to 3b; realisations differ and repeat; `truth` adds no estimate; a
  truth override reproduces step 2's GD solves exactly, and a KF override
  reports a plausible receiver velocity error.

### Results (from 30 s; RMSE pooled over 5 runs for truth, 20 for gnss and kf)

| Case | Aircraft pos (truth / gnss / kf), m | Aircraft vel (truth / gnss / kf), m/s | Receiver vel error (gnss / kf), m/s | GD correction error (truth / gnss / kf), Hz |
|---|---|---|---|---|
| static | 81.3 / 81.3 / 81.2 | 3.58 / 3.67 / 3.58 | 1.07 / 0.17 | 0.00 / 1.04 / 0.16 |
| linear-east-walk | 81.0 / 81.6 / 81.5 | 3.60 / 3.69 / 3.60 | 1.07 / 0.17 | 0.01 / 1.04 / 0.16 |
| linear-east-vehicle | 82.9 / 83.8 / 84.1 | 3.85 / 3.95 / 3.88 | 1.07 / 0.45 | 0.10 / 1.09 / 0.47 |
| linear-north-walk | 81.1 / 81.4 / 81.4 | 3.58 / 3.68 / 3.59 | 1.07 / 0.17 | 0.01 / 1.03 / 0.16 |
| linear-north-vehicle | 83.3 / 83.0 / 83.0 | 3.66 / 3.74 / 3.67 | 1.07 / 0.45 | 0.08 / 1.00 / 0.43 |
| oscillate-walk | 81.4 / 81.6 / 81.5 | 3.58 / 3.67 / 3.59 | 1.07 / 0.33 | 0.01 / 1.04 / 0.26 |
| oscillate-vehicle | 81.7 / 81.7 / 81.4 | 3.58 / 3.72 / 3.72 | 1.29 / 1.22 | 0.06 / 1.16 / 0.93 |
| circle-walk | 81.6 / 81.6 / 81.7 | 3.57 / 3.67 / 3.58 | 1.07 / 0.22 | 0.01 / 1.04 / 0.21 |
| circle-vehicle | 81.0 / 81.1 / 81.1 | 3.58 / 3.69 / 3.64 | 1.12 / 0.70 | 0.08 / 1.09 / 0.68 |
| **all cases** | **81.7 / 81.9 / 81.9** | **3.62 / 3.72 / 3.65** | **1.10 / 0.54** | **0.05 / 1.06 / 0.46** |

(receiver position error: gnss 2.29 m, kf 2.07 m RMS horizontal)

### Findings
1. **GNSS-based trajectories cost GD almost nothing.** Over all cases the KF
   adds 0.2 m of aircraft position RMSE (81.7 -> 81.9 m) and 0.03 m/s of
   velocity RMSE (3.62 -> 3.65 m/s); raw GNSS adds 0.2 m and 0.10 m/s. Both
   differences are far smaller than the run-to-run spread (whiskers).
2. **Position:** a ~2 m receiver position error is invisible against the
   ~80 m aircraft error, as predicted (about 0.8-1.9 m of aircraft error per
   metre of receiver error, added in quadrature to 80 m).
3. **Velocity:** the KF matches the truth to within 0.01-0.02 m/s wherever its
   receiver velocity error is below ~0.5 m/s (static, walking, straight
   driving). Raw GNSS is consistently ~0.1 m/s worse (its 1.07 m/s receiver
   velocity error gives a ~1 Hz Doppler correction error).
4. **Only the turning vehicle cases show the KF's lag:** oscillate-vehicle
   (receiver velocity error 1.22 m/s, the same as raw) and circle-vehicle
   (0.70 m/s), +0.14 and +0.06 m/s of aircraft velocity error.
5. **GD tolerates much more receiver velocity error than the 0.58 m/s
   estimate.** Even ~1.2 m/s (about 1 Hz of correction error, half the 2 Hz
   measurement noise) adds only ~4% to the aircraft velocity RMSE: the
   correction error adds in quadrature to the measurement noise, and the
   0.58 m/s figure assumed the worst geometry on every channel at once.

### Measured velocity tolerance (`velocity_tolerance.png`)
- Every GNSS- and KF-based solve after 30 s, binned by the receiver velocity
  error GD was given (0.1 m/s bins, at least 100 solves each).
- GD's Doppler correction error grows linearly, about 0.95 Hz per m/s of
  receiver velocity error, and reaches the 2 Hz measurement noise at about
  2.1 m/s.
- Aircraft velocity RMSE stays at the truth-fed level (3.6-3.9 m/s against
  3.62) up to about 1.7 m/s of receiver velocity error, rises to about
  4.0-4.3 m/s at 1.8-2.1 m/s and is noisy beyond (few solves).
- So the practical tolerance is about 1.5 m/s of receiver velocity error, not
  the 0.58 m/s worst-case estimate. The KF stays below that in every case
  (worst bin about 1.5 m/s, in the turning vehicle cases).
- Caveat: bins mix motion cases with different geometry; the largest receiver
  errors come almost entirely from raw GNSS and the turning vehicle cases. The
  lowest raw-GNSS bins (2.9-3.2 m/s) are few solves and partly that mix.

### Next (not started)
- 3d: Monte Carlo over the GNSS parameters, with ranges wide enough to find
  where GD starts to degrade.
