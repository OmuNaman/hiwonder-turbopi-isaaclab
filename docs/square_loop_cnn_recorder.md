# Square Loop Environment and Simple CNN Recorder

This standalone repo includes a square-loop arena plus a simple autonomous recorder aimed at the legacy no-text CNN path-following pipeline.

The current recommended square-loop workflow is:

- inspect the arena with `scripts/view_turbopi_square.py`
- collect episodes with `scripts/record_turbopi_square_simple.py`
- train with `cnn_policy.train`
- run the learned policy with `scripts/drive_turbopi_square_cnn.py --control_mode kinematic`

## Environment

The square environment is intentionally plain:

- a dark floor for contrast
- a white taped square path on the floor
- outer boundary walls
- the TurboPi spawned at a consistent pose on the left side of the square

The square route is a floor-marked loop rather than an inner-walled corridor. That keeps the geometry easy to inspect and avoids unnecessary contact problems while still producing clean path-following episodes.

Default square geometry:

- taped route corners: `(-0.45, -0.45) -> (0.45, -0.45) -> (0.45, 0.45) -> (-0.45, 0.45)`
- full-square start pose: `(-0.45, 0.0, 0.04)`
- `square_ccw` start yaw: `-pi/2`
- `square_cw` start yaw: `+pi/2`
- floor half-extent: `1.40 m`

## Scripts

### View the square environment

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/view_turbopi_square.py --livestream 2
```

Useful alternate view:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/view_turbopi_square.py --view robot --direction counterclockwise --livestream 2
```

### Record autonomous episodes

The simple recorder supports:

- `--route segment`
- `--route square_ccw`
- `--route square_cw`

Record full counterclockwise laps:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/record_turbopi_square_simple.py --headless --route square_ccw --num_episodes 8
```

Fast bulk generation:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/record_turbopi_square_simple.py --headless --route square_ccw --num_episodes 50 --target_speed 0.30 --lookahead_distance 0.16 --corner_slowdown_distance 0.30 --square_max_wz 1.25 --max_episode_time 25
```

Livestream inspection:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/record_turbopi_square_simple.py --livestream 2 --view chase --route square_ccw --num_episodes 10
```

One point-to-point segment per episode:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/record_turbopi_square_simple.py --headless --route segment --start bl --goal br --num_episodes 8
```

For full-square routes, the recorder automatically uses a more efficient simulation step in headless mode while preserving the saved control rate. That improves throughput without changing the dataset layout.

## Output Structure

By default, episodes are written under:

```text
turbopi_standalone/data/cnn_square_loop/session_simple_<timestamp>/
```

Each saved episode is directly compatible with the legacy CNN loader layout:

```text
session_simple_YYYYMMDD_HHMMSS/
├── collection_summary.json
├── session_info.json
├── task_mapping.json
├── episode_00000/
│   ├── video.mp4
│   ├── data.parquet
│   └── episode_info.json
├── episode_00001/
│   ├── video.mp4
│   ├── data.parquet
│   └── episode_info.json
└── ...
```

## Parquet Columns

The important compatibility fields are:

- `task`
- `task_index`
- `direction`
- `state`
- `action`

Additional debugging columns are also saved:

- `frame_index`
- `timestamp`
- `command`
- `body_velocity`
- `track_error`
- `lap_progress`

`state` stores the previous normalized action. `action` stores the current normalized action issued by the autonomous teacher.

`episode_info.json` also records summary stats such as:

- `mean_track_error`
- `p90_track_error`
- `max_track_error`
- `mean_abs_action_vx`
- `mean_abs_action_vy`
- `mean_abs_action_wz`
- `mean_action_vy_vx_ratio`

## Recorder Behavior

The simple recorder is built to avoid the earlier startup and timeline deadlocks:

- it enables cameras explicitly
- it calls `sim.play()` after reset
- it resumes automatically if the timeline becomes paused
- it warms up the camera after reset before recording frame 0
- it never waits for keyboard focus
- it has explicit timeout and stuck detection
- it stops the robot cleanly on interrupt

The full-square teacher is intentionally simple and inspectable:

- it follows known square geometry
- it keeps `vy` near zero
- it uses forward speed plus yaw to take corners
- it records one full lap as one episode

## Success and Failure Rules

For `square_ccw` and `square_cw`, lap completion is geometry-based, not time-based.

At each control step, the recorder projects the robot root `x,y` position onto the nearest segment of the square and converts that to a normalized `track_phase`:

- left edge: `0.00 -> 0.25`
- top edge: `0.25 -> 0.50`
- right edge: `0.50 -> 0.75`
- bottom edge: `0.75 -> 1.00`

It then accumulates signed forward progress:

- `square_cw` counts positive phase advance
- `square_ccw` flips the sign and counts the same forward motion
- wrap-around across the start line is handled explicitly

With the default settings, a full-square rollout is accepted when:

- `lap_progress >= 0.97`
- the robot returns near the start phase
- the robot returns near the start heading
- commanded forward distance covers at least `min_square_distance_ratio` of the square perimeter

With the default settings, a rollout is rejected when:

- `track_error > 0.25 m`
- no useful forward progress for `8.0 s`
- camera warmup stays too flat
- the app closes or the state becomes invalid
- total rollout time exceeds `45.0 s`

Unlike the older dynamic square recorder, the simple recorder does not apply a second layer of post-lap quality gates before saving. It saves every rollout that actually satisfies the route-completion and safety checks above, while still writing track-error and action summary metrics into the episode metadata for later inspection.

## Dependency Note

Direct `data.parquet` export needs a parquet engine in the Isaac Lab Python environment:

```bash
/workspace/isaaclab/_isaac_sim/python.sh -m pip install pyarrow
```

The legacy training loader also expects PyAV when reading `video.mp4`:

```bash
/workspace/isaaclab/_isaac_sim/python.sh -m pip install av
```

## Known Limitations

- The simple square recorder uses a kinematic teacher, not full wheel-ground dynamics.
- The dataset is directly compatible with the legacy CNN loader, but deployment caps still need to be chosen in the inference script.
- The current default inference mode is also kinematic so training and deployment match.
- The older `scripts/record_turbopi_square_cnn.py` still exists for comparison, but it is no longer the recommended square-loop collection path for fresh clones.
