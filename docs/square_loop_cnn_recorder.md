# Square Loop Environment and Automatic CNN Recorder

This standalone repo now includes a square-loop arena and an autonomous recorder aimed at the legacy no-text CNN path-following pipeline.

## Environment

The environment is intentionally simple:

- a dark floor for contrast
- a white taped square path on the floor
- boundary walls around the outside of the arena
- the TurboPi spawned on the left side of the square in a consistent start pose

The square route is a visual floor-marked loop rather than a corridor with inner walls. That choice keeps the geometry easy to inspect, avoids unnecessary contact problems, and makes the automatic teacher more reliable.

The teacher is tuned to behave like a forward-facing rover, not a sideways-sliding mecanum demo:

- it aligns the robot heading with the local path tangent
- it slows forward speed when heading error is large
- it uses `wz` aggressively at corners
- it keeps `vy` near zero except for tiny corrective nudges

## Scripts

### View the square environment

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/view_turbopi_square.py --livestream 2
```

Useful options:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/view_turbopi_square.py --view robot --direction counterclockwise --livestream 2
```

### Record autonomous episodes

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/record_turbopi_square_cnn.py --headless --num_episodes 8
```

For bulk generation, prefer headless mode. In headless runs the recorder now auto-selects a render interval that matches the control cadence instead of rendering every physics step, which improves wall-clock throughput without changing the saved episode structure. Actual minute-scale throughput still depends on the machine and is unlikely with the current single-robot recorder; a batched multi-environment recorder is the real path for very large fast runs.

Example bulk command:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/record_turbopi_square_cnn.py --headless --directions counterclockwise --num_episodes 50 --target_speed 0.24 --control_hz 8
```

Example with alternating directions and livestream inspection:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/record_turbopi_square_cnn.py --livestream 2 --view chase --directions clockwise,counterclockwise --num_episodes 6
```

By default the recorder stays in `clockwise` only. Use `--directions clockwise,counterclockwise` when you want a mixed-direction dataset instead of a single-direction inspection run.

## Output Structure

By default, episodes are written under:

```text
turbopi_standalone/data/cnn_square_loop/<session_name>/
```

Each saved episode is directly compatible with the legacy CNN loader layout:

```text
session_YYYYMMDD_HHMMSS/
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

`episode_info.json` also records action-style summary stats for each accepted lap:

- `mean_abs_action_vx`
- `mean_abs_action_vy`
- `mean_abs_action_wz`
- `mean_action_vy_vx_ratio`

## Recorder Behavior

The recorder is built to avoid the startup deadlocks from the teleop workflow:

- it enables cameras explicitly
- it calls `sim.play()` after reset
- it resumes automatically if the timeline becomes paused
- it warms up the camera after reset before recording frame 0
- it refuses to save episodes if the RGB stream is too flat or washed out
- it never waits for keyboard focus
- it has per-episode timeout and stuck detection
- it exits after a bounded number of failed attempts
- it stops the robot cleanly on interrupt

## Success and Failure Rules

An episode is accepted when:

- signed lap progress reaches the completion threshold
- the rollout has nontrivial motion
- the robot stays within the allowed track error

A rollout is rejected if it:

- times out
- goes off track
- stops making lap progress
- falls into an invalid physics state
- moves too slowly overall

Failed attempts are not saved as dataset episodes, but the session summary keeps count.

## How Lap Completion Is Measured

Lap completion is geometry-based, not time-based.

At each control step, the recorder projects the robot root `x,y` position onto the nearest segment of the square centerline and converts that to a normalized `track_phase`:

- left edge: `0.00 -> 0.25`
- top edge: `0.25 -> 0.50`
- right edge: `0.50 -> 0.75`
- bottom edge: `0.75 -> 1.00`

The recorder then accumulates signed forward progress:

- clockwise episodes count positive phase advance
- counterclockwise episodes count the same progress with the sign flipped
- wrap-around across the start/finish boundary is handled explicitly
- backward motion does not increase the lap counter

Direction is not inferred from the motion after the fact. It is chosen up front from the `--directions` argument and then used consistently for:

- the start yaw
- the tangent direction used by the teacher
- the signed lap-progress accumulation
- the expected segment order for the completion check

With the default settings, a rollout is accepted once `lap_progress >= 0.97`.

It must also return close to the start phase and close to the start heading before the lap can close cleanly.

With the default settings, a rollout is rejected when any of these triggers fire first:

- `track_error > 0.30 m`
- no forward lap progress for `5.0 s` after an initial `3.0 s` grace period
- invalid state or NaN physics data
- invalid or effectively blank camera frames during warmup
- total rollout time exceeds `45.0 s`
- mean planar body speed is below `0.03 m/s`

That rule is the reason the recorder can stop cleanly without a human pressing a key: it knows where the robot is on the square, how far around the loop it has progressed, and whether it is still making useful forward progress.

## Quality Gates

A rollout that technically finishes the loop is still rejected unless it meets the default dataset-quality thresholds:

- `mean_track_error <= 0.06 m`
- `p90_track_error <= 0.12 m`
- `max_track_error <= 0.18 m`
- `frames_over_0.15m_ratio <= 0.05`
- `mean_image_std >= 8.0`

The recorder also requires the robot to complete the square segment cycle in order and return to the start-phase neighborhood before the lap can be accepted. That makes corner-cutting much less likely to slip through as a fake “complete lap.”

The recorder also rejects laps that rely too much on strafing:

- `mean |vy| / mean |vx| <= 0.20` by default

That gate is there specifically to protect the CNN dataset from crab-walking episodes where the robot slides along the line instead of facing the direction of travel.

## Dependency Note

Direct `data.parquet` export needs a parquet engine in the Isaac Lab Python environment. If it is missing, install:

```bash
/workspace/isaaclab/_isaac_sim/python.sh -m pip install pyarrow
```

The legacy training loader also expects PyAV when reading `video.mp4`:

```bash
/workspace/isaaclab/_isaac_sim/python.sh -m pip install av
```

## Known Limitations

- The autonomous teacher follows known square geometry, not pixels, so the dataset is clean but still simulator-generated.
- The saved action normalization matches the CNN action space, but deployment caps still need to be chosen in the downstream drive script.
- Episodes are saved only after success, so there is no raw failed-rollout archive in this standalone repo.
- Omega feedback is enabled by default in the autonomous recorder so turns track the commanded yaw rate more faithfully. Use `--disable_omega_feedback` only when you are debugging the low-level drive mapping.
- Older sessions recorded before the camera warmup and RGB-quality checks may contain washed-out `video.mp4` files and should be treated with caution.
