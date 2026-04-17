# Hiwonder TurboPi for Isaac Lab

This repository packages the Hiwonder TurboPi robot for Isaac Lab.

Using this repo, you can:

- load the TurboPi into Isaac Lab
- inspect the robot in a clean standalone scene
- inspect the robot in a square-loop collection arena
- switch to the robot-mounted camera
- teleoperate the robot from Isaac Lab
- record simple autonomous square-loop episodes in the legacy no-text CNN dataset format
- train the no-text CNN directly inside this repo
- run the trained CNN with inference control that matches the recorder
- run the same setup on a local machine or on Isaac Lab running on Brev

The repo includes a ready-to-load USD, the original URDF and meshes, a standalone viewer, and a keyboard teleop script so you can bring the robot up quickly without digging through the RL or dataset code.

## Workspace Layout

The commands in this README assume this layout:

```text
/workspace
├── isaaclab/
└── turbopi_standalone/
```

In other words:

- Isaac Lab is checked out at `/workspace/isaaclab`
- this TurboPi repo is checked out at `/workspace/turbopi_standalone`

If you want to record datasets and train the CNN in a fresh clone, install the two Python extras in the Isaac Lab environment once:

```bash
/workspace/isaaclab/_isaac_sim/python.sh -m pip install pyarrow av
```

## Quick Start

These quick-start commands are Brev-focused. They launch Isaac Lab with livestream enabled so you can see and use the robot remotely.

From `/workspace`, run:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/view_turbopi.py --livestream 2
```

That launches the TurboPi in a standalone Isaac Lab scene with a clean overview camera.

For keyboard teleoperation:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/teleop_turbopi.py --livestream 2
```

Once the stream is visible, click inside the Isaac viewport once so the browser gives keyboard focus to the session.

To start directly from the robot-mounted camera:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/view_turbopi.py --view robot --livestream 2
```

## Repository Layout

```text
turbopi_standalone/
├── README.md
├── assets/
│   └── turbopi/
│       ├── turbopi.usd
│       ├── configuration/
│       └── turbopi_description/
│           ├── meshes/
│           └── urdf/
└── scripts/
    ├── common.py
    ├── cnn_dataset.py
    ├── drive_turbopi_square_cnn.py
    ├── mecanum_builder.py
    ├── record_turbopi_square_simple.py
    ├── record_turbopi_straight_cnn.py
    ├── square_loop.py
    ├── teleop_turbopi.py
    ├── view_turbopi.py
    └── view_turbopi_square.py
```

## Main Commands

These commands use the TurboPi repo from `/workspace/turbopi_standalone` while launching Isaac Lab from `/workspace/isaaclab`.

View the robot in a plain standalone scene:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/view_turbopi.py
```

Drive the robot with the keyboard:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/teleop_turbopi.py
```

Use the robot-mounted camera:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/teleop_turbopi.py --view robot
```

Use the external chase camera:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/teleop_turbopi.py --view chase
```

Override the USD if you are iterating on another asset file:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/view_turbopi.py --asset_usd /absolute/path/to/turbopi.usd
```

Run for a fixed smoke-test duration:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/view_turbopi.py --duration 3
```

View the square-loop collection arena:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/view_turbopi_square.py --livestream 2
```

Record simple autonomous square-loop CNN episodes:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/record_turbopi_square_simple.py --headless --route square_ccw --num_episodes 8
```

Run a trained no-text CNN checkpoint in the same square arena:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/drive_turbopi_square_cnn.py --livestream 2 --view chase --direction counterclockwise --control_mode kinematic --checkpoint /workspace/turbopi_standalone/runs/cnn_square_simple/<run_name>/checkpoints/best.pt
```

Record a teleoperated square trace you can replay exactly later:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/teleop_trace_turbopi_square.py --livestream 2
```

Replay that trace exactly by recorded poses:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/replay_turbopi_square_trace.py --trace_dir /workspace/turbopi_standalone/data/teleop_traces/<trace_name> --mode pose --livestream 2
```

Run a quick headless physics diagnostic for forward, strafe, and yaw:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/debug_turbopi_physics.py --headless --scene square
```

Measure the same diagnostics with yaw-rate compensation enabled:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/debug_turbopi_physics.py --headless --scene square --omega_feedback
```

You can also run the diagnostics in the straight-line arena (useful to confirm that strafe is no longer sign-flipped and yaw tracks the command without overshoot):

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/debug_turbopi_physics.py --headless --scene straight --direction forward
```

View the straight-line arena in a livestream:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/view_turbopi_straight.py --livestream 2
```

Record autonomous straight-line (start to end) episodes:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/record_turbopi_straight_cnn.py --headless --num_episodes 8
```

## Teleop Controls

The teleop script uses Isaac Lab's built-in `Se2Keyboard`.

- `Up`: forward
- `Down`: backward
- `Left`: strafe left (body-left, ROS `+vy`)
- `Right`: strafe right (body-right, ROS `-vy`)
- `Z`: yaw counterclockwise (ROS `+wz`)
- `X`: yaw clockwise (ROS `-wz`)
- `L`: zero the current command
- `R`: reset the robot to its start pose
- `C`: cycle `overview -> chase -> robot`
- `Esc`: close the app

The arrow-key directions now match the standard ROS convention (`+vy = body-left`, `+wz = CCW`). Older versions of this repo shipped with mirrored rollers that inverted strafe, so keyboard labels previously read the opposite way. See [Physics Fixes](#physics-fixes) for details.

## Camera Behavior

There are three useful camera modes in this repo:

1. `overview`
   A normal static perspective view that is handy for inspecting the robot and the scene.

2. `robot`
   A camera prim attached under `camera_link`, with the same optical offset and orientation used by the TurboPi square-loop work. This is the view to use when you want the robot's onboard perspective.

3. `chase`
   A follow camera that stays behind the robot while you drive it around.

The viewer script can start in any of these modes through `--view`, and the teleop script can cycle between them live with `C`.

## Assets

The robot assets are committed directly in this repo so the package stays self-contained:

- USD: [`assets/turbopi/turbopi.usd`](assets/turbopi/turbopi.usd)
- URDF: [`assets/turbopi/turbopi_description/urdf/turbopi.urdf`](assets/turbopi/turbopi_description/urdf/turbopi.urdf)
- Meshes: [`assets/turbopi/turbopi_description/meshes`](assets/turbopi/turbopi_description/meshes)

That means you do not need the older `/workspace/turbopi` path for this repository to work.

## Implementation Notes

The standalone scripts intentionally keep the scope small:

- they spawn a single TurboPi articulation
- they add the procedural mecanum rollers before `sim.reset()`
- they hold the small arm mast in its default pose so the mounted camera stays usable
- they convert `(vx, vy, wz)` body commands into wheel velocities with the same mecanum mapping used in the RL environment

The repo stays intentionally small, but it now covers the whole practical standalone workflow:

- plain viewer and keyboard teleop
- simple square-loop and straight-line dataset recording
- local CNN training
- CNN inference inside the same square arena

The current recommended square-loop path is `record_turbopi_square_simple.py` for data collection plus `drive_turbopi_square_cnn.py --control_mode kinematic` for deployment.

## Physics Fixes

An earlier revision had two problems that only showed up once you stopped driving straight:

1. **Strafe direction was inverted.** Commanded `+vy` produced motion in the `-vy` body direction (verified in `data/physics_diag/current_square.json`: `target_vy=+0.15`, `mean_vy=-0.141`, `vy_ratio=-0.94`). The cause was a mirrored roller X-pattern in `scripts/mecanum_builder.py` (LF `+45`, RF `-45`, LB `-45`, RB `+45`) relative to the standard ROS mecanum IK in `scripts/common.py`. The roller angles are now flipped so `+vy` moves the robot left, matching ROS convention.
2. **Yaw tracking was weak and overshot.** Commanded `wz=+0.80` produced `mean_wz=+1.68` (a ~2× overshoot) while individual wheels barely tracked their targets. Two changes address this:
   - The procedural rollers now use 16 capsules per wheel at radius 5 mm with center distance 28 mm. The older 12-capsule, 3.5 mm setup covered less than half of each wheel's rim, so the wheels effectively bumped along without consistent ground contact.
   - Wheel actuator damping raised from `5.0` to `20.0` in `build_turbopi_cfg`, so the implicit velocity actuator tracks its commanded wheel speeds within the effort limit instead of taking the whole episode to converge.
   - Roller friction is `static=1.0, dynamic=0.85` and combines via `multiply`, matching the floor material so wheel-ground interaction is predictable.

These are enough to get clean open-loop strafe and yaw. The `OmegaTracker` closed-loop compensator is still available on the teleop path and on the legacy dynamic recorder when you want it, but the current simple square recorder and default CNN inference path both use the more predictable kinematic square flow described below.

## Straight-Line Dataset Flow

The straight-line flow defines **one episode as one traversal** of a taped lane from the start end to the finish end. It is the simpler alternative to the square loop: no corners, no lap accounting, just "drive from A to B without wandering off the line."

Default geometry (see `scripts/straight_line.py`):

- centerline tape from `(-1.00, 0.0)` to `(+1.00, 0.0)` meters
- lane half-width: `0.40` meters (side walls at `y = +/-0.44`)
- floor extends `0.30` meters past each end
- forward runs start at `(-1.00, 0.0, 0.04)` with yaw `0`
- reverse runs start at `(+1.00, 0.0, 0.04)` with yaw `pi`

### View the straight-line arena

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/view_turbopi_straight.py --livestream 2
```

### Record autonomous straight-line episodes

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/record_turbopi_straight_cnn.py --headless --num_episodes 8
```

Alternate forward and reverse while watching in livestream mode:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/record_turbopi_straight_cnn.py --livestream 2 --view chase --directions forward,reverse --num_episodes 6
```

Episodes are written by default to:

```text
/workspace/turbopi_standalone/data/cnn_straight_line/<session_name>/
```

Each saved episode matches the same folder layout as the square-loop flow (`video.mp4`, `data.parquet`, `episode_info.json`), and the session-level metadata (`session_info.json`, `task_mapping.json`, `collection_summary.json`) uses `track_layout = straight_tape_line`, `episode_definition = one_full_autonomous_traversal`, and task names `forward` / `reverse`.

### How a run is judged complete

On every control tick the recorder computes signed progress along the intended travel direction (signed so reverse runs count too) and tracks the unsigned lateral offset from the centerline.

A rollout is **accepted** when:

- the robot is within `finish_tolerance` meters of the finish end (default `0.08 m`)
- the heading is within `finish_yaw_tolerance` of the goal yaw (default `0.35 rad`)
- the mean planar speed over the run is at least `min_body_speed`
- the track-error quality gates (mean/p90/peak) pass the same way as the square flow

A rollout is **rejected** if:

- `track_error > max_track_error` (default `0.25 m`) — the robot wandered off the lane
- no forward progress for `stuck_timeout_s` seconds after the grace period
- the physics state becomes invalid
- it times out at `episode_time_s`
- the RGB stream is effectively flat
- the mean absolute lateral command is too large compared to forward command (`mean_action_vy_vx_ratio > 0.20`), to avoid saving crab-walking runs

The field `lap_progress` in the saved parquet is the normalized progress in `[0, 1]`, matching the field name already used by the square-loop episodes, so downstream training code doesn't need special handling.

## Square-Loop Dataset Flow

The current recommended square-loop pipeline is:

- [`scripts/view_turbopi_square.py`](scripts/view_turbopi_square.py) to inspect the arena
- [`scripts/record_turbopi_square_simple.py`](scripts/record_turbopi_square_simple.py) to generate episodes
- [`scripts/drive_turbopi_square_cnn.py`](scripts/drive_turbopi_square_cnn.py) to run the trained model

The environment is intentionally plain:

- a dark floor
- a white taped square route
- outer arena walls
- the TurboPi starting from a consistent pose on the left side of the square

The simple recorder supports three routes:

- `--route segment`: record one named start-to-goal edge
- `--route square_ccw`: record one full counterclockwise square lap per episode
- `--route square_cw`: record one full clockwise square lap per episode

For the full-square routes, the teacher behaves like a forward-facing rover rather than a sliding mecanum demo:

- it keeps `vy` near zero
- it drives mainly with forward speed plus yaw
- it blends heading early into the next corner
- it records one full lap as one episode
- it writes directly to the same `video.mp4` + `data.parquet` + `episode_info.json` layout used by the legacy CNN loader

Default square geometry:

- taped route centerline corners: `(-0.45, -0.45) -> (0.45, -0.45) -> (0.45, 0.45) -> (-0.45, 0.45)` meters
- the full-square start pose is the left midpoint `(-0.45, 0.0, 0.04)`
- `square_ccw` starts at yaw `-pi/2`
- `square_cw` starts at yaw `+pi/2`
- floor half-extent: `1.40` meters
- wall center planes: `x/y = +/-1.42` meters

### View the square-loop arena

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/view_turbopi_square.py --livestream 2
```

### Record autonomous CNN-format episodes

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/record_turbopi_square_simple.py --headless --route square_ccw --num_episodes 8
```

For high-throughput collection, use headless mode. The simple recorder automatically relaxes the full-square physics step to a more efficient setting in headless runs while keeping the saved control cadence unchanged. That makes the published command much faster than the older square recorder without changing the dataset format.

Example: 50 fast counterclockwise episodes for bulk dataset generation:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/record_turbopi_square_simple.py --headless --route square_ccw --num_episodes 50 --target_speed 0.30 --lookahead_distance 0.16 --corner_slowdown_distance 0.30 --square_max_wz 1.25 --max_episode_time 25
```

Watch the same recorder in livestream mode:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/record_turbopi_square_simple.py --livestream 2 --view chase --route square_ccw --num_episodes 10
```

Episodes are written by default to:

```text
/workspace/turbopi_standalone/data/cnn_square_loop/session_simple_<timestamp>/
```

Each saved episode has:

```text
episode_00000/
├── video.mp4
├── data.parquet
└── episode_info.json
```

There is also session-level metadata:

- `session_info.json`
- `task_mapping.json`
- `collection_summary.json`

### How Lap Completion Is Decided

For `square_ccw` and `square_cw`, the recorder does not accept an episode based on time alone.

On every control tick, it projects the robot root `x,y` position onto the nearest edge of the square and converts that to a normalized `track_phase` in `[0, 1)`:

- left edge: `0.00 -> 0.25`
- top edge: `0.25 -> 0.50`
- right edge: `0.50 -> 0.75`
- bottom edge: `0.75 -> 1.00`

It then accumulates only forward signed phase progress into `lap_progress`:

- `square_cw`: positive phase advance counts
- `square_ccw`: negative phase advance is flipped and counts
- wrap-around at `1.0 -> 0.0` is handled explicitly

By default, a rollout is accepted when:

- `lap_progress >= 0.97`
- the robot returns near the start phase
- the robot returns near the start heading
- commanded forward distance covers at least `min_square_distance_ratio` of the square perimeter

By default, a rollout is rejected if:

- `track_error > 0.25` meters
- it makes no useful forward progress for `8.0` seconds
- the simulation app closes or the state goes invalid
- the camera stays too flat during warmup
- it times out at `45.0` seconds

Unlike the older dynamic square recorder, the simple recorder does not apply a second wave of post-lap quality gates before saving. Instead, it saves every rollout that actually satisfies the route-completion and safety checks above, while still writing summary metrics such as mean/p90/max track error and `mean_action_vy_vx_ratio` into the episode metadata for later inspection.

### Segment Mode

If you want a simpler teacher than the full square, use `--route segment` together with named points like `bl`, `br`, `tr`, `tl`, `lm`, `rm`, `tm`, `bm`, or `center`. For example:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/record_turbopi_square_simple.py --headless --route segment --start bl --goal br --num_episodes 8
```

That records one point-to-point traversal per episode in the same dataset format.

### Legacy Dynamic Recorder

The older [`scripts/record_turbopi_square_cnn.py`](scripts/record_turbopi_square_cnn.py) is still in the repo for comparison, but the simple recorder above is the recommended square-loop path for new clones because it is easier to tune, faster to inspect, and now matches the default CNN inference path.

See [Square Loop Recorder](./docs/square_loop_cnn_recorder.md) for the full workflow and [No-Text CNN Pipeline](./docs/no_text_cnn_pipeline.md) for the architecture and data-path explanation based on your legacy reference files.

## Train the CNN

The standalone repo now includes a local `cnn_policy` package so you can train directly against the recorded square-loop dataset without switching repos.

Train on all recorded sessions under the default dataset root:

```bash
cd /workspace/turbopi_standalone
PYTHONPATH=/workspace/turbopi_standalone /workspace/isaaclab/_isaac_sim/python.sh -m cnn_policy.train --episodes-dir /workspace/turbopi_standalone/data/cnn_square_loop --run-dir /workspace/turbopi_standalone/runs/cnn_square_simple --device cuda
```

Train on a single session:

```bash
cd /workspace/turbopi_standalone
PYTHONPATH=/workspace/turbopi_standalone /workspace/isaaclab/_isaac_sim/python.sh -m cnn_policy.train --episodes-dir /workspace/turbopi_standalone/data/cnn_square_loop/session_simple_<timestamp> --run-dir /workspace/turbopi_standalone/runs/cnn_square_simple --device cuda
```

If you only have one session, the training code warns and skips validation because the split happens at the session level, not the episode level.

## Run the Trained Policy

After training, you can load a checkpoint and watch the CNN drive the TurboPi in the same square-loop arena.

Livestream with chase camera:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/drive_turbopi_square_cnn.py --livestream 2 --view chase --direction counterclockwise --control_mode kinematic --checkpoint /workspace/turbopi_standalone/runs/cnn_square_simple/<run_name>/checkpoints/best.pt
```

Livestream from the robot-mounted camera:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/drive_turbopi_square_cnn.py --livestream 2 --view robot --direction counterclockwise --control_mode kinematic --checkpoint /workspace/turbopi_standalone/runs/cnn_square_simple/<run_name>/checkpoints/best.pt
```

Useful options:

- `--direction counterclockwise` to match the square-CCW training set
- `--control_mode dynamic` if you explicitly want wheel-physics deployment instead of the default kinematic mode
- `--duration 30`
- `--policy_device cpu`
- `--disable_auto_reset`

## Teleop Trace and Replay

If you want to drive the robot yourself and then reproduce exactly what happened, use the teleop trace flow instead of the autonomous teacher.

### Record the trace

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/teleop_trace_turbopi_square.py --livestream 2
```

This records:

- root position
- root orientation
- world and body velocities
- raw keyboard command as body twist `[vx, vy, wz]`
- filtered body command after first-order smoothing
- wheel targets
- track error and track phase

The interactive teleop, trace, and command-replay scripts use a small closed-loop yaw-rate compensator by default to tame Isaac Lab's residual mecanum over-rotation. The raw autonomous teacher keeps this compensation off by default so its existing tuning stays stable.

Output goes under:

```text
/workspace/turbopi_standalone/data/teleop_traces/<trace_name>/
```

### Replay the trace

Exact pose replay:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/replay_turbopi_square_trace.py --trace_dir /workspace/turbopi_standalone/data/teleop_traces/<trace_name> --mode pose --livestream 2
```

Physics replay from the recorded body commands:

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/replay_turbopi_square_trace.py --trace_dir /workspace/turbopi_standalone/data/teleop_traces/<trace_name> --mode command --livestream 2
```

## Detailed Usage

### Local desktop launch

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/teleop_turbopi.py
```

### Remote livestream launch

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/teleop_turbopi.py --livestream 2
```

### Start with slower keyboard motion

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/teleop_turbopi.py --vx 0.20 --vy 0.15 --wz 0.80
```

### Tighten or loosen the command smoothing

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/teleop_turbopi.py --command_filter 0.15
```

Higher values react faster. Lower values feel softer.

### Skip the procedural rollers

```bash
cd /workspace
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/view_turbopi.py --no_rollers
```

That is mostly useful for debugging asset loading or isolating wheel-contact issues. It will not behave like a real mecanum base for strafe or in-place yaw.

## Troubleshooting

### I launched teleop and nothing responds

Keyboard teleop needs an interactive viewport. Use a normal GUI session, or launch with `--livestream 2` on a remote machine.

### I only want to inspect the model and camera pose

Use the plain viewer:

```bash
./isaaclab/isaaclab.sh -p /workspace/turbopi_standalone/scripts/view_turbopi.py --view robot --livestream 2
```

### I need the source robot description

Use the bundled URDF at [`assets/turbopi/turbopi_description/urdf/turbopi.urdf`](assets/turbopi/turbopi_description/urdf/turbopi.urdf).

### I need the already-converted simulation asset

Use the bundled USD at [`assets/turbopi/turbopi.usd`](assets/turbopi/turbopi.usd).

### Autonomous recorder says parquet export is unavailable

Install a parquet engine in the Isaac Lab Python environment:

```bash
/workspace/isaaclab/_isaac_sim/python.sh -m pip install pyarrow
```

### The CNN training loader cannot read `video.mp4`

Install PyAV in the Isaac Lab Python environment:

```bash
/workspace/isaaclab/_isaac_sim/python.sh -m pip install av
```
