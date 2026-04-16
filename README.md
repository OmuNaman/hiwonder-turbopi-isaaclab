# Hiwonder TurboPi for Isaac Lab

This repository packages the Hiwonder TurboPi robot for Isaac Lab.

Using this repo, you can:

- load the TurboPi into Isaac Lab
- inspect the robot in a clean standalone scene
- switch to the robot-mounted camera
- teleoperate the robot from Isaac Lab
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
    ├── mecanum_builder.py
    ├── teleop_turbopi.py
    └── view_turbopi.py
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

## Teleop Controls

The teleop script uses Isaac Lab's built-in `Se2Keyboard`.

- `Up`: forward
- `Down`: backward
- `Left`: strafe right
- `Right`: strafe left
- `Z`: positive yaw
- `X`: negative yaw
- `L`: zero the current command
- `R`: reset the robot to its start pose
- `C`: cycle `overview -> chase -> robot`
- `Esc`: close the app

The lateral key mapping follows Isaac Lab's SE(2) keyboard device exactly, so the labels above match what the script actually sends.

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

This is not the RL environment, not the dataset collector, and not the square-loop trainer. It is the lightweight entry point for loading, viewing, and driving the Hiwonder TurboPi in Isaac Lab.

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

That is mostly useful for debugging asset loading or isolating wheel-contact issues.

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
