"""Drive the TurboPi once around the taped square and save one CNN episode.

No teacher config, no omega tracker, no phase projection. The robot walks
through a hard-coded list of waypoints: corner -> in-place turn -> corner
-> in-place turn -> ... -> back to start. Each drive leg uses a plain P
controller on the heading-to-waypoint error; each turn leg uses a plain P
controller on the yaw error. That is all.

    cd /workspace
    ./isaaclab/isaaclab.sh -p \
        /workspace/turbopi_standalone/scripts/record_turbopi_square_simple.py \
        --livestream 2 --view chase --direction clockwise --num_episodes 1
"""

from __future__ import annotations

import argparse
import math
import os
import signal
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "cnn_square_loop"

parser = argparse.ArgumentParser(description="Drive the TurboPi one full lap with a simple waypoint driver.")
parser.add_argument("--asset_usd", type=str, default=None)
parser.add_argument("--view", type=str, choices=("overview", "chase", "robot"), default="chase")
parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
parser.add_argument("--session_name", type=str, default=None)
parser.add_argument("--dataset_name", type=str, default="turbopi_square_loop_cnn_simple")
parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument("--direction", type=str, choices=("clockwise", "counterclockwise"), default="clockwise")
parser.add_argument("--physics_dt", type=float, default=1.0 / 120.0)
parser.add_argument("--control_hz", type=float, default=10.0)
parser.add_argument("--image_width", type=int, default=160)
parser.add_argument("--image_height", type=int, default=120)
parser.add_argument("--square_half_extent", type=float, default=0.45)
parser.add_argument("--floor_half_extent", type=float, default=1.40)
parser.add_argument("--wall_height", type=float, default=0.55)
parser.add_argument("--wall_thickness", type=float, default=0.04)
parser.add_argument("--tape_width", type=float, default=0.08)
parser.add_argument("--target_speed", type=float, default=0.18, help="Cruise forward speed during drive legs (m/s).")
parser.add_argument("--min_speed_scale", type=float, default=0.25, help="Min forward speed as a fraction of cruise when approaching a waypoint or off-heading.")
parser.add_argument("--approach_distance", type=float, default=0.12, help="Distance to target at which the robot starts slowing down (m).")
parser.add_argument("--position_tolerance", type=float, default=0.04, help="How close to the waypoint counts as arrived (m).")
parser.add_argument("--yaw_tolerance", type=float, default=0.06, help="How close to the target yaw counts as done turning (rad).")
parser.add_argument("--turn_rate", type=float, default=1.1, help="Max yaw rate during in-place turns (rad/s).")
parser.add_argument("--turn_gain", type=float, default=3.0, help="P gain on yaw error during in-place turns.")
parser.add_argument("--drive_heading_gain", type=float, default=1.5, help="P gain on heading-to-waypoint error while driving.")
parser.add_argument("--max_wz_during_drive", type=float, default=0.6, help="Cap on wz correction while driving (rad/s).")
parser.add_argument("--settle_steps", type=int, default=24)
parser.add_argument("--cooldown_steps", type=int, default=12)
parser.add_argument("--camera_warmup_steps", type=int, default=18)
parser.add_argument("--min_image_std", type=float, default=8.0)
parser.add_argument("--max_episode_time", type=float, default=60.0)
parser.add_argument("--no_rollers", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if os.environ.get("DISPLAY") is None and not args_cli.headless:
    print("[INFO] DISPLAY is not set. Enabling headless rendering.")
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.math import euler_xyz_from_quat

from cnn_dataset import CNNSessionWriter, EpisodeFrame, EpisodeResult
from common import (
    activate_view_mode,
    get_arm_joint_ids,
    get_viewport,
    get_wheel_joint_ids,
    hold_arm_posture,
    reset_robot_pose,
    resolve_asset_usd,
    spawn_turbopi,
    twist_to_wheel_targets,
    update_chase_camera,
)
from square_loop import (
    SquareTrackSceneCfg,
    TASK_INDEX,
    TASKS,
    build_robot_camera_sensor,
    design_square_loop_scene,
)


# ---------------------------------------------------------------------------
# Waypoint plan
# ---------------------------------------------------------------------------


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def initial_yaw(direction: str) -> float:
    return math.pi / 2.0 if direction == "clockwise" else -math.pi / 2.0


def build_plan(h: float, direction: str):
    """Return a list of (kind, target) tuples that describe one full lap.

    kind == "drive"  -> target is a (x, y) world waypoint to drive to
    kind == "turn"   -> target is a yaw angle in [-pi, pi] to rotate in place to
    """
    if direction == "clockwise":
        # Start at (-h, 0), facing +pi/2 (north). Go north, east, south, west, north.
        return [
            ("drive", (-h, +h)),
            ("turn",  0.0),
            ("drive", (+h, +h)),
            ("turn",  -math.pi / 2.0),
            ("drive", (+h, -h)),
            ("turn",  math.pi),
            ("drive", (-h, -h)),
            ("turn",  math.pi / 2.0),
            ("drive", (-h, 0.0)),
        ]
    # counterclockwise: start at (-h, 0) facing -pi/2 (south).
    return [
        ("drive", (-h, -h)),
        ("turn",  0.0),
        ("drive", (+h, -h)),
        ("turn",  math.pi / 2.0),
        ("drive", (+h, +h)),
        ("turn",  math.pi),
        ("drive", (-h, +h)),
        ("turn",  -math.pi / 2.0),
        ("drive", (-h, 0.0)),
    ]


# ---------------------------------------------------------------------------
# Track error helper (unsigned distance to the taped square)
# ---------------------------------------------------------------------------


def distance_to_tape(x: float, y: float, h: float) -> float:
    clamp_x = max(-h, min(h, x))
    clamp_y = max(-h, min(h, y))
    d_left = math.hypot(x - (-h), y - clamp_y)
    d_right = math.hypot(x - (+h), y - clamp_y)
    d_top = math.hypot(x - clamp_x, y - (+h))
    d_bottom = math.hypot(x - clamp_x, y - (-h))
    return min(d_left, d_right, d_top, d_bottom)


# ---------------------------------------------------------------------------
# Sim helpers
# ---------------------------------------------------------------------------


def ensure_sim_playing(sim):
    if not sim.is_playing():
        sim.play()


def rgb_frame(camera) -> np.ndarray:
    image = camera.data.output["rgb"]
    if image is None or image.numel() == 0:
        raise RuntimeError("Camera sensor has no RGB data yet.")
    rgb = image[0, ..., :3].detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.clip(rgb, 0.0, 255.0)
            if rgb.max() <= 1.0:
                rgb = rgb * 255.0
        else:
            rgb = np.clip(rgb, 0, 255)
        rgb = rgb.astype(np.uint8)
    return rgb


def steps_per_control(physics_dt: float, control_hz: float) -> int:
    control_dt = 1.0 / max(control_hz, 1e-6)
    return max(1, int(round(control_dt / physics_dt)))


def apply_command(robot, wheel_joint_ids, arm_joint_ids, command_vec: tuple[float, float, float]):
    command_t = torch.tensor([command_vec], dtype=torch.float32, device=robot.device)
    wheel_targets = twist_to_wheel_targets(command_t, robot.device)
    robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_joint_ids)
    hold_arm_posture(robot, arm_joint_ids)
    robot.write_data_to_sim()


def step_n(sim, robot, camera, wheel_joint_ids, arm_joint_ids, command_vec, substeps, physics_dt, viewport, active_view):
    for _ in range(substeps):
        if not simulation_app.is_running():
            return False
        ensure_sim_playing(sim)
        apply_command(robot, wheel_joint_ids, arm_joint_ids, command_vec)
        sim.step()
        robot.update(physics_dt)
        if active_view == "chase":
            update_chase_camera(robot, viewport)
    camera.update(dt=substeps * physics_dt)
    return True


def get_pose(robot) -> tuple[float, float, float]:
    x = float(robot.data.root_pos_w[0, 0].item())
    y = float(robot.data.root_pos_w[0, 1].item())
    _, _, yaw_t = euler_xyz_from_quat(robot.data.root_quat_w)
    yaw = wrap_to_pi(float(yaw_t[0].item()))
    return x, y, yaw


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


def compute_command(kind: str, target, pose: tuple[float, float, float]) -> tuple[tuple[float, float, float], bool]:
    """Return ((vx, vy, wz), done) for the current phase."""
    x, y, yaw = pose

    if kind == "drive":
        tx, ty = target
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        if dist < args_cli.position_tolerance:
            return (0.0, 0.0, 0.0), True

        desired_yaw = math.atan2(dy, dx)
        yaw_error = wrap_to_pi(desired_yaw - yaw)

        approach_scale = min(1.0, dist / max(args_cli.approach_distance, 1e-6))
        heading_scale = max(args_cli.min_speed_scale, math.cos(yaw_error))
        vx = args_cli.target_speed * approach_scale * heading_scale
        vx = max(args_cli.target_speed * args_cli.min_speed_scale, min(args_cli.target_speed, vx))

        wz = args_cli.drive_heading_gain * yaw_error
        wz = max(-args_cli.max_wz_during_drive, min(args_cli.max_wz_during_drive, wz))
        return (vx, 0.0, wz), False

    if kind == "turn":
        target_yaw = float(target)
        yaw_error = wrap_to_pi(target_yaw - yaw)
        if abs(yaw_error) < args_cli.yaw_tolerance:
            return (0.0, 0.0, 0.0), True
        wz = args_cli.turn_gain * yaw_error
        wz = max(-args_cli.turn_rate, min(args_cli.turn_rate, wz))
        return (0.0, 0.0, wz), False

    return (0.0, 0.0, 0.0), True


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


class StopFlag:
    def __init__(self):
        self.requested = False

    def request(self, signum, _frame):
        self.requested = True
        print(f"\n[INFO] Received signal {signum}. Finishing cleanup.", flush=True)


def warm_camera(sim, robot, camera, wheel_joint_ids, arm_joint_ids, viewport, active_view, steps, physics_dt, min_std):
    last_std = 0.0
    for _ in range(max(1, steps)):
        ok = step_n(sim, robot, camera, wheel_joint_ids, arm_joint_ids, (0.0, 0.0, 0.0), 1, physics_dt, viewport, active_view)
        if not ok:
            return False, last_std
        try:
            last_std = float(np.asarray(rgb_frame(camera), dtype=np.float32).std())
        except Exception:
            last_std = 0.0
        if last_std >= min_std:
            return True, last_std
    return False, last_std


def run_episode(*, sim, robot, camera, wheel_joint_ids, arm_joint_ids, viewport, active_view, scene_cfg, direction, stop_flag):
    h = scene_cfg.square_half_extent
    start_pos = (-h, 0.0, 0.04)
    start_yaw = initial_yaw(direction)
    reset_robot_pose(robot, position=start_pos, yaw=start_yaw)

    physics_dt = float(args_cli.physics_dt)
    control_dt = 1.0 / max(args_cli.control_hz, 1e-6)
    substeps = steps_per_control(physics_dt, args_cli.control_hz)
    max_steps = max(1, int(math.ceil(args_cli.max_episode_time / control_dt)))

    ok = step_n(sim, robot, camera, wheel_joint_ids, arm_joint_ids, (0.0, 0.0, 0.0), max(1, args_cli.settle_steps), physics_dt, viewport, active_view)
    if not ok:
        raise RuntimeError("Simulation app closed during reset settle.")

    camera_ready, warm_std = warm_camera(
        sim, robot, camera, wheel_joint_ids, arm_joint_ids, viewport, active_view,
        args_cli.camera_warmup_steps, physics_dt, args_cli.min_image_std,
    )
    if not camera_ready:
        print(f"[WARN] Camera warmup std stayed at {warm_std:.2f}; continuing anyway.", flush=True)

    plan = build_plan(h, direction)
    phase_index = 0
    frames: list[EpisodeFrame] = []
    track_errors: list[float] = []
    body_speeds: list[float] = []
    image_stds: list[float] = []
    action_history: list[np.ndarray] = []
    prev_action = np.zeros(3, dtype=np.float32)

    MAX_COMMAND = np.array([0.45, 0.35, 2.0], dtype=np.float32)

    last_print_at = -1.0
    terminal_reason = "timeout"
    success = False

    print(f"[INFO] Starting {direction} simple lap. {len(plan)} phases.", flush=True)

    for step_index in range(max_steps):
        if stop_flag.requested:
            terminal_reason = "interrupted"
            break
        if not simulation_app.is_running():
            terminal_reason = "app_closed"
            break
        if phase_index >= len(plan):
            terminal_reason = "lap_complete"
            success = True
            break

        pose = get_pose(robot)
        kind, target = plan[phase_index]
        command_vec, phase_done = compute_command(kind, target, pose)
        if phase_done:
            phase_index += 1
            # Hold zero for one control tick so the turn -> drive transition
            # doesn't smear together.
            if not step_n(sim, robot, camera, wheel_joint_ids, arm_joint_ids, (0.0, 0.0, 0.0), substeps, physics_dt, viewport, active_view):
                terminal_reason = "app_closed"
                break
            continue

        episode_time = step_index * control_dt
        track_err = distance_to_tape(pose[0], pose[1], h)
        track_errors.append(track_err)

        body_lin = robot.data.root_lin_vel_b[0, :2].detach().cpu().numpy()
        body_ang = robot.data.root_ang_vel_b[0, 2:3].detach().cpu().numpy()
        body_vel = np.concatenate([body_lin, body_ang]).astype(np.float32)
        body_speeds.append(float(np.linalg.norm(body_lin)))

        image = rgb_frame(camera)
        image_stds.append(float(np.asarray(image, dtype=np.float32).std()))

        command_np = np.array(command_vec, dtype=np.float32)
        action_np = np.clip(command_np / MAX_COMMAND, -1.0, 1.0)
        action_history.append(action_np)

        frames.append(
            EpisodeFrame(
                image_rgb=image,
                timestamp=float(episode_time),
                state=prev_action.copy(),
                action=action_np.copy(),
                command=command_np.copy(),
                body_velocity=body_vel,
                track_error=float(track_err),
                lap_progress=float(phase_index / max(1, len(plan))),
            )
        )
        prev_action = action_np

        if not step_n(sim, robot, camera, wheel_joint_ids, arm_joint_ids, tuple(command_vec), substeps, physics_dt, viewport, active_view):
            terminal_reason = "app_closed"
            break

        if episode_time - last_print_at >= 1.0:
            last_print_at = episode_time
            print(
                f"[INFO] {direction:16s} t={episode_time:5.1f}s phase={phase_index:2d}/{len(plan)} "
                f"kind={kind:5s} pose=({pose[0]:+0.2f},{pose[1]:+0.2f},{pose[2]:+0.2f}) "
                f"err={track_err:0.3f} cmd=[{command_vec[0]:+0.2f},{command_vec[1]:+0.2f},{command_vec[2]:+0.2f}]",
                flush=True,
            )

    step_n(sim, robot, camera, wheel_joint_ids, arm_joint_ids, (0.0, 0.0, 0.0), max(1, args_cli.cooldown_steps), physics_dt, viewport, active_view)

    duration_s = len(frames) * control_dt
    mean_track_error = float(np.mean(track_errors)) if track_errors else float("inf")
    p90_track_error = float(np.quantile(track_errors, 0.9)) if track_errors else float("inf")
    max_track_error = float(np.max(track_errors)) if track_errors else float("inf")
    frames_over_010_ratio = float(np.mean(np.asarray(track_errors) > 0.10)) if track_errors else 1.0
    frames_over_015_ratio = float(np.mean(np.asarray(track_errors) > 0.15)) if track_errors else 1.0
    mean_image_std = float(np.mean(image_stds)) if image_stds else 0.0
    min_image_std_val = float(np.min(image_stds)) if image_stds else 0.0
    mean_speed = float(np.mean(body_speeds)) if body_speeds else 0.0
    action_array = np.asarray(action_history, dtype=np.float32) if action_history else np.zeros((0, 3), dtype=np.float32)
    mean_abs = np.mean(np.abs(action_array), axis=0) if len(action_array) > 0 else np.zeros(3, dtype=np.float32)
    mean_vy_vx = float(mean_abs[1] / max(float(mean_abs[0]), 1e-6))
    final_lap_progress = float(phase_index / max(1, len(plan)))

    return EpisodeResult(
        direction=direction,
        task_name=direction,
        task_index=TASK_INDEX[direction],
        frames=frames,
        success=success,
        terminal_reason=terminal_reason,
        final_lap_progress=final_lap_progress,
        mean_track_error=mean_track_error,
        p90_track_error=p90_track_error,
        max_track_error=max_track_error,
        frames_over_010_ratio=frames_over_010_ratio,
        frames_over_015_ratio=frames_over_015_ratio,
        mean_image_std=mean_image_std,
        min_image_std=min_image_std_val,
        mean_abs_action_vx=float(mean_abs[0]),
        mean_abs_action_vy=float(mean_abs[1]),
        mean_abs_action_wz=float(mean_abs[2]),
        mean_action_vy_vx_ratio=mean_vy_vx,
        mean_speed=mean_speed,
        duration_s=duration_s,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_session_name() -> str:
    return args_cli.session_name or datetime.utcnow().strftime("session_simple_%Y%m%d_%H%M%S")


def main() -> None:
    physics_dt = float(args_cli.physics_dt)
    substeps = steps_per_control(physics_dt, args_cli.control_hz)
    livestream_enabled = bool(getattr(args_cli, "livestream", 0))
    render_interval = substeps if args_cli.headless and not livestream_enabled else 1

    scene_cfg = SquareTrackSceneCfg(
        square_half_extent=args_cli.square_half_extent,
        floor_half_extent=args_cli.floor_half_extent,
        tape_width=args_cli.tape_width,
        wall_height=args_cli.wall_height,
        wall_thickness=args_cli.wall_thickness,
    )

    sim_cfg = sim_utils.SimulationCfg(dt=physics_dt, render_interval=render_interval, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    design_square_loop_scene(scene_cfg)
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)
    camera = build_robot_camera_sensor(width=args_cli.image_width, height=args_cli.image_height)

    sim.reset()
    camera.update(dt=0.0)
    sim.play()

    wheel_joint_ids = get_wheel_joint_ids(robot)
    arm_joint_ids = get_arm_joint_ids(robot)
    viewport = get_viewport()
    active_view = activate_view_mode(args_cli.view, sim, robot, viewport)

    writer = CNNSessionWriter(
        output_root=Path(args_cli.output_dir),
        session_name=build_session_name(),
        dataset_name=args_cli.dataset_name,
        fps=args_cli.control_hz,
        image_width=args_cli.image_width,
        image_height=args_cli.image_height,
        episode_time_s=args_cli.max_episode_time,
        control_hz=args_cli.control_hz,
        physics_dt=physics_dt,
        tasks=TASKS,
        track_layout="square_tape_loop",
        episode_definition="one_full_autonomous_lap",
    )

    stop_flag = StopFlag()
    signal.signal(signal.SIGINT, stop_flag.request)
    signal.signal(signal.SIGTERM, stop_flag.request)

    print()
    print("=" * 60)
    print("  TurboPi Simple Waypoint Lap Recorder")
    print("=" * 60)
    print(f"  TurboPi USD    : {resolve_asset_usd(args_cli.asset_usd)}")
    print(f"  Output session : {writer.session_dir}")
    print(f"  Direction      : {args_cli.direction}")
    print(f"  Episodes       : {args_cli.num_episodes}")
    print(f"  Control rate   : {args_cli.control_hz:.1f} Hz")
    print(f"  Target speed   : {args_cli.target_speed:.2f} m/s")
    print(f"  Turn rate      : {args_cli.turn_rate:.2f} rad/s (gain {args_cli.turn_gain:.1f})")
    print(f"  Drive heading  : gain {args_cli.drive_heading_gain:.1f}, wz cap {args_cli.max_wz_during_drive:.2f}")
    print(f"  Tolerances     : pos {args_cli.position_tolerance:.02f} m, yaw {args_cli.yaw_tolerance:.02f} rad")
    print(f"  Start view     : {active_view}")
    print()

    saved = 0
    attempts = 0
    try:
        while saved < args_cli.num_episodes and simulation_app.is_running() and not stop_flag.requested:
            attempts += 1
            result = run_episode(
                sim=sim, robot=robot, camera=camera,
                wheel_joint_ids=wheel_joint_ids, arm_joint_ids=arm_joint_ids,
                viewport=viewport, active_view=active_view,
                scene_cfg=scene_cfg, direction=args_cli.direction,
                stop_flag=stop_flag,
            )
            if result.success and result.frames:
                episode_dir = writer.save_episode(saved, result)
                saved += 1
                print(
                    f"[INFO] Saved episode_{saved - 1:05d} [{args_cli.direction}] frames={len(result.frames)} "
                    f"final_phase={result.final_lap_progress:.2f} mean_err={result.mean_track_error:.3f} -> {episode_dir}",
                    flush=True,
                )
            else:
                writer.record_failure()
                print(
                    f"[WARN] Attempt {attempts} failed [{args_cli.direction}] reason={result.terminal_reason} "
                    f"frames={len(result.frames)} final_phase={result.final_lap_progress:.2f}",
                    flush=True,
                )
    finally:
        print()
        print(f"[INFO] Session complete : {writer.session_dir}", flush=True)
        print(f"[INFO] Saved episodes   : {saved}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
