"""Autonomous TurboPi square-loop recorder that writes legacy CNN episodes."""

from __future__ import annotations

import argparse
import math
import os
import signal
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "cnn_square_loop"

parser = argparse.ArgumentParser(description="Record autonomous TurboPi square-loop episodes for the no-text CNN pipeline.")
parser.add_argument("--asset_usd", type=str, default=None, help="Optional override for the TurboPi USD.")
parser.add_argument(
    "--view",
    type=str,
    choices=("overview", "chase", "robot"),
    default="overview",
    help="Initial viewport mode when a GUI or livestream is available.",
)
parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Root directory for saved sessions.")
parser.add_argument("--session_name", type=str, default=None, help="Optional session directory name.")
parser.add_argument("--dataset_name", type=str, default="turbopi_square_loop_cnn", help="Dataset name stored in session metadata.")
parser.add_argument("--num_episodes", type=int, default=8, help="Number of successful episodes to save.")
parser.add_argument(
    "--directions",
    type=str,
    default="clockwise",
    help="Comma-separated direction cycle, e.g. 'clockwise,counterclockwise'.",
)
parser.add_argument("--max_attempts", type=int, default=0, help="Maximum total rollout attempts before exiting. 0 uses a safe default.")
parser.add_argument("--physics_dt", type=float, default=1.0 / 120.0, help="Physics step in seconds.")
parser.add_argument("--control_hz", type=float, default=10.0, help="Control and recording frequency in Hz.")
parser.add_argument(
    "--render_interval",
    type=int,
    default=0,
    help="Physics steps per render. 0 uses an automatic setting: control-rate rendering for headless runs, every physics step otherwise.",
)
parser.add_argument("--image_width", type=int, default=160, help="Recorded RGB width.")
parser.add_argument("--image_height", type=int, default=120, help="Recorded RGB height.")
parser.add_argument("--episode_time_s", type=float, default=45.0, help="Hard timeout per rollout in seconds.")
parser.add_argument("--lap_completion_threshold", type=float, default=0.97, help="Lap progress threshold for success.")
parser.add_argument("--stuck_timeout_s", type=float, default=5.0, help="Abort if signed lap progress stalls for this long.")
parser.add_argument("--stuck_grace_period_s", type=float, default=3.0, help="Ignore stuck checks during startup.")
parser.add_argument("--max_track_error", type=float, default=0.30, help="Abort if the robot drifts farther than this from the square path.")
parser.add_argument("--min_body_speed", type=float, default=0.03, help="Minimum mean planar speed required for an accepted rollout.")
parser.add_argument("--max_mean_track_error", type=float, default=0.06, help="Reject accepted laps whose mean track error is higher than this.")
parser.add_argument("--max_p90_track_error", type=float, default=0.12, help="Reject accepted laps whose p90 track error is higher than this.")
parser.add_argument("--max_peak_track_error", type=float, default=0.18, help="Reject accepted laps whose peak track error is higher than this.")
parser.add_argument("--max_frames_over_015_ratio", type=float, default=0.05, help="Reject accepted laps if more than this ratio of frames exceed 0.15 m track error.")
parser.add_argument("--min_image_std", type=float, default=8.0, help="Reject laps if the camera image stream is too flat or washed out.")
parser.add_argument("--max_mean_action_vy_vx_ratio", type=float, default=0.20, help="Reject laps if the mean absolute lateral action is too large relative to forward action.")
parser.add_argument("--camera_warmup_steps", type=int, default=18, help="Extra zero-command steps used to warm up the camera after reset.")
parser.add_argument("--start_phase_tolerance", type=float, default=0.06, help="Required closeness to the start phase when closing a lap.")
parser.add_argument("--start_yaw_tolerance", type=float, default=0.35, help="Required closeness to the start heading when closing a lap, in radians.")
parser.add_argument("--settle_steps", type=int, default=24, help="Zero-command steps after reset before an episode starts.")
parser.add_argument("--cooldown_steps", type=int, default=12, help="Zero-command steps after an episode ends.")
parser.add_argument("--square_half_extent", type=float, default=0.45, help="Half-size of the taped square path in meters.")
parser.add_argument("--floor_half_extent", type=float, default=1.40, help="Half-size of the visible arena floor in meters.")
parser.add_argument("--tape_width", type=float, default=0.08, help="Width of the square path marker in meters.")
parser.add_argument("--wall_height", type=float, default=0.55, help="Outer boundary wall height in meters.")
parser.add_argument("--wall_thickness", type=float, default=0.04, help="Outer boundary wall thickness in meters.")
parser.add_argument("--target_speed", type=float, default=0.18, help="Nominal teacher speed in m/s.")
parser.add_argument("--lookahead_distance", type=float, default=0.05, help="Lookahead distance along the square path in meters.")
parser.add_argument("--boundary_gain", type=float, default=0.8, help="Gain for lateral correction back to the square boundary.")
parser.add_argument("--heading_gain", type=float, default=5.4, help="Gain for yaw tracking.")
parser.add_argument("--cross_track_heading_gain", type=float, default=2.4, help="Extra yaw correction that turns the robot toward the lookahead point instead of sliding sideways.")
parser.add_argument("--lateral_gain", type=float, default=0.45, help="Small residual lateral correction gain in body coordinates.")
parser.add_argument("--corner_blend_distance", type=float, default=0.10, help="Distance from a corner where tangent blending begins.")
parser.add_argument("--corner_slowdown_distance", type=float, default=0.14, help="Distance from a corner where speed reduction begins.")
parser.add_argument("--max_lateral_speed", type=float, default=0.045, help="Cap for lateral correction speed in m/s.")
parser.add_argument("--track_error_slowdown", type=float, default=0.40, help="Slowdown factor applied when the robot drifts off-center.")
parser.add_argument("--min_tracking_scale", type=float, default=0.55, help="Minimum speed scale when track-error slowdown is active.")
parser.add_argument("--heading_slowdown_angle", type=float, default=0.75, help="Heading error angle beyond which forward speed is strongly reduced.")
parser.add_argument("--strafe_suppression_angle", type=float, default=0.35, help="Heading error angle beyond which lateral strafe is strongly suppressed.")
parser.add_argument("--max_vx", type=float, default=0.45, help="Maximum forward body command in m/s.")
parser.add_argument("--max_vy", type=float, default=0.35, help="Maximum lateral body command in m/s.")
parser.add_argument("--max_wz", type=float, default=2.00, help="Maximum yaw command in rad/s.")
parser.add_argument("--enable_omega_feedback", dest="enable_omega_feedback", action="store_true", default=True, help="Enable closed-loop yaw-rate compensation during autonomous recording (default: on).")
parser.add_argument("--disable_omega_feedback", dest="enable_omega_feedback", action="store_false", help="Disable closed-loop yaw-rate compensation during autonomous recording.")
parser.add_argument("--omega_feedback_gain", type=float, default=0.6, help="Closed-loop yaw-rate feedback gain (additive on top of the desired wz).")
parser.add_argument("--omega_measure_alpha", type=float, default=0.2, help="EMA factor for measured yaw rate in the compensator.")
parser.add_argument("--no_rollers", action="store_true", help="Skip procedural mecanum roller generation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if os.environ.get("DISPLAY") is None and not args_cli.headless:
    print("[INFO] DISPLAY is not set. Enabling headless rendering for autonomous recording.")
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.math import euler_xyz_from_quat

from cnn_dataset import CNNSessionWriter, EpisodeFrame, EpisodeResult
from common import (
    OmegaTracker,
    OmegaTrackerCfg,
    activate_view_mode,
    get_arm_joint_ids,
    get_viewport,
    get_wheel_joint_ids,
    hold_arm_posture,
    resolve_asset_usd,
    reset_robot_pose,
    spawn_turbopi,
    twist_to_wheel_targets,
    update_chase_camera,
)
from square_loop import (
    ControlLimits,
    SquareTrackSceneCfg,
    SquareTrackTeacher,
    TASK_INDEX,
    TASKS,
    TeacherControllerCfg,
    build_robot_camera_sensor,
    design_square_loop_scene,
    observe_track_state,
    start_pose_for_direction,
)


@dataclass(frozen=True)
class EpisodeRunResult:
    result: EpisodeResult | None
    saved: bool
    reason: str


class StopFlag:
    def __init__(self) -> None:
        self.requested = False

    def request(self, signum: int, _frame) -> None:
        self.requested = True
        print(f"\n[INFO] Received signal {signum}. Finishing the current cleanup.", flush=True)


class LapSegmentTracker:
    """Track ordered square-edge transitions so corner-cutting does not look like a full lap."""

    def __init__(self, *, initial_segment: int, direction: str):
        self.direction = direction
        self.step = 1 if direction == "clockwise" else -1
        self.current_segment = int(initial_segment)
        self.expected_segment = (self.current_segment + self.step) % 4
        self.completed_transitions = 0

    def update(self, segment_index: int) -> None:
        segment_index = int(segment_index)
        if segment_index == self.current_segment:
            return
        if segment_index == self.expected_segment:
            self.completed_transitions += 1
        self.current_segment = segment_index
        self.expected_segment = (self.current_segment + self.step) % 4

    @property
    def cycle_complete(self) -> bool:
        return self.completed_transitions >= 4


def parse_direction_cycle(raw: str) -> list[str]:
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not items:
        raise ValueError("At least one direction must be provided.")
    invalid = sorted(set(items) - set(TASKS))
    if invalid:
        raise ValueError(f"Unsupported direction(s): {', '.join(invalid)}")
    return items


def ensure_sim_playing(sim: sim_utils.SimulationContext) -> None:
    if not sim.is_playing():
        sim.play()


def rgb_frame_from_camera(camera) -> np.ndarray:
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


def apply_body_command(
    robot,
    wheel_joint_ids: list[int],
    arm_joint_ids: list[int],
    command: torch.Tensor,
    *,
    physics_dt: float,
    omega_tracker: OmegaTracker | None,
    max_omega_command: float,
) -> torch.Tensor:
    applied_command = command.view(1, 3)
    if omega_tracker is not None:
        applied_command = omega_tracker.compensate(
            applied_command,
            robot.data.root_ang_vel_b[:, 2],
            dt=physics_dt,
            command_limit=max_omega_command,
        )
    wheel_targets = twist_to_wheel_targets(applied_command, robot.device)
    robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_joint_ids)
    hold_arm_posture(robot, arm_joint_ids)
    robot.write_data_to_sim()
    return applied_command[0]


def zero_command(device: str | torch.device) -> torch.Tensor:
    return torch.zeros(3, dtype=torch.float32, device=device)


def signed_phase_delta(current_phase: float, previous_phase: float, direction: str) -> float:
    delta = current_phase - previous_phase
    if delta < -0.5:
        delta += 1.0
    if delta > 0.5:
        delta -= 1.0
    sign = 1.0 if direction == "clockwise" else -1.0
    return max(0.0, sign * delta)


def wrapped_phase_error(phase_a: float, phase_b: float) -> float:
    delta = (phase_a - phase_b + 0.5) % 1.0 - 0.5
    return abs(delta)


def wrapped_angle_error(angle_a: float, angle_b: float) -> float:
    delta = (angle_a - angle_b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(delta)


def image_std_rgb(image_rgb: np.ndarray) -> float:
    return float(np.asarray(image_rgb, dtype=np.float32).std())


def steps_per_control(physics_dt: float, control_hz: float) -> int:
    control_dt = 1.0 / max(control_hz, 1e-6)
    return max(1, int(round(control_dt / physics_dt)))


def step_simulation(
    *,
    sim: sim_utils.SimulationContext,
    robot,
    camera,
    wheel_joint_ids: list[int],
    arm_joint_ids: list[int],
    command: torch.Tensor,
    substeps: int,
    physics_dt: float,
    viewport,
    active_view: str,
    omega_tracker: OmegaTracker | None,
    max_omega_command: float,
) -> bool:
    for _ in range(substeps):
        if not simulation_app.is_running():
            return False
        ensure_sim_playing(sim)
        apply_body_command(
            robot,
            wheel_joint_ids,
            arm_joint_ids,
            command,
            physics_dt=physics_dt,
            omega_tracker=omega_tracker,
            max_omega_command=max_omega_command,
        )
        sim.step()
        robot.update(physics_dt)
        if active_view == "chase":
            update_chase_camera(robot, viewport)
    camera.update(dt=substeps * physics_dt)
    return True


def settle_robot(
    *,
    sim: sim_utils.SimulationContext,
    robot,
    camera,
    wheel_joint_ids: list[int],
    arm_joint_ids: list[int],
    steps: int,
    viewport,
    active_view: str,
    omega_tracker: OmegaTracker | None,
    max_omega_command: float,
) -> bool:
    return step_simulation(
        sim=sim,
        robot=robot,
        camera=camera,
        wheel_joint_ids=wheel_joint_ids,
        arm_joint_ids=arm_joint_ids,
        command=zero_command(robot.device),
        substeps=max(1, steps),
        physics_dt=args_cli.physics_dt,
        viewport=viewport,
        active_view=active_view,
        omega_tracker=omega_tracker,
        max_omega_command=max_omega_command,
    )


def warm_up_camera(
    *,
    sim: sim_utils.SimulationContext,
    robot,
    camera,
    wheel_joint_ids: list[int],
    arm_joint_ids: list[int],
    viewport,
    active_view: str,
    omega_tracker: OmegaTracker | None,
    max_omega_command: float,
    warmup_steps: int,
    min_image_std: float,
) -> tuple[bool, float]:
    """Advance a few rendered frames and verify that the RGB stream is not blank."""
    last_std = 0.0
    for _ in range(max(1, warmup_steps)):
        if not step_simulation(
            sim=sim,
            robot=robot,
            camera=camera,
            wheel_joint_ids=wheel_joint_ids,
            arm_joint_ids=arm_joint_ids,
            command=zero_command(robot.device),
            substeps=1,
            physics_dt=args_cli.physics_dt,
            viewport=viewport,
            active_view=active_view,
            omega_tracker=omega_tracker,
            max_omega_command=max_omega_command,
        ):
            return False, last_std
        try:
            last_std = image_std_rgb(rgb_frame_from_camera(camera))
        except Exception:
            last_std = 0.0
        if last_std >= min_image_std:
            return True, last_std
    return False, last_std


def run_episode(
    *,
    direction: str,
    sim: sim_utils.SimulationContext,
    robot,
    camera,
    wheel_joint_ids: list[int],
    arm_joint_ids: list[int],
    viewport,
    active_view: str,
    scene_cfg: SquareTrackSceneCfg,
    teacher: SquareTrackTeacher,
    stop_flag: StopFlag,
    omega_tracker: OmegaTracker | None,
    max_omega_command: float,
) -> EpisodeResult:
    start_position, start_yaw = start_pose_for_direction(scene_cfg, direction)
    reset_robot_pose(robot, position=start_position, yaw=start_yaw)
    teacher.reset(direction)
    if omega_tracker is not None:
        omega_tracker.reset()
    if not settle_robot(
        sim=sim,
        robot=robot,
        camera=camera,
        wheel_joint_ids=wheel_joint_ids,
        arm_joint_ids=arm_joint_ids,
        steps=args_cli.settle_steps,
        viewport=viewport,
        active_view=active_view,
        omega_tracker=omega_tracker,
        max_omega_command=max_omega_command,
    ):
        raise RuntimeError("Simulation app closed during reset settle.")

    tracking = observe_track_state(robot, scene_cfg)
    previous_phase = tracking.track_phase
    lap_progress = 0.0
    last_progress_time = 0.0
    frames: list[EpisodeFrame] = []
    track_errors: list[float] = []
    body_speeds: list[float] = []
    image_stds: list[float] = []
    action_history: list[np.ndarray] = []
    segment_tracker = LapSegmentTracker(initial_segment=int(math.floor(tracking.track_phase * 4.0)) % 4, direction=direction)
    control_dt = 1.0 / max(args_cli.control_hz, 1e-6)
    control_substeps = steps_per_control(args_cli.physics_dt, args_cli.control_hz)
    max_steps = max(1, int(math.ceil(args_cli.episode_time_s / control_dt)))
    terminal_reason = "timeout"
    success = False
    last_print_at = -1.0
    start_phase = tracking.track_phase
    _, _, start_yaw_tensor = euler_xyz_from_quat(robot.data.root_quat_w)
    start_yaw_actual = float(start_yaw_tensor[0].item())

    camera_ready, warmup_std = warm_up_camera(
        sim=sim,
        robot=robot,
        camera=camera,
        wheel_joint_ids=wheel_joint_ids,
        arm_joint_ids=arm_joint_ids,
        viewport=viewport,
        active_view=active_view,
        omega_tracker=omega_tracker,
        max_omega_command=max_omega_command,
        warmup_steps=args_cli.camera_warmup_steps,
        min_image_std=args_cli.min_image_std,
    )
    if not camera_ready:
        print(f"[WARN] Camera warmup failed; image std stayed at {warmup_std:0.2f}.", flush=True)
        return EpisodeResult(
            direction=direction,
            task_name=direction,
            task_index=TASK_INDEX[direction],
            frames=[],
            success=False,
            terminal_reason="invalid_camera",
            final_lap_progress=0.0,
            mean_track_error=float("inf"),
            p90_track_error=float("inf"),
            max_track_error=float("inf"),
            frames_over_010_ratio=1.0,
            frames_over_015_ratio=1.0,
            mean_image_std=warmup_std,
            min_image_std=warmup_std,
            mean_speed=0.0,
            duration_s=0.0,
        )

    print(f"[INFO] Starting {direction} rollout.", flush=True)

    for step_index in range(max_steps):
        if stop_flag.requested:
            terminal_reason = "interrupted"
            break
        if not simulation_app.is_running():
            terminal_reason = "app_closed"
            break

        ensure_sim_playing(sim)
        episode_time = step_index * control_dt
        tracking = observe_track_state(robot, scene_cfg)
        segment_tracker.update(int(math.floor(tracking.track_phase * 4.0)) % 4)
        lap_progress += signed_phase_delta(tracking.track_phase, previous_phase, direction)
        previous_phase = tracking.track_phase

        planar_speed = float(torch.linalg.norm(tracking.body_velocity).item())
        track_errors.append(tracking.track_error)
        body_speeds.append(planar_speed)
        if lap_progress > 1e-4:
            last_progress_time = episode_time

        if tracking.has_nan:
            print(
                f"[WARN] Invalid state: pos={tracking.position_w.tolist()} "
                f"height={tracking.height:0.4f} has_nan={tracking.has_nan}",
                flush=True,
            )
            terminal_reason = "invalid_state"
            break
        _, _, current_yaw_tensor = euler_xyz_from_quat(robot.data.root_quat_w)
        current_yaw = float(current_yaw_tensor[0].item())
        if (
            lap_progress >= args_cli.lap_completion_threshold
            and segment_tracker.cycle_complete
            and wrapped_phase_error(tracking.track_phase, start_phase) <= args_cli.start_phase_tolerance
            and wrapped_angle_error(current_yaw, start_yaw_actual) <= args_cli.start_yaw_tolerance
        ):
            terminal_reason = "lap_complete"
            success = True
            break
        if tracking.track_error > args_cli.max_track_error:
            terminal_reason = "off_track"
            break
        if episode_time >= args_cli.stuck_grace_period_s and episode_time - last_progress_time >= args_cli.stuck_timeout_s:
            terminal_reason = "stuck_no_progress"
            break

        image_rgb = rgb_frame_from_camera(camera)
        image_std = image_std_rgb(image_rgb)
        image_stds.append(image_std)
        state, action, command, info = teacher.compute_action(robot)
        action_np = action.detach().cpu().numpy().astype(np.float32)
        action_history.append(action_np)
        frames.append(
            EpisodeFrame(
                image_rgb=image_rgb,
                timestamp=episode_time,
                state=state.detach().cpu().numpy(),
                action=action_np,
                command=command.detach().cpu().numpy(),
                body_velocity=torch.cat((tracking.body_velocity, tracking.body_ang_velocity)).detach().cpu().numpy(),
                track_error=tracking.track_error,
                lap_progress=lap_progress,
            )
        )

        if not step_simulation(
            sim=sim,
            robot=robot,
            camera=camera,
            wheel_joint_ids=wheel_joint_ids,
            arm_joint_ids=arm_joint_ids,
            command=command,
            substeps=control_substeps,
            physics_dt=args_cli.physics_dt,
            viewport=viewport,
            active_view=active_view,
            omega_tracker=omega_tracker,
            max_omega_command=max_omega_command,
        ):
            terminal_reason = "app_closed"
            break

        if episode_time - last_print_at >= 1.0:
            last_print_at = episode_time
            print(
                f"[INFO] {direction:16s} t={episode_time:5.1f}s frames={len(frames):4d} "
                f"lap={lap_progress:0.3f} err={tracking.track_error:0.3f} "
                f"img_std={image_std:0.1f} cmd=[{float(action[0]):0.2f}, {float(action[1]):0.2f}, {float(action[2]):0.2f}]",
                flush=True,
            )

    settle_robot(
        sim=sim,
        robot=robot,
        camera=camera,
        wheel_joint_ids=wheel_joint_ids,
        arm_joint_ids=arm_joint_ids,
        steps=args_cli.cooldown_steps,
        viewport=viewport,
        active_view=active_view,
        omega_tracker=omega_tracker,
        max_omega_command=max_omega_command,
    )

    duration_s = len(frames) * control_dt
    mean_track_error = float(np.mean(track_errors)) if track_errors else float("inf")
    p90_track_error = float(np.quantile(track_errors, 0.9)) if track_errors else float("inf")
    max_track_error = float(np.max(track_errors)) if track_errors else float("inf")
    frames_over_010_ratio = float(np.mean(np.asarray(track_errors) > 0.10)) if track_errors else 1.0
    frames_over_015_ratio = float(np.mean(np.asarray(track_errors) > 0.15)) if track_errors else 1.0
    mean_image_std = float(np.mean(image_stds)) if image_stds else 0.0
    min_image_std = float(np.min(image_stds)) if image_stds else 0.0
    action_array = np.asarray(action_history, dtype=np.float32) if action_history else np.zeros((0, 3), dtype=np.float32)
    mean_abs_action = np.mean(np.abs(action_array), axis=0) if len(action_array) > 0 else np.array([0.0, 0.0, 0.0], dtype=np.float32)
    mean_action_vy_vx_ratio = float(mean_abs_action[1] / max(float(mean_abs_action[0]), 1e-6))
    mean_speed = float(np.mean(body_speeds)) if body_speeds else 0.0
    if success and mean_speed < args_cli.min_body_speed:
        success = False
        terminal_reason = "too_slow"
    if success and mean_track_error > args_cli.max_mean_track_error:
        success = False
        terminal_reason = "quality_mean_track_error"
    if success and p90_track_error > args_cli.max_p90_track_error:
        success = False
        terminal_reason = "quality_p90_track_error"
    if success and max_track_error > args_cli.max_peak_track_error:
        success = False
        terminal_reason = "quality_peak_track_error"
    if success and frames_over_015_ratio > args_cli.max_frames_over_015_ratio:
        success = False
        terminal_reason = "quality_frames_over_015_ratio"
    if success and mean_image_std < args_cli.min_image_std:
        success = False
        terminal_reason = "quality_flat_rgb"
    if success and mean_action_vy_vx_ratio > args_cli.max_mean_action_vy_vx_ratio:
        success = False
        terminal_reason = "quality_lateral_action_ratio"

    return EpisodeResult(
        direction=direction,
        task_name=direction,
        task_index=TASK_INDEX[direction],
        frames=frames,
        success=success,
        terminal_reason=terminal_reason,
        final_lap_progress=lap_progress,
        mean_track_error=mean_track_error,
        p90_track_error=p90_track_error,
        max_track_error=max_track_error,
        frames_over_010_ratio=frames_over_010_ratio,
        frames_over_015_ratio=frames_over_015_ratio,
        mean_image_std=mean_image_std,
        min_image_std=min_image_std,
        mean_abs_action_vx=float(mean_abs_action[0]),
        mean_abs_action_vy=float(mean_abs_action[1]),
        mean_abs_action_wz=float(mean_abs_action[2]),
        mean_action_vy_vx_ratio=mean_action_vy_vx_ratio,
        mean_speed=mean_speed,
        duration_s=duration_s,
    )


def build_session_name() -> str:
    return args_cli.session_name or datetime.utcnow().strftime("session_%Y%m%d_%H%M%S")


def main() -> None:
    direction_cycle = parse_direction_cycle(args_cli.directions)
    control_substeps = steps_per_control(args_cli.physics_dt, args_cli.control_hz)
    if args_cli.render_interval > 0:
        render_interval = args_cli.render_interval
    else:
        livestream_enabled = bool(getattr(args_cli, "livestream", 0))
        render_interval = control_substeps if args_cli.headless and not livestream_enabled else 1
    scene_cfg = SquareTrackSceneCfg(
        square_half_extent=args_cli.square_half_extent,
        floor_half_extent=args_cli.floor_half_extent,
        tape_width=args_cli.tape_width,
        wall_height=args_cli.wall_height,
        wall_thickness=args_cli.wall_thickness,
    )
    limits = ControlLimits(max_vx=args_cli.max_vx, max_vy=args_cli.max_vy, max_wz=args_cli.max_wz)
    teacher_cfg = TeacherControllerCfg(
        target_speed=args_cli.target_speed,
        lookahead_distance=args_cli.lookahead_distance,
        boundary_gain=args_cli.boundary_gain,
        heading_gain=args_cli.heading_gain,
        cross_track_heading_gain=args_cli.cross_track_heading_gain,
        lateral_gain=args_cli.lateral_gain,
        corner_blend_distance=args_cli.corner_blend_distance,
        corner_slowdown_distance=args_cli.corner_slowdown_distance,
        max_lateral_speed=args_cli.max_lateral_speed,
        track_error_slowdown=args_cli.track_error_slowdown,
        min_tracking_scale=args_cli.min_tracking_scale,
        heading_slowdown_angle=args_cli.heading_slowdown_angle,
        strafe_suppression_angle=args_cli.strafe_suppression_angle,
    )
    session_name = build_session_name()
    max_attempts = args_cli.max_attempts if args_cli.max_attempts > 0 else max(args_cli.num_episodes * 3, args_cli.num_episodes + 2)

    sim_cfg = sim_utils.SimulationCfg(dt=args_cli.physics_dt, render_interval=render_interval, device=args_cli.device)
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
    teacher = SquareTrackTeacher(scene_cfg=scene_cfg, limits=limits, controller_cfg=teacher_cfg, device=robot.device)
    omega_tracker = None
    if args_cli.enable_omega_feedback:
        omega_tracker = OmegaTracker(
            robot.num_instances,
            robot.device,
            OmegaTrackerCfg(
                feedback_gain=args_cli.omega_feedback_gain,
                measurement_alpha=args_cli.omega_measure_alpha,
                command_limit=max(args_cli.max_wz, 2.0),
            ),
        )
    writer = CNNSessionWriter(
        output_root=Path(args_cli.output_dir),
        session_name=session_name,
        dataset_name=args_cli.dataset_name,
        fps=args_cli.control_hz,
        image_width=args_cli.image_width,
        image_height=args_cli.image_height,
        episode_time_s=args_cli.episode_time_s,
        control_hz=args_cli.control_hz,
        physics_dt=args_cli.physics_dt,
    )

    stop_flag = StopFlag()
    signal.signal(signal.SIGINT, stop_flag.request)
    signal.signal(signal.SIGTERM, stop_flag.request)

    print()
    print("=" * 60)
    print("  TurboPi Autonomous CNN Recorder")
    print("=" * 60)
    print(f"  TurboPi USD      : {resolve_asset_usd(args_cli.asset_usd)}")
    print(f"  Output session   : {writer.session_dir}")
    print(f"  Directions       : {', '.join(direction_cycle)}")
    print(f"  Requested eps    : {args_cli.num_episodes}")
    print(f"  Max attempts     : {max_attempts}")
    print(f"  Recording rate   : {args_cli.control_hz:.1f} Hz")
    print(f"  Render interval  : {render_interval} physics steps")
    print(f"  Image size       : {args_cli.image_width}x{args_cli.image_height}")
    print(f"  Lap threshold    : {args_cli.lap_completion_threshold:.2f}")
    print(f"  Max track error  : {args_cli.max_track_error:.2f} m")
    print(
        f"  Finish gate      : phase<={args_cli.start_phase_tolerance:.2f}, "
        f"yaw<={args_cli.start_yaw_tolerance:.2f} rad"
    )
    print(
        f"  Stuck timeout    : {args_cli.stuck_timeout_s:.1f}s "
        f"(after {args_cli.stuck_grace_period_s:.1f}s grace)"
    )
    print(f"  Speed floor      : {args_cli.min_body_speed:.2f} m/s")
    print(f"  Quality gates    : mean<={args_cli.max_mean_track_error:.2f} p90<={args_cli.max_p90_track_error:.2f} peak<={args_cli.max_peak_track_error:.2f}")
    print(f"  RGB gate         : mean std >= {args_cli.min_image_std:.1f}")
    print(f"  Lateral gate     : mean |vy| / mean |vx| <= {args_cli.max_mean_action_vy_vx_ratio:.2f}")
    print(f"  Start view       : {active_view}")
    if omega_tracker is not None:
        print(
            f"  Omega ctrl       : enabled (gain={args_cli.omega_feedback_gain:.2f}, "
            f"alpha={args_cli.omega_measure_alpha:.2f})"
        )
    else:
        print("  Omega ctrl       : off")
    print()

    successful = 0
    attempts = 0
    failure_reasons: Counter[str] = Counter()

    try:
        while simulation_app.is_running() and not stop_flag.requested and successful < args_cli.num_episodes and attempts < max_attempts:
            direction = direction_cycle[successful % len(direction_cycle)]
            attempts += 1
            result = run_episode(
                direction=direction,
                sim=sim,
                robot=robot,
                camera=camera,
                wheel_joint_ids=wheel_joint_ids,
                arm_joint_ids=arm_joint_ids,
                viewport=viewport,
                active_view=active_view,
                scene_cfg=scene_cfg,
                teacher=teacher,
                stop_flag=stop_flag,
                omega_tracker=omega_tracker,
                max_omega_command=args_cli.max_wz,
            )

            if result.success and result.frames:
                episode_dir = writer.save_episode(successful, result)
                successful += 1
                print(
                    f"[INFO] Saved episode_{successful - 1:05d} [{direction}] "
                    f"frames={len(result.frames)} lap={result.final_lap_progress:0.3f} "
                    f"mean_err={result.mean_track_error:0.3f} -> {episode_dir}",
                    flush=True,
                )
            else:
                failure_reasons[result.terminal_reason] += 1
                writer.record_failure()
                print(
                    f"[WARN] Rollout attempt {attempts} failed [{direction}] "
                    f"reason={result.terminal_reason} frames={len(result.frames)} "
                    f"lap={result.final_lap_progress:0.3f} mean_err={result.mean_track_error:0.3f}",
                    flush=True,
                )

        if successful < args_cli.num_episodes and not stop_flag.requested:
            print(
                f"[WARN] Stopping after {attempts} attempts with {successful}/{args_cli.num_episodes} saved episodes.",
                flush=True,
            )
    finally:
        try:
            settle_robot(
                sim=sim,
                robot=robot,
                camera=camera,
                wheel_joint_ids=wheel_joint_ids,
                arm_joint_ids=arm_joint_ids,
                steps=max(args_cli.cooldown_steps, 1),
                viewport=viewport,
                active_view=active_view,
                omega_tracker=omega_tracker,
                max_omega_command=args_cli.max_wz,
            )
        except Exception:
            pass
        print()
        print(f"[INFO] Session complete    : {writer.session_dir}", flush=True)
        print(f"[INFO] Saved episodes      : {successful}", flush=True)
        print(f"[INFO] Failed attempts     : {sum(failure_reasons.values())}", flush=True)
        if failure_reasons:
            print(f"[INFO] Failure breakdown   : {dict(failure_reasons)}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
