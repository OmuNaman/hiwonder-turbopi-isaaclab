"""Teleoperate the TurboPi in the square arena and record an exact trace."""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleoperate the TurboPi in the square arena while recording pose/velocity/command traces.")
parser.add_argument("--asset_usd", type=str, default=None, help="Optional override for the TurboPi USD.")
parser.add_argument(
    "--view",
    type=str,
    choices=("overview", "chase", "robot"),
    default="overview",
    help="Initial viewport mode.",
)
parser.add_argument(
    "--direction",
    type=str,
    choices=("clockwise", "counterclockwise"),
    default="clockwise",
    help="Start pose orientation.",
)
parser.add_argument(
    "--duration",
    type=float,
    default=0.0,
    help="Optional simulation duration in seconds. Use 0 to run until the app is closed.",
)
parser.add_argument("--vx", type=float, default=0.35, help="Forward/backward speed in m/s for a full key press.")
parser.add_argument("--vy", type=float, default=0.25, help="Lateral speed in m/s for a full key press.")
parser.add_argument("--wz", type=float, default=1.20, help="Yaw rate in rad/s for a full key press.")
parser.add_argument(
    "--command_filter",
    type=float,
    default=0.25,
    help="First-order smoothing factor in [0, 1]. Higher values track the keyboard more tightly.",
)
parser.add_argument("--disable_omega_feedback", action="store_true", help="Send yaw commands directly without closed-loop compensation.")
parser.add_argument("--omega_feedback_gain", type=float, default=2.0, help="Closed-loop yaw-rate feedback gain.")
parser.add_argument("--omega_measure_alpha", type=float, default=0.2, help="EMA factor for measured yaw rate in the compensator.")
parser.add_argument("--output_dir", type=str, default="data/teleop_traces", help="Root directory for saved traces.")
parser.add_argument("--trace_name", type=str, default=None, help="Optional trace directory name.")
parser.add_argument("--square_half_extent", type=float, default=0.45, help="Half-size of the taped square path in meters.")
parser.add_argument("--floor_half_extent", type=float, default=1.40, help="Half-size of the visible arena floor in meters.")
parser.add_argument("--tape_width", type=float, default=0.08, help="Width of the square path marker in meters.")
parser.add_argument("--wall_height", type=float, default=0.55, help="Outer boundary wall height in meters.")
parser.add_argument("--wall_thickness", type=float, default=0.04, help="Outer boundary wall thickness in meters.")
parser.add_argument("--no_rollers", action="store_true", help="Skip procedural mecanum roller generation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import pandas as pd
import torch

import isaaclab.sim as sim_utils
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.utils.math import euler_xyz_from_quat

from common import (
    OmegaTracker,
    OmegaTrackerCfg,
    TURBOPI_URDF,
    activate_view_mode,
    cycle_view_mode,
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
    SquareTrackSceneCfg,
    compute_square_track_frame,
    design_square_loop_scene,
    outer_wall_half_extent,
    square_corners_xy,
    start_pose_for_direction,
)
from trace_utils import ensure_trace_dir, resolve_parquet_engine, utc_now, write_json


def resolve_trace_name() -> str:
    return args_cli.trace_name or datetime.utcnow().strftime("trace_%Y%m%d_%H%M%S")


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=1, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = SquareTrackSceneCfg(
        square_half_extent=args_cli.square_half_extent,
        floor_half_extent=args_cli.floor_half_extent,
        tape_width=args_cli.tape_width,
        wall_height=args_cli.wall_height,
        wall_thickness=args_cli.wall_thickness,
    )
    design_square_loop_scene(scene_cfg)
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)

    sim.reset()
    start_position, start_yaw = start_pose_for_direction(scene_cfg, args_cli.direction)
    reset_robot_pose(robot, position=start_position, yaw=start_yaw)
    sim.play()

    wheel_joint_ids = get_wheel_joint_ids(robot)
    arm_joint_ids = get_arm_joint_ids(robot)
    filtered_command = torch.zeros((robot.num_instances, 3), dtype=torch.float32, device=robot.device)
    omega_tracker = None
    if not args_cli.disable_omega_feedback:
        omega_tracker = OmegaTracker(
            robot.num_instances,
            robot.device,
            OmegaTrackerCfg(
                feedback_gain=args_cli.omega_feedback_gain,
                measurement_alpha=args_cli.omega_measure_alpha,
                command_limit=max(args_cli.wz, 2.0),
            ),
        )

    viewport = get_viewport()
    state = {"view_mode": activate_view_mode(args_cli.view, sim, robot, viewport)}

    try:
        teleop = Se2Keyboard(
            Se2KeyboardCfg(
                sim_device=args_cli.device,
                v_x_sensitivity=args_cli.vx,
                v_y_sensitivity=args_cli.vy,
                omega_z_sensitivity=args_cli.wz,
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to create the keyboard teleop device. Run with a local GUI session or use --livestream 2."
        ) from exc

    trace_name = resolve_trace_name()
    trace_dir = ensure_trace_dir(Path(args_cli.output_dir), trace_name)
    parquet_engine = resolve_parquet_engine()
    records: list[dict[str, float | int | str | bool]] = []
    reset_count = 0
    route_corners = square_corners_xy(scene_cfg)
    wall_half = outer_wall_half_extent(scene_cfg)
    alpha = float(max(0.0, min(1.0, args_cli.command_filter)))

    def reset_cb() -> None:
        nonlocal reset_count
        reset_count += 1
        reset_robot_pose(robot, position=start_position, yaw=start_yaw)
        filtered_command.zero_()
        if omega_tracker is not None:
            omega_tracker.reset()
        print("[INFO] Robot reset to the trace start pose.")

    def cycle_camera_cb() -> None:
        state["view_mode"] = activate_view_mode(cycle_view_mode(state["view_mode"]), sim, robot, viewport)
        print(f"[INFO] View mode -> {state['view_mode']}")

    def close_cb() -> None:
        simulation_app.close()

    teleop.add_callback("R", reset_cb)
    teleop.add_callback("C", cycle_camera_cb)
    teleop.add_callback("ESCAPE", close_cb)
    teleop.reset()

    print(f"[INFO] TurboPi USD   : {resolve_asset_usd(args_cli.asset_usd)}")
    print(f"[INFO] TurboPi URDF  : {TURBOPI_URDF}")
    print(f"[INFO] Trace dir     : {trace_dir}")
    print(f"[INFO] Start pose    : pos={start_position}, yaw={start_yaw:.3f} rad")
    print(f"[INFO] Square route  : {route_corners}")
    print(f"[INFO] Arena walls   : x/y = +/-{wall_half:.3f} m")
    print(f"[INFO] Initial view  : {state['view_mode']}")
    print("[INFO] Controls      : Up/Down drive, Left/Right strafe, Z/X yaw, R reset, C camera, Esc close.")
    print(
        "[INFO] Command model : raw_command=[vx, vy, wz] in body coordinates "
        f"(m/s, m/s, rad/s), full-key sensitivities=[{args_cli.vx:.3f}, {args_cli.vy:.3f}, {args_cli.wz:.3f}]"
    )
    print(
        f"[INFO] Filter        : filtered = (1-{alpha:.2f}) * prev + {alpha:.2f} * raw, "
        "then converted to wheel joint velocity targets."
    )
    if omega_tracker is not None:
        print(
            f"[INFO] Omega ctrl    : enabled (gain={args_cli.omega_feedback_gain:.2f}, "
            f"alpha={args_cli.omega_measure_alpha:.2f})"
        )
    print("[INFO] Livestream    : click once inside the Isaac viewport before using the keyboard in a remote session.")
    if viewport is None and args_cli.view != "overview":
        print("[INFO] No interactive viewport is available in this launch mode, so the script fell back to overview.")

    sim_dt = float(sim_cfg.dt)
    elapsed = 0.0
    last_log_time = -1.0

    try:
        while simulation_app.is_running():
            if not sim.is_playing():
                sim.play()
                continue

            raw_command = teleop.advance().view(robot.num_instances, 3)
            filtered_command.mul_(1.0 - alpha).add_(alpha * raw_command)
            applied_command = filtered_command
            if omega_tracker is not None:
                applied_command = omega_tracker.compensate(
                    filtered_command,
                    robot.data.root_ang_vel_b[:, 2],
                    dt=sim_dt,
                    command_limit=args_cli.wz,
                )
            wheel_targets = twist_to_wheel_targets(applied_command, robot.device)

            robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_joint_ids)
            hold_arm_posture(robot, arm_joint_ids)
            robot.write_data_to_sim()

            sim.step()
            robot.update(sim_dt)

            if state["view_mode"] == "chase":
                update_chase_camera(robot, viewport)

            root_pos_w = robot.data.root_pos_w[0]
            root_quat_w = robot.data.root_quat_w[0]
            root_lin_vel_w = robot.data.root_lin_vel_w[0]
            root_ang_vel_w = robot.data.root_ang_vel_w[0]
            root_lin_vel_b = robot.data.root_lin_vel_b[0]
            root_ang_vel_b = robot.data.root_ang_vel_b[0]
            _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
            _, _, track_error, track_phase = compute_square_track_frame(
                root_pos_w[:2].unsqueeze(0), scene_cfg.square_half_extent
            )

            records.append(
                {
                    "step_index": len(records),
                    "sim_time": elapsed,
                    "reset_count": reset_count,
                    "view_mode": state["view_mode"],
                    "square_half_extent": scene_cfg.square_half_extent,
                    "root_pos_x": float(root_pos_w[0].item()),
                    "root_pos_y": float(root_pos_w[1].item()),
                    "root_pos_z": float(root_pos_w[2].item()),
                    "root_quat_w": float(root_quat_w[0].item()),
                    "root_quat_x": float(root_quat_w[1].item()),
                    "root_quat_y": float(root_quat_w[2].item()),
                    "root_quat_z": float(root_quat_w[3].item()),
                    "yaw": float(yaw[0].item()),
                    "root_lin_vel_w_x": float(root_lin_vel_w[0].item()),
                    "root_lin_vel_w_y": float(root_lin_vel_w[1].item()),
                    "root_lin_vel_w_z": float(root_lin_vel_w[2].item()),
                    "root_ang_vel_w_x": float(root_ang_vel_w[0].item()),
                    "root_ang_vel_w_y": float(root_ang_vel_w[1].item()),
                    "root_ang_vel_w_z": float(root_ang_vel_w[2].item()),
                    "root_lin_vel_b_x": float(root_lin_vel_b[0].item()),
                    "root_lin_vel_b_y": float(root_lin_vel_b[1].item()),
                    "root_lin_vel_b_z": float(root_lin_vel_b[2].item()),
                    "root_ang_vel_b_x": float(root_ang_vel_b[0].item()),
                    "root_ang_vel_b_y": float(root_ang_vel_b[1].item()),
                    "root_ang_vel_b_z": float(root_ang_vel_b[2].item()),
                    "raw_command_vx": float(raw_command[0, 0].item()),
                    "raw_command_vy": float(raw_command[0, 1].item()),
                    "raw_command_wz": float(raw_command[0, 2].item()),
                    "filtered_command_vx": float(filtered_command[0, 0].item()),
                    "filtered_command_vy": float(filtered_command[0, 1].item()),
                    "filtered_command_wz": float(filtered_command[0, 2].item()),
                    "applied_command_vx": float(applied_command[0, 0].item()),
                    "applied_command_vy": float(applied_command[0, 1].item()),
                    "applied_command_wz": float(applied_command[0, 2].item()),
                    "wheel_target_lf": float(wheel_targets[0, 0].item()),
                    "wheel_target_lb": float(wheel_targets[0, 1].item()),
                    "wheel_target_rf": float(wheel_targets[0, 2].item()),
                    "wheel_target_rb": float(wheel_targets[0, 3].item()),
                    "track_error": float(track_error[0].item()),
                    "track_phase": float(track_phase[0].item()),
                    "is_command_zero": bool(torch.allclose(applied_command[0], torch.zeros(3, device=robot.device), atol=1e-5)),
                }
            )

            elapsed += sim_dt
            if elapsed - last_log_time >= 1.0:
                last_log_time = elapsed
                print(
                    f"[INFO] t={elapsed:5.1f}s pos=[{float(root_pos_w[0]): .3f}, {float(root_pos_w[1]): .3f}] "
                    f"vel_b=[{float(root_lin_vel_b[0]): .3f}, {float(root_lin_vel_b[1]): .3f}, {float(root_ang_vel_b[2]): .3f}] "
                    f"cmd_des=[{float(filtered_command[0, 0]): .3f}, {float(filtered_command[0, 1]): .3f}, {float(filtered_command[0, 2]): .3f}] "
                    f"cmd_app=[{float(applied_command[0, 0]): .3f}, {float(applied_command[0, 1]): .3f}, {float(applied_command[0, 2]): .3f}]"
                )

            if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
                break
    finally:
        dataframe = pd.DataFrame(records)
        parquet_path = trace_dir / "trace.parquet"
        info_path = trace_dir / "trace_info.json"
        if not dataframe.empty:
            dataframe.to_parquet(parquet_path, engine=parquet_engine, index=False)

        write_json(
            info_path,
            {
                "created_at": utc_now(),
                "trace_name": trace_name,
                "direction": args_cli.direction,
                "asset_usd": str(resolve_asset_usd(args_cli.asset_usd)),
                "physics_dt": sim_dt,
                "duration_s": elapsed,
                "num_steps": len(records),
                "square_half_extent": scene_cfg.square_half_extent,
                "square_corners_xy": [list(corner) for corner in route_corners],
                "outer_wall_half_extent": wall_half,
                "floor_half_extent": scene_cfg.floor_half_extent,
                "tape_width": scene_cfg.tape_width,
                "wall_height": scene_cfg.wall_height,
                "wall_thickness": scene_cfg.wall_thickness,
                "start_pose": {
                    "position": list(start_position),
                    "yaw_rad": start_yaw,
                },
                "vx_sensitivity": args_cli.vx,
                "vy_sensitivity": args_cli.vy,
                "wz_sensitivity": args_cli.wz,
                "command_filter": alpha,
                "omega_feedback_enabled": omega_tracker is not None,
                "omega_feedback_gain": args_cli.omega_feedback_gain if omega_tracker is not None else 0.0,
                "omega_measure_alpha": args_cli.omega_measure_alpha if omega_tracker is not None else 0.0,
                "command_semantics": {
                    "raw_command": "[vx, vy, wz] body twist from keyboard",
                    "filtered_command": "first-order filtered body twist",
                    "applied_command": "body twist after yaw-rate compensation, used for wheel targets",
                    "units": {
                        "vx": "m/s",
                        "vy": "m/s",
                        "wz": "rad/s",
                        "wheel_targets": "rad/s",
                    },
                },
                "reset_count": reset_count,
                "parquet_file": parquet_path.name if parquet_path.exists() else None,
            },
        )
        print(f"[INFO] Trace saved    : {trace_dir}")
        print(f"[INFO] Steps recorded : {len(records)}")


if __name__ == "__main__":
    main()
    simulation_app.close()
