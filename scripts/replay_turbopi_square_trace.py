"""Replay a recorded TurboPi square teleop trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a recorded TurboPi square teleop trace.")
parser.add_argument("--trace_dir", type=str, required=True, help="Directory containing trace.parquet and trace_info.json.")
parser.add_argument(
    "--mode",
    type=str,
    choices=("pose", "command"),
    default="pose",
    help="Replay exact recorded poses or replay recorded commands through physics.",
)
parser.add_argument(
    "--view",
    type=str,
    choices=("overview", "chase", "robot"),
    default="overview",
    help="Initial viewport mode.",
)
parser.add_argument("--disable_omega_feedback", action="store_true", help="Replay commands directly without closed-loop yaw compensation.")
parser.add_argument("--omega_feedback_gain", type=float, default=2.0, help="Closed-loop yaw-rate feedback gain.")
parser.add_argument("--omega_measure_alpha", type=float, default=0.2, help="EMA factor for measured yaw rate in the compensator.")
parser.add_argument("--asset_usd", type=str, default=None, help="Optional override for the TurboPi USD.")
parser.add_argument("--no_rollers", action="store_true", help="Skip procedural mecanum roller generation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import pandas as pd
import torch

import isaaclab.sim as sim_utils

from common import (
    OmegaTracker,
    OmegaTrackerCfg,
    activate_view_mode,
    get_arm_joint_ids,
    get_viewport,
    get_wheel_joint_ids,
    hold_arm_posture,
    resolve_asset_usd,
    spawn_turbopi,
    twist_to_wheel_targets,
    update_chase_camera,
)
from square_loop import SquareTrackSceneCfg, design_square_loop_scene


def load_trace(trace_dir: Path) -> tuple[pd.DataFrame, dict]:
    info_path = trace_dir / "trace_info.json"
    parquet_path = trace_dir / "trace.parquet"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing trace info: {info_path}")
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing trace parquet: {parquet_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    dataframe = pd.read_parquet(parquet_path)
    if dataframe.empty:
        raise RuntimeError(f"Trace is empty: {parquet_path}")
    return dataframe, info


def main() -> None:
    trace_dir = Path(args_cli.trace_dir).expanduser().resolve()
    trace_df, trace_info = load_trace(trace_dir)

    physics_dt = float(trace_info.get("physics_dt", 1.0 / 120.0))
    sim_cfg = sim_utils.SimulationCfg(dt=physics_dt, render_interval=1, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = SquareTrackSceneCfg(
        square_half_extent=float(trace_info.get("square_half_extent", 0.45)),
        floor_half_extent=float(trace_info.get("floor_half_extent", 1.40)),
        tape_width=float(trace_info.get("tape_width", 0.08)),
        wall_height=float(trace_info.get("wall_height", 0.55)),
        wall_thickness=float(trace_info.get("wall_thickness", 0.04)),
    )
    design_square_loop_scene(scene_cfg)
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)

    sim.reset()
    sim.play()

    wheel_joint_ids = get_wheel_joint_ids(robot)
    arm_joint_ids = get_arm_joint_ids(robot)
    viewport = get_viewport()
    active_view = activate_view_mode(args_cli.view, sim, robot, viewport)
    has_applied_columns = {"applied_command_vx", "applied_command_vy", "applied_command_wz"}.issubset(trace_df.columns)
    omega_tracker = None
    if args_cli.mode == "command" and not has_applied_columns and not args_cli.disable_omega_feedback:
        omega_tracker = OmegaTracker(
            robot.num_instances,
            robot.device,
            OmegaTrackerCfg(
                feedback_gain=args_cli.omega_feedback_gain,
                measurement_alpha=args_cli.omega_measure_alpha,
                command_limit=max(float(trace_info.get("wz_sensitivity", 1.2)), 2.0),
            ),
        )

    print(f"[INFO] Trace dir      : {trace_dir}")
    print(f"[INFO] Replay mode    : {args_cli.mode}")
    print(f"[INFO] TurboPi USD    : {resolve_asset_usd(args_cli.asset_usd)}")
    print(f"[INFO] Trace steps    : {len(trace_df)}")
    print(f"[INFO] Physics dt     : {physics_dt}")
    print(f"[INFO] Initial view   : {active_view}")
    if "square_corners_xy" in trace_info:
        print(f"[INFO] Square route   : {trace_info['square_corners_xy']}")
    if "outer_wall_half_extent" in trace_info:
        print(f"[INFO] Arena walls    : x/y = +/-{float(trace_info['outer_wall_half_extent']):.3f} m")
    if "start_pose" in trace_info:
        print(f"[INFO] Start pose     : {trace_info['start_pose']}")
    if args_cli.mode == "command":
        if has_applied_columns:
            print("[INFO] Command source : applied_command_* columns from the trace")
        elif omega_tracker is not None:
            print(
                f"[INFO] Omega ctrl     : enabled (gain={args_cli.omega_feedback_gain:.2f}, "
                f"alpha={args_cli.omega_measure_alpha:.2f})"
            )

    zero_wheels = torch.zeros((robot.num_instances, 4), dtype=torch.float32, device=robot.device)
    sim_time = 0.0
    last_log_time = -1.0

    try:
        for row_idx, row in trace_df.iterrows():
            if not simulation_app.is_running():
                break
            if not sim.is_playing():
                sim.play()

            if args_cli.mode == "pose":
                root_pose = torch.tensor(
                    [
                        [
                            float(row["root_pos_x"]),
                            float(row["root_pos_y"]),
                            float(row["root_pos_z"]),
                            float(row["root_quat_w"]),
                            float(row["root_quat_x"]),
                            float(row["root_quat_y"]),
                            float(row["root_quat_z"]),
                        ]
                    ],
                    dtype=torch.float32,
                    device=robot.device,
                )
                root_velocity = torch.tensor(
                    [
                        [
                            float(row["root_lin_vel_w_x"]),
                            float(row["root_lin_vel_w_y"]),
                            float(row["root_lin_vel_w_z"]),
                            float(row["root_ang_vel_w_x"]),
                            float(row["root_ang_vel_w_y"]),
                            float(row["root_ang_vel_w_z"]),
                        ]
                    ],
                    dtype=torch.float32,
                    device=robot.device,
                )
                robot.write_root_pose_to_sim(root_pose)
                robot.write_root_velocity_to_sim(root_velocity)
                robot.set_joint_velocity_target(zero_wheels, joint_ids=wheel_joint_ids)
            else:
                if has_applied_columns:
                    command = torch.tensor(
                        [
                            float(row["applied_command_vx"]),
                            float(row["applied_command_vy"]),
                            float(row["applied_command_wz"]),
                        ],
                        dtype=torch.float32,
                        device=robot.device,
                    ).view(1, 3)
                else:
                    desired_command = torch.tensor(
                        [
                            float(row["filtered_command_vx"]),
                            float(row["filtered_command_vy"]),
                            float(row["filtered_command_wz"]),
                        ],
                        dtype=torch.float32,
                        device=robot.device,
                    ).view(1, 3)
                    command = desired_command
                    if omega_tracker is not None:
                        command = omega_tracker.compensate(
                            desired_command,
                            robot.data.root_ang_vel_b[:, 2],
                            dt=physics_dt,
                            command_limit=float(trace_info.get("wz_sensitivity", 1.2)),
                        )
                wheel_targets = twist_to_wheel_targets(command, robot.device)
                robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_joint_ids)

            hold_arm_posture(robot, arm_joint_ids)
            robot.write_data_to_sim()

            sim.step()
            robot.update(physics_dt)

            if active_view == "chase":
                update_chase_camera(robot, viewport)

            sim_time += physics_dt
            if sim_time - last_log_time >= 1.0:
                last_log_time = sim_time
                print(
                    f"[INFO] replay t={sim_time:5.1f}s row={row_idx + 1}/{len(trace_df)} "
                    f"pos=[{float(row['root_pos_x']): .3f}, {float(row['root_pos_y']): .3f}]"
                )
    finally:
        print(f"[INFO] Replay complete: {trace_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
