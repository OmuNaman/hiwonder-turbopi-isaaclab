"""Standalone TurboPi straight-line viewer for Isaac Lab."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Launch the standalone TurboPi inside a straight-line collection arena.")
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
    choices=("forward", "reverse"),
    default="forward",
    help="Start pose orientation to preview.",
)
parser.add_argument(
    "--duration",
    type=float,
    default=0.0,
    help="Optional simulation duration in seconds. Use 0 to run until the app is closed.",
)
parser.add_argument("--line_half_length", type=float, default=1.00, help="Half-length of the taped centerline in meters.")
parser.add_argument("--lane_half_width", type=float, default=0.40, help="Half-width of the lane in meters.")
parser.add_argument("--end_margin", type=float, default=0.30, help="Extra floor past each end before the walls in meters.")
parser.add_argument("--tape_width", type=float, default=0.08, help="Width of the centerline tape in meters.")
parser.add_argument("--wall_height", type=float, default=0.55, help="Arena wall height in meters.")
parser.add_argument("--wall_thickness", type=float, default=0.04, help="Arena wall thickness in meters.")
parser.add_argument("--no_rollers", action="store_true", help="Skip procedural mecanum roller generation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils

from common import (
    TURBOPI_URDF,
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
from straight_line import (
    StraightLineSceneCfg,
    design_straight_line_scene,
    start_pose_for_direction,
)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=1, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    scene_cfg = StraightLineSceneCfg(
        line_half_length=args_cli.line_half_length,
        lane_half_width=args_cli.lane_half_width,
        end_margin=args_cli.end_margin,
        tape_width=args_cli.tape_width,
        wall_height=args_cli.wall_height,
        wall_thickness=args_cli.wall_thickness,
    )
    design_straight_line_scene(scene_cfg)
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)

    sim.reset()
    start_position, start_yaw = start_pose_for_direction(scene_cfg, args_cli.direction)
    reset_robot_pose(robot, position=start_position, yaw=start_yaw)
    sim.play()

    wheel_joint_ids = get_wheel_joint_ids(robot)
    arm_joint_ids = get_arm_joint_ids(robot)
    idle_targets = twist_to_wheel_targets(torch.zeros((robot.num_instances, 3), device=robot.device), robot.device)

    viewport = get_viewport()
    active_view = activate_view_mode(args_cli.view, sim, robot, viewport)

    print(f"[INFO] TurboPi USD       : {resolve_asset_usd(args_cli.asset_usd)}")
    print(f"[INFO] TurboPi URDF      : {TURBOPI_URDF}")
    print(f"[INFO] Lane half length  : {scene_cfg.line_half_length:.2f} m")
    print(f"[INFO] Lane half width   : {scene_cfg.lane_half_width:.2f} m")
    print(f"[INFO] Start direction   : {args_cli.direction}")
    print(f"[INFO] Initial view      : {active_view}")
    if viewport is None and args_cli.view != "overview":
        print("[INFO] No interactive viewport is available in this launch mode, so the script fell back to overview.")

    sim_dt = float(sim_cfg.dt)
    elapsed = 0.0

    while simulation_app.is_running():
        if not sim.is_playing():
            sim.play()

        robot.set_joint_velocity_target(idle_targets, joint_ids=wheel_joint_ids)
        hold_arm_posture(robot, arm_joint_ids)
        robot.write_data_to_sim()

        sim.step()
        robot.update(sim_dt)

        if active_view == "chase":
            update_chase_camera(robot, viewport)

        elapsed += sim_dt
        if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
