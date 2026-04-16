"""Standalone TurboPi viewer for Isaac Lab."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Launch the standalone TurboPi bundle inside Isaac Lab.")
parser.add_argument("--asset_usd", type=str, default=None, help="Optional override for the TurboPi USD.")
parser.add_argument(
    "--view",
    type=str,
    choices=("overview", "chase", "robot"),
    default="overview",
    help="Initial viewport mode.",
)
parser.add_argument(
    "--duration",
    type=float,
    default=0.0,
    help="Optional simulation duration in seconds. Use 0 to run until the app is closed.",
)
parser.add_argument("--no_rollers", action="store_true", help="Skip procedural mecanum roller generation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils

from common import (
    activate_view_mode,
    design_basic_scene,
    get_arm_joint_ids,
    get_viewport,
    get_wheel_joint_ids,
    hold_arm_posture,
    reset_robot,
    resolve_asset_usd,
    spawn_turbopi,
    TURBOPI_URDF,
    twist_to_wheel_targets,
    update_chase_camera,
)


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=1, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    design_basic_scene()
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)

    sim.reset()
    reset_robot(robot)
    sim.play()

    wheel_joint_ids = get_wheel_joint_ids(robot)
    arm_joint_ids = get_arm_joint_ids(robot)
    idle_targets = twist_to_wheel_targets(torch.zeros((robot.num_instances, 3), device=robot.device), robot.device)

    viewport = get_viewport()
    active_view = activate_view_mode(args_cli.view, sim, robot, viewport)

    print(f"[INFO] TurboPi USD   : {resolve_asset_usd(args_cli.asset_usd)}")
    print(f"[INFO] TurboPi URDF  : {TURBOPI_URDF}")
    print(f"[INFO] Initial view  : {active_view}")
    if viewport is None and args_cli.view != "overview":
        print("[INFO] No interactive viewport is available in this launch mode, so the script fell back to overview.")

    sim_dt = sim.get_physics_dt()
    elapsed = 0.0

    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.play()
            continue

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
