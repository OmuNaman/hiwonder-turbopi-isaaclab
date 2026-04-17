"""Run simple TurboPi physics diagnostics for forward/strafe/yaw commands."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure how the TurboPi responds to simple body-twist commands.")
parser.add_argument("--asset_usd", type=str, default=None, help="Optional override for the TurboPi USD.")
parser.add_argument("--scene", type=str, choices=("basic", "square", "straight"), default="square", help="Scene to use for the test.")
parser.add_argument("--direction", type=str, default="clockwise", help="Start pose orientation. Square: clockwise|counterclockwise. Straight: forward|reverse.")
parser.add_argument("--test_duration", type=float, default=2.5, help="Seconds to hold each constant command.")
parser.add_argument("--settle_duration", type=float, default=0.8, help="Seconds to settle at zero command before each test.")
parser.add_argument("--vx_test", type=float, default=0.20, help="Forward test command in m/s.")
parser.add_argument("--vy_test", type=float, default=0.15, help="Strafe test command in m/s.")
parser.add_argument("--wz_test", type=float, default=0.80, help="Yaw test command in rad/s.")
parser.add_argument("--output", type=str, default="", help="Optional JSON file to write the summary to.")
parser.add_argument("--omega_feedback", action="store_true", help="Enable closed-loop omega compensation during the test.")
parser.add_argument("--omega_feedback_gain", type=float, default=0.6, help="Yaw-rate feedback gain when omega compensation is enabled (additive on top of desired wz).")
parser.add_argument("--omega_measure_alpha", type=float, default=0.2, help="EMA factor for measured omega when compensation is enabled.")
parser.add_argument("--no_rollers", action="store_true", help="Skip procedural mecanum roller generation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils.math import euler_xyz_from_quat

from common import (
    OmegaTracker,
    OmegaTrackerCfg,
    design_basic_scene,
    get_arm_joint_ids,
    get_wheel_joint_ids,
    hold_arm_posture,
    reset_robot,
    reset_robot_pose,
    spawn_turbopi,
    twist_to_wheel_targets,
)
from square_loop import SquareTrackSceneCfg, design_square_loop_scene
from square_loop import start_pose_for_direction as square_start_pose_for_direction
from straight_line import StraightLineSceneCfg, design_straight_line_scene
from straight_line import start_pose_for_direction as straight_start_pose_for_direction


@dataclass(frozen=True)
class CommandCase:
    name: str
    command: tuple[float, float, float]


@dataclass(frozen=True)
class CommandResult:
    name: str
    target_vx: float
    target_vy: float
    target_wz: float
    mean_vx: float
    mean_vy: float
    mean_wz: float
    mean_heading_rate: float
    mean_speed_xy: float
    vx_ratio: float
    vy_ratio: float
    wz_ratio: float
    delta_x: float
    delta_y: float
    delta_yaw: float
    target_wheel_lf: float
    target_wheel_lb: float
    target_wheel_rf: float
    target_wheel_rb: float
    mean_wheel_lf: float
    mean_wheel_lb: float
    mean_wheel_rf: float
    mean_wheel_rb: float


def safe_ratio(measured: float, target: float) -> float:
    if abs(target) < 1e-6:
        return 0.0
    return measured / target


def apply_body_command(robot, wheel_joint_ids: list[int], arm_joint_ids: list[int], command_t: torch.Tensor) -> None:
    wheel_targets = twist_to_wheel_targets(command_t.view(1, 3), robot.device)
    robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_joint_ids)
    hold_arm_posture(robot, arm_joint_ids)
    robot.write_data_to_sim()


def run_for_duration(
    *,
    sim: sim_utils.SimulationContext,
    robot,
    wheel_joint_ids: list[int],
    arm_joint_ids: list[int],
    command: tuple[float, float, float],
    duration_s: float,
    physics_dt: float,
    omega_tracker: OmegaTracker | None,
) -> dict[str, np.ndarray | float]:
    command_t = torch.tensor(command, dtype=torch.float32, device=robot.device)
    num_steps = max(1, int(round(duration_s / physics_dt)))
    start_pos = robot.data.root_pos_w[0].detach().cpu().numpy().copy()
    start_yaw = float(torch.atan2(
        2.0 * (robot.data.root_quat_w[0, 0] * robot.data.root_quat_w[0, 3] + robot.data.root_quat_w[0, 1] * robot.data.root_quat_w[0, 2]),
        1.0 - 2.0 * (robot.data.root_quat_w[0, 2] ** 2 + robot.data.root_quat_w[0, 3] ** 2),
    ).item())

    body_vel_samples: list[np.ndarray] = []
    yaw_rate_samples: list[float] = []
    yaw_samples: list[float] = []
    wheel_target_samples: list[np.ndarray] = []
    wheel_vel_samples: list[np.ndarray] = []
    for _ in range(num_steps):
        if not sim.is_playing():
            sim.play()
        applied_command_t = command_t.view(1, 3)
        if omega_tracker is not None:
            applied_command_t = omega_tracker.compensate(
                applied_command_t,
                robot.data.root_ang_vel_b[:, 2],
                dt=physics_dt,
                command_limit=max(abs(float(command[2])), 2.0),
            )
        wheel_targets = twist_to_wheel_targets(applied_command_t, robot.device)
        robot.set_joint_velocity_target(wheel_targets, joint_ids=wheel_joint_ids)
        hold_arm_posture(robot, arm_joint_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(physics_dt)
        body_vel = robot.data.root_lin_vel_b[0].detach().cpu().numpy().copy()
        body_vel_samples.append(body_vel)
        yaw_rate_samples.append(float(robot.data.root_ang_vel_b[0, 2].item()))
        _, _, yaw_t = euler_xyz_from_quat(robot.data.root_quat_w)
        yaw_samples.append(float(yaw_t[0].item()))
        wheel_target_samples.append(wheel_targets[0].detach().cpu().numpy().copy())
        wheel_vel_samples.append(robot.data.joint_vel[0, wheel_joint_ids].detach().cpu().numpy().copy())

    end_pos = robot.data.root_pos_w[0].detach().cpu().numpy().copy()
    end_yaw = float(torch.atan2(
        2.0 * (robot.data.root_quat_w[0, 0] * robot.data.root_quat_w[0, 3] + robot.data.root_quat_w[0, 1] * robot.data.root_quat_w[0, 2]),
        1.0 - 2.0 * (robot.data.root_quat_w[0, 2] ** 2 + robot.data.root_quat_w[0, 3] ** 2),
    ).item())

    return {
        "body_vel_samples": np.asarray(body_vel_samples, dtype=np.float32),
        "yaw_rate_samples": np.asarray(yaw_rate_samples, dtype=np.float32),
        "yaw_samples": np.asarray(yaw_samples, dtype=np.float32),
        "wheel_target_samples": np.asarray(wheel_target_samples, dtype=np.float32),
        "wheel_vel_samples": np.asarray(wheel_vel_samples, dtype=np.float32),
        "delta_pos": end_pos - start_pos,
        "delta_yaw": end_yaw - start_yaw,
        "physics_dt": physics_dt,
    }


def summarize_case(name: str, command: tuple[float, float, float], raw_result: dict[str, np.ndarray | float]) -> CommandResult:
    body_vel_samples = raw_result["body_vel_samples"]
    yaw_rate_samples = raw_result["yaw_rate_samples"]
    tail_start = len(body_vel_samples) // 2
    tail_body = body_vel_samples[tail_start:]
    tail_yaw = yaw_rate_samples[tail_start:]
    tail_heading = np.unwrap(raw_result["yaw_samples"][tail_start:])
    tail_wheel_targets = raw_result["wheel_target_samples"][tail_start:]
    tail_wheel_vels = raw_result["wheel_vel_samples"][tail_start:]
    physics_dt = float(raw_result["physics_dt"])

    mean_vx = float(np.mean(tail_body[:, 0]))
    mean_vy = float(np.mean(tail_body[:, 1]))
    mean_wz = float(np.mean(tail_yaw))
    mean_heading_rate = 0.0
    if len(tail_heading) >= 2:
        mean_heading_rate = float((tail_heading[-1] - tail_heading[0]) / max(len(tail_heading) - 1, 1) / physics_dt)
    mean_speed_xy = float(np.mean(np.linalg.norm(tail_body[:, :2], axis=1)))
    delta_pos = np.asarray(raw_result["delta_pos"], dtype=np.float32)
    delta_yaw = float(raw_result["delta_yaw"])

    return CommandResult(
        name=name,
        target_vx=float(command[0]),
        target_vy=float(command[1]),
        target_wz=float(command[2]),
        mean_vx=mean_vx,
        mean_vy=mean_vy,
        mean_wz=mean_wz,
        mean_heading_rate=mean_heading_rate,
        mean_speed_xy=mean_speed_xy,
        vx_ratio=safe_ratio(mean_vx, command[0]),
        vy_ratio=safe_ratio(mean_vy, command[1]),
        wz_ratio=safe_ratio(mean_wz, command[2]),
        delta_x=float(delta_pos[0]),
        delta_y=float(delta_pos[1]),
        delta_yaw=delta_yaw,
        target_wheel_lf=float(np.mean(tail_wheel_targets[:, 0])),
        target_wheel_lb=float(np.mean(tail_wheel_targets[:, 1])),
        target_wheel_rf=float(np.mean(tail_wheel_targets[:, 2])),
        target_wheel_rb=float(np.mean(tail_wheel_targets[:, 3])),
        mean_wheel_lf=float(np.mean(tail_wheel_vels[:, 0])),
        mean_wheel_lb=float(np.mean(tail_wheel_vels[:, 1])),
        mean_wheel_rf=float(np.mean(tail_wheel_vels[:, 2])),
        mean_wheel_rb=float(np.mean(tail_wheel_vels[:, 3])),
    )


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, render_interval=1, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    square_cfg = SquareTrackSceneCfg()
    straight_cfg = StraightLineSceneCfg()
    if args_cli.scene == "square":
        design_square_loop_scene(square_cfg)
    elif args_cli.scene == "straight":
        design_straight_line_scene(straight_cfg)
    else:
        design_basic_scene()
    robot = spawn_turbopi(asset_usd=args_cli.asset_usd, add_rollers=not args_cli.no_rollers)

    sim.reset()
    if args_cli.scene == "square":
        start_position, start_yaw = square_start_pose_for_direction(square_cfg, args_cli.direction)
        reset_robot_pose(robot, position=start_position, yaw=start_yaw)
    elif args_cli.scene == "straight":
        direction = args_cli.direction if args_cli.direction in ("forward", "reverse") else "forward"
        start_position, start_yaw = straight_start_pose_for_direction(straight_cfg, direction)
        reset_robot_pose(robot, position=start_position, yaw=start_yaw)
    else:
        reset_robot(robot)
    sim.play()

    wheel_joint_ids = get_wheel_joint_ids(robot)
    arm_joint_ids = get_arm_joint_ids(robot)
    physics_dt = float(sim_cfg.dt)
    settle_steps = max(1, int(round(args_cli.settle_duration / physics_dt)))

    cases = [
        CommandCase(name="forward", command=(args_cli.vx_test, 0.0, 0.0)),
        CommandCase(name="strafe", command=(0.0, args_cli.vy_test, 0.0)),
        CommandCase(name="yaw", command=(0.0, 0.0, args_cli.wz_test)),
    ]

    results: list[CommandResult] = []
    omega_tracker = None
    if args_cli.omega_feedback:
        omega_tracker = OmegaTracker(
            robot.num_instances,
            robot.device,
            OmegaTrackerCfg(
                feedback_gain=args_cli.omega_feedback_gain,
                measurement_alpha=args_cli.omega_measure_alpha,
                command_limit=max(abs(args_cli.wz_test), 2.0),
            ),
        )
    for case in cases:
        if args_cli.scene == "square":
            start_position, start_yaw = square_start_pose_for_direction(square_cfg, args_cli.direction)
            reset_robot_pose(robot, position=start_position, yaw=start_yaw)
        elif args_cli.scene == "straight":
            direction = args_cli.direction if args_cli.direction in ("forward", "reverse") else "forward"
            start_position, start_yaw = straight_start_pose_for_direction(straight_cfg, direction)
            reset_robot_pose(robot, position=start_position, yaw=start_yaw)
        else:
            reset_robot(robot)
        if omega_tracker is not None:
            omega_tracker.reset()

        zero_command = torch.zeros(3, dtype=torch.float32, device=robot.device)
        for _ in range(settle_steps):
            apply_body_command(robot, wheel_joint_ids, arm_joint_ids, zero_command)
            sim.step()
            robot.update(physics_dt)

        raw_result = run_for_duration(
            sim=sim,
            robot=robot,
            wheel_joint_ids=wheel_joint_ids,
            arm_joint_ids=arm_joint_ids,
            command=case.command,
            duration_s=args_cli.test_duration,
            physics_dt=physics_dt,
            omega_tracker=omega_tracker,
        )
        result = summarize_case(case.name, case.command, raw_result)
        results.append(result)

        print(
            f"[DIAG] {result.name:7s} target=[{result.target_vx: .3f}, {result.target_vy: .3f}, {result.target_wz: .3f}] "
            f"mean=[{result.mean_vx: .3f}, {result.mean_vy: .3f}, {result.mean_wz: .3f}] "
            f"heading_rate={result.mean_heading_rate: .3f} "
            f"ratio=[{result.vx_ratio: .2f}, {result.vy_ratio: .2f}, {result.wz_ratio: .2f}] "
            f"delta=[{result.delta_x: .3f}, {result.delta_y: .3f}, {result.delta_yaw: .3f}] "
            f"wheel_target=[{result.target_wheel_lf: .2f}, {result.target_wheel_lb: .2f}, {result.target_wheel_rf: .2f}, {result.target_wheel_rb: .2f}] "
            f"wheel_vel=[{result.mean_wheel_lf: .2f}, {result.mean_wheel_lb: .2f}, {result.mean_wheel_rf: .2f}, {result.mean_wheel_rb: .2f}]"
        )

    payload = {
        "scene": args_cli.scene,
        "direction": args_cli.direction,
        "no_rollers": bool(args_cli.no_rollers),
        "omega_feedback": bool(args_cli.omega_feedback),
        "test_duration": args_cli.test_duration,
        "settle_duration": args_cli.settle_duration,
        "results": [asdict(result) for result in results],
    }

    if args_cli.output:
        output_path = Path(args_cli.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[DIAG] Wrote summary to {output_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
