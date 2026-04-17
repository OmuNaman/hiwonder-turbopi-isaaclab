"""Straight-line collection arena and teacher for the TurboPi.

One episode = one traversal of the lane from the start end to the finish end.
Reuses the same TurboPi articulation, camera sensor, and CNN dataset writer
as the square-loop flow, but uses a simpler scene and a simpler teacher so
the robot mostly drives forward and the data focuses on visual lane
following.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse

from common import ROBOT_CAMERA_PATH

DirectionName = Literal["forward", "reverse"]

DIRECTION_TO_SIGN: dict[DirectionName, float] = {
    "forward": +1.0,
    "reverse": -1.0,
}

TASK_INDEX: dict[DirectionName, int] = {
    "forward": 0,
    "reverse": 1,
}

TASKS: tuple[DirectionName, ...] = ("forward", "reverse")


@dataclass(frozen=True)
class StraightLineSceneCfg:
    """Geometry and visuals for the straight-line collection arena."""

    # Half-length of the taped centerline along body-x direction.
    # Robot starts at -line_half_length and exits at +line_half_length.
    line_half_length: float = 1.00
    # Half-width of the lane (distance from centerline to arena wall).
    lane_half_width: float = 0.40
    # Margin past the ends of the tape before the arena walls close the lane.
    end_margin: float = 0.30
    tape_width: float = 0.08
    wall_height: float = 0.55
    wall_thickness: float = 0.04
    floor_z: float = 0.001
    tape_z: float = 0.003
    floor_color: tuple[float, float, float] = (0.14, 0.14, 0.16)
    tape_color: tuple[float, float, float] = (0.95, 0.95, 0.94)
    wall_color: tuple[float, float, float] = (0.10, 0.10, 0.12)
    start_height: float = 0.04


@dataclass(frozen=True)
class StraightControlLimits:
    """Normalisation ceiling used by the recorder and the teacher."""

    max_vx: float = 0.45
    max_vy: float = 0.35
    max_wz: float = 2.00

    def as_tensor(self, device: str | torch.device) -> torch.Tensor:
        return torch.tensor([self.max_vx, self.max_vy, self.max_wz], dtype=torch.float32, device=device)


@dataclass(frozen=True)
class StraightTeacherCfg:
    """Simple forward-driving teacher that corrects heading and lateral drift."""

    target_speed: float = 0.20
    min_forward_speed: float = 0.05
    # Lateral and heading gains are in body frame. Signs match ROS convention
    # (+vy = left, +wz = CCW). Body-left lateral error contributes +wz so the
    # robot turns its nose back toward the line; once near the line the small
    # lateral command nudges it the rest of the way. Gains picked so the
    # discrete heading loop (control rate 10 Hz, plant gain ~1x) has its
    # poles on the positive real axis, avoiding the underdamped ring that
    # the square teacher exhibited with heading_gain=5.4.
    heading_gain: float = 2.2
    cross_track_heading_gain: float = 0.8
    lateral_gain: float = 0.35
    max_lateral_speed: float = 0.045
    strafe_suppression_angle: float = 0.35
    heading_slowdown_angle: float = 0.70
    track_error_slowdown: float = 0.35
    min_tracking_scale: float = 0.55
    command_filter_alpha_xy: float = 0.30
    command_filter_alpha_wz: float = 0.30


@dataclass(frozen=True)
class LineObservation:
    """Compact robot/line state used by the recorder and teacher."""

    position_w: torch.Tensor
    # Unsigned perpendicular distance from the centerline.
    track_error: float
    # Signed progress along the intended travel direction, in meters
    # relative to the start. Goes from 0 at the start pose to roughly
    # 2 * line_half_length at the goal.
    progress: float
    # Signed lateral offset from the centerline in body frame:
    # positive = body-left. Sign matches ROS convention.
    lateral_offset: float
    body_velocity: torch.Tensor
    body_ang_velocity: torch.Tensor
    height: float
    has_nan: bool


def direction_sign(direction: DirectionName) -> float:
    return DIRECTION_TO_SIGN[direction]


def start_pose_for_direction(
    scene_cfg: StraightLineSceneCfg, direction: DirectionName
) -> tuple[tuple[float, float, float], float]:
    """Return the spawn position at the lane start, yawed toward the finish.

    Forward runs start at -line_half_length facing +x (yaw 0).
    Reverse runs start at +line_half_length facing -x (yaw pi).
    """
    sign = direction_sign(direction)
    start_x = -sign * scene_cfg.line_half_length
    yaw = 0.0 if direction == "forward" else math.pi
    return (start_x, 0.0, scene_cfg.start_height), yaw


def end_x_for_direction(scene_cfg: StraightLineSceneCfg, direction: DirectionName) -> float:
    return direction_sign(direction) * scene_cfg.line_half_length


def design_straight_line_scene(scene_cfg: StraightLineSceneCfg) -> None:
    """Spawn a dark floor, a visible taped centerline, and boundary walls."""
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.85,
            restitution=0.0,
        )
    )
    ground_cfg.func("/World/ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2600.0, color=(0.78, 0.78, 0.78))
    light_cfg.func("/World/Light", light_cfg)

    floor_length = 2.0 * (scene_cfg.line_half_length + scene_cfg.end_margin)
    floor_width = 2.0 * (scene_cfg.lane_half_width + 0.5 * scene_cfg.wall_thickness)

    floor_cfg = sim_utils.CuboidCfg(
        size=(floor_length, floor_width, 0.002),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=scene_cfg.floor_color, roughness=0.95),
    )
    floor_cfg.func("/World/StraightLine/Floor", floor_cfg, translation=(0.0, 0.0, scene_cfg.floor_z))

    tape_length = 2.0 * scene_cfg.line_half_length + scene_cfg.tape_width
    tape_cfg = sim_utils.CuboidCfg(
        size=(tape_length, scene_cfg.tape_width, 0.002),
        collision_props=None,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=scene_cfg.tape_color, roughness=0.65),
    )
    tape_cfg.func("/World/StraightLine/TapeCenter", tape_cfg, translation=(0.0, 0.0, scene_cfg.tape_z))

    wall_half_y = scene_cfg.lane_half_width + 0.5 * scene_cfg.wall_thickness
    wall_z = 0.5 * scene_cfg.wall_height

    long_wall_cfg = sim_utils.CuboidCfg(
        size=(floor_length + scene_cfg.wall_thickness, scene_cfg.wall_thickness, scene_cfg.wall_height),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=scene_cfg.wall_color, roughness=0.90),
    )
    long_wall_cfg.func(
        "/World/StraightLine/WallLeft",
        long_wall_cfg,
        translation=(0.0, +wall_half_y, wall_z),
    )
    long_wall_cfg.func(
        "/World/StraightLine/WallRight",
        long_wall_cfg,
        translation=(0.0, -wall_half_y, wall_z),
    )

    end_wall_cfg = sim_utils.CuboidCfg(
        size=(scene_cfg.wall_thickness, 2.0 * wall_half_y + scene_cfg.wall_thickness, scene_cfg.wall_height),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=scene_cfg.wall_color, roughness=0.90),
    )
    end_x = 0.5 * floor_length + 0.5 * scene_cfg.wall_thickness
    end_wall_cfg.func(
        "/World/StraightLine/WallBack",
        end_wall_cfg,
        translation=(-end_x, 0.0, wall_z),
    )
    end_wall_cfg.func(
        "/World/StraightLine/WallFront",
        end_wall_cfg,
        translation=(+end_x, 0.0, wall_z),
    )


def build_robot_camera_sensor(*, width: int, height: int) -> Camera:
    """Attach a camera sensor to the existing robot camera prim."""
    camera_cfg = CameraCfg(
        prim_path=ROBOT_CAMERA_PATH,
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=None,
    )
    return Camera(camera_cfg)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def observe_line_state(robot, scene_cfg: StraightLineSceneCfg, direction: DirectionName) -> LineObservation:
    """Read the current robot state in the straight-line frame."""
    root_pos_w = robot.data.root_pos_w[0].detach().clone()
    sign = direction_sign(direction)
    # Signed progress along the intended direction, measured from start pose.
    progress = float((sign * root_pos_w[0] + scene_cfg.line_half_length).item())
    # Perpendicular distance from the centerline is just |y|.
    y_world = float(root_pos_w[1].item())
    track_error = abs(y_world)

    # Lateral offset in the body frame: positive means the centerline is to
    # the robot's LEFT (+y_body). For forward runs the body x matches world x,
    # so body_y = world_y. For reverse runs the robot faces -x, and body_y is
    # mirrored relative to world. Compute from the actual quaternion so the
    # value stays correct even while the robot is yawed slightly.
    quat_w = robot.data.root_quat_w
    offset_w = torch.zeros((1, 3), dtype=torch.float32, device=robot.device)
    offset_w[0, 1] = -y_world  # vector from robot to nearest point on centerline (y=0)
    offset_b = quat_apply_inverse(quat_w, offset_w)
    lateral_offset = float(offset_b[0, 1].item())

    body_velocity = robot.data.root_lin_vel_b[0, :2].detach().clone()
    body_ang_velocity = robot.data.root_ang_vel_b[0, 2:3].detach().clone()
    height = float(root_pos_w[2].item())
    has_nan = bool(torch.isnan(robot.data.root_pos_w).any().item())
    return LineObservation(
        position_w=root_pos_w,
        track_error=track_error,
        progress=progress,
        lateral_offset=lateral_offset,
        body_velocity=body_velocity,
        body_ang_velocity=body_ang_velocity,
        height=height,
        has_nan=has_nan,
    )


class StraightLineTeacher:
    """Deterministic teacher that drives the robot straight down the lane."""

    def __init__(
        self,
        *,
        scene_cfg: StraightLineSceneCfg,
        limits: StraightControlLimits,
        controller_cfg: StraightTeacherCfg,
        device: str | torch.device,
    ):
        self.scene_cfg = scene_cfg
        self.limits = limits
        self.cfg = controller_cfg
        self.device = device
        self.max_command = limits.as_tensor(device)
        self.previous_action = torch.zeros(3, dtype=torch.float32, device=device)
        self.direction: DirectionName = "forward"

    def reset(self, direction: DirectionName) -> None:
        self.direction = direction
        self.previous_action.zero_()

    def compute_action(self, robot) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
        """Return previous normalized action, current normalized action, and body-twist command."""
        sign = direction_sign(self.direction)
        quat_w = robot.data.root_quat_w
        _, _, yaw = euler_xyz_from_quat(quat_w)
        # Desired heading is +x for forward and +pi (or -pi) for reverse.
        path_yaw = torch.zeros_like(yaw) if sign > 0.0 else torch.full_like(yaw, math.pi)
        yaw_error = wrap_to_pi(path_yaw - yaw)
        turn_error = torch.abs(yaw_error)

        # Signed world-y (robot position) -> lateral error in body frame.
        y_world = robot.data.root_pos_w[:, 1]
        offset_w = torch.zeros((robot.num_instances, 3), dtype=torch.float32, device=self.device)
        offset_w[:, 1] = -y_world
        offset_b = quat_apply_inverse(quat_w, offset_w)
        lateral_offset_b = offset_b[:, 1]
        track_error = torch.abs(y_world)

        track_speed_scale = torch.clamp(
            1.0 - self.cfg.track_error_slowdown * torch.clamp(track_error, max=0.15) / 0.15,
            min=self.cfg.min_tracking_scale,
            max=1.0,
        )
        heading_speed_scale = torch.clamp(
            1.0 - turn_error / max(self.cfg.heading_slowdown_angle, 1e-4),
            min=0.10,
            max=1.0,
        )
        speed = torch.full_like(yaw, self.cfg.target_speed)
        speed = speed * torch.minimum(track_speed_scale, heading_speed_scale)
        speed = torch.clamp(speed, min=self.cfg.min_forward_speed, max=self.limits.max_vx)

        strafe_scale = torch.clamp(
            1.0 - turn_error / max(self.cfg.strafe_suppression_angle, 1e-4),
            min=0.0,
            max=1.0,
        )
        lateral_command = torch.clamp(
            self.cfg.lateral_gain * lateral_offset_b * strafe_scale,
            min=-self.cfg.max_lateral_speed,
            max=self.cfg.max_lateral_speed,
        )
        forward_command = torch.clamp(
            speed * torch.clamp(torch.cos(yaw_error), min=0.15, max=1.0),
            min=self.cfg.min_forward_speed,
            max=self.limits.max_vx,
        )
        # Positive lateral_offset_b means the centerline is on the body-left,
        # so we want a CCW (+wz) turn to point the nose toward the line.
        desired_wz = self.cfg.heading_gain * yaw_error + self.cfg.cross_track_heading_gain * lateral_offset_b

        command = torch.zeros((robot.num_instances, 3), dtype=torch.float32, device=self.device)
        command[:, 0] = forward_command
        command[:, 1] = lateral_command
        command[:, 2] = desired_wz
        command = torch.clamp(command, min=-self.max_command, max=self.max_command)

        raw_action = torch.clamp(command / self.max_command, -1.0, 1.0)
        action = raw_action.clone()
        keep_xy = 1.0 - self.cfg.command_filter_alpha_xy
        keep_wz = 1.0 - self.cfg.command_filter_alpha_wz
        action[:, :2] = keep_xy * self.previous_action[:2].unsqueeze(0) + self.cfg.command_filter_alpha_xy * raw_action[:, :2]
        action[:, 2] = keep_wz * self.previous_action[2] + self.cfg.command_filter_alpha_wz * raw_action[:, 2]
        action = torch.clamp(action, -1.0, 1.0)

        filtered_command = action * self.max_command
        state = self.previous_action.clone()
        self.previous_action.copy_(action[0])
        info = {
            "track_error": float(track_error[0].item()),
            "lateral_offset_b": float(lateral_offset_b[0].item()),
            "target_yaw_error": float(yaw_error[0].item()),
        }
        return state, action[0].clone(), filtered_command[0].clone(), info
