from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.utils.math import quat_apply

from mecanum_builder import add_all_mecanum_rollers

ViewMode = Literal["overview", "chase", "robot"]


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = BUNDLE_ROOT / "assets" / "turbopi"
TURBOPI_USD = ASSET_ROOT / "turbopi.usd"
TURBOPI_URDF = ASSET_ROOT / "turbopi_description" / "urdf" / "turbopi.urdf"

ROBOT_PRIM_PATH = "/World/TurboPi"
ROBOT_CAMERA_PATH = f"{ROBOT_PRIM_PATH}/camera_link/RobotCamera"
CHASE_CAMERA_PATH = "/World/TurboPiChaseCamera"
PERSPECTIVE_CAMERA_PATH = "/OmniverseKit_Persp"

CAMERA_LINK_TO_SENSOR_POS = (0.015, 0.0, 0.0)
CAMERA_LINK_TO_SENSOR_ROT = (0.965926, 0.0, -0.258819, 0.0)

WHEEL_RADIUS = 0.033
TRACK_WIDTH = 0.130
WHEEL_BASE = 0.119
LW_SUM_HALF = (WHEEL_BASE + TRACK_WIDTH) / 2.0

WHEEL_FORWARD_SIGN = {
    "wheel_lf_joint": -1.0,
    "wheel_lb_joint": +1.0,
    "wheel_rf_joint": -1.0,
    "wheel_rb_joint": +1.0,
}

VIEW_MODES: tuple[ViewMode, ...] = ("overview", "chase", "robot")


def resolve_asset_usd(asset_usd: str | None = None) -> Path:
    """Return the USD path to load for the standalone TurboPi bundle."""
    usd_path = Path(asset_usd).expanduser().resolve() if asset_usd else TURBOPI_USD
    if not usd_path.is_file():
        raise FileNotFoundError(
            f"TurboPi USD not found: {usd_path}. Expected the bundled asset at {TURBOPI_USD} or an override via"
            " --asset_usd."
        )
    return usd_path


def build_turbopi_cfg(asset_usd: str | None = None, prim_path: str = ROBOT_PRIM_PATH) -> ArticulationCfg:
    """Create the articulation config used by the standalone viewer and teleop scripts."""
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(resolve_asset_usd(asset_usd)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.04)),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["wheel_.*_joint"],
                velocity_limit_sim=35.0,
                effort_limit_sim=0.22,
                stiffness=0.0,
                damping=5.0,
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[23]"],
                velocity_limit_sim=3.0,
                effort_limit_sim=2.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "rollers": ImplicitActuatorCfg(
                joint_names_expr=[".*_roller_.*_joint"],
                velocity_limit_sim=100.0,
                effort_limit_sim=0.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )


def design_basic_scene() -> None:
    """Spawn a simple floor and light rig around the TurboPi."""
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        )
    )
    ground_cfg.func("/World/ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2600.0, color=(0.78, 0.78, 0.78))
    light_cfg.func("/World/Light", light_cfg)


def _set_camera_common_attrs(camera: UsdGeom.Camera) -> None:
    camera.CreateFocalLengthAttr(8.5)
    camera.CreateFocusDistanceAttr(400.0)
    camera.CreateHorizontalApertureAttr(10.0)
    camera.CreateVerticalApertureAttr(7.5)
    camera_prim = camera.GetPrim()
    coi_attr = camera_prim.GetProperty("omni:kit:centerOfInterest")
    if not coi_attr or not coi_attr.IsValid():
        camera_prim.CreateAttribute(
            "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
        ).Set(Gf.Vec3d(0.0, 0.0, -10.0))


def ensure_robot_camera(camera_path: str = ROBOT_CAMERA_PATH) -> str:
    """Create a robot-mounted camera prim under ``camera_link``."""
    stage = get_current_stage()
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera_prim = camera.GetPrim()
    _set_camera_common_attrs(camera)

    xformable = UsdGeom.Xformable(camera_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*CAMERA_LINK_TO_SENSOR_POS))
    xformable.AddOrientOp().Set(Gf.Quatf(*CAMERA_LINK_TO_SENSOR_ROT))
    return camera_path


def ensure_chase_camera(camera_path: str = CHASE_CAMERA_PATH) -> str:
    """Create the external follow camera used by teleop."""
    stage = get_current_stage()
    camera = UsdGeom.Camera.Define(stage, camera_path)
    _set_camera_common_attrs(camera)
    return camera_path


def spawn_turbopi(asset_usd: str | None = None, add_rollers: bool = True) -> Articulation:
    """Spawn the standalone TurboPi articulation and attach its helper cameras."""
    robot = Articulation(build_turbopi_cfg(asset_usd=asset_usd))
    stage = get_current_stage()
    if add_rollers:
        add_all_mecanum_rollers(ROBOT_PRIM_PATH, stage)
    ensure_robot_camera()
    ensure_chase_camera()
    return robot


def set_overview_camera(sim: sim_utils.SimulationContext) -> None:
    """Place the main viewport in a sane overview pose."""
    sim.set_camera_view(eye=[1.9, -1.9, 1.35], target=[0.0, 0.0, 0.08])


def get_viewport():
    """Return the main viewport if the current launch mode exposes one."""
    try:
        from omni.kit.viewport.utility import get_viewport_from_window_name

        return get_viewport_from_window_name("Viewport")
    except Exception:
        return None


def activate_view_mode(
    view_mode: ViewMode, sim: sim_utils.SimulationContext, robot: Articulation, viewport=None
) -> ViewMode:
    """Switch between overview, chase, and robot-mounted camera views."""
    if viewport is None:
        set_overview_camera(sim)
        return "overview"

    if view_mode == "overview":
        viewport.set_active_camera(PERSPECTIVE_CAMERA_PATH)
        set_overview_camera(sim)
        return "overview"

    if view_mode == "robot":
        viewport.set_active_camera(ROBOT_CAMERA_PATH)
        return "robot"

    viewport.set_active_camera(CHASE_CAMERA_PATH)
    update_chase_camera(robot, viewport)
    return "chase"


def cycle_view_mode(current_mode: ViewMode) -> ViewMode:
    """Advance to the next camera mode."""
    current_index = VIEW_MODES.index(current_mode)
    return VIEW_MODES[(current_index + 1) % len(VIEW_MODES)]


def update_chase_camera(
    robot: Articulation,
    viewport,
    camera_path: str = CHASE_CAMERA_PATH,
    eye_offset: tuple[float, float, float] = (-0.8, 0.0, 0.34),
    target_offset: tuple[float, float, float] = (0.22, 0.0, 0.10),
) -> None:
    """Update the chase camera to follow the robot base."""
    if viewport is None:
        return

    from omni.kit.viewport.utility.camera_state import ViewportCameraState

    base_pos = robot.data.root_pos_w[0]
    base_quat = robot.data.root_quat_w[0]

    eye_offset_t = torch.tensor([eye_offset], dtype=torch.float32, device=robot.device)
    target_offset_t = torch.tensor([target_offset], dtype=torch.float32, device=robot.device)

    eye_world = quat_apply(base_quat.unsqueeze(0), eye_offset_t)[0] + base_pos
    target_world = quat_apply(base_quat.unsqueeze(0), target_offset_t)[0] + base_pos

    camera_state = ViewportCameraState(camera_path, viewport)
    camera_state.set_position_world(
        Gf.Vec3d(float(eye_world[0]), float(eye_world[1]), float(eye_world[2])), True
    )
    camera_state.set_target_world(
        Gf.Vec3d(float(target_world[0]), float(target_world[1]), float(target_world[2])), True
    )


def _resolve_joint_ids(robot: Articulation, joint_names: list[str]) -> list[int]:
    joint_ids: list[int] = []
    for joint_name in joint_names:
        ids, _ = robot.find_joints(joint_name)
        joint_ids.append(int(ids[0]))
    return joint_ids


def get_wheel_joint_ids(robot: Articulation) -> list[int]:
    """Return wheel joint ids in a stable left-front to right-back order."""
    return _resolve_joint_ids(robot, ["wheel_lf_joint", "wheel_lb_joint", "wheel_rf_joint", "wheel_rb_joint"])


def get_arm_joint_ids(robot: Articulation) -> list[int]:
    """Return the arm joints that should be held in their default pose."""
    joint_ids, _ = robot.find_joints("joint[23]")
    return [int(joint_id) for joint_id in joint_ids]


def hold_arm_posture(robot: Articulation, arm_joint_ids: list[int]) -> None:
    """Keep the little camera mast from sagging while the base drives around."""
    if not arm_joint_ids:
        return
    robot.set_joint_position_target(robot.data.default_joint_pos[:, arm_joint_ids], joint_ids=arm_joint_ids)


def twist_to_wheel_targets(command: torch.Tensor | list[float] | tuple[float, float, float], device: str) -> torch.Tensor:
    """Convert body twist commands ``(vx, vy, wz)`` into mecanum wheel velocity targets."""
    command_t = torch.as_tensor(command, dtype=torch.float32, device=device)
    if command_t.ndim == 1:
        command_t = command_t.unsqueeze(0)

    vx = command_t[:, 0]
    vy = command_t[:, 1]
    wz = command_t[:, 2]

    omega_lf = (vx - vy - LW_SUM_HALF * wz) / WHEEL_RADIUS
    omega_lb = (vx + vy - LW_SUM_HALF * wz) / WHEEL_RADIUS
    omega_rf = (vx + vy + LW_SUM_HALF * wz) / WHEEL_RADIUS
    omega_rb = (vx - vy + LW_SUM_HALF * wz) / WHEEL_RADIUS

    wheel_targets = torch.stack((omega_lf, omega_lb, omega_rf, omega_rb), dim=-1)
    wheel_signs = torch.tensor(
        [
            WHEEL_FORWARD_SIGN["wheel_lf_joint"],
            WHEEL_FORWARD_SIGN["wheel_lb_joint"],
            WHEEL_FORWARD_SIGN["wheel_rf_joint"],
            WHEEL_FORWARD_SIGN["wheel_rb_joint"],
        ],
        dtype=torch.float32,
        device=device,
    )
    return wheel_targets * wheel_signs


def reset_robot(robot: Articulation, position: tuple[float, float, float] = (0.0, 0.0, 0.04)) -> None:
    """Reset the robot root and joints to a clean startup pose."""
    default_root_state = robot.data.default_root_state.clone()
    default_root_state[:, 0:3] = torch.tensor(position, dtype=torch.float32, device=robot.device)
    default_root_state[:, 7:] = 0.0

    robot.write_root_pose_to_sim(default_root_state[:, :7])
    robot.write_root_velocity_to_sim(default_root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.reset()
