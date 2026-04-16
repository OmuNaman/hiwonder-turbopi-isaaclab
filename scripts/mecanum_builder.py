# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers to add simple physical mecanum rollers to the TurboPi wheels."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim.spawners.materials.physics_materials import spawn_rigid_body_material
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sim.utils import bind_physics_material


@dataclass
class MecanumWheelConfig:
    num_rollers: int = 9
    roller_radius: float = 0.0035
    roller_height: float = 0.014
    roller_center_dist: float = 0.029
    roller_angle_deg: float = 45.0
    roller_mass: float = 0.001
    prefix: str = ""
    wheel_pos_in_robot: tuple[float, float, float] = (0.0, 0.0, 0.0)


WHEEL_CONFIGS = {
    "wheel_lf_link": MecanumWheelConfig(
        roller_angle_deg=+45.0,
        prefix="lf",
        wheel_pos_in_robot=(0.057814, 0.064852, 0.032474),
    ),
    "wheel_rf_link": MecanumWheelConfig(
        roller_angle_deg=-45.0,
        prefix="rf",
        wheel_pos_in_robot=(0.057814, -0.065148, 0.032474),
    ),
    "wheel_lb_link": MecanumWheelConfig(
        roller_angle_deg=-45.0,
        prefix="lb",
        wheel_pos_in_robot=(-0.061513, 0.064852, 0.032474),
    ),
    "wheel_rb_link": MecanumWheelConfig(
        roller_angle_deg=+45.0,
        prefix="rb",
        wheel_pos_in_robot=(-0.061513, -0.065148, 0.032474),
    ),
}

ROLLER_MATERIAL_CFG = RigidBodyMaterialCfg(
    static_friction=0.8,
    dynamic_friction=0.6,
    restitution=0.0,
    friction_combine_mode="average",
)


def _quat_from_two_vectors(v_from: Gf.Vec3d, v_to: Gf.Vec3d) -> Gf.Quatd:
    v_from = v_from.GetNormalized()
    v_to = v_to.GetNormalized()
    dot = Gf.Dot(v_from, v_to)

    if dot > 0.999999:
        return Gf.Quatd(1.0, 0.0, 0.0, 0.0)
    if dot < -0.999999:
        ortho = Gf.Cross(Gf.Vec3d(1, 0, 0), v_from)
        if ortho.GetLength() < 1e-6:
            ortho = Gf.Cross(Gf.Vec3d(0, 1, 0), v_from)
        ortho = ortho.GetNormalized()
        return Gf.Quatd(0.0, ortho[0], ortho[1], ortho[2])

    axis = Gf.Cross(v_from, v_to)
    quat = Gf.Quatd(1.0 + dot, axis[0], axis[1], axis[2])
    return quat.GetNormalized()


def _to_quatf(quat: Gf.Quatd) -> Gf.Quatf:
    return Gf.Quatf(
        float(quat.GetReal()),
        float(quat.GetImaginary()[0]),
        float(quat.GetImaginary()[1]),
        float(quat.GetImaginary()[2]),
    )


def _compute_roller_transform(index: int, cfg: MecanumWheelConfig) -> tuple[Gf.Vec3d, Gf.Quatd]:
    theta = 2.0 * math.pi * index / cfg.num_rollers
    alpha = math.radians(cfg.roller_angle_deg)

    pos = Gf.Vec3d(
        cfg.roller_center_dist * math.cos(theta),
        0.0,
        cfg.roller_center_dist * math.sin(theta),
    )

    spin_axis = Gf.Vec3d(
        -math.sin(alpha) * math.sin(theta),
        math.cos(alpha),
        math.sin(alpha) * math.cos(theta),
    )
    quat = _quat_from_two_vectors(Gf.Vec3d(0, 0, 1), spin_axis)
    return pos, quat


def _disable_hub_collision(stage: Usd.Stage, wheel_link_path: str) -> None:
    wheel_prim = stage.GetPrimAtPath(wheel_link_path)
    if not wheel_prim.IsValid():
        return
    for descendant in Usd.PrimRange(wheel_prim):
        if descendant.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(descendant).GetCollisionEnabledAttr().Set(False)


def _create_roller_material(stage: Usd.Stage, material_path: str) -> None:
    spawn_rigid_body_material(material_path, ROLLER_MATERIAL_CFG)


def _create_roller(
    stage: Usd.Stage,
    robot_prim_path: str,
    wheel_link_path: str,
    roller_index: int,
    cfg: MecanumWheelConfig,
    material_path: str,
) -> None:
    pos_in_wheel, quat = _compute_roller_transform(roller_index, cfg)
    roller_name = f"{cfg.prefix}_roller_{roller_index}"
    roller_path = f"{robot_prim_path}/{roller_name}"
    capsule_path = f"{roller_path}/capsule"
    joint_path = f"{robot_prim_path}/{roller_name}_joint"

    wp = cfg.wheel_pos_in_robot
    init_pos = Gf.Vec3d(
        wp[0] + pos_in_wheel[0],
        wp[1] + pos_in_wheel[1],
        wp[2] + pos_in_wheel[2],
    )

    roller_xform = UsdGeom.Xform.Define(stage, roller_path)
    roller_xform.AddTranslateOp().Set(init_pos)
    roller_xform.AddOrientOp().Set(_to_quatf(quat))

    capsule = UsdGeom.Capsule.Define(stage, capsule_path)
    capsule.CreateRadiusAttr().Set(cfg.roller_radius)
    capsule.CreateHeightAttr().Set(cfg.roller_height)
    capsule.CreateAxisAttr().Set("Z")

    roller_prim = stage.GetPrimAtPath(roller_path)
    UsdPhysics.RigidBodyAPI.Apply(roller_prim)

    capsule_prim = stage.GetPrimAtPath(capsule_path)
    UsdPhysics.CollisionAPI.Apply(capsule_prim)
    UsdPhysics.MassAPI.Apply(roller_prim).CreateMassAttr().Set(cfg.roller_mass)

    bind_physics_material(capsule_path, material_path)

    joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(wheel_link_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(roller_path)])
    joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(float(pos_in_wheel[0]), float(pos_in_wheel[1]), float(pos_in_wheel[2]))
    )
    joint.CreateLocalRot0Attr().Set(_to_quatf(quat))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateAxisAttr().Set("Z")
    joint.CreateBreakForceAttr().Set(1e20)
    joint.CreateBreakTorqueAttr().Set(1e20)


def add_all_mecanum_rollers(robot_prim_path: str, stage: Usd.Stage) -> None:
    """Adds physical mecanum rollers to all four wheels.

    Call after spawning the robot under ``env_0`` and before ``sim.reset()``.
    """

    material_path = f"{robot_prim_path}/RollerMaterial"
    _create_roller_material(stage, material_path)

    for wheel_name, cfg in WHEEL_CONFIGS.items():
        wheel_path = f"{robot_prim_path}/{wheel_name}"
        _disable_hub_collision(stage, wheel_path)
        for i in range(cfg.num_rollers):
            _create_roller(stage, robot_prim_path, wheel_path, i, cfg, material_path)
