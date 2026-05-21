from __future__ import annotations

import math

import mujoco
import numpy as np

from .config import (
    CAR_CIRCLE_CENTER,
    CAR_CIRCLE_RADIUS,
    CAR_LINEAR_SPEED,
    CAR_MAX_FORCE,
    CAR_POSITION_KD,
    CAR_POSITION_KP,
    LANDING_CAR_BODY_NAME,
    LANDING_CAR_X_JOINT_NAME,
    LANDING_CAR_Y_JOINT_NAME,
    WIND_CHANGE_INTERVAL,
)


def landing_car_target_position(time: float) -> np.ndarray:
    angular_speed = CAR_LINEAR_SPEED / CAR_CIRCLE_RADIUS
    angle = angular_speed * time
    return CAR_CIRCLE_CENTER + np.array(
        [
            CAR_CIRCLE_RADIUS * math.cos(angle),
            CAR_CIRCLE_RADIUS * math.sin(angle),
            0.0,
        ]
    )


def landing_car_velocity(time: float) -> np.ndarray:
    angular_speed = CAR_LINEAR_SPEED / CAR_CIRCLE_RADIUS
    angle = angular_speed * time
    return CAR_LINEAR_SPEED * np.array([-math.sin(angle), math.cos(angle)])


def landing_car_yaw(time: float) -> float:
    angular_speed = CAR_LINEAR_SPEED / CAR_CIRCLE_RADIUS
    angle = angular_speed * time
    return angle + math.pi * 0.5


class CarController:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LANDING_CAR_BODY_NAME)
        if self.body_id < 0:
            raise RuntimeError(f"Could not find body '{LANDING_CAR_BODY_NAME}'.")

        joint_names = (LANDING_CAR_X_JOINT_NAME, LANDING_CAR_Y_JOINT_NAME)
        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            for joint_name in joint_names
        ]
        if any(joint_id < 0 for joint_id in joint_ids):
            raise RuntimeError("Could not find all landing car joints.")

        self.dof_adrs = [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
        self.mass = float(model.body_subtreemass[self.body_id])
        self.status = "car: initializing"

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        target_pos = landing_car_target_position(data.time)
        target_velocity = landing_car_velocity(data.time)

        pos_error = target_pos[:2] - data.xpos[self.body_id, :2]
        car_velocity = np.array([data.qvel[self.dof_adrs[0]], data.qvel[self.dof_adrs[1]]])
        vel_error = target_velocity - car_velocity
        force_xy = self.mass * (CAR_POSITION_KP * pos_error + CAR_POSITION_KD * vel_error)
        force_norm = float(np.linalg.norm(force_xy))
        if force_norm > CAR_MAX_FORCE:
            force_xy *= CAR_MAX_FORCE / force_norm

        data.qfrc_applied[self.dof_adrs[0]] += force_xy[0]
        data.qfrc_applied[self.dof_adrs[1]] += force_xy[1]
        self.status = (
            f"car: err=({pos_error[0]:+.2f},{pos_error[1]:+.2f}) "
            f"force={force_xy[0]:+.0f},{force_xy[1]:+.0f}N"
        )


class WindDisturbance:
    def __init__(self, body_id: int, strength: float) -> None:
        self.body_id = body_id
        self.strength = max(0.0, strength)
        self.current_segment = -1
        self.force_world = np.zeros(3)
        self.status = "wind: calm"

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self.strength <= 0.0:
            self.force_world[:] = 0.0
            self.status = "wind: disabled"
            return

        segment = int(data.time // WIND_CHANGE_INTERVAL)
        if segment != self.current_segment:
            self.current_segment = segment
            angle = self._segment_angle(segment)
            self.force_world = self.strength * np.array([math.cos(angle), math.sin(angle), 0.0])

        mujoco.mj_applyFT(
            model,
            data,
            self.force_world,
            np.zeros(3),
            data.xpos[self.body_id],
            self.body_id,
            data.qfrc_applied,
        )
        self.status = f"wind: ({self.force_world[0]:+.1f},{self.force_world[1]:+.1f})N"

    @staticmethod
    def _segment_angle(segment: int) -> float:
        value = math.sin(segment * 12.9898 + 78.233) * 43758.5453
        fraction = value - math.floor(value)
        return fraction * 2.0 * math.pi
