from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .config import (
    CAR_CIRCLE_CENTER,
    CAR_CIRCLE_RADIUS,
    CAR_LINEAR_SPEED,
    CAR_MAX_FORCE,
    CAR_POSITION_KD,
    CAR_POSITION_KP,
    CAR_SINE_AMPLITUDE,
    CAR_SINE_START,
    CAR_SINE_WAVELENGTH,
    CAR_STRAIGHT_START,
    LANDING_CAR_BODY_NAME,
    LANDING_CAR_X_JOINT_NAME,
    LANDING_CAR_Y_JOINT_NAME,
    WIND_CHANGE_INTERVAL,
)

CarMotion = str


@dataclass(frozen=True)
class CarTrajectory:
    motion: CarMotion = "circle"
    speed: float = CAR_LINEAR_SPEED

    def __post_init__(self) -> None:
        object.__setattr__(self, "speed", max(0.0, float(self.speed)))

    def position(self, time: float) -> np.ndarray:
        speed = self.speed
        if self.motion == "straight":
            return CAR_STRAIGHT_START + np.array([0.0, speed * time, 0.0])
        if self.motion == "sine":
            phase = 2.0 * math.pi * speed * time / CAR_SINE_WAVELENGTH
            return CAR_SINE_START + np.array(
                [
                    CAR_SINE_AMPLITUDE * math.sin(phase),
                    speed * time,
                    0.0,
                ]
            )

        angular_speed = speed / CAR_CIRCLE_RADIUS if CAR_CIRCLE_RADIUS > 0 else 0.0
        angle = angular_speed * time
        return CAR_CIRCLE_CENTER + np.array(
            [
                CAR_CIRCLE_RADIUS * math.cos(angle),
                CAR_CIRCLE_RADIUS * math.sin(angle),
                0.0,
            ]
        )

    def velocity(self, time: float) -> np.ndarray:
        speed = self.speed
        if self.motion == "straight":
            return np.array([0.0, speed])
        if self.motion == "sine":
            phase = 2.0 * math.pi * speed * time / CAR_SINE_WAVELENGTH
            lateral_velocity = (
                CAR_SINE_AMPLITUDE
                * (2.0 * math.pi * speed / CAR_SINE_WAVELENGTH)
                * math.cos(phase)
            )
            return np.array([lateral_velocity, speed])

        angular_speed = speed / CAR_CIRCLE_RADIUS if CAR_CIRCLE_RADIUS > 0 else 0.0
        angle = angular_speed * time
        return speed * np.array([-math.sin(angle), math.cos(angle)])

    def yaw(self, time: float) -> float:
        velocity = self.velocity(time)
        if np.linalg.norm(velocity) < 1e-6:
            return 0.0
        return math.atan2(float(velocity[1]), float(velocity[0]))


def landing_car_target_position(time: float) -> np.ndarray:
    return CarTrajectory().position(time)


def landing_car_velocity(time: float) -> np.ndarray:
    return CarTrajectory().velocity(time)


def landing_car_yaw(time: float) -> float:
    return CarTrajectory().yaw(time)


class CarController:
    def __init__(self, model: mujoco.MjModel, trajectory: CarTrajectory | None = None) -> None:
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
        self.trajectory = CarTrajectory() if trajectory is None else trajectory

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        target_pos = self.trajectory.position(data.time)
        target_velocity = self.trajectory.velocity(data.time)

        pos_error = target_pos[:2] - data.xpos[self.body_id, :2]
        car_velocity = np.array([data.qvel[self.dof_adrs[0]], data.qvel[self.dof_adrs[1]]])
        vel_error = target_velocity - car_velocity
        force_xy = self.mass * (CAR_POSITION_KP * pos_error + CAR_POSITION_KD * vel_error)
        force_norm = float(np.linalg.norm(force_xy))
        if force_norm > CAR_MAX_FORCE:
            force_xy *= CAR_MAX_FORCE / force_norm

        data.qfrc_applied[self.dof_adrs[0]] += force_xy[0]
        data.qfrc_applied[self.dof_adrs[1]] += force_xy[1]


class WindDisturbance:
    def __init__(self, body_id: int, strength: float) -> None:
        self.body_id = body_id
        self.strength = max(0.0, strength)
        self.current_segment = -1
        self.force_world = np.zeros(3)

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
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

    @staticmethod
    def _segment_angle(segment: int) -> float:
        value = math.sin(segment * 12.9898 + 78.233) * 43758.5453
        fraction = value - math.floor(value)
        return fraction * 2.0 * math.pi
