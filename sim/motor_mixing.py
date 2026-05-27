from __future__ import annotations

import math

import mujoco
import numpy as np

from .config import (
    ALTITUDE_KD,
    ALTITUDE_KP,
    ARM_LENGTH,
    ATTITUDE_KD,
    ATTITUDE_KP,
    CAPTURE_MAX_TILT_TARGET,
    CAPTURE_POSITION_KD,
    CAPTURE_POSITION_KP,
    CAPTURE_YAW_SOFT_LIMIT,
    MAX_POSITION_INTEGRAL,
    MAX_ROTOR_THRUST,
    MAX_TILT_TARGET,
    MAX_YAW_TORQUE,
    POSITION_KD,
    POSITION_KI,
    POSITION_KP,
    QUADCOPTER_BODY_NAME,
    ROOT_JOINT_NAME,
    ROTOR_SITE_NAMES,
    YAW_KD,
    YAW_KP,
    YAW_ONLY_LEVEL_KD,
    YAW_ONLY_LEVEL_KP,
    YAW_PRIORITY_FULL_ERROR,
    YAW_PRIORITY_MIN_XY_SCALE,
    YAW_TARGET_RATE_LIMIT,
    YAW_TORQUE_COEFF,
    YAW_TRAVEL_SPEED_THRESHOLD,
)
from .math_utils import rotation_to_euler_xyz, wrap_angle


class DroneController:
    """Low-level controller that maps target state to rotor thrust and yaw torque."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, QUADCOPTER_BODY_NAME
        )
        if self.body_id < 0:
            raise RuntimeError(f"Could not find body '{QUADCOPTER_BODY_NAME}'.")

        root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ROOT_JOINT_NAME)
        if root_joint_id < 0:
            raise RuntimeError(f"Could not find joint '{ROOT_JOINT_NAME}'.")

        self.qvel_adr = int(model.jnt_dofadr[root_joint_id])
        self.site_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            for site_name in ROTOR_SITE_NAMES
        ]
        if any(site_id < 0 for site_id in self.site_ids):
            raise RuntimeError("Could not find all rotor thrust sites.")

        mujoco.mj_forward(model, data)
        self.mass = float(model.body_subtreemass[self.body_id])
        self.target_xy = data.xpos[self.body_id, :2].copy()
        self.target_velocity_xy = np.zeros(2)
        self.position_error_integral = np.zeros(2)
        self.target_height = float(data.xpos[self.body_id, 2])
        rotation = data.xmat[self.body_id].reshape(3, 3)
        _, _, self.target_yaw = rotation_to_euler_xyz(rotation)

        # see https://cookierobotics.com/066/ for the theory
        mixer = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [ARM_LENGTH, -ARM_LENGTH, ARM_LENGTH, -ARM_LENGTH],
                [-ARM_LENGTH, -ARM_LENGTH, ARM_LENGTH, ARM_LENGTH],
                [YAW_TORQUE_COEFF, -YAW_TORQUE_COEFF, -YAW_TORQUE_COEFF, YAW_TORQUE_COEFF],
            ]
        )
        self.inverse_mixer = np.linalg.inv(mixer)
        self.last_forces = np.zeros(4)
        self.capture_mode = False
        self.motors_enabled = True
        self.xy_control_enabled = True
        self.yaw_rate_target: float | None = None
        self.status = "hover: initializing"

    def mix_rotors(self, total_thrust: float, tau_x: float, tau_y: float, tau_z: float) -> np.ndarray:
        total_thrust = float(np.clip(total_thrust, 0.0, MAX_ROTOR_THRUST * 4.0))
        tau_z = float(np.clip(tau_z, -MAX_YAW_TORQUE, MAX_YAW_TORQUE))

        # This block attempts to apply the full yaw torque, then scales it back if it would cause any rotor to saturate. Otherwise the drone will destabilize.
        base_command = np.array([total_thrust, 0.0, 0.0, tau_z])
        base_forces = self.inverse_mixer @ base_command
        if base_forces.min() < 0.0 or base_forces.max() > MAX_ROTOR_THRUST:
            yaw_scale = 1.0
            for scale in np.linspace(0.9, 0.0, 10):
                candidate = self.inverse_mixer @ np.array([total_thrust, 0.0, 0.0, tau_z * scale])
                if candidate.min() >= 0.0 and candidate.max() <= MAX_ROTOR_THRUST:
                    base_forces = candidate
                    yaw_scale = scale
                    break
            tau_z *= yaw_scale

        # This block applies the roll/pitch torques with the minimum scaling necessary to avoid saturation.
        rp_forces = self.inverse_mixer @ np.array([0.0, tau_x, tau_y, 0.0])
        rp_scale = 1.0
        for index, delta in enumerate(rp_forces):
            # if rotor needs more thrust...
            if delta > 0.0:
                # ...but is already near max, scale back the roll/pitch commands
                rp_scale = min(rp_scale, (MAX_ROTOR_THRUST - base_forces[index]) / delta)
            # if rotor needs less thrust...
            elif delta < 0.0:
                # ...but is already near zero, scale back the roll/pitch commands
                rp_scale = min(rp_scale, -base_forces[index] / delta)
        rp_scale = float(np.clip(rp_scale, 0.0, 1.0))

        return np.clip(base_forces + rp_forces * rp_scale, 0.0, MAX_ROTOR_THRUST)

    def set_target_xy(self, target_xy: np.ndarray, target_velocity_xy: np.ndarray) -> None:
        self.target_xy = target_xy.copy()
        self.target_velocity_xy = target_velocity_xy.copy()
        

    def set_target_yaw_from_velocity(self, velocity_xy: np.ndarray, dt: float) -> None:
        speed = float(np.linalg.norm(velocity_xy))
        if speed > YAW_TRAVEL_SPEED_THRESHOLD:
            desired_yaw = math.atan2(float(velocity_xy[1]), float(velocity_xy[0]))
            yaw_step = float(
                np.clip(
                    wrap_angle(desired_yaw - self.target_yaw),
                    -YAW_TARGET_RATE_LIMIT * dt,
                    YAW_TARGET_RATE_LIMIT * dt,
                )
            )
            self.target_yaw = wrap_angle(self.target_yaw + yaw_step)

    def approach_target_height(self, target_height: float, dt: float, rate: float) -> None:
        max_step = rate * dt
        height_delta = float(np.clip(target_height - self.target_height, -max_step, max_step))
        self.target_height += height_delta

    def set_capture_mode(self, enabled: bool) -> None:
        if enabled and not self.capture_mode:
            self.position_error_integral[:] = 0.0
        self.capture_mode = enabled

    def cut_motors(self) -> None:
        self.motors_enabled = False
        self.last_forces[:] = 0.0

    def set_yaw_rate_target(self, yaw_rate: float | None) -> None:
        self.yaw_rate_target = yaw_rate

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        data.qfrc_applied[:] = 0.0
        if not self.motors_enabled:
            self.status = "land: motors off"
            return

        rotation = data.xmat[self.body_id].reshape(3, 3)
        _, _, yaw = rotation_to_euler_xyz(rotation)

        linear_velocity = data.qvel[self.qvel_adr : self.qvel_adr + 3]
        angular_velocity_body = data.qvel[self.qvel_adr + 3 : self.qvel_adr + 6]

        height = float(data.xpos[self.body_id, 2])
        height_error = self.target_height - height
        xy_error = self.target_xy - data.xpos[self.body_id, :2]
        self.position_error_integral += xy_error * float(model.opt.timestep)
        self.position_error_integral = np.clip(
            self.position_error_integral,
            -MAX_POSITION_INTEGRAL,
            MAX_POSITION_INTEGRAL,
        )
        vertical_velocity = float(linear_velocity[2])
        xy_velocity = linear_velocity[:2] - self.target_velocity_xy
        gravity = abs(float(model.opt.gravity[2]))
        body_up_world_z = max(0.25, float(rotation[2, 2]))

        yaw_error = abs(wrap_angle(self.target_yaw - yaw))
        if self.yaw_rate_target is None:
            yaw_priority = float(np.clip(yaw_error / YAW_PRIORITY_FULL_ERROR, 0.0, 1.0))
            xy_scale = 1.0 - (1.0 - YAW_PRIORITY_MIN_XY_SCALE) * yaw_priority
        else:
            xy_scale = 1.0

        # in the terminal phase when landing, the drone should move more sharply to capitalize on its velocity already being aligned, to avoid drifting.
        capture_gains_enabled = self.capture_mode and yaw_error < CAPTURE_YAW_SOFT_LIMIT
        position_kp = CAPTURE_POSITION_KP if capture_gains_enabled else POSITION_KP
        position_kd = CAPTURE_POSITION_KD if capture_gains_enabled else POSITION_KD
        position_ki = 0.0 if capture_gains_enabled else POSITION_KI
        max_tilt_target = CAPTURE_MAX_TILT_TARGET if capture_gains_enabled else MAX_TILT_TARGET
        max_tilt_target *= xy_scale

        # so called "pid controller"
        desired_accel_xy = (
            position_kp * xy_error
            + position_ki * self.position_error_integral
            - position_kd * xy_velocity
        ) * xy_scale
        if not self.xy_control_enabled:
            desired_accel_xy[:] = 0.0

        # go from desired horizontal acceleration to desired tilt angle, and clamp to avoid demanding impossible tilts
        desired_tilt = np.array([desired_accel_xy[0], desired_accel_xy[1], gravity])
        desired_tilt /= np.linalg.norm(desired_tilt)
        max_horizontal_tilt = math.sin(max_tilt_target)
        horizontal_tilt = float(np.linalg.norm(desired_tilt[:2]))
        if horizontal_tilt > max_horizontal_tilt:
            desired_tilt[:2] *= max_horizontal_tilt / horizontal_tilt
            desired_tilt[2] = math.sqrt(max(0.0, 1.0 - max_horizontal_tilt**2))

        total_thrust = (
            self.mass * gravity
            + ALTITUDE_KP * height_error
            - ALTITUDE_KD * vertical_velocity
        ) / body_up_world_z

        attitude_kp = YAW_ONLY_LEVEL_KP if self.yaw_rate_target is not None else ATTITUDE_KP
        attitude_kd = YAW_ONLY_LEVEL_KD if self.yaw_rate_target is not None else ATTITUDE_KD
        current_up_world = rotation[:, 2]
        level_error_world = np.cross(current_up_world, desired_tilt)
        level_error_body = rotation.T @ level_error_world
        tau_x = attitude_kp * level_error_body[0] - attitude_kd * angular_velocity_body[0]
        tau_y = attitude_kp * level_error_body[1] - attitude_kd * angular_velocity_body[1]
        if self.yaw_rate_target is None:
            tau_z = -YAW_KP * wrap_angle(yaw - self.target_yaw) - YAW_KD * angular_velocity_body[2]
        else:
            tau_z = YAW_KD * (self.yaw_rate_target - angular_velocity_body[2])
        tau_z = float(np.clip(tau_z, -MAX_YAW_TORQUE, MAX_YAW_TORQUE))

        rotor_forces = self.mix_rotors(total_thrust, tau_x, tau_y, 0.0)
        self.last_forces = rotor_forces

        # Apply forces at each rotor site
        for force, site_id in zip(rotor_forces, self.site_ids):
            force_world = rotation @ np.array([0.0, 0.0, force])
            mujoco.mj_applyFT(
                model,
                data,
                force_world,
                np.zeros(3),
                data.site_xpos[site_id],
                self.body_id,
                data.qfrc_applied,
            )

        # Apply yaw torque at the center of mass
        mujoco.mj_applyFT(
            model,
            data,
            np.zeros(3),
            rotation @ np.array([0.0, 0.0, tau_z]),
            data.xpos[self.body_id],
            self.body_id,
            data.qfrc_applied,
        )

        mode = "capture" if capture_gains_enabled else "align" if self.capture_mode else "hold"
        self.status = (
            f"{mode}, motors={rotor_forces[0]:.1f},{rotor_forces[1]:.1f},"
            f"{rotor_forces[2]:.1f},{rotor_forces[3]:.1f}N"
        )
