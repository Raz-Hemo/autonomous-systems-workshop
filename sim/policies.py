from __future__ import annotations

import math
from abc import ABC, abstractmethod

import mujoco
import numpy as np

from .config import (
    APPROACH_BEHIND_DISTANCE,
    APPROACH_BLEND_CLEARANCE,
    CAPTURE_CLEARANCE,
    CAPTURE_DESCENT_RATE,
    CAPTURE_TRIGGER_CLEARANCE,
    CAPTURE_TRIGGER_XY_ERROR,
    CAPTURE_YAW_SOFT_LIMIT,
    CUTOFF_CLEARANCE,
    CUTOFF_REL_SPEED,
    CUTOFF_XY_ERROR,
    IMAGE_CENTERING_GAIN,
    LANDING_CLEARANCE,
    LANDING_DESCENT_RATE,
    LANDING_PLATFORM_GEOM_NAME,
    MAX_IMAGE_CENTERING_OFFSET,
    MPC_BEHIND_DISTANCES,
    MPC_APPROACH_ERROR_WEIGHT,
    MPC_CLEARANCE_WEIGHT,
    MPC_CLEARANCES,
    MPC_COMMIT_DISTANCE,
    MPC_CONTROL_EFFORT_WEIGHT,
    MPC_FOV_MARGIN,
    MPC_FOV_PENALTY,
    MPC_GOOD_FOV_SLACK,
    MPC_HORIZON_SECONDS,
    MPC_LANDING_ERROR_WEIGHT,
    MPC_MIN_BEHIND_DISTANCE,
    MPC_MIN_CLEARANCE,
    MPC_PLOP_CLEARANCE,
    MPC_PLOP_CUTOFF_CLEARANCE,
    MPC_PLOP_CUTOFF_REL_SPEED,
    MPC_PLOP_CUTOFF_XY_ERROR,
    MPC_PLOP_DESCENT_RATE,
    MPC_PLOP_PHASE,
    MPC_PLOP_LEAD_TIME,
    MPC_PLOP_XY_DISTANCE,
    MPC_PREDICTION_STEPS,
    MPC_APPROACH_PHASE_RATE,
    MPC_BAD_FOV_SLACK,
    MPC_RETREAT_PHASE_RATE,
    MPC_RESPONSE_TIME,
    MPC_REPLAN_INTERVAL,
    MPC_SIDE_OFFSETS,
    MPC_TERMINAL_PROGRESS_WEIGHT,
)
from .math_utils import rotation_to_euler_xyz, wrap_angle
from .motor_mixing import DroneController
from .vision import ArucoVision
from .world import CarController, CarTrajectory


class BehaviorPolicy(ABC):
    """High-level behavior policy that decides controller targets each step."""

    def __init__(self, controller: DroneController) -> None:
        self.controller = controller
        self.status = "policy: initializing"

    @abstractmethod
    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Update targets, apply drone control, and apply any policy-owned forces."""


class StableYawPolicy(BehaviorPolicy):
    def __init__(self, controller: DroneController, yaw_rate: float) -> None:
        super().__init__(controller)
        self.controller.xy_control_enabled = False
        self.controller.set_yaw_rate_target(yaw_rate)
        self.status = "policy: yaw-only"

    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        mujoco.mj_forward(model, data)
        self.controller.apply(model, data)
        self.status = "policy: yaw-only"


class ChasePlopPolicy(BehaviorPolicy):
    def __init__(
        self,
        model: mujoco.MjModel,
        controller: DroneController,
        vision: ArucoVision,
        drone_camera_id: int,
        car_trajectory: CarTrajectory | None = None,
    ) -> None:
        super().__init__(controller)
        self.vision = vision
        self.drone_camera_id = drone_camera_id
        self.car_trajectory = CarTrajectory() if car_trajectory is None else car_trajectory
        self.car_controller = CarController(model, self.car_trajectory)
        self.landing_platform_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, LANDING_PLATFORM_GEOM_NAME
        )
        if self.landing_platform_geom_id < 0:
            raise RuntimeError(f"Could not find geom '{LANDING_PLATFORM_GEOM_NAME}'.")
        self.status = "policy: chase"

    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        mujoco.mj_forward(model, data)

        platform_xy = data.geom_xpos[self.landing_platform_geom_id, :2]
        platform_velocity = self.car_trajectory.velocity(data.time)
        platform_top_height = (
            data.geom_xpos[self.landing_platform_geom_id, 2]
            + model.geom_size[self.landing_platform_geom_id, 2]
        )
        clearance = float(data.xpos[self.controller.body_id, 2] - platform_top_height)

        target_xy = approach_target_xy(platform_xy, platform_velocity, clearance)
        target_xy += image_centering_offset_xy(
            self.vision.image_error,
            data.cam_xmat[self.drone_camera_id].reshape(3, 3),
        )
        self.controller.set_target_xy(target_xy, platform_velocity)
        self.controller.set_target_yaw_from_velocity(platform_velocity, float(model.opt.timestep))

        xy_error_norm = float(np.linalg.norm(platform_xy - data.xpos[self.controller.body_id, :2]))
        relative_xy_velocity = (
            data.qvel[self.controller.qvel_adr : self.controller.qvel_adr + 2]
            - platform_velocity
        )
        relative_xy_speed = float(np.linalg.norm(relative_xy_velocity))
        drone_rotation = data.xmat[self.controller.body_id].reshape(3, 3)
        _, _, drone_yaw = rotation_to_euler_xyz(drone_rotation)
        yaw_error = abs(wrap_angle(self.controller.target_yaw - drone_yaw))

        should_capture = (
            xy_error_norm < CAPTURE_TRIGGER_XY_ERROR
            and clearance < CAPTURE_TRIGGER_CLEARANCE
        )
        self.controller.set_capture_mode(should_capture)
        landing_clearance = CAPTURE_CLEARANCE if should_capture else LANDING_CLEARANCE
        descent_rate = CAPTURE_DESCENT_RATE if should_capture else LANDING_DESCENT_RATE
        if yaw_error > CAPTURE_YAW_SOFT_LIMIT:
            descent_rate *= 0.25
        self.controller.approach_target_height(
            platform_top_height + landing_clearance,
            float(model.opt.timestep),
            descent_rate,
        )
        if (
            should_capture
            and xy_error_norm < CUTOFF_XY_ERROR
            and relative_xy_speed < CUTOFF_REL_SPEED
            and clearance < CUTOFF_CLEARANCE
        ):
            self.controller.cut_motors()

        self.controller.apply(model, data)
        if self.controller.motors_enabled:
            self.controller.status += f" yawerr={math.degrees(yaw_error):.0f}deg"

        self.car_controller.apply(model, data)
        mode = "capture" if should_capture else "chase"
        self.status = f"policy: {mode}"


class MPCFoVPolicy(ChasePlopPolicy):
    """Sampled MPC-like landing policy with a soft field-of-view constraint."""

    def __init__(
        self,
        model: mujoco.MjModel,
        controller: DroneController,
        vision: ArucoVision,
        drone_camera_id: int,
        car_trajectory: CarTrajectory | None = None,
    ) -> None:
        super().__init__(model, controller, vision, drone_camera_id, car_trajectory)
        self.next_plan_time = -math.inf
        self.cached_target_xy = controller.target_xy.copy()
        self.cached_clearance = LANDING_CLEARANCE
        self.cached_cost = math.inf
        self.cached_fov_slack = 0.0
        self.approach_phase = 0.0
        self.final_plop_enabled = False

    def step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        mujoco.mj_forward(model, data)

        platform_xy = data.geom_xpos[self.landing_platform_geom_id, :2]
        platform_velocity = self.car_trajectory.velocity(data.time)
        platform_top_height = (
            data.geom_xpos[self.landing_platform_geom_id, 2]
            + model.geom_size[self.landing_platform_geom_id, 2]
        )
        clearance = float(data.xpos[self.controller.body_id, 2] - platform_top_height)

        if data.time >= self.next_plan_time:
            self._update_approach_phase(MPC_REPLAN_INTERVAL)
            (
                self.cached_target_xy,
                self.cached_clearance,
                self.cached_cost,
                self.cached_fov_slack,
            ) = self._choose_mpc_target(
                model,
                data,
                platform_xy,
                platform_velocity,
                platform_top_height,
            )
            self.next_plan_time = data.time + MPC_REPLAN_INTERVAL

        target_xy = self.cached_target_xy.copy()
        target_clearance = self.cached_clearance
        xy_error_norm = float(np.linalg.norm(platform_xy - data.xpos[self.controller.body_id, :2]))
        if (
            not self.final_plop_enabled
            and self.approach_phase >= MPC_PLOP_PHASE
            and xy_error_norm < MPC_PLOP_XY_DISTANCE
        ):
            self.final_plop_enabled = True

        final_plop = self.final_plop_enabled
        if final_plop:
            target_xy = platform_xy + platform_velocity * MPC_PLOP_LEAD_TIME
            target_clearance = MPC_PLOP_CLEARANCE

        if not final_plop:
            target_xy += image_centering_offset_xy(
                self.vision.image_error,
                data.cam_xmat[self.drone_camera_id].reshape(3, 3),
            )

        self.controller.set_target_xy(target_xy, platform_velocity)
        self.controller.set_target_yaw_from_velocity(platform_velocity, float(model.opt.timestep))

        relative_xy_velocity = (
            data.qvel[self.controller.qvel_adr : self.controller.qvel_adr + 2]
            - platform_velocity
        )
        relative_xy_speed = float(np.linalg.norm(relative_xy_velocity))
        drone_rotation = data.xmat[self.controller.body_id].reshape(3, 3)
        _, _, drone_yaw = rotation_to_euler_xyz(drone_rotation)
        yaw_error = abs(wrap_angle(self.controller.target_yaw - drone_yaw))

        should_capture = (
            final_plop
            or (
                xy_error_norm < CAPTURE_TRIGGER_XY_ERROR
                and clearance < CAPTURE_TRIGGER_CLEARANCE
                and self.cached_fov_slack > -0.15
            )
        )
        self.controller.set_capture_mode(should_capture)

        landing_clearance = target_clearance if final_plop else CAPTURE_CLEARANCE if should_capture else target_clearance
        descent_rate = (
            MPC_PLOP_DESCENT_RATE
            if final_plop
            else CAPTURE_DESCENT_RATE
            if should_capture
            else LANDING_DESCENT_RATE
        )
        if yaw_error > CAPTURE_YAW_SOFT_LIMIT:
            descent_rate *= 0.25
        self.controller.approach_target_height(
            platform_top_height + landing_clearance,
            float(model.opt.timestep),
            descent_rate,
        )
        if final_plop:
            should_cut_motors = (
                xy_error_norm < MPC_PLOP_CUTOFF_XY_ERROR
                and relative_xy_speed < MPC_PLOP_CUTOFF_REL_SPEED
                and clearance < MPC_PLOP_CUTOFF_CLEARANCE
            )
        else:
            should_cut_motors = (
                should_capture
                and xy_error_norm < CUTOFF_XY_ERROR
                and relative_xy_speed < CUTOFF_REL_SPEED
                and clearance < CUTOFF_CLEARANCE
            )
        if should_cut_motors:
            self.controller.cut_motors()

        self.controller.apply(model, data)
        if self.controller.motors_enabled:
            self.controller.status += (
                f" yawerr={math.degrees(yaw_error):.0f}deg "
                f"mpc={self.cached_cost:.1f} fov={self.cached_fov_slack:+.2f} "
                f"phase={self.approach_phase:.2f} plop={int(final_plop)}"
            )

        self.car_controller.apply(model, data)
        mode = "plop" if final_plop else "capture" if should_capture else "mpc-fov"
        self.status = f"policy: {mode}"

    def _update_approach_phase(self, dt: float) -> None:
        if self.cached_fov_slack > MPC_GOOD_FOV_SLACK:
            self.approach_phase += MPC_APPROACH_PHASE_RATE * dt
        elif self.cached_fov_slack < MPC_BAD_FOV_SLACK:
            self.approach_phase -= MPC_RETREAT_PHASE_RATE * dt
        else:
            slack_range = MPC_GOOD_FOV_SLACK - MPC_BAD_FOV_SLACK
            confidence = (self.cached_fov_slack - MPC_BAD_FOV_SLACK) / slack_range
            self.approach_phase += MPC_APPROACH_PHASE_RATE * 0.35 * confidence * dt

        self.approach_phase = float(np.clip(self.approach_phase, 0.0, 1.0))

    def _choose_mpc_target(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        platform_xy: np.ndarray,
        platform_velocity: np.ndarray,
        platform_top_height: float,
    ) -> tuple[np.ndarray, float, float, float]:
        speed = float(np.linalg.norm(platform_velocity))
        if speed > 1e-6:
            forward = platform_velocity / speed
        else:
            forward = np.array([1.0, 0.0])
        behind = -forward
        side = np.array([-forward[1], forward[0]])

        drone_pos = data.xpos[self.controller.body_id].copy()
        drone_vel = data.qvel[self.controller.qvel_adr : self.controller.qvel_adr + 3].copy()

        best_target_xy = platform_xy.copy()
        best_clearance = LANDING_CLEARANCE
        best_cost = math.inf
        best_slack = -math.inf

        behind_distances = self._phase_scaled_values(MPC_BEHIND_DISTANCES, MPC_MIN_BEHIND_DISTANCE)
        clearances = self._phase_scaled_values(MPC_CLEARANCES, MPC_MIN_CLEARANCE)

        for behind_distance in behind_distances:
            for side_offset in MPC_SIDE_OFFSETS:
                candidate_xy = platform_xy + behind * behind_distance + side * side_offset
                for candidate_clearance in clearances:
                    candidate_z = platform_top_height + candidate_clearance
                    candidate_pos = np.array([candidate_xy[0], candidate_xy[1], candidate_z])
                    cost, slack = self._score_candidate(
                        model,
                        data,
                        drone_pos,
                        drone_vel,
                        platform_xy,
                        platform_velocity,
                        platform_top_height,
                        candidate_pos,
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_slack = slack
                        best_target_xy = candidate_xy
                        best_clearance = candidate_clearance

        return best_target_xy, best_clearance, best_cost, best_slack

    def _phase_scaled_values(self, values: tuple[float, ...], minimum: float) -> tuple[float, ...]:
        return tuple(
            max(minimum, value * (1.0 - self.approach_phase) + minimum * self.approach_phase)
            for value in values
        )

    def _score_candidate(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        drone_pos: np.ndarray,
        drone_vel: np.ndarray,
        platform_xy: np.ndarray,
        platform_velocity: np.ndarray,
        platform_top_height: float,
        candidate_pos: np.ndarray,
    ) -> tuple[float, float]:
        cost = 0.0
        min_slack = math.inf
        approach_offset = candidate_pos - np.array(
            [platform_xy[0], platform_xy[1], platform_top_height]
        )
        drone_to_platform = float(np.linalg.norm(drone_pos[:2] - platform_xy))
        commit_blend = 1.0 - float(np.clip(drone_to_platform / MPC_COMMIT_DISTANCE, 0.0, 1.0))
        commit_blend = max(commit_blend, self.approach_phase)
        reference_now = (
            np.array([platform_xy[0], platform_xy[1], platform_top_height])
            + approach_offset * (1.0 - commit_blend)
        )
        target_offset = reference_now - drone_pos
        response = max(1e-3, MPC_RESPONSE_TIME)

        times = np.linspace(
            MPC_HORIZON_SECONDS / MPC_PREDICTION_STEPS,
            MPC_HORIZON_SECONDS,
            MPC_PREDICTION_STEPS,
        )
        final_landing_error = drone_to_platform
        for horizon_time in times:
            alpha = 1.0 - math.exp(-float(horizon_time) / response)
            predicted_drone = drone_pos + drone_vel * horizon_time * (1.0 - alpha)
            predicted_drone += target_offset * alpha
            predicted_platform_xy = platform_xy + platform_velocity * horizon_time
            predicted_platform = np.array(
                [predicted_platform_xy[0], predicted_platform_xy[1], platform_top_height]
            )
            predicted_reference = predicted_platform + approach_offset * (1.0 - commit_blend)

            approach_error = np.linalg.norm(predicted_drone - predicted_reference)
            landing_error = np.linalg.norm(predicted_drone[:2] - predicted_platform_xy)
            final_landing_error = float(landing_error)
            clearance = predicted_drone[2] - platform_top_height
            fov_violation, slack = self._fov_violation(
                model, data, predicted_drone, predicted_platform, platform_velocity
            )
            min_slack = min(min_slack, slack)
            cost += (
                MPC_APPROACH_ERROR_WEIGHT * approach_error**2
                + MPC_LANDING_ERROR_WEIGHT * commit_blend * landing_error**2
                + MPC_CLEARANCE_WEIGHT * max(0.0, clearance - LANDING_CLEARANCE) ** 2
                + MPC_FOV_PENALTY * fov_violation**2
            )

        progress_weight = MPC_TERMINAL_PROGRESS_WEIGHT * (0.25 + self.approach_phase)
        cost += progress_weight * final_landing_error**2
        cost += MPC_CONTROL_EFFORT_WEIGHT * float(np.linalg.norm(candidate_pos - drone_pos) ** 2)
        return cost, min_slack

    def _fov_violation(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        predicted_drone: np.ndarray,
        predicted_platform: np.ndarray,
        platform_velocity: np.ndarray,
    ) -> tuple[float, float]:
        speed = float(np.linalg.norm(platform_velocity))
        if speed > 1e-6:
            yaw = math.atan2(float(platform_velocity[1]), float(platform_velocity[0]))
        else:
            rotation = data.xmat[self.controller.body_id].reshape(3, 3)
            _, _, yaw = rotation_to_euler_xyz(rotation)

        world_from_body_yaw = np.array(
            [
                [math.cos(yaw), -math.sin(yaw), 0.0],
                [math.sin(yaw), math.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        current_body_mat = data.xmat[self.controller.body_id].reshape(3, 3)
        current_camera_mat = data.cam_xmat[self.drone_camera_id].reshape(3, 3)
        camera_from_body = current_body_mat.T @ current_camera_mat
        camera_mat = world_from_body_yaw @ camera_from_body

        target_camera = camera_mat.T @ (predicted_platform - predicted_drone)
        forward_depth = -target_camera[2]
        if forward_depth <= 1e-3:
            return 1.0, -1.0

        fovy = math.radians(float(model.cam_fovy[self.drone_camera_id]))
        half_y = math.tan(fovy * 0.5)
        half_x = half_y * (16.0 / 9.0)
        normalized_x = abs(float(target_camera[0]) / forward_depth) / max(1e-6, half_x)
        normalized_y = abs(float(target_camera[1]) / forward_depth) / max(1e-6, half_y)
        normalized_edge = max(normalized_x, normalized_y)
        slack = MPC_FOV_MARGIN - normalized_edge
        return max(0.0, -slack), slack


def approach_target_xy(
    platform_xy: np.ndarray, platform_velocity: np.ndarray, clearance: float
) -> np.ndarray:
    speed = float(np.linalg.norm(platform_velocity))
    if speed < 1e-6:
        return platform_xy.copy()

    behind_direction = -platform_velocity / speed
    blend = float(np.clip(clearance / APPROACH_BLEND_CLEARANCE, 0.0, 1.0))
    return platform_xy + behind_direction * APPROACH_BEHIND_DISTANCE * blend


def image_centering_offset_xy(
    image_error: np.ndarray,
    camera_mat: np.ndarray,
) -> np.ndarray:
    error_norm = float(np.linalg.norm(image_error))
    if error_norm < 1e-3:
        return np.zeros(2)

    camera_right_xy = camera_mat[:2, 0]
    camera_up_xy = camera_mat[:2, 1]
    correction_xy = (
        image_error[0] * camera_right_xy
        - image_error[1] * camera_up_xy
    ) * IMAGE_CENTERING_GAIN

    correction_norm = float(np.linalg.norm(correction_xy))
    if correction_norm > MAX_IMAGE_CENTERING_OFFSET:
        correction_xy *= MAX_IMAGE_CENTERING_OFFSET / correction_norm
    return correction_xy
