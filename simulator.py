from __future__ import annotations

import argparse
import math

import glfw
import mujoco
import numpy as np

from sim.config import (
    CAR_LINEAR_SPEED,
    DRONE_CAMERA_DOWN_ANGLE_DEG,
    DRONE_CAMERA_NAME,
    LANDING_PLATFORM_GEOM_NAME,
    MODEL_PATH,
    VISION_HEIGHT,
    VISION_WIDTH,
    WIND_FORCE_N,
)
from sim.motor_mixing import DroneController
from sim.policies import BehaviorPolicy, ChasePlopPolicy, MPCFoVPolicy, StableYawPolicy
from sim.rendering import (
    AutoTrackingCamera,
    add_camera_frustum,
    add_debug_arrow,
    add_debug_sphere,
    add_line,
    draw_detection_bbox,
    draw_viewport_border,
    key_callback,
)
from sim.vision import ArucoVision, ensure_aruco_marker_texture
from sim.world import CarTrajectory, WindDisturbance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the quadcopter landing simulator.")
    parser.add_argument(
        "--wind-strength",
        type=float,
        default=WIND_FORCE_N,
        help="Horizontal wind force in Newtons. Use 0 to disable wind.",
    )
    parser.add_argument(
        "--policy",
        choices=("chase-plop", "mpc-fov", "yaw-only"),
        default="chase-plop",
        help="High-level behavior policy to run.",
    )
    parser.add_argument(
        "--yaw-only-rate",
        type=float,
        default=None,
        help="Yaw rate in deg/s for --policy yaw-only",
    )
    parser.add_argument(
        "--car-motion",
        choices=("circle", "straight", "sine"),
        default="circle",
        help="Platform trajectory shape.",
    )
    parser.add_argument(
        "--car-speed",
        type=float,
        default=CAR_LINEAR_SPEED,
        help="Nominal platform speed in m/s.",
    )
    parser.add_argument(
        "--drone-camera-down-angle",
        type=float,
        default=DRONE_CAMERA_DOWN_ANGLE_DEG,
        help="Drone camera pitch in degrees downward from horizontal.",
    )
    return parser.parse_args()


def make_policy(
    args: argparse.Namespace,
    model: mujoco.MjModel,
    controller: DroneController,
    vision: ArucoVision,
    drone_camera_id: int,
    car_trajectory: CarTrajectory,
) -> BehaviorPolicy:
    if args.policy == "yaw-only":
        yaw_rate = 0.0 if args.yaw_only_rate is None else math.radians(args.yaw_only_rate)
        return StableYawPolicy(controller, yaw_rate)
    if args.policy == "mpc-fov":
        return MPCFoVPolicy(model, controller, vision, drone_camera_id, car_trajectory)
    if args.policy == "chase-plop":
        return ChasePlopPolicy(model, controller, vision, drone_camera_id, car_trajectory)
    raise ValueError(f"Unsupported policy: {args.policy}")


def set_drone_camera_down_angle(
    model: mujoco.MjModel,
    camera_id: int,
    down_angle_deg: float,
) -> None:
    angle = math.radians(float(np.clip(down_angle_deg, 0.0, 89.9)))
    forward_axis = np.array([math.cos(angle), 0.0, -math.sin(angle)])
    x_axis = np.array([0.0, -1.0, 0.0])
    z_axis = -forward_axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    camera_mat = np.column_stack((x_axis, y_axis, z_axis)).reshape(-1)
    camera_quat = np.zeros(4)
    mujoco.mju_mat2Quat(camera_quat, camera_mat)
    model.cam_quat[camera_id] = camera_quat
    model.cam_mat0[camera_id] = camera_mat


def main() -> None:
    args = parse_args()
    ensure_aruco_marker_texture()

    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW.")

    window = glfw.create_window(1280, 720, "Quadcopter Landing Simulator", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create the viewer window.")

    try:
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        drone_camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, DRONE_CAMERA_NAME
        )
        if drone_camera_id < 0:
            raise RuntimeError(f"Could not find camera '{DRONE_CAMERA_NAME}' in {MODEL_PATH}.")
        set_drone_camera_down_angle(model, drone_camera_id, args.drone_camera_down_angle)
        landing_platform_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, LANDING_PLATFORM_GEOM_NAME
        )
        if landing_platform_geom_id < 0:
            raise RuntimeError(
                f"Could not find geom '{LANDING_PLATFORM_GEOM_NAME}' in {MODEL_PATH}."
            )

        camera = AutoTrackingCamera()
        drone_camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(drone_camera)
        drone_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        drone_camera.fixedcamid = drone_camera_id

        vision = ArucoVision(model, drone_camera_id)

        controller = DroneController(model, data)
        car_trajectory = CarTrajectory(args.car_motion, args.car_speed)
        policy = make_policy(args, model, controller, vision, drone_camera_id, car_trajectory)
        wind = WindDisturbance(controller.body_id, args.wind_strength)

        option = mujoco.MjvOption()
        scene = mujoco.MjvScene(model, maxgeom=10000)
        drone_scene = mujoco.MjvScene(model, maxgeom=10000)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

        glfw.set_key_callback(window, key_callback)

        last_time = glfw.get_time()
        while not glfw.window_should_close(window):
            now = glfw.get_time()
            dt = max(1e-6, now - last_time)
            last_time = now

            glfw.poll_events()
            while data.time < now:
                policy.step(model, data)
                wind.apply(model, data)
                mujoco.mj_step(model, data)
            vision.poll()

            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
            drone_view_width = min(640, max(320, int(viewport_width * 0.38)))
            drone_view_height = int(drone_view_width * 9 / 16)
            drone_viewport = mujoco.MjrRect(16, 16, drone_view_width, drone_view_height)
            drone_aspect = drone_view_width / max(1, drone_view_height)
            camera.update(
                data.xpos[controller.body_id],
                data.geom_xpos[landing_platform_geom_id],
                viewport_width / max(1, viewport_height),
                dt,
            )

            mujoco.mjv_updateScene(
                model,
                data,
                option,
                None,
                camera.camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )
            add_camera_frustum(model, data, scene, drone_camera_id, drone_aspect)

            platform_marker_pos = data.geom_xpos[landing_platform_geom_id].copy()
            platform_marker_pos[2] += 0.25
            add_debug_sphere(scene, platform_marker_pos, 0.06, np.array([0.0, 1.0, 0.2, 1.0]))

            hold_marker_pos = np.array(
                [
                    controller.target_xy[0],
                    controller.target_xy[1],
                    controller.target_height,
                ]
            )
            add_debug_sphere(scene, hold_marker_pos, 0.08, np.array([1.0, 0.9, 0.0, 1.0]))

            wind_arrow_start = data.xpos[controller.body_id] + np.array([0.0, 0.0, 0.35])
            add_debug_arrow(
                scene,
                wind_arrow_start,
                wind.force_world * 0.9,
                np.array([1.0, 0.25, 0.0, 1.0]),
            )
            if vision.target_world is not None:
                add_line(
                    scene,
                    data.cam_xpos[drone_camera_id],
                    vision.target_world,
                    np.array([1.0, 0.05, 0.0, 1.0]),
                )
            vision_estimate = vision.dead_reckoned_target_world(data.time)
            if vision_estimate is not None:
                add_debug_sphere(
                    scene,
                    vision_estimate,
                    0.07,
                    np.array([1.0, 0.0, 1.0, 1.0]),
                )
            mujoco.mjr_render(viewport, scene, context)

            mujoco.mjv_updateScene(
                model,
                data,
                option,
                None,
                drone_camera,
                mujoco.mjtCatBit.mjCAT_ALL,
                drone_scene,
            )
            mujoco.mjr_render(drone_viewport, drone_scene, context)
            draw_detection_bbox(drone_viewport, vision.marker_bbox)
            draw_viewport_border(drone_viewport)

            if vision.should_update(data.time):
                vision_viewport = mujoco.MjrRect(0, 0, VISION_WIDTH, VISION_HEIGHT)
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
                if context.currentBuffer == mujoco.mjtFramebuffer.mjFB_OFFSCREEN:
                    mujoco.mjr_render(vision_viewport, drone_scene, context)
                    vision_rgb = np.empty((VISION_HEIGHT, VISION_WIDTH, 3), dtype=np.uint8)
                    vision_depth = np.empty((VISION_HEIGHT, VISION_WIDTH), dtype=np.float32)
                    mujoco.mjr_readPixels(vision_rgb, vision_depth, vision_viewport, context)
                    vision.submit(
                        vision_rgb,
                        data.cam_xpos[drone_camera_id],
                        data.cam_xmat[drone_camera_id].reshape(3, 3),
                        data.time,
                    )
                else:
                    vision.status = "vision: offscreen buffer unavailable"
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, context)

            overlay = "Auto camera | Esc/Space quit"
            status_panel = (
                f"{controller.status}\n"
                f"{vision.overlay_status()}"
            )
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                overlay,
                "",
                context,
            )
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                viewport,
                "",
                status_panel,
                context,
            )
            glfw.swap_buffers(window)
    finally:
        vision.shutdown()
        glfw.terminate()


if __name__ == "__main__":
    main()
