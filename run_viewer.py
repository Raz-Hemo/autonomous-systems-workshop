from __future__ import annotations

import math
from pathlib import Path

import glfw
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "basic_scene.xml"
DRONE_CAMERA_NAME = "drone_cam"
LANDING_CAR_BODY_NAME = "landing_car"
CAR_CIRCLE_CENTER = np.array([10.0, 0.0, 0.22])
CAR_CIRCLE_RADIUS = 2.2
CAR_LINEAR_SPEED = 1.1


class FreeCamera:
    def __init__(self) -> None:
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = np.array([0.0, 0.0, 0.7])
        self.camera.distance = 3.4
        self.camera.azimuth = 135.0
        self.camera.elevation = -25.0

        self.move_speed = 2.7
        self.mouse_sensitivity = 0.18
        self.last_cursor: tuple[float, float] | None = None
        self.mouse_captured = False
        self.ignore_next_mouse_delta = False

    def update_keyboard(self, window: glfw._GLFWwindow, dt: float) -> None:
        yaw = math.radians(self.camera.azimuth)
        forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
        movement = np.zeros(3)

        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            movement += forward
        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            movement -= forward
        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            movement += right
        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            movement -= right
        if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
            movement += np.array([0.0, 0.0, 1.0])
        if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
            movement -= np.array([0.0, 0.0, 1.0])

        norm = np.linalg.norm(movement)
        if norm > 0:
            self.camera.lookat[:] += movement / norm * self.move_speed * dt

    def cursor_callback(self, window: glfw._GLFWwindow, xpos: float, ypos: float) -> None:
        if self.last_cursor is None:
            self.last_cursor = (xpos, ypos)
            return

        dx = xpos - self.last_cursor[0]
        dy = ypos - self.last_cursor[1]
        self.last_cursor = (xpos, ypos)

        if not self.mouse_captured or self.ignore_next_mouse_delta:
            self.ignore_next_mouse_delta = False
            return

        self.camera.azimuth -= dx * self.mouse_sensitivity
        self.camera.elevation = float(
            np.clip(self.camera.elevation - dy * self.mouse_sensitivity, -89.0, 10.0)
        )

    def mouse_button_callback(
        self, window: glfw._GLFWwindow, button: int, action: int, mods: int
    ) -> None:
        if button != glfw.MOUSE_BUTTON_RIGHT:
            return

        self.mouse_captured = action == glfw.PRESS
        glfw.set_input_mode(
            window,
            glfw.CURSOR,
            glfw.CURSOR_DISABLED if self.mouse_captured else glfw.CURSOR_NORMAL,
        )
        self.last_cursor = glfw.get_cursor_pos(window)
        self.ignore_next_mouse_delta = self.mouse_captured

    def scroll_callback(self, window: glfw._GLFWwindow, xoffset: float, yoffset: float) -> None:
        self.camera.distance = float(np.clip(self.camera.distance * (0.9 ** yoffset), 0.4, 15.0))


def key_callback(window: glfw._GLFWwindow, key: int, scancode: int, action: int, mods: int) -> None:
    if action == glfw.PRESS and key in (glfw.KEY_ESCAPE, glfw.KEY_SPACE):
        glfw.set_window_should_close(window, True)


def yaw_to_quat(yaw: float) -> np.ndarray:
    half_yaw = yaw * 0.5
    return np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)])


def update_landing_car(data: mujoco.MjData, mocap_id: int, time: float) -> None:
    angular_speed = CAR_LINEAR_SPEED / CAR_CIRCLE_RADIUS
    angle = angular_speed * time
    tangent_yaw = angle + math.pi * 0.5

    data.mocap_pos[mocap_id] = CAR_CIRCLE_CENTER + np.array(
        [
            CAR_CIRCLE_RADIUS * math.cos(angle),
            CAR_CIRCLE_RADIUS * math.sin(angle),
            0.0,
        ]
    )
    data.mocap_quat[mocap_id] = yaw_to_quat(tangent_yaw)


def add_line(scene: mujoco.MjvScene, start: np.ndarray, end: np.ndarray, rgba: np.ndarray) -> None:
    if scene.ngeom >= scene.maxgeom:
        return

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).reshape(-1),
        rgba,
    )
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2.0, start, end)
    scene.ngeom += 1


def draw_viewport_border(viewport: mujoco.MjrRect, thickness: int = 2) -> None:
    red = (1.0, 0.0, 0.0, 1.0)
    left = viewport.left
    bottom = viewport.bottom
    width = viewport.width
    height = viewport.height

    mujoco.mjr_rectangle(mujoco.MjrRect(left, bottom, width, thickness), *red)
    mujoco.mjr_rectangle(mujoco.MjrRect(left, bottom + height - thickness, width, thickness), *red)
    mujoco.mjr_rectangle(mujoco.MjrRect(left, bottom, thickness, height), *red)
    mujoco.mjr_rectangle(mujoco.MjrRect(left + width - thickness, bottom, thickness, height), *red)


def add_camera_frustum(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene: mujoco.MjvScene,
    camera_id: int,
    aspect: float,
) -> None:
    camera_pos = data.cam_xpos[camera_id].copy()
    camera_mat = data.cam_xmat[camera_id].reshape(3, 3).copy()

    fovy = math.radians(float(model.cam_fovy[camera_id]))
    ground_z = 0.0
    ground_offset = np.array([0.0, 0.0, 0.05])
    half_y = math.tan(fovy * 0.5)
    half_x = half_y * aspect

    corner_dirs = [
        camera_mat @ np.array([half_x, half_y, -1.0]),
        camera_mat @ np.array([-half_x, half_y, -1.0]),
        camera_mat @ np.array([-half_x, -half_y, -1.0]),
        camera_mat @ np.array([half_x, -half_y, -1.0]),
    ]

    ground_hits: list[np.ndarray | None] = []
    for direction in corner_dirs:
        if abs(direction[2]) < 1e-6:
            ground_hits.append(None)
            continue

        distance = (ground_z - camera_pos[2]) / direction[2]
        if distance <= 0:
            ground_hits.append(None)
            continue

        ground_hits.append(camera_pos + direction * distance + ground_offset)

    rgba = np.array([0.0, 0.95, 1.0, 1.0])
    for index in range(4):
        hit = ground_hits[index]
        next_hit = ground_hits[(index + 1) % 4]

        if hit is not None:
            add_line(scene, camera_pos, hit, rgba)
        if hit is not None and next_hit is not None:
            add_line(scene, hit, next_hit, rgba)


def main() -> None:
    if not glfw.init():
        raise RuntimeError("Could not initialize GLFW.")

    window = glfw.create_window(1280, 720, "Quadcopter Landing Simulator", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create the viewer window.")

    glfw.make_context_current(window)
    glfw.swap_interval(1)

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    drone_camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, DRONE_CAMERA_NAME
    )
    if drone_camera_id < 0:
        raise RuntimeError(f"Could not find camera '{DRONE_CAMERA_NAME}' in {MODEL_PATH}.")
    landing_car_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, LANDING_CAR_BODY_NAME
    )
    if landing_car_body_id < 0:
        raise RuntimeError(f"Could not find body '{LANDING_CAR_BODY_NAME}' in {MODEL_PATH}.")
    landing_car_mocap_id = int(model.body_mocapid[landing_car_body_id])
    if landing_car_mocap_id < 0:
        raise RuntimeError(f"Body '{LANDING_CAR_BODY_NAME}' is not a mocap body.")

    camera = FreeCamera()
    drone_camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(drone_camera)
    drone_camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    drone_camera.fixedcamid = drone_camera_id

    option = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    drone_scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    glfw.set_key_callback(window, key_callback)
    glfw.set_cursor_pos_callback(window, camera.cursor_callback)
    glfw.set_mouse_button_callback(window, camera.mouse_button_callback)
    glfw.set_scroll_callback(window, camera.scroll_callback)

    last_time = glfw.get_time()

    while not glfw.window_should_close(window):
        now = glfw.get_time()
        dt = max(1e-6, now - last_time)
        last_time = now

        glfw.poll_events()
        camera.update_keyboard(window, dt)
        update_landing_car(data, landing_car_mocap_id, now)

        while data.time < now:
            update_landing_car(data, landing_car_mocap_id, data.time)
            mujoco.mj_step(model, data)

        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        drone_view_width = max(220, int(viewport_width * 0.26))
        drone_view_height = max(124, int(drone_view_width * 9 / 16))
        drone_viewport = mujoco.MjrRect(16, 16, drone_view_width, drone_view_height)
        drone_aspect = drone_view_width / max(1, drone_view_height)

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
        draw_viewport_border(drone_viewport)

        overlay = "WASD move | Q/E up-down | right mouse look | wheel zoom | Esc/Space quit"
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            overlay,
            "",
            context,
        )
        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
