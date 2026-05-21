from __future__ import annotations

import math

import glfw
import mujoco
import numpy as np


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

        # Normalize movement to have consistent speed in all directions, including diagonals
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


def add_debug_sphere(
    scene: mujoco.MjvScene,
    position: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
    if scene.ngeom >= scene.maxgeom:
        return

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        position,
        np.eye(3).reshape(-1),
        rgba,
    )
    scene.ngeom += 1


def add_debug_arrow(
    scene: mujoco.MjvScene,
    start: np.ndarray,
    vector: np.ndarray,
    rgba: np.ndarray,
) -> None:
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        return

    end = start + vector
    add_line(scene, start, end, rgba)

    direction = vector / length
    side = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0.0, 0.0])
    else:
        side /= np.linalg.norm(side)

    head_length = min(0.35, length * 0.28)
    head_width = head_length * 0.45
    add_line(scene, end, end - direction * head_length + side * head_width, rgba)
    add_line(scene, end, end - direction * head_length - side * head_width, rgba)


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
