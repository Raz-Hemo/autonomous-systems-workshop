from __future__ import annotations

import math

import numpy as np


def yaw_to_quat(yaw: float) -> np.ndarray:
    half_yaw = yaw * 0.5
    return np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)])


def rotation_to_euler_xyz(rotation: np.ndarray) -> tuple[float, float, float]:
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return roll, pitch, yaw


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
