import cv2
import mujoco
import numpy as np
import pytest

from sim.config import ARUCO_MARKER_ID, MODEL_PATH, DRONE_CAMERA_NAME
from sim.vision import ArucoVision


@pytest.fixture
def vision() -> ArucoVision:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, DRONE_CAMERA_NAME
    )
    detector = ArucoVision(model, camera_id)
    try:
        yield detector
    finally:
        detector.shutdown()


def test_detects_rotated_marker_orientation(vision: ArucoVision) -> None:
    image_rgb = _marker_frame(rotation=cv2.ROTATE_90_CLOCKWISE)
    read_pixels_style_image = np.flipud(image_rgb)

    target_world, image_error, marker_angle, target_rvec, marker_bbox, status = (
        vision._detect(
            read_pixels_style_image,
            np.zeros(3),
            np.eye(3),
        )
    )

    assert target_world is not None, status
    assert target_rvec is not None
    assert marker_bbox is not None
    assert image_error[0] == pytest.approx(0.0, abs=0.08)
    assert image_error[1] == pytest.approx(0.0, abs=0.08)
    assert marker_angle is not None
    assert marker_angle == pytest.approx(90.0, abs=6.0)


def test_detects_marker_under_perspective_tilt(vision: ArucoVision) -> None:
    image_rgb = _perspective_marker_frame()
    read_pixels_style_image = np.flipud(image_rgb)

    target_world, image_error, marker_angle, target_rvec, marker_bbox, status = (
        vision._detect(
            read_pixels_style_image,
            np.zeros(3),
            np.eye(3),
        )
    )

    assert target_world is not None, status
    assert target_rvec is not None
    assert marker_bbox is not None
    assert marker_angle is not None
    assert image_error[0] == pytest.approx(-0.06, abs=0.08)
    assert image_error[1] == pytest.approx(-0.04, abs=0.08)
    assert np.linalg.norm(target_rvec) > 0.1


def _marker_frame(rotation: int | None = None) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker = cv2.aruco.generateImageMarker(dictionary, ARUCO_MARKER_ID, 120)
    if rotation is not None:
        marker = cv2.rotate(marker, rotation)

    image = np.full((480, 640, 3), 255, dtype=np.uint8)
    y0 = (image.shape[0] - marker.shape[0]) // 2
    x0 = (image.shape[1] - marker.shape[1]) // 2
    image[y0 : y0 + marker.shape[0], x0 : x0 + marker.shape[1]] = cv2.cvtColor(
        marker, cv2.COLOR_GRAY2RGB
    )
    return image


def _perspective_marker_frame() -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    marker = cv2.aruco.generateImageMarker(dictionary, ARUCO_MARKER_ID, 160)
    board = np.full((220, 220), 255, dtype=np.uint8)
    offset = (board.shape[0] - marker.shape[0]) // 2
    board[offset : offset + marker.shape[0], offset : offset + marker.shape[1]] = marker

    image = np.full((720, 1280, 3), 255, dtype=np.uint8)
    src = np.array(
        [
            [0.0, 0.0],
            [float(board.shape[1] - 1), 0.0],
            [float(board.shape[1] - 1), float(board.shape[0] - 1)],
            [0.0, float(board.shape[0] - 1)],
        ],
        dtype=np.float32,
    )
    dst = np.array(
        [
            [475.0, 235.0],
            [780.0, 185.0],
            [720.0, 475.0],
            [520.0, 505.0],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        cv2.cvtColor(board, cv2.COLOR_GRAY2RGB),
        transform,
        (image.shape[1], image.shape[0]),
        borderValue=(255, 255, 255),
    )
    mask = np.any(warped != 255, axis=2)
    image[mask] = warped[mask]
    return image
