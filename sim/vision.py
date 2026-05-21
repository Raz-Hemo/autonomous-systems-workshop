from __future__ import annotations

import math
import queue
import threading
import time

import mujoco
import numpy as np

from .config import (
    ARUCO_MARKER_ID,
    ARUCO_MARKER_LENGTH,
    ARUCO_TEXTURE_PATH,
    VISION_FPS,
)

try:
    import cv2
except ImportError:
    cv2 = None


def ensure_aruco_marker_texture() -> bool:
    if cv2 is None or not hasattr(cv2, "aruco"):
        return False

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_size = 384
    marker = np.zeros((marker_size, marker_size), dtype=np.uint8)
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(
            dictionary, ARUCO_MARKER_ID, marker_size, marker, 1
        )
    else:
        marker = cv2.aruco.drawMarker(dictionary, ARUCO_MARKER_ID, marker_size)

    board = np.full((512, 512), 255, dtype=np.uint8)
    offset = (board.shape[0] - marker_size) // 2
    board[offset : offset + marker_size, offset : offset + marker_size] = marker

    ARUCO_TEXTURE_PATH.parent.mkdir(exist_ok=True)
    return bool(cv2.imwrite(str(ARUCO_TEXTURE_PATH), board))


class ArucoVision:
    def __init__(self, model: mujoco.MjModel, camera_id: int) -> None:
        self.enabled = cv2 is not None and hasattr(cv2, "aruco")
        self.camera_id = camera_id
        self.target_world: np.ndarray | None = None
        self.image_error = np.zeros(2)
        self.marker_bbox: tuple[int, int, int, int, int, int] | None = None
        self.last_roi: tuple[int, int, int, int] | None = None
        self.last_update_time = -math.inf
        self.frame_count = 0
        self.pending = False
        self.status = "vision: OpenCV ArUco unavailable"
        self.worker_status = "idle"
        self.worker_mode = "none"

        if not self.enabled:
            return

        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 53
        self.parameters.adaptiveThreshWinSizeStep = 8
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.minMarkerPerimeterRate = 0.015
        self.parameters.maxMarkerPerimeterRate = 0.8
        self.parameters.polygonalApproxAccuracyRate = 0.06
        self.detector = (
            cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
            if hasattr(cv2.aruco, "ArucoDetector")
            else None
        )
        self.fovy = math.radians(float(model.cam_fovy[camera_id]))
        self.input_queue: queue.Queue[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = (
            queue.Queue(maxsize=1)
        )
        self.output_queue: queue.Queue[
            tuple[np.ndarray | None, np.ndarray, tuple[int, int, int, int, int, int] | None, str]
        ] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker_loop, name="aruco-vision", daemon=True)
        self.worker.start()
        self.status = "vision: searching"

    def should_update(self, time: float) -> bool:
        return (
            self.enabled
            and not self.pending
            and time - self.last_update_time >= 1.0 / VISION_FPS
        )

    def submit(
        self,
        image_rgb: np.ndarray,
        camera_pos: np.ndarray,
        camera_mat: np.ndarray,
        time: float,
    ) -> None:
        if not self.enabled:
            self.target_world = None
            return

        self.last_update_time = time
        self.frame_count += 1
        self.pending = True
        self.worker_mode = self._roi_label(image_rgb.shape[1], image_rgb.shape[0])
        self.worker_status = "processing"
        try:
            self.input_queue.put_nowait(
                (image_rgb.copy(), camera_pos.copy(), camera_mat.copy(), time)
            )
        except queue.Full:
            self.pending = False
            self.worker_status = "busy"

    def poll(self) -> None:
        if not self.enabled:
            return

        try:
            target_world, image_error, marker_bbox, status = self.output_queue.get_nowait()
        except queue.Empty:
            return

        self.target_world = target_world
        self.image_error = image_error
        self.marker_bbox = marker_bbox
        self.status = status
        self.pending = False
        self.worker_status = "idle"

    def shutdown(self) -> None:
        if not self.enabled:
            return

        self.stop_event.set()
        self.worker.join(timeout=0.5)

    def overlay_status(self) -> str:
        if not self.enabled:
            return self.status
        return f"{self.status} | worker: {self.worker_status} {self.worker_mode}"

    def _roi_label(self, width: int, height: int) -> str:
        if self.last_roi is None:
            return "full"

        x0, y0, x1, y1 = self.last_roi
        area_ratio = ((x1 - x0) * (y1 - y0)) / max(1, width * height)
        return f"roi {area_ratio * 100:.0f}%"

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                image_rgb, camera_pos, camera_mat, _time = self.input_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            started_at = time.perf_counter()
            target_world, image_error, marker_bbox, status = self._detect(
                image_rgb, camera_pos, camera_mat
            )
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            status = f"{status} {elapsed_ms:.0f}ms"
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    break
            self.output_queue.put((target_world, image_error, marker_bbox, status))

    def _detect(
        self,
        image_rgb: np.ndarray,
        camera_pos: np.ndarray,
        camera_mat: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray, tuple[int, int, int, int, int, int] | None, str]:
        image_rgb = np.flipud(image_rgb)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape

        corners = []
        ids = None
        used_roi = False
        roi_label = "full"
        if self.last_roi is not None:
            x0, y0, x1, y1 = self.last_roi
            roi_gray = gray[y0:y1, x0:x1]
            if roi_gray.size > 0:
                roi_area_ratio = ((x1 - x0) * (y1 - y0)) / max(1, width * height)
                roi_label = f"roi {roi_area_ratio * 100:.0f}%"
                corners, ids = self._detect_markers(roi_gray)
                if ids is not None:
                    offset = np.array([[[x0, y0]]], dtype=np.float32)
                    corners = [corner + offset for corner in corners]
                    used_roi = True

        if ids is None:
            corners, ids = self._detect_markers(gray)
            used_roi = False
            roi_label = "full"

        if ids is None:
            self.last_roi = None
            return None, np.zeros(2), None, "vision: searching"

        marker_indices = np.flatnonzero(ids.reshape(-1) == ARUCO_MARKER_ID)
        if len(marker_indices) == 0:
            self.last_roi = None
            return None, np.zeros(2), None, "vision: marker id not found"

        marker_corners = [corners[int(marker_indices[0])]]
        marker_bbox = self._make_bbox(marker_corners[0], width, height)
        self.last_roi = self._make_roi(marker_corners[0], width, height)
        marker_center = marker_corners[0].reshape(-1, 2).mean(axis=0)
        image_error = np.array(
            [
                (marker_center[0] - width * 0.5) / (width * 0.5),
                (marker_center[1] - height * 0.5) / (height * 0.5),
            ]
        )

        focal = height / (2.0 * math.tan(self.fovy * 0.5))
        camera_matrix = np.array(
            [
                [focal, 0.0, width * 0.5],
                [0.0, focal, height * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.zeros(5)
        _, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            marker_corners, ARUCO_MARKER_LENGTH, camera_matrix, distortion
        )

        tvec_cv = tvecs[0, 0]
        target_camera = np.array([tvec_cv[0], -tvec_cv[1], -tvec_cv[2]])
        target_world = camera_pos + camera_mat @ target_camera
        search_mode = roi_label if used_roi else "full"
        status = (
            f"vision: id {ARUCO_MARKER_ID} "
            f"x={tvec_cv[0]:.2f} y={tvec_cv[1]:.2f} z={tvec_cv[2]:.2f}m "
            f"img=({image_error[0]:+.2f},{image_error[1]:+.2f}) "
            f"@ {VISION_FPS:.0f}Hz {search_mode}"
        )
        return target_world, image_error, marker_bbox, status

    def _detect_markers(
        self, gray: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray | None]:
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.parameters
            )
        return corners, ids

    def _make_roi(
        self, marker_corners: np.ndarray, width: int, height: int
    ) -> tuple[int, int, int, int]:
        points = marker_corners.reshape(-1, 2)
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        marker_span = max(max_xy - min_xy)
        margin = max(140, int(marker_span * 2.2))

        x0 = max(0, int(min_xy[0]) - margin)
        y0 = max(0, int(min_xy[1]) - margin)
        x1 = min(width, int(max_xy[0]) + margin)
        y1 = min(height, int(max_xy[1]) + margin)
        return x0, y0, x1, y1

    def _make_bbox(
        self, marker_corners: np.ndarray, width: int, height: int
    ) -> tuple[int, int, int, int, int, int]:
        points = marker_corners.reshape(-1, 2)
        min_xy = points.min(axis=0)
        max_xy = points.max(axis=0)
        x0 = max(0, int(math.floor(float(min_xy[0]))))
        y0 = max(0, int(math.floor(float(min_xy[1]))))
        x1 = min(width, int(math.ceil(float(max_xy[0]))))
        y1 = min(height, int(math.ceil(float(max_xy[1]))))
        return x0, y0, x1, y1, width, height
