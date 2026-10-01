"""Fixed-camera background-difference segmentation for conveyor pieces.

Frames are streamed one at a time; the full video is never held in memory.
Candidate blobs are filtered by morphology noise removal and validated by
area/bbox size. Touching (merged) clusters are reported separately: they are
never treated as a clean piece, never fed to the tracker and never produce a
pass verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .vision import Detection


@dataclass(frozen=True)
class SegmentConfig:
    roi: tuple[int, int, int, int]  # x1, y1, x2, y2 in full-frame pixels
    diff_threshold: int = 32        # grayscale difference threshold
    blur_sigma: float = 2.0
    min_area: int = 400
    max_piece_area: int = 20000
    min_side: int = 18
    max_side: int = 260
    max_aspect: float = 3.0
    crop_margin: int = 8

    def validate(self, width: int, height: int) -> None:
        x1, y1, x2, y2 = self.roi
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("检测区矩形超出画面或宽高为零")
        if x2 - x1 < 32 or y2 - y1 < 32:
            raise ValueError("检测区过小")
        if self.diff_threshold < 5 or self.diff_threshold > 250:
            raise ValueError("差分阈值应在 5..250 灰度之间")
        if self.min_area <= 0 or self.max_piece_area <= self.min_area:
            raise ValueError("面积过滤参数无效")
        if self.min_side <= 0 or self.max_side <= self.min_side:
            raise ValueError("边长过滤参数无效")
        if self.max_aspect < 1.0:
            raise ValueError("长宽比下限不能小于 1")


class BackgroundSegmenter:
    def __init__(self, background_rgb: np.ndarray, config: SegmentConfig):
        if background_rgb.ndim != 3 or background_rgb.shape[2] != 3:
            raise ValueError("background must be an RGB image")
        h, w = background_rgb.shape[:2]
        config.validate(w, h)
        self.config = config
        background = cv2.cvtColor(background_rgb, cv2.COLOR_RGB2GRAY)
        ksize = int(round(config.blur_sigma * 6)) | 1
        self.background = cv2.GaussianBlur(background, (ksize, ksize), config.blur_sigma)
        x1, y1, x2, y2 = config.roi
        self.roi = (x1, y1, x2, y2)
        self.roi_background = self.background[y1:y2, x1:x2]
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame_rgb: np.ndarray) -> tuple[list[Detection], list[Detection]]:
        x1, y1, x2, y2 = self.roi
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        ksize = int(round(self.config.blur_sigma * 6)) | 1
        blurred = cv2.GaussianBlur(
            gray, (ksize, ksize), self.config.blur_sigma
        )
        diff = cv2.absdiff(blurred[y1:y2, x1:x2], self.roi_background)
        _, mask = cv2.threshold(
            diff, self.config.diff_threshold, 255, cv2.THRESH_BINARY
        )
        # Open removes small noise specks; close fills piece texture gaps.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        clean: list[Detection] = []
        merged: list[Detection] = []
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for label in range(1, n):
            rx, ry, rw, rh, area = stats[label]
            if area < self.config.min_area:
                continue  # residual noise
            bbox = np.array(
                [x1 + rx, y1 + ry, x1 + rx + rw, y1 + ry + rh], dtype=np.int64
            )
            det = Detection(bbox=bbox, area=float(area), merged=False)
            aspect = max(rw, rh) / max(min(rw, rh), 1)
            too_big = (
                area > self.config.max_piece_area
                or rw > self.config.max_side
                or rh > self.config.max_side
            )
            too_small = rw < self.config.min_side or rh < self.config.min_side
            if too_big:
                # Oversized blob: touching pieces or an intrusion. Keep it
                # out of tracking and of any automatic verdict.
                det.merged = True
                merged.append(det)
                continue
            if too_small or aspect > self.config.max_aspect:
                continue  # fails the validated piece shape envelope
            clean.append(det)
        return clean, merged

    def crop_rgb(
        self, frame_rgb: np.ndarray, det: Detection
    ) -> np.ndarray | None:
        h, w = frame_rgb.shape[:2]
        m = self.config.crop_margin
        x1 = max(int(det.bbox[0]) - m, 0)
        y1 = max(int(det.bbox[1]) - m, 0)
        x2 = min(int(det.bbox[2]) + m, w)
        y2 = min(int(det.bbox[3]) + m, h)
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        return np.ascontiguousarray(frame_rgb[y1:y2, x1:x2])
