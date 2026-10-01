"""Fixed-camera background-difference candidate detection on a conveyor.

The camera and an empty-belt background image are required.  Only the
configured rectangular detection region (ROI) is analysed; frames are read
one at a time by the caller and never buffered wholesale.

Detected blobs are validated by pixel area, side length and fill ratio.
Touching/overlapping pieces produce one *merged* candidate: it is flagged and
excluded from tracker association and from quality decisions instead of being
treated as one piece or merged into an existing track.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionParams:
    min_area: int = 1200
    max_area: int = 6500
    min_side: int = 24
    diff_threshold: int = 35
    blur_sigma: float = 1.2
    morph_iterations: int = 2
    min_fill_ratio: float = 0.35
    split_erode_px: int = 15


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]  # x, y, w, h in full-frame coordinates
    area: float
    fill_ratio: float
    merged: bool
    mask: np.ndarray  # foreground mask cropped to the bbox (uint8 0/255)


class BackgroundDetector:
    """Per-frame background subtraction restricted to an ROI."""

    def __init__(
        self,
        background_bgr: np.ndarray,
        roi: tuple[int, int, int, int],
        params: DetectionParams,
    ):
        x, y, w, h = roi
        if w <= 0 or h <= 0:
            raise ValueError("detection region must have positive size")
        self.roi = (int(x), int(y), int(w), int(h))
        self.params = params
        bg = self._prepare(background_bgr)
        x0, y0, rw, rh = self.roi
        if not (0 <= x0 and 0 <= y0 and x0 + rw <= bg.shape[1] and y0 + rh <= bg.shape[0]):
            raise ValueError("detection region lies outside the video frame")
        self.bg_roi = bg[y0 : y0 + rh, x0 : x0 + rw]
        kernel_size = max(3, int(round(params.blur_sigma * 4)) | 1)
        self._blur = (kernel_size, kernel_size)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def _prepare(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        ksize = max(3, int(round(self.params.blur_sigma * 4)) | 1)
        return cv2.GaussianBlur(gray, (ksize, ksize), self.params.blur_sigma)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        p = self.params
        x0, y0, rw, rh = self.roi
        roi = frame_bgr[y0 : y0 + rh, x0 : x0 + rw]
        if roi.shape[:2] != self.bg_roi.shape:
            raise ValueError("frame size does not match the background image")
        gray = self._prepare(frame_bgr)[y0 : y0 + rh, x0 : x0 + rw]
        diff = cv2.absdiff(gray, self.bg_roi)
        _, mask = cv2.threshold(diff, p.diff_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self._kernel,
            iterations=max(1, p.morph_iterations - 1),
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, self._kernel, iterations=p.morph_iterations
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        out: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < p.min_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bw < p.min_side or bh < p.min_side:
                continue
            box_area = float(bw * bh)
            fill = area / box_area if box_area else 0.0
            if fill < p.min_fill_ratio:
                continue
            cm = np.zeros((bh, bw), dtype=np.uint8)
            shifted = contour - np.array([[[bx, by]]], dtype=np.int32)
            cv2.drawContours(cm, [shifted], -1, 255, thickness=cv2.FILLED)
            merged = area > p.max_area or self._splits_into_parts(cm, p)
            out.append(
                Detection(
                    bbox=(x0 + bx, y0 + by, int(bw), int(bh)),
                    area=area,
                    fill_ratio=fill,
                    merged=merged,
                    mask=cm,
                )
            )
        return out

    @staticmethod
    def _splits_into_parts(crop_mask: np.ndarray, p: DetectionParams) -> bool:
        """Heuristic: erosion that splits a blob reveals touching pieces."""
        size = max(3, int(p.split_erode_px) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        eroded = cv2.erode(crop_mask, kernel, iterations=1)
        count, _ = cv2.connectedComponents(eroded)
        # count includes the background label; two surviving cores = merged.
        return count - 1 >= 2


def validate_params(params: DetectionParams) -> None:
    if not (0 < params.min_area <= params.max_area):
        raise ValueError("require 0 < min_area <= max_area")
    if params.min_side <= 0:
        raise ValueError("min_side must be positive")
    if not (1 <= params.diff_threshold <= 254):
        raise ValueError("diff_threshold must be in [1, 254]")
    if not 0.0 < params.min_fill_ratio <= 1.0:
        raise ValueError("min_fill_ratio must be in (0, 1]")
    if params.morph_iterations < 1 or params.morph_iterations > 10:
        raise ValueError("morph_iterations must be in [1, 10]")
