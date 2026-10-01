"""Tests for background detection, Kalman/appearance tracking and counting."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from anomaly_inspector.detection import (
    BackgroundDetector, DetectionParams, validate_params,
)
from anomaly_inspector.tracker import MultiObjectTracker, Observation

W, H = 320, 200
LINE_X = 160
ROI = (10, 10, W - 20, H - 20)


def _frame(pieces):
    frame = np.full((H, W, 3), 40, dtype=np.uint8)
    for cx, cy, size, gray in pieces:
        x0, y0 = int(cx - size / 2), int(cy - size / 2)
        frame[max(0, y0):y0 + size, max(0, x0):x0 + size] = gray
    return frame


def test_detection_filters_noise_and_marks_merged():
    bg = _frame([])
    detector = BackgroundDetector(
        bg, ROI, DetectionParams(min_area=500, max_area=4000, min_side=15)
    )
    frame = bg.copy()
    cv2.rectangle(frame, (60, 60), (64, 64), 230, -1)
    cv2.rectangle(frame, (120, 70), (170, 120), (250, 250, 250), -1)
    cv2.rectangle(frame, (200, 60), (269, 130), (250, 250, 250), -1)
    dets = detector.detect(frame)
    merged_flags = sorted(d.merged for d in dets)
    assert len(dets) == 2
    assert merged_flags == [False, True]


def test_param_validation():
    with pytest.raises(ValueError):
        validate_params(DetectionParams(min_area=10, max_area=5))
    with pytest.raises(ValueError):
        validate_params(DetectionParams(diff_threshold=300))


def test_tracker_identity_and_single_directional_count():
    tracker = MultiObjectTracker(10.0, LINE_X, "lr", max_missed=3)
    crossed = []
    emb1 = np.array([1.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.0, 1.0], dtype=np.float32)

    def obs(frame_idx, cx, emb):
        return Observation(frame_idx, (int(cx - 20), 80, 40, 40), 1600.0, emb)

    seq1 = [40, 60, 80, None, None, 140, 180]
    for f, cx in enumerate(seq1):
        observations = [] if cx is None else [obs(f, cx, emb1)]
        _, ids = tracker.update(f, observations)
        crossed.extend(ids)
    assert crossed == [1]

    _, ids = tracker.update(7, [obs(7, 150, emb1)])
    assert ids == []
    _, ids = tracker.update(8, [obs(8, 180, emb1)])
    assert ids == []

    _, ids = tracker.update(9, [obs(9, 180, emb2)])
    assert ids == []
    for f, cx in zip([10, 11, 12], [150, 100, 50]):
        _, ids = tracker.update(f, [obs(f, cx, emb2)])
        assert 2 not in ids


def test_predicted_crossing_does_not_count():
    tracker = MultiObjectTracker(10.0, LINE_X, "lr", max_missed=5)
    for f, cx in enumerate([60, 90, 120, 140]):
        tracker.update(f, [Observation(f, (cx - 15, 80, 30, 30), 900.0)])
    for f in range(4, 9):
        _, ids = tracker.update(f, [])
        assert ids == []
    assert not tracker.tracks[0].crossed


def test_appearance_gate_rejects_swap():
    tracker = MultiObjectTracker(10.0, LINE_X, "lr")
    e1 = np.array([1.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 1.0], dtype=np.float32)
    tracker.update(0, [Observation(0, (50, 80, 30, 30), 900, e1)])
    tracker.update(1, [Observation(1, (52, 80, 30, 30), 900, e2)])
    assert len(tracker.tracks) == 2
