"""Streaming conveyor-video processing: detect, track, count, classify.

Frames are decoded one at a time with OpenCV (the video is never held in
memory).  When a track makes a real-observation directional crossing, its
most recent complete, non-merged crop is checked with the original anomaly
algorithm using the detector snapshot fixed at job start.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from . import imaging
from .detection import BackgroundDetector, Detection, DetectionParams
from .features import LAYER2_GRID
from .memory import image_score
from .tracker import MultiObjectTracker, Observation

MAX_VIDEO_BYTES = 200 * 1024 * 1024
VIDEO_EXTENSIONS = (".mp4", ".avi")
MIN_FPS, MAX_FPS = 1.0, 120.0
MIN_FRAME_SIDE = 120
MAX_FRAME_SIDE = 3840
ROI_MARGIN = 6  # evidence crops must sit this far inside the ROI
RECENT_EVIDENCE = 6
EVIDENCE_PAD = 6  # px of belt kept around a piece crop (avoids edge artefacts)
LINE_CLEARANCE = 48  # px: evidence must stay clear of the drawn count line


@dataclass
class VideoConfig:
    roi: tuple[int, int, int, int]
    line_x: int
    direction: str  # "lr" | "rl"
    params: DetectionParams = field(default_factory=DetectionParams)


@dataclass
class PieceResult:
    track_id: int
    time_seconds: float
    frame: int
    score: float | None
    threshold: float | None
    verdict: str  # "ok" | "defect" | "review"
    review_reason: str | None
    evidence_crop: bytes | None
    evidence_heatmap: bytes | None
    crossing_bbox: tuple[int, int, int, int]


def validate_video(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("video could not be opened; use MP4 (H.264/MPEG-4) or AVI")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or not np.isfinite(fps) or not (MIN_FPS <= fps <= MAX_FPS):
        cap.release()
        raise ValueError(
            f"unsupported frame rate {fps}; require {MIN_FPS}-{MAX_FPS} fps"
        )
    if not (MIN_FRAME_SIDE <= width <= MAX_FRAME_SIDE) or not (
        MIN_FRAME_SIDE <= height <= MAX_FRAME_SIDE
    ):
        cap.release()
        raise ValueError(
            f"unsupported frame size {width}x{height}; each side must be in "
            f"[{MIN_FRAME_SIDE}, {MAX_FRAME_SIDE}] px"
        )
    if frames <= 0:
        cap.release()
        raise ValueError("video has no readable frames")
    return cap


def validate_config(config: VideoConfig, width: int, height: int) -> None:
    x, y, w, h = config.roi
    if w < 40 or h < 40:
        raise ValueError("detection region must be at least 40x40 px")
    if not (0 <= x and 0 <= y and x + w <= width and y + h <= height):
        raise ValueError("detection region lies outside the video frame")
    if not (0 <= config.line_x < width):
        raise ValueError("counting line x must lie inside the video frame")
    if config.direction not in ("lr", "rl"):
        raise ValueError("direction must be 'lr' or 'rl'")
    from .detection import validate_params

    validate_params(config.params)


def crop_embedding(extractor, crop_bgr: np.ndarray) -> np.ndarray:
    """L2-normalised mean local descriptor of a piece crop."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    descriptors = extractor.extract_image(imaging.preprocess(image))
    pooled = descriptors.mean(dim=0)
    norm = torch.linalg.vector_norm(pooled)
    if float(norm) > 0:
        pooled = pooled / norm
    return pooled.numpy().astype(np.float32)


def inspect_crop(
    extractor, crop_bgr: np.ndarray, bank: torch.Tensor, threshold: float
) -> dict:
    """Run the original anomaly algorithm on one piece crop."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    descriptors = extractor.extract_image(imaging.preprocess(image))
    score, distances = image_score(descriptors, bank)
    grid = distances.reshape(LAYER2_GRID, LAYER2_GRID).numpy()
    heat = imaging.colorize_distance(grid, image.size, threshold)
    overlay = imaging.overlay_png(image, heat, 0.45)
    return {
        "score": score,
        "threshold": float(threshold),
               "is_anomaly": bool(score > threshold),
        "crop_png": _png_bytes(image),
        "overlay_png": overlay,
    }


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _crop_frame(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return frame[y : y + h, x : x + w].copy()


def _padded_crop(
    frame: np.ndarray, bbox: tuple[int, int, int, int], pad: int = EVIDENCE_PAD
) -> np.ndarray:
    x, y, w, h = bbox
    height, width = frame.shape[:2]
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + w + pad)
    y1 = min(height, y + h + pad)
    return frame[y0:y1, x0:x1].copy()


def _reliable_evidence(det: Detection, roi: tuple[int, int, int, int]) -> bool:
    """A clean, complete, non-merged, inside-ROI observation."""
    if det.merged:
        return False
    x, y, w, h = det.bbox
    rx, ry, rw, rh = roi
    return (
        x >= rx + ROI_MARGIN
        and y >= ry + ROI_MARGIN
        and x + w <= rx + rw - ROI_MARGIN
        and y + h <= ry + rh - ROI_MARGIN
        and det.fill_ratio >= 0.45
    )


def _clear_of_line(bbox: tuple[int, int, int, int], line_x: int) -> bool:
    x, _, w, _ = bbox
    pad = EVIDENCE_PAD
    return x - pad > line_x + LINE_CLEARANCE or x + w + pad < line_x - LINE_CLEARANCE


class CancelledError(RuntimeError):
    pass


def process_video(
    video_path: str | Path,
    background_bgr: np.ndarray,
    config: VideoConfig,
    state,
    snapshot,
    cancel_event,
    progress_cb,
) -> dict:
    """Run the full pipeline; raises CancelledError when stopped.

    The video is decoded frame by frame.  Returns pieces and delta-encoded
    track histories for the playback overlay.
    """
    cap = validate_video(video_path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if background_bgr.shape[1] != width or background_bgr.shape[0] != height:
            raise ValueError(
                "background image must have the same dimensions as the video"
            )
        validate_config(config, width, height)
        detector = BackgroundDetector(background_bgr, config.roi, config.params)
        tracker = MultiObjectTracker(fps, config.line_x, config.direction)

        pieces: list[PieceResult] = []
        recent: dict[int, list[tuple[int, tuple[int, int, int, int]]]] = {}
        frame_index = -1
        current_frame = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            current_frame = frame
            if cancel_event.is_set():
                raise CancelledError()

            detections = detector.detect(frame)
            observations: list[Observation] = []
            clean_by_box: dict[tuple[int, int, int, int], Detection] = {}
            for det in detections:
                if det.merged:
                    # Touching pieces never merge tracks or force a verdict.
                    continue
                embedding = crop_embedding(
                    state.extractor, _padded_crop(frame, det.bbox)
                )
                observations.append(
                    Observation(
                        frame=frame_index,
                        bbox=det.bbox,
                        area=det.area,
                        embedding=embedding,
                    )
                )
                clean_by_box[det.bbox] = det

            _, crossed_ids = tracker.update(frame_index, observations)

            for track in tracker.tracks:
                last = track.history.get(frame_index)
                if last is None or not last[4]:
                    continue
                box = (last[0], last[1], last[2], last[3])
                det = clean_by_box.get(box)
                if (
                    det is not None
                    and _reliable_evidence(det, config.roi)
                    and _clear_of_line(det.bbox, config.line_x)
                ):
                    bucket = recent.setdefault(track.track_id, [])
                    bucket.append((frame_index, box))
                    del bucket[:-RECENT_EVIDENCE]

            for track_id in crossed_ids:
                track = tracker._track(track_id)
                pieces.append(
                    _classify_piece(
                        track,
                        frame,
                        cap,
                        snapshot,
                        state,
                        recent.get(track_id, []),
                    )
                )

            if frame_index % 5 == 0:
                progress_cb(frame_index + 1, total)
        processed = frame_index + 1
        progress_cb(processed, max(total, processed))
    finally:
        cap.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frames": processed,
        "pieces": [_piece_dict(p) for p in pieces],
        "tracks": [
            _encode_track(t)
            for t in tracker.tracks
            if t.crossed or t.hits >= 2
        ],
        "_pieces": pieces,
    }


def _classify_piece(
    track,
    current_frame: np.ndarray,
    cap: cv2.VideoCapture,
    snapshot,
    state,
    candidates: list[tuple[int, tuple[int, int, int, int]]],
) -> PieceResult:
    crossing = track.crossing
    result = PieceResult(
        track_id=track.track_id,
        time_seconds=float(crossing.time_seconds),
        frame=int(crossing.frame),
        score=None,
        threshold=float(snapshot.threshold),
        verdict="review",
        review_reason=None,
        evidence_crop=None,
        evidence_heatmap=None,
        crossing_bbox=crossing.bbox,
    )
    if not candidates:
        result.review_reason = "缺少完整且无粘连的可靠证据裁片"
        return result
    # Nearest-in-time complete, non-merged crop before/at the crossing.
    evidence_frame_idx, box = candidates[-1]
    if evidence_frame_idx == int(crossing.frame):
        evidence_frame = current_frame
    else:
        was = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cap.set(cv2.CAP_PROP_POS_FRAMES, evidence_frame_idx)
        ok, evidence_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, was)
        if not ok:
            result.review_reason = "证据帧无法重新解码"
            return result
    crop = _padded_crop(evidence_frame, box)
    try:
        inspection = inspect_crop(
            state.extractor, crop, snapshot.bank, float(snapshot.threshold)
        )
    except Exception as exc:  # evidence failure is never a fake pass/fail
        result.review_reason = f"证据检测失败: {exc}"
        return result
    result.score = inspection["score"]
    result.verdict = "defect" if inspection["is_anomaly"] else "ok"
    result.evidence_crop = inspection["crop_png"]
    result.evidence_heatmap = inspection["overlay_png"]
    return result


def _piece_dict(piece: PieceResult) -> dict:
    return {
        "track_id": piece.track_id,
        "time_seconds": round(piece.time_seconds, 3),
        "frame": piece.frame,
        "score": None if piece.score is None else round(piece.score, 6),
        "threshold": piece.threshold,
        "verdict": piece.verdict,
        "review_reason": piece.review_reason,
        "has_crop": piece.evidence_crop is not None,
        "has_heatmap": piece.evidence_heatmap is not None,
        "crossing_bbox": list(piece.crossing_bbox),
    }


def _encode_track(track) -> dict:
    """Delta-encode per-frame boxes; final 1/0 flags real vs predicted."""
    frames = sorted(track.history)
    encoded: list[int] = []
    prev_f = -1
    prev_xy = [0, 0, 0, 0]
    for f in frames:
        x, y, w, h, observed = track.history[f]
        encoded.extend(
            [
                f - prev_f - 1,
                x - prev_xy[0],
                y - prev_xy[1],
                w - prev_xy[2],
                h - prev_xy[3],
                1 if observed else 0,
            ]
        )
        prev_f = f
        prev_xy = [x, y, w, h]
    return {
        "track_id": track.track_id,
        "crossed": track.crossed,
        "crossing_frame": track.crossing.frame if track.crossing else None,
        "data": encoded,
    }
