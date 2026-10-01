"""Streaming conveyor video processing: validate, segment, track, inspect."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .detector import DetectorSnapshot
from .features import FeatureExtractor
from .segmentation import BackgroundSegmenter, SegmentConfig
from .vision import MultiObjectTracker

SUPPORTED_VIDEO_FORMATS = ("mp4", "mov", "avi")
MAX_VIDEO_BYTES = 200 * 1024 * 1024
MIN_FPS = 5.0
MAX_FPS = 30.0
MIN_SIDE = 240
MAX_LONG_SIDE = 1920
MAX_DURATION_SEC = 120.0

VIDEO_PREPROCESS_INFO = {
    "accepted_formats": [e.upper() for e in SUPPORTED_VIDEO_FORMATS],
    "max_video_bytes": MAX_VIDEO_BYTES,
    "fps_range": [MIN_FPS, MAX_FPS],
    "min_frame_side": MIN_SIDE,
    "max_frame_long_side": MAX_LONG_SIDE,
    "max_duration_sec": MAX_DURATION_SEC,
    "streaming": "frames decoded one at a time, never whole-video",
}


@dataclass(frozen=True)
class VideoParams:
    roi: tuple[int, int, int, int]
    line_x: int
    direction: str  # ltr | rtl
    diff_threshold: int = 32
    min_area: int = 400
    max_piece_area: int = 20000
    min_side: int = 18
    max_side: int = 160

    def validate(self, width: int, height: int) -> None:
        x1, y1, x2, y2 = self.roi
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("检测区超出画面范围或宽高为零")
        # Require a margin so a piece has a clean, fully visible crop on both
        # sides of the line; a line on the ROI edge cannot be evidenced.
        piece_margin = self.max_side
        if not (x1 + piece_margin <= self.line_x <= x2 - piece_margin):
            raise ValueError(
                "计数线必须位于检测区内部，并与检测区左右边界至少保留一个工件宽度"
            )
        if self.direction not in ("ltr", "rtl"):
            raise ValueError("行进方向必须为 ltr 或 rtl")
        if self.min_area <= 0 or self.max_piece_area <= self.min_area:
            raise ValueError("面积参数无效")
        if self.min_side <= 0 or self.max_side <= self.min_side:
            raise ValueError("工件尺寸参数无效")


class CancelledError(Exception):
    """Raised when the client cancels a video job mid-processing."""


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise CancelledError("job cancelled by user")


@dataclass
class Progress:
    processed_frames: int = 0
    total_frames: int = 0
    counted: int = 0
    on_update: object = None  # optional callback(Progress)

    def as_dict(self) -> dict:
        return {
            "processed_frames": self.processed_frames,
            "total_frames": self.total_frames,
            "percent": (
                round(100.0 * self.processed_frames / self.total_frames, 1)
                if self.total_frames
                else 0.0
            ),
            "counted": self.counted,
        }

    def notify(self) -> None:
        if self.on_update is not None:
            self.on_update(self)


def validate_video_file(path: str, background_path: str) -> dict:
    """Decode metadata only (no frame buffering) and validate geometry."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("无法解码该视频，请使用可打开的 MP4/MOV/AVI")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("视频不含可读取的帧")
    finally:
        cap.release()

    if not (MIN_FPS <= fps <= MAX_FPS):
        raise ValueError(
            f"帧率 {fps:.2f} 超出支持范围 {MIN_FPS:g}..{MAX_FPS:g} fps"
        )
    if min(width, height) < MIN_SIDE or max(width, height) > MAX_LONG_SIDE:
        raise ValueError(
            f"分辨率 {width}x{height} 超出支持范围：短边 >= {MIN_SIDE}，"
            f"长边 <= {MAX_LONG_SIDE}"
        )
    if frames <= 0:
        raise ValueError("无法确定视频帧数（容器缺少帧数索引）")
    duration = frames / fps
    if duration > MAX_DURATION_SEC:
        raise ValueError(f"视频时长 {duration:.1f}s 超过 {MAX_DURATION_SEC:g}s 上限")

    bg = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
    if bg is None:
        raise ValueError("空背景图无法解码")
    bh, bw = bg.shape[:2]
    if (bw, bh) != (width, height):
        raise ValueError(
            f"背景图尺寸 {bw}x{bh} 与视频 {width}x{height} 不一致"
        )
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frames": frames,
        "duration_sec": duration,
    }


def suggest_params(width: int, height: int) -> dict:
    """Default geometry: ROI margins, central vertical line, left-to-right."""
    margin_x = max(24, width // 16)
    margin_y = max(24, height // 8)
    roi = (margin_x, margin_y, width - margin_x, height - margin_y)
    return {
        "roi": list(roi),
        "line_x": width // 2,
        "direction": "ltr",
        "diff_threshold": 32,
        "min_area": 400,
        "max_piece_area": min(max(12000, (width * height) // 24), 20000),
        "min_side": 18,
        "max_side": min(120, (height - 2 * margin_y) - 8),
    }


def process_video(
    video_path: str,
    background_rgb: np.ndarray,
    params: VideoParams,
    extractor: FeatureExtractor,
    snapshot: DetectorSnapshot,
    progress: Progress,
    cancel: CancellationToken,
) -> dict:
    """Run the full streaming pipeline; return serializable job results."""
    params.validate(background_rgb.shape[1], background_rgb.shape[0])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("cannot open video")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress.total_frames = total

    seg_config = SegmentConfig(
        roi=params.roi,
        diff_threshold=params.diff_threshold,
        min_area=params.min_area,
        max_piece_area=params.max_piece_area,
        min_side=params.min_side,
        max_side=params.max_side,
    )
    seg_config.validate(width, height)
    segmenter = BackgroundSegmenter(background_rgb, seg_config)
    tracker = MultiObjectTracker(
        line_x=params.line_x, direction=params.direction
    )

    results: list[dict] = []
    tracks_path: dict[int, list[list[float]]] = {}
    merged_frames: list[int] = []
    frame_index = 0
    try:
        while True:
            cancel.raise_if_cancelled()
            ok, bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            time_sec = frame_index / fps
            clean, merged = segmenter.detect(frame_rgb)
            if merged:
                merged_frames.append(frame_index)
            crops = [segmenter.crop_rgb(frame_rgb, d) for d in clean]
            descriptions = [_appearance(extractor, crop) for crop in crops]
            events = tracker.update(
                clean,
                descriptions,
                frame_index,
                time_sec,
                crops,
                (width, height),
            )
            for event in events:
                record = _classify(event, snapshot, extractor, line_x=params.line_x)
                results.append(record)
                progress.counted = len(results)

            for track in tracker.tracks:
                if (
                    track.confirmed
                    and track.last_obs_frame == frame_index
                    and track.last_obs_bbox is not None
                ):
                    cx = (track.last_obs_bbox[0] + track.last_obs_bbox[2]) / 2.0
                    cy = (track.last_obs_bbox[1] + track.last_obs_bbox[3]) / 2.0
                    points = tracks_path.setdefault(track.track_id, [])
                    points.append([frame_index, round(float(cx), 1), round(float(cy), 1)])

            frame_index += 1
            progress.processed_frames = frame_index
            if frame_index % 10 == 0:
                progress.notify()
    finally:
        cap.release()
    progress.notify()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frames_processed": frame_index,
        "total_frames": total,
        "line_x": params.line_x,
        "roi": list(params.roi),
        "direction": params.direction,
        "tracks": [
            {"track_id": tid, "points": points}
            for tid, points in sorted(tracks_path.items())
        ],
        "merged_frames": merged_frames,
        "results": results,
        "detector": {
            "built_at": snapshot.built_at,
            "bank_sha256": snapshot.bank_sha256,
            "threshold": snapshot.threshold,
        },
    }


def _appearance(extractor: FeatureExtractor, crop: np.ndarray | None):
    if crop is None:
        return None
    from .imaging import preprocess

    image = Image.fromarray(crop)
    desc = extractor.extract_image(preprocess(image)).mean(dim=0)
    vector = desc.numpy()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _classify(
    event: dict,
    snapshot: DetectorSnapshot,
    extractor: FeatureExtractor,
    line_x: int,
) -> dict:
    crop = event.get("evidence_crop")
    record = {
        "track_id": event["track_id"],
        "time_sec": round(event["time_sec"], 3),
        "frame": event["frame"],
        "evidence_frame": event.get("evidence_frame"),
        "bbox": event["bbox"],
        "line_x": line_x,
        "score": None,
        "threshold": snapshot.threshold,
        "verdict": "review_required",
        "reason": "",
    }
    if crop is None:
        record["reason"] = "缺少完整、无粘连的工件裁剪证据"
        return record
    try:
        outcome = snapshot.inspect_pil(extractor, Image.fromarray(crop))
    except Exception as exc:
        record["reason"] = f"证据检测失败：{exc}"
        return record
    record["score"] = round(outcome["score"], 6)
    record["verdict"] = "anomaly" if outcome["is_anomaly"] else "ok"
    record["evidence_shape"] = list(crop.shape)
    record["heat"] = outcome["heat"]
    record["crop"] = crop
    return record
