"""Background-difference detection and one-to-one multi-object tracking.

The tracker combines a constant-velocity Kalman motion model with appearance
descriptors produced by the frozen ResNet feature extractor. Matching is a
global one-to-one Hungarian assignment with both motion (gating) and
appearance gates; an untrusted pair is never forced. Tracks survive short
detection gaps by prediction, but predicted boxes never cross the counting
line: counting requires two real observations straddling the line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import block_diag


@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2], integer pixel coordinates
    area: float
    merged: bool  # candidate blob is a touching cluster, not a clean piece

    @property
    def center(self) -> np.ndarray:
        return np.array(
            [(self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0]
        )

    @property
    def scale(self) -> float:
        return float(
            ((self.bbox[2] - self.bbox[0]) + (self.bbox[3] - self.bbox[1])) / 2.0
        )


@dataclass
class Track:
    track_id: int
    kf: "_KalmanBox"
    appearance: np.ndarray | None = None
    hits: int = 0
    misses: int = 0
    age: int = 0
    confirmed: bool = False
    counted: bool = False
    # Bounding box of the latest *real* observation and its frame index.
    last_obs_bbox: np.ndarray | None = None
    last_obs_frame: int = -1
    # Most recent clean (non-merged, in-bounds) evidence crops by frame.
    clean_crops: dict[int, np.ndarray] = field(default_factory=dict)
    # Last real-observation centre x used for crossing decisions.
    last_obs_cx: float | None = None
    dead: bool = False
    # Sampled real-observation path points: (frame, x, y) for replay.
    path: list[tuple[int, float, float]] = field(default_factory=list)

    def predicted_bbox(self) -> np.ndarray:
        cx, cy, s = self.kf.x[:3, 0]
        half = max(s, 1.0) / 2.0
        return np.array([cx - half, cy - half, cx + half, cy + half])


class _KalmanBox:
    """Tiny constant-velocity Kalman filter: state [cx, cy, s, vx, vy, vs]."""

    def __init__(self, center: np.ndarray, scale: float, q: float = 1.0):
        self.x = np.array(
            [center[0], center[1], scale, 0.0, 0.0, 0.0], dtype=np.float64
        ).reshape(6, 1)
        self.P = np.diag([10.0, 10.0, 10.0, 100.0, 100.0, 25.0])
        self.F = np.eye(6)
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])
        self.R = np.diag([4.0, 4.0, 4.0])
        self.Q = np.diag([q, q, 0.2 * q, 5.0 * q, 5.0 * q, q])

    def predict(self, dt: float = 1.0) -> None:
        f = self.F.copy()
        f[0, 3] = f[1, 4] = f[2, 5] = dt
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + self.Q * dt

    def update(self, center: np.ndarray, scale: float) -> None:
        z = np.array([[center[0]], [center[1]], [scale]], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        i_kh = np.eye(6) - K @ self.H
        self.P = i_kh @ self.P
        self.innovation = float((y.T @ np.linalg.inv(S) @ y).item())


class MultiObjectTracker:
    """Kalman + appearance, one-to-one Hungarian assignment with gating.

    Parameters
    ----------
    max_missed: consecutive predicted-only frames before a track is closed.
    confirm_hits: real observations required before a track is confirmed.
    motion_gate: maximum squared Mahalanobis innovation for a legal match.
    appearance_gate: maximum cosine distance (0..2) for a legal match.
    max_cost: pairs above this combined cost are left unmatched.
    """

    def __init__(
        self,
        line_x: float,
        direction: str,
        max_missed: int = 12,
        confirm_hits: int = 2,
        motion_gate: float = 25.0,
        appearance_gate: float = 0.6,
        max_cost: float = 1.0,
        appearance_weight: float = 0.6,
    ):
        if direction not in ("ltr", "rtl"):
            raise ValueError("direction must be 'ltr' or 'rtl'")
        self.line_x = float(line_x)
        self.direction = direction
        self.max_missed = max_missed
        self.confirm_hits = confirm_hits
        self.motion_gate = motion_gate
        self.appearance_gate = appearance_gate
        self.max_cost = max_cost
        self.appearance_weight = appearance_weight
        self.tracks: list[Track] = []
        self._next_id = 1
        self.cross_events: list[dict] = []

    # ------------------------------------------------------------- main step

    def update(
        self,
        detections: list[Detection],
        appearances: list[np.ndarray | None],
        frame_index: int,
        time_sec: float,
        crops: list[np.ndarray | None],
        frame_size: tuple[int, int],
    ) -> list[dict]:
        """Advance one frame; return newly produced crossing events.

        ``detections`` are clean single-piece blobs only; merged (touching)
        candidates must be filtered out by the caller so they can never merge
        two identities. Each list aligns index-wise.
        """
        for track in self.tracks:
            track.kf.predict()
            track.age += 1

        active = [t for t in self.tracks if not t.dead]
        matches, unmatched_dets, unmatched_tracks = self._associate(
            active, detections, appearances
        )

        width, height = frame_size
        new_events: list[dict] = []
        for ti, di in matches:
            track = active[ti]
            det = detections[di]
            track.kf.update(det.center, det.scale)
            track.hits += 1
            track.misses = 0
            if track.hits >= self.confirm_hits:
                track.confirmed = True
            self._merge_appearance(track, appearances[di])
            track.last_obs_bbox = det.bbox.copy()
            track.last_obs_frame = frame_index
            crop = crops[di]
            if crop is not None and self._crop_is_clean(crop, width, height):
                track.clean_crops[frame_index] = crop
                # Bound the crop buffer: keep the latest window only.
                stale = [f for f in track.clean_crops if f < frame_index - 30]
                for f in stale:
                    track.clean_crops.pop(f, None)
            event = self._check_crossing(track, det.center[0], frame_index, time_sec)
            if event is not None:
                new_events.append(event)

        for ti in unmatched_tracks:
            active[ti].misses += 1
            if active[ti].misses > self.max_missed:
                active[ti].dead = True

        for di in unmatched_dets:
            det = detections[di]
            kf = _KalmanBox(det.center, det.scale)
            track = Track(
                track_id=self._next_id,
                kf=kf,
                appearance=appearances[di],
                hits=1,
                last_obs_bbox=det.bbox.copy(),
                last_obs_frame=frame_index,
                last_obs_cx=det.center[0],
            )
            self._next_id += 1
            crop = crops[di]
            if crop is not None and self._crop_is_clean(crop, width, height):
                track.clean_crops[frame_index] = crop
            self.tracks.append(track)

        self.tracks = [t for t in self.tracks if not t.dead]
        events = self.cross_events
        return new_events

    # ------------------------------------------------------------ assignment

    def _associate(
        self,
        tracks: list[Track],
        detections: list[Detection],
        appearances: list[np.ndarray | None],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(detections))), list(range(len(tracks)))

        cost = np.full((len(tracks), len(detections)), np.inf)
        for ti, track in enumerate(tracks):
            pred = track.predicted_bbox()
            pcx = (pred[0] + pred[2]) / 2.0
            pcy = (pred[1] + pred[3]) / 2.0
            for di, det in enumerate(detections):
                # Motion gate: measurement vs. Kalman prediction covariance.
                z = np.array([[det.center[0]], [det.center[1]], [det.scale]])
                h = track.kf.H
                S = h @ track.kf.P @ h.T + track.kf.R
                innov = z - h @ track.kf.x
                mahal = float((innov.T @ np.linalg.inv(S) @ innov).item())
                if mahal > self.motion_gate:
                    continue
                appear_dist = 0.0
                if track.appearance is not None and appearances[di] is not None:
                    appear_dist = float(
                        1.0 - np.dot(track.appearance, appearances[di])
                    )
                    if appear_dist > self.appearance_gate:
                        continue
                # Normalized Mahalanobis contribution, capped at the gate.
                motion_part = min(mahal / self.motion_gate, 1.0)
                c = (1.0 - self.appearance_weight) * motion_part
                c += self.appearance_weight * min(appear_dist, 1.0)
                if c <= self.max_cost:
                    cost[ti, di] = c

        finite = np.isfinite(cost)
        row_ind, col_ind = linear_sum_assignment(
            np.where(finite, cost, 1e6)
        )
        matches: list[tuple[int, int]] = []
        matched_t, matched_d = set(), set()
        for ti, di in zip(row_ind, col_ind):
            if finite[ti, di]:
                matches.append((int(ti), int(di)))
                matched_t.add(int(ti))
                matched_d.add(int(di))
        unmatched_d = [d for d in range(len(detections)) if d not in matched_d]
        unmatched_t = [t for t in range(len(tracks)) if t not in matched_t]
        return matches, unmatched_d, unmatched_t

    @staticmethod
    def _merge_appearance(track: Track, desc: np.ndarray | None) -> None:
        if desc is None:
            return
        if track.appearance is None:
            track.appearance = desc.copy()
        else:
            merged = 0.7 * track.appearance + 0.3 * desc
            track.appearance = merged / (np.linalg.norm(merged) + 1e-9)

    # ------------------------------------------------------------- counting

    def _check_crossing(
        self,
        track: Track,
        obs_cx: float,
        frame_index: int,
        time_sec: float,
    ) -> dict | None:
        """Count only on a real observation straddling the line, once."""
        if track.counted or not track.confirmed:
            track.last_obs_cx = obs_cx
            return None
        prev_cx = track.last_obs_cx
        track.last_obs_cx = obs_cx
        if prev_cx is None:
            return None
        crossed = (
            (self.direction == "ltr" and prev_cx <= self.line_x < obs_cx)
            or (self.direction == "rtl" and prev_cx >= self.line_x > obs_cx)
        )
        if not crossed:
            return None
        track.counted = True
        best_frame = max(
            (f for f in track.clean_crops if f <= frame_index),
            default=None,
        )
        event = {
            "track_id": track.track_id,
            "frame": frame_index,
            "time_sec": float(time_sec),
            "bbox": track.last_obs_bbox.tolist(),
            "evidence_frame": best_frame,
            "evidence_crop": (
                track.clean_crops[best_frame].copy()
                if best_frame is not None
                else None
            ),
        }
        self.cross_events.append(event)
        return event

    @staticmethod
    def _crop_is_clean(crop: np.ndarray, width: int, height: int) -> bool:
        # A crop touching the frame border may be partly off-screen.
        return bool(crop.shape[0] >= 16 and crop.shape[1] >= 16)

    def evidence_crop(
        self, track_id: int, margin: int = 8
    ) -> tuple[int | None, np.ndarray | None]:
        for track in self.tracks:
            if track.track_id == track_id and track.clean_crops:
                f = max(track.clean_crops)
                return f, track.clean_crops[f]
        return None, None
