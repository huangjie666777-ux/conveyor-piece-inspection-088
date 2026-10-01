"""Piece identity tracking and directional line-crossing counting.

Each track combines:

* a constant-velocity Kalman filter on (cx, cy, w, h) for motion prediction
  (OpenCV's cv2.KalmanFilter);
* an appearance descriptor (L2-normalised) maintained as an exponential moving
  average of the existing feature extractor's crop descriptors;
* one-to-one Hungarian assignment (SciPy) on a Mahalanobis + appearance cost,
  with hard gates that reject implausible pairs.

Missed detections coast the track for a few frames (prediction only) so short
occlusions keep the identity.  Predicted boxes are never counted: a crossing
is registered only on a real observation, only in the configured direction,
and at most once per track (jitter and reverse re-crossing cannot double
count).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Chi-square gate with 4 degrees of freedom (cx, cy, w, h residual).
# The 99.5% quantile is 14.86; 16 leaves headroom for 2-3 frame coasts while
# still rejecting jumps to unrelated candidates (which also fail appearance).
MAHALANOBIS_GATE = 16.0
MIN_APPEARANCE_COSINE = 0.15
MAX_MISSED_FRAMES = 8


@dataclass
class Observation:
    frame: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area: float
    embedding: np.ndarray | None = None


@dataclass
class Crossing:
    frame: int
    time_seconds: float
    bbox: tuple[int, int, int, int]


@dataclass
class Track:
    track_id: int
    kf: cv2.KalmanFilter
    embedding: np.ndarray | None = None
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    age: int = 0
    hits: int = 0
    missed: int = 0
    last_real_frame: int = -10
    last_real_cx: float | None = None
    crossed: bool = False
    crossing: Crossing | None = None
    # frame -> (x, y, w, h, observed): real vs coasted prediction.
    history: dict[int, tuple[int, int, int, int, bool]] = field(default_factory=dict)
    active: bool = True

    def predicted_bbox(self) -> tuple[int, int, int, int]:
        state = self.kf.statePost.reshape(-1)
        cx, cy, w, h = state[0], state[1], state[2], state[3]
        return _box_from_center(cx, cy, max(float(w), 1.0), max(float(h), 1.0))


def _box_from_center(cx, cy, w, h) -> tuple[int, int, int, int]:
    return (
        int(round(cx - w / 2.0)),
        int(round(cy - h / 2.0)),
        int(round(w)),
        int(round(h)),
    )


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def _make_filter(delta_t: float, obs: Observation) -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(6, 4)
    kf.transitionMatrix = np.array(
        [
            [1, 0, 0, 0, delta_t, 0],
            [0, 1, 0, 0, 0, delta_t],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    kf.measurementMatrix = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ],
        dtype=np.float32,
    )
    # Velocity process noise is large enough that a 2-3 frame coast still
    # leaves the re-acquired observation inside the Mahalanobis gate.
    kf.processNoiseCov = np.diag(
        np.array([1, 1, 0.5, 0.5, 40, 40], dtype=np.float32)
    )
    kf.measurementNoiseCov = np.diag(
        np.array([4, 4, 4, 4], dtype=np.float32)
    )
    cx, cy = box_center(obs.bbox)
    _, _, w, h = obs.bbox
    kf.statePost = np.array(
        [cx, cy, w, h, 0, 0], dtype=np.float32
    ).reshape(6, 1)
    kf.errorCovPost = np.diag(
        np.array([4, 4, 9, 9, 25, 25], dtype=np.float32)
    )
    return kf


class MultiObjectTracker:
    """Kalman + appearance tracker with strict directional counting."""

    def __init__(
        self,
        fps: float,
        line_x: int,
        direction: str,
        max_missed: int = MAX_MISSED_FRAMES,
    ):
        if direction not in ("lr", "rl"):
            raise ValueError("direction must be 'lr' (left to right) or 'rl'")
        if not 0.0 < fps <= 240.0:
            raise ValueError("fps must be in (0, 240]")
        self.delta_t = 1.0 / float(fps)
        self.line_x = float(line_x)
        self.direction = direction
        self.max_missed = max_missed
        self.tracks: list[Track] = []
        self._next_id = 1
        self.newly_crossed: list[Track] = []

    def update(
        self, frame: int, observations: list[Observation]
    ) -> tuple[list[Track], list[int]]:
        """Advance one frame.

        observations must already exclude merged/unreliable candidates.
        Returns (tracks, crossed_track_ids); crossings are only emitted in the
        frame they happen, and only for confirmed real observations.
        """
        self.newly_crossed = []
        predicted: dict[int, np.ndarray] = {}
        for track in self.tracks:
            if not track.active:
                continue
            track.age += 1
            predicted[track.track_id] = track.kf.predict().reshape(-1)

        assignments = self._assign(observations, predicted)
        matched_tracks = {t_id for t_id, _ in assignments}
        matched_obs = {o_idx for _, o_idx in assignments}

        for track_id, obs_idx in assignments:
            track = self._track(track_id)
            obs = observations[obs_idx]
            cx, cy = box_center(obs.bbox)
            _, _, w, h = obs.bbox
            measurement = np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1)
            track.kf.correct(measurement)
            track.missed = 0
            track.hits += 1
            track.bbox = obs.bbox
            track.history[frame] = (*obs.bbox, True)
            self._update_embedding(track, obs.embedding)
            self._check_crossing(track, frame, cx)
            track.last_real_frame = frame
            track.last_real_cx = cx

        for track in self.tracks:
            if not track.active or track.track_id in matched_tracks:
                continue
            track.missed += 1
            state = predicted[track.track_id]
            track.bbox = _box_from_center(
                state[0], state[1], max(float(state[2]), 1.0),
                max(float(state[3]), 1.0),
            )
            track.history[frame] = (*track.bbox, False)
            # This runs after assignment: a track that reaches exactly
            # max_missed may still have been re-acquired in this same frame
            # (matched tracks skip this branch).  Retire only when the
            # following frame also has no observation.
            if track.missed > self.max_missed:
                track.active = False

        for obs_idx, obs in enumerate(observations):
            if obs_idx not in matched_obs:
                track = Track(
                    track_id=self._next_id,
                    kf=_make_filter(self.delta_t, obs),
                    embedding=(
                        obs.embedding.copy()
                        if obs.embedding is not None
                        else None
                    ),
                    bbox=obs.bbox,
                    age=1,
                    hits=1,
                    missed=0,
                    last_real_frame=frame,
                    last_real_cx=box_center(obs.bbox)[0],
                )
                track.history[frame] = (*obs.bbox, True)
                self._next_id += 1
                self.tracks.append(track)

        return self.tracks, [t.track_id for t in self.newly_crossed]

    def _assign(
        self,
        observations: list[Observation],
        predicted: dict[int, np.ndarray],
    ) -> list[tuple[int, int]]:
        active = [t for t in self.tracks if t.active]
        if not active or not observations:
            return []
        cost = np.full(
            (len(active), len(observations)), np.inf, dtype=np.float64
        )
        for i, track in enumerate(active):
            cov = track.kf.errorCovPre
            innovation_cov = (
                track.kf.measurementMatrix
                @ cov
                @ track.kf.measurementMatrix.T
                + track.kf.measurementNoiseCov
            )
            inv_cov = np.linalg.pinv(innovation_cov)
            state = predicted[track.track_id]
            for j, obs in enumerate(observations):
                cx, cy = box_center(obs.bbox)
                z = np.array([cx, cy, obs.bbox[2], obs.bbox[3]], dtype=np.float64)
                zhat = state[[0, 1, 2, 3]].astype(np.float64)
                delta = z - zhat
                mahal = float(np.sqrt(max(delta @ inv_cov @ delta, 0.0)))
                if mahal > MAHALANOBIS_GATE:
                    continue
                if track.embedding is not None and obs.embedding is not None:
                    cosine = float(track.embedding @ obs.embedding)
                    if cosine < MIN_APPEARANCE_COSINE:
                        continue
                    appearance = 1.0 - cosine
                else:
                    appearance = 0.0
                cost[i, j] = mahal * 0.25 + appearance * 8.0
        finite = np.isfinite(cost)
        safe = np.where(finite, cost, 1e9)
        row_ind, col_ind = linear_sum_assignment(safe)
        assignments: list[tuple[int, int]] = []
        for i, j in zip(row_ind, col_ind):
            if finite[i, j]:
                assignments.append((active[i].track_id, int(j)))
        return assignments

    @staticmethod
    def _update_embedding(track: Track, embedding: np.ndarray | None) -> None:
        if embedding is None:
            return
        if track.embedding is None:
            track.embedding = embedding.copy()
        else:
            merged = 0.7 * track.embedding + 0.3 * embedding
            track.embedding = merged / np.linalg.norm(merged)

    def _check_crossing(self, track: Track, frame: int, cx: float) -> None:
        # Only real observations reach this method.  A crossing needs the
        # previous real observation on the entry side, so a track first seen
        # beyond the line cannot count; the crossed latch prevents jitter or
        # reverse re-crossing from counting twice.
        if track.crossed or track.last_real_cx is None:
            return
        prev_cx = track.last_real_cx
        if self.direction == "lr":
            crossed = prev_cx <= self.line_x < cx
        else:
            crossed = prev_cx >= self.line_x > cx
        if crossed:
            track.crossed = True
            track.crossing = Crossing(
                frame=frame,
                time_seconds=frame * self.delta_t,
                bbox=track.bbox,
            )
            self.newly_crossed.append(track)

    def _track(self, track_id: int) -> Track:
        return next(t for t in self.tracks if t.track_id == track_id)
