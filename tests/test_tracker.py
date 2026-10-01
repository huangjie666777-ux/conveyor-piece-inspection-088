import numpy as np

from anomaly_inspector.vision import Detection, MultiObjectTracker


def _det(cx, cy, side=40.0):
    half = side / 2.0
    return Detection(
        bbox=np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.int64),
        area=side * side,
        merged=False,
    )


def _desc(seed, dim=16):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


CROP = np.zeros((40, 40, 3), np.uint8)


def _step(tracker, frame, dets, descs=None, crops=None, size=(400, 300)):
    descs = descs or [_desc(i + 1) for i in range(len(dets))]
    crops = crops if crops is not None else [CROP for _ in dets]
    return tracker.update(dets, descs, frame, frame / 10.0, crops, size)


def test_direction_cross_counts_once_per_track():
    tracker = MultiObjectTracker(line_x=100, direction="ltr")
    events = []
    for frame in range(12):
        events += _step(tracker, frame, [_det(20 + frame * 12, 100)])
    assert len(events) == 1
    assert events[0]["track_id"] == 1

    # Jitter/re-crossing the same line never counts the same track twice.
    tracker2 = MultiObjectTracker(line_x=100, direction="ltr")
    xs = [20, 60, 95, 105, 90, 105, 110, 98, 108, 130]
    total = []
    for frame, cx in enumerate(xs):
        total += _step(tracker2, frame, [_det(cx, 100)])
    assert len(total) == 1

    # Right-to-left direction ignores left-to-right movers.
    tracker3 = MultiObjectTracker(line_x=100, direction="rtl")
    total = []
    for frame in range(12):
        total += _step(tracker3, frame, [_det(20 + frame * 12, 100)])
    assert total == []


def test_short_gaps_keep_identity():
    tracker = MultiObjectTracker(line_x=300, direction="ltr")
    events = []
    for frame in range(26):
        cx = 20 + frame * 12
        dets = [] if 10 <= frame <= 13 else [_det(cx, 100)]
        events += _step(tracker, frame, dets)
    assert len(events) == 1
    assert events[0]["track_id"] == 1


def test_one_to_one_assignment_and_appearance_gate():
    # Two consistent identities at different rows; one-to-one assignment
    # must keep the upper and lower tracks distinct even when close in x.
    tracker = MultiObjectTracker(line_x=500, direction="ltr")
    ids_per_frame = []
    for frame in range(6):
        d1 = _det(40 + frame * 10, 80)
        d2 = _det(60 + frame * 10, 200)
        events = _step(
            tracker, frame, [d1, d2], [_desc(1), _desc(2)],
        )
        ids_per_frame.append(events)
    track_ids = {t.track_id for t in tracker.tracks}
    assert track_ids == {1, 2}

    # A far-away appearance description must be rejected by the appearance
    # gate even when the motion gate would accept the position.
    tracker3 = MultiObjectTracker(
        line_x=500, direction="ltr", appearance_gate=0.05
    )
    _step(tracker3, 0, [_det(100, 100)], [_desc(1)])
    _step(tracker3, 1, [_det(110, 100)], [_desc(2)])
    # The mismatched-looking object starts a new identity rather than being
    # force-merged with the existing track.
    assert len(tracker3.tracks) == 2


def test_predicted_box_does_not_count():
    tracker = MultiObjectTracker(line_x=100, direction="ltr", max_missed=10)
    events = []
    for frame, cx in [(0, 20), (1, 40), (2, 60), (3, 80)]:
        events += _step(tracker, frame, [_det(cx, 100)])
    # Then predictions alone would cross the line; they must not count.
    for frame in range(4, 12):
        events += _step(tracker, frame, [])
    assert events == []


def test_missing_evidence_marks_review():
    tracker = MultiObjectTracker(line_x=100, direction="ltr")
    events = []
    for frame in range(12):
        cx = 20 + frame * 12
        events += _step(
            tracker, frame, [_det(cx, 100)], crops=[None]
        )
    assert len(events) == 1
    assert events[0]["evidence_crop"] is None
    assert events[0]["evidence_frame"] is None
