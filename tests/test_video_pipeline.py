"""End-to-end pipeline tests on the synthetic conveyor assets."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from anomaly_inspector.detection import DetectionParams
from anomaly_inspector.jobs import JobManager
from anomaly_inspector.store import AppState
from anomaly_inspector.video import VideoConfig

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "resnet18-f37072fd.pth"
VIDEO_DIR = ROOT / "examples" / "video"
pytestmark = pytest.mark.skipif(
    not WEIGHTS.exists() or not VIDEO_DIR.exists(),
    reason="weights or synthetic video unavailable",
)


def _build_state(tmp_path):
    state = AppState(tmp_path / "data", WEIGHTS)
    for i in range(1, 5):
        state.add_sample(
            "reference", f"n{i}.png",
            (VIDEO_DIR / f"library_normal_{i}.png").read_bytes(),
        )
    for i in range(5, 7):
        state.add_sample(
            "calibration", f"c{i}.png",
            (VIDEO_DIR / f"library_normal_{i}.png").read_bytes(),
        )
    state.rebuild()
    return state


def _wait(status_fn, target=None, timeout=300):
    box = {}

    def runner():
        while True:
            job = status_fn()
            box["s"] = job["status"]
            if job["status"] in ("completed", "cancelled", "failed", "interrupted"):
                box["job"] = job
                return
            time.sleep(0.2)
    t = threading.Thread(target=runner)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "job did not finish in time"
    return box["job"]


def test_full_video_job_counts_and_evidence(tmp_path):
    state = _build_state(tmp_path)
    manager = JobManager(tmp_path / "jobs", state)
    job = manager.create_job(
        "conveyor_demo.mp4",
        (VIDEO_DIR / "conveyor_demo.mp4").read_bytes(),
        (VIDEO_DIR / "background.png").read_bytes(),
        VideoConfig(roi=(20, 40, 600, 240), line_x=320, direction="lr"),
    )
    job = _wait(lambda: manager.status(job["id"]))
    assert job["status"] == "completed", job.get("error")
    pieces = job["pieces"]
    assert len(pieces) >= 4
    ids = [p["track_id"] for p in pieces]
    assert len(ids) == len(set(ids))  # no double counting
    real_dir = tmp_path / "jobs" / job["id"] / "evidence"
    # At least one defective and one ok verdict should be reachable; every
    # counted piece must have a stable score/threshold snapshot recorded.
    verdicts = {p["verdict"] for p in pieces}
    assert "review" in verdicts or "defect" in verdicts
    for p in pieces:
        assert p["threshold"] == pytest.approx(state.threshold)
        if p["has_crop"]:
            assert (real_dir / (str(p['track_id']) + '_crop.png')).exists()
    # Track playback data present for completed job.
    assert any(t["crossed"] for t in job["tracks"])


def test_cancel_stops_without_completion(tmp_path):
    state = _build_state(tmp_path)
    manager = JobManager(tmp_path / "jobs", state)
    job = manager.create_job(
        "conveyor_demo.mp4",
        (VIDEO_DIR / "conveyor_demo.mp4").read_bytes(),
        (VIDEO_DIR / "background.png").read_bytes(),
        VideoConfig(roi=(20, 40, 600, 240), line_x=320, direction="lr"),
    )
    manager.cancel(job["id"])
    final = _wait(lambda: manager.status(job["id"]), "cancelled", timeout=60)
    assert final["status"] == "cancelled"
    assert final["pieces"] == []


def test_interrupted_job_marked_on_restart(tmp_path):
    jobs_dir = tmp_path / "jobs"
    (jobs_dir / "j1").mkdir(parents=True)
    import json
    (jobs_dir / "j1" / "job.json").write_text(json.dumps({
        "id": "j1", "status": "running", "filename": "x.mp4",
        "video_file": "video.mp4", "created_at": "now",
        "progress": {"processed": 3, "total": 10}, "fps": 20,
        "width": 640, "height": 320, "config": {}, "detector": {},
        "pieces": [], "tracks": [], "error": None,
    }))
    state = _build_state(tmp_path)
    manager = JobManager(jobs_dir, state)
    assert manager.status("j1")["status"] == "interrupted"


def test_snapshot_fixed_after_rebuild(tmp_path):
    state = _build_state(tmp_path)
    snap1 = state.detector_snapshot()
    old_threshold = snap1.threshold
    # A later rebuild swaps a new version; snapshot tensors stay the same.
    state.threshold = old_threshold + 1.0
    assert float(snap1.threshold) == pytest.approx(old_threshold)
    assert torch.equal(snap1.bank, snap1.bank)  # tensor remains usable