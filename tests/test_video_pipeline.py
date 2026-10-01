import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from anomaly_inspector.jobs import JobStore
from anomaly_inspector.segmentation import BackgroundSegmenter, SegmentConfig
from anomaly_inspector.store import AppState
from anomaly_inspector.video_pipeline import (
    CancellationToken,
    VideoParams,
    suggest_params,
    validate_video_file,
)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "resnet18-f37072fd.pth"
pytestmark = pytest.mark.skipif(
    not WEIGHTS.exists(), reason="resnet18 weights unavailable"
)


def _write_video(path: Path, frames, fps=10.0, size=(320, 240)):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def test_segmenter_filters_noise_and_flags_touching_cluster():
    h, w = 120, 160
    background = np.full((h, w, 3), 80, np.uint8)
    config = SegmentConfig(
        roi=(0, 0, w, h), diff_threshold=30, min_area=200,
        max_piece_area=2500, min_side=12, max_side=60,
    )
    segmenter = BackgroundSegmenter(background, config)
    frame = background.copy()
    frame[50:90, 30:70] = 200  # clean 40x40 piece
    frame[5:9, 5:9] = 250      # tiny noise speck, below min_area
    clean, merged = segmenter.detect(frame)
    assert len(clean) == 1
    assert merged == []

    frame2 = background.copy()
    frame2[30:100, 90:150] = 200  # oversized touching cluster
    clean2, merged2 = segmenter.detect(frame2)
    assert clean2 == []
    assert len(merged2) == 1
    assert merged2[0].merged is True


def test_video_validation_dimensions_fps_and_size_mismatch(tmp_path):
    background = np.full((240, 320, 3), 80, np.uint8)
    frames = [background.copy() for _ in range(5)]
    video = tmp_path / "v.mp4"
    _write_video(video, frames, fps=10.0, size=(320, 240))
    bg_path = tmp_path / "bg.png"
    Image.fromarray(background).save(bg_path)
    info = validate_video_file(str(video), str(bg_path))
    assert info["frames"] == 5
    assert info["width"] == 320

    wrong = tmp_path / "wrong.png"
    Image.fromarray(np.full((240, 300, 3), 80, np.uint8)).save(wrong)
    with pytest.raises(ValueError, match="不一致"):
        validate_video_file(str(video), str(wrong))


def test_suggest_params_place_line_inside_roi():
    s = suggest_params(320, 240)
    x1, _, x2, _ = s["roi"]
    assert x1 + s["max_side"] <= s["line_x"] <= x2 - s["max_side"]
    assert s["direction"] == "ltr"


def test_job_cancel_does_not_report_completed(tmp_path):
    background = np.full((240, 320, 3), 80, np.uint8)
    frames = [background.copy() for _ in range(60)]
    video = tmp_path / "v.mp4"
    _write_video(video, frames, fps=10.0, size=(320, 240))
    bg_path = tmp_path / "bg.png"
    Image.fromarray(background).save(bg_path)

    state = AppState(tmp_path / "data", WEIGHTS)
    _seed_bank(state)
    store = JobStore(tmp_path / "data", state)
    token = CancellationToken()
    token.cancel()
    params = VideoParams(
        roi=(0, 0, 320, 240), line_x=160, direction="ltr",
        max_piece_area=50000, max_side=80,
    )
    from anomaly_inspector.video_pipeline import process_video, Progress

    with pytest.raises(Exception):
        process_video(
            str(video), background, params, state.extractor,
            state.snapshot(), Progress(), token,
        )


def _png(seed, size=224):
    rng = np.random.default_rng(seed)
    arr = np.clip(rng.normal(150 + seed % 9, 20, (size, size, 3)), 0, 255)
    import io
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _seed_bank(state):
    for seed in (1, 2, 3, 4):
        state.add_sample("reference", f"r{seed}.png", _png(seed))
    for seed in (11, 12, 13, 14):
        state.add_sample("calibration", f"c{seed}.png", _png(seed))
    state.rebuild()


def test_end_to_end_synthetic_video_counts_and_evidence(tmp_path):
    demo = ROOT / "examples" / "video"
    if not (demo / "conveyor_demo.mp4").exists():
        pytest.skip("synthetic conveyor demo not generated")
    state = AppState(tmp_path / "data", WEIGHTS)
    bank = demo / "bank"
    for p in sorted(bank.glob("reference_*.png")):
        state.add_sample("reference", p.name, p.read_bytes())
    for p in sorted(bank.glob("calibration_*.png")):
        state.add_sample("calibration", p.name, p.read_bytes())
    state.rebuild()

    store = JobStore(tmp_path / "data", state)
    hint = json.loads((demo / "params_hint.json").read_text())
    video_bytes = (demo / "conveyor_demo.mp4").read_bytes()
    bg_bytes = (demo / "conveyor_background.png").read_bytes()
    params = VideoParams(
        roi=tuple(hint["roi"]),
        line_x=hint["line_x"],
        direction=hint["direction"],
        diff_threshold=hint["diff_threshold"],
        min_area=hint["min_area"],
        max_piece_area=hint["max_piece_area"],
        min_side=hint["min_side"],
        max_side=hint["max_side"],
    )
    meta = store.create(
        "jobdemo", "conveyor_demo.mp4", video_bytes, bg_bytes, params
    )
    store._thread["jobdemo"].join(timeout=120)
    job = store.get("jobdemo")
    assert job["status"] == "completed"
    verdicts = {r["track_id"]: r["verdict"] for r in job["results"]}
    assert len(job["results"]) == 3  # touching pair never merges/counts
    assert "anomaly" in verdicts.values()
    assert "ok" in verdicts.values()
    assert job["merged_frames"]  # touching frames were recorded
    anomaly_id = next(
        r["track_id"] for r in job["results"] if r["verdict"] == "anomaly"
    )
    crop = store.evidence_png("jobdemo", anomaly_id, "crop")
    heat = store.evidence_png("jobdemo", anomaly_id, "heat")
    assert crop[:8] == b"\x89PNG\r\n\x1a\n"
    assert heat[:8] == b"\x89PNG\r\n\x1a\n"


def test_interrupted_job_marked_on_restart(tmp_path):
    state = AppState(tmp_path / "data", WEIGHTS)
    store = JobStore(tmp_path / "data", state)
    jobs_dir = tmp_path / "data" / "jobs" / "stale"
    (jobs_dir / "evidence").mkdir(parents=True)
    (jobs_dir / "job.json").write_text(json.dumps({
        "job_id": "stale", "status": "running", "created_at": "t",
        "results": [], "frames_processed": 3, "total_frames": 10,
    }))
    store.recover_stale()
    job = store.get("stale")
    assert job["status"] == "interrupted"
    assert "中断" in job["error"]
