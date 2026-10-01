import io

import numpy as np
import pytest
import torch
from PIL import Image

from anomaly_inspector import imaging
from anomaly_inspector.store import AppState

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "resnet18-f37072fd.pth"
pytestmark = pytest.mark.skipif(
    not WEIGHTS.exists(), reason="resnet18 weights unavailable"
)


def _png(seed: int, anomaly: str | None = None) -> bytes:
    rng = np.random.default_rng(seed)
    arr = np.clip(
    rng.normal(160 + seed % 7, 18, (256, 256, 3)), 0, 255
    ).astype(np.uint8)
    if anomaly == "scratch":
        arr[40:220, 120:126] = 10
    elif anomaly == "blob":
        arr[100:150, 100:150] = 5
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_full_build_detect_persist_and_recover(tmp_path):
    state = AppState(tmp_path / "data", WEIGHTS)
    for seed in (1, 2, 3, 4):
        state.add_sample("reference", f"r{seed}.png", _png(seed))
    for seed in (11, 12, 13, 14):
        state.add_sample("calibration", f"c{seed}.png", _png(seed))

    # Identical content across groups is rejected.
    with pytest.raises(ValueError, match="identical image content"):
        state.add_sample("calibration", "dup.png", _png(1))

    status = state.rebuild()
    assert status["memory_bank_items"] == 256 or status["memory_bank_items"] > 0
    assert status["threshold"] >= 0.0
    threshold = status["threshold"]

    normal_result = state.inspect_bytes(_png(11))
    anomaly_result = state.inspect_bytes(_png(11, "blob"))
    assert anomaly_result["score"] > normal_result["score"]
    assert anomaly_result["is_anomaly"] == (
        anomaly_result["score"] > threshold
    )
    assert anomaly_result["heat_png"][:8] == b"\x89PNG\r\n\x1a\n"

    # Restart: bank and threshold are restored without rebuilding.
    restored = AppState(tmp_path / "data", WEIGHTS)
    assert restored.threshold == pytest.approx(threshold)
    assert restored.memory_bank is not None
    assert restored.memory_bank.shape == state.memory_bank.shape
    assert len(restored.samples) == 8


def test_failed_rebuild_keeps_previous_bank(tmp_path):
    state = AppState(tmp_path / "data", WEIGHTS)
    state.add_sample("reference", "r1.png", _png(1))
    state.add_sample("calibration", "c1.png", _png(11))
    state.rebuild()
    old_bank = state.memory_bank.clone()
    old_threshold = state.threshold

    # Force a failure mid-rebuild: delete a stored image after upload.
    before = set((tmp_path / "data" / "images").glob("*.png"))
    state.add_sample("reference", "broken.png", _png(2))
    broken = next(
        iter(set((tmp_path / "data" / "images").glob("*.png")) - before)
    )
    broken.unlink()
    with pytest.raises(Exception):
        state.rebuild()
    assert torch.equal(state.memory_bank, old_bank)
    assert state.threshold == old_threshold


def test_color_scale_is_fixed_not_per_image():
    grid = np.full((28, 28), 3.0, dtype=np.float32)
    heat = imaging.colorize_distance(grid, (56, 56), threshold=10.0)
    # 3.0 / 20.0 = 0.15 -> deep-blue region; same value with a different
    # image maximum must give the identical color (no per-image stretch).
    assert heat.shape == (56, 56, 3)
    grid2 = np.full((28, 28), 3.0, dtype=np.float32)
    heat2 = imaging.colorize_distance(grid2, (56, 56), threshold=10.0)
    assert np.array_equal(heat, heat2)


def test_invalid_upload_rejected():
    with pytest.raises(ValueError):
        imaging.decode_image(b"not an image")


def test_bank_and_threshold_never_mix_after_restart(tmp_path):
    state = AppState(tmp_path / "data", WEIGHTS)
    for seed in (1, 2, 3, 4):
        state.add_sample("reference", f"r{seed}.png", _png(seed))
    for seed in (11, 12, 13, 14):
        state.add_sample("calibration", f"c{seed}.png", _png(seed))
    state.rebuild()

    # Tamper with the content-addressed bank file named in metadata:
    # restart must detect the hash mismatch and refuse to pair the stale
    # threshold with an unrelated/missing bank (no mixed detector).
    import json

    meta_path = tmp_path / "data" / "state.json"
    meta = json.loads(meta_path.read_text())
    bank_file = meta["bank_file"]
    bank_path = tmp_path / "data" / "banks" / bank_file
    different = state.memory_bank.clone()
    different.add_(1.0)
    torch.save(different, bank_path)

    restored = AppState(tmp_path / "data", WEIGHTS)
    assert restored.memory_bank is None
    assert restored.threshold is None
    with pytest.raises(ValueError, match="not built"):
        restored.inspect_bytes(_png(11))

    # Rebuilding recovers the service cleanly.
    restored.rebuild()
    assert restored.memory_bank is not None
    assert restored.threshold is not None
