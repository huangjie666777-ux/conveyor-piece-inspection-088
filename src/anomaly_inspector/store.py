"""Persistent application state: uploaded samples, memory bank and threshold.

Rebuild writes new state to temporary files first; a failed rebuild keeps the
previous bank usable. Detection snapshots the active bank under a lock, so a
rebuild can never make a single detection mix two bank versions.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import imaging
from .features import LAYER2_GRID, FeatureExtractor
from .memory import MAX_MEMORY_ITEMS, greedy_farthest_points, image_score
from .stats import linear_quantile

CALIBRATION_QUANTILE = 0.95


@dataclass(frozen=True)
class Sample:
    sample_id: str
    filename: str
    content_id: str  # hash of the decoded/resized pixel content
    width: int
    height: int
    group: str  # "reference" | "calibration"


class AppState:
    def __init__(self, data_dir: str | Path, weights_path: str | Path):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.state_path = self.data_dir / "state.json"
        self.bank_path = self.data_dir / "memory_bank.pt"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.extractor = FeatureExtractor(weights_path)
        self.samples: dict[str, Sample] = {}
        self.memory_bank: torch.Tensor | None = None
        self.threshold: float | None = None
        self.calibration_scores: list[float] = []
        self.built_at: str | None = None
        self.reference_count: int = 0
        self.calibration_count: int = 0
        self._load()

    # ------------------------------------------------------------------ upload

    @staticmethod
    def content_hash(image: Image.Image) -> str:
        resized = image.resize(
            (imaging.INPUT_SIZE, imaging.INPUT_SIZE), Image.BILINEAR
        )
        digest = hashlib.sha256(
            np.asarray(resized, dtype=np.uint8).tobytes()
        ).hexdigest()
        return f"resized224-{digest[:32]}"

    def add_sample(self, group: str, filename: str, data: bytes) -> Sample:
        if group not in ("reference", "calibration"):
            raise ValueError("group must be reference or calibration")
        raw_sha = hashlib.sha256(data).hexdigest()[:16]
        image = imaging.decode_image(data)
        content_id = self.content_hash(image)
        with self._lock:
            for sample in self.samples.values():
                if sample.content_id == content_id:
                    raise ValueError(
                        "identical image content already exists in the "
                        f"{sample.group} set; the two sets must not overlap "
                        "and duplicates are not allowed"
                    )
            sample_id = f"{group[0]}-{raw_sha}"
            if sample_id in self.samples:
                raise ValueError("this file has already been uploaded")
            sample = Sample(
                sample_id=sample_id,
                filename=filename,
                content_id=content_id,
                width=image.width,
                height=image.height,
                group=group,
            )
            self.images_dir.joinpath(sample_id + ".png").write_bytes(
                _png_bytes(image)
            )
            self.samples[sample_id] = sample
            self._persist_metadata()
        return sample

    def delete_sample(self, sample_id: str) -> None:
        with self._lock:
            sample = self.samples.pop(sample_id)
            path = self.images_dir / (sample_id + ".png")
            if path.exists():
                path.unlink()
            self._persist_metadata()

    def list_samples(self, group: str) -> list[Sample]:
        with self._lock:
            return [s for s in self.samples.values() if s.group == group]

    def sample_png(self, sample_id: str) -> bytes:
        with self._lock:
            if sample_id not in self.samples:
                raise KeyError(sample_id)
            return (self.images_dir / (sample_id + ".png")).read_bytes()

    # ------------------------------------------------------------------ build

    def rebuild(self) -> dict:
        """Rebuild the memory bank and threshold; old state survives failure."""
        import datetime

        with self._lock:
            refs = [s for s in self.samples.values() if s.group == "reference"]
            cals = [s for s in self.samples.values() if s.group == "calibration"]
            if not refs:
                raise ValueError("upload at least one reference normal image")
            if not cals:
                raise ValueError(
                    "upload at least one independent calibration normal image"
                )

            ref_desc = torch.cat(
                [self._descriptors(s) for s in refs], dim=0
            )
            bank = greedy_farthest_points(ref_desc, MAX_MEMORY_ITEMS)

            cal_scores = []
            for sample in cals:
                desc = self._descriptors(sample)
                score, _ = image_score(desc, bank)
                cal_scores.append(score)
            threshold = linear_quantile(cal_scores, CALIBRATION_QUANTILE)

            tmp_bank = self.bank_path.with_suffix(".pt.tmp")
            torch.save(bank, tmp_bank)
            built_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

            new_meta = self._metadata_dict()
            new_meta.update(
                {
                    "threshold": threshold,
                    "calibration_scores": cal_scores,
                    "built_at": built_at,
                    "memory_bank_items": int(bank.shape[0]),
                }
            )
            tmp_meta = self.state_path.with_suffix(".json.tmp")
            tmp_meta.write_text(json.dumps(new_meta, indent=2))

            # Commit phase: replace files, then swap the in-memory version.
            os.replace(tmp_bank, self.bank_path)
            os.replace(tmp_meta, self.state_path)
            self.memory_bank = bank
            self.threshold = threshold
            self.calibration_scores = cal_scores
            self.built_at = built_at
            return self.status()

    # ---------------------------------------------------------------- detect

    def inspect_bytes(self, data: bytes) -> dict:
        loaded = imaging.load_image_bytes(data, hashlib.sha256(data).hexdigest())
        with self._lock:
            if self.memory_bank is None or self.threshold is None:
                raise ValueError(
                    "memory bank not built yet; add samples and build first"
                )
            # Snapshot under the lock; inference uses a fixed bank version.
            bank = self.memory_bank
            threshold = self.threshold

        descriptors = self.extractor.extract_image(loaded.tensor)
        score, distances = image_score(descriptors, bank)
        grid = distances.reshape(LAYER2_GRID, LAYER2_GRID).numpy()

        heat = imaging.colorize_distance(
            grid, loaded.original.size, threshold
        )
        threshold_zero = threshold == 0.0
        return {
            "score": score,
            "threshold": threshold,
            "is_anomaly": bool(score > threshold),
            "threshold_is_zero": threshold_zero,
            "color_scale_vmax": (2.0 * threshold) if not threshold_zero else None,
            "distance_grid": grid.tolist(),
            "original_png": _png_bytes(loaded.original),
            "heat_png": _png_bytes(Image.fromarray(heat)),
            "width": loaded.original.width,
            "height": loaded.original.height,
        }

    # ---------------------------------------------------------------- status

    def status(self) -> dict:
        with self._lock:
            return {
                "reference_count": sum(
                    1 for s in self.samples.values() if s.group == "reference"
                ),
                "calibration_count": sum(
                    1 for s in self.samples.values() if s.group == "calibration"
                ),
                "memory_bank_items": (
                    int(self.memory_bank.shape[0])
                    if self.memory_bank is not None
                    else 0
                ),
                "threshold": self.threshold,
                "threshold_is_zero": self.threshold == 0.0,
                "calibration_scores": list(self.calibration_scores),
                "built_at": self.built_at,
                "preprocess": imaging.PREPROCESS_INFO,
            }

    # ------------------------------------------------------------- persistence

    def _descriptors(self, sample: Sample) -> torch.Tensor:
        path = self.images_dir / (sample.sample_id + ".png")
        image = Image.open(path).convert("RGB")
        return self.extractor.extract_image(imaging.preprocess(image))

    def _metadata_dict(self) -> dict:
        return {
            "samples": [vars(s) for s in self.samples.values()],
            "threshold": self.threshold,
            "calibration_scores": self.calibration_scores,
            "built_at": self.built_at,
            "preprocess": imaging.PREPROCESS_INFO,
        }

    def _persist_metadata(self) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._metadata_dict(), indent=2))
        os.replace(tmp, self.state_path)

    def _load(self) -> None:
        if self.state_path.exists():
            meta = json.loads(self.state_path.read_text())
            for raw in meta.get("samples", []):
                sample = Sample(**raw)
                if (self.images_dir / (sample.sample_id + ".png")).exists():
                    self.samples[sample.sample_id] = sample
            self.threshold = meta.get("threshold")
            self.calibration_scores = meta.get("calibration_scores", [])
            self.built_at = meta.get("built_at")
        if self.bank_path.exists():
            try:
                bank = torch.load(self.bank_path, map_location="cpu")
                if isinstance(bank, torch.Tensor) and bank.ndim == 2:
                    self.memory_bank = bank
            except Exception:
                # Corrupt bank: keep metadata but report an unbuilt detector
                # rather than crashing startup.
                self.memory_bank = None


def _png_bytes(image: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
