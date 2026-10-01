"""Persistent application state: uploaded samples, memory bank and threshold.

Rebuild commits through a single atomic metadata swap: bank tensors are
stored in content-addressed files and the metadata names the exact bank file
and its hash. A crash at any point (or a failed metadata write) therefore
leaves either the old (bank, threshold) pair or the new pair, never a new
bank paired with an old threshold after restart. Detection and video jobs
snapshot the active pair under a lock, so a rebuild can never make a single
task mix two detector versions.
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
from .detector import DetectorSnapshot
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
        self.banks_dir = self.data_dir / "banks"
        self.state_path = self.data_dir / "state.json"
        self.legacy_bank_path = self.data_dir / "memory_bank.pt"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.banks_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.extractor = FeatureExtractor(weights_path)
        self.samples: dict[str, Sample] = {}
        self.memory_bank: torch.Tensor | None = None
        self.threshold: float | None = None
        self.calibration_scores: list[float] = []
        self.built_at: str | None = None
        self.bank_file: str | None = None
        self.bank_sha256: str | None = None
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

            built_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
            bank_sha = DetectorSnapshot.bank_hash(bank)
            bank_file = f"{bank_sha}.pt"
            # Write the content-addressed bank durably first. A brand-new
            # file name can never be half-read as the previous bank.
            tmp_bank = self.banks_dir / f".{bank_file}.tmp"
            torch.save(bank, tmp_bank)
            os.replace(tmp_bank, self.banks_dir / bank_file)

            new_meta = self._metadata_dict()
            new_meta.update(
                {
                    "threshold": threshold,
                    "calibration_scores": cal_scores,
                    "built_at": built_at,
                    "memory_bank_items": int(bank.shape[0]),
                    "bank_file": bank_file,
                    "bank_sha256": bank_sha,
                }
            )
            tmp_meta = self.state_path.with_suffix(".json.tmp")
            # Single commit point: one atomic rename publishes the
            # threshold and the bank file name/hash together.
            with open(tmp_meta, "w") as handle:
                handle.write(json.dumps(new_meta, indent=2))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_meta, self.state_path)
            self.memory_bank = bank
            self.threshold = threshold
            self.calibration_scores = cal_scores
            self.built_at = built_at
            self.bank_file = bank_file
            self.bank_sha256 = bank_sha
            return self.status()

    # ---------------------------------------------------------------- detect

    def inspect_bytes(self, data: bytes) -> dict:
        loaded = imaging.load_image_bytes(data, hashlib.sha256(data).hexdigest())
        snapshot = self.snapshot()
        outcome = snapshot.inspect_pil(self.extractor, loaded.original)
        return {
            "score": outcome["score"],
            "threshold": outcome["threshold"],
            "is_anomaly": outcome["is_anomaly"],
            "threshold_is_zero": outcome["threshold_is_zero"],
            "color_scale_vmax": outcome["color_scale_vmax"],
            "original_png": _png_bytes(loaded.original),
            "heat_png": _png_bytes(Image.fromarray(outcome["heat"])),
            "width": loaded.original.width,
            "height": loaded.original.height,
        }

    def snapshot(self) -> DetectorSnapshot:
        """Fix one calibrated detector version under the lock.

        Video jobs call this exactly once at start; later rebuilds cannot
        change the detector version used by that job.
        """
        with self._lock:
            if self.memory_bank is None or self.threshold is None:
                raise ValueError(
                    "memory bank not built yet; add samples and build first"
                )
            return DetectorSnapshot(
                bank=self.memory_bank,
                threshold=float(self.threshold),
                built_at=self.built_at,
                bank_sha256=self.bank_sha256 or "",
            )

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
            "bank_file": self.bank_file,
            "bank_sha256": self.bank_sha256,
            "preprocess": imaging.PREPROCESS_INFO,
        }

    def _persist_metadata(self) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._metadata_dict(), indent=2))
        os.replace(tmp, self.state_path)

    def _load(self) -> None:
        meta: dict = {}
        if self.state_path.exists():
            meta = json.loads(self.state_path.read_text())
            for raw in meta.get("samples", []):
                sample = Sample(**raw)
                if (self.images_dir / (sample.sample_id + ".png")).exists():
                    self.samples[sample.sample_id] = sample
        bank_file = meta.get("bank_file")
        bank = self._load_bank(bank_file) if bank_file else None
        if bank is None and bank_file is None:
            # Migration from the pre-content-addressed single-file layout.
            bank = self._load_bank(None, legacy=self.legacy_bank_path)
        if bank is not None:
            actual_sha = DetectorSnapshot.bank_hash(bank)
            expected_sha = meta.get("bank_sha256")
            # A stored threshold is only valid together with its exact bank.
            if expected_sha and actual_sha != expected_sha:
                bank = None
        if bank is not None:
            self.memory_bank = bank
            self.threshold = meta.get("threshold")
            self.calibration_scores = meta.get("calibration_scores", [])
            self.built_at = meta.get("built_at")
            self.bank_file = bank_file or f"{DetectorSnapshot.bank_hash(bank)}.pt"
            self.bank_sha256 = DetectorSnapshot.bank_hash(bank)
            if not (self.banks_dir / self.bank_file).exists():
                torch.save(bank, self.banks_dir / self.bank_file)
        else:
            # Mismatch/missing/corrupt bank: never pair a stale threshold
            # with an unrelated bank after restart.
            self.memory_bank = None
            self.threshold = None
            self.calibration_scores = []
            self.built_at = None
        self._prune_bank_files()

    def _load_bank(
        self,
        bank_file: str | None,
        legacy: Path | None = None,
    ) -> torch.Tensor | None:
        path = (
            (self.banks_dir / bank_file)
            if bank_file is not None
            else legacy
        )
        if path is None or not path.exists():
            return None
        try:
            bank = torch.load(path, map_location="cpu")
        except Exception:
            return None
        if isinstance(bank, torch.Tensor) and bank.ndim == 2:
            return bank
        return None

    def _prune_bank_files(self) -> None:
        keep = {self.bank_file} if self.bank_file else set()
        for path in self.banks_dir.glob("*.pt"):
            if path.name not in keep:
                try:
                    path.unlink()
                except OSError:
                    pass


def _png_bytes(image: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
