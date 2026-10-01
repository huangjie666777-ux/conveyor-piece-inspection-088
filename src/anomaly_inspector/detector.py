"""Immutable detector snapshot used by both single-image and video jobs.

A video job fixes one calibrated detector version at start time. A later
rebuild of the memory bank never changes the detector used by that job, even
if the job is inspected again after a server restart: the job stores the
bank content hash and refuses a mismatched bank.
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass

import torch
from PIL import Image

from . import imaging
from .features import LAYER2_GRID, FeatureExtractor
from .memory import image_score


@dataclass(frozen=True)
class DetectorSnapshot:
    bank: torch.Tensor
    threshold: float
    built_at: str | None
    bank_sha256: str

    @staticmethod
    def bank_hash(bank: torch.Tensor) -> str:
        tensor = bank.detach().cpu().contiguous()
        digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        return f"sha256-{digest[:32]}"

    def inspect_pil(self, extractor: FeatureExtractor, image: Image.Image) -> dict:
        tensor = imaging.preprocess(image)
        descriptors = extractor.extract_image(tensor)
        score, distances = image_score(descriptors, self.bank)
        grid = distances.reshape(LAYER2_GRID, LAYER2_GRID).numpy()
        heat = imaging.colorize_distance(grid, image.size, self.threshold)
        threshold_zero = self.threshold == 0.0
        return {
            "score": score,
            "threshold": self.threshold,
            "is_anomaly": bool(score > self.threshold),
            "threshold_is_zero": threshold_zero,
            "color_scale_vmax": (
                2.0 * self.threshold if not threshold_zero else None
            ),
            "distance_grid": grid,
            "heat": heat,
        }
