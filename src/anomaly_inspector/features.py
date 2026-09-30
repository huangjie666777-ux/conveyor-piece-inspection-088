"""Frozen ResNet18 local feature extraction (layer2 + layer3 descriptors)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet18

from .imaging import INPUT_SIZE

# layer2 outputs 28x28 (stride 8), layer3 outputs 14x14 (stride 16).
LAYER2_GRID = INPUT_SIZE // 8
FEATURE_DIM = 128 + 256  # concatenated descriptor dimensionality


class FeatureExtractor:
    """ResNet18 up to layer3, frozen, running in eval mode on CPU."""

    def __init__(self, weights_path: str | Path):
        weights_path = Path(weights_path)
        model = resnet18(weights=None)
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model

    @torch.inference_mode()
    def extract(self, batch: torch.Tensor) -> torch.Tensor:
        """Return local descriptors as (B, 28, 28, 384)."""
        x = self.model.conv1(batch)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x2 = self.model.layer2(x)  # (B, 128, 28, 28)
        x3 = self.model.layer3(x2)  # (B, 256, 14, 14)

        # Align spatial sizes: upsample layer3 back to the layer2 grid.
        x3_up = F.interpolate(
            x3,
            size=(LAYER2_GRID, LAYER2_GRID),
            mode="bilinear",
            align_corners=False,
        )
        concat = torch.cat([x2, x3_up], dim=1)  # (B, 384, 28, 28)
        return concat.permute(0, 2, 3, 1).contiguous()

    @torch.inference_mode()
    def extract_image(self, normalized_image: torch.Tensor) -> torch.Tensor:
        """Extract descriptors for a single image; returns (784, 384)."""
        batch = normalized_image.unsqueeze(0)
        return self.extract(batch).reshape(-1, FEATURE_DIM)
