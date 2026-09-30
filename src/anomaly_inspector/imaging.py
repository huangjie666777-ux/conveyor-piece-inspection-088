"""Image loading, preprocessing and heatmap rendering.

All detector input uses whole-image resize to 224x224 followed by ImageNet
normalization (no center crop, per the application requirements).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

INPUT_SIZE = 224
SUPPORTED_FORMATS = ("PNG", "JPEG")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PREPROCESS_INFO = {
    "input_size": [INPUT_SIZE, INPUT_SIZE],
    "resize": "bilinear whole-image resize (no center crop)",
    "color_space": "RGB",
    "normalization": "ImageNet per-channel mean/std",
    "mean": list(IMAGENET_MEAN),
    "std": list(IMAGENET_STD),
    "accepted_formats": ["PNG", "JPEG"],
    "max_upload_bytes": MAX_UPLOAD_BYTES,
}

_mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_std = torch.tensor(IMAGENET_STD).view(3, 1, 1)


@dataclass(frozen=True)
class LoadedImage:
    """An image accepted by the API."""

    original: Image.Image
    tensor: torch.Tensor  # float32 (3, 224, 224), ImageNet normalized
    content_sha: str      # sha256 of the normalized upload bytes


def decode_image(data: bytes) -> Image.Image:
    """Decode and validate an uploaded PNG/JPEG; return an RGB PIL image."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"file too large: {len(data)} bytes, limit is {MAX_UPLOAD_BYTES}"
        )
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:  # Pillow raises a variety of error types
        raise ValueError("file is not a valid PNG or JPEG image") from exc
    fmt = (image.format or "").upper()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported format {fmt or 'unknown'}; only PNG/JPEG")
    return image.convert("RGB")


def preprocess(image: Image.Image) -> torch.Tensor:
    """Whole-image bilinear resize to 224x224 then ImageNet normalization."""
    resized = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0  # HWC RGB
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - _mean) / _std


def load_image_bytes(data: bytes, content_sha: str) -> LoadedImage:
    image = decode_image(data)
    return LoadedImage(original=image, tensor=preprocess(image), content_sha=content_sha)


def colorize_distance(
    distance_map: np.ndarray,
    original_size: tuple[int, int],
    threshold: float,
) -> np.ndarray:
    """Upscale the low-resolution distance map and apply a fixed color scale.

    ``original_size`` is (width, height). The map is bilinearly resized to the
    original image resolution. The color upper bound is fixed at
    ``2 * threshold`` so colors stay comparable across images; when the
    threshold is zero the raw maximum is used (and documented upstream).
    Values are never normalized per image.
    """
    src = Image.fromarray(distance_map.astype(np.float32), mode="F")
    upscaled = np.asarray(src.resize(original_size, Image.BILINEAR), dtype=np.float32)

    if threshold and threshold > 0:
        vmax = 2.0 * threshold
    else:
        vmax = float(upscaled.max()) if upscaled.max() > 0 else 1.0
    normalized = np.clip(upscaled / vmax, 0.0, 1.0)

    # Turbo-like JET colormap without matplotlib (RGB uint8), piecewise linear.
    heat = _jet_rgb(normalized)
    return heat


def _jet_rgb(values: np.ndarray) -> np.ndarray:
    """Apply the JET colormap (blue -> red) to an array in [0, 1]."""
    v = np.clip(values * 4.0, 0.0, 4.0)
    red = np.clip(np.minimum(v - 1.5, 4.5 - v), 0.0, 1.0)
    green = np.clip(np.minimum(v - 0.5, 3.5 - v), 0.0, 1.0)
    blue = np.clip(np.minimum(v + 0.5, 2.5 - v), 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    return (rgb * 255.0).round().astype(np.uint8)


def overlay_png(base: Image.Image, heat: np.ndarray, alpha: float) -> bytes:
    """Alpha-blend the RGB heatmap over the original and encode as PNG."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    base_arr = np.asarray(base.convert("RGB"), dtype=np.float32)
    blended = base_arr * (1.0 - alpha) + heat.astype(np.float32) * alpha
    out = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()
