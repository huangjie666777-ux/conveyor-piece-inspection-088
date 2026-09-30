"""Deterministic greedy farthest-point selection and nearest-neighbour scoring."""

from __future__ import annotations

import torch

MAX_MEMORY_ITEMS = 256


def greedy_farthest_points(
    descriptors: torch.Tensor, max_items: int = MAX_MEMORY_ITEMS
) -> torch.Tensor:
    """Build a fixed-size memory bank with deterministic greedy FPS.

    Only distances to the currently selected set are tracked, so no full
    sample x sample distance matrix is ever constructed. The first selected
    point is always index 0; ties are broken by smallest index, giving the
    same bank for identical input regardless of platform.
    """
    n = descriptors.shape[0]
    k = min(max_items, n)
    selected = torch.empty(k, dtype=torch.long)
    selected[0] = 0
    min_dist = torch.full((n,), float("inf"), dtype=torch.float32)
    used = torch.zeros(n, dtype=torch.bool)
    used[0] = True

    for step in range(k):
        anchor = descriptors[selected[step]].unsqueeze(0)
        dist = torch.cdist(descriptors, anchor).squeeze(1)
        min_dist = torch.minimum(min_dist, dist)
        if step + 1 < k:
            masked = min_dist.masked_fill(used, -float("inf"))
            nxt = int(torch.argmax(masked).item())
            selected[step + 1] = nxt
            used[nxt] = True
    return descriptors[selected]


def local_distances(
    descriptors: torch.Tensor, memory_bank: torch.Tensor
) -> torch.Tensor:
    """Nearest-memory Euclidean distance for each local descriptor.

    Returns a 1-D tensor aligned row-wise with ``descriptors``.
    """
    # Chunk the query side to bound peak memory on CPU.
    chunks = []
    for start in range(0, descriptors.shape[0], 1024):
        block = descriptors[start : start + 1024]
        dist = torch.cdist(block, memory_bank)
        chunks.append(dist.min(dim=1).values)
    return torch.cat(chunks)


def image_score(
    descriptors: torch.Tensor, memory_bank: torch.Tensor
) -> tuple[float, torch.Tensor]:
    """Score an image as the maximum local nearest-neighbour distance."""
    distances = local_distances(descriptors, memory_bank)
    return float(distances.max().item()), distances
