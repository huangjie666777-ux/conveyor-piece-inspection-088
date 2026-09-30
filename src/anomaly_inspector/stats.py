"""Threshold statistics from independent calibration images."""

from __future__ import annotations


def linear_quantile(values: list[float], q: float) -> float:
    """Linear-interpolated sample quantile (NumPy "linear" / R type 7).

    ``q`` is in [0, 1]. Sorted value at virtual index ``(n - 1) * q``.
    """
    if not values:
        raise ValueError("cannot compute a quantile without calibration scores")
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
