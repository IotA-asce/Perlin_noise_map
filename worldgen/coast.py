from __future__ import annotations

import numpy as np


def box_blur2d(a: np.ndarray, *, radius: int) -> np.ndarray:
    """Fast 2D box blur using summed-area tables."""

    x = np.asarray(a, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("a must be a 2D array")

    r = int(radius)
    if r <= 0:
        return x.copy()

    H, W = x.shape
    p = np.pad(x, ((r, r), (r, r)), mode="edge")

    # Integral image with a leading 0 row/col.
    s = np.pad(p, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    s = np.cumsum(np.cumsum(s, axis=0), axis=1)

    y0 = np.arange(H, dtype=np.int32)
    x0 = np.arange(W, dtype=np.int32)
    y1 = y0 + 2 * r + 1
    x1 = x0 + 2 * r + 1

    # Region sums for each output cell.
    A = s[y1[:, None], x1[None, :]]
    B = s[y0[:, None], x1[None, :]]
    C = s[y1[:, None], x0[None, :]]
    D = s[y0[:, None], x0[None, :]]
    area = float((2 * r + 1) * (2 * r + 1))
    return (A - B - C + D) / area


def coastline_smooth(
    height01: np.ndarray,
    *,
    water_level: float,
    band: float = 0.08,
    radius: int = 2,
    strength: float = 0.5,
) -> np.ndarray:
    """Smooth terrain primarily near the coastline.

    band defines how wide (in height units) the coastline influence region is.
    """

    h = np.asarray(height01, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("height01 must be a 2D array")

    water_level = float(water_level)
    band = float(max(band, 0.0))
    strength = float(np.clip(float(strength), 0.0, 1.0))

    if strength == 0.0 or band == 0.0:
        return h.copy()

    blurred = box_blur2d(h, radius=int(radius))
    # 1 near shoreline, 0 far away.
    d = np.abs(h - water_level)
    w = np.clip(1.0 - (d / max(band, 1e-9)), 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    return np.clip(h * (1.0 - strength * w) + blurred * (strength * w), 0.0, 1.0)


def beach_deposit(
    height01: np.ndarray,
    *,
    water_level: float,
    shore_level: float,
    amount: float = 0.015,
) -> np.ndarray:
    """Deposit sand in the shore band by raising heights slightly."""

    h = np.asarray(height01, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("height01 must be a 2D array")

    water_level = float(water_level)
    shore_level = float(shore_level)
    amount = float(amount)
    if amount <= 0.0:
        return h.copy()

    lo = min(water_level, shore_level)
    hi = max(water_level, shore_level)
    t = np.clip((h - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    # Peak deposit near mid-shore.
    w = 4.0 * t * (1.0 - t)
    out = h + amount * w
    return np.clip(out, 0.0, 1.0)
