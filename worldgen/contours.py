from __future__ import annotations

import numpy as np


def contour_mask(height01: np.ndarray, *, interval: float = 0.05) -> np.ndarray:
    h = np.asarray(height01, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("height01 must be a 2D array")
    interval = float(interval)
    if interval <= 0.0:
        return np.zeros_like(h, dtype=bool)

    q = np.floor(h / interval).astype(np.int32)
    m = np.zeros_like(q, dtype=bool)
    m[:, 1:] |= q[:, 1:] != q[:, :-1]
    m[1:, :] |= q[1:, :] != q[:-1, :]
    return m


def apply_mask_overlay(
    rgb01: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[float, float, float] = (0.05, 0.05, 0.05),
    alpha: float = 0.45,
) -> np.ndarray:
    rgb = np.asarray(rgb01, dtype=np.float64)
    m = np.asarray(mask).astype(bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb01 must be HxWx3")
    if m.shape != rgb.shape[:2]:
        raise ValueError("mask must be HxW")

    a = float(np.clip(float(alpha), 0.0, 1.0))
    c = np.array(color, dtype=np.float64)
    out = rgb.copy()
    out[m] = out[m] * (1.0 - a) + c * a
    return np.clip(out, 0.0, 1.0)
