from __future__ import annotations

import numpy as np


def jittered_points(
    *,
    seed: int,
    height: int,
    width: int,
    cell: int,
    probability: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate deterministic jittered grid points.

    Returns (xs, ys) in pixel coordinates.
    """

    H = int(height)
    W = int(width)
    c = max(int(cell), 1)
    p = float(np.clip(float(probability), 0.0, 1.0))

    ys = np.arange(0, H, c, dtype=np.int32)
    xs = np.arange(0, W, c, dtype=np.int32)
    if ys.size == 0 or xs.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    rng = np.random.default_rng(int(seed))
    keep = rng.random((ys.size, xs.size)) < p
    jx = rng.random((ys.size, xs.size))
    jy = rng.random((ys.size, xs.size))

    gx, gy = np.meshgrid(xs, ys)
    px = gx.astype(np.float64) + jx * float(c)
    py = gy.astype(np.float64) + jy * float(c)

    sel = keep.reshape(-1)
    px = px.reshape(-1)[sel]
    py = py.reshape(-1)[sel]

    px = np.clip(px, 0.0, max(float(W - 1), 0.0))
    py = np.clip(py, 0.0, max(float(H - 1), 0.0))
    return px, py


def filter_points_by_mask(
    xs: np.ndarray,
    ys: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mask).astype(bool)
    if m.ndim != 2:
        raise ValueError("mask must be 2D")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("xs and ys must have the same shape")

    xi = np.clip(np.round(x).astype(np.int32), 0, m.shape[1] - 1)
    yi = np.clip(np.round(y).astype(np.int32), 0, m.shape[0] - 1)
    keep = m[yi, xi]
    return x[keep], y[keep]
