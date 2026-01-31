from __future__ import annotations

import numpy as np


def flow_direction_d8(height: np.ndarray) -> np.ndarray:
    """Return downstream flat indices per cell using a strict D8 rule.

    Each cell chooses the lowest *strictly lower* neighbor among 8 neighbors.
    Cells without a lower neighbor become sinks (downstream = -1).
    """

    h = np.asarray(height, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("height must be a 2D array")

    H, W = h.shape
    p = np.pad(h, 1, mode="edge")
    c = p[1:-1, 1:-1]

    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    best = np.full((H, W), np.inf, dtype=np.float64)
    best_dy = np.zeros((H, W), dtype=np.int32)
    best_dx = np.zeros((H, W), dtype=np.int32)

    for dy, dx in offsets:
        neigh = p[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
        lower = neigh < c
        take = lower & (neigh < best)
        best[take] = neigh[take]
        best_dy[take] = int(dy)
        best_dx[take] = int(dx)

    has = best < np.inf
    ys = np.arange(H, dtype=np.int32)[:, None]
    xs = np.arange(W, dtype=np.int32)[None, :]

    to_y = ys + best_dy
    to_x = xs + best_dx

    valid = has & (to_y >= 0) & (to_y < H) & (to_x >= 0) & (to_x < W)
    downstream = np.full((H, W), -1, dtype=np.int32)
    downstream[valid] = (to_y[valid] * W + to_x[valid]).astype(np.int32)
    return downstream


def flow_accumulation_d8(height: np.ndarray, downstream: np.ndarray) -> np.ndarray:
    """Compute flow accumulation (cells) given D8 downstream indices."""

    h = np.asarray(height, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("height must be a 2D array")

    ds = np.asarray(downstream, dtype=np.int32)
    if ds.shape != h.shape:
        raise ValueError("downstream must have the same shape as height")

    H, W = h.shape
    n = int(H * W)
    flat_h = h.reshape(-1)
    flat_ds = ds.reshape(-1)

    order = np.argsort(flat_h, kind="mergesort")[::-1]
    acc = np.ones(n, dtype=np.float64)

    for i in order:
        j = int(flat_ds[int(i)])
        if j >= 0:
            acc[j] += acc[int(i)]
    return acc.reshape(H, W)


def river_mask_from_accumulation(acc: np.ndarray, *, threshold: float) -> np.ndarray:
    a = np.asarray(acc, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("acc must be a 2D array")
    return a >= float(threshold)


def carve_rivers(
    height01: np.ndarray,
    acc: np.ndarray,
    river_mask: np.ndarray,
    *,
    depth: float = 0.06,
    exponent: float = 0.5,
) -> np.ndarray:
    """Carve river channels into a 0..1 heightmap (returns a new array)."""

    h = np.asarray(height01, dtype=np.float64)
    a = np.asarray(acc, dtype=np.float64)
    m = np.asarray(river_mask).astype(bool)
    if h.ndim != 2:
        raise ValueError("height01 must be a 2D array")
    if a.shape != h.shape or m.shape != h.shape:
        raise ValueError("acc and river_mask must match height01 shape")

    if not np.any(m):
        return h.copy()

    depth = float(depth)
    exponent = float(exponent)

    a0 = a[m]
    amin = float(np.min(a0))
    amax = float(np.max(a0))
    if amax == amin:
        carved = h.copy()
        carved[m] = np.clip(carved[m] - depth, 0.0, 1.0)
        return carved

    an = (a - amin) / (amax - amin)
    carve = depth * np.power(np.clip(an, 0.0, 1.0), exponent)
    carved = h.copy()
    carved[m] = np.clip(carved[m] - carve[m], 0.0, 1.0)
    return carved
