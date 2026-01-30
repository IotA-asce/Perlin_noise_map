from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def fade(t: np.ndarray) -> np.ndarray:
    """Quintic fade curve used by Improved Perlin Noise (2002)."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def make_permutation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    p = rng.permutation(256).astype(np.int32)
    return np.concatenate([p, p])


@dataclass(frozen=True)
class Corner2D:
    gx: float
    gy: float
    dx: float
    dy: float
    dot: float


_GRAD2 = np.array(
    [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [1.0, -1.0],
        [-1.0, -1.0],
    ],
    dtype=np.float64,
)
_GRAD2 /= np.linalg.norm(_GRAD2, axis=1, keepdims=True)


def grad2_from_hash(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = (h & 7).astype(np.int32)
    g = _GRAD2[idx]
    return g[..., 0], g[..., 1]


_GRAD1 = np.array([1.0, -1.0], dtype=np.float64)


def grad1_from_hash(h: np.ndarray) -> np.ndarray:
    idx = (h & 1).astype(np.int32)
    return _GRAD1[idx]


_GRAD3 = np.array(
    [
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, -1.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.0, -1.0, -1.0],
    ],
    dtype=np.float64,
)
_GRAD3 /= np.linalg.norm(_GRAD3, axis=1, keepdims=True)


def grad3_from_hash(h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = (h % 12).astype(np.int32)
    g = _GRAD3[idx]
    return g[..., 0], g[..., 1], g[..., 2]
