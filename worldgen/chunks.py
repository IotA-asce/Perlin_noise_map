from __future__ import annotations

from typing import Any, cast

import numpy as np

from worldgen.pipeline import practical_pipeline


def chunk_world_size(*, chunk_size_px: int, scale: float) -> float:
    return float(int(chunk_size_px)) / max(float(scale), 1e-9)


def chunk_origin(
    *, chunk_x: int, chunk_y: int, chunk_size_px: int, scale: float
) -> tuple[float, float]:
    s = chunk_world_size(chunk_size_px=int(chunk_size_px), scale=float(scale))
    return float(int(chunk_x)) * s, float(int(chunk_y)) * s


def generate_chunk(
    *,
    seed: int,
    chunk_x: int,
    chunk_y: int,
    chunk_size_px: int,
    scale: float,
    dtype: object = np.float64,
    **params: Any,
) -> dict[str, np.ndarray | float | None]:
    """Chunk contract: (seed, chunk_x, chunk_y, params) -> deterministic outputs."""

    left, top = chunk_origin(
        chunk_x=int(chunk_x),
        chunk_y=int(chunk_y),
        chunk_size_px=int(chunk_size_px),
        scale=float(scale),
    )
    return practical_pipeline(
        seed=int(seed),
        width=int(chunk_size_px),
        height=int(chunk_size_px),
        view_left=float(left),
        view_top=float(top),
        scale=float(scale),
        dtype=dtype,
        **cast(dict[str, Any], params),
    )
