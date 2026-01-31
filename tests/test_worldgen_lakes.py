from __future__ import annotations

import numpy as np

from worldgen.lakes import fill_depressions_priority_flood


def test_fill_depressions_priority_flood_shape_and_bounds() -> None:
    z = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 0.2, 0.2, 1.0],
            [1.0, 0.2, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    filled, depth = fill_depressions_priority_flood(z)
    assert filled.shape == z.shape
    assert depth.shape == z.shape
    assert float(np.min(depth)) >= 0.0
    assert bool(np.all(filled >= z))


def test_fill_depressions_priority_flood_deterministic() -> None:
    rng = np.random.default_rng(0)
    z = rng.random((32, 32), dtype=np.float64)
    a = fill_depressions_priority_flood(z)
    b = fill_depressions_priority_flood(z)
    assert np.allclose(a[0], b[0])
    assert np.allclose(a[1], b[1])
