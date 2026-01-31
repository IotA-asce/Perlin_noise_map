from __future__ import annotations

import numpy as np

from worldgen.hydrology import (
    carve_rivers,
    flow_accumulation_d8,
    flow_direction_d8,
    river_mask_from_accumulation,
)


def test_flow_direction_shape_and_sinks() -> None:
    # Monotone slope to bottom-right.
    y = np.linspace(1.0, 0.0, 16, dtype=np.float64)[:, None]
    x = np.linspace(1.0, 0.0, 16, dtype=np.float64)[None, :]
    z = y + x

    ds = flow_direction_d8(z)
    assert ds.shape == z.shape

    # Lowest corner is a sink.
    assert int(ds[-1, -1]) == -1


def test_flow_accumulation_deterministic() -> None:
    rng = np.random.default_rng(0)
    z = rng.random((32, 32), dtype=np.float64)
    ds = flow_direction_d8(z)
    a1 = flow_accumulation_d8(z, ds)
    a2 = flow_accumulation_d8(z, ds)
    assert np.allclose(a1, a2)


def test_river_mask_and_carving() -> None:
    rng = np.random.default_rng(2)
    z = rng.random((32, 32), dtype=np.float64)
    ds = flow_direction_d8(z)
    acc = flow_accumulation_d8(z, ds)
    rivers = river_mask_from_accumulation(acc, threshold=float(np.quantile(acc, 0.98)))
    carved = carve_rivers(z, acc, rivers, depth=0.05)
    assert carved.shape == z.shape
    assert float(np.min(carved)) >= 0.0
    assert float(np.max(carved)) <= 1.0
    if bool(np.any(rivers)):
        assert float(np.mean(carved[rivers])) <= float(np.mean(z[rivers]))
