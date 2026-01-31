from __future__ import annotations

import numpy as np

from worldgen.coast import beach_deposit, box_blur2d, coastline_smooth


def test_box_blur2d_shape() -> None:
    rng = np.random.default_rng(0)
    z = rng.random((21, 17), dtype=np.float64)
    b = box_blur2d(z, radius=3)
    assert b.shape == z.shape


def test_coastline_smooth_deterministic() -> None:
    rng = np.random.default_rng(1)
    z = rng.random((32, 32), dtype=np.float64)
    a = coastline_smooth(z, water_level=0.45, band=0.1, radius=2, strength=0.6)
    b = coastline_smooth(z, water_level=0.45, band=0.1, radius=2, strength=0.6)
    assert np.allclose(a, b)


def test_beach_deposit_bounds() -> None:
    z = np.linspace(0.0, 1.0, 128, dtype=np.float64)
    z = np.tile(z[None, :], (64, 1))
    out = beach_deposit(z, water_level=0.45, shore_level=0.50, amount=0.03)
    assert out.shape == z.shape
    assert float(np.min(out)) >= 0.0
    assert float(np.max(out)) <= 1.0
