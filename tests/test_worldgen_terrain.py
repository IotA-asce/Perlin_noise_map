from __future__ import annotations

import numpy as np

from worldgen.terrain import hillshade01, slope01, terrain_colormap, terrain_masks


def test_slope01_shape_and_bounds() -> None:
    h = np.linspace(0.0, 1.0, 64, dtype=np.float64)
    z = np.tile(h[None, :], (32, 1))
    s = slope01(z)
    assert s.shape == z.shape
    assert float(np.min(s)) >= 0.0
    assert float(np.max(s)) <= 1.0


def test_hillshade01_shape_and_bounds() -> None:
    rng = np.random.default_rng(0)
    z = rng.random((64, 64), dtype=np.float64)
    sh = hillshade01(z, azimuth_deg=315.0, altitude_deg=45.0, z_factor=2.0)
    assert sh.shape == z.shape
    assert float(np.min(sh)) >= 0.0
    assert float(np.max(sh)) <= 1.0


def test_terrain_masks_basic_partition() -> None:
    z = np.zeros((8, 8), dtype=np.float64)
    z[:, :2] = 0.1
    z[:, 2:4] = 0.48
    z[:, 4:6] = 0.60
    z[:, 6:] = 0.90

    s = np.zeros_like(z)
    masks = terrain_masks(
        z,
        water_level=0.45,
        shore_level=0.50,
        mountain_level=0.75,
        snowline=0.85,
        slope01_map=s,
    )

    assert masks.water.shape == z.shape
    assert masks.snow.shape == z.shape

    covered = masks.water | masks.shore | masks.land | masks.mountain
    assert bool(np.all(covered))
    assert bool(np.all(~(masks.water & masks.mountain)))


def test_terrain_colormap_shape_and_range() -> None:
    rng = np.random.default_rng(1)
    z = rng.random((32, 40), dtype=np.float64)
    s = slope01(z)
    sh = hillshade01(z)
    rgb = terrain_colormap(z, slope01_map=s, shade01=sh)
    assert rgb.shape == (32, 40, 3)
    assert float(np.min(rgb)) >= 0.0
    assert float(np.max(rgb)) <= 1.0
