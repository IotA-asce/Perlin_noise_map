from __future__ import annotations

import numpy as np

from worldgen.climate import apply_climate_palette, climate_biome_map
from worldgen.contours import apply_mask_overlay, contour_mask
from worldgen.paths import astar_path
from worldgen.tiles import tiles_zip_from_rgb
from worldgen.vegetation import filter_points_by_mask, jittered_points


def test_climate_biome_map_shape() -> None:
    rng = np.random.default_rng(0)
    h = rng.random((32, 40), dtype=np.float64)
    t = rng.random((32, 40), dtype=np.float64)
    m = rng.random((32, 40), dtype=np.float64)
    b = climate_biome_map(h, t, m, water_level=0.45, snowline=0.83)
    assert b.shape == h.shape


def test_apply_climate_palette_shape() -> None:
    rng = np.random.default_rng(1)
    rgb = rng.random((16, 16, 3), dtype=np.float64)
    biome = rng.integers(0, 9, size=(16, 16), dtype=np.uint8)
    out = apply_climate_palette(rgb, biome, strength=0.5)
    assert out.shape == rgb.shape


def test_jittered_points_deterministic() -> None:
    x1, y1 = jittered_points(seed=123, height=128, width=128, cell=16, probability=0.5)
    x2, y2 = jittered_points(seed=123, height=128, width=128, cell=16, probability=0.5)
    assert np.allclose(x1, x2)
    assert np.allclose(y1, y2)


def test_filter_points_by_mask() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[5:, 5:] = True
    xs = np.array([1, 8], dtype=np.float64)
    ys = np.array([1, 8], dtype=np.float64)
    xo, yo = filter_points_by_mask(xs, ys, mask)
    assert xo.shape == yo.shape
    assert xo.size == 1


def test_astar_path_basic() -> None:
    cost = np.ones((10, 10), dtype=np.float64)
    cost[5, :] = np.inf
    cost[5, 5] = 1.0
    path = astar_path(cost, start=(0, 0), goal=(9, 9))
    assert path
    assert path[0] == (0, 0)
    assert path[-1] == (9, 9)


def test_contour_mask_shape() -> None:
    z = np.linspace(0.0, 1.0, 100, dtype=np.float64).reshape(10, 10)
    m = contour_mask(z, interval=0.1)
    assert m.shape == z.shape


def test_apply_mask_overlay_shape() -> None:
    rgb = np.zeros((10, 10, 3), dtype=np.float64)
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, :] = True
    out = apply_mask_overlay(rgb, mask)
    assert out.shape == rgb.shape


def test_tiles_zip_from_rgb_nonempty() -> None:
    rgb = np.zeros((64, 64, 3), dtype=np.float64)
    data = tiles_zip_from_rgb(rgb, z=0, grid=2, tile_size=32)
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > 0
