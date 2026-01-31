from __future__ import annotations

import numpy as np

from perlin.map2d import noise_map_2d


def test_noise_map_2d_shape_and_deterministic() -> None:
    z1 = noise_map_2d(
        seed=0,
        basis="perlin",
        grad_set="diag8",
        width=64,
        height=48,
        scale=120.0,
        octaves=4,
        lacunarity=2.0,
        persistence=0.5,
        variant="fbm",
        warp_amp=1.25,
        warp_scale=1.5,
        warp_octaves=2,
        offset_x=0.0,
        offset_y=0.0,
        normalize=False,
        tileable=False,
    )
    z2 = noise_map_2d(
        seed=0,
        basis="perlin",
        grad_set="diag8",
        width=64,
        height=48,
        scale=120.0,
        octaves=4,
        lacunarity=2.0,
        persistence=0.5,
        variant="fbm",
        warp_amp=1.25,
        warp_scale=1.5,
        warp_octaves=2,
        offset_x=0.0,
        offset_y=0.0,
        normalize=False,
        tileable=False,
    )
    assert z1.shape == (48, 64)
    assert np.allclose(z1, z2)


def test_noise_map_2d_normalize_bounds() -> None:
    z = noise_map_2d(
        seed=1,
        basis="perlin",
        grad_set="diag8",
        width=64,
        height=64,
        scale=80.0,
        octaves=3,
        lacunarity=2.0,
        persistence=0.55,
        variant="fbm",
        warp_amp=0.0,
        warp_scale=1.0,
        warp_octaves=1,
        offset_x=0.0,
        offset_y=0.0,
        normalize=True,
        tileable=False,
    )
    assert float(np.min(z)) >= 0.0
    assert float(np.max(z)) <= 1.0


def test_noise_map_2d_tileable_edges_match() -> None:
    z = noise_map_2d(
        seed=0,
        basis="perlin",
        grad_set="diag8",
        width=96,
        height=72,
        scale=100.0,
        octaves=4,
        lacunarity=2.0,
        persistence=0.5,
        variant="fbm",
        warp_amp=1.25,
        warp_scale=1.5,
        warp_octaves=2,
        offset_x=3.7,
        offset_y=-1.2,
        normalize=False,
        tileable=True,
    )
    assert np.allclose(z[:, 0], z[:, -1])
    assert np.allclose(z[0, :], z[-1, :])


def test_noise_map_2d_supports_value_basis() -> None:
    z = noise_map_2d(
        seed=0,
        basis="value",
        grad_set="diag8",
        width=32,
        height=32,
        scale=60.0,
        octaves=3,
        lacunarity=2.0,
        persistence=0.5,
        variant="turbulence",
        warp_amp=0.0,
        warp_scale=1.0,
        warp_octaves=1,
        offset_x=0.0,
        offset_y=0.0,
        normalize=False,
        tileable=False,
    )
    assert z.shape == (32, 32)
    assert np.isfinite(z).all()
