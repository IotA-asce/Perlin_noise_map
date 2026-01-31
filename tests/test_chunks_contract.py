from __future__ import annotations

import numpy as np

from worldgen.chunks import chunk_origin, chunk_world_size, generate_chunk


def test_chunk_origin_and_size() -> None:
    cw = chunk_world_size(chunk_size_px=256, scale=128.0)
    assert cw == 2.0
    left, top = chunk_origin(chunk_x=-2, chunk_y=3, chunk_size_px=256, scale=128.0)
    assert left == -4.0
    assert top == 6.0


def test_generate_chunk_deterministic_for_same_inputs() -> None:
    params = dict(
        basis="perlin",
        grad2="diag8",
        noise_variant="fbm",
        warp_amp=1.25,
        warp_scale=1.5,
        warp_octaves=2,
        octaves=4,
        lacunarity=2.0,
        persistence=0.5,
        z_scale=80.0,
        water_level=0.45,
        shore_width=0.05,
        mountain_level=0.75,
        snowline=0.83,
        shade_az=315.0,
        shade_alt=45.0,
        shade_strength=0.55,
        river_q=0.985,
        river_carve=True,
        river_depth=0.06,
        fill_lakes=True,
        coast_smooth=True,
        coast_radius=2,
        coast_strength=0.6,
        beach=True,
        beach_amount=0.02,
        thermal_on=False,
        thermal_iter=0,
        thermal_talus=0.02,
        thermal_strength=0.35,
        hydraulic_on=False,
        hyd_iter=0,
        hyd_rain=0.01,
        hyd_evap=0.5,
        hyd_flow=0.5,
        hyd_capacity=4.0,
        hyd_erosion=0.3,
        hyd_deposition=0.3,
    )

    a = generate_chunk(
        seed=0,
        chunk_x=1,
        chunk_y=-2,
        chunk_size_px=64,
        scale=120.0,
        dtype=np.float64,
        **params,
    )
    b = generate_chunk(
        seed=0,
        chunk_x=1,
        chunk_y=-2,
        chunk_size_px=64,
        scale=120.0,
        dtype=np.float64,
        **params,
    )

    for k in ["terr_river01", "acc", "rgb", "biome"]:
        assert np.allclose(np.asarray(a[k]), np.asarray(b[k]))
