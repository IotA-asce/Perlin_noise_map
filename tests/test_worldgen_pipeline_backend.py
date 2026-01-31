from __future__ import annotations

import numpy as np

from worldgen.pipeline import practical_pipeline


def test_practical_pipeline_fast_backend_close_to_reference() -> None:
    params = dict(
        seed=0,
        basis="perlin",
        grad2="diag8",
        noise_variant="fbm",
        warp_amp=1.25,
        warp_scale=1.5,
        warp_octaves=2,
        scale=120.0,
        octaves=4,
        lacunarity=2.0,
        persistence=0.5,
        width=64,
        height=64,
        view_left=0.0,
        view_top=0.0,
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
        thermal_on=True,
        thermal_iter=10,
        thermal_talus=0.02,
        thermal_strength=0.35,
        hydraulic_on=True,
        hyd_iter=10,
        hyd_rain=0.01,
        hyd_evap=0.5,
        hyd_flow=0.5,
        hyd_capacity=4.0,
        hyd_erosion=0.3,
        hyd_deposition=0.3,
    )

    ref = practical_pipeline(**params, dtype=np.float64)
    fast = practical_pipeline(**params, dtype=np.float32)

    href = np.asarray(ref["terr_river01"], dtype=np.float64)
    hfast = np.asarray(fast["terr_river01"], dtype=np.float64)
    assert href.shape == hfast.shape
    # Float32 is approximate but should be close.
    assert float(np.max(np.abs(href - hfast))) < 2e-3
