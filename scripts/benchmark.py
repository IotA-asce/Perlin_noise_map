from __future__ import annotations

import time

import numpy as np

from perlin.map2d import noise_map_2d
from worldgen.pipeline import practical_pipeline


def _timeit(label: str, fn) -> float:
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0
    print(f"{label}: {ms:.2f} ms")
    return ms


def main() -> None:
    """Quick CPU benchmark.

    Intended targets (laptop-class CPU):
    - Explore noise map 512x512: < ~100ms
    - Practical chunk 256x256: < ~250ms (depends heavily on erosion/hydrology toggles)
    """

    seed = 0
    basis = "perlin"
    grad2 = "diag8"

    _timeit(
        "Explore: noise_map_2d 512x512",
        lambda: noise_map_2d(
            seed=seed,
            basis=basis,
            grad_set=grad2,
            width=512,
            height=512,
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
            normalize=True,
            tileable=False,
        ),
    )

    params = dict(
        seed=seed,
        basis=basis,
        grad2=grad2,
        noise_variant="fbm",
        warp_amp=1.25,
        warp_scale=1.5,
        warp_octaves=2,
        scale=120.0,
        octaves=4,
        lacunarity=2.0,
        persistence=0.5,
        width=256,
        height=256,
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
        thermal_iter=30,
        thermal_talus=0.02,
        thermal_strength=0.35,
        hydraulic_on=True,
        hyd_iter=30,
        hyd_rain=0.01,
        hyd_evap=0.5,
        hyd_flow=0.5,
        hyd_capacity=4.0,
        hyd_erosion=0.3,
        hyd_deposition=0.3,
    )

    def run_ref() -> None:
        out = practical_pipeline(**params, dtype=np.float64)
        _ = float(np.mean(np.asarray(out["terr_river01"])))

    def run_fast() -> None:
        out = practical_pipeline(**params, dtype=np.float32)
        _ = float(np.mean(np.asarray(out["terr_river01"])))

    _timeit("Practical: chunk 256x256 (reference)", run_ref)
    _timeit("Practical: chunk 256x256 (fast float32)", run_fast)


if __name__ == "__main__":
    main()
