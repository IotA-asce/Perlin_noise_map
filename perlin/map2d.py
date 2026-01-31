from __future__ import annotations

import math

import numpy as np

from perlin.noise_2d import (
    Perlin2D,
    domain_warp2,
    fbm2,
    ridged2,
    tileable2d,
    turbulence2,
)
from perlin.value_noise_2d import ValueNoise2D


def noise_map_2d(
    *,
    seed: int,
    basis: str,
    grad_set: str,
    width: int,
    height: int,
    scale: float,
    octaves: int,
    lacunarity: float,
    persistence: float,
    variant: str,
    warp_amp: float,
    warp_scale: float,
    warp_octaves: int,
    offset_x: float,
    offset_y: float,
    normalize: bool,
    tileable: bool,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    """Generate a deterministic 2D noise map.

    This is UI-agnostic and intended to be the shared reference implementation
    used by Streamlit and benchmarks.
    """

    basis = str(basis)
    grad_set = str(grad_set)
    if basis == "perlin":
        noise = Perlin2D(seed=int(seed), grad_set=grad_set)
    elif basis == "value":
        noise = ValueNoise2D(seed=int(seed))
    else:
        raise ValueError(f"unknown basis: {basis}")

    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0")

    scale = float(scale)
    scale = max(scale, 1e-9)

    offset_x = float(offset_x)
    offset_y = float(offset_y)

    period_x = 0.0
    period_y = 0.0
    if bool(tileable):
        period_x = (float(width) - 1.0) / scale
        period_y = (float(height) - 1.0) / scale
        xs = np.linspace(offset_x, offset_x + period_x, width, dtype=np.float64)
        ys = np.linspace(offset_y, offset_y + period_y, height, dtype=np.float64)
    else:
        xs = (np.arange(width, dtype=np.float64) / scale) + offset_x
        ys = (np.arange(height, dtype=np.float64) / scale) + offset_y

    xg, yg = np.meshgrid(xs, ys)

    variant = str(variant)
    warp_amp = float(warp_amp)
    warp_scale = float(warp_scale)
    warp_octaves = int(warp_octaves)

    def base(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        if variant == "fbm":
            return fbm2(
                noise,
                xx,
                yy,
                octaves=int(octaves),
                lacunarity=float(lacunarity),
                persistence=float(persistence),
            )
        if variant == "turbulence":
            return turbulence2(
                noise,
                xx,
                yy,
                octaves=int(octaves),
                lacunarity=float(lacunarity),
                persistence=float(persistence),
            )
        if variant == "ridged":
            return ridged2(
                noise,
                xx,
                yy,
                octaves=int(octaves),
                lacunarity=float(lacunarity),
                persistence=float(persistence),
            )
        if variant == "domain_warp":
            return domain_warp2(
                noise,
                xx,
                yy,
                octaves=int(octaves),
                lacunarity=float(lacunarity),
                persistence=float(persistence),
                warp_amp=float(warp_amp),
                warp_scale=float(warp_scale),
                warp_octaves=int(warp_octaves),
                warp_lacunarity=float(lacunarity),
                warp_persistence=float(persistence),
            )
        raise ValueError(f"unknown variant: {variant}")

    z = (
        tileable2d(base, xg, yg, period_x=period_x, period_y=period_y)
        if bool(tileable)
        else base(xg, yg)
    )

    if dtype is not None:
        z = np.asarray(z, dtype=dtype)

    if not bool(normalize):
        return z

    z = np.asarray(z, dtype=np.float64)
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    if math.isclose(zmin, zmax):
        return np.zeros_like(z)
    return (z - zmin) / (zmax - zmin)
