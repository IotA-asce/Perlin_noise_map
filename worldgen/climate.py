from __future__ import annotations

import numpy as np


def climate_biome_map(
    height01: np.ndarray,
    temp01: np.ndarray,
    moist01: np.ndarray,
    *,
    water_level: float,
    snowline: float,
) -> np.ndarray:
    """Classify a simple climate biome map.

    Codes:
    - 0 water
    - 4 snow/ice
    - 5 desert
    - 6 grassland
    - 7 forest
    - 8 tundra
    """

    h = np.asarray(height01, dtype=np.float64)
    t = np.asarray(temp01, dtype=np.float64)
    m = np.asarray(moist01, dtype=np.float64)
    if h.ndim != 2 or t.ndim != 2 or m.ndim != 2:
        raise ValueError("height01/temp01/moist01 must be 2D arrays")
    if t.shape != h.shape or m.shape != h.shape:
        raise ValueError("temp01/moist01 must match height01 shape")

    water_level = float(water_level)
    snowline = float(snowline)

    out = np.full(h.shape, 6, dtype=np.uint8)  # grassland

    water = h < water_level
    out[water] = np.uint8(0)

    snow = h >= snowline
    out[snow] = np.uint8(4)

    land = (~water) & (~snow)

    tundra = land & (t < 0.25)
    out[tundra] = np.uint8(8)

    desert = land & (m < 0.22) & (~tundra)
    out[desert] = np.uint8(5)

    forest = land & (m > 0.68) & (~tundra)
    out[forest] = np.uint8(7)

    return out


def apply_climate_palette(
    rgb01: np.ndarray, biome: np.ndarray, *, strength: float = 0.85
) -> np.ndarray:
    """Tint a terrain RGB map based on climate biome codes."""

    rgb = np.asarray(rgb01, dtype=np.float64)
    b = np.asarray(biome, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb01 must be HxWx3")
    if b.shape != rgb.shape[:2]:
        raise ValueError("biome must be HxW")

    strength = float(np.clip(float(strength), 0.0, 1.0))
    out = rgb.copy()

    # Palette tints (keeps shading/detail by blending).
    desert = np.array([0.78, 0.69, 0.40], dtype=np.float64)
    grass = np.array([0.18, 0.50, 0.22], dtype=np.float64)
    forest = np.array([0.10, 0.35, 0.16], dtype=np.float64)
    tundra = np.array([0.46, 0.54, 0.44], dtype=np.float64)

    def blend(mask: np.ndarray, color: np.ndarray) -> None:
        out[mask] = out[mask] * (1.0 - strength) + color * strength

    blend(b == 5, desert)
    blend(b == 6, grass)
    blend(b == 7, forest)
    blend(b == 8, tundra)
    return np.clip(out, 0.0, 1.0)
