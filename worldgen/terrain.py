from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    edge0 = float(edge0)
    edge1 = float(edge1)
    if edge1 <= edge0:
        return np.zeros_like(x, dtype=np.float64)
    t = (np.asarray(x, dtype=np.float64) - edge0) / (edge1 - edge0)
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def slope01(height: np.ndarray) -> np.ndarray:
    """Return a 0..1 normalized slope magnitude map."""
    h = np.asarray(height, dtype=np.float64)
    dzdy, dzdx = np.gradient(h)
    s = np.sqrt(dzdx * dzdx + dzdy * dzdy)
    smin = float(np.min(s))
    smax = float(np.max(s))
    if smax == smin:
        return np.zeros_like(s)
    return (s - smin) / (smax - smin)


def hillshade01(
    height: np.ndarray,
    *,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Approximate hillshade lighting in 0..1.

    Uses a standard gradient-based normal estimate (Horn-style hillshade).
    """

    h = np.asarray(height, dtype=np.float64) * float(z_factor)
    dzdy, dzdx = np.gradient(h)

    az = np.deg2rad(float(azimuth_deg))
    alt = np.deg2rad(float(altitude_deg))

    lx = np.cos(alt) * np.sin(az)
    ly = np.cos(alt) * np.cos(az)
    lz = np.sin(alt)

    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(h)
    nrm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= nrm
    ny /= nrm
    nz /= nrm

    shade = (nx * lx) + (ny * ly) + (nz * lz)
    return np.clip(shade, 0.0, 1.0)


@dataclass(frozen=True)
class TerrainMasks:
    water: np.ndarray
    shore: np.ndarray
    land: np.ndarray
    mountain: np.ndarray
    snow: np.ndarray


def terrain_masks(
    height01: np.ndarray,
    *,
    water_level: float,
    shore_level: float,
    mountain_level: float,
    snowline: float,
    slope01_map: np.ndarray | None = None,
    snow_slope_start: float = 0.35,
    snow_slope_end: float = 0.75,
) -> TerrainMasks:
    h = np.asarray(height01, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("height01 must be a 2D array")

    water_level = float(water_level)
    shore_level = float(shore_level)
    mountain_level = float(mountain_level)
    snowline = float(snowline)

    water = h < water_level
    shore = (h >= water_level) & (h < shore_level)
    mountain = h >= mountain_level
    land = (~water) & (~shore) & (~mountain)

    if slope01_map is None:
        slope01_map = slope01(h)
    s = np.asarray(slope01_map, dtype=np.float64)
    if s.shape != h.shape:
        raise ValueError("slope01_map must have the same shape as height01")

    snow_height = _smoothstep(snowline, 1.0, h)
    snow_slope = 1.0 - _smoothstep(float(snow_slope_start), float(snow_slope_end), s)
    snow = (snow_height * snow_slope) > 0.5

    return TerrainMasks(
        water=water, shore=shore, land=land, mountain=mountain, snow=snow
    )


def terrain_colormap(
    height01: np.ndarray,
    *,
    water_level: float = 0.45,
    shore_level: float = 0.50,
    mountain_level: float = 0.75,
    snowline: float = 0.83,
    slope01_map: np.ndarray | None = None,
    shade01: np.ndarray | None = None,
    shade_strength: float = 0.55,
    rivers: np.ndarray | None = None,
    river_strength: float = 0.75,
) -> np.ndarray:
    """Return an RGB terrain visualization in 0..1 float64.

    Inputs are expected to be normalized height values (0..1). This function is
    deterministic and does not mutate inputs.
    """

    h = np.asarray(height01, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("height01 must be a 2D array")

    water_level = float(water_level)
    shore_level = float(shore_level)
    mountain_level = float(mountain_level)
    snowline = float(snowline)

    if not (0.0 <= water_level <= 1.0):
        raise ValueError("water_level must be in [0, 1]")
    if not (0.0 <= shore_level <= 1.0):
        raise ValueError("shore_level must be in [0, 1]")
    if not (0.0 <= mountain_level <= 1.0):
        raise ValueError("mountain_level must be in [0, 1]")

    if slope01_map is None:
        slope01_map = slope01(h)
    s = np.asarray(slope01_map, dtype=np.float64)
    if s.shape != h.shape:
        raise ValueError("slope01_map must have the same shape as height01")

    masks = terrain_masks(
        h,
        water_level=water_level,
        shore_level=shore_level,
        mountain_level=mountain_level,
        snowline=snowline,
        slope01_map=s,
    )

    rgb = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.float64)

    # Water: deep -> shallow.
    if np.any(masks.water):
        t = np.clip(h / max(water_level, 1e-9), 0.0, 1.0)
        deep = np.array([0.02, 0.12, 0.28], dtype=np.float64)
        shallow = np.array([0.10, 0.48, 0.62], dtype=np.float64)
        water_rgb = deep + (shallow - deep) * t[..., None]
        rgb[masks.water] = water_rgb[masks.water]

    # Shore: sand band.
    if np.any(masks.shore):
        t = _smoothstep(water_level, max(shore_level, water_level + 1e-9), h)
        sand0 = np.array([0.76, 0.70, 0.50], dtype=np.float64)
        sand1 = np.array([0.63, 0.60, 0.42], dtype=np.float64)
        shore_rgb = sand0 + (sand1 - sand0) * t[..., None]
        rgb[masks.shore] = shore_rgb[masks.shore]

    # Land: greens -> dry -> rocky.
    if np.any(masks.land):
        t = _smoothstep(shore_level, mountain_level, h)
        low = np.array([0.12, 0.44, 0.16], dtype=np.float64)
        mid = np.array([0.46, 0.52, 0.23], dtype=np.float64)
        high = np.array([0.52, 0.50, 0.42], dtype=np.float64)
        land_rgb = np.where(
            (t[..., None] < 0.5),
            low + (mid - low) * (t[..., None] * 2.0),
            mid + (high - mid) * ((t[..., None] - 0.5) * 2.0),
        )
        rgb[masks.land] = land_rgb[masks.land]

    # Mountains: rock -> bright rock.
    if np.any(masks.mountain):
        t = _smoothstep(mountain_level, 1.0, h)
        rock0 = np.array([0.40, 0.38, 0.36], dtype=np.float64)
        rock1 = np.array([0.66, 0.66, 0.65], dtype=np.float64)
        mountain_rgb = rock0 + (rock1 - rock0) * t[..., None]
        rgb[masks.mountain] = mountain_rgb[masks.mountain]

    # Snow/ice: altitude-based, reduced on steep slopes.
    snow_h = _smoothstep(snowline, 1.0, h)
    snow_s = 1.0 - _smoothstep(0.35, 0.75, s)
    snow = np.clip(snow_h * snow_s, 0.0, 1.0)
    snow_rgb = np.array([0.92, 0.94, 0.98], dtype=np.float64)
    rgb = rgb * (1.0 - snow[..., None]) + snow_rgb * snow[..., None]

    # Shading.
    if shade01 is not None:
        sh = np.asarray(shade01, dtype=np.float64)
        if sh.shape != h.shape:
            raise ValueError("shade01 must have the same shape as height01")
        shade_strength = float(shade_strength)
        shade_strength = float(np.clip(shade_strength, 0.0, 1.0))
        # Lift shadows a bit so colors remain readable.
        light = (1.0 - shade_strength) + shade_strength * (0.35 + 0.65 * sh)
        rgb = np.clip(rgb * light[..., None], 0.0, 1.0)

    # River overlay.
    if rivers is not None:
        r = np.asarray(rivers)
        if r.shape != h.shape:
            raise ValueError("rivers must have the same shape as height01")
        river_strength = float(np.clip(float(river_strength), 0.0, 1.0))
        river_rgb = np.array([0.05, 0.40, 0.72], dtype=np.float64)
        m = r.astype(bool)
        rgb[m] = rgb[m] * (1.0 - river_strength) + river_rgb * river_strength

    return rgb
