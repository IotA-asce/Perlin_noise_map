from __future__ import annotations

import numpy as np

from perlin.map2d import noise_map_2d
from worldgen.coast import beach_deposit, coastline_smooth
from worldgen.erosion import hydraulic_erosion, thermal_erosion
from worldgen.hydrology import (
    carve_rivers,
    flow_accumulation_d8,
    flow_direction_d8,
    river_mask_from_accumulation,
)
from worldgen.lakes import fill_depressions_priority_flood
from worldgen.terrain import hillshade01, slope01, terrain_colormap, terrain_masks


def practical_pipeline(
    *,
    seed: int,
    basis: str,
    grad2: str,
    noise_variant: str,
    warp_amp: float,
    warp_scale: float,
    warp_octaves: int,
    scale: float,
    octaves: int,
    lacunarity: float,
    persistence: float,
    width: int,
    height: int,
    view_left: float,
    view_top: float,
    z_scale: float,
    water_level: float,
    shore_width: float,
    mountain_level: float,
    snowline: float,
    shade_az: float,
    shade_alt: float,
    shade_strength: float,
    river_q: float,
    river_carve: bool,
    river_depth: float,
    fill_lakes: bool,
    coast_smooth: bool,
    coast_radius: int,
    coast_strength: float,
    beach: bool,
    beach_amount: float,
    thermal_on: bool,
    thermal_iter: int,
    thermal_talus: float,
    thermal_strength: float,
    hydraulic_on: bool,
    hyd_iter: int,
    hyd_rain: float,
    hyd_evap: float,
    hyd_flow: float,
    hyd_capacity: float,
    hyd_erosion: float,
    hyd_deposition: float,
    dtype: np.dtype | type[np.floating] = np.float64,
) -> dict[str, np.ndarray | float | None]:
    """Worldgen pipeline used by the Practical page.

    This is a pure function (no Streamlit dependency) so it can be used by
    benchmarks and chunk contracts.
    """

    z_raw = noise_map_2d(
        seed=int(seed),
        basis=str(basis),
        grad_set=str(grad2),
        width=int(width),
        height=int(height),
        scale=float(scale),
        octaves=int(octaves),
        lacunarity=float(lacunarity),
        persistence=float(persistence),
        variant=str(noise_variant),
        warp_amp=float(warp_amp),
        warp_scale=float(warp_scale),
        warp_octaves=int(warp_octaves),
        offset_x=float(view_left),
        offset_y=float(view_top),
        normalize=False,
        tileable=False,
        dtype=np.dtype(np.float64),
    )

    if str(noise_variant) in {"turbulence", "ridged"}:
        base01 = np.clip(np.asarray(z_raw, dtype=np.float64), 0.0, 1.0)
    else:
        base01 = np.clip((np.asarray(z_raw, dtype=np.float64) + 1.0) * 0.5, 0.0, 1.0)

    terr01 = np.asarray(base01, dtype=np.float64)
    hyd_water = None
    hyd_sediment = None

    if bool(thermal_on):
        terr01 = thermal_erosion(
            terr01,
            iterations=int(thermal_iter),
            talus=float(thermal_talus),
            strength=float(thermal_strength),
        )

    if bool(hydraulic_on):
        terr01, hyd_water, hyd_sediment = hydraulic_erosion(
            terr01,
            iterations=int(hyd_iter),
            rain=float(hyd_rain),
            evaporation=float(hyd_evap),
            flow_rate=float(hyd_flow),
            capacity=float(hyd_capacity),
            erosion=float(hyd_erosion),
            deposition=float(hyd_deposition),
        )

    lake_depth = np.zeros_like(terr01)
    if bool(fill_lakes):
        terr01, lake_depth = fill_depressions_priority_flood(terr01)

    shore_level = min(1.0, float(water_level) + float(shore_width))

    if bool(coast_smooth):
        terr01 = coastline_smooth(
            terr01,
            water_level=float(water_level),
            band=float(shore_width) * 2.5,
            radius=int(coast_radius),
            strength=float(coast_strength),
        )

    if bool(beach):
        terr01 = beach_deposit(
            terr01,
            water_level=float(water_level),
            shore_level=float(shore_level),
            amount=float(beach_amount),
        )

    s01 = slope01(terr01)
    shade01 = hillshade01(
        terr01,
        azimuth_deg=float(shade_az),
        altitude_deg=float(shade_alt),
        z_factor=2.0,
    )

    ds = flow_direction_d8(terr01)
    acc = flow_accumulation_d8(terr01, ds)
    thr = float(np.quantile(acc, float(river_q)))
    rivers = river_mask_from_accumulation(acc, threshold=thr)

    terr_river01 = terr01
    if bool(river_carve):
        terr_river01 = carve_rivers(
            terr01,
            acc,
            rivers,
            depth=float(river_depth),
        )

    masks = terrain_masks(
        terr_river01,
        water_level=float(water_level),
        shore_level=float(shore_level),
        mountain_level=float(mountain_level),
        snowline=float(snowline),
        slope01_map=s01,
    )
    biome = np.full(terr_river01.shape, 2, dtype=np.uint8)
    biome[masks.water] = np.uint8(0)
    biome[masks.shore] = np.uint8(1)
    biome[masks.mountain] = np.uint8(3)
    biome[masks.snow] = np.uint8(4)

    rgb = terrain_colormap(
        terr_river01,
        water_level=float(water_level),
        shore_level=float(shore_level),
        mountain_level=float(mountain_level),
        snowline=float(snowline),
        slope01_map=s01,
        shade01=shade01,
        shade_strength=float(shade_strength),
        rivers=None,
    )

    out: dict[str, np.ndarray | float | None] = {
        "base01": np.asarray(base01),
        "terr01": np.asarray(terr01),
        "terr_river01": np.asarray(terr_river01),
        "s01": np.asarray(s01),
        "shade01": np.asarray(shade01),
        "lake_depth": np.asarray(lake_depth),
        "ds": np.asarray(ds),
        "acc": np.asarray(acc),
        "thr": float(thr),
        "rivers": np.asarray(rivers),
        "rgb": np.asarray(rgb),
        "biome": np.asarray(biome),
        "shore_level": float(shore_level),
        "hyd_water": None if hyd_water is None else np.asarray(hyd_water),
        "hyd_sediment": None if hyd_sediment is None else np.asarray(hyd_sediment),
        "z_scale": float(z_scale),
    }

    use_f32 = (dtype == np.float32) or (dtype == np.dtype(np.float32))

    if bool(use_f32):
        for k in [
            "base01",
            "terr01",
            "terr_river01",
            "s01",
            "shade01",
            "lake_depth",
            "acc",
            "rgb",
        ]:
            out[k] = np.asarray(out[k], dtype=np.float32)
        if out["hyd_water"] is not None:
            out["hyd_water"] = np.asarray(out["hyd_water"], dtype=np.float32)
        if out["hyd_sediment"] is not None:
            out["hyd_sediment"] = np.asarray(out["hyd_sediment"], dtype=np.float32)

    return out
