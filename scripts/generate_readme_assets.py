from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _rgb01_to_pil(rgb01: np.ndarray) -> Image.Image:
    rgb = np.asarray(rgb01, dtype=np.float64)
    rgb = np.clip(rgb, 0.0, 1.0)
    img = (rgb * 255.0).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


def _gray01_to_pil(gray01: np.ndarray) -> Image.Image:
    g = np.asarray(gray01, dtype=np.float64)
    g = np.clip(g, 0.0, 1.0)
    img = (g * 255.0).astype(np.uint8)
    return Image.fromarray(img, mode="L")


def _overlay(
    rgb01: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[float, float, float],
    alpha: float,
) -> np.ndarray:
    rgb = np.asarray(rgb01, dtype=np.float64)
    m = np.asarray(mask).astype(bool)
    a = float(np.clip(float(alpha), 0.0, 1.0))
    c = np.array(color, dtype=np.float64)
    out = rgb.copy()
    out[m] = out[m] * (1.0 - a) + c * a
    return np.clip(out, 0.0, 1.0)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from worldgen.erosion import thermal_erosion_frames
    from worldgen.pipeline import practical_pipeline
    from worldgen.terrain import hillshade01, terrain_colormap

    out_dir = root / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # A representative Practical scene.
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
        width=512,
        height=512,
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
    out = practical_pipeline(**params, dtype=np.float64)

    rgb = np.asarray(out["rgb"], dtype=np.float64)
    rivers = np.asarray(out["rivers"]).astype(bool)
    lake_depth = np.asarray(out["lake_depth"], dtype=np.float64)

    if float(np.max(lake_depth)) > 0.0:
        lake_mask = lake_depth > 1e-12
        rgb = _overlay(rgb, lake_mask, color=(0.20, 0.62, 0.92), alpha=0.22)

    rgb = _overlay(rgb, rivers, color=(0.05, 0.40, 0.72), alpha=0.75)
    _rgb01_to_pil(rgb).save(out_dir / "practical_terrain.png")

    # Hydrology accumulation preview (normalized log).
    acc = np.asarray(out["acc"], dtype=np.float64)
    loga = np.log1p(acc)
    mx = float(np.max(loga))
    g = (loga / mx) if mx > 0.0 else np.zeros_like(loga)
    _gray01_to_pil(g).save(out_dir / "practical_flow_accumulation.png")

    # A small thermal erosion animation.
    base01 = np.asarray(out["base01"], dtype=np.float64)
    base_small = base01[::2, ::2]
    iters = 60
    every = 3
    frames_h = thermal_erosion_frames(
        base_small,
        iterations=iters,
        talus=0.02,
        strength=0.35,
        every=every,
    )

    water_level = float(params["water_level"])
    shore_level = min(1.0, water_level + float(params["shore_width"]))

    pil_frames: list[Image.Image] = []
    for i in range(int(frames_h.shape[0])):
        h = np.asarray(frames_h[i], dtype=np.float64)
        sh = hillshade01(h, azimuth_deg=315.0, altitude_deg=45.0, z_factor=2.0)
        rgbf = terrain_colormap(
            h,
            water_level=water_level,
            shore_level=shore_level,
            mountain_level=float(params["mountain_level"]),
            snowline=float(params["snowline"]),
            shade01=sh,
            shade_strength=0.6,
        )
        pil_frames.append(_rgb01_to_pil(rgbf))

    duration_ms = int(math.ceil(1000.0 / 12.0))
    pil_frames[0].save(
        out_dir / "thermal_erosion.gif",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


if __name__ == "__main__":
    main()
