from __future__ import annotations

import io
import json
import math
import time
from typing import Any, cast
from urllib.parse import urlencode

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from perlin.map2d import noise_map_2d
from perlin.noise_2d import Perlin2D
from st_components.hotkeys import hotkeys
from st_components.live_slider import live_slider
from ui.styles import inject_global_styles
from viz.export import array_to_npy_bytes, array_to_png_bytes, heightmap_to_obj_bytes
from viz.step_2d import (
    fade_curve_figure,
    perlin2d_cell_figure,
    scanline_dots_figure,
    scanline_figure,
    scanline_series_from_debug,
)
from worldgen.climate import apply_climate_palette, climate_biome_map
from worldgen.contours import apply_mask_overlay, contour_mask
from worldgen.erosion import (
    hydraulic_erosion_frames,
    thermal_erosion,
    thermal_erosion_frames,
)
from worldgen.paths import astar_path
from worldgen.pipeline import practical_pipeline
from worldgen.tiles import tiles_zip_from_rgb
from worldgen.vegetation import filter_points_by_mask, jittered_points

st.set_page_config(
    page_title="Perlin Noise Map",
    page_icon="~",
    layout="wide",
)

inject_global_styles()


def _qp_get(name: str, default: str) -> str:
    try:
        raw = st.query_params.get(name)
    except Exception:
        raw = None

    if raw is None:
        return default
    if isinstance(raw, list):
        return str(raw[0]) if raw else default
    return str(raw)


def _qp_bool(name: str, default: bool) -> bool:
    raw = _qp_get(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "t", "yes", "y", "on"}


def _qp_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    try:
        v = int(float(_qp_get(name, str(default))))
    except ValueError:
        v = default
    return max(min_value, min(max_value, v))


def _qp_float(
    name: str, default: float, *, min_value: float, max_value: float
) -> float:
    try:
        v = float(_qp_get(name, str(default)))
    except ValueError:
        v = default
    return max(min_value, min(max_value, v))


def _set_query_params(params: dict[str, str]) -> None:
    try:
        st.query_params.clear()
        st.query_params.update(params)
    except Exception:
        st.experimental_set_query_params(**params)


def _nav_log(event: str, payload: dict[str, object] | None = None) -> None:
    if "nav_log" not in st.session_state:
        st.session_state["nav_log"] = []
    item = {
        "ts": time.time(),
        "event": str(event),
        "payload": dict(payload or {}),
    }
    st.session_state["nav_log"].append(item)
    st.session_state["nav_log"] = st.session_state["nav_log"][-60:]


def _freeze_params(params: dict[str, object]) -> tuple[tuple[str, str], ...]:
    items: list[tuple[str, str]] = []
    for k in sorted(params.keys()):
        items.append((str(k), repr(params[k])))
    return tuple(items)


def _snap_to_scale(x: float, *, scale: float) -> float:
    s = max(float(scale), 1e-9)
    return float(round(float(x) * s) / s)


def _chunk_cache_state() -> dict[str, Any]:
    if "chunk_cache" not in st.session_state:
        st.session_state["chunk_cache"] = {
            "order": [],
            "data": {},
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }
    return cast(dict[str, Any], st.session_state["chunk_cache"])


def _chunk_cache_get(key: object) -> object | None:
    stt = _chunk_cache_state()
    data = cast(dict[object, object], stt["data"])
    if key not in data:
        stt["misses"] = int(stt.get("misses", 0)) + 1
        return None
    stt["hits"] = int(stt.get("hits", 0)) + 1
    order = cast(list[object], stt["order"])
    try:
        order.remove(key)
    except ValueError:
        pass
    order.append(key)
    return data[key]


def _chunk_cache_put(key: object, value: object, *, max_items: int) -> None:
    stt = _chunk_cache_state()
    data = cast(dict[object, object], stt["data"])
    order = cast(list[object], stt["order"])

    if key in data:
        data[key] = value
        try:
            order.remove(key)
        except ValueError:
            pass
        order.append(key)
        return

    data[key] = value
    order.append(key)

    max_items = int(max_items)
    while max_items > 0 and len(order) > max_items:
        old = order.pop(0)
        data.pop(old, None)
        stt["evictions"] = int(stt.get("evictions", 0)) + 1


_COLOR_SCALES = ["Viridis", "Cividis", "Turbo", "IceFire", "Earth"]
_QUALITY = ["Fast", "Balanced", "Full"]
_NOISE_VARIANTS = {
    "fbm": "fBm",
    "turbulence": "Turbulence",
    "ridged": "Ridged",
    "domain_warp": "Domain warp",
}
_BASES = {
    "perlin": "Perlin (gradient)",
    "value": "Value noise",
}
_GRAD2_SETS = {
    "diag8": "8-dir (default)",
    "axis4": "Axis-only 4-dir",
    "circle16": "16-dir circle",
}

default_page = _qp_get("page", "Explore")
if default_page not in {"Explore", "Practical", "Learn"}:
    default_page = "Explore"

default_seed = _qp_int("seed", 0, min_value=0, max_value=2**31 - 1)
default_scale = _qp_float("scale", 120.0, min_value=5.0, max_value=600.0)
default_octaves = _qp_int("octaves", 4, min_value=1, max_value=10)
default_lacunarity = _qp_float("lacunarity", 2.0, min_value=1.0, max_value=4.0)
default_persistence = _qp_float("persistence", 0.5, min_value=0.0, max_value=1.0)
default_width = _qp_int("width", 256, min_value=64, max_value=1024)
default_height = _qp_int("height", 256, min_value=64, max_value=1024)
default_offset_x = _qp_float("offset_x", 0.0, min_value=-50.0, max_value=50.0)
default_offset_y = _qp_float("offset_y", 0.0, min_value=-50.0, max_value=50.0)
default_z_scale = _qp_float("z_scale", 80.0, min_value=0.0, max_value=200.0)
default_res3d = _qp_int("res3d", 128, min_value=64, max_value=256)

default_normalize = _qp_bool("normalize", True)
default_tileable = _qp_bool("tileable", False)
default_show_hist = _qp_bool("show_hist", False)
default_show_colorbar = _qp_bool("show_colorbar", False)
default_show3d = _qp_bool("show3d", True)
default_live_drag = _qp_bool("live_drag", True)
default_throttle_ms = _qp_int("throttle_ms", 50, min_value=10, max_value=250)

# Practical terrain defaults.
default_water_level = _qp_float("water_level", 0.45, min_value=0.0, max_value=1.0)
default_shore_width = _qp_float("shore_width", 0.05, min_value=0.0, max_value=0.25)
default_mountain_level = _qp_float("mountain_level", 0.75, min_value=0.0, max_value=1.0)
default_snowline = _qp_float("snowline", 0.83, min_value=0.0, max_value=1.0)
default_shade_az = _qp_float("shade_az", 315.0, min_value=0.0, max_value=360.0)
default_shade_alt = _qp_float("shade_alt", 45.0, min_value=0.0, max_value=90.0)
default_shade_strength = _qp_float("shade_strength", 0.55, min_value=0.0, max_value=1.0)
default_river_q = _qp_float("river_q", 0.985, min_value=0.9, max_value=0.999)
default_river_carve = _qp_bool("river_carve", True)
default_river_depth = _qp_float("river_depth", 0.06, min_value=0.0, max_value=0.2)
default_fill_lakes = _qp_bool("fill_lakes", True)
default_coast_smooth = _qp_bool("coast_smooth", True)
default_coast_radius = _qp_int("coast_radius", 2, min_value=0, max_value=12)
default_coast_strength = _qp_float("coast_strength", 0.6, min_value=0.0, max_value=1.0)
default_beach = _qp_bool("beach", True)
default_beach_amount = _qp_float("beach_amount", 0.02, min_value=0.0, max_value=0.1)
default_erosion = _qp_bool("erosion", False)
default_erosion_iter = _qp_int("erosion_iter", 30, min_value=0, max_value=250)
default_erosion_talus = _qp_float("erosion_talus", 0.02, min_value=0.0, max_value=0.1)
default_erosion_strength = _qp_float(
    "erosion_strength", 0.35, min_value=0.0, max_value=1.0
)
default_hydraulic = _qp_bool("hydraulic", False)
default_hyd_iter = _qp_int("hyd_iter", 40, min_value=0, max_value=250)
default_hyd_rain = _qp_float("hyd_rain", 0.01, min_value=0.0, max_value=0.05)
default_hyd_evap = _qp_float("hyd_evap", 0.5, min_value=0.0, max_value=1.0)
default_hyd_flow = _qp_float("hyd_flow", 0.5, min_value=0.0, max_value=1.0)
default_hyd_capacity = _qp_float("hyd_capacity", 4.0, min_value=0.0, max_value=10.0)
default_hyd_erosion = _qp_float("hyd_erosion", 0.3, min_value=0.0, max_value=1.0)
default_hyd_deposition = _qp_float("hyd_deposition", 0.3, min_value=0.0, max_value=1.0)
default_player_x = _qp_float("player_x", 0.0, min_value=-1e6, max_value=1e6)
default_player_y = _qp_float("player_y", 0.0, min_value=-1e6, max_value=1e6)
default_nav_step_px = _qp_float("nav_step_px", 64.0, min_value=1.0, max_value=512.0)
default_chunk_size_px = _qp_int("chunk_size_px", 256, min_value=32, max_value=2048)
default_show_chunk_grid = _qp_bool("show_chunk_grid", False)
default_chunk_cache_n = _qp_int("chunk_cache_n", 24, min_value=0, max_value=256)

default_backend = _qp_get("backend", "Reference")
if default_backend not in {"Reference", "Fast"}:
    default_backend = "Reference"
default_constraints = _qp_bool("constraints", False)
default_water_behavior = _qp_get("water_behavior", "Slow")
if default_water_behavior not in {"Slow", "Block"}:
    default_water_behavior = "Slow"
default_water_slow = _qp_float("water_slow", 0.35, min_value=0.05, max_value=1.0)
default_slope_block = _qp_float("slope_block", 0.75, min_value=0.0, max_value=1.0)
default_slope_cost = _qp_float("slope_cost", 2.0, min_value=0.0, max_value=10.0)

# Extras.
default_climate = _qp_bool("climate", False)
default_climate_scale = _qp_float(
    "climate_scale", 600.0, min_value=20.0, max_value=2000.0
)
default_climate_strength = _qp_float(
    "climate_strength", 0.85, min_value=0.0, max_value=1.0
)
default_veg = _qp_bool("veg", True)
default_veg_cell = _qp_int("veg_cell", 18, min_value=6, max_value=64)
default_veg_p = _qp_float("veg_p", 0.55, min_value=0.0, max_value=1.0)
default_rocks = _qp_bool("rocks", True)
default_trails = _qp_bool("trails", False)
default_trail_tx = _qp_float("trail_tx", 0.85, min_value=0.0, max_value=1.0)
default_trail_ty = _qp_float("trail_ty", 0.25, min_value=0.0, max_value=1.0)
default_cartographic = _qp_bool("carto", False)
default_contour_interval = _qp_float(
    "contour_interval", 0.05, min_value=0.01, max_value=0.2
)
default_contour_alpha = _qp_float("contour_alpha", 0.45, min_value=0.0, max_value=1.0)
default_tiles_grid = _qp_int("tiles_grid", 4, min_value=1, max_value=8)
default_tiles_size = _qp_int("tiles_size", 256, min_value=64, max_value=512)
default_tiles_z = _qp_int("tiles_z", 0, min_value=0, max_value=12)

default_colorscale = _qp_get("colorscale", "Viridis")
if default_colorscale not in _COLOR_SCALES:
    default_colorscale = "Viridis"

default_quality = _qp_get("quality", "Balanced")
if default_quality not in _QUALITY:
    default_quality = "Balanced"

_PRESETS: dict[str, dict[str, str]] = {
    "Terrain": {
        "basis": "perlin",
        "noise": "ridged",
        "scale": "180.0",
        "octaves": "6",
        "lacunarity": "2.0",
        "persistence": "0.5",
        "colorscale": "Earth",
        "shade": "Slope",
        "z_scale": "120.0",
    },
    "Clouds": {
        "basis": "perlin",
        "noise": "fbm",
        "scale": "320.0",
        "octaves": "5",
        "lacunarity": "2.0",
        "persistence": "0.6",
        "colorscale": "Cividis",
        "shade": "Height",
        "z_scale": "70.0",
    },
    "Marble": {
        "basis": "perlin",
        "noise": "turbulence",
        "scale": "90.0",
        "octaves": "6",
        "lacunarity": "2.0",
        "persistence": "0.55",
        "colorscale": "IceFire",
        "shade": "Height",
        "z_scale": "60.0",
    },
    "Islands": {
        "basis": "perlin",
        "noise": "domain_warp",
        "warp_amp": "1.8",
        "warp_scale": "2.0",
        "warp_octaves": "2",
        "scale": "260.0",
        "octaves": "5",
        "lacunarity": "2.0",
        "persistence": "0.5",
        "colorscale": "Earth",
        "shade": "Height",
        "z_scale": "85.0",
    },
    "Ridged mountains": {
        "basis": "perlin",
        "noise": "ridged",
        "scale": "140.0",
        "octaves": "7",
        "lacunarity": "2.0",
        "persistence": "0.45",
        "colorscale": "Earth",
        "shade": "Slope",
        "z_scale": "150.0",
    },
}

default_noise = _qp_get("noise", "fbm")
if default_noise not in _NOISE_VARIANTS:
    default_noise = "fbm"

default_basis = _qp_get("basis", "perlin")
if default_basis not in _BASES:
    default_basis = "perlin"

default_grad2 = _qp_get("grad2", "diag8")
if default_grad2 not in _GRAD2_SETS:
    default_grad2 = "diag8"

default_warp_amp = _qp_float("warp_amp", 1.25, min_value=0.0, max_value=10.0)
default_warp_scale = _qp_float("warp_scale", 1.5, min_value=0.05, max_value=10.0)
default_warp_octaves = _qp_int("warp_octaves", 2, min_value=1, max_value=8)

default_shade = _qp_get("shade", "Height")
if default_shade not in {"Height", "Slope"}:
    default_shade = "Height"


@st.cache_data(show_spinner=False)
def _noise_map(
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
) -> np.ndarray:
    return noise_map_2d(
        seed=int(seed),
        basis=str(basis),
        grad_set=str(grad_set),
        width=int(width),
        height=int(height),
        scale=float(scale),
        octaves=int(octaves),
        lacunarity=float(lacunarity),
        persistence=float(persistence),
        variant=str(variant),
        warp_amp=float(warp_amp),
        warp_scale=float(warp_scale),
        warp_octaves=int(warp_octaves),
        offset_x=float(offset_x),
        offset_y=float(offset_y),
        normalize=bool(normalize),
        tileable=bool(tileable),
    )


@st.cache_data(show_spinner=False)
def _practical_pipeline(
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
    backend: str,
) -> dict[str, np.ndarray | float | bool | None]:
    dtype = np.float32 if str(backend) == "Fast" else np.float64

    out = practical_pipeline(
        seed=int(seed),
        basis=str(basis),
        grad2=str(grad2),
        noise_variant=str(noise_variant),
        warp_amp=float(warp_amp),
        warp_scale=float(warp_scale),
        warp_octaves=int(warp_octaves),
        scale=float(scale),
        octaves=int(octaves),
        lacunarity=float(lacunarity),
        persistence=float(persistence),
        width=int(width),
        height=int(height),
        view_left=float(view_left),
        view_top=float(view_top),
        z_scale=float(z_scale),
        water_level=float(water_level),
        shore_width=float(shore_width),
        mountain_level=float(mountain_level),
        snowline=float(snowline),
        shade_az=float(shade_az),
        shade_alt=float(shade_alt),
        shade_strength=float(shade_strength),
        river_q=float(river_q),
        river_carve=bool(river_carve),
        river_depth=float(river_depth),
        fill_lakes=bool(fill_lakes),
        coast_smooth=bool(coast_smooth),
        coast_radius=int(coast_radius),
        coast_strength=float(coast_strength),
        beach=bool(beach),
        beach_amount=float(beach_amount),
        thermal_on=bool(thermal_on),
        thermal_iter=int(thermal_iter),
        thermal_talus=float(thermal_talus),
        thermal_strength=float(thermal_strength),
        hydraulic_on=bool(hydraulic_on),
        hyd_iter=int(hyd_iter),
        hyd_rain=float(hyd_rain),
        hyd_evap=float(hyd_evap),
        hyd_flow=float(hyd_flow),
        hyd_capacity=float(hyd_capacity),
        hyd_erosion=float(hyd_erosion),
        hyd_deposition=float(hyd_deposition),
        dtype=dtype,
    )

    return dict(out)


@st.cache_data(show_spinner=False)
def _climate_fields(
    *,
    seed: int,
    basis: str,
    grad2: str,
    width: int,
    height: int,
    climate_scale: float,
    view_left: float,
    view_top: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Temperature and moisture: use decorrelated fBm fields.
    zt = _noise_map(
        seed=int(seed) + 101,
        basis=str(basis),
        grad_set=str(grad2),
        width=int(width),
        height=int(height),
        scale=float(climate_scale),
        octaves=4,
        lacunarity=2.0,
        persistence=0.5,
        variant="fbm",
        warp_amp=0.0,
        warp_scale=1.0,
        warp_octaves=1,
        offset_x=float(view_left) + 19.1,
        offset_y=float(view_top) + 7.3,
        normalize=False,
        tileable=False,
    )
    zm = _noise_map(
        seed=int(seed) + 202,
        basis=str(basis),
        grad_set=str(grad2),
        width=int(width),
        height=int(height),
        scale=float(climate_scale),
        octaves=4,
        lacunarity=2.0,
        persistence=0.5,
        variant="fbm",
        warp_amp=0.0,
        warp_scale=1.0,
        warp_octaves=1,
        offset_x=float(view_left) - 11.8,
        offset_y=float(view_top) + 47.2,
        normalize=False,
        tileable=False,
    )

    temp01 = np.clip((np.asarray(zt, dtype=np.float64) + 1.0) * 0.5, 0.0, 1.0)
    moist01 = np.clip((np.asarray(zm, dtype=np.float64) + 1.0) * 0.5, 0.0, 1.0)
    return temp01, moist01


def _heatmap(
    z: np.ndarray,
    *,
    colorscale: str,
    show_colorbar: bool,
    height: int = 520,
) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            colorscale=colorscale,
            showscale=bool(show_colorbar),
            hovertemplate="value=%{z:.4f}<extra></extra>",
            colorbar=dict(thickness=12),
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=int(height),
    )
    fig.update_yaxes(
        autorange="reversed",
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )
    return fig


def _rgb_figure(
    rgb01: np.ndarray,
    *,
    marker_xy: tuple[float, float] | None = None,
    marker_label: str = "Player",
    height: int = 520,
) -> go.Figure:
    rgb = np.asarray(rgb01, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb01 must be HxWx3")

    img = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    fig = go.Figure(data=go.Image(z=img))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=int(height))
    fig.update_yaxes(autorange="reversed", showticklabels=False, showgrid=False)
    fig.update_xaxes(showticklabels=False, showgrid=False)

    if marker_xy is not None:
        x, y = marker_xy
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker=dict(
                    size=10,
                    color="rgba(255,255,255,0.95)",
                    line=dict(width=2, color="rgba(0,0,0,0.75)"),
                ),
                text=[marker_label],
                textposition="top center",
                textfont=dict(color="rgba(255,255,255,0.95)", size=12),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return fig


def _rgb_to_png_bytes(rgb01: np.ndarray) -> bytes:
    rgb = np.asarray(rgb01, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb01 must be HxWx3")
    img = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(img, mode="RGB").save(out, format="PNG")
    return out.getvalue()


def _rgb_overlay(
    rgb01: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[float, float, float],
    alpha: float,
) -> np.ndarray:
    rgb = np.asarray(rgb01, dtype=np.float64)
    m = np.asarray(mask).astype(bool)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb01 must be HxWx3")
    if m.shape != rgb.shape[:2]:
        raise ValueError("mask must be HxW")

    a = float(np.clip(float(alpha), 0.0, 1.0))
    c = np.array(color, dtype=np.float64)
    out = rgb.copy()
    out[m] = out[m] * (1.0 - a) + c * a
    return np.clip(out, 0.0, 1.0)


def _surface(
    z: np.ndarray,
    *,
    z_scale: float,
    colorscale: str,
    surfacecolor: np.ndarray | None = None,
) -> go.Figure:
    fig = go.Figure(
        data=go.Surface(
            z=z * z_scale,
            colorscale=colorscale,
            surfacecolor=surfacecolor,
            showscale=False,
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
    )
    return fig


def _slope01(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    dzdy, dzdx = np.gradient(z)
    s = np.sqrt(dzdx * dzdx + dzdy * dzdy)
    smin = float(np.min(s))
    smax = float(np.max(s))
    if smax == smin:
        return np.zeros_like(s)
    return (s - smin) / (smax - smin)


def _histogram(z: np.ndarray) -> go.Figure:
    fig = go.Figure(
        data=go.Histogram(
            x=z.reshape(-1),
            nbinsx=80,
            marker=dict(color="rgba(255,255,255,0.75)"),
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=260,
        title="Value distribution",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


header = st.container()
with header:
    left, right = st.columns([5, 2], vertical_alignment="center")
    with left:
        st.markdown(
            """
            <div class="pn-header">
              <div class="pn-title">Perlin Noise Map</div>
              <div class="pn-subtitle">
                Interactive Perlin/noise lab: explore, compare,
                and inspect intermediate steps.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if st.button("Randomize seed", use_container_width=True):
            st.query_params["seed"] = str(
                np.random.default_rng().integers(0, 2**31 - 1)
            )
            st.rerun()
        if st.button("Reset view", use_container_width=True):
            _set_query_params(
                {
                    "page": "Explore",
                    "basis": "perlin",
                    "grad2": "diag8",
                    "noise": "fbm",
                    "seed": "0",
                    "scale": "120.0",
                    "octaves": "4",
                    "lacunarity": "2.0",
                    "persistence": "0.5",
                    "width": "256",
                    "height": "256",
                    "offset_x": "0.0",
                    "offset_y": "0.0",
                    "z_scale": "80.0",
                    "res3d": "128",
                    "shade": "Height",
                    "normalize": "1",
                    "tileable": "0",
                    "colorscale": "Viridis",
                    "show_hist": "0",
                }
            )
            st.rerun()


with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Page",
        ["Explore", "Practical", "Learn"],
        index=(
            0 if default_page == "Explore" else 1 if default_page == "Practical" else 2
        ),
        label_visibility="collapsed",
    )

    st.divider()
    update_mode = st.radio(
        "Updates",
        ["Live", "Apply"],
        index=0,
        horizontal=True,
        help=(
            "Live: reruns when widgets change (fast iteration). "
            "Apply: batch updates via an Apply button (helps with heavier settings)."
        ),
    )

    default_params = {
        "page": str(page),
        "basis": str(default_basis),
        "grad2": str(default_grad2),
        "noise": str(default_noise),
        "warp_amp": float(default_warp_amp),
        "warp_scale": float(default_warp_scale),
        "warp_octaves": int(default_warp_octaves),
        "seed": int(default_seed),
        "scale": float(default_scale),
        "octaves": int(default_octaves),
        "lacunarity": float(default_lacunarity),
        "persistence": float(default_persistence),
        "width": int(default_width),
        "height": int(default_height),
        "offset_x": float(default_offset_x),
        "offset_y": float(default_offset_y),
        "z_scale": float(default_z_scale),
        "res3d": int(default_res3d),
        "shade": str(default_shade),
        "show3d": bool(default_show3d),
        "normalize": bool(default_normalize),
        "tileable": bool(default_tileable),
        "colorscale": str(default_colorscale),
        "show_colorbar": bool(default_show_colorbar),
        "show_hist": bool(default_show_hist),
    }

    if "applied_params" not in st.session_state:
        st.session_state["applied_params"] = dict(default_params)

    if "practical_params" not in st.session_state:
        st.session_state["practical_params"] = {
            "water_level": float(default_water_level),
            "shore_width": float(default_shore_width),
            "mountain_level": float(default_mountain_level),
            "snowline": float(default_snowline),
            "shade_az": float(default_shade_az),
            "shade_alt": float(default_shade_alt),
            "shade_strength": float(default_shade_strength),
            "river_q": float(default_river_q),
            "river_carve": bool(default_river_carve),
            "river_depth": float(default_river_depth),
            "fill_lakes": bool(default_fill_lakes),
            "coast_smooth": bool(default_coast_smooth),
            "coast_radius": int(default_coast_radius),
            "coast_strength": float(default_coast_strength),
            "beach": bool(default_beach),
            "beach_amount": float(default_beach_amount),
            "erosion": bool(default_erosion),
            "erosion_iter": int(default_erosion_iter),
            "erosion_talus": float(default_erosion_talus),
            "erosion_strength": float(default_erosion_strength),
            "hydraulic": bool(default_hydraulic),
            "hyd_iter": int(default_hyd_iter),
            "hyd_rain": float(default_hyd_rain),
            "hyd_evap": float(default_hyd_evap),
            "hyd_flow": float(default_hyd_flow),
            "hyd_capacity": float(default_hyd_capacity),
            "hyd_erosion": float(default_hyd_erosion),
            "hyd_deposition": float(default_hyd_deposition),
            "player_x": float(default_player_x),
            "player_y": float(default_player_y),
            "nav_step_px": float(default_nav_step_px),
            "chunk_size_px": int(default_chunk_size_px),
            "show_chunk_grid": bool(default_show_chunk_grid),
            "chunk_cache_n": int(default_chunk_cache_n),
            "backend": str(default_backend),
            "constraints": bool(default_constraints),
            "water_behavior": str(default_water_behavior),
            "water_slow": float(default_water_slow),
            "slope_block": float(default_slope_block),
            "slope_cost": float(default_slope_cost),
            "climate": bool(default_climate),
            "climate_scale": float(default_climate_scale),
            "climate_strength": float(default_climate_strength),
            "veg": bool(default_veg),
            "veg_cell": int(default_veg_cell),
            "veg_p": float(default_veg_p),
            "rocks": bool(default_rocks),
            "trails": bool(default_trails),
            "trail_tx": float(default_trail_tx),
            "trail_ty": float(default_trail_ty),
            "carto": bool(default_cartographic),
            "contour_interval": float(default_contour_interval),
            "contour_alpha": float(default_contour_alpha),
            "tiles_grid": int(default_tiles_grid),
            "tiles_size": int(default_tiles_size),
            "tiles_z": int(default_tiles_z),
        }

    if update_mode == "Apply":
        applied = dict(st.session_state["applied_params"])

        with st.form("controls_form", border=False):
            st.header("Parameters")
            basis = st.selectbox(
                "Basis",
                list(_BASES.keys()),
                index=list(_BASES.keys()).index(str(applied["basis"])),
                format_func=lambda k: _BASES[str(k)],
            )
            grad2 = str(applied["grad2"])
            if str(basis) == "perlin":
                grad2 = st.selectbox(
                    "Gradient set",
                    list(_GRAD2_SETS.keys()),
                    index=list(_GRAD2_SETS.keys()).index(str(applied["grad2"])),
                    format_func=lambda k: _GRAD2_SETS[str(k)],
                )

            noise_variant = st.selectbox(
                "Noise type",
                list(_NOISE_VARIANTS.keys()),
                index=list(_NOISE_VARIANTS.keys()).index(str(applied["noise"])),
                format_func=lambda k: _NOISE_VARIANTS[str(k)],
            )

            warp_amp = float(applied["warp_amp"])
            warp_scale = float(applied["warp_scale"])
            warp_octaves = int(applied["warp_octaves"])
            if str(noise_variant) == "domain_warp":
                st.subheader("Domain warp")
                warp_amp = st.slider(
                    "Warp amplitude",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(warp_amp),
                    step=0.05,
                )
                warp_scale = st.slider(
                    "Warp scale",
                    min_value=0.05,
                    max_value=10.0,
                    value=float(warp_scale),
                    step=0.05,
                )
                warp_octaves = st.slider(
                    "Warp octaves",
                    min_value=1,
                    max_value=8,
                    value=int(warp_octaves),
                    step=1,
                )

            seed = st.number_input(
                "Seed",
                min_value=0,
                max_value=2**31 - 1,
                value=int(applied["seed"]),
                step=1,
            )
            scale = st.slider(
                "Scale (bigger = smoother)",
                min_value=5.0,
                max_value=600.0,
                value=float(applied["scale"]),
            )
            octaves = st.slider(
                "Octaves",
                min_value=1,
                max_value=10,
                value=int(applied["octaves"]),
            )
            lacunarity = st.slider(
                "Lacunarity",
                min_value=1.0,
                max_value=4.0,
                value=float(applied["lacunarity"]),
                step=0.05,
            )
            persistence = st.slider(
                "Persistence",
                min_value=0.0,
                max_value=1.0,
                value=float(applied["persistence"]),
                step=0.01,
            )

            st.divider()
            st.subheader("Display")
            normalize = st.toggle("Normalize to 0..1", value=bool(applied["normalize"]))
            tileable = st.toggle(
                "Tileable (seamless edges)", value=bool(applied["tileable"])
            )
            colorscale = st.selectbox(
                "Colorscale",
                _COLOR_SCALES,
                index=_COLOR_SCALES.index(str(applied["colorscale"])),
            )
            show_colorbar = st.toggle(
                "Show colorbar",
                value=bool(applied.get("show_colorbar", default_show_colorbar)),
            )
            show_hist = st.toggle("Show histogram", value=bool(applied["show_hist"]))

            st.divider()
            st.subheader("Viewport")
            width = st.slider(
                "Width",
                min_value=64,
                max_value=1024,
                value=int(applied["width"]),
                step=64,
            )
            height = st.slider(
                "Height",
                min_value=64,
                max_value=1024,
                value=int(applied["height"]),
                step=64,
            )
            offset_x = st.slider(
                "Offset X",
                min_value=-50.0,
                max_value=50.0,
                value=float(applied["offset_x"]),
                step=0.5,
            )
            offset_y = st.slider(
                "Offset Y",
                min_value=-50.0,
                max_value=50.0,
                value=float(applied["offset_y"]),
                step=0.5,
            )

            st.divider()
            st.subheader("3D")
            show3d = st.toggle(
                "Enable 3D view",
                value=bool(applied.get("show3d", default_show3d)),
            )
            res3d = st.slider(
                "3D resolution",
                min_value=64,
                max_value=256,
                value=int(applied["res3d"]),
                step=32,
            )
            shade_mode = st.selectbox(
                "Shading",
                ["Height", "Slope"],
                index=0 if str(applied["shade"]) == "Height" else 1,
            )
            z_scale = st.slider(
                "Height scale",
                min_value=0.0,
                max_value=200.0,
                value=float(applied["z_scale"]),
                step=5.0,
            )

            applied_now = st.form_submit_button(
                "Apply changes",
                type="primary",
                use_container_width=True,
            )

        if applied_now:
            st.session_state["applied_params"] = {
                "page": str(page),
                "basis": str(basis),
                "grad2": str(grad2),
                "noise": str(noise_variant),
                "warp_amp": float(warp_amp),
                "warp_scale": float(warp_scale),
                "warp_octaves": int(warp_octaves),
                "seed": int(seed),
                "scale": float(scale),
                "octaves": int(octaves),
                "lacunarity": float(lacunarity),
                "persistence": float(persistence),
                "width": int(width),
                "height": int(height),
                "offset_x": float(offset_x),
                "offset_y": float(offset_y),
                "z_scale": float(z_scale),
                "res3d": int(res3d),
                "shade": str(shade_mode),
                "show3d": bool(show3d),
                "normalize": bool(normalize),
                "tileable": bool(tileable),
                "colorscale": str(colorscale),
                "show_colorbar": bool(show_colorbar),
                "show_hist": bool(show_hist),
            }

        params = dict(st.session_state["applied_params"])
        params["page"] = str(page)
        live_drag = False
        throttle_ms = 60
        quality = str(default_quality)
    else:
        st.header("Parameters")

        st.subheader("Live preview")
        live_drag = st.toggle(
            "Update while dragging",
            value=bool(default_live_drag),
            help="Uses a custom slider that emits values continuously.",
        )
        throttle_ms = st.slider(
            "Drag throttle (ms)",
            min_value=10,
            max_value=250,
            value=int(default_throttle_ms),
            step=10,
            help="Higher = fewer reruns while dragging.",
        )

        quality = st.selectbox(
            "Quality",
            _QUALITY,
            index=_QUALITY.index(default_quality),
            help="Uses lower resolution while dragging, then refines on release.",
        )

        st.divider()
        basis = st.selectbox(
            "Basis",
            list(_BASES.keys()),
            index=list(_BASES.keys()).index(default_basis),
            format_func=lambda k: _BASES[str(k)],
        )
        grad2 = str(default_grad2)
        if str(basis) == "perlin":
            grad2 = st.selectbox(
                "Gradient set",
                list(_GRAD2_SETS.keys()),
                index=list(_GRAD2_SETS.keys()).index(default_grad2),
                format_func=lambda k: _GRAD2_SETS[str(k)],
            )
        noise_variant = st.selectbox(
            "Noise type",
            list(_NOISE_VARIANTS.keys()),
            index=list(_NOISE_VARIANTS.keys()).index(default_noise),
            format_func=lambda k: _NOISE_VARIANTS[str(k)],
        )

        warp_amp = float(default_warp_amp)
        warp_scale = float(default_warp_scale)
        warp_octaves = int(default_warp_octaves)
        if str(noise_variant) == "domain_warp":
            st.subheader("Domain warp")
            warp_amp = st.slider(
                "Warp amplitude",
                min_value=0.0,
                max_value=10.0,
                value=float(default_warp_amp),
                step=0.05,
            )
            warp_scale = st.slider(
                "Warp scale",
                min_value=0.05,
                max_value=10.0,
                value=float(default_warp_scale),
                step=0.05,
            )
            warp_octaves = st.slider(
                "Warp octaves",
                min_value=1,
                max_value=8,
                value=int(default_warp_octaves),
                step=1,
            )
        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=2**31 - 1,
            value=int(default_seed),
            step=1,
        )
        scale_final = True
        if live_drag:
            scale, scale_final = live_slider(
                label="Scale (bigger = smoother)",
                min_value=5.0,
                max_value=600.0,
                value=float(default_scale),
                step=1.0,
                throttle_ms=int(throttle_ms),
                key="live_scale",
            )
        else:
            scale = st.slider(
                "Scale (bigger = smoother)",
                min_value=5.0,
                max_value=600.0,
                value=float(default_scale),
            )
        octaves = st.slider(
            "Octaves", min_value=1, max_value=10, value=int(default_octaves)
        )
        lacunarity = st.slider(
            "Lacunarity",
            min_value=1.0,
            max_value=4.0,
            value=float(default_lacunarity),
            step=0.05,
        )
        persistence = st.slider(
            "Persistence",
            min_value=0.0,
            max_value=1.0,
            value=float(default_persistence),
            step=0.01,
        )

        st.divider()
        st.subheader("Display")
        normalize = st.toggle("Normalize to 0..1", value=bool(default_normalize))
        tileable = st.toggle("Tileable (seamless edges)", value=bool(default_tileable))
        colorscale = st.selectbox(
            "Colorscale",
            _COLOR_SCALES,
            index=_COLOR_SCALES.index(default_colorscale),
        )
        show_colorbar = st.toggle(
            "Show colorbar",
            value=bool(default_show_colorbar),
        )
        show_hist = st.toggle("Show histogram", value=bool(default_show_hist))

        st.divider()
        st.subheader("Viewport")
        width = st.slider(
            "Width", min_value=64, max_value=1024, value=int(default_width), step=64
        )
        height = st.slider(
            "Height", min_value=64, max_value=1024, value=int(default_height), step=64
        )
        ox_final = True
        oy_final = True
        if live_drag:
            offset_x, ox_final = live_slider(
                label="Offset X",
                min_value=-50.0,
                max_value=50.0,
                value=float(default_offset_x),
                step=0.1,
                throttle_ms=int(throttle_ms),
                key="live_offset_x",
            )
            offset_y, oy_final = live_slider(
                label="Offset Y",
                min_value=-50.0,
                max_value=50.0,
                value=float(default_offset_y),
                step=0.1,
                throttle_ms=int(throttle_ms),
                key="live_offset_y",
            )
        else:
            offset_x = st.slider(
                "Offset X",
                min_value=-50.0,
                max_value=50.0,
                value=float(default_offset_x),
                step=0.5,
            )
            offset_y = st.slider(
                "Offset Y",
                min_value=-50.0,
                max_value=50.0,
                value=float(default_offset_y),
                step=0.5,
            )

        st.divider()
        st.subheader("3D")
        show3d = st.toggle(
            "Enable 3D view",
            value=bool(default_show3d),
        )
        res3d = st.slider(
            "3D resolution",
            min_value=64,
            max_value=256,
            value=int(default_res3d),
            step=32,
        )
        shade_mode = st.selectbox(
            "Shading",
            ["Height", "Slope"],
            index=0 if default_shade == "Height" else 1,
        )
        zs_final = True
        if live_drag:
            z_scale, zs_final = live_slider(
                label="Height scale",
                min_value=0.0,
                max_value=200.0,
                value=float(default_z_scale),
                step=1.0,
                throttle_ms=int(throttle_ms),
                key="live_z_scale",
            )
        else:
            z_scale = st.slider(
                "Height scale",
                min_value=0.0,
                max_value=200.0,
                value=float(default_z_scale),
                step=5.0,
            )

        is_dragging = bool(live_drag) and not (
            scale_final and ox_final and oy_final and zs_final
        )

        params = {
            "page": str(page),
            "basis": str(basis),
            "grad2": str(grad2),
            "noise": str(noise_variant),
            "live_drag": bool(live_drag),
            "throttle_ms": int(throttle_ms),
            "quality": str(quality),
            "is_dragging": bool(is_dragging),
            "warp_amp": float(warp_amp),
            "warp_scale": float(warp_scale),
            "warp_octaves": int(warp_octaves),
            "seed": int(seed),
            "scale": float(scale),
            "octaves": int(octaves),
            "lacunarity": float(lacunarity),
            "persistence": float(persistence),
            "width": int(width),
            "height": int(height),
            "offset_x": float(offset_x),
            "offset_y": float(offset_y),
            "z_scale": float(z_scale),
            "res3d": int(res3d),
            "shade": str(shade_mode),
            "show3d": bool(show3d),
            "normalize": bool(normalize),
            "tileable": bool(tileable),
            "colorscale": str(colorscale),
            "show_colorbar": bool(show_colorbar),
            "show_hist": bool(show_hist),
        }

    # Values used by the rest of the app.
    basis = str(params["basis"])
    grad2 = str(params["grad2"])
    noise_variant = str(params["noise"])
    warp_amp = float(params["warp_amp"])
    warp_scale = float(params["warp_scale"])
    warp_octaves = int(params["warp_octaves"])
    seed = int(params["seed"])
    scale = float(params["scale"])
    octaves = int(params["octaves"])
    lacunarity = float(params["lacunarity"])
    persistence = float(params["persistence"])
    width = int(params["width"])
    height = int(params["height"])
    offset_x = float(params["offset_x"])
    offset_y = float(params["offset_y"])
    z_scale = float(params["z_scale"])
    res3d = int(params["res3d"])
    shade_mode = str(params["shade"])
    normalize = bool(params["normalize"])
    tileable = bool(params["tileable"])
    colorscale = str(params["colorscale"])
    show_hist = bool(params["show_hist"])
    show_colorbar = bool(params.get("show_colorbar", default_show_colorbar))
    live_drag = bool(params.get("live_drag", False))
    throttle_ms = int(params.get("throttle_ms", 60))
    quality = str(params.get("quality", default_quality))
    if quality not in _QUALITY:
        quality = "Balanced"
    is_dragging = bool(params.get("is_dragging", False))
    show3d = bool(params.get("show3d", default_show3d))

    st.divider()
    st.subheader("Share")
    pp = dict(st.session_state.get("practical_params", {}))
    params_for_url = {
        "page": str(page),
        "basis": str(basis),
        "grad2": str(grad2),
        "noise": str(noise_variant),
        "live_drag": "1" if bool(live_drag) else "0",
        "throttle_ms": str(int(throttle_ms)),
        "quality": str(quality),
        "warp_amp": str(float(warp_amp)),
        "warp_scale": str(float(warp_scale)),
        "warp_octaves": str(int(warp_octaves)),
        "seed": str(int(seed)),
        "scale": str(float(scale)),
        "octaves": str(int(octaves)),
        "lacunarity": str(float(lacunarity)),
        "persistence": str(float(persistence)),
        "width": str(int(width)),
        "height": str(int(height)),
        "offset_x": str(float(offset_x)),
        "offset_y": str(float(offset_y)),
        "z_scale": str(float(z_scale)),
        "res3d": str(int(res3d)),
        "shade": str(shade_mode),
        "show3d": "1" if bool(show3d) else "0",
        "normalize": "1" if bool(normalize) else "0",
        "tileable": "1" if bool(tileable) else "0",
        "colorscale": str(colorscale),
        "show_colorbar": "1" if bool(show_colorbar) else "0",
        "show_hist": "1" if bool(show_hist) else "0",
        "water_level": str(float(pp.get("water_level", default_water_level))),
        "shore_width": str(float(pp.get("shore_width", default_shore_width))),
        "mountain_level": str(float(pp.get("mountain_level", default_mountain_level))),
        "snowline": str(float(pp.get("snowline", default_snowline))),
        "shade_az": str(float(pp.get("shade_az", default_shade_az))),
        "shade_alt": str(float(pp.get("shade_alt", default_shade_alt))),
        "shade_strength": str(float(pp.get("shade_strength", default_shade_strength))),
        "river_q": str(float(pp.get("river_q", default_river_q))),
        "river_carve": "1" if bool(pp.get("river_carve", default_river_carve)) else "0",
        "river_depth": str(float(pp.get("river_depth", default_river_depth))),
        "fill_lakes": "1" if bool(pp.get("fill_lakes", default_fill_lakes)) else "0",
        "coast_smooth": "1"
        if bool(pp.get("coast_smooth", default_coast_smooth))
        else "0",
        "coast_radius": str(int(pp.get("coast_radius", default_coast_radius))),
        "coast_strength": str(float(pp.get("coast_strength", default_coast_strength))),
        "beach": "1" if bool(pp.get("beach", default_beach)) else "0",
        "beach_amount": str(float(pp.get("beach_amount", default_beach_amount))),
        "erosion": "1" if bool(pp.get("erosion", default_erosion)) else "0",
        "erosion_iter": str(int(pp.get("erosion_iter", default_erosion_iter))),
        "erosion_talus": str(float(pp.get("erosion_talus", default_erosion_talus))),
        "erosion_strength": str(
            float(pp.get("erosion_strength", default_erosion_strength))
        ),
        "hydraulic": "1" if bool(pp.get("hydraulic", default_hydraulic)) else "0",
        "hyd_iter": str(int(pp.get("hyd_iter", default_hyd_iter))),
        "hyd_rain": str(float(pp.get("hyd_rain", default_hyd_rain))),
        "hyd_evap": str(float(pp.get("hyd_evap", default_hyd_evap))),
        "hyd_flow": str(float(pp.get("hyd_flow", default_hyd_flow))),
        "hyd_capacity": str(float(pp.get("hyd_capacity", default_hyd_capacity))),
        "hyd_erosion": str(float(pp.get("hyd_erosion", default_hyd_erosion))),
        "hyd_deposition": str(float(pp.get("hyd_deposition", default_hyd_deposition))),
        "player_x": str(float(pp.get("player_x", default_player_x))),
        "player_y": str(float(pp.get("player_y", default_player_y))),
        "nav_step_px": str(float(pp.get("nav_step_px", default_nav_step_px))),
        "chunk_size_px": str(int(pp.get("chunk_size_px", default_chunk_size_px))),
        "show_chunk_grid": "1"
        if bool(pp.get("show_chunk_grid", default_show_chunk_grid))
        else "0",
        "chunk_cache_n": str(int(pp.get("chunk_cache_n", default_chunk_cache_n))),
        "backend": str(pp.get("backend", default_backend)),
        "constraints": "1" if bool(pp.get("constraints", default_constraints)) else "0",
        "water_behavior": str(pp.get("water_behavior", default_water_behavior)),
        "water_slow": str(float(pp.get("water_slow", default_water_slow))),
        "slope_block": str(float(pp.get("slope_block", default_slope_block))),
        "slope_cost": str(float(pp.get("slope_cost", default_slope_cost))),
        "climate": "1" if bool(pp.get("climate", default_climate)) else "0",
        "climate_scale": str(float(pp.get("climate_scale", default_climate_scale))),
        "climate_strength": str(
            float(pp.get("climate_strength", default_climate_strength))
        ),
        "veg": "1" if bool(pp.get("veg", default_veg)) else "0",
        "veg_cell": str(int(pp.get("veg_cell", default_veg_cell))),
        "veg_p": str(float(pp.get("veg_p", default_veg_p))),
        "rocks": "1" if bool(pp.get("rocks", default_rocks)) else "0",
        "trails": "1" if bool(pp.get("trails", default_trails)) else "0",
        "trail_tx": str(float(pp.get("trail_tx", default_trail_tx))),
        "trail_ty": str(float(pp.get("trail_ty", default_trail_ty))),
        "carto": "1" if bool(pp.get("carto", default_cartographic)) else "0",
        "contour_interval": str(
            float(pp.get("contour_interval", default_contour_interval))
        ),
        "contour_alpha": str(float(pp.get("contour_alpha", default_contour_alpha))),
        "tiles_grid": str(int(pp.get("tiles_grid", default_tiles_grid))),
        "tiles_size": str(int(pp.get("tiles_size", default_tiles_size))),
        "tiles_z": str(int(pp.get("tiles_z", default_tiles_z))),
    }
    if st.button("Update URL with current settings"):
        _set_query_params(params_for_url)

    st.code(f"?{urlencode(params_for_url)}", language="text")

    st.divider()
    st.subheader("Presets")
    preset_name = st.selectbox(
        "Preset",
        list(_PRESETS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    if st.button("Apply preset", use_container_width=True):
        new_params = dict(params_for_url)
        new_params.update(_PRESETS[str(preset_name)])

        # If we're in Apply mode, reset the stored applied params so the preset
        # becomes the new baseline on rerun.
        if update_mode == "Apply":
            st.session_state.pop("applied_params", None)

        _set_query_params(new_params)
        st.rerun()

    st.divider()
    st.subheader("Snapshots")
    st.caption("Save/restore parameter sets for this session.")

    if "snapshots" not in st.session_state:
        st.session_state["snapshots"] = []

    snap_name = st.text_input("Name", value="", placeholder="e.g. My terrain")
    if st.button("Save snapshot", use_container_width=True):
        name = snap_name.strip() or f"Snapshot {len(st.session_state['snapshots']) + 1}"
        st.session_state["snapshots"].append(
            {"name": name, "params": dict(params_for_url)}
        )
        st.rerun()

    if st.session_state["snapshots"]:
        snap_idx = st.selectbox(
            "",
            list(range(len(st.session_state["snapshots"]))),
            format_func=lambda i: st.session_state["snapshots"][int(i)]["name"],
            label_visibility="collapsed",
        )
        cols = st.columns(2)
        with cols[0]:
            if st.button("Load", use_container_width=True):
                if update_mode == "Apply":
                    st.session_state.pop("applied_params", None)
                _set_query_params(
                    st.session_state["snapshots"][int(snap_idx)]["params"]
                )
                st.rerun()
        with cols[1]:
            if st.button("Delete", use_container_width=True):
                del st.session_state["snapshots"][int(snap_idx)]
                st.rerun()

hk = hotkeys(
    enabled=True,
    allowed=["r", "0", "h", "c", "t", "3", "p", "l", "w", "a", "s", "d"],
    key="hotkeys",
)

if isinstance(hk, dict) and "ts" in hk:
    ts = int(hk.get("ts", 0))
    if ts and ts != int(st.session_state.get("hotkey_ts", 0)):
        st.session_state["hotkey_ts"] = ts
        k = str(hk.get("key", ""))

        new = dict(params_for_url)

        if k == "r":
            new["seed"] = str(np.random.default_rng().integers(0, 2**31 - 1))
        elif k == "0":
            new = {
                "page": "Explore",
                "basis": "perlin",
                "grad2": "diag8",
                "noise": "fbm",
                "warp_amp": "1.25",
                "warp_scale": "1.5",
                "warp_octaves": "2",
                "seed": "0",
                "scale": "120.0",
                "octaves": "4",
                "lacunarity": "2.0",
                "persistence": "0.5",
                "width": "256",
                "height": "256",
                "offset_x": "0.0",
                "offset_y": "0.0",
                "z_scale": "80.0",
                "res3d": "128",
                "shade": "Height",
                "show3d": "1",
                "normalize": "1",
                "tileable": "0",
                "colorscale": "Viridis",
                "show_colorbar": "0",
                "show_hist": "0",
                "live_drag": "1",
                "throttle_ms": "50",
                "quality": "Balanced",
            }
        elif k == "h":
            new["show_hist"] = "0" if bool(show_hist) else "1"
        elif k == "c":
            new["show_colorbar"] = "0" if bool(show_colorbar) else "1"
        elif k == "t":
            new["tileable"] = "0" if bool(tileable) else "1"
        elif k == "3":
            new["show3d"] = "0" if bool(show3d) else "1"
        elif k == "p":
            if str(page) == "Explore":
                new["page"] = "Practical"
            elif str(page) == "Practical":
                new["page"] = "Learn"
            else:
                new["page"] = "Explore"
        elif k in {"w", "a", "s", "d"}:
            if str(page) == "Practical":
                pp = dict(st.session_state.get("practical_params", {}))
                step_px = float(pp.get("nav_step_px", default_nav_step_px))

                step = step_px / max(float(scale), 1e-9)
                move_dx = 0.0
                move_dy = 0.0
                if k == "w":
                    move_dy -= 1.0
                elif k == "s":
                    move_dy += 1.0
                elif k == "a":
                    move_dx -= 1.0
                elif k == "d":
                    move_dx += 1.0

                factor = 1.0
                blocked = False
                if bool(pp.get("constraints", default_constraints)):
                    last_h = st.session_state.get("practical_last_height")
                    last_s = st.session_state.get("practical_last_slope")
                    if isinstance(last_h, np.ndarray) and isinstance(
                        last_s, np.ndarray
                    ):
                        Hh, Ww = last_h.shape
                        cxp = int((Ww - 1) // 2)
                        cyp = int((Hh - 1) // 2)
                        tx = int(cxp + round(move_dx * float(step_px)))
                        ty = int(cyp + round(move_dy * float(step_px)))
                        if 0 <= tx < int(Ww) and 0 <= ty < int(Hh):
                            h = float(last_h[ty, tx])
                            sl = float(last_s[ty, tx])
                            water = h < float(
                                pp.get("water_level", default_water_level)
                            )
                            if sl >= float(pp.get("slope_block", default_slope_block)):
                                blocked = True
                            elif (
                                water
                                and str(
                                    pp.get("water_behavior", default_water_behavior)
                                )
                                == "Block"
                            ):
                                blocked = True
                            else:
                                sc = float(pp.get("slope_cost", default_slope_cost))
                                if sc > 0.0:
                                    factor /= 1.0 + sc * sl
                                if (
                                    water
                                    and str(
                                        pp.get("water_behavior", default_water_behavior)
                                    )
                                    == "Slow"
                                ):
                                    factor *= float(
                                        pp.get("water_slow", default_water_slow)
                                    )

                px = float(pp.get("player_x", default_player_x))
                py = float(pp.get("player_y", default_player_y))
                if not blocked:
                    px += move_dx * step * factor
                    py += move_dy * step * factor

                pp["player_x"] = float(px)
                pp["player_y"] = float(py)
                st.session_state["practical_params"] = pp

                view_left = float(px) - (float(width) / (2.0 * max(float(scale), 1e-9)))
                view_top = float(py) - (float(height) / (2.0 * max(float(scale), 1e-9)))
                new["player_x"] = str(float(px))
                new["player_y"] = str(float(py))
                new["offset_x"] = str(float(view_left))
                new["offset_y"] = str(float(view_top))
        elif k == "l":
            new["live_drag"] = "0" if bool(live_drag) else "1"

        if update_mode == "Apply":
            st.session_state.pop("applied_params", None)

        _set_query_params(new)
        st.rerun()


width_render = int(width)
height_render = int(height)
res3d_render = int(res3d)
rendering_preview = False

if bool(is_dragging) and str(quality) != "Full":
    rendering_preview = True
    max2 = 256 if str(quality) == "Fast" else 512
    width_render = min(width_render, max2)
    height_render = min(height_render, max2)

    max3 = 96 if str(quality) == "Fast" else 160
    res3d_render = min(res3d_render, max3)

t0 = time.perf_counter()
z01 = _noise_map(
    seed=int(seed),
    basis=str(basis),
    grad_set=str(grad2),
    width=int(width_render),
    height=int(height_render),
    scale=float(scale),
    octaves=int(octaves),
    lacunarity=float(lacunarity),
    persistence=float(persistence),
    variant=str(noise_variant),
    warp_amp=float(warp_amp),
    warp_scale=float(warp_scale),
    warp_octaves=int(warp_octaves),
    offset_x=float(offset_x),
    offset_y=float(offset_y),
    normalize=bool(normalize),
    tileable=bool(tileable),
)
t1 = time.perf_counter()
st.session_state["perf_2d_ms"] = (t1 - t0) * 1000.0
st.session_state["perf_2d_res"] = (int(width_render), int(height_render))

zmin = float(np.min(z01))
zmax = float(np.max(z01))


if page == "Explore":
    tab2d, tab3d = st.tabs(["2D Map", "3D Terrain"])

    with tab2d:
        st.subheader("2D Noise Map")
        zmean = float(np.mean(z01))
        zstd = float(np.std(z01))
        c0, c1, c2, c3 = st.columns(4)
        c0.metric("min", f"{zmin:.4f}")
        c1.metric("max", f"{zmax:.4f}")
        c2.metric("mean", f"{zmean:.4f}")
        c3.metric("std", f"{zstd:.4f}")

        nav, mini = st.columns([2, 3], vertical_alignment="top")
        with nav:
            st.markdown("**Viewport**")
            pan_step = st.number_input(
                "Pan step",
                min_value=0.1,
                max_value=20.0,
                value=1.0,
                step=0.1,
                key="pan_step",
            )
            r0, r1, r2, r3 = st.columns(4)
            if r0.button("Left", use_container_width=True, key="pan_left"):
                params_for_url["offset_x"] = str(float(offset_x) - float(pan_step))
                if update_mode == "Apply":
                    st.session_state.pop("applied_params", None)
                _set_query_params(params_for_url)
                st.rerun()
            if r1.button("Right", use_container_width=True, key="pan_right"):
                params_for_url["offset_x"] = str(float(offset_x) + float(pan_step))
                if update_mode == "Apply":
                    st.session_state.pop("applied_params", None)
                _set_query_params(params_for_url)
                st.rerun()
            if r2.button("Up", use_container_width=True, key="pan_up"):
                params_for_url["offset_y"] = str(float(offset_y) - float(pan_step))
                if update_mode == "Apply":
                    st.session_state.pop("applied_params", None)
                _set_query_params(params_for_url)
                st.rerun()
            if r3.button("Down", use_container_width=True, key="pan_down"):
                params_for_url["offset_y"] = str(float(offset_y) + float(pan_step))
                if update_mode == "Apply":
                    st.session_state.pop("applied_params", None)
                _set_query_params(params_for_url)
                st.rerun()

        with mini:
            st.markdown("**Minimap**")
            zmini = _noise_map(
                seed=int(seed),
                basis=str(basis),
                grad_set=str(grad2),
                width=160,
                height=160,
                scale=float(scale),
                octaves=int(octaves),
                lacunarity=float(lacunarity),
                persistence=float(persistence),
                variant=str(noise_variant),
                warp_amp=float(warp_amp),
                warp_scale=float(warp_scale),
                warp_octaves=int(warp_octaves),
                offset_x=float(offset_x),
                offset_y=float(offset_y),
                normalize=True,
                tileable=bool(tileable),
            )
            st.plotly_chart(
                _heatmap(zmini, colorscale=str(colorscale), show_colorbar=False),
                width="stretch",
                key="minimap",
            )
        st.plotly_chart(
            _heatmap(z01, colorscale=str(colorscale), show_colorbar=show_colorbar),
            width="stretch",
            key="explore_heatmap",
        )

        with st.expander("Value probe"):
            if rendering_preview:
                st.caption(
                    (
                        f"Preview mode: probing {z01.shape[1]}x{z01.shape[0]} "
                        "(release to refine)."
                    )
                )

            max_x = max(int(z01.shape[1]) - 1, 0)
            max_y = max(int(z01.shape[0]) - 1, 0)
            ix = st.slider("x index", 0, max_x, 0, key="probe_x")
            iy = st.slider("y index", 0, max_y, 0, key="probe_y")
            v = float(z01[int(iy), int(ix)])
            st.metric("value", f"{v:.6f}")

        with st.expander("Cross-section"):
            mode = st.radio(
                "Cross-section mode",
                ["Row", "Column"],
                horizontal=True,
                label_visibility="collapsed",
                key="xsec_mode",
            )
            if mode == "Row":
                max_i = max(int(z01.shape[0]) - 1, 0)
                idx = st.slider("row index", 0, max_i, min(0, max_i), key="xsec_row")
                y = z01[int(idx), :]
                x = np.arange(y.shape[0])
                title = f"Row {int(idx)}"
            else:
                max_i = max(int(z01.shape[1]) - 1, 0)
                idx = st.slider("column index", 0, max_i, min(0, max_i), key="xsec_col")
                y = z01[:, int(idx)]
                x = np.arange(y.shape[0])
                title = f"Column {int(idx)}"

            fig = go.Figure(
                data=go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0.85)", width=2),
                    showlegend=False,
                )
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                height=260,
                title=title,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, width="stretch", key="xsec_plot")

        with st.expander("Diff mode"):
            if rendering_preview:
                st.info("Release the slider to run diff comparisons.")
            else:
                options: list[tuple[str, str]] = []
                for name in _PRESETS.keys():
                    options.append((f"preset:{name}", f"Preset: {name}"))
                for i, s in enumerate(st.session_state.get("snapshots", [])):
                    options.append((f"snap:{i}", f"Snapshot: {s['name']}"))

                if not options:
                    st.caption("Create a snapshot or use presets to compare.")
                else:
                    choice = st.selectbox(
                        "Compare against",
                        options,
                        format_func=lambda t: t[1],
                        key="diff_choice",
                    )

                    key = choice[0]
                    label = choice[1]

                    other = dict(params_for_url)
                    if str(key).startswith("preset:"):
                        preset = str(key).split(":", 1)[1]
                        other.update(_PRESETS[preset])
                    elif str(key).startswith("snap:"):
                        idx = int(str(key).split(":", 1)[1])
                        other.update(st.session_state["snapshots"][idx]["params"])

                    def _to_int(k: str, fallback: int) -> int:
                        try:
                            return int(float(other.get(k, str(fallback))))
                        except ValueError:
                            return int(fallback)

                    def _to_float(k: str, fallback: float) -> float:
                        try:
                            return float(other.get(k, str(fallback)))
                        except ValueError:
                            return float(fallback)

                    z_other = _noise_map(
                        seed=_to_int("seed", seed),
                        basis=str(other.get("basis", basis)),
                        grad_set=str(other.get("grad2", grad2)),
                        width=int(width_render),
                        height=int(height_render),
                        scale=_to_float("scale", scale),
                        octaves=_to_int("octaves", octaves),
                        lacunarity=_to_float("lacunarity", lacunarity),
                        persistence=_to_float("persistence", persistence),
                        variant=str(other.get("noise", noise_variant)),
                        warp_amp=_to_float("warp_amp", warp_amp),
                        warp_scale=_to_float("warp_scale", warp_scale),
                        warp_octaves=_to_int("warp_octaves", warp_octaves),
                        offset_x=_to_float("offset_x", offset_x),
                        offset_y=_to_float("offset_y", offset_y),
                        normalize=bool(normalize),
                        tileable=bool(tileable),
                    )

                    diff = z01 - z_other
                    c0, c1, c2 = st.columns(3)
                    with c0:
                        st.markdown("**Current**")
                        st.plotly_chart(
                            _heatmap(
                                z01,
                                colorscale=str(colorscale),
                                show_colorbar=False,
                            ),
                            width="stretch",
                            key="diff_current",
                        )
                    with c1:
                        st.markdown(f"**{label}**")
                        st.plotly_chart(
                            _heatmap(
                                z_other,
                                colorscale=str(colorscale),
                                show_colorbar=False,
                            ),
                            width="stretch",
                            key="diff_other",
                        )
                    with c2:
                        st.markdown("**Difference**")
                        st.plotly_chart(
                            _heatmap(
                                diff,
                                colorscale="IceFire",
                                show_colorbar=False,
                            ),
                            width="stretch",
                            key="diff_delta",
                        )
        if show_hist:
            st.plotly_chart(_histogram(z01), width="stretch", key="explore_hist")

        with st.expander("Compare: Perlin vs Value noise"):
            if rendering_preview:
                st.info("Release the slider to compute comparisons.")
            else:
                perlin_z = _noise_map(
                    seed=int(seed),
                    basis="perlin",
                    grad_set=str(grad2),
                    width=int(width_render),
                    height=int(height_render),
                    scale=float(scale),
                    octaves=int(octaves),
                    lacunarity=float(lacunarity),
                    persistence=float(persistence),
                    variant=str(noise_variant),
                    warp_amp=float(warp_amp),
                    warp_scale=float(warp_scale),
                    warp_octaves=int(warp_octaves),
                    offset_x=float(offset_x),
                    offset_y=float(offset_y),
                    normalize=bool(normalize),
                    tileable=bool(tileable),
                )
                value_z = _noise_map(
                    seed=int(seed),
                    basis="value",
                    grad_set=str(grad2),
                    width=int(width_render),
                    height=int(height_render),
                    scale=float(scale),
                    octaves=int(octaves),
                    lacunarity=float(lacunarity),
                    persistence=float(persistence),
                    variant=str(noise_variant),
                    warp_amp=float(warp_amp),
                    warp_scale=float(warp_scale),
                    warp_octaves=int(warp_octaves),
                    offset_x=float(offset_x),
                    offset_y=float(offset_y),
                    normalize=bool(normalize),
                    tileable=bool(tileable),
                )

                col0, col1 = st.columns(2)
                with col0:
                    st.markdown("**Perlin (gradient)**")
                    st.plotly_chart(
                        _heatmap(
                            perlin_z,
                            colorscale=str(colorscale),
                            show_colorbar=False,
                        ),
                        width="stretch",
                        key="compare_perlin_heatmap",
                    )
                with col1:
                    st.markdown("**Value noise**")
                    st.plotly_chart(
                        _heatmap(
                            value_z,
                            colorscale=str(colorscale),
                            show_colorbar=False,
                        ),
                        width="stretch",
                        key="compare_value_heatmap",
                    )

        if str(basis) == "perlin":
            with st.expander("Artifacts: different gradient sets"):
                st.write(
                    "Axis-only gradients tend to emphasize grid-aligned artifacts. "
                    "Compare them to the default 8-direction set."
                )

                diag8_z = _noise_map(
                    seed=int(seed),
                    basis="perlin",
                    grad_set="diag8",
                    width=int(width_render),
                    height=int(height_render),
                    scale=float(scale),
                    octaves=int(octaves),
                    lacunarity=float(lacunarity),
                    persistence=float(persistence),
                    variant=str(noise_variant),
                    warp_amp=float(warp_amp),
                    warp_scale=float(warp_scale),
                    warp_octaves=int(warp_octaves),
                    offset_x=float(offset_x),
                    offset_y=float(offset_y),
                    normalize=bool(normalize),
                    tileable=bool(tileable),
                )
                axis4_z = _noise_map(
                    seed=int(seed),
                    basis="perlin",
                    grad_set="axis4",
                    width=int(width_render),
                    height=int(height_render),
                    scale=float(scale),
                    octaves=int(octaves),
                    lacunarity=float(lacunarity),
                    persistence=float(persistence),
                    variant=str(noise_variant),
                    warp_amp=float(warp_amp),
                    warp_scale=float(warp_scale),
                    warp_octaves=int(warp_octaves),
                    offset_x=float(offset_x),
                    offset_y=float(offset_y),
                    normalize=bool(normalize),
                    tileable=bool(tileable),
                )

                col0, col1 = st.columns(2)
                with col0:
                    st.markdown("**diag8**")
                    st.plotly_chart(
                        _heatmap(
                            diag8_z,
                            colorscale=str(colorscale),
                            show_colorbar=False,
                        ),
                        width="stretch",
                        key="gradset_diag8_heatmap",
                    )
                with col1:
                    st.markdown("**axis4**")
                    st.plotly_chart(
                        _heatmap(
                            axis4_z,
                            colorscale=str(colorscale),
                            show_colorbar=False,
                        ),
                        width="stretch",
                        key="gradset_axis4_heatmap",
                    )

        if str(noise_variant) == "domain_warp":
            with st.expander("What is domain warping?"):
                st.write(
                    "Domain warping perturbs the input coordinates (x, y) with another "
                    "noise field, then samples the base noise at the "
                    "warped coordinates."
                )

                base_z = _noise_map(
                    seed=int(seed),
                    basis="perlin",
                    grad_set=str(grad2),
                    width=int(width_render),
                    height=int(height_render),
                    scale=float(scale),
                    octaves=int(octaves),
                    lacunarity=float(lacunarity),
                    persistence=float(persistence),
                    variant="fbm",
                    warp_amp=float(warp_amp),
                    warp_scale=float(warp_scale),
                    warp_octaves=int(warp_octaves),
                    offset_x=float(offset_x),
                    offset_y=float(offset_y),
                    normalize=bool(normalize),
                    tileable=bool(tileable),
                )

                col0, col1 = st.columns(2)
                with col0:
                    st.markdown("**Base (fBm)**")
                    st.plotly_chart(
                        _heatmap(
                            base_z, colorscale=str(colorscale), show_colorbar=False
                        ),
                        width="stretch",
                        key="domainwarp_base_heatmap",
                    )
                with col1:
                    st.markdown("**Warped**")
                    st.plotly_chart(
                        _heatmap(z01, colorscale=str(colorscale), show_colorbar=False),
                        width="stretch",
                        key="domainwarp_warped_heatmap",
                    )

        with st.expander("Export"):
            params = {
                "page": str(page),
                "basis": str(basis),
                "grad2": str(grad2),
                "noise": str(noise_variant),
                "warp_amp": float(warp_amp),
                "warp_scale": float(warp_scale),
                "warp_octaves": int(warp_octaves),
                "seed": int(seed),
                "scale": float(scale),
                "octaves": int(octaves),
                "lacunarity": float(lacunarity),
                "persistence": float(persistence),
                "width": int(width),
                "height": int(height),
                "offset_x": float(offset_x),
                "offset_y": float(offset_y),
                "normalize": bool(normalize),
                "tileable": bool(tileable),
                "colorscale": str(colorscale),
                "show_colorbar": bool(show_colorbar),
                "show_hist": bool(show_hist),
                "res3d": int(res3d),
                "shade": str(shade_mode),
                "z_scale": float(z_scale),
            }

            base_name = (
                st.text_input(
                    "Base filename",
                    value="perlin_map",
                    help="Used for exported files.",
                ).strip()
                or "perlin_map"
            )
            export_kind = st.selectbox(
                "Data export",
                ["PNG (grayscale)", "NumPy (.npy)"],
                index=0,
            )

            if export_kind == "PNG (grayscale)":
                png = array_to_png_bytes(z01)
                st.image(png, caption="Preview (grayscale)")
                st.download_button(
                    "Download PNG",
                    data=png,
                    file_name=f"{base_name}.png",
                    mime="image/png",
                )
            else:
                st.download_button(
                    "Download .npy",
                    data=array_to_npy_bytes(z01),
                    file_name=f"{base_name}.npy",
                    mime="application/octet-stream",
                )

            st.download_button(
                "Download params.json",
                data=json.dumps(params, indent=2, sort_keys=True),
                file_name=f"{base_name}.params.json",
                mime="application/json",
            )
            st.caption(
                "To share this exact state: click 'Update URL with current settings' "
                "in the sidebar, then copy the browser URL."
            )

    with tab3d:
        if not show3d:
            st.subheader("3D Heightmap")
            st.info("3D is disabled (toggle 'Enable 3D view' in the sidebar).")
        else:
            st.subheader("3D Heightmap")
            if rendering_preview:
                st.caption(
                    f"resolution={int(res3d_render)}x{int(res3d_render)} (preview)"
                )
            else:
                st.caption(f"resolution={int(res3d)}x{int(res3d)}")

            t3a = time.perf_counter()
            z3d = _noise_map(
                seed=int(seed),
                basis=str(basis),
                grad_set=str(grad2),
                width=int(res3d_render),
                height=int(res3d_render),
                scale=float(scale),
                octaves=int(octaves),
                lacunarity=float(lacunarity),
                persistence=float(persistence),
                variant=str(noise_variant),
                warp_amp=float(warp_amp),
                warp_scale=float(warp_scale),
                warp_octaves=int(warp_octaves),
                offset_x=float(offset_x),
                offset_y=float(offset_y),
                normalize=bool(normalize),
                tileable=bool(tileable),
            )
            t3b = time.perf_counter()
            st.session_state["perf_3d_ms"] = (t3b - t3a) * 1000.0
            st.session_state["perf_3d_res"] = (int(res3d_render), int(res3d_render))

            surfacecolor = _slope01(z3d) if shade_mode == "Slope" else None
            z3_min = float(np.min(z3d))
            z3_max = float(np.max(z3d))
            z3_mean = float(np.mean(z3d))
            z3_std = float(np.std(z3d))
            c0, c1, c2, c3 = st.columns(4)
            c0.metric("min", f"{z3_min:.4f}")
            c1.metric("max", f"{z3_max:.4f}")
            c2.metric("mean", f"{z3_mean:.4f}")
            c3.metric("std", f"{z3_std:.4f}")
            st.plotly_chart(
                _surface(
                    z3d,
                    z_scale=float(z_scale),
                    colorscale=str(colorscale),
                    surfacecolor=surfacecolor,
                ),
                width="stretch",
                key="explore_surface",
            )

            with st.expander("Export 3D"):
                st.download_button(
                    "Download mesh (OBJ)",
                    data=heightmap_to_obj_bytes(z3d, z_scale=float(z_scale)),
                    file_name="terrain.obj",
                    mime="text/plain",
                )
                st.download_button(
                    "Download heightmap (.npy)",
                    data=array_to_npy_bytes(z3d),
                    file_name="heightmap.npy",
                    mime="application/octet-stream",
                )
elif page == "Practical":
    st.subheader("Practical: Terrain Generator")
    st.caption(
        "A learning-first worldgen sandbox: height -> biomes, rivers, and weathering. "
        "Use WASD (or the buttons) to move the player marker across an endless map."
    )

    pp = dict(st.session_state.get("practical_params", {}))
    if "practical_player_init" not in st.session_state:
        # If the user arrives here from Explore with a non-zero viewport, center
        # the player on the current view (unless player coords were explicitly set).
        if math.isclose(
            float(pp.get("player_x", 0.0)), float(default_player_x)
        ) and math.isclose(float(pp.get("player_y", 0.0)), float(default_player_y)):
            px0 = float(offset_x) + (
                float(width_render) / (2.0 * max(float(scale), 1e-9))
            )
            py0 = float(offset_y) + (
                float(height_render) / (2.0 * max(float(scale), 1e-9))
            )
            pp["player_x"] = float(px0)
            pp["player_y"] = float(py0)
            st.session_state["practical_params"] = pp
        st.session_state["practical_player_init"] = True

    left_col, right_col = st.columns([0.95, 1.55], gap="large")

    with left_col:
        if "practical_debug" not in st.session_state:
            st.session_state["practical_debug"] = False
        debug_mode = st.toggle("Debug mode", key="practical_debug")

        st.markdown("**Controls**")

        with st.form("practical_settings", border=False):
            tabs = st.tabs(
                ["Terrain", "Hydrology", "Weathering", "Navigation", "Extras"]
            )
            with tabs[0]:
                st.markdown("**Terrain + lighting**")
                water_level = st.slider(
                    "Water level",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("water_level", default_water_level)),
                    step=0.01,
                )
                shore_width = st.slider(
                    "Shore width",
                    min_value=0.0,
                    max_value=0.25,
                    value=float(pp.get("shore_width", default_shore_width)),
                    step=0.01,
                )
                mountain_level = st.slider(
                    "Mountain level",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("mountain_level", default_mountain_level)),
                    step=0.01,
                )
                snowline = st.slider(
                    "Snowline",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("snowline", default_snowline)),
                    step=0.01,
                )
                st.divider()
                shade_az = st.slider(
                    "Light azimuth (deg)",
                    min_value=0.0,
                    max_value=360.0,
                    value=float(pp.get("shade_az", default_shade_az)),
                    step=1.0,
                )
                shade_alt = st.slider(
                    "Light altitude (deg)",
                    min_value=0.0,
                    max_value=90.0,
                    value=float(pp.get("shade_alt", default_shade_alt)),
                    step=1.0,
                )
                shade_strength = st.slider(
                    "Shade strength",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("shade_strength", default_shade_strength)),
                    step=0.05,
                )

            with tabs[1]:
                st.markdown("**Rivers + lakes**")
                river_q = st.slider(
                    "River density (quantile)",
                    min_value=0.9,
                    max_value=0.999,
                    value=float(pp.get("river_q", default_river_q)),
                    step=0.001,
                    help=(
                        "Higher = fewer rivers. "
                        "Uses flow accumulation quantiles for a stable knob."
                    ),
                )
                river_carve = st.toggle(
                    "Carve rivers",
                    value=bool(pp.get("river_carve", default_river_carve)),
                )
                river_depth = st.slider(
                    "River carve depth",
                    min_value=0.0,
                    max_value=0.2,
                    value=float(pp.get("river_depth", default_river_depth)),
                    step=0.01,
                )
                fill_lakes = st.toggle(
                    "Fill depressions (lakes)",
                    value=bool(pp.get("fill_lakes", default_fill_lakes)),
                )

                st.divider()
                st.markdown("**Coast**")
                coast_smooth = st.toggle(
                    "Smooth coastline",
                    value=bool(pp.get("coast_smooth", default_coast_smooth)),
                )
                coast_radius = st.slider(
                    "Smooth radius",
                    min_value=0,
                    max_value=12,
                    value=int(pp.get("coast_radius", default_coast_radius)),
                    step=1,
                )
                coast_strength = st.slider(
                    "Smooth strength",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("coast_strength", default_coast_strength)),
                    step=0.05,
                )
                beach = st.toggle(
                    "Beach deposit",
                    value=bool(pp.get("beach", default_beach)),
                )
                beach_amount = st.slider(
                    "Beach amount",
                    min_value=0.0,
                    max_value=0.1,
                    value=float(pp.get("beach_amount", default_beach_amount)),
                    step=0.005,
                )

            with tabs[2]:
                st.markdown("**Erosion**")
                erosion = st.toggle(
                    "Thermal erosion",
                    value=bool(pp.get("erosion", default_erosion)),
                )
                erosion_iter = st.slider(
                    "Iterations",
                    min_value=0,
                    max_value=250,
                    value=int(pp.get("erosion_iter", default_erosion_iter)),
                    step=5,
                )
                erosion_talus = st.slider(
                    "Talus",
                    min_value=0.0,
                    max_value=0.1,
                    value=float(pp.get("erosion_talus", default_erosion_talus)),
                    step=0.005,
                )
                erosion_strength = st.slider(
                    "Strength",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("erosion_strength", default_erosion_strength)),
                    step=0.05,
                )

                st.divider()
                hydraulic = st.toggle(
                    "Hydraulic erosion",
                    value=bool(pp.get("hydraulic", default_hydraulic)),
                )
                hyd_iter = st.slider(
                    "Iterations (hydraulic)",
                    min_value=0,
                    max_value=250,
                    value=int(pp.get("hyd_iter", default_hyd_iter)),
                    step=5,
                )
                hyd_rain = st.slider(
                    "Rain",
                    min_value=0.0,
                    max_value=0.05,
                    value=float(pp.get("hyd_rain", default_hyd_rain)),
                    step=0.001,
                )
                hyd_evap = st.slider(
                    "Evaporation",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("hyd_evap", default_hyd_evap)),
                    step=0.05,
                )
                hyd_flow = st.slider(
                    "Flow rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("hyd_flow", default_hyd_flow)),
                    step=0.05,
                )
                hyd_capacity = st.slider(
                    "Sediment capacity",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(pp.get("hyd_capacity", default_hyd_capacity)),
                    step=0.25,
                )
                hyd_erosion = st.slider(
                    "Erode rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("hyd_erosion", default_hyd_erosion)),
                    step=0.05,
                )
                hyd_deposition = st.slider(
                    "Deposit rate",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("hyd_deposition", default_hyd_deposition)),
                    step=0.05,
                )

            with tabs[3]:
                st.markdown("**Movement**")
                nav_step_px = st.slider(
                    "Move step (px)",
                    min_value=1.0,
                    max_value=512.0,
                    value=float(pp.get("nav_step_px", default_nav_step_px)),
                    step=1.0,
                )
                chunk_size_px = st.slider(
                    "Chunk size (px)",
                    min_value=32,
                    max_value=2048,
                    value=int(pp.get("chunk_size_px", default_chunk_size_px)),
                    step=32,
                )
                show_chunk_grid = st.toggle(
                    "Show chunk grid",
                    value=bool(pp.get("show_chunk_grid", default_show_chunk_grid)),
                )
                chunk_cache_n = st.slider(
                    "Chunk cache (LRU)",
                    min_value=0,
                    max_value=256,
                    value=int(pp.get("chunk_cache_n", default_chunk_cache_n)),
                    step=4,
                )
                backend = st.selectbox(
                    "Backend",
                    ["Reference", "Fast"],
                    index=0
                    if str(pp.get("backend", default_backend)) == "Reference"
                    else 1,
                    help=(
                        "Fast uses float32 internal arrays "
                        "(approximate but deterministic)."
                    ),
                )

                st.divider()
                st.markdown("**Constraints**")
                constraints = st.toggle(
                    "Enable movement constraints",
                    value=bool(pp.get("constraints", default_constraints)),
                )
                water_behavior = st.selectbox(
                    "Water",
                    ["Slow", "Block"],
                    index=0
                    if str(pp.get("water_behavior", default_water_behavior)) == "Slow"
                    else 1,
                )
                water_slow = st.slider(
                    "Water speed factor",
                    min_value=0.05,
                    max_value=1.0,
                    value=float(pp.get("water_slow", default_water_slow)),
                    step=0.05,
                )
                slope_block = st.slider(
                    "Slope block threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("slope_block", default_slope_block)),
                    step=0.05,
                )
                slope_cost = st.slider(
                    "Slope cost",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(pp.get("slope_cost", default_slope_cost)),
                    step=0.25,
                )

            with tabs[4]:
                st.markdown("**View + extras**")
                climate = st.toggle(
                    "Climate biomes",
                    value=bool(pp.get("climate", default_climate)),
                )
                climate_scale = st.slider(
                    "Climate scale",
                    min_value=20.0,
                    max_value=2000.0,
                    value=float(pp.get("climate_scale", default_climate_scale)),
                    step=10.0,
                )
                climate_strength = st.slider(
                    "Climate strength",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("climate_strength", default_climate_strength)),
                    step=0.05,
                )
                carto = st.toggle(
                    "Cartographic mode",
                    value=bool(pp.get("carto", default_cartographic)),
                )
                contour_interval = st.slider(
                    "Contour interval",
                    min_value=0.01,
                    max_value=0.2,
                    value=float(pp.get("contour_interval", default_contour_interval)),
                    step=0.01,
                )
                contour_alpha = st.slider(
                    "Contour alpha",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("contour_alpha", default_contour_alpha)),
                    step=0.05,
                )

                st.divider()
                veg = st.toggle(
                    "Vegetation",
                    value=bool(pp.get("veg", default_veg)),
                )
                veg_cell = st.slider(
                    "Vegetation spacing (px)",
                    min_value=6,
                    max_value=64,
                    value=int(pp.get("veg_cell", default_veg_cell)),
                    step=2,
                )
                veg_p = st.slider(
                    "Vegetation density",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("veg_p", default_veg_p)),
                    step=0.05,
                )
                rocks = st.toggle(
                    "Rocks",
                    value=bool(pp.get("rocks", default_rocks)),
                )

                st.divider()
                trails = st.toggle(
                    "Trails (A*)",
                    value=bool(pp.get("trails", default_trails)),
                )
                trail_tx = st.slider(
                    "Trail target X (0..1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("trail_tx", default_trail_tx)),
                    step=0.01,
                )
                trail_ty = st.slider(
                    "Trail target Y (0..1)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("trail_ty", default_trail_ty)),
                    step=0.01,
                )

                st.divider()
                tiles_grid = st.slider(
                    "Tiles grid",
                    min_value=1,
                    max_value=8,
                    value=int(pp.get("tiles_grid", default_tiles_grid)),
                    step=1,
                )
                tiles_size = st.slider(
                    "Tile size",
                    min_value=64,
                    max_value=512,
                    value=int(pp.get("tiles_size", default_tiles_size)),
                    step=32,
                )
                tiles_z = st.slider(
                    "Tile zoom (label)",
                    min_value=0,
                    max_value=12,
                    value=int(pp.get("tiles_z", default_tiles_z)),
                    step=1,
                )

            apply_practical = st.form_submit_button(
                "Apply",
                type="primary",
                use_container_width=True,
            )

        if apply_practical:
            pp = {
                "water_level": float(water_level),
                "shore_width": float(shore_width),
                "mountain_level": float(mountain_level),
                "snowline": float(snowline),
                "shade_az": float(shade_az),
                "shade_alt": float(shade_alt),
                "shade_strength": float(shade_strength),
                "river_q": float(river_q),
                "river_carve": bool(river_carve),
                "river_depth": float(river_depth),
                "fill_lakes": bool(fill_lakes),
                "coast_smooth": bool(coast_smooth),
                "coast_radius": int(coast_radius),
                "coast_strength": float(coast_strength),
                "beach": bool(beach),
                "beach_amount": float(beach_amount),
                "erosion": bool(erosion),
                "erosion_iter": int(erosion_iter),
                "erosion_talus": float(erosion_talus),
                "erosion_strength": float(erosion_strength),
                "hydraulic": bool(hydraulic),
                "hyd_iter": int(hyd_iter),
                "hyd_rain": float(hyd_rain),
                "hyd_evap": float(hyd_evap),
                "hyd_flow": float(hyd_flow),
                "hyd_capacity": float(hyd_capacity),
                "hyd_erosion": float(hyd_erosion),
                "hyd_deposition": float(hyd_deposition),
                "player_x": float(pp.get("player_x", default_player_x)),
                "player_y": float(pp.get("player_y", default_player_y)),
                "nav_step_px": float(nav_step_px),
                "chunk_size_px": int(chunk_size_px),
                "show_chunk_grid": bool(show_chunk_grid),
                "chunk_cache_n": int(chunk_cache_n),
                "backend": str(backend),
                "constraints": bool(constraints),
                "water_behavior": str(water_behavior),
                "water_slow": float(water_slow),
                "slope_block": float(slope_block),
                "slope_cost": float(slope_cost),
                "climate": bool(climate),
                "climate_scale": float(climate_scale),
                "climate_strength": float(climate_strength),
                "veg": bool(veg),
                "veg_cell": int(veg_cell),
                "veg_p": float(veg_p),
                "rocks": bool(rocks),
                "trails": bool(trails),
                "trail_tx": float(trail_tx),
                "trail_ty": float(trail_ty),
                "carto": bool(carto),
                "contour_interval": float(contour_interval),
                "contour_alpha": float(contour_alpha),
                "tiles_grid": int(tiles_grid),
                "tiles_size": int(tiles_size),
                "tiles_z": int(tiles_z),
            }
            st.session_state["practical_params"] = pp

            new = dict(params_for_url)
            new.update(
                {
                    "water_level": str(float(pp["water_level"])),
                    "shore_width": str(float(pp["shore_width"])),
                    "mountain_level": str(float(pp["mountain_level"])),
                    "snowline": str(float(pp["snowline"])),
                    "shade_az": str(float(pp["shade_az"])),
                    "shade_alt": str(float(pp["shade_alt"])),
                    "shade_strength": str(float(pp["shade_strength"])),
                    "river_q": str(float(pp["river_q"])),
                    "river_carve": "1" if bool(pp["river_carve"]) else "0",
                    "river_depth": str(float(pp["river_depth"])),
                    "fill_lakes": "1" if bool(pp["fill_lakes"]) else "0",
                    "coast_smooth": "1" if bool(pp["coast_smooth"]) else "0",
                    "coast_radius": str(int(pp["coast_radius"])),
                    "coast_strength": str(float(pp["coast_strength"])),
                    "beach": "1" if bool(pp["beach"]) else "0",
                    "beach_amount": str(float(pp["beach_amount"])),
                    "erosion": "1" if bool(pp["erosion"]) else "0",
                    "erosion_iter": str(int(pp["erosion_iter"])),
                    "erosion_talus": str(float(pp["erosion_talus"])),
                    "erosion_strength": str(float(pp["erosion_strength"])),
                    "hydraulic": "1" if bool(pp["hydraulic"]) else "0",
                    "hyd_iter": str(int(pp["hyd_iter"])),
                    "hyd_rain": str(float(pp["hyd_rain"])),
                    "hyd_evap": str(float(pp["hyd_evap"])),
                    "hyd_flow": str(float(pp["hyd_flow"])),
                    "hyd_capacity": str(float(pp["hyd_capacity"])),
                    "hyd_erosion": str(float(pp["hyd_erosion"])),
                    "hyd_deposition": str(float(pp["hyd_deposition"])),
                    "nav_step_px": str(float(pp["nav_step_px"])),
                    "chunk_size_px": str(int(pp["chunk_size_px"])),
                    "show_chunk_grid": "1" if bool(pp["show_chunk_grid"]) else "0",
                    "chunk_cache_n": str(int(pp["chunk_cache_n"])),
                    "backend": str(pp["backend"]),
                    "constraints": "1" if bool(pp["constraints"]) else "0",
                    "water_behavior": str(pp["water_behavior"]),
                    "water_slow": str(float(pp["water_slow"])),
                    "slope_block": str(float(pp["slope_block"])),
                    "slope_cost": str(float(pp["slope_cost"])),
                    "climate": "1" if bool(pp["climate"]) else "0",
                    "climate_scale": str(float(pp["climate_scale"])),
                    "climate_strength": str(float(pp["climate_strength"])),
                    "veg": "1" if bool(pp["veg"]) else "0",
                    "veg_cell": str(int(pp["veg_cell"])),
                    "veg_p": str(float(pp["veg_p"])),
                    "rocks": "1" if bool(pp["rocks"]) else "0",
                    "trails": "1" if bool(pp["trails"]) else "0",
                    "trail_tx": str(float(pp["trail_tx"])),
                    "trail_ty": str(float(pp["trail_ty"])),
                    "carto": "1" if bool(pp["carto"]) else "0",
                    "contour_interval": str(float(pp["contour_interval"])),
                    "contour_alpha": str(float(pp["contour_alpha"])),
                    "tiles_grid": str(int(pp["tiles_grid"])),
                    "tiles_size": str(int(pp["tiles_size"])),
                    "tiles_z": str(int(pp["tiles_z"])),
                }
            )
            _set_query_params(new)
            st.rerun()

    pp = dict(st.session_state.get("practical_params", {}))
    player_x = float(pp.get("player_x", default_player_x))
    player_y = float(pp.get("player_y", default_player_y))
    nav_step_px = float(pp.get("nav_step_px", default_nav_step_px))
    chunk_size_px = int(pp.get("chunk_size_px", default_chunk_size_px))
    show_chunk_grid = bool(pp.get("show_chunk_grid", default_show_chunk_grid))
    chunk_cache_n = int(pp.get("chunk_cache_n", default_chunk_cache_n))
    backend = str(pp.get("backend", default_backend))
    step_world = nav_step_px / max(float(scale), 1e-9)

    player_x = _snap_to_scale(float(player_x), scale=float(scale))
    player_y = _snap_to_scale(float(player_y), scale=float(scale))
    view_left = _snap_to_scale(
        player_x - (float(width_render) / (2.0 * max(float(scale), 1e-9))),
        scale=float(scale),
    )
    view_top = _snap_to_scale(
        player_y - (float(height_render) / (2.0 * max(float(scale), 1e-9))),
        scale=float(scale),
    )

    chunk_size_px = max(int(chunk_size_px), 32)
    chunk_world = float(chunk_size_px) / max(float(scale), 1e-9)

    view_w_world = float(int(width_render) - 1) / max(float(scale), 1e-9)
    view_h_world = float(int(height_render) - 1) / max(float(scale), 1e-9)

    cx0 = int(math.floor(view_left / chunk_world))
    cy0 = int(math.floor(view_top / chunk_world))
    cx1 = int(math.floor((view_left + view_w_world) / chunk_world))
    cy1 = int(math.floor((view_top + view_h_world) / chunk_world))

    seed_i = int(seed)
    basis_s = str(basis)
    grad2_s = str(grad2)
    noise_variant_s = str(noise_variant)
    warp_amp_f = float(warp_amp)
    warp_scale_f = float(warp_scale)
    warp_octaves_i = int(warp_octaves)
    scale_f = float(scale)
    octaves_i = int(octaves)
    lacunarity_f = float(lacunarity)
    persistence_f = float(persistence)
    z_scale_f = float(z_scale)

    water_level_f = float(pp.get("water_level", default_water_level))
    shore_width_f = float(pp.get("shore_width", default_shore_width))
    mountain_level_f = float(pp.get("mountain_level", default_mountain_level))
    snowline_f = float(pp.get("snowline", default_snowline))
    shade_az_f = float(pp.get("shade_az", default_shade_az))
    shade_alt_f = float(pp.get("shade_alt", default_shade_alt))
    shade_strength_f = float(pp.get("shade_strength", default_shade_strength))
    river_q_f = float(pp.get("river_q", default_river_q))
    river_carve_b = bool(pp.get("river_carve", default_river_carve))
    river_depth_f = float(pp.get("river_depth", default_river_depth))
    fill_lakes_b = bool(pp.get("fill_lakes", default_fill_lakes))
    coast_smooth_b = bool(pp.get("coast_smooth", default_coast_smooth))
    coast_radius_i = int(pp.get("coast_radius", default_coast_radius))
    coast_strength_f = float(pp.get("coast_strength", default_coast_strength))
    beach_b = bool(pp.get("beach", default_beach))
    beach_amount_f = float(pp.get("beach_amount", default_beach_amount))
    thermal_on_b = bool(pp.get("erosion", default_erosion))
    thermal_iter_i = int(pp.get("erosion_iter", default_erosion_iter))
    thermal_talus_f = float(pp.get("erosion_talus", default_erosion_talus))
    thermal_strength_f = float(pp.get("erosion_strength", default_erosion_strength))
    hydraulic_on_b = bool(pp.get("hydraulic", default_hydraulic))
    hyd_iter_i = int(pp.get("hyd_iter", default_hyd_iter))
    hyd_rain_f = float(pp.get("hyd_rain", default_hyd_rain))
    hyd_evap_f = float(pp.get("hyd_evap", default_hyd_evap))
    hyd_flow_f = float(pp.get("hyd_flow", default_hyd_flow))
    hyd_capacity_f = float(pp.get("hyd_capacity", default_hyd_capacity))
    hyd_erosion_f = float(pp.get("hyd_erosion", default_hyd_erosion))
    hyd_deposition_f = float(pp.get("hyd_deposition", default_hyd_deposition))

    pipeline_params: dict[str, object] = {
        "seed": seed_i,
        "basis": basis_s,
        "grad2": grad2_s,
        "noise_variant": noise_variant_s,
        "warp_amp": warp_amp_f,
        "warp_scale": warp_scale_f,
        "warp_octaves": warp_octaves_i,
        "scale": scale_f,
        "octaves": octaves_i,
        "lacunarity": lacunarity_f,
        "persistence": persistence_f,
        "z_scale": z_scale_f,
        "water_level": water_level_f,
        "shore_width": shore_width_f,
        "mountain_level": mountain_level_f,
        "snowline": snowline_f,
        "shade_az": shade_az_f,
        "shade_alt": shade_alt_f,
        "shade_strength": shade_strength_f,
        "river_q": river_q_f,
        "river_carve": river_carve_b,
        "river_depth": river_depth_f,
        "fill_lakes": fill_lakes_b,
        "coast_smooth": coast_smooth_b,
        "coast_radius": coast_radius_i,
        "coast_strength": coast_strength_f,
        "beach": beach_b,
        "beach_amount": beach_amount_f,
        "thermal_on": thermal_on_b,
        "thermal_iter": thermal_iter_i,
        "thermal_talus": thermal_talus_f,
        "thermal_strength": thermal_strength_f,
        "hydraulic_on": hydraulic_on_b,
        "hyd_iter": hyd_iter_i,
        "hyd_rain": hyd_rain_f,
        "hyd_evap": hyd_evap_f,
        "hyd_flow": hyd_flow_f,
        "hyd_capacity": hyd_capacity_f,
        "hyd_erosion": hyd_erosion_f,
        "hyd_deposition": hyd_deposition_f,
        "backend": str(backend),
    }
    frozen = _freeze_params(pipeline_params)

    def get_chunk(chunk_x: int, chunk_y: int) -> dict[str, object]:
        key = (frozen, int(chunk_x), int(chunk_y), int(chunk_size_px))
        cached = _chunk_cache_get(key)
        if cached is not None:
            return cast(dict[str, object], cached)

        left = float(int(chunk_x)) * chunk_world
        top = float(int(chunk_y)) * chunk_world
        out = _practical_pipeline(
            seed=int(seed_i),
            basis=str(basis_s),
            grad2=str(grad2_s),
            noise_variant=str(noise_variant_s),
            warp_amp=float(warp_amp_f),
            warp_scale=float(warp_scale_f),
            warp_octaves=int(warp_octaves_i),
            scale=float(scale_f),
            octaves=int(octaves_i),
            lacunarity=float(lacunarity_f),
            persistence=float(persistence_f),
            width=int(chunk_size_px),
            height=int(chunk_size_px),
            view_left=float(left),
            view_top=float(top),
            z_scale=float(z_scale_f),
            water_level=float(water_level_f),
            shore_width=float(shore_width_f),
            mountain_level=float(mountain_level_f),
            snowline=float(snowline_f),
            shade_az=float(shade_az_f),
            shade_alt=float(shade_alt_f),
            shade_strength=float(shade_strength_f),
            river_q=float(river_q_f),
            river_carve=bool(river_carve_b),
            river_depth=float(river_depth_f),
            fill_lakes=bool(fill_lakes_b),
            coast_smooth=bool(coast_smooth_b),
            coast_radius=int(coast_radius_i),
            coast_strength=float(coast_strength_f),
            beach=bool(beach_b),
            beach_amount=float(beach_amount_f),
            thermal_on=bool(thermal_on_b),
            thermal_iter=int(thermal_iter_i),
            thermal_talus=float(thermal_talus_f),
            thermal_strength=float(thermal_strength_f),
            hydraulic_on=bool(hydraulic_on_b),
            hyd_iter=int(hyd_iter_i),
            hyd_rain=float(hyd_rain_f),
            hyd_evap=float(hyd_evap_f),
            hyd_flow=float(hyd_flow_f),
            hyd_capacity=float(hyd_capacity_f),
            hyd_erosion=float(hyd_erosion_f),
            hyd_deposition=float(hyd_deposition_f),
            backend=str(backend),
        )
        _chunk_cache_put(key, out, max_items=int(chunk_cache_n))
        return cast(dict[str, object], out)

    def stitch(field: str) -> np.ndarray:
        rows: list[np.ndarray] = []
        for yy in range(cy0, cy1 + 1):
            parts: list[np.ndarray] = []
            for xx in range(cx0, cx1 + 1):
                parts.append(np.asarray(get_chunk(xx, yy)[field]))
            rows.append(np.concatenate(parts, axis=1))
        return np.concatenate(rows, axis=0)

    def stitch_opt(field: str) -> np.ndarray | None:
        rows: list[np.ndarray] = []
        any_present = False
        for yy in range(cy0, cy1 + 1):
            parts: list[np.ndarray] = []
            for xx in range(cx0, cx1 + 1):
                v = get_chunk(xx, yy).get(field)
                if v is None:
                    v = np.zeros((chunk_size_px, chunk_size_px), dtype=np.float64)
                else:
                    any_present = True
                parts.append(np.asarray(v))
            rows.append(np.concatenate(parts, axis=1))
        return None if not any_present else np.concatenate(rows, axis=0)

    full_base01 = stitch("base01")
    full_terr01 = stitch("terr01")
    full_terr_river01 = stitch("terr_river01")
    full_s01 = stitch("s01")
    full_shade01 = stitch("shade01")
    full_lake_depth = stitch("lake_depth")
    full_acc = stitch("acc")
    full_rivers = stitch("rivers").astype(bool)
    full_rgb = stitch("rgb")
    full_biome = stitch("biome").astype(np.uint8)
    full_hyd_water = stitch_opt("hyd_water")
    full_hyd_sediment = stitch_opt("hyd_sediment")

    x0 = int(round((view_left - (float(cx0) * chunk_world)) * float(scale)))
    y0 = int(round((view_top - (float(cy0) * chunk_world)) * float(scale)))
    x1 = x0 + int(width_render)
    y1 = y0 + int(height_render)

    base01 = np.asarray(full_base01[y0:y1, x0:x1], dtype=np.float64)
    terr01 = np.asarray(full_terr01[y0:y1, x0:x1], dtype=np.float64)
    terr_river01 = np.asarray(full_terr_river01[y0:y1, x0:x1], dtype=np.float64)
    s01 = np.asarray(full_s01[y0:y1, x0:x1], dtype=np.float64)
    shade01 = np.asarray(full_shade01[y0:y1, x0:x1], dtype=np.float64)
    lake_depth = np.asarray(full_lake_depth[y0:y1, x0:x1], dtype=np.float64)
    acc = np.asarray(full_acc[y0:y1, x0:x1], dtype=np.float64)
    rivers = np.asarray(full_rivers[y0:y1, x0:x1]).astype(bool)
    rgb = np.asarray(full_rgb[y0:y1, x0:x1], dtype=np.float64)
    biome = np.asarray(full_biome[y0:y1, x0:x1], dtype=np.uint8)

    hyd_water = (
        None
        if full_hyd_water is None
        else np.asarray(full_hyd_water[y0:y1, x0:x1], dtype=np.float64)
    )
    hyd_sediment = (
        None
        if full_hyd_sediment is None
        else np.asarray(full_hyd_sediment[y0:y1, x0:x1], dtype=np.float64)
    )

    seam_delta = np.zeros((int(height_render), int(width_render)), dtype=np.float64)
    # Mark only chunk boundaries.
    for k in range(1, int(cx1 - cx0 + 1)):
        bx = int(k * chunk_size_px)
        lx = bx - x0
        if 1 <= lx < int(width_render):
            seam_delta[:, lx] = np.abs(terr_river01[:, lx - 1] - terr_river01[:, lx])
    for k in range(1, int(cy1 - cy0 + 1)):
        by = int(k * chunk_size_px)
        ly = by - y0
        if 1 <= ly < int(height_render):
            seam_delta[ly, :] = np.abs(terr_river01[ly - 1, :] - terr_river01[ly, :])

    shore_level = min(
        1.0,
        float(pp.get("water_level", default_water_level))
        + float(pp.get("shore_width", default_shore_width)),
    )

    # Used for hotkey-based movement constraints.
    st.session_state["practical_last_height"] = terr_river01
    st.session_state["practical_last_slope"] = s01

    if "practical_overlays" not in st.session_state:
        st.session_state["practical_overlays"] = {
            "rivers": True,
            "lakes": True,
            "veg": True,
            "rocks": True,
            "trails": True,
        }
    overlays = cast(dict[str, bool], st.session_state["practical_overlays"])

    rgb_vis = rgb
    temp01 = None
    moist01 = None
    climate_biome = None
    if bool(pp.get("climate", default_climate)):
        temp01, moist01 = _climate_fields(
            seed=int(seed),
            basis=str(basis),
            grad2=str(grad2),
            width=int(width_render),
            height=int(height_render),
            climate_scale=float(pp.get("climate_scale", default_climate_scale)),
            view_left=float(view_left),
            view_top=float(view_top),
        )
        climate_biome = climate_biome_map(
            terr_river01,
            temp01,
            moist01,
            water_level=float(pp.get("water_level", default_water_level)),
            snowline=float(pp.get("snowline", default_snowline)),
        )
        rgb_vis = apply_climate_palette(
            rgb_vis,
            climate_biome,
            strength=float(pp.get("climate_strength", default_climate_strength)),
        )

    if bool(pp.get("carto", default_cartographic)):
        cm = contour_mask(
            terr_river01,
            interval=float(pp.get("contour_interval", default_contour_interval)),
        )
        rgb_vis = apply_mask_overlay(
            rgb_vis,
            cm,
            color=(0.05, 0.05, 0.05),
            alpha=float(pp.get("contour_alpha", default_contour_alpha)),
        )
        water = terr_river01 < float(pp.get("water_level", default_water_level))
        edge = np.zeros_like(water, dtype=bool)
        edge[:, 1:] |= water[:, 1:] != water[:, :-1]
        edge[1:, :] |= water[1:, :] != water[:-1, :]
        rgb_vis = apply_mask_overlay(rgb_vis, edge, color=(0.02, 0.02, 0.02), alpha=0.6)

    # Hydrology overlays: keep these after cartographic tint so they stay visible.
    if bool(overlays.get("lakes", True)) and bool(
        pp.get("fill_lakes", default_fill_lakes)
    ):
        lake_mask = np.asarray(lake_depth, dtype=np.float64) > 1e-12
        rgb_vis = _rgb_overlay(
            rgb_vis,
            lake_mask,
            color=(0.20, 0.62, 0.92),
            alpha=0.22,
        )

    if bool(overlays.get("rivers", True)):
        rgb_vis = _rgb_overlay(
            rgb_vis,
            rivers,
            color=(0.05, 0.40, 0.72),
            alpha=0.75,
        )

    veg_x = np.array([], dtype=np.float64)
    veg_y = np.array([], dtype=np.float64)
    rock_x = np.array([], dtype=np.float64)
    rock_y = np.array([], dtype=np.float64)
    if bool(pp.get("veg", default_veg)) or bool(pp.get("rocks", default_rocks)):
        water = terr_river01 < float(pp.get("water_level", default_water_level))
        land_mask = (~water) & (
            terr_river01 < float(pp.get("mountain_level", default_mountain_level))
        )
        land_mask &= s01 < 0.65
        mount_mask = terr_river01 >= float(
            pp.get("mountain_level", default_mountain_level)
        )
        mount_mask |= s01 > 0.7

        if bool(pp.get("veg", default_veg)):
            veg_x, veg_y = jittered_points(
                seed=int(seed) + 17,
                height=int(height_render),
                width=int(width_render),
                cell=int(pp.get("veg_cell", default_veg_cell)),
                probability=float(pp.get("veg_p", default_veg_p)),
            )
            veg_x, veg_y = filter_points_by_mask(veg_x, veg_y, land_mask)

        if bool(pp.get("rocks", default_rocks)):
            rock_x, rock_y = jittered_points(
                seed=int(seed) + 33,
                height=int(height_render),
                width=int(width_render),
                cell=max(int(pp.get("veg_cell", default_veg_cell)) + 6, 8),
                probability=0.35,
            )
            rock_x, rock_y = filter_points_by_mask(rock_x, rock_y, mount_mask)

    trail_path_xy = None
    if bool(pp.get("trails", default_trails)):
        H, W = terr_river01.shape
        step = max(1, int(max(H, W) // 192))
        cost = 1.0 + 3.0 * s01
        water = terr_river01 < float(pp.get("water_level", default_water_level))
        cost = np.where(water, np.inf, cost)
        cost_ds = cost[::step, ::step]
        sy = int((H - 1) // 2) // step
        sx = int((W - 1) // 2) // step
        gy = int(float(pp.get("trail_ty", default_trail_ty)) * float(H - 1)) // step
        gx = int(float(pp.get("trail_tx", default_trail_tx)) * float(W - 1)) // step
        path = astar_path(cost_ds, start=(sy, sx), goal=(gy, gx))
        if path:
            xs = [int(x * step) for (y, x) in path]
            ys = [int(y * step) for (y, x) in path]
            trail_path_xy = (xs, ys)

    with right_col:
        with st.form("practical_quick", border=False):
            qc0, qc1, qc2, qc3, qc4 = st.columns(5)
            with qc0:
                qc_water = st.slider(
                    "Water",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("water_level", default_water_level)),
                    step=0.01,
                )
            with qc1:
                qc_mtn = st.slider(
                    "Mountains",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(pp.get("mountain_level", default_mountain_level)),
                    step=0.01,
                )
            with qc2:
                qc_rivers = st.toggle(
                    "Rivers",
                    value=bool(pp.get("river_carve", default_river_carve)),
                )
            with qc3:
                qc_erosion = st.toggle(
                    "Erosion",
                    value=bool(pp.get("erosion", default_erosion)),
                )
            with qc4:
                qc_backend = st.selectbox(
                    "Backend",
                    ["Reference", "Fast"],
                    index=0
                    if str(pp.get("backend", default_backend)) == "Reference"
                    else 1,
                )

            qc_apply = st.form_submit_button(
                "Apply quick",
                type="secondary",
                use_container_width=True,
            )

        if qc_apply:
            pp2 = dict(st.session_state.get("practical_params", {}))
            pp2["water_level"] = float(qc_water)
            pp2["mountain_level"] = float(qc_mtn)
            pp2["river_carve"] = bool(qc_rivers)
            pp2["erosion"] = bool(qc_erosion)
            pp2["backend"] = str(qc_backend)
            st.session_state["practical_params"] = pp2

            new = dict(params_for_url)
            new["water_level"] = str(float(pp2["water_level"]))
            new["mountain_level"] = str(float(pp2["mountain_level"]))
            new["river_carve"] = "1" if bool(pp2["river_carve"]) else "0"
            new["erosion"] = "1" if bool(pp2["erosion"]) else "0"
            new["backend"] = str(pp2["backend"])
            _set_query_params(new)
            st.rerun()

        cache_state = _chunk_cache_state()
        hits = int(cache_state.get("hits", 0))
        misses = int(cache_state.get("misses", 0))
        evictions = int(cache_state.get("evictions", 0))
        size = len(cast(list[object], cache_state.get("order", [])))

        s0, s1, s2, s3, s4 = st.columns(5)
        s0.metric("chunk", f"({cx0},{cy0})..({cx1},{cy1})")
        s1.metric("cache", str(size))
        s2.metric("hits", str(hits))
        s3.metric("misses", str(misses))
        s4.metric("backend", str(backend))

        if bool(debug_mode):
            st.caption(f"LRU evictions: {evictions}")

        with st.expander("Overlays", expanded=True):
            oc0, oc1, oc2 = st.columns(3)
            with oc0:
                overlays["rivers"] = st.toggle(
                    "Rivers",
                    value=bool(overlays.get("rivers", True)),
                    key="ov_rivers",
                )
                overlays["lakes"] = st.toggle(
                    "Lakes",
                    value=bool(overlays.get("lakes", True)),
                    key="ov_lakes",
                )
            with oc1:
                overlays["veg"] = st.toggle(
                    "Vegetation",
                    value=bool(overlays.get("veg", True)),
                    key="ov_veg",
                )
                overlays["rocks"] = st.toggle(
                    "Rocks",
                    value=bool(overlays.get("rocks", True)),
                    key="ov_rocks",
                )
            with oc2:
                overlays["trails"] = st.toggle(
                    "Trails",
                    value=bool(overlays.get("trails", True)),
                    key="ov_trails",
                )

            st.session_state["practical_overlays"] = overlays

        tabs_right = st.tabs(["Viewport", "Hydrology", "Weathering", "Export"])

    with tabs_right[0]:
        st.markdown("**Terrain map**")

        t2d, t3d = st.tabs(["2D", "3D"])
        with t2d:
            fig2 = _rgb_figure(
                rgb_vis,
                marker_xy=(
                    float(width_render - 1) / 2.0,
                    float(height_render - 1) / 2.0,
                ),
                marker_label="You",
                height=480,
            )

            if bool(show_chunk_grid) and int(chunk_size_px) > 0:
                cw = float(int(chunk_size_px)) / max(float(scale), 1e-9)
                wx0 = float(view_left)
                wx1 = float(view_left) + (
                    float(width_render - 1) / max(float(scale), 1e-9)
                )
                wy0 = float(view_top)
                wy1 = float(view_top) + (
                    float(height_render - 1) / max(float(scale), 1e-9)
                )

                kx0 = int(math.floor(wx0 / cw))
                kx1 = int(math.floor(wx1 / cw)) + 1
                ky0 = int(math.floor(wy0 / cw))
                ky1 = int(math.floor(wy1 / cw)) + 1

                for k in range(kx0, kx1 + 1):
                    xw = float(k) * cw
                    xp = (xw - wx0) * float(scale)
                    if 0.0 <= xp <= float(width_render - 1):
                        fig2.add_shape(
                            type="line",
                            x0=xp,
                            x1=xp,
                            y0=0,
                            y1=float(height_render - 1),
                            line=dict(color="rgba(255,255,255,0.22)", width=1),
                        )
                for k in range(ky0, ky1 + 1):
                    yw = float(k) * cw
                    yp = (yw - wy0) * float(scale)
                    if 0.0 <= yp <= float(height_render - 1):
                        fig2.add_shape(
                            type="line",
                            x0=0,
                            x1=float(width_render - 1),
                            y0=yp,
                            y1=yp,
                            line=dict(color="rgba(255,255,255,0.22)", width=1),
                        )

            if veg_x.size and bool(overlays.get("veg", True)):
                fig2.add_trace(
                    go.Scatter(
                        x=veg_x,
                        y=veg_y,
                        mode="markers",
                        marker=dict(size=3, color="rgba(40,150,70,0.85)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            if rock_x.size and bool(overlays.get("rocks", True)):
                fig2.add_trace(
                    go.Scatter(
                        x=rock_x,
                        y=rock_y,
                        mode="markers",
                        marker=dict(size=3, color="rgba(110,110,110,0.8)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            if trail_path_xy is not None and bool(overlays.get("trails", True)):
                xs, ys = trail_path_xy
                fig2.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color="rgba(255,255,255,0.85)", width=2),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

            st.plotly_chart(fig2, width="stretch", key="practical_terrain_rgb")

            if bool(debug_mode):
                with st.expander("Inspector layers"):
                    c0, c1 = st.columns(2)
                    with c0:
                        st.markdown("**Height (0..1)**")
                        st.plotly_chart(
                            _heatmap(
                                terr_river01,
                                colorscale="Earth",
                                show_colorbar=True,
                                height=420,
                            ),
                            width="stretch",
                            key="practical_height",
                        )
                    with c1:
                        st.markdown("**Slope (0..1)**")
                        st.plotly_chart(
                            _heatmap(
                                s01,
                                colorscale="Cividis",
                                show_colorbar=True,
                                height=420,
                            ),
                            width="stretch",
                            key="practical_slope",
                        )

                    if bool(show_chunk_grid):
                        st.markdown("**Seam delta (chunk edges)**")
                        sd = np.asarray(seam_delta, dtype=np.float64)
                        if float(np.max(sd)) > 0.0:
                            sd = sd / float(np.max(sd))
                        st.plotly_chart(
                            _heatmap(
                                sd,
                                colorscale="Magma",
                                show_colorbar=True,
                                height=420,
                            ),
                            width="stretch",
                            key="practical_seams",
                        )

                    if temp01 is not None and moist01 is not None:
                        c2, c3 = st.columns(2)
                        with c2:
                            st.markdown("**Temperature**")
                            st.plotly_chart(
                                _heatmap(
                                    np.asarray(temp01, dtype=np.float64),
                                    colorscale="Turbo",
                                    show_colorbar=True,
                                    height=420,
                                ),
                                width="stretch",
                                key="practical_temp",
                            )
                        with c3:
                            st.markdown("**Moisture**")
                            st.plotly_chart(
                                _heatmap(
                                    np.asarray(moist01, dtype=np.float64),
                                    colorscale="Viridis",
                                    show_colorbar=True,
                                    height=420,
                                ),
                                width="stretch",
                                key="practical_moist",
                            )

        with t3d:
            h3 = np.asarray(terr_river01, dtype=np.float64)
            H, W = h3.shape
            cx = float(W - 1) / 2.0
            cy = float(H - 1) / 2.0
            cz = float(h3[int(H // 2), int(W // 2)] * float(z_scale))

            wl = float(pp.get("water_level", default_water_level))

            cs = [
                [0.0, "rgb(5,30,70)"],
                [max(0.0, min(1.0, wl)), "rgb(25,120,160)"],
                [max(0.0, min(1.0, shore_level)), "rgb(195,180,130)"],
                [
                    max(
                        0.0,
                        min(
                            1.0, float(pp.get("mountain_level", default_mountain_level))
                        ),
                    ),
                    "rgb(80,80,75)",
                ],
                [
                    max(0.0, min(1.0, float(pp.get("snowline", default_snowline)))),
                    "rgb(235,240,250)",
                ],
                [1.0, "rgb(255,255,255)"],
            ]

            fig3 = go.Figure()
            fig3.add_trace(
                go.Surface(
                    z=h3 * float(z_scale),
                    surfacecolor=h3,
                    colorscale=cs,
                    showscale=False,
                )
            )
            fig3.add_trace(
                go.Surface(
                    z=np.full_like(h3, wl * float(z_scale)),
                    opacity=0.35,
                    colorscale=[[0.0, "rgb(20,110,150)"], [1.0, "rgb(20,110,150)"]],
                    showscale=False,
                )
            )
            fig3.add_trace(
                go.Scatter3d(
                    x=[cx],
                    y=[cy],
                    z=[cz + 2.0],
                    mode="markers",
                    marker=dict(size=4, color="rgba(255,255,255,0.95)"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
            fig3.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                height=520,
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    aspectmode="data",
                ),
            )
            st.plotly_chart(fig3, width="stretch", key="practical_terrain_3d")

    with tabs_right[1]:
        st.markdown("**Hydrology layers**")
        loga = np.log1p(acc)
        if float(np.max(loga)) > 0.0:
            loga = loga / float(np.max(loga))
        st.plotly_chart(
            _heatmap(loga, colorscale="Viridis", show_colorbar=True, height=460),
            width="stretch",
            key="practical_accum",
        )

        if bool(pp.get("fill_lakes", default_fill_lakes)):
            ld = np.asarray(lake_depth, dtype=np.float64)
            if float(np.max(ld)) > 0.0:
                ld = ld / float(np.max(ld))
            st.plotly_chart(
                _heatmap(ld, colorscale="IceFire", show_colorbar=True, height=460),
                width="stretch",
                key="practical_lake_depth",
            )

    with tabs_right[2]:
        st.markdown("**Weathering (erosion before/after)**")
        if bool(pp.get("erosion", False)) or bool(pp.get("hydraulic", False)):
            diff = terr01 - base01
            c0, c1, c2 = st.columns(3)
            with c0:
                st.markdown("**Before**")
                st.plotly_chart(
                    _heatmap(
                        base01, colorscale="Earth", show_colorbar=False, height=340
                    ),
                    width="stretch",
                    key="practical_before",
                )
            with c1:
                st.markdown("**After**")
                st.plotly_chart(
                    _heatmap(
                        terr01, colorscale="Earth", show_colorbar=False, height=340
                    ),
                    width="stretch",
                    key="practical_after",
                )
            with c2:
                st.markdown("**Delta**")
                st.plotly_chart(
                    _heatmap(
                        diff, colorscale="IceFire", show_colorbar=False, height=340
                    ),
                    width="stretch",
                    key="practical_delta",
                )

            if hyd_water is not None and hyd_sediment is not None:
                c3, c4 = st.columns(2)
                with c3:
                    st.markdown("**Water**")
                    w = np.asarray(hyd_water, dtype=np.float64)
                    if float(np.max(w)) > 0.0:
                        w = w / float(np.max(w))
                    st.plotly_chart(
                        _heatmap(
                            w, colorscale="Cividis", show_colorbar=False, height=320
                        ),
                        width="stretch",
                        key="practical_water",
                    )
                with c4:
                    st.markdown("**Sediment**")
                    s = np.asarray(hyd_sediment, dtype=np.float64)
                    if float(np.max(s)) > 0.0:
                        s = s / float(np.max(s))
                    st.plotly_chart(
                        _heatmap(
                            s, colorscale="Viridis", show_colorbar=False, height=320
                        ),
                        width="stretch",
                        key="practical_sediment",
                    )

            st.divider()
            st.markdown("**Animation**")
            max_frames = st.slider(
                "Max frames",
                min_value=10,
                max_value=120,
                value=60,
                step=5,
                key="anim_max_frames",
            )
            fps = st.slider(
                "FPS",
                min_value=2,
                max_value=30,
                value=12,
                step=1,
                key="anim_fps",
            )

            modes = []
            if bool(pp.get("erosion", False)):
                modes.append("Thermal")
            if bool(pp.get("hydraulic", False)):
                modes.append("Hydraulic")
            mode = st.selectbox("Mode", modes, key="anim_mode")

            if mode == "Thermal":
                it = int(pp.get("erosion_iter", default_erosion_iter))
                every = max(1, int(math.ceil(max(it, 1) / float(max_frames))))
                frames = thermal_erosion_frames(
                    base01,
                    iterations=it,
                    talus=float(pp.get("erosion_talus", default_erosion_talus)),
                    strength=float(
                        pp.get("erosion_strength", default_erosion_strength)
                    ),
                    every=every,
                )
                idx = st.slider(
                    "Frame",
                    min_value=0,
                    max_value=int(frames.shape[0] - 1),
                    value=0,
                    step=1,
                    key="anim_frame",
                )
                st.caption(f"stride={every} iters, frames={int(frames.shape[0])}")
                ph = st.empty()
                ph.plotly_chart(
                    _heatmap(frames[int(idx)], colorscale="Earth", show_colorbar=False),
                    width="stretch",
                    key="thermal_anim_frame",
                )
                cols = st.columns(2)
                play = cols[0].button(
                    "Play", use_container_width=True, key="play_thermal"
                )
                if play:
                    for j in range(int(frames.shape[0])):
                        ph.plotly_chart(
                            _heatmap(
                                frames[int(j)],
                                colorscale="Earth",
                                show_colorbar=False,
                            ),
                            width="stretch",
                        )
                        time.sleep(1.0 / max(float(fps), 1.0))

            else:
                it = int(pp.get("hyd_iter", default_hyd_iter))
                every = max(1, int(math.ceil(max(it, 1) / float(max_frames))))
                hf, wf, sf = hydraulic_erosion_frames(
                    thermal_erosion(
                        base01,
                        iterations=int(pp.get("erosion_iter", default_erosion_iter))
                        if bool(pp.get("erosion", False))
                        else 0,
                        talus=float(pp.get("erosion_talus", default_erosion_talus)),
                        strength=float(
                            pp.get("erosion_strength", default_erosion_strength)
                        ),
                    ),
                    iterations=it,
                    rain=float(pp.get("hyd_rain", default_hyd_rain)),
                    evaporation=float(pp.get("hyd_evap", default_hyd_evap)),
                    flow_rate=float(pp.get("hyd_flow", default_hyd_flow)),
                    capacity=float(pp.get("hyd_capacity", default_hyd_capacity)),
                    erosion=float(pp.get("hyd_erosion", default_hyd_erosion)),
                    deposition=float(pp.get("hyd_deposition", default_hyd_deposition)),
                    every=every,
                )
                idx = st.slider(
                    "Frame",
                    min_value=0,
                    max_value=int(hf.shape[0] - 1),
                    value=0,
                    step=1,
                    key="anim_frame_h",
                )
                st.caption(f"stride={every} iters, frames={int(hf.shape[0])}")
                ph = st.empty()
                c0, c1, c2 = st.columns(3)
                with c0:
                    ph0 = st.empty()
                with c1:
                    ph1 = st.empty()
                with c2:
                    ph2 = st.empty()

                def _render_h(i: int) -> None:
                    h = hf[int(i)]
                    w = wf[int(i)]
                    s = sf[int(i)]
                    if float(np.max(w)) > 0.0:
                        w = w / float(np.max(w))
                    if float(np.max(s)) > 0.0:
                        s = s / float(np.max(s))
                    ph0.plotly_chart(
                        _heatmap(h, colorscale="Earth", show_colorbar=False),
                        width="stretch",
                    )
                    ph1.plotly_chart(
                        _heatmap(w, colorscale="Cividis", show_colorbar=False),
                        width="stretch",
                    )
                    ph2.plotly_chart(
                        _heatmap(s, colorscale="Viridis", show_colorbar=False),
                        width="stretch",
                    )

                _render_h(int(idx))

                play = st.button("Play", use_container_width=True, key="play_hyd")
                if play:
                    for j in range(int(hf.shape[0])):
                        _render_h(int(j))
                        time.sleep(1.0 / max(float(fps), 1.0))
        else:
            st.info(
                "Enable thermal and/or hydraulic erosion in Practical settings "
                "to see before/after."
            )

    with tabs_right[3]:
        st.markdown("**Export**")

        st.markdown("**Navigation**")
        nav_caption = (
            f"Player: x={player_x:.3f}, y={player_y:.3f} "
            f"(step={step_world:.3f} world units)"
        )
        st.caption(nav_caption)

        if bool(debug_mode):
            with st.expander("Navigator logs"):
                if st.button(
                    "Clear logs", use_container_width=True, key="clear_nav_logs"
                ):
                    st.session_state["nav_log"] = []
                    st.rerun()
                log = list(st.session_state.get("nav_log", []))
                if not log:
                    st.caption("No navigator events yet.")
                else:
                    st.json(log[-25:])

        if int(chunk_size_px) > 0:
            cw = float(int(chunk_size_px)) / max(float(scale), 1e-9)
            cx = int(math.floor(float(player_x) / cw))
            cy = int(math.floor(float(player_y) / cw))
            st.caption(f"Chunk: ({cx}, {cy})  size={int(chunk_size_px)}px")

            with st.form("chunk_form", border=False):
                c0, c1, c2 = st.columns([1, 1, 1])
                with c0:
                    chunk_x = st.number_input(
                        "Chunk x",
                        value=int(cx),
                        step=1,
                        key="chunk_x",
                    )
                with c1:
                    chunk_y = st.number_input(
                        "Chunk y",
                        value=int(cy),
                        step=1,
                        key="chunk_y",
                    )
                with c2:
                    go_chunk = st.form_submit_button(
                        "Go to chunk",
                        type="secondary",
                        use_container_width=True,
                    )

            if go_chunk:
                _nav_log(
                    "go_chunk",
                    {
                        "from": {"x": float(player_x), "y": float(player_y)},
                        "to_chunk": {"x": int(chunk_x), "y": int(chunk_y)},
                    },
                )
                player_x = (float(int(chunk_x)) + 0.5) * cw
                player_y = (float(int(chunk_y)) + 0.5) * cw
                pp["player_x"] = float(player_x)
                pp["player_y"] = float(player_y)
                st.session_state["practical_params"] = pp

                view_left = float(player_x) - (
                    float(width_render) / (2.0 * max(float(scale), 1e-9))
                )
                view_top = float(player_y) - (
                    float(height_render) / (2.0 * max(float(scale), 1e-9))
                )
                new = dict(params_for_url)
                new["player_x"] = str(float(player_x))
                new["player_y"] = str(float(player_y))
                new["offset_x"] = str(float(view_left))
                new["offset_y"] = str(float(view_top))
                _set_query_params(new)
                st.rerun()

        with st.expander("Export region"):
            st.download_button(
                "Download terrain preview (PNG)",
                data=_rgb_to_png_bytes(rgb_vis),
                file_name="terrain.png",
                mime="image/png",
            )
            st.download_button(
                "Download heightmap (NPY)",
                data=array_to_npy_bytes(terr_river01),
                file_name="heightmap.npy",
                mime="application/octet-stream",
            )
            st.download_button(
                "Download biome map (NPY)",
                data=array_to_npy_bytes(biome),
                file_name="biome.npy",
                mime="application/octet-stream",
            )
            st.download_button(
                "Download river mask (NPY)",
                data=array_to_npy_bytes(rivers.astype(np.uint8)),
                file_name="rivers.npy",
                mime="application/octet-stream",
            )
            meta = {
                "seed": int(seed),
                "noise": str(noise_variant),
                "basis": str(basis),
                "grad2": str(grad2),
                "scale": float(scale),
                "octaves": int(octaves),
                "lacunarity": float(lacunarity),
                "persistence": float(persistence),
                "player_x": float(player_x),
                "player_y": float(player_y),
                "view_left": float(view_left),
                "view_top": float(view_top),
                "water_level": float(pp.get("water_level", default_water_level)),
                "shore_level": float(shore_level),
                "mountain_level": float(
                    pp.get("mountain_level", default_mountain_level)
                ),
                "snowline": float(pp.get("snowline", default_snowline)),
            }
            st.download_button(
                "Download metadata (JSON)",
                data=json.dumps(meta, indent=2).encode("utf-8"),
                file_name="region.json",
                mime="application/json",
            )

            st.download_button(
                "Download tileset (ZIP)",
                data=tiles_zip_from_rgb(
                    rgb_vis,
                    z=int(pp.get("tiles_z", default_tiles_z)),
                    grid=int(pp.get("tiles_grid", default_tiles_grid)),
                    tile_size=int(pp.get("tiles_size", default_tiles_size)),
                ),
                file_name="tiles.zip",
                mime="application/zip",
            )

            st.divider()
            st.markdown("**Export chunk region (k x k chunks)**")
            with st.form("chunk_export_form", border=False):
                export_k = st.slider(
                    "k",
                    min_value=1,
                    max_value=8,
                    value=3,
                    step=1,
                    help="Exports a stitched region aligned to chunk boundaries.",
                    key="export_k",
                )
                do_export = st.form_submit_button(
                    "Build chunk export",
                    type="primary",
                    use_container_width=True,
                )

            if do_export:
                if float(chunk_world) <= 0.0:
                    st.error("Invalid chunk size.")
                else:
                    cx = int(math.floor(float(player_x) / float(chunk_world)))
                    cy = int(math.floor(float(player_y) / float(chunk_world)))
                    half = int(export_k) // 2
                    x0c = int(cx - half)
                    y0c = int(cy - half)
                    x1c = int(x0c + int(export_k) - 1)
                    y1c = int(y0c + int(export_k) - 1)

                    def stitch_region(field: str) -> np.ndarray:
                        rows: list[np.ndarray] = []
                        for yy in range(y0c, y1c + 1):
                            parts: list[np.ndarray] = []
                            for xx in range(x0c, x1c + 1):
                                parts.append(np.asarray(get_chunk(xx, yy)[field]))
                            rows.append(np.concatenate(parts, axis=1))
                        return np.concatenate(rows, axis=0)

                    region_rgb = stitch_region("rgb")
                    region_height = stitch_region("terr_river01")
                    region_biome = stitch_region("biome").astype(np.uint8)
                    region_rivers = stitch_region("rivers").astype(np.uint8)

                    meta2 = dict(meta)
                    meta2.update(
                        {
                            "export": {
                                "kind": "chunk_region",
                                "chunk_size_px": int(chunk_size_px),
                                "chunk_world": float(chunk_world),
                                "k": int(export_k),
                                "chunks": {
                                    "x0": int(x0c),
                                    "x1": int(x1c),
                                    "y0": int(y0c),
                                    "y1": int(y1c),
                                },
                                "origin": {
                                    "left": float(x0c) * float(chunk_world),
                                    "top": float(y0c) * float(chunk_world),
                                },
                            }
                        }
                    )

                    st.download_button(
                        "Download chunk region (PNG)",
                        data=_rgb_to_png_bytes(region_rgb),
                        file_name="chunk_region.png",
                        mime="image/png",
                    )
                    st.download_button(
                        "Download chunk heightmap (NPY)",
                        data=array_to_npy_bytes(region_height),
                        file_name="chunk_height.npy",
                        mime="application/octet-stream",
                    )
                    st.download_button(
                        "Download chunk biome (NPY)",
                        data=array_to_npy_bytes(region_biome),
                        file_name="chunk_biome.npy",
                        mime="application/octet-stream",
                    )
                    st.download_button(
                        "Download chunk rivers (NPY)",
                        data=array_to_npy_bytes(region_rivers),
                        file_name="chunk_rivers.npy",
                        mime="application/octet-stream",
                    )
                    st.download_button(
                        "Download chunk metadata (JSON)",
                        data=json.dumps(meta2, indent=2).encode("utf-8"),
                        file_name="chunk_region.json",
                        mime="application/json",
                    )

        c0, c1, c2, c3 = st.columns(4)
        move_dx = 0.0
        move_dy = 0.0
        if c0.button("Left (A)", use_container_width=True, key="nav_left"):
            move_dx -= 1.0
        if c1.button("Right (D)", use_container_width=True, key="nav_right"):
            move_dx += 1.0
        if c2.button("Up (W)", use_container_width=True, key="nav_up"):
            move_dy -= 1.0
        if c3.button("Down (S)", use_container_width=True, key="nav_down"):
            move_dy += 1.0

        if move_dx != 0.0 or move_dy != 0.0:
            factor = 1.0
            blocked = False
            reason = ""

            if bool(pp.get("constraints", default_constraints)):
                cx = int((int(width_render) - 1) // 2)
                cy = int((int(height_render) - 1) // 2)
                tx = int(cx + round(move_dx * float(nav_step_px)))
                ty = int(cy + round(move_dy * float(nav_step_px)))

                if 0 <= tx < int(width_render) and 0 <= ty < int(height_render):
                    h = float(terr_river01[ty, tx])
                    sl = float(s01[ty, tx])
                    water = h < float(pp.get("water_level", default_water_level))

                    if sl >= float(pp.get("slope_block", default_slope_block)):
                        blocked = True
                        reason = "too steep"
                    elif (
                        water
                        and str(pp.get("water_behavior", default_water_behavior))
                        == "Block"
                    ):
                        blocked = True
                        reason = "water"
                    else:
                        sc = float(pp.get("slope_cost", default_slope_cost))
                        if sc > 0.0:
                            factor /= 1.0 + sc * sl
                        if (
                            water
                            and str(pp.get("water_behavior", default_water_behavior))
                            == "Slow"
                        ):
                            factor *= float(pp.get("water_slow", default_water_slow))

            if blocked:
                _nav_log(
                    "move_blocked",
                    {
                        "reason": str(reason),
                        "pos": {"x": float(player_x), "y": float(player_y)},
                    },
                )
                st.warning(f"Movement blocked: {reason}.")
            else:
                _nav_log(
                    "move",
                    {
                        "from": {"x": float(player_x), "y": float(player_y)},
                        "delta": {
                            "dx": float(move_dx),
                            "dy": float(move_dy),
                            "factor": float(factor),
                        },
                    },
                )
                player_x += move_dx * step_world * factor
                player_y += move_dy * step_world * factor

        with st.form("teleport_form", border=False):
            c4, c5 = st.columns(2)
            with c4:
                px_new = st.number_input(
                    "Teleport x",
                    value=float(player_x),
                    step=float(step_world),
                    key="teleport_x",
                )
            with c5:
                py_new = st.number_input(
                    "Teleport y",
                    value=float(player_y),
                    step=float(step_world),
                    key="teleport_y",
                )
            teleport = st.form_submit_button(
                "Teleport",
                type="primary",
                use_container_width=True,
            )

        if teleport:
            _nav_log(
                "teleport",
                {
                    "from": {"x": float(player_x), "y": float(player_y)},
                    "to": {"x": float(px_new), "y": float(py_new)},
                },
            )
            player_x = float(px_new)
            player_y = float(py_new)

            pp["player_x"] = float(player_x)
            pp["player_y"] = float(player_y)
            st.session_state["practical_params"] = pp

            view_left = float(player_x) - (
                float(width_render) / (2.0 * max(float(scale), 1e-9))
            )
            view_top = float(player_y) - (
                float(height_render) / (2.0 * max(float(scale), 1e-9))
            )
            new = dict(params_for_url)
            new["player_x"] = str(float(player_x))
            new["player_y"] = str(float(player_y))
            new["offset_x"] = str(float(view_left))
            new["offset_y"] = str(float(view_top))
            _set_query_params(new)
            st.rerun()

        old_x = float(pp.get("player_x", default_player_x))
        old_y = float(pp.get("player_y", default_player_y))
        moved = (abs(player_x - old_x) > 1e-12) or (abs(player_y - old_y) > 1e-12)
        if moved:
            pp["player_x"] = float(player_x)
            pp["player_y"] = float(player_y)
            st.session_state["practical_params"] = pp

            view_left = float(player_x) - (
                float(width_render) / (2.0 * max(float(scale), 1e-9))
            )
            view_top = float(player_y) - (
                float(height_render) / (2.0 * max(float(scale), 1e-9))
            )
            new = dict(params_for_url)
            new["player_x"] = str(float(player_x))
            new["player_y"] = str(float(player_y))
            new["offset_x"] = str(float(view_left))
            new["offset_y"] = str(float(view_top))
            _set_query_params(new)
            st.rerun()

else:
    st.subheader("Learn: Perlin Noise (2D)")
    st.caption("Pick a point inside a lattice cell and inspect each intermediate step.")

    perlin = Perlin2D(seed=int(seed), grad_set=str(grad2))
    px = st.slider(
        "x",
        min_value=0.0,
        max_value=10.0,
        value=2.25,
        step=0.05,
        key="learn_px",
    )
    py = st.slider(
        "y",
        min_value=0.0,
        max_value=10.0,
        value=3.75,
        step=0.05,
        key="learn_py",
    )
    debug = perlin.debug_point(px, py)

    t_cell, t_fade, t_interp, t_scan, t_raw = st.tabs(
        ["Cell", "Fade", "Interpolation", "Scanline", "Raw"]
    )

    with t_cell:
        st.markdown("**Lattice cell + gradient vectors**")
        st.plotly_chart(perlin2d_cell_figure(debug), width="stretch", key="learn_cell")

    with t_fade:
        st.markdown("**Fade curves**")
        st.plotly_chart(
            fade_curve_figure(t_value=float(debug["relative"]["xf"]), title="fade(x)"),
            width="stretch",
            key="learn_fade_x",
        )
        st.plotly_chart(
            fade_curve_figure(t_value=float(debug["relative"]["yf"]), title="fade(y)"),
            width="stretch",
            key="learn_fade_y",
        )

    with t_interp:
        st.markdown("**Interpolation values**")
        st.write(
            {
                "u": debug["fade"]["u"],
                "v": debug["fade"]["v"],
                "x_lerp0": debug["interpolation"]["x_lerp0"],
                "x_lerp1": debug["interpolation"]["x_lerp1"],
                "noise": debug["noise"],
            }
        )

    with t_scan:
        st.markdown("**Scanline animator**")
        steps = st.slider(
            "Scan steps",
            min_value=32,
            max_value=512,
            value=256,
            step=32,
            key="learn_scan_steps",
        )
        series = scanline_series_from_debug(debug, steps=int(steps))
        st.plotly_chart(scanline_figure(series), width="stretch", key="learn_scanline")
        st.plotly_chart(
            scanline_dots_figure(series),
            width="stretch",
            key="learn_scan_dots",
        )

    with t_raw:
        st.json(debug)


with st.sidebar:
    st.divider()
    st.subheader("Performance")
    p2 = float(st.session_state.get("perf_2d_ms", 0.0))
    r2 = st.session_state.get("perf_2d_res", (0, 0))
    st.write(f"2D compute: {p2:.1f} ms  (res={r2[0]}x{r2[1]})")

    p3 = st.session_state.get("perf_3d_ms")
    r3 = st.session_state.get("perf_3d_res")
    if p3 is not None and r3 is not None:
        p3 = float(p3)
        st.write(f"3D compute: {p3:.1f} ms  (res={r3[0]}x{r3[1]})")

    fps = 1000.0 / max(p2, 1e-6)
    st.caption(f"Estimated max refresh: ~{fps:.1f} FPS (2D compute only)")

    st.divider()
    st.subheader("Validation")
    pixels = int(width) * int(height)
    if pixels >= 1024 * 1024:
        st.warning(
            "Large 2D resolution selected. Consider Apply mode or lower Quality "
            "while dragging."
        )
    elif pixels >= 512 * 512:
        st.info("Moderate 2D resolution. Live preview may feel heavier.")

    if bool(live_drag) and str(quality) == "Full":
        st.warning("Quality=Full with live dragging can rerun frequently.")

    if rendering_preview:
        st.caption("Preview LOD is active; release the slider to refine.")
