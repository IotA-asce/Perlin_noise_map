from __future__ import annotations

import json
import math
from urllib.parse import urlencode

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from perlin.noise_2d import Perlin2D, fbm2, tileable_fbm2
from viz.export import array_to_png_bytes
from viz.step_2d import (
    fade_curve_figure,
    perlin2d_cell_figure,
    scanline_dots_figure,
    scanline_figure,
    scanline_series_from_debug,
)

st.set_page_config(
    page_title="Perlin Noise Map",
    page_icon="~",
    layout="wide",
)


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


_COLOR_SCALES = ["Viridis", "Cividis", "Turbo", "IceFire", "Earth"]

default_page = _qp_get("page", "Explore")
if default_page not in {"Explore", "Learn"}:
    default_page = "Explore"

default_seed = _qp_int("seed", 0, min_value=0, max_value=2**31 - 1)
default_scale = _qp_float("scale", 120.0, min_value=5.0, max_value=600.0)
default_octaves = _qp_int("octaves", 4, min_value=1, max_value=10)
default_lacunarity = _qp_float("lacunarity", 2.0, min_value=1.0, max_value=4.0)
default_persistence = _qp_float("persistence", 0.5, min_value=0.0, max_value=1.0)
default_width = _qp_int("width", 256, min_value=64, max_value=512)
default_height = _qp_int("height", 256, min_value=64, max_value=512)
default_offset_x = _qp_float("offset_x", 0.0, min_value=-50.0, max_value=50.0)
default_offset_y = _qp_float("offset_y", 0.0, min_value=-50.0, max_value=50.0)
default_z_scale = _qp_float("z_scale", 80.0, min_value=0.0, max_value=200.0)
default_res3d = _qp_int("res3d", 128, min_value=64, max_value=256)

default_normalize = _qp_bool("normalize", True)
default_tileable = _qp_bool("tileable", False)
default_show_hist = _qp_bool("show_hist", False)

default_colorscale = _qp_get("colorscale", "Viridis")
if default_colorscale not in _COLOR_SCALES:
    default_colorscale = "Viridis"

default_shade = _qp_get("shade", "Height")
if default_shade not in {"Height", "Slope"}:
    default_shade = "Height"


@st.cache_data(show_spinner=False)
def _noise_map(
    *,
    seed: int,
    width: int,
    height: int,
    scale: float,
    octaves: int,
    lacunarity: float,
    persistence: float,
    offset_x: float,
    offset_y: float,
    normalize: bool,
    tileable: bool,
) -> np.ndarray:
    perlin = Perlin2D(seed=seed)

    scale = max(scale, 1e-9)
    if tileable:
        period_x = (float(width) - 1.0) / scale
        period_y = (float(height) - 1.0) / scale
        xs = np.linspace(offset_x, offset_x + period_x, width, dtype=np.float64)
        ys = np.linspace(offset_y, offset_y + period_y, height, dtype=np.float64)
    else:
        xs = (np.arange(width, dtype=np.float64) / scale) + offset_x
        ys = (np.arange(height, dtype=np.float64) / scale) + offset_y
    xg, yg = np.meshgrid(xs, ys)

    if tileable:
        z = tileable_fbm2(
            perlin,
            xg,
            yg,
            period_x=period_x,
            period_y=period_y,
            octaves=octaves,
            lacunarity=lacunarity,
            persistence=persistence,
        )
    else:
        z = fbm2(
            perlin,
            xg,
            yg,
            octaves=octaves,
            lacunarity=lacunarity,
            persistence=persistence,
        )

    if not normalize:
        return z

    # Normalize for display.
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    if math.isclose(zmin, zmax):
        return np.zeros_like(z)
    return (z - zmin) / (zmax - zmin)


def _heatmap(z: np.ndarray, *, colorscale: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            colorscale=colorscale,
            showscale=False,
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=520,
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


st.title("Perlin Noise Map")
st.caption("Interactive Perlin noise explorer (2D + 3D) with a learning-first focus.")


with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "",
        ["Explore", "Learn"],
        index=0 if default_page == "Explore" else 1,
        label_visibility="collapsed",
    )

    st.divider()
    st.header("Parameters")
    seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=2**31 - 1,
        value=int(default_seed),
        step=1,
    )
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
    show_hist = st.toggle("Show histogram", value=bool(default_show_hist))

    st.divider()
    st.subheader("Viewport")
    width = st.slider(
        "Width", min_value=64, max_value=512, value=int(default_width), step=32
    )
    height = st.slider(
        "Height", min_value=64, max_value=512, value=int(default_height), step=32
    )
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
    z_scale = st.slider(
        "Height scale",
        min_value=0.0,
        max_value=200.0,
        value=float(default_z_scale),
        step=5.0,
    )

    st.divider()
    st.subheader("Share")
    params_for_url = {
        "page": str(page),
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
        "normalize": "1" if bool(normalize) else "0",
        "tileable": "1" if bool(tileable) else "0",
        "colorscale": str(colorscale),
        "show_hist": "1" if bool(show_hist) else "0",
    }
    if st.button("Update URL with current settings"):
        _set_query_params(params_for_url)

    st.code(f"?{urlencode(params_for_url)}", language="text")

z01 = _noise_map(
    seed=int(seed),
    width=int(width),
    height=int(height),
    scale=float(scale),
    octaves=int(octaves),
    lacunarity=float(lacunarity),
    persistence=float(persistence),
    offset_x=float(offset_x),
    offset_y=float(offset_y),
    normalize=bool(normalize),
    tileable=bool(tileable),
)

zmin = float(np.min(z01))
zmax = float(np.max(z01))


if page == "Explore":
    tab2d, tab3d = st.tabs(["2D Map", "3D Terrain"])

    with tab2d:
        st.subheader("2D Noise Map")
        st.caption(f"min={zmin:.4f}, max={zmax:.4f}")
        st.plotly_chart(
            _heatmap(z01, colorscale=str(colorscale)), use_container_width=True
        )
        if show_hist:
            st.plotly_chart(_histogram(z01), use_container_width=True)

        with st.expander("Export"):
            params = {
                "page": str(page),
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
                "show_hist": bool(show_hist),
                "res3d": int(res3d),
                "shade": str(shade_mode),
                "z_scale": float(z_scale),
            }
            st.download_button(
                "Download PNG (grayscale)",
                data=array_to_png_bytes(z01),
                file_name="perlin_map.png",
                mime="image/png",
            )
            st.download_button(
                "Download params.json",
                data=json.dumps(params, indent=2, sort_keys=True),
                file_name="params.json",
                mime="application/json",
            )
            st.caption(
                "To share this exact state: click 'Update URL with current settings' "
                "in the sidebar, then copy the browser URL."
            )

    with tab3d:
        st.subheader("3D Heightmap")
        st.caption(f"resolution={int(res3d)}x{int(res3d)}")

        z3d = _noise_map(
            seed=int(seed),
            width=int(res3d),
            height=int(res3d),
            scale=float(scale),
            octaves=int(octaves),
            lacunarity=float(lacunarity),
            persistence=float(persistence),
            offset_x=float(offset_x),
            offset_y=float(offset_y),
            normalize=bool(normalize),
            tileable=bool(tileable),
        )
        surfacecolor = _slope01(z3d) if shade_mode == "Slope" else None
        st.plotly_chart(
            _surface(
                z3d,
                z_scale=float(z_scale),
                colorscale=str(colorscale),
                surfacecolor=surfacecolor,
            ),
            use_container_width=True,
        )
else:
    st.subheader("Step-by-step Generation (Work in progress)")
    st.write(
        (
            "This page will evolve into a full, inspectable breakdown of Perlin noise "
            "generation: grid corners, gradient vectors, dot products, fade curve "
            "values, interpolation, and octaves."
        )
    )

    perlin = Perlin2D(seed=int(seed))
    st.markdown("**Inspect a single point**")
    px = st.slider("x", min_value=0.0, max_value=10.0, value=2.25, step=0.05)
    py = st.slider("y", min_value=0.0, max_value=10.0, value=3.75, step=0.05)
    debug = perlin.debug_point(px, py)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Cell + gradients**")
        st.plotly_chart(perlin2d_cell_figure(debug), use_container_width=True)

    with col_b:
        st.markdown("**Fade curves**")
        st.plotly_chart(
            fade_curve_figure(t_value=float(debug["relative"]["xf"]), title="fade(x)"),
            use_container_width=True,
        )
        st.plotly_chart(
            fade_curve_figure(t_value=float(debug["relative"]["yf"]), title="fade(y)"),
            use_container_width=True,
        )

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

    st.markdown("**Scanline animator**")
    steps = st.slider("Scan steps", min_value=32, max_value=512, value=256, step=32)
    series = scanline_series_from_debug(debug, steps=int(steps))
    st.plotly_chart(scanline_figure(series), use_container_width=True)
    st.plotly_chart(scanline_dots_figure(series), use_container_width=True)

    with st.expander("Raw debug JSON"):
        st.json(debug)
