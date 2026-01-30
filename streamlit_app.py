from __future__ import annotations

import json
import math
from urllib.parse import urlencode

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from perlin.noise_2d import (
    Perlin2D,
    domain_warp2,
    fbm2,
    ridged2,
    tileable2d,
    turbulence2,
)
from perlin.value_noise_2d import ValueNoise2D
from viz.export import array_to_npy_bytes, array_to_png_bytes, heightmap_to_obj_bytes
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
if default_page not in {"Explore", "Learn"}:
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

default_colorscale = _qp_get("colorscale", "Viridis")
if default_colorscale not in _COLOR_SCALES:
    default_colorscale = "Viridis"

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
    basis = str(basis)
    grad_set = str(grad_set)
    if basis == "perlin":
        noise = Perlin2D(seed=seed, grad_set=grad_set)
    elif basis == "value":
        noise = ValueNoise2D(seed=seed)
    else:
        raise ValueError(f"unknown basis: {basis}")

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
                octaves=octaves,
                lacunarity=lacunarity,
                persistence=persistence,
            )
        if variant == "turbulence":
            return turbulence2(
                noise,
                xx,
                yy,
                octaves=octaves,
                lacunarity=lacunarity,
                persistence=persistence,
            )
        if variant == "ridged":
            return ridged2(
                noise,
                xx,
                yy,
                octaves=octaves,
                lacunarity=lacunarity,
                persistence=persistence,
            )
        if variant == "domain_warp":
            return domain_warp2(
                noise,
                xx,
                yy,
                octaves=octaves,
                lacunarity=lacunarity,
                persistence=persistence,
                warp_amp=warp_amp,
                warp_scale=warp_scale,
                warp_octaves=warp_octaves,
                warp_lacunarity=lacunarity,
                warp_persistence=persistence,
            )
        raise ValueError(f"unknown variant: {variant}")

    z = (
        tileable2d(base, xg, yg, period_x=period_x, period_y=period_y)
        if tileable
        else base(xg, yg)
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
        "Page",
        ["Explore", "Learn"],
        index=0 if default_page == "Explore" else 1,
        label_visibility="collapsed",
    )

    st.divider()
    st.header("Parameters")
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
        "Width", min_value=64, max_value=1024, value=int(default_width), step=64
    )
    height = st.slider(
        "Height", min_value=64, max_value=1024, value=int(default_height), step=64
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
        "basis": str(basis),
        "grad2": str(grad2),
        "noise": str(noise_variant),
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
        st.plotly_chart(_heatmap(z01, colorscale=str(colorscale)), width="stretch")
        if show_hist:
            st.plotly_chart(_histogram(z01), width="stretch")

        with st.expander("Compare: Perlin vs Value noise"):
            perlin_z = _noise_map(
                seed=int(seed),
                basis="perlin",
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
                offset_x=float(offset_x),
                offset_y=float(offset_y),
                normalize=bool(normalize),
                tileable=bool(tileable),
            )
            value_z = _noise_map(
                seed=int(seed),
                basis="value",
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
                offset_x=float(offset_x),
                offset_y=float(offset_y),
                normalize=bool(normalize),
                tileable=bool(tileable),
            )

            col0, col1 = st.columns(2)
            with col0:
                st.markdown("**Perlin (gradient)**")
                st.plotly_chart(
                    _heatmap(perlin_z, colorscale=str(colorscale)),
                    width="stretch",
                )
            with col1:
                st.markdown("**Value noise**")
                st.plotly_chart(
                    _heatmap(value_z, colorscale=str(colorscale)),
                    width="stretch",
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
                    offset_x=float(offset_x),
                    offset_y=float(offset_y),
                    normalize=bool(normalize),
                    tileable=bool(tileable),
                )
                axis4_z = _noise_map(
                    seed=int(seed),
                    basis="perlin",
                    grad_set="axis4",
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
                    offset_x=float(offset_x),
                    offset_y=float(offset_y),
                    normalize=bool(normalize),
                    tileable=bool(tileable),
                )

                col0, col1 = st.columns(2)
                with col0:
                    st.markdown("**diag8**")
                    st.plotly_chart(
                        _heatmap(diag8_z, colorscale=str(colorscale)),
                        width="stretch",
                    )
                with col1:
                    st.markdown("**axis4**")
                    st.plotly_chart(
                        _heatmap(axis4_z, colorscale=str(colorscale)),
                        width="stretch",
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
                    width=int(width),
                    height=int(height),
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
                        _heatmap(base_z, colorscale=str(colorscale)),
                        width="stretch",
                    )
                with col1:
                    st.markdown("**Warped**")
                    st.plotly_chart(
                        _heatmap(z01, colorscale=str(colorscale)),
                        width="stretch",
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
            basis=str(basis),
            grad_set=str(grad2),
            width=int(res3d),
            height=int(res3d),
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
        surfacecolor = _slope01(z3d) if shade_mode == "Slope" else None
        st.plotly_chart(
            _surface(
                z3d,
                z_scale=float(z_scale),
                colorscale=str(colorscale),
                surfacecolor=surfacecolor,
            ),
            width="stretch",
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
else:
    st.subheader("Step-by-step Generation (Work in progress)")
    st.write(
        (
            "This page will evolve into a full, inspectable breakdown of Perlin noise "
            "generation: grid corners, gradient vectors, dot products, fade curve "
            "values, interpolation, and octaves."
        )
    )

    perlin = Perlin2D(seed=int(seed), grad_set=str(grad2))
    st.markdown("**Inspect a single point**")
    px = st.slider("x", min_value=0.0, max_value=10.0, value=2.25, step=0.05)
    py = st.slider("y", min_value=0.0, max_value=10.0, value=3.75, step=0.05)
    debug = perlin.debug_point(px, py)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Cell + gradients**")
        st.plotly_chart(perlin2d_cell_figure(debug), width="stretch")

    with col_b:
        st.markdown("**Fade curves**")
        st.plotly_chart(
            fade_curve_figure(t_value=float(debug["relative"]["xf"]), title="fade(x)"),
            width="stretch",
        )
        st.plotly_chart(
            fade_curve_figure(t_value=float(debug["relative"]["yf"]), title="fade(y)"),
            width="stretch",
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
    st.plotly_chart(scanline_figure(series), width="stretch")
    st.plotly_chart(scanline_dots_figure(series), width="stretch")

    with st.expander("Raw debug JSON"):
        st.json(debug)
