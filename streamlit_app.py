from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from perlin.noise_2d import Perlin2D, fbm2

st.set_page_config(
    page_title="Perlin Noise Map",
    page_icon="~",
    layout="wide",
)


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
) -> np.ndarray:
    perlin = Perlin2D(seed=seed)

    scale = max(scale, 1e-9)
    xs = (np.arange(width, dtype=np.float64) / scale) + offset_x
    ys = (np.arange(height, dtype=np.float64) / scale) + offset_y
    xg, yg = np.meshgrid(xs, ys)

    z = fbm2(
        perlin,
        xg,
        yg,
        octaves=octaves,
        lacunarity=lacunarity,
        persistence=persistence,
    )

    # Normalize for display.
    zmin = float(np.min(z))
    zmax = float(np.max(z))
    if math.isclose(zmin, zmax):
        return np.zeros_like(z)
    return (z - zmin) / (zmax - zmin)


def _heatmap(z01: np.ndarray) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=z01,
            colorscale="Viridis",
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


def _surface(z01: np.ndarray, *, z_scale: float) -> go.Figure:
    fig = go.Figure(
        data=go.Surface(
            z=z01 * z_scale,
            colorscale="Viridis",
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


st.title("Perlin Noise Map")
st.caption("Interactive Perlin noise explorer (2D + 3D) with a learning-first focus.")


with st.sidebar:
    st.header("Navigation")
    page = st.radio("", ["Explore", "Learn"], label_visibility="collapsed")

    st.divider()
    st.header("Parameters")
    seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=0, step=1)
    scale = st.slider(
        "Scale (bigger = smoother)", min_value=5.0, max_value=600.0, value=120.0
    )
    octaves = st.slider("Octaves", min_value=1, max_value=10, value=4)
    lacunarity = st.slider(
        "Lacunarity", min_value=1.0, max_value=4.0, value=2.0, step=0.05
    )
    persistence = st.slider(
        "Persistence", min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )

    st.divider()
    st.subheader("Viewport")
    width = st.slider("Width", min_value=64, max_value=512, value=256, step=32)
    height = st.slider("Height", min_value=64, max_value=512, value=256, step=32)
    offset_x = st.slider(
        "Offset X", min_value=-50.0, max_value=50.0, value=0.0, step=0.5
    )
    offset_y = st.slider(
        "Offset Y", min_value=-50.0, max_value=50.0, value=0.0, step=0.5
    )

    st.divider()
    st.subheader("3D")
    z_scale = st.slider(
        "Height scale", min_value=0.0, max_value=200.0, value=80.0, step=5.0
    )

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
)


if page == "Explore":
    tab2d, tab3d = st.tabs(["2D Map", "3D Terrain"])

    with tab2d:
        st.subheader("2D Noise Map")
        st.plotly_chart(_heatmap(z01), use_container_width=True)

    with tab3d:
        st.subheader("3D Heightmap")
        st.plotly_chart(_surface(z01, z_scale=float(z_scale)), use_container_width=True)
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
    st.json(debug)
