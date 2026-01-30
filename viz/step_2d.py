from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from perlin.core import fade


def perlin2d_cell_figure(debug: dict) -> go.Figure:
    rel = debug["relative"]
    xf = float(rel["xf"])
    yf = float(rel["yf"])

    corners = debug["corners"]
    corners_xy = {
        "c00": (0.0, 0.0),
        "c10": (1.0, 0.0),
        "c01": (0.0, 1.0),
        "c11": (1.0, 1.0),
    }

    fig = go.Figure()

    # Cell outline.
    fig.add_trace(
        go.Scatter(
            x=[0, 1, 1, 0, 0],
            y=[0, 0, 1, 1, 0],
            mode="lines",
            line=dict(color="rgba(255,255,255,0.6)", width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Corners.
    fig.add_trace(
        go.Scatter(
            x=[0, 1, 0, 1],
            y=[0, 0, 1, 1],
            mode="markers+text",
            marker=dict(size=10, color="rgba(255,255,255,0.9)"),
            text=["(0,0)", "(1,0)", "(0,1)", "(1,1)"],
            textposition="top center",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Point inside the cell.
    fig.add_trace(
        go.Scatter(
            x=[xf],
            y=[yf],
            mode="markers+text",
            marker=dict(size=12, color="#ffb000"),
            text=[f"(x={xf:.3f}, y={yf:.3f})"],
            textposition="bottom center",
            showlegend=False,
        )
    )

    # Dot product labels near corners.
    dot_x = []
    dot_y = []
    dot_txt = []
    for key, (cx, cy) in corners_xy.items():
        c = corners[key]
        dot = float(c["dot"])
        dot_x.append(cx + 0.05)
        dot_y.append(cy + 0.05)
        dot_txt.append(f"{key}: dot={dot:.3f}")

    fig.add_trace(
        go.Scatter(
            x=dot_x,
            y=dot_y,
            mode="text",
            text=dot_txt,
            textfont=dict(color="rgba(255,255,255,0.9)", size=12),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Gradient arrows (annotations).
    arrow_scale = 0.35
    annotations = []
    for key, (cx, cy) in corners_xy.items():
        c = corners[key]
        gx = float(c["gx"])
        gy = float(c["gy"])
        x_end = cx + gx * arrow_scale
        y_end = cy + gy * arrow_scale
        annotations.append(
            dict(
                x=x_end,
                y=y_end,
                ax=cx,
                ay=cy,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="rgba(0, 200, 255, 0.9)",
                text="",
            )
        )

    fig.update_layout(
        annotations=annotations,
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(range=[-0.15, 1.15], visible=False)
    fig.update_yaxes(range=[-0.15, 1.15], visible=False, scaleanchor="x")
    return fig


def fade_curve_figure(*, t_value: float, title: str) -> go.Figure:
    t = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    y = fade(t)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.8)", width=2),
            showlegend=False,
        )
    )

    tv = float(np.clip(t_value, 0.0, 1.0))
    fv = float(fade(np.array(tv, dtype=np.float64)))
    fig.add_trace(
        go.Scatter(
            x=[tv],
            y=[fv],
            mode="markers+text",
            marker=dict(size=10, color="#ffb000"),
            text=[f"t={tv:.3f}\nfade={fv:.3f}"],
            textposition="top center",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    return fig


def scanline_series_from_debug(debug: dict, *, steps: int = 256) -> dict:
    """Build a 1D scanline (x in [0, 1)) within the current lattice cell.

    Uses the corner gradients from `debug` and computes dot products + interpolation
    across x while keeping y fixed.
    """

    steps = int(steps)
    steps = max(8, steps)

    xf0 = float(debug["relative"]["xf"])
    yf = float(debug["relative"]["yf"])
    v = float(debug["fade"]["v"])

    t = np.linspace(0.0, 1.0, steps, endpoint=False, dtype=np.float64)
    u = fade(t)

    c00 = debug["corners"]["c00"]
    c10 = debug["corners"]["c10"]
    c01 = debug["corners"]["c01"]
    c11 = debug["corners"]["c11"]

    gx00, gy00 = float(c00["gx"]), float(c00["gy"])
    gx10, gy10 = float(c10["gx"]), float(c10["gy"])
    gx01, gy01 = float(c01["gx"]), float(c01["gy"])
    gx11, gy11 = float(c11["gx"]), float(c11["gy"])

    y0 = yf
    y1 = yf - 1.0
    x0 = t
    x1 = t - 1.0

    d00 = gx00 * x0 + gy00 * y0
    d10 = gx10 * x1 + gy10 * y0
    d01 = gx01 * x0 + gy01 * y1
    d11 = gx11 * x1 + gy11 * y1

    x_lerp0 = d00 + u * (d10 - d00)
    x_lerp1 = d01 + u * (d11 - d01)
    noise = x_lerp0 + v * (x_lerp1 - x_lerp0)

    return {
        "t": t,
        "xf": xf0,
        "dots": {"d00": d00, "d10": d10, "d01": d01, "d11": d11},
        "lerp": {"x_lerp0": x_lerp0, "x_lerp1": x_lerp1, "noise": noise},
    }


def scanline_figure(series: dict) -> go.Figure:
    t = series["t"]
    xf = float(series["xf"])
    noise = series["lerp"]["noise"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=noise,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.85)", width=2),
            name="noise",
        )
    )
    fig.add_vline(x=xf, line_width=2, line_dash="dot", line_color="#ffb000")

    fig.update_layout(
        title="Scanline: noise(x, y_fixed) within one cell",
        margin=dict(l=0, r=0, t=40, b=0),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(range=[0, 1])
    return fig


def scanline_dots_figure(series: dict) -> go.Figure:
    t = series["t"]
    xf = float(series["xf"])
    dots = series["dots"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=dots["d00"], mode="lines", name="dot c00"))
    fig.add_trace(go.Scatter(x=t, y=dots["d10"], mode="lines", name="dot c10"))
    fig.add_trace(go.Scatter(x=t, y=dots["d01"], mode="lines", name="dot c01"))
    fig.add_trace(go.Scatter(x=t, y=dots["d11"], mode="lines", name="dot c11"))
    fig.add_vline(x=xf, line_width=2, line_dash="dot", line_color="#ffb000")

    fig.update_layout(
        title="Scanline: corner dot products vs x",
        margin=dict(l=0, r=0, t=40, b=0),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(range=[0, 1])
    return fig
