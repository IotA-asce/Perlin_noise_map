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
