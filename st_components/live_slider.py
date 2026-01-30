from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_FRONTEND_DIST = Path(__file__).parent / "live_slider_frontend" / "dist"

_live_slider = components.declare_component(
    "live_slider",
    path=str(_FRONTEND_DIST),
)


def live_slider(
    *,
    label: str,
    min_value: float,
    max_value: float,
    value: float,
    step: float,
    throttle_ms: int = 60,
    key: str,
) -> tuple[float, bool]:
    """A slider that emits values continuously while dragging.

    Returns (value, is_final). is_final is True when the user releases the handle.
    """

    payload = _live_slider(
        label=label,
        min_value=float(min_value),
        max_value=float(max_value),
        value=float(value),
        step=float(step),
        throttle_ms=int(throttle_ms),
        key=key,
        default={"value": float(value), "is_final": True},
    )

    if payload is None:
        return float(value), True

    try:
        v = float(payload.get("value", value))
    except (TypeError, ValueError, AttributeError):
        v = float(value)

    try:
        is_final = bool(payload.get("is_final", True))
    except Exception:
        is_final = True

    return v, is_final
