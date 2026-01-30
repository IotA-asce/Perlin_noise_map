from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_FRONTEND_DIST = Path(__file__).parent / "hotkeys_frontend" / "dist"

_hotkeys = components.declare_component(
    "hotkeys",
    path=str(_FRONTEND_DIST),
)


def hotkeys(
    *,
    enabled: bool,
    allowed: list[str],
    key: str,
) -> dict | None:
    """Capture keypresses from the browser and return events.

    Returns a dict like: {"key": "r", "code": "KeyR", "ts": 1234567890}.
    """

    return _hotkeys(
        enabled=bool(enabled),
        allowed=[str(k) for k in allowed],
        key=key,
        default=None,
    )
