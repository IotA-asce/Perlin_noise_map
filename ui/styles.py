from __future__ import annotations

import streamlit as st

APP_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap');

/* Global typography */
html, body, [class*="st-"] {
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system,
    BlinkMacSystemFont, "Segoe UI", sans-serif;
}

h1, h2, h3, h4 {
  font-family: "Space Grotesk", ui-sans-serif, system-ui, -apple-system,
    BlinkMacSystemFont, "Segoe UI", sans-serif;
  letter-spacing: -0.02em;
}

/* Softer page background */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(
      1200px 800px at 20% 10%,
      rgba(15, 118, 110, 0.10),
      rgba(0,0,0,0) 60%
    ),
    radial-gradient(
      900px 700px at 90% 20%,
      rgba(245, 158, 11, 0.08),
      rgba(0,0,0,0) 55%
    ),
    linear-gradient(180deg, rgba(255,255,255,0.35), rgba(0,0,0,0) 45%);
}

/* Sidebar polish */
[data-testid="stSidebar"] {
  border-right: 1px solid rgba(17, 24, 39, 0.08);
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
  gap: 0.75rem;
}

/* Tighten default container padding a bit */
[data-testid="stAppViewContainer"] > .main {
  padding-top: 1.25rem;
}

/* Subtle card styling for expanders */
details {
  border: 1px solid rgba(17, 24, 39, 0.08) !important;
  border-radius: 12px !important;
  background: rgba(255, 255, 255, 0.65) !important;
  backdrop-filter: blur(6px);
}

details summary {
  padding: 0.35rem 0.75rem !important;
}
"""


def inject_global_styles() -> None:
    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)
