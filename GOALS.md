# Project Goals

This document captures what this repo is trying to become: a practical and visual deep-dive into Perlin noise (and close relatives), built around a Python implementation and a Streamlit-based interactive explorer.

## North Star

Make Perlin noise understandable.

Not just "generate noise", but make each mathematical decision visible: gradients, dot products, smoothing, interpolation, octave composition, and why parameter changes look the way they do.

## Core Goals

- [ ] Step-by-step visualization of how Perlin noise is generated.
- [ ] A 3D map rendered in real time with interactivity.
- [ ] Clean, well-tested Perlin implementations (1D/2D/3D) with deterministic seeding.
- [ ] A learning-first codebase: readable, modular, and easy to extend.

## Milestones

### Milestone 0: Project Scaffolding

- [x] Streamlit app skeleton with navigation: "Learn" (step-by-step) and "Explore" (parameter playground).
- [x] Package layout (`perlin/`) separated from UI.
- [x] Basic test harness (pytest) and formatting/linting (ruff).

### Milestone 1: Correctness-First Perlin (Improved Perlin)

- [x] Implement fade, lerp, gradient hashing, permutation table.
- [x] Implement `noise1d`, `noise2d`, `noise3d`.
- [x] Add reproducibility tests: same seed + params => same output.
- [x] Add sanity tests: output bounds, continuity checks, and simple reference fixtures.

### Milestone 2: Step-by-Step Visualization (2D)

- [x] Visualize lattice/grid points and gradient vectors.
- [x] Visualize the dot product contributions at each corner.
- [x] Visualize fade curves and interpolation weights.
- [x] Build a "single cell inspector" that zooms into one lattice cell and shows intermediate values.
- [x] Build a "scanline animator" that shows how moving input changes contributions over time.

### Milestone 3: 2D Map Explorer

- [x] Interactive controls: seed, frequency/scale, octaves, lacunarity, persistence.
- [x] Options: normalize vs raw, color maps, histogram of values.
- [ ] Tileable/seamless mode for texture generation.
- [ ] Export: PNG (2D), parameter JSON, and reproducible "share link" state (Streamlit query params).

### Milestone 4: Real-Time 3D Terrain

- [ ] Render a heightmap surface with interactive camera controls (rotate/zoom/pan).
- [ ] Real-time parameter updates without sluggishness (caching, downsample/LOD, incremental updates).
- [ ] Optional shading modes: flat, smooth, slope/curvature based coloring.
- [ ] Export: mesh (OBJ/PLY) or heightmap for external tools.

### Milestone 5: Beyond Basic Perlin

- [ ] Fractal Brownian Motion (fBm), turbulence, ridged multifractal.
- [ ] Domain warping (single + multi-stage) with visual explanation.
- [ ] Alternative gradient sets and artifacts exploration (grid-alignment, directional bias).
- [ ] Comparisons: value noise vs gradient noise; Perlin vs Simplex (optional).

## Performance Targets

- [ ] 2D: interactive at typical sizes (e.g., 256x256 to 1024x1024) with responsive UI.
- [ ] 3D: "feels real-time" for moderate grids (e.g., 128x128 to 256x256) with smooth camera interaction.

## Non-Goals (For Now)

- [ ] Procedural world streaming / chunk systems at game-engine scale.
- [ ] GPU-only implementations as the default (OK as an optional later track).

## Definition of Done (For This Repo)

- [ ] The app can teach Perlin noise step-by-step to someone who knows basic vectors and interpolation.
- [ ] The implementation is test-covered, deterministic, and modular.
- [ ] The 3D view is interactive and updates quickly enough to encourage experimentation.
