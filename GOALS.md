# Project Goals

This document captures what this repo is trying to become: a practical and visual deep-dive into Perlin noise (and close relatives), built around a Python implementation and a Streamlit-based interactive explorer.

## North Star

Make Perlin noise understandable.

Not just "generate noise", but make each mathematical decision visible: gradients, dot products, smoothing, interpolation, octave composition, and why parameter changes look the way they do.

## Core Goals

- [x] Step-by-step visualization of how Perlin noise is generated.
- [x] A 3D map rendered in real time with interactivity.
- [x] Clean, well-tested Perlin implementations (1D/2D/3D) with deterministic seeding.
- [x] A learning-first codebase: readable, modular, and easy to extend.

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
- [x] Tileable/seamless mode for texture generation.
- [x] Export: PNG (2D), parameter JSON, and reproducible "share link" state (Streamlit query params).

### Milestone 4: Real-Time 3D Terrain

- [x] Render a heightmap surface with interactive camera controls (rotate/zoom/pan).
- [x] Real-time parameter updates without sluggishness (caching, downsample/LOD, incremental updates).
- [x] Optional shading modes: flat, smooth, slope/curvature based coloring.
- [x] Export: mesh (OBJ/PLY) or heightmap for external tools.

### Milestone 5: Beyond Basic Perlin

- [x] Fractal Brownian Motion (fBm), turbulence, ridged multifractal.
- [x] Domain warping (single + multi-stage) with visual explanation.
- [x] Alternative gradient sets and artifacts exploration (grid-alignment, directional bias).
- [x] Comparisons: value noise vs gradient noise; Perlin vs Simplex (optional).

## UI Enhancements (Backlog)

- [x] Establish a cohesive visual theme (typography, spacing scale, colors, surfaces, shadows).
- [x] Add a modern app layout with a persistent header, compact sidebar, and clear section hierarchy.
- [x] Create a "Live preview" mode that updates maps continuously while adjusting controls (with throttling).
- [x] Add a "Apply" / "Pause updates" mode for heavy configs (batch updates via form) as a performance escape hatch.
- [x] Add a performance HUD: last render time, cache hits, resolution, and estimated FPS.
- [x] Add responsive layout rules for mobile/tablet (stacking, smaller charts, collapsible panels).
- [x] Improve chart framing: titles, legends, axis hiding, consistent margins, and value readouts.
- [x] Add a colorbar + value probe (hover shows value, min/max/mean shown persistently).
- [x] Add tasteful motion: initial load reveal, subtle transitions for chart updates, no jitter.
- [x] Add presets: "Terrain", "Marble", "Clouds", "Islands", "Ridged mountains", etc.
- [x] Add parameter bookmarking in-session (save/restore snapshots) beyond URL query params.
- [x] Add an inline diff mode: compare A vs B settings side-by-side with synchronized controls.
- [x] Add an interactive cross-section tool: pick a row/column and plot the 1D slice.
- [x] Add a minimap / viewport navigator for offsets (drag to pan, wheel to zoom).
- [ ] Add keyboard shortcuts for common actions (reset, randomize seed, toggle histogram, toggle 3D).
- [x] Add input validation UX: inline warnings, clamping indicators, and disabled states when invalid.
- [x] Add a "quality" selector: LOD for 2D/3D with progressive refinement (fast preview then refine).
- [x] Add a dedicated "Learn" design: step cards, breadcrumbs, and a consistent inspector panel.
- [x] Replace the most latency-sensitive sliders with a custom Streamlit component that emits values while dragging.
- [x] Improve exports UX: export panel with format options, naming, and a preview of what will be exported.

## Performance Targets

- [x] 2D: interactive at typical sizes (e.g., 256x256 to 1024x1024) with responsive UI.
- [x] 3D: "feels real-time" for moderate grids (e.g., 128x128 to 256x256) with smooth camera interaction.

## Non-Goals (For Now)

- [ ] Procedural world streaming / chunk systems at game-engine scale.
- [ ] GPU-only implementations as the default (OK as an optional later track).

## Definition of Done (For This Repo)

- [x] The app can teach Perlin noise step-by-step to someone who knows basic vectors and interpolation.
- [x] The implementation is test-covered, deterministic, and modular.
- [x] The 3D view is interactive and updates quickly enough to encourage experimentation.
