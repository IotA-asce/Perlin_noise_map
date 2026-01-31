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

### Milestone 6: Practical Usage - Terrain Map (Biomes)

- [x] Add a new "Practical" page (or new Explore tabs) focused on worldgen outputs, not raw noise.
- [x] Terrain classification from height: water/shore/land/hills/mountains with user-set thresholds.
- [x] Height-based color ramp per region (e.g. shallow->deep water; lowland->highland; rock->snow).
- [x] Slope-aware snow/ice placement: snowline depends on altitude, with slope reducing accumulation.
- [x] Add hillshade / lighting controls (azimuth, altitude) to make terrain readable.
- [x] Add a 3D terrain view for the practical map (surface + water plane).
- [x] Inspectability: show intermediate layers (base height, slope, masks, final composite) with toggles.

### Milestone 7: Hydrology - Rivers And Lakes

- [x] Flow direction + flow accumulation from heightmap (deterministic; stable for same inputs).
- [x] River network extraction: thresholded accumulation -> river mask; adjustable density.
- [x] River carving (simple channel deepening) + bank shaping; keep the underlying "raw" height visible.
- [x] Lake filling: local minima handling (basic depression fill) and lake level visualization.
- [x] River showcase view: overlay rivers on terrain, plus a dedicated "flow" debug view.

### Milestone 8: Weathering And Erosion (Learning-First)

- [x] Thermal erosion (talus) pass with before/after + iteration scrubber.
- [x] Simple hydraulic erosion pass with rainfall/evaporation and sediment transport visualization.
- [x] Show erosion/weathering as an animation (playback + scrubber) with deterministic frame generation.
- [x] Coastline smoothing + beach deposition as an optional post-process.
- [x] Performance: support low-iteration preview + refine mode; cache deterministic passes.

### Milestone 9: World Navigation And "Endless" Map (Stretch)

- [x] Chunked generation: infinite coordinates via (chunk_x, chunk_y) with deterministic seeding.
- [x] Seam handling: match chunk edges (tileable/chunk-stitch strategy) and provide a debug overlay.
- [x] "Player" marker that can move (WASD/hotkeys/buttons) and updates the view offset.
- [x] Teleport sets player coordinates, recenters viewport, and updates share URL state.
- [x] Add navigator debug logs for movement/teleport/chunk actions (helps diagnose state updates).
- [x] Basic collision/constraints demo: water is slow/blocked; steep slopes cost more (optional).
- [x] Export a region around the player (heightmap + biome map + river mask) for external tools.

### Milestone 10: Extras (Nice Practical Additions)

- [x] Biomes from climate: temperature + moisture maps (noise-driven) -> biome palette (desert/forest/tundra).
- [x] Vegetation/rocks placement as points (Poisson disk or blue noise) with deterministic seeds.
- [x] Roads/trails: least-cost path between points using slope/water penalties.
- [x] Contours and map labels: contour lines, coastline outline, and a clean cartographic style mode.
- [x] Export tileset: generate map tiles (z/x/y) for slippy-map viewers.

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
- [x] Add keyboard shortcuts for common actions (reset, randomize seed, toggle histogram, toggle 3D).
- [x] Add input validation UX: inline warnings, clamping indicators, and disabled states when invalid.
- [x] Add a "quality" selector: LOD for 2D/3D with progressive refinement (fast preview then refine).
- [x] Add a dedicated "Learn" design: step cards, breadcrumbs, and a consistent inspector panel.
- [x] Replace the most latency-sensitive sliders with a custom Streamlit component that emits values while dragging.
- [x] Improve exports UX: export panel with format options, naming, and a preview of what will be exported.

### Practical UI Polish (Backlog)

- [x] Replace the top-of-page "Practical settings" expander with a cleaner layout (sidebar panel or a dedicated settings column) so the viewport stays visible.
- [x] Re-organize Practical controls into tabbed groups (Terrain / Hydrology / Weathering / Navigation / Extras) instead of long vertical stacks.
- [x] Create a compact "Quick Controls" row (water level, mountain level, rivers on/off, erosion on/off, backend) without scrolling.
- [x] Remove duplicated controls across the Practical settings and Navigator tab (single source of truth for navigation + export).
- [x] Consolidate repeated map renders across Terrain/Rivers tabs: one viewport with overlay toggles (rivers, lakes, trails, chunk grid, seams, vegetation).
- [x] Standardize control labels and units (world vs px, iteration naming, erosion parameter naming) and align to a consistent typography hierarchy.
- [x] Make debug-only UI clearly optional: collapse logs/seams/layer inspectors by default and add a "Debug mode" toggle.
- [x] Improve spatial balance: avoid equal 3-column settings when one column becomes extremely tall; use adaptive widths and consistent spacing.
- [x] Improve mobile behavior: collapse secondary panels, avoid nested tabs where possible, and reduce chart heights for stacked layouts.
- [x] Add cache/status mini-panel in Practical (chunk coords, cache size, cache hits/misses, backend) for confidence and troubleshooting.

## Performance Targets

- [x] 2D: interactive at typical sizes (e.g., 256x256 to 1024x1024) with responsive UI.
- [x] 3D: "feels real-time" for moderate grids (e.g., 128x128 to 256x256) with smooth camera interaction.

## Scope Boundaries (Finite Goals Instead Of Non-Goals)

These items are intentionally *bounded* so the project stays learning-first.

### "World Streaming" (Streamlit-Scale, Not Engine-Scale)

Instead of aiming for a game-engine runtime world streamer, we keep a finite, testable target:

- [x] Chunk cache (LRU): keep the last N chunks computed (height, rivers, biome, RGB) to make navigation feel instant.
- [x] Chunk contract: a single function that maps (seed, chunk_x, chunk_y, params) -> deterministic chunk outputs.
- [x] Chunk size invariants: fixed pixel size per chunk and a clear world-units mapping.
- [x] Seam debugging: show chunk boundaries and a seam-delta overlay (edge mismatch heatmap).
- [x] No background threads required: all generation stays deterministic and debuggable from the UI.
- [x] Export contract: export a stitched region (k x k chunks) with metadata that reproduces it.

Explicitly out of scope for now:

- Real-time physics, meshes streamed to a game engine, server-authoritative multiplayer, persistence/serialization for huge worlds.

### Acceleration (Optional), Not GPU-Only

Instead of making GPU the default path, we target a finite “fast enough on CPU” baseline and keep acceleration as optional:

- [x] CPU-first reference: NumPy implementation remains the source of truth for determinism and tests.
- [x] Optional acceleration track: add a faster backend (e.g., float32) behind a toggle, matching outputs within tolerance.
- [x] Benchmark harness: simple timing script + documented target resolutions/latency for Explore and Practical.
- [x] Consistent results: any accelerated backend must preserve determinism for the same seed/params.

Explicitly out of scope for now:

- Making GPU mandatory, requiring CUDA, or maintaining multiple GPU backends as the primary implementation.

## Definition of Done (For This Repo)

- [x] The app can teach Perlin noise step-by-step to someone who knows basic vectors and interpolation.
- [x] The implementation is test-covered, deterministic, and modular.
- [x] The 3D view is interactive and updates quickly enough to encourage experimentation.
