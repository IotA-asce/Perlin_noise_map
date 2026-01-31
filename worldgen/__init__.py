from __future__ import annotations

from worldgen.chunks import chunk_origin, chunk_world_size, generate_chunk
from worldgen.climate import apply_climate_palette, climate_biome_map
from worldgen.coast import beach_deposit, box_blur2d, coastline_smooth
from worldgen.contours import apply_mask_overlay, contour_mask
from worldgen.erosion import (
    hydraulic_erosion,
    hydraulic_erosion_frames,
    thermal_erosion,
    thermal_erosion_frames,
)
from worldgen.hydrology import (
    carve_rivers,
    flow_accumulation_d8,
    flow_direction_d8,
    river_mask_from_accumulation,
)
from worldgen.lakes import fill_depressions_priority_flood
from worldgen.paths import astar_path
from worldgen.pipeline import practical_pipeline
from worldgen.terrain import (
    hillshade01,
    slope01,
    terrain_colormap,
    terrain_masks,
)
from worldgen.tiles import tiles_zip_from_rgb
from worldgen.vegetation import filter_points_by_mask, jittered_points

__all__ = [
    "apply_climate_palette",
    "apply_mask_overlay",
    "astar_path",
    "chunk_origin",
    "chunk_world_size",
    "carve_rivers",
    "beach_deposit",
    "box_blur2d",
    "coastline_smooth",
    "climate_biome_map",
    "contour_mask",
    "flow_accumulation_d8",
    "flow_direction_d8",
    "hillshade01",
    "fill_depressions_priority_flood",
    "filter_points_by_mask",
    "river_mask_from_accumulation",
    "slope01",
    "generate_chunk",
    "terrain_colormap",
    "terrain_masks",
    "hydraulic_erosion",
    "hydraulic_erosion_frames",
    "practical_pipeline",
    "thermal_erosion",
    "thermal_erosion_frames",
    "tiles_zip_from_rgb",
    "jittered_points",
]
