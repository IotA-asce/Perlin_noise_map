from __future__ import annotations

import heapq

import numpy as np


def fill_depressions_priority_flood(
    height: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill depressions using a priority-flood (depression fill).

    Returns (filled_height, lake_depth), where:
    - filled_height is the minimally raised surface so every cell can drain to the edge
    - lake_depth = filled_height - height

    This is deterministic and intended for inspectable learning demos.
    """

    h = np.asarray(height, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("height must be a 2D array")

    H, W = h.shape
    if H == 0 or W == 0:
        return h.copy(), np.zeros_like(h)

    filled = np.empty_like(h)
    visited = np.zeros((H, W), dtype=bool)

    heap: list[tuple[float, int, int]] = []

    def push(y: int, x: int) -> None:
        if visited[y, x]:
            return
        visited[y, x] = True
        v = float(h[y, x])
        filled[y, x] = v
        heapq.heappush(heap, (v, y, x))

    # Initialize with boundary cells.
    for x in range(W):
        push(0, x)
        if H > 1:
            push(H - 1, x)
    for y in range(1, H - 1):
        push(y, 0)
        if W > 1:
            push(y, W - 1)

    # Flood inward: any lower interior cell is raised to current spill elevation.
    while heap:
        v, y, x = heapq.heappop(heap)
        v = float(v)

        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            if visited[ny, nx]:
                continue
            visited[ny, nx] = True
            hv = float(h[ny, nx])
            fv = hv if hv >= v else v
            filled[ny, nx] = fv
            heapq.heappush(heap, (fv, ny, nx))

    lake_depth = np.clip(filled - h, 0.0, np.inf)
    return filled, lake_depth
