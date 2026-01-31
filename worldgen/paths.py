from __future__ import annotations

import heapq

import numpy as np


def astar_path(
    cost: np.ndarray,
    *,
    start: tuple[int, int],
    goal: tuple[int, int],
    diag: bool = True,
) -> list[tuple[int, int]]:
    """A* path on a 2D cost grid.

    start/goal are (y, x). Cells with non-finite cost are treated as blocked.
    """

    c = np.asarray(cost, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError("cost must be a 2D array")

    H, W = c.shape
    sy, sx = int(start[0]), int(start[1])
    gy, gx = int(goal[0]), int(goal[1])
    if not (0 <= sy < H and 0 <= sx < W and 0 <= gy < H and 0 <= gx < W):
        return []

    if not np.isfinite(c[sy, sx]) or not np.isfinite(c[gy, gx]):
        return []

    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if bool(diag):
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def h(y: int, x: int) -> float:
        return float(abs(y - gy) + abs(x - gx))

    INF = float("inf")
    gscore = np.full((H, W), INF, dtype=np.float64)
    came_y = np.full((H, W), -1, dtype=np.int32)
    came_x = np.full((H, W), -1, dtype=np.int32)

    gscore[sy, sx] = 0.0
    heap: list[tuple[float, int, int]] = [(h(sy, sx), sy, sx)]

    while heap:
        _, y, x = heapq.heappop(heap)
        if y == gy and x == gx:
            break

        gyx = float(gscore[y, x])
        for dy, dx in offsets:
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            if not np.isfinite(c[ny, nx]):
                continue
            step = float(c[ny, nx])
            if dy != 0 and dx != 0:
                step *= 1.41421356237
            ng = gyx + step
            if ng < float(gscore[ny, nx]):
                gscore[ny, nx] = ng
                came_y[ny, nx] = int(y)
                came_x[ny, nx] = int(x)
                heapq.heappush(heap, (ng + h(ny, nx), ny, nx))

    if came_y[gy, gx] == -1 and (gy, gx) != (sy, sx):
        return []

    path: list[tuple[int, int]] = []
    y, x = gy, gx
    path.append((y, x))
    while (y, x) != (sy, sx):
        py = int(came_y[y, x])
        px = int(came_x[y, x])
        if py < 0 or px < 0:
            return []
        y, x = py, px
        path.append((y, x))

    path.reverse()
    return path
