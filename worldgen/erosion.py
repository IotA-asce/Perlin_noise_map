from __future__ import annotations

import numpy as np

from worldgen.hydrology import flow_direction_d8


def thermal_erosion(
    height01: np.ndarray,
    *,
    iterations: int = 30,
    talus: float = 0.02,
    strength: float = 0.35,
) -> np.ndarray:
    """Simple deterministic thermal erosion (talus) on a 0..1 heightmap.

    Uses 4-neighborhood transfers. This is meant for visual intuition, not
    physically exact simulation.
    """

    h = np.asarray(height01, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("height01 must be a 2D array")

    iterations = int(iterations)
    if iterations < 0:
        raise ValueError("iterations must be >= 0")
    talus = float(talus)
    strength = float(strength)
    strength = float(np.clip(strength, 0.0, 1.0))

    out = h.copy()

    for _ in range(iterations):
        p = np.pad(out, 1, mode="edge")
        c = p[1:-1, 1:-1]
        n = p[:-2, 1:-1]
        s = p[2:, 1:-1]
        w = p[1:-1, :-2]
        e = p[1:-1, 2:]

        dn = np.maximum(0.0, (c - n) - talus)
        ds = np.maximum(0.0, (c - s) - talus)
        dw = np.maximum(0.0, (c - w) - talus)
        de = np.maximum(0.0, (c - e) - talus)

        out_n = strength * 0.25 * dn
        out_s = strength * 0.25 * ds
        out_w = strength * 0.25 * dw
        out_e = strength * 0.25 * de

        outflow = out_n + out_s + out_w + out_e
        inflow = np.zeros_like(out)

        inflow[:-1, :] += out_s[:-1, :]
        inflow[1:, :] += out_n[1:, :]
        inflow[:, :-1] += out_e[:, :-1]
        inflow[:, 1:] += out_w[:, 1:]

        out = np.clip(out - outflow + inflow, 0.0, 1.0)

    return out


def thermal_erosion_frames(
    height01: np.ndarray,
    *,
    iterations: int = 30,
    talus: float = 0.02,
    strength: float = 0.35,
    every: int = 1,
) -> np.ndarray:
    """Return intermediate frames of thermal erosion.

    Frames include the initial height at frame 0.
    Output shape: (frames, H, W)
    """

    h = np.asarray(height01, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("height01 must be a 2D array")

    iterations = int(iterations)
    if iterations < 0:
        raise ValueError("iterations must be >= 0")
    talus = float(talus)
    strength = float(strength)
    strength = float(np.clip(strength, 0.0, 1.0))

    every = int(every)
    if every <= 0:
        raise ValueError("every must be >= 1")

    out = h.copy()
    frames: list[np.ndarray] = [out.copy()]

    for i in range(iterations):
        p = np.pad(out, 1, mode="edge")
        c = p[1:-1, 1:-1]
        n = p[:-2, 1:-1]
        s = p[2:, 1:-1]
        w = p[1:-1, :-2]
        e = p[1:-1, 2:]

        dn = np.maximum(0.0, (c - n) - talus)
        ds = np.maximum(0.0, (c - s) - talus)
        dw = np.maximum(0.0, (c - w) - talus)
        de = np.maximum(0.0, (c - e) - talus)

        out_n = strength * 0.25 * dn
        out_s = strength * 0.25 * ds
        out_w = strength * 0.25 * dw
        out_e = strength * 0.25 * de

        outflow = out_n + out_s + out_w + out_e
        inflow = np.zeros_like(out)

        inflow[:-1, :] += out_s[:-1, :]
        inflow[1:, :] += out_n[1:, :]
        inflow[:, :-1] += out_e[:, :-1]
        inflow[:, 1:] += out_w[:, 1:]

        out = np.clip(out - outflow + inflow, 0.0, 1.0)

        if ((i + 1) % every) == 0 or (i + 1) == iterations:
            frames.append(out.copy())

    return np.stack(frames, axis=0)


def hydraulic_erosion(
    height01: np.ndarray,
    *,
    iterations: int = 40,
    rain: float = 0.01,
    evaporation: float = 0.5,
    flow_rate: float = 0.5,
    capacity: float = 4.0,
    erosion: float = 0.3,
    deposition: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple deterministic hydraulic erosion for visualization.

    This is intentionally lightweight (D8 routing + sediment capacity) and is
    designed for inspectability rather than physical accuracy.

    Returns (height01, water, sediment).
    """

    h = np.asarray(height01, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("height01 must be a 2D array")

    iterations = int(iterations)
    if iterations < 0:
        raise ValueError("iterations must be >= 0")

    rain = float(rain)
    evaporation = float(evaporation)
    flow_rate = float(flow_rate)
    capacity = float(capacity)
    erosion = float(erosion)
    deposition = float(deposition)

    evaporation = float(np.clip(evaporation, 0.0, 1.0))
    flow_rate = float(np.clip(flow_rate, 0.0, 1.0))
    erosion = float(np.clip(erosion, 0.0, 1.0))
    deposition = float(np.clip(deposition, 0.0, 1.0))

    out = h.copy()
    water = np.zeros_like(out)
    sediment = np.zeros_like(out)

    H, W = out.shape
    n = int(H * W)

    for _ in range(iterations):
        water += rain

        ds = flow_direction_d8(out)
        dsf = ds.reshape(-1)
        hf = out.reshape(-1)
        wf = water.reshape(-1)
        sf = sediment.reshape(-1)

        # Route a fraction of water to the downstream cell.
        outflow = wf * flow_rate
        wf2 = wf - outflow
        inflow = np.zeros(n, dtype=np.float64)
        m = dsf >= 0
        if bool(np.any(m)):
            np.add.at(inflow, dsf[m].astype(np.int64), outflow[m])
        wf2 += inflow

        # Capacity increases with water amount and local drop.
        drop = np.zeros(n, dtype=np.float64)
        if bool(np.any(m)):
            j = dsf[m].astype(np.int64)
            drop[m] = np.maximum(0.0, hf[m] - hf[j])

        cap = capacity * drop * wf2

        # Erode if below capacity, deposit if above.
        need = cap - sf
        er = erosion * np.maximum(0.0, need)
        dep = deposition * np.maximum(0.0, -need)

        hf2 = hf - er + dep
        sf2 = sf + er - dep

        out = np.clip(hf2.reshape(H, W), 0.0, 1.0)
        water = wf2.reshape(H, W) * (1.0 - evaporation)
        sediment = np.clip(sf2.reshape(H, W), 0.0, np.inf)

    return out, water, sediment


def hydraulic_erosion_frames(
    height01: np.ndarray,
    *,
    iterations: int = 40,
    rain: float = 0.01,
    evaporation: float = 0.5,
    flow_rate: float = 0.5,
    capacity: float = 4.0,
    erosion: float = 0.3,
    deposition: float = 0.3,
    every: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return intermediate frames of hydraulic erosion.

    Frames include the initial state at frame 0.
    Returns (height_frames, water_frames, sediment_frames).
    Each output shape: (frames, H, W)
    """

    h = np.asarray(height01, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("height01 must be a 2D array")

    iterations = int(iterations)
    if iterations < 0:
        raise ValueError("iterations must be >= 0")

    rain = float(rain)
    evaporation = float(evaporation)
    flow_rate = float(flow_rate)
    capacity = float(capacity)
    erosion = float(erosion)
    deposition = float(deposition)

    evaporation = float(np.clip(evaporation, 0.0, 1.0))
    flow_rate = float(np.clip(flow_rate, 0.0, 1.0))
    erosion = float(np.clip(erosion, 0.0, 1.0))
    deposition = float(np.clip(deposition, 0.0, 1.0))

    every = int(every)
    if every <= 0:
        raise ValueError("every must be >= 1")

    out = h.copy()
    water = np.zeros_like(out)
    sediment = np.zeros_like(out)

    h_frames: list[np.ndarray] = [out.copy()]
    w_frames: list[np.ndarray] = [water.copy()]
    s_frames: list[np.ndarray] = [sediment.copy()]

    H, W = out.shape
    n = int(H * W)

    for i in range(iterations):
        water += rain

        ds = flow_direction_d8(out)
        dsf = ds.reshape(-1)
        hf = out.reshape(-1)
        wf = water.reshape(-1)
        sf = sediment.reshape(-1)

        outflow = wf * flow_rate
        wf2 = wf - outflow
        inflow = np.zeros(n, dtype=np.float64)
        m = dsf >= 0
        if bool(np.any(m)):
            np.add.at(inflow, dsf[m].astype(np.int64), outflow[m])
        wf2 += inflow

        drop = np.zeros(n, dtype=np.float64)
        if bool(np.any(m)):
            j = dsf[m].astype(np.int64)
            drop[m] = np.maximum(0.0, hf[m] - hf[j])

        cap = capacity * drop * wf2

        need = cap - sf
        er = erosion * np.maximum(0.0, need)
        dep = deposition * np.maximum(0.0, -need)

        hf2 = hf - er + dep
        sf2 = sf + er - dep

        out = np.clip(hf2.reshape(H, W), 0.0, 1.0)
        water = wf2.reshape(H, W) * (1.0 - evaporation)
        sediment = np.clip(sf2.reshape(H, W), 0.0, np.inf)

        if ((i + 1) % every) == 0 or (i + 1) == iterations:
            h_frames.append(out.copy())
            w_frames.append(water.copy())
            s_frames.append(sediment.copy())

    return (
        np.stack(h_frames, axis=0),
        np.stack(w_frames, axis=0),
        np.stack(s_frames, axis=0),
    )
