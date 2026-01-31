from __future__ import annotations

import numpy as np

from worldgen.erosion import (
    hydraulic_erosion,
    hydraulic_erosion_frames,
    thermal_erosion,
    thermal_erosion_frames,
)


def test_thermal_erosion_shape_and_bounds() -> None:
    rng = np.random.default_rng(0)
    z = rng.random((48, 64), dtype=np.float64)
    out = thermal_erosion(z, iterations=10, talus=0.02, strength=0.35)
    assert out.shape == z.shape
    assert float(np.min(out)) >= 0.0
    assert float(np.max(out)) <= 1.0


def test_thermal_erosion_deterministic() -> None:
    rng = np.random.default_rng(1)
    z = rng.random((32, 32), dtype=np.float64)
    a = thermal_erosion(z, iterations=8, talus=0.03, strength=0.5)
    b = thermal_erosion(z, iterations=8, talus=0.03, strength=0.5)
    assert np.allclose(a, b)


def test_hydraulic_erosion_shape_and_bounds() -> None:
    rng = np.random.default_rng(3)
    z = rng.random((40, 48), dtype=np.float64)
    out, water, sediment = hydraulic_erosion(
        z,
        iterations=10,
        rain=0.01,
        evaporation=0.5,
        flow_rate=0.5,
        capacity=4.0,
        erosion=0.3,
        deposition=0.3,
    )
    assert out.shape == z.shape
    assert water.shape == z.shape
    assert sediment.shape == z.shape
    assert float(np.min(out)) >= 0.0
    assert float(np.max(out)) <= 1.0
    assert float(np.min(water)) >= 0.0
    assert float(np.min(sediment)) >= 0.0


def test_hydraulic_erosion_deterministic() -> None:
    rng = np.random.default_rng(4)
    z = rng.random((24, 24), dtype=np.float64)
    a = hydraulic_erosion(z, iterations=6)
    b = hydraulic_erosion(z, iterations=6)
    assert np.allclose(a[0], b[0])
    assert np.allclose(a[1], b[1])
    assert np.allclose(a[2], b[2])


def test_thermal_erosion_frames_shape() -> None:
    rng = np.random.default_rng(5)
    z = rng.random((20, 24), dtype=np.float64)
    frames = thermal_erosion_frames(z, iterations=9, every=2)
    assert frames.ndim == 3
    assert frames.shape[1:] == z.shape
    assert float(np.min(frames)) >= 0.0
    assert float(np.max(frames)) <= 1.0


def test_hydraulic_erosion_frames_shape() -> None:
    rng = np.random.default_rng(6)
    z = rng.random((18, 18), dtype=np.float64)
    hf, wf, sf = hydraulic_erosion_frames(z, iterations=7, every=3)
    assert hf.shape[1:] == z.shape
    assert wf.shape == hf.shape
    assert sf.shape == hf.shape
    assert float(np.min(hf)) >= 0.0
    assert float(np.max(hf)) <= 1.0
    assert float(np.min(wf)) >= 0.0
    assert float(np.min(sf)) >= 0.0
