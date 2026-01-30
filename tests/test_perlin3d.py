import numpy as np

from perlin.noise_3d import Perlin3D


def test_perlin3d_deterministic_for_seed():
    p1 = Perlin3D(seed=123)
    p2 = Perlin3D(seed=123)
    x = np.array([0.1, 1.25, 10.5])
    y = np.array([0.2, 2.75, 9.0])
    z = np.array([0.3, 0.5, 1.0])
    assert np.allclose(p1.noise(x, y, z), p2.noise(x, y, z))


def test_perlin3d_changes_with_seed():
    p1 = Perlin3D(seed=1)
    p2 = Perlin3D(seed=2)
    x = np.array([0.1, 1.25, 10.5])
    y = np.array([0.2, 2.75, 9.0])
    z = np.array([0.3, 0.5, 1.0])
    assert not np.allclose(p1.noise(x, y, z), p2.noise(x, y, z))


def test_perlin3d_shape_finite_and_reasonable_range():
    p = Perlin3D(seed=0)
    xg, yg, zg = np.meshgrid(
        np.linspace(0.0, 2.0, 16),
        np.linspace(0.0, 2.0, 12),
        np.linspace(0.0, 2.0, 8),
        indexing="xy",
    )
    out = p.noise(xg, yg, zg)
    assert out.shape == xg.shape
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) < 2.0
