import numpy as np

from perlin.noise_2d import Perlin2D, fbm2


def test_perlin2d_deterministic_for_seed():
    p1 = Perlin2D(seed=123)
    p2 = Perlin2D(seed=123)
    x = np.array([0.1, 1.25, 10.5])
    y = np.array([0.2, 2.75, 9.0])
    assert np.allclose(p1.noise(x, y), p2.noise(x, y))


def test_perlin2d_changes_with_seed():
    p1 = Perlin2D(seed=1)
    p2 = Perlin2D(seed=2)
    x = np.array([0.1, 1.25, 10.5])
    y = np.array([0.2, 2.75, 9.0])
    assert not np.allclose(p1.noise(x, y), p2.noise(x, y))


def test_fbm2_shape_and_finite():
    p = Perlin2D(seed=0)
    xg, yg = np.meshgrid(np.linspace(0, 3, 64), np.linspace(0, 3, 32))
    z = fbm2(p, xg, yg, octaves=4, lacunarity=2.0, persistence=0.5)
    assert z.shape == xg.shape
    assert np.isfinite(z).all()
