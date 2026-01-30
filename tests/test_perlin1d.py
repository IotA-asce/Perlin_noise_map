import numpy as np

from perlin.noise_1d import Perlin1D


def test_perlin1d_deterministic_for_seed():
    p1 = Perlin1D(seed=123)
    p2 = Perlin1D(seed=123)
    x = np.array([0.1, 1.25, 10.5])
    assert np.allclose(p1.noise(x), p2.noise(x))


def test_perlin1d_changes_with_seed():
    p1 = Perlin1D(seed=1)
    p2 = Perlin1D(seed=2)
    x = np.array([0.1, 1.25, 10.5])
    assert not np.allclose(p1.noise(x), p2.noise(x))


def test_perlin1d_shape_finite_and_reasonable_range():
    p = Perlin1D(seed=0)
    x = np.linspace(0.0, 10.0, 513)
    z = p.noise(x)
    assert z.shape == x.shape
    assert np.isfinite(z).all()
    assert float(np.max(np.abs(z))) < 2.0


def test_perlin1d_continuity_small_step():
    p = Perlin1D(seed=0)
    x = np.linspace(0.0, 10.0, 256)
    dx = 1e-4
    z0 = p.noise(x)
    z1 = p.noise(x + dx)
    assert float(np.max(np.abs(z1 - z0))) < 0.1
