import numpy as np

from perlin.value_noise_2d import ValueNoise2D


def test_value_noise2d_deterministic_for_seed():
    n1 = ValueNoise2D(seed=123)
    n2 = ValueNoise2D(seed=123)
    x = np.array([0.1, 1.25, 10.5])
    y = np.array([0.2, 2.75, 9.0])
    assert np.allclose(n1.noise(x, y), n2.noise(x, y))


def test_value_noise2d_changes_with_seed():
    n1 = ValueNoise2D(seed=1)
    n2 = ValueNoise2D(seed=2)
    x = np.array([0.1, 1.25, 10.5])
    y = np.array([0.2, 2.75, 9.0])
    assert not np.allclose(n1.noise(x, y), n2.noise(x, y))


def test_value_noise2d_shape_finite_and_reasonable_range():
    n = ValueNoise2D(seed=0)
    xg, yg = np.meshgrid(np.linspace(0, 3, 64), np.linspace(0, 3, 32))
    out = n.noise(xg, yg)
    assert out.shape == xg.shape
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) <= 1.0
