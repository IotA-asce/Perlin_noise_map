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


def test_perlin2d_reasonable_range():
    p = Perlin2D(seed=0)
    xg, yg = np.meshgrid(np.linspace(0, 5, 64), np.linspace(0, 5, 64))
    z = p.noise(xg, yg)
    assert float(np.max(np.abs(z))) < 2.0


def test_perlin2d_continuity_small_step():
    p = Perlin2D(seed=0)
    xg, yg = np.meshgrid(np.linspace(0, 5, 64), np.linspace(0, 5, 64))
    d = 1e-4
    z0 = p.noise(xg, yg)
    z1 = p.noise(xg + d, yg)
    assert float(np.max(np.abs(z1 - z0))) < 0.1


def test_perlin2d_reference_values():
    p = Perlin2D(seed=0)
    pts = np.array(
        [
            [0.1, 0.2],
            [1.25, 2.75],
            [10.5, 9.0],
            [2.25, 3.75],
        ],
        dtype=np.float64,
    )
    out = p.noise(pts[:, 0], pts[:, 1])
    expected = np.array(
        [
            -0.19122562499637968,
            0.024773190814829205,
            -0.07322330470336313,
            -0.2761086726516018,
        ],
        dtype=np.float64,
    )
    assert np.allclose(out, expected)
