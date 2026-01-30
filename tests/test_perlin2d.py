import numpy as np

from perlin.noise_2d import Perlin2D, fbm2, ridged2, tileable_fbm2, turbulence2
from viz.step_2d import scanline_series_from_debug


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


def test_perlin2d_debug_point_matches_noise():
    p = Perlin2D(seed=0)
    x = 2.25
    y = 3.75
    dbg = p.debug_point(x, y)
    out = float(p.noise(np.array(x), np.array(y)))
    assert np.allclose(dbg["noise"], out)
    assert "interpolation" in dbg
    assert "x_lerp0" in dbg["interpolation"]
    assert "x_lerp1" in dbg["interpolation"]


def test_scanline_series_matches_perlin_noise():
    p = Perlin2D(seed=0)
    x = 2.25
    y = 3.75
    dbg = p.debug_point(x, y)
    series = scanline_series_from_debug(dbg, steps=128)

    x_base = float(np.floor(x))
    y_base = float(np.floor(y))
    y_rel = float(dbg["relative"]["yf"])

    xs = x_base + series["t"]
    ys = y_base + y_rel
    expected = p.noise(xs, ys)
    assert np.allclose(series["lerp"]["noise"], expected)


def test_tileable_fbm2_periodic_in_x_and_y():
    p = Perlin2D(seed=0)
    period_x = 3.25
    period_y = 4.5

    x = np.array([0.1, 1.25, 10.5], dtype=np.float64)
    y = np.array([0.2, 2.75, 9.0], dtype=np.float64)

    z0 = tileable_fbm2(
        p,
        x,
        y,
        period_x=period_x,
        period_y=period_y,
        octaves=4,
        lacunarity=2.0,
        persistence=0.5,
    )
    zx = tileable_fbm2(
        p,
        x + period_x,
        y,
        period_x=period_x,
        period_y=period_y,
        octaves=4,
        lacunarity=2.0,
        persistence=0.5,
    )
    zy = tileable_fbm2(
        p,
        x,
        y + period_y,
        period_x=period_x,
        period_y=period_y,
        octaves=4,
        lacunarity=2.0,
        persistence=0.5,
    )

    assert np.allclose(z0, zx)
    assert np.allclose(z0, zy)


def test_turbulence2_shape_finite_and_deterministic():
    p1 = Perlin2D(seed=0)
    p2 = Perlin2D(seed=0)
    xg, yg = np.meshgrid(np.linspace(0, 3, 64), np.linspace(0, 3, 32))
    z1 = turbulence2(p1, xg, yg, octaves=4, lacunarity=2.0, persistence=0.5)
    z2 = turbulence2(p2, xg, yg, octaves=4, lacunarity=2.0, persistence=0.5)
    assert z1.shape == xg.shape
    assert np.isfinite(z1).all()
    assert np.allclose(z1, z2)


def test_ridged2_shape_finite_and_deterministic():
    p1 = Perlin2D(seed=0)
    p2 = Perlin2D(seed=0)
    xg, yg = np.meshgrid(np.linspace(0, 3, 64), np.linspace(0, 3, 32))
    z1 = ridged2(p1, xg, yg, octaves=4, lacunarity=2.0, persistence=0.5)
    z2 = ridged2(p2, xg, yg, octaves=4, lacunarity=2.0, persistence=0.5)
    assert z1.shape == xg.shape
    assert np.isfinite(z1).all()
    assert np.allclose(z1, z2)
