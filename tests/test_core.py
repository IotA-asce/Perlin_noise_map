import numpy as np

from perlin.core import fade, lerp


def test_fade_endpoints():
    t = np.array([0.0, 1.0], dtype=np.float64)
    out = fade(t)
    assert out[0] == 0.0
    assert out[1] == 1.0


def test_lerp_basic():
    a = np.array([0.0, 10.0])
    b = np.array([10.0, 20.0])
    t = np.array([0.0, 0.5])
    out = lerp(a, b, t)
    assert np.allclose(out, np.array([0.0, 15.0]))
