import io

import numpy as np
from PIL import Image

from viz.export import array_to_png_bytes


def test_array_to_png_bytes_roundtrip():
    z = np.arange(12, dtype=np.float64).reshape(3, 4)
    data = array_to_png_bytes(z)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"

    img = Image.open(io.BytesIO(data))
    assert img.size == (4, 3)


def test_array_to_png_bytes_constant_map():
    z = np.full((5, 6), 7.0, dtype=np.float64)
    data = array_to_png_bytes(z)
    img = Image.open(io.BytesIO(data))
    arr = np.array(img)
    assert arr.min() == 0
    assert arr.max() == 0
