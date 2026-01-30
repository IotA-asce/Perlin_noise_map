import io

import numpy as np
from PIL import Image

from viz.export import array_to_npy_bytes, array_to_png_bytes, heightmap_to_obj_bytes


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


def test_array_to_npy_bytes_roundtrip():
    z = np.arange(6, dtype=np.int32).reshape(2, 3)
    data = array_to_npy_bytes(z)
    out = np.load(io.BytesIO(data))
    assert np.array_equal(out, z)


def test_heightmap_to_obj_bytes_vertex_and_face_counts():
    z = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float64)
    obj = heightmap_to_obj_bytes(z, z_scale=2.0).decode("utf-8")

    v_lines = [ln for ln in obj.splitlines() if ln.startswith("v ")]
    f_lines = [ln for ln in obj.splitlines() if ln.startswith("f ")]
    assert len(v_lines) == 6
    assert len(f_lines) == 4
