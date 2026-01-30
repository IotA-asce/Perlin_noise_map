from __future__ import annotations

import io

import numpy as np
from PIL import Image


def array_to_png_bytes(z: np.ndarray) -> bytes:
    """Convert a 2D array to an 8-bit grayscale PNG.

    Values are min/max normalized to [0, 255]. Degenerate (constant) arrays
    become all zeros.
    """

    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("expected a 2D array")

    zmin = float(np.min(z))
    zmax = float(np.max(z))
    if zmax == zmin:
        img = np.zeros(z.shape, dtype=np.uint8)
    else:
        zn = (z - zmin) / (zmax - zmin)
        img = np.clip(zn * 255.0, 0.0, 255.0).astype(np.uint8)

    out = io.BytesIO()
    Image.fromarray(img, mode="L").save(out, format="PNG")
    return out.getvalue()
