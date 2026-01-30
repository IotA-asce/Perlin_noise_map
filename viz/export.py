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


def array_to_npy_bytes(z: np.ndarray) -> bytes:
    z = np.asarray(z)
    out = io.BytesIO()
    np.save(out, z)
    return out.getvalue()


def heightmap_to_obj_bytes(z: np.ndarray, *, z_scale: float = 1.0) -> bytes:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("expected a 2D array")

    z_scale = float(z_scale)
    h, w = z.shape
    if h < 2 or w < 2:
        raise ValueError("heightmap must be at least 2x2")

    def vid(x: int, y: int) -> int:
        return y * w + x + 1

    lines: list[str] = []
    lines.append("# Perlin Noise Map heightmap\n")
    lines.append(f"# grid={w}x{h}\n")

    for y in range(h):
        for x in range(w):
            lines.append(f"v {x:.6f} {y:.6f} {(z[y, x] * z_scale):.6f}\n")

    for y in range(h - 1):
        for x in range(w - 1):
            v00 = vid(x, y)
            v10 = vid(x + 1, y)
            v01 = vid(x, y + 1)
            v11 = vid(x + 1, y + 1)
            lines.append(f"f {v00} {v10} {v01}\n")
            lines.append(f"f {v10} {v11} {v01}\n")

    return "".join(lines).encode("utf-8")
