from __future__ import annotations

import io
import zipfile

import numpy as np
from PIL import Image


def tiles_zip_from_rgb(
    rgb01: np.ndarray,
    *,
    z: int = 0,
    grid: int = 4,
    tile_size: int = 256,
) -> bytes:
    """Split an RGB image into a z/x/y.png tile zip (single zoom level).

    This is a practical export helper (not a full slippy-map world projection).
    """

    rgb = np.asarray(rgb01, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb01 must be HxWx3")

    z = int(z)
    g = max(int(grid), 1)
    ts = max(int(tile_size), 16)

    img = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    pil = Image.fromarray(img, mode="RGB")

    W, H = pil.size
    tile_w = max(W // g, 1)
    tile_h = max(H // g, 1)

    out = io.BytesIO()
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for y in range(g):
            for x in range(g):
                left = x * tile_w
                top = y * tile_h
                right = W if x == g - 1 else (x + 1) * tile_w
                bottom = H if y == g - 1 else (y + 1) * tile_h

                tile = pil.crop((left, top, right, bottom)).resize(
                    (ts, ts), resample=Image.BILINEAR
                )
                buf = io.BytesIO()
                tile.save(buf, format="PNG")
                zf.writestr(f"{z}/{x}/{y}.png", buf.getvalue())

    return out.getvalue()
