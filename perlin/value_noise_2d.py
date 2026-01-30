from __future__ import annotations

import numpy as np

from .core import fade, lerp, make_permutation


class ValueNoise2D:
    """2D value noise (lattice values + smooth interpolation)."""

    def __init__(self, *, seed: int = 0):
        self.seed = int(seed)
        self.perm = make_permutation(self.seed)

    def noise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        xi0 = np.floor(x).astype(np.int32) & 255
        yi0 = np.floor(y).astype(np.int32) & 255
        xi1 = (xi0 + 1) & 255
        yi1 = (yi0 + 1) & 255

        xf = x - np.floor(x)
        yf = y - np.floor(y)
        u = fade(xf)
        v = fade(yf)

        p = self.perm
        aa = p[p[xi0] + yi0]
        ab = p[p[xi0] + yi1]
        ba = p[p[xi1] + yi0]
        bb = p[p[xi1] + yi1]

        # Map hashed values into [-1, 1].
        vaa = (aa.astype(np.float64) / 255.0) * 2.0 - 1.0
        vab = (ab.astype(np.float64) / 255.0) * 2.0 - 1.0
        vba = (ba.astype(np.float64) / 255.0) * 2.0 - 1.0
        vbb = (bb.astype(np.float64) / 255.0) * 2.0 - 1.0

        x_lerp0 = lerp(vaa, vba, u)
        x_lerp1 = lerp(vab, vbb, u)
        return lerp(x_lerp0, x_lerp1, v)
