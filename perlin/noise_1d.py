from __future__ import annotations

import numpy as np

from .core import fade, grad1_from_hash, lerp, make_permutation


class Perlin1D:
    def __init__(self, *, seed: int = 0):
        self.seed = int(seed)
        self.perm = make_permutation(self.seed)

    def noise(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)

        xi0 = np.floor(x).astype(np.int32) & 255
        xi1 = (xi0 + 1) & 255

        xf = x - np.floor(x)
        u = fade(xf)

        p = self.perm
        a = p[xi0]
        b = p[xi1]

        ga = grad1_from_hash(a)
        gb = grad1_from_hash(b)

        d0 = ga * xf
        d1 = gb * (xf - 1.0)
        return lerp(d0, d1, u)
