from __future__ import annotations

import numpy as np

from .core import fade, grad3_from_hash, lerp, make_permutation


class Perlin3D:
    def __init__(self, *, seed: int = 0):
        self.seed = int(seed)
        self.perm = make_permutation(self.seed)

    def noise(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        xi0 = np.floor(x).astype(np.int32) & 255
        yi0 = np.floor(y).astype(np.int32) & 255
        zi0 = np.floor(z).astype(np.int32) & 255

        xi1 = (xi0 + 1) & 255
        yi1 = (yi0 + 1) & 255
        zi1 = (zi0 + 1) & 255

        xf = x - np.floor(x)
        yf = y - np.floor(y)
        zf = z - np.floor(z)

        u = fade(xf)
        v = fade(yf)
        w = fade(zf)

        p = self.perm

        x0 = xf
        y0 = yf
        z0 = zf
        x1 = xf - 1.0
        y1 = yf - 1.0
        z1 = zf - 1.0

        aaa = p[p[p[xi0] + yi0] + zi0]
        aab = p[p[p[xi0] + yi0] + zi1]
        aba = p[p[p[xi0] + yi1] + zi0]
        abb = p[p[p[xi0] + yi1] + zi1]
        baa = p[p[p[xi1] + yi0] + zi0]
        bab = p[p[p[xi1] + yi0] + zi1]
        bba = p[p[p[xi1] + yi1] + zi0]
        bbb = p[p[p[xi1] + yi1] + zi1]

        gxa, gya, gza = grad3_from_hash(aaa)
        gx1, gy1, gz1 = grad3_from_hash(baa)
        gx2, gy2, gz2 = grad3_from_hash(aba)
        gx3, gy3, gz3 = grad3_from_hash(bba)
        gx4, gy4, gz4 = grad3_from_hash(aab)
        gx5, gy5, gz5 = grad3_from_hash(bab)
        gx6, gy6, gz6 = grad3_from_hash(abb)
        gx7, gy7, gz7 = grad3_from_hash(bbb)

        d000 = gxa * x0 + gya * y0 + gza * z0
        d100 = gx1 * x1 + gy1 * y0 + gz1 * z0
        d010 = gx2 * x0 + gy2 * y1 + gz2 * z0
        d110 = gx3 * x1 + gy3 * y1 + gz3 * z0
        d001 = gx4 * x0 + gy4 * y0 + gz4 * z1
        d101 = gx5 * x1 + gy5 * y0 + gz5 * z1
        d011 = gx6 * x0 + gy6 * y1 + gz6 * z1
        d111 = gx7 * x1 + gy7 * y1 + gz7 * z1

        x_lerp00 = lerp(d000, d100, u)
        x_lerp10 = lerp(d010, d110, u)
        y_lerp0 = lerp(x_lerp00, x_lerp10, v)

        x_lerp01 = lerp(d001, d101, u)
        x_lerp11 = lerp(d011, d111, u)
        y_lerp1 = lerp(x_lerp01, x_lerp11, v)

        return lerp(y_lerp0, y_lerp1, w)
