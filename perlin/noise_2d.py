from __future__ import annotations

import math

import numpy as np

from .core import Corner2D, fade, grad2_from_hash, lerp, make_permutation


class Perlin2D:
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

        gxaa, gyaa = grad2_from_hash(aa)
        gxab, gyab = grad2_from_hash(ab)
        gxba, gyba = grad2_from_hash(ba)
        gxbb, gybb = grad2_from_hash(bb)

        x0 = xf
        y0 = yf
        x1 = xf - 1.0
        y1 = yf - 1.0

        d00 = gxaa * x0 + gyaa * y0
        d01 = gxab * x0 + gyab * y1
        d10 = gxba * x1 + gyba * y0
        d11 = gxbb * x1 + gybb * y1

        x_lerp0 = lerp(d00, d10, u)
        x_lerp1 = lerp(d01, d11, u)
        return lerp(x_lerp0, x_lerp1, v)

    def debug_point(self, x: float, y: float) -> dict:
        # Scalar breakdown for teaching/inspection.
        xf = float(x)
        yf = float(y)
        xi0 = int(np.floor(xf)) & 255
        yi0 = int(np.floor(yf)) & 255
        xi1 = (xi0 + 1) & 255
        yi1 = (yi0 + 1) & 255

        xrel = xf - math.floor(xf)
        yrel = yf - math.floor(yf)

        u = float(fade(np.array(xrel, dtype=np.float64)))
        v = float(fade(np.array(yrel, dtype=np.float64)))

        p = self.perm
        aa = int(p[p[xi0] + yi0])
        ab = int(p[p[xi0] + yi1])
        ba = int(p[p[xi1] + yi0])
        bb = int(p[p[xi1] + yi1])

        def corner(h: int, dx: float, dy: float) -> Corner2D:
            gx, gy = grad2_from_hash(np.array(h, dtype=np.int32))
            gx = float(gx)
            gy = float(gy)
            return Corner2D(gx=gx, gy=gy, dx=dx, dy=dy, dot=(gx * dx + gy * dy))

        c00 = corner(aa, xrel, yrel)
        c10 = corner(ba, xrel - 1.0, yrel)
        c01 = corner(ab, xrel, yrel - 1.0)
        c11 = corner(bb, xrel - 1.0, yrel - 1.0)

        x_lerp0 = lerp(np.array(c00.dot), np.array(c10.dot), np.array(u))
        x_lerp1 = lerp(np.array(c01.dot), np.array(c11.dot), np.array(u))
        n = float(lerp(x_lerp0, x_lerp1, np.array(v)))

        return {
            "seed": self.seed,
            "input": {"x": xf, "y": yf},
            "cell": {"xi0": xi0, "yi0": yi0, "xi1": xi1, "yi1": yi1},
            "relative": {"xf": xrel, "yf": yrel},
            "fade": {"u": u, "v": v},
            "hash": {"aa": aa, "ab": ab, "ba": ba, "bb": bb},
            "corners": {
                "c00": c00.__dict__,
                "c10": c10.__dict__,
                "c01": c01.__dict__,
                "c11": c11.__dict__,
            },
            "noise": n,
        }


def fbm2(
    perlin: Perlin2D,
    x: np.ndarray,
    y: np.ndarray,
    *,
    octaves: int = 4,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    octaves = int(octaves)
    lacunarity = float(lacunarity)
    persistence = float(persistence)

    amp = 1.0
    freq = 1.0
    total = np.zeros_like(x, dtype=np.float64)
    amp_sum = 0.0

    for _ in range(max(octaves, 1)):
        total += amp * perlin.noise(x * freq, y * freq)
        amp_sum += amp
        amp *= persistence
        freq *= lacunarity

    if amp_sum == 0.0:
        return total
    return total / amp_sum
