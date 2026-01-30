from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from .core import Corner2D, fade, grad2_from_hash, grad2_table, lerp, make_permutation


class Noise2D(Protocol):
    def noise(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:  # pragma: no cover
        ...


class Perlin2D:
    def __init__(self, *, seed: int = 0, grad_set: str = "diag8"):
        self.seed = int(seed)
        self.perm = make_permutation(self.seed)
        self.grad_set = str(grad_set)
        self.grad_table = grad2_table(self.grad_set)

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

        gxaa, gyaa = grad2_from_hash(aa, grad_table=self.grad_table)
        gxab, gyab = grad2_from_hash(ab, grad_table=self.grad_table)
        gxba, gyba = grad2_from_hash(ba, grad_table=self.grad_table)
        gxbb, gybb = grad2_from_hash(bb, grad_table=self.grad_table)

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
            gx, gy = grad2_from_hash(
                np.array(h, dtype=np.int32),
                grad_table=self.grad_table,
            )
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
            "interpolation": {
                "x_lerp0": float(x_lerp0),
                "x_lerp1": float(x_lerp1),
            },
            "noise": n,
        }


def fbm2(
    noise: Noise2D,
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
        total += amp * noise.noise(x * freq, y * freq)
        amp_sum += amp
        amp *= persistence
        freq *= lacunarity

    if amp_sum == 0.0:
        return total
    return total / amp_sum


def turbulence2(
    noise: Noise2D,
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
        total += amp * np.abs(noise.noise(x * freq, y * freq))
        amp_sum += amp
        amp *= persistence
        freq *= lacunarity

    if amp_sum == 0.0:
        return total
    return total / amp_sum


def ridged2(
    noise: Noise2D,
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
        signal = 1.0 - np.abs(noise.noise(x * freq, y * freq))
        signal = signal * signal
        total += amp * signal
        amp_sum += amp
        amp *= persistence
        freq *= lacunarity

    if amp_sum == 0.0:
        return total
    return total / amp_sum


def domain_warp2(
    perlin: Noise2D,
    x: np.ndarray,
    y: np.ndarray,
    *,
    octaves: int = 4,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
    warp_amp: float = 1.0,
    warp_scale: float = 1.0,
    warp_octaves: int = 2,
    warp_lacunarity: float = 2.0,
    warp_persistence: float = 0.5,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    warp_amp = float(warp_amp)
    warp_scale = float(warp_scale)
    if warp_scale <= 0.0:
        raise ValueError("warp_scale must be > 0")

    # Two decorrelated fields to perturb x and y.
    dx = fbm2(
        perlin,
        (x + 19.1) * warp_scale,
        (y + 47.2) * warp_scale,
        octaves=warp_octaves,
        lacunarity=warp_lacunarity,
        persistence=warp_persistence,
    )
    dy = fbm2(
        perlin,
        (x - 11.8) * warp_scale,
        (y + 7.3) * warp_scale,
        octaves=warp_octaves,
        lacunarity=warp_lacunarity,
        persistence=warp_persistence,
    )

    xw = x + warp_amp * dx
    yw = y + warp_amp * dy
    return fbm2(
        perlin,
        xw,
        yw,
        octaves=octaves,
        lacunarity=lacunarity,
        persistence=persistence,
    )


def tileable2d(
    func,
    x: np.ndarray,
    y: np.ndarray,
    *,
    period_x: float,
    period_y: float,
) -> np.ndarray:
    """Make an arbitrary 2D noise function tileable with given periods.

    This wraps `func(x, y)` into a periodic function with period `period_x` and
    `period_y` using a standard 4-sample blend.
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    period_x = float(period_x)
    period_y = float(period_y)
    if period_x <= 0.0 or period_y <= 0.0:
        raise ValueError("period_x and period_y must be > 0")

    x0 = np.mod(x, period_x)
    y0 = np.mod(y, period_y)
    sx = x0 / period_x
    sy = y0 / period_y

    n00 = func(x0, y0)
    n10 = func(x0 - period_x, y0)
    n01 = func(x0, y0 - period_y)
    n11 = func(x0 - period_x, y0 - period_y)

    nx0 = n00 + sx * (n10 - n00)
    nx1 = n01 + sx * (n11 - n01)
    return nx0 + sy * (nx1 - nx0)


def tileable_fbm2(
    perlin: Noise2D,
    x: np.ndarray,
    y: np.ndarray,
    *,
    period_x: float,
    period_y: float,
    octaves: int = 4,
    lacunarity: float = 2.0,
    persistence: float = 0.5,
) -> np.ndarray:
    def _base(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        return fbm2(
            perlin,
            xx,
            yy,
            octaves=octaves,
            lacunarity=lacunarity,
            persistence=persistence,
        )

    return tileable2d(_base, x, y, period_x=period_x, period_y=period_y)
