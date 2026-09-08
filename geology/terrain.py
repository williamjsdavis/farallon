"""Fixed terrain heights sampled independently of the geological history."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _coordinates(values: Any, name: str) -> np.ndarray:
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite coordinate vector")
    return result


class TerrainGrid:
    """A bilinear height raster in world kilometres, indexed as (y, x).

    Source axes may ascend or descend and need not be uniformly spaced. They
    must be strictly monotonic; a one-sample axis is also supported. Inputs are
    copied into read-only arrays, with source axes normalized to ascending order.

    Samples beyond the source *sample centres* clamp to the nearest edge. This
    includes the outer half-cell margin when sampling a raster footprint: no
    slope is extrapolated beyond the measured centres, and no wrap occurs.
    """

    def __init__(self, heights_km: Any, xs: Any, ys: Any):
        xs = _coordinates(xs, "xs")
        ys = _coordinates(ys, "ys")
        heights = np.array(heights_km, dtype=np.float64, copy=True)
        if heights.shape != (len(ys), len(xs)) or not np.isfinite(heights).all():
            raise ValueError("heights_km must be finite with shape (len(ys), len(xs))")
        for axis, name in ((xs, "xs"), (ys, "ys")):
            differences = np.diff(axis)
            if len(axis) > 1 and not (np.all(differences > 0) or np.all(differences < 0)):
                raise ValueError(f"{name} must be strictly monotonic")
        if len(xs) > 1 and xs[0] > xs[-1]:
            xs, heights = xs[::-1], heights[:, ::-1]
        if len(ys) > 1 and ys[0] > ys[-1]:
            ys, heights = ys[::-1], heights[::-1, :]
        self.xs = np.ascontiguousarray(xs)
        self.ys = np.ascontiguousarray(ys)
        self.heights_km = np.ascontiguousarray(heights)
        for array in (self.xs, self.ys, self.heights_km):
            array.setflags(write=False)

    @classmethod
    def from_npz(cls, path: str | Path) -> TerrainGrid:
        """Load heights_km, xs and ys from a pickle-free NumPy archive."""
        with np.load(path, allow_pickle=False) as data:
            missing = {"heights_km", "xs", "ys"} - set(data.files)
            if missing:
                raise ValueError(f"Terrain NPZ is missing keys: {', '.join(sorted(missing))}")
            return cls(data["heights_km"], data["xs"], data["ys"])

    def sample(self, xs: Any, ys: Any) -> np.ndarray:
        """Return float64 heights with shape (len(ys), len(xs)), in query order."""
        xs, ys = _coordinates(xs, "sample xs"), _coordinates(ys, "sample ys")
        # Interpolating fractional indices supports uneven source spacing and
        # clamps at both ends. Singleton source axes become index zero.
        x = np.interp(xs, self.xs, np.arange(len(self.xs), dtype=np.float64))
        y = np.interp(ys, self.ys, np.arange(len(self.ys), dtype=np.float64))
        x0, y0 = x.astype(np.intp), y.astype(np.intp)
        x1, y1 = np.minimum(x0 + 1, len(self.xs) - 1), np.minimum(y0 + 1, len(self.ys) - 1)
        wx, wy = x - x0, (y - y0)[:, None]
        lower = self.heights_km[y0[:, None], x0] * (1 - wx) + self.heights_km[y0[:, None], x1] * wx
        upper = self.heights_km[y1[:, None], x0] * (1 - wx) + self.heights_km[y1[:, None], x1] * wx
        return np.ascontiguousarray(lower * (1 - wy) + upper * wy)
