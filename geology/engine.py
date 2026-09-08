"""Fast categorical geology from reversible analytic events.

The event program is explicit and the output is an explicit labeled volume.
This is a kinematic approximation, not a mechanical or balanced fold model.
Folds preserve *vertical* intervals, not normal-to-bedding thickness.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit

from .history import PARAM_SPECS, HistoryError, history_to_program, parse_history, validate_history


_ANTICLINE, _TILT, _FAULT, _INTRUSION, _ERODE = 1, 2, 3, 4, 5
_WIDTH = 16


def _pack(history: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history = validate_history(history)
    initial = history["events"][0]
    codes, rows = [], []
    for event in history["events"][1:]:
        kind = event["type"]
        p = np.zeros(_WIDTH, dtype=np.float64)
        if kind == "anticline":
            if event["uplift"] == 0:
                continue
            az = math.radians(event["azimuth"])
            p[:12] = [event["x"], event["y"], math.sin(az), math.cos(az),
                       event["uplift"], math.tan(math.radians(event["dip_ne"])),
                       math.tan(math.radians(event["dip_sw"])), event["hinge"],
                       event["nw_length"], event["se_length"],
                       math.tan(math.radians(event["plunge_nw"])),
                       math.tan(math.radians(event["plunge_se"]))]
            code = _ANTICLINE
        elif kind == "tilt":
            if event["angle"] == 0:
                continue
            az, angle = math.radians(event["azimuth"]), math.radians(-event["angle"])
            p[:7] = [event["x"], event["y"], event["z"], math.sin(az),
                      math.cos(az), math.cos(angle), math.sin(angle)]
            code = _TILT
        elif kind == "fault":
            if event["slip"] == 0:
                continue
            az, dip = math.radians(event["azimuth"]), math.radians(event["dip"])
            sa, ca, sd, cd, slip = math.sin(az), math.cos(az), math.sin(dip), math.cos(dip), event["slip"]
            # Strike azimuth; plane dips to its right. Positive slip moves the
            # positive-normal block down dip. The slip is tangent to the plane.
            p[:9] = [event["x"], event["y"], event["z"], ca * sd, -sa * sd, cd,
                      slip * ca * cd, -slip * sa * cd, -slip * sd]
            code = _FAULT
        elif kind == "intrusion":
            p[:7] = [event["x"], event["y"], event["z"], 1 / event["rx"],
                      1 / event["ry"], 1 / event["rz"], event["unit"]]
            code = _INTRUSION
        else:
            p[0] = event["level"]
            code = _ERODE
        codes.append(code)
        rows.append(p)
    return (np.asarray(codes, dtype=np.int8),
            np.asarray(rows, dtype=np.float64).reshape((-1, _WIDTH)),
            np.asarray(initial["levels"], dtype=np.float64),
            np.asarray(initial["units"], dtype=np.uint8))


@njit(cache=True, inline="always")
def _positive_part(t: float, width: float) -> float:
    """C1 onset of a linear ramp, identically zero before its onset."""
    if t <= 0.0:
        return 0.0
    if t < width:
        return t * t / (2.0 * width)
    return t - 0.5 * width


@njit(cache=True, inline="always")
def _fold_height(x: float, y: float, p: np.ndarray) -> float:
    dx, dy = x - p[0], y - p[1]
    # For azimuth=135: u points NE and v points SE.
    u, v = -p[3] * dx + p[2] * dy, p[2] * dx + p[3] * dy
    slope = p[5] if u >= 0 else p[6]
    cross_drop = slope * (math.sqrt(u * u + p[7] * p[7]) - p[7])
    end_drop = p[10] * _positive_part(-v - p[8], p[7])
    end_drop += p[11] * _positive_part(v - p[9], p[7])
    # A finite uplift leaves a flat far field; the foot is a sharp bend. This
    # bounded simplification makes zero uplift a genuine identity operation.
    return max(0.0, p[4] - cross_drop - end_drop)


@njit(cache=True, inline="always")
def _label(x: float, y: float, z: float, codes: np.ndarray, params: np.ndarray,
           levels: np.ndarray, units: np.ndarray) -> int:
    for i in range(len(codes) - 1, -1, -1):
        code, p = codes[i], params[i]
        if code == _ERODE:
            if z > p[0]:
                return 0
        elif code == _ANTICLINE:
            z -= _fold_height(x, y, p)
        elif code == _TILT:
            dx, dy, dz = x - p[0], y - p[1], z - p[2]
            dot = p[3] * dx + p[4] * dy
            x = p[0] + dx * p[5] + p[4] * dz * p[6] + p[3] * dot * (1.0 - p[5])
            y = p[1] + dy * p[5] - p[3] * dz * p[6] + p[4] * dot * (1.0 - p[5])
            z = p[2] + dz * p[5] + (p[3] * dy - p[4] * dx) * p[6]
        elif code == _FAULT:
            side = (x - p[0]) * p[3] + (y - p[1]) * p[4] + (z - p[2]) * p[5]
            if side > 0.0:
                x -= p[6]
                y -= p[7]
                z -= p[8]
        elif code == _INTRUSION:
            dx, dy, dz = (x - p[0]) * p[3], (y - p[1]) * p[4], (z - p[2]) * p[5]
            if dx * dx + dy * dy + dz * dz <= 1.0:
                return int(p[6])
    index = 0
    # Half-open strata: an exact contact belongs to the upper/younger unit.
    while index < len(levels) and z >= levels[index]:
        index += 1
    return int(units[index])


@njit(cache=True, nogil=True)
def _points_kernel(points: np.ndarray, codes: np.ndarray, params: np.ndarray,
                   levels: np.ndarray, units: np.ndarray) -> np.ndarray:
    result = np.empty(len(points), dtype=np.uint8)
    for i in range(len(points)):
        result[i] = _label(points[i, 0], points[i, 1], points[i, 2], codes, params, levels, units)
    return result


@njit(cache=True, nogil=True)
def _map_kernel(xs: np.ndarray, ys: np.ndarray, terrain: np.ndarray,
                codes: np.ndarray, params: np.ndarray, levels: np.ndarray,
                units: np.ndarray) -> np.ndarray:
    result = np.empty((len(ys), len(xs)), dtype=np.uint8)
    for j in range(len(ys)):
        for i in range(len(xs)):
            result[j, i] = _label(xs[i], ys[j], terrain[j, i] - 1e-8, codes, params, levels, units)
    return result


@njit(cache=True, nogil=True)
def _volume_kernel(bounds: np.ndarray, nx: int, ny: int, nz: int, codes: np.ndarray,
                   params: np.ndarray, levels: np.ndarray, units: np.ndarray) -> np.ndarray:
    result = np.empty((nz, ny, nx), dtype=np.uint8)
    dx, dy, dz = (bounds[1] - bounds[0]) / nx, (bounds[3] - bounds[2]) / ny, (bounds[5] - bounds[4]) / nz
    for k in range(nz):
        z = bounds[4] + (k + 0.5) * dz
        for j in range(ny):
            y = bounds[2] + (j + 0.5) * dy
            for i in range(nx):
                x = bounds[0] + (i + 0.5) * dx
                result[k, j, i] = _label(x, y, z, codes, params, levels, units)
    return result


def evaluate_points(history: Any, points: Any) -> np.ndarray:
    """Return categorical uint8 labels for N×3 world points; 0 denotes air."""
    packed = _pack(history)
    array = np.ascontiguousarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise ValueError("points must be a finite N×3 numeric array")
    if len(array) > 20_000_000:
        raise ValueError("At most 20 million points per evaluation")
    return _points_kernel(array, *packed)


def render_map(history: Any, xs: Any, ys: Any, terrain: Any = None) -> np.ndarray:
    """Sample rock just below the fixed observation surface.

    xs and ys are world coordinate vectors, in caller-selected order. Descending
    ys gives a north-up image. terrain is a scalar or (len(ys), len(xs)) array;
    None means z=0. Air at the nominated terrain is retained as a mismatch.
    """
    packed = _pack(history)
    xs, ys = np.ascontiguousarray(xs, dtype=np.float64), np.ascontiguousarray(ys, dtype=np.float64)
    if xs.ndim != 1 or ys.ndim != 1 or not len(xs) or not len(ys):
        raise ValueError("xs and ys must be nonempty coordinate vectors")
    if not np.isfinite(xs).all() or not np.isfinite(ys).all() or len(xs) * len(ys) > 4_194_304:
        raise ValueError("Map coordinates must be finite, with at most 2048² samples")
    shape = (len(ys), len(xs))
    if terrain is None or np.ndim(terrain) == 0:
        height = 0.0 if terrain is None else float(terrain)
        if not math.isfinite(height):
            raise ValueError("terrain must be finite")
        elevations = np.full(shape, height, dtype=np.float64)
    else:
        elevations = np.ascontiguousarray(terrain, dtype=np.float64)
        if elevations.shape != shape or not np.isfinite(elevations).all():
            raise ValueError("terrain must match the map shape and be finite")
    return _map_kernel(xs, ys, elevations, *packed)


def render_volume(history: Any, bounds: Any, resolution: Any = 96) -> np.ndarray:
    """Sample voxel centres in a z,y,x array; all volume axes ascend.

    bounds=[xmin,xmax,ymin,ymax,zmin,zmax]. resolution is an integer or an
    (nx,ny,nz) tuple. The returned shape is (nz,ny,nx), not (nx,ny,nz).
    """
    packed = _pack(history)
    box = np.ascontiguousarray(bounds, dtype=np.float64)
    if box.shape != (6,) or not np.isfinite(box).all() or any(box[i] >= box[i + 1] for i in (0, 2, 4)):
        raise ValueError("bounds requires six finite values with positive extents")
    if isinstance(resolution, (int, np.integer)) and not isinstance(resolution, bool):
        sizes = (int(resolution),) * 3
    elif isinstance(resolution, (list, tuple)) and len(resolution) == 3:
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in resolution):
            raise ValueError("resolution dimensions must be integers")
        sizes = tuple(int(v) for v in resolution)
    else:
        raise ValueError("resolution must be an integer or (nx,ny,nz)")
    if any(n < 1 or n > 512 for n in sizes) or math.prod(sizes) > 16_777_216:
        raise ValueError("Volume resolution must fit within 256³ total voxels")
    return _volume_kernel(box, *sizes, *packed)
