"""Display-only geological cartography; never used to create scoring labels."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


INK = (24, 29, 34, 255)
OUTSIDE = (233, 231, 224, 255)
STYLE_VERSION = "source-contacts-folds-v1"


def observed_cartography(source: Image.Image, labels: np.ndarray,
                         mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Use exact legend colors at observations and original map ink/cover elsewhere.

    Retaining the source RGB in excluded pixels is not an inferred bedrock label.
    In particular, yellow Quaternary cover remains cover and is never scored.
    """
    image = np.array(source.convert("RGBA").resize(
        (labels.shape[1], labels.shape[0]), Image.Resampling.NEAREST))
    image[image[..., 3] == 0] = OUTSIDE
    image[mask] = palette[labels[mask]]
    image[..., 3] = 255
    return image


def predicted_cartography(labels: np.ndarray, palette: np.ndarray,
                          axes: list[dict], bounds: dict, scale: int = 4) -> np.ndarray:
    """Draw continuous half-pixel contacts and dashed model construction axes.

    Supersampling leaves thin rock bands visible. Raw prediction/3D textures
    remain available separately, without cartographic ink baked into the data.
    """
    height, width = labels.shape
    image_width = width * scale
    image_height = max(1, round(image_width * (bounds["ymax"] - bounds["ymin"])
                                / (bounds["xmax"] - bounds["xmin"])))
    sx, sy = image_width / width, image_height / height
    # Work in the map's physical aspect ratio so transverse arrows really are
    # perpendicular to the axis after display, including on a nonsquare map.
    image = Image.fromarray(palette[labels]).resize(
        (image_width, image_height), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    stroke = max(1, round(scale / 2))
    for row, col in np.argwhere(labels[:, 1:] != labels[:, :-1]):
        x, y = float(col + 1) * sx, float(row) * sy
        draw.line([(x, y), (x, y + sy)], fill=INK, width=stroke)
    for row, col in np.argwhere(labels[1:, :] != labels[:-1, :]):
        x, y = float(col) * sx, float(row + 1) * sy
        draw.line([(x, y), (x + sx, y)], fill=INK, width=stroke)

    def pixel(point):
        return np.array([(point[0] - bounds["xmin"]) / (bounds["xmax"] - bounds["xmin"]) * image_width,
                         (bounds["ymax"] - point[1]) / (bounds["ymax"] - bounds["ymin"]) * image_height])

    def line(points, ink=INK, thickness=0.9):
        draw.line([tuple(p) for p in points], fill=ink, width=max(1, round(thickness * scale)))

    for axis in axes:
        if axis.get("status") != "visible" or len(axis.get("polyline", [])) < 2:
            continue
        start, end = map(pixel, (axis["polyline"][0], axis["polyline"][-1]))
        length = float(np.linalg.norm(end - start))
        if length < 12 * scale:
            continue
        tangent = (end - start) / length
        normal = np.array([-tangent[1], tangent[0]])
        # Dash model axes to distinguish their inferred position from source ink.
        for distance in np.arange(0, length, 8 * scale):
            points = [start + distance * tangent, start + min(distance + 5 * scale, length) * tangent]
            line(points, (249, 246, 231, 230), 1.7)
            line(points)
        fractions = (0.38, 0.72) if length > 140 * scale else (0.5,)
        for fraction in fractions:
            center = start + fraction * (end - start)
            for side in (-1, 1):
                outward = side * normal
                tail_distance, tip_distance = ((2, 9) if axis["kind"] == "anticline" else (10, 2))
                tail = center + tail_distance * scale * outward
                tip = center + tip_distance * scale * outward
                direction = (tip - tail) / np.linalg.norm(tip - tail)
                wing = np.array([-direction[1], direction[0]])
                arrow = [tip - 3 * scale * direction + 2 * scale * wing, tip,
                         tip - 3 * scale * direction - 2 * scale * wing]
                line([tail, tip], (249, 246, 231, 255), 2.2)
                line(arrow, (249, 246, 231, 255), 2.2)
                line([tail, tip], thickness=1.0)
                line(arrow, thickness=1.0)
    return np.asarray(image)
