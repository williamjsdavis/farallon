"""Independent categorical map scores; renderer colors never affect the result."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt


def _labels(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a two-dimensional integer label array")
    return array


def _contacts(labels: np.ndarray, mask: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Internal contact pixels keyed by their unordered material pair.

    Both neighboring pixels must be observed. Image edges and the perimeter
    of a no-data hole are consequently never treated as geological contacts.
    """
    contacts: dict[tuple[int, int], np.ndarray] = {}
    for axis in (0, 1):
        left = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        right = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        first, second = labels[left], labels[right]
        changed = mask[left] & mask[right] & (first != second)
        yy, xx = np.nonzero(changed)
        if yy.size == 0:
            continue
        a, b = first[changed], second[changed]
        pairs = np.column_stack((np.minimum(a, b), np.maximum(a, b)))
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        for index, pair in enumerate(unique):
            key = (int(pair[0]), int(pair[1]))
            boundary = contacts.setdefault(key, np.zeros(labels.shape, dtype=bool))
            keep = inverse == index
            rows, columns = yy[keep], xx[keep]
            boundary[rows, columns] = True
            boundary[rows + (axis == 0), columns + (axis == 1)] = True
    return contacts


def prepare_boundary_mask(mask: np.ndarray, exclusion_reasons: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """Bridge only narrow uncertain-color gaps for approximate map contacts.

    Reason codes: 0 observed unit, 1 outside, 2 legacy unobserved cover, 3 unmatched color.
    Reason 3 includes printed linework but is not guaranteed to be linework;
    the radius is therefore deliberately small. Outside/legacy cover plus a pixel
    margin remain excluded. This mask must never replace the IoU mask.
    """
    observed, reasons = np.asarray(mask, dtype=bool), np.asarray(exclusion_reasons)
    if observed.ndim != 2 or reasons.shape != observed.shape:
        raise ValueError("mask and exclusion_reasons must have the same two-dimensional shape")
    protected = binary_dilation((reasons == 1) | (reasons == 2), iterations=1)
    near_observed = distance_transform_edt(~observed) <= radius
    return (observed | ((reasons == 3) & near_observed)) & ~protected


class MapScorer:
    """Cache target contact distances for repeated evaluation of one fixed map."""

    def __init__(self, target: np.ndarray, mask: np.ndarray | None = None, boundary_mask: np.ndarray | None = None):
        self.target = _labels(target, "target").copy()
        self.mask = np.ones(self.target.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
        if self.mask.shape != self.target.shape:
            raise ValueError("mask and target must have the same shape")
        self.count = int(np.count_nonzero(self.mask))
        if self.count == 0:
            raise ValueError("the observation mask must contain at least one pixel")
        self.target_values = self.target[self.mask]
        self.target_units = np.unique(self.target_values)
        self.boundary_mask = self.mask if boundary_mask is None else np.asarray(boundary_mask, dtype=bool).copy()
        if self.boundary_mask.shape != self.target.shape:
            raise ValueError("boundary_mask and target must have the same shape")
        boundary_target = self.target
        to_fill = self.boundary_mask & ~self.mask
        if np.any(to_fill):
            nearest = distance_transform_edt(~self.mask, return_distances=False, return_indices=True)
            boundary_target = self.target.copy()
            boundary_target[to_fill] = self.target[tuple(nearest[:, to_fill])]
        self.target_contacts = _contacts(boundary_target, self.boundary_mask)
        self.target_distances = {
            pair: distance_transform_edt(~boundary)
            for pair, boundary in self.target_contacts.items()
        }
        self.target_contact_count = sum(int(b.sum()) for b in self.target_contacts.values())
        self.missing_penalty = max(1.0, math.hypot(self.target.shape[0] - 1, self.target.shape[1] - 1))
        self.boundary_scale = max(2.0, self.missing_penalty * 0.04)

    def __call__(self, prediction: np.ndarray) -> dict:
        prediction = _labels(prediction, "prediction")
        if prediction.shape != self.target.shape:
            raise ValueError("prediction and target must have the same shape")
        values = prediction[self.mask]
        units = np.union1d(self.target_units, np.unique(values))
        per_unit: dict[str, float] = {}
        for unit in units:
            predicted = values == unit
            observed = self.target_values == unit
            union = int(np.count_nonzero(predicted | observed))
            intersection = int(np.count_nonzero(predicted & observed))
            per_unit[str(int(unit))] = intersection / union
        miou = float(np.mean(list(per_unit.values())))
        accuracy = float(np.count_nonzero(values == self.target_values) / self.count)

        predicted_contacts = _contacts(prediction, self.boundary_mask)
        predicted_count = sum(int(b.sum()) for b in predicted_contacts.values())
        if not predicted_count and not self.target_contact_count:
            boundary_error = 0.0
        elif not predicted_count or not self.target_contact_count:
            boundary_error = self.missing_penalty
        else:
            forward, backward = 0.0, 0.0
            for pair, boundary in predicted_contacts.items():
                distance = self.target_distances.get(pair)
                if distance is None:
                    forward += float(boundary.sum()) * self.missing_penalty
                else:
                    forward += float(distance[boundary].sum())
            for pair, boundary in self.target_contacts.items():
                predicted_boundary = predicted_contacts.get(pair)
                if predicted_boundary is None:
                    backward += float(boundary.sum()) * self.missing_penalty
                else:
                    backward += float(distance_transform_edt(~predicted_boundary)[boundary].sum())
            boundary_error = 0.5 * (forward / predicted_count + backward / self.target_contact_count)
        boundary_quality = math.exp(-boundary_error / self.boundary_scale)
        return {
            "miou": miou,
            "accuracy": accuracy,
            "boundary_error_px": float(boundary_error),
            "score": float(0.8 * miou + 0.2 * boundary_quality),
            "per_unit": per_unit,
            "evaluated_pixels": self.count,
            "target_contact_pixels": self.target_contact_count,
            "prediction_contact_pixels": predicted_count,
        }


def score_map(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None, *, boundary_mask: np.ndarray | None = None) -> dict:
    """Score observed classes, including classes missing from either map.

    ``score`` is a fixed ranking: 80% macro IoU and 20% an exponentially
    decreasing contact-distance quality. Contact distances compare matching
    material pairs, penalizing absent pairs by the raster diagonal.
    """
    return MapScorer(target, mask, boundary_mask)(prediction)


def mismatch_rgba(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Red mismatches, faint green agreements, transparent unobserved pixels."""
    prediction, target = _labels(prediction, "prediction"), _labels(target, "target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    observed = np.ones(target.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if observed.shape != target.shape:
        raise ValueError("mask and target must have the same shape")
    rgba = np.zeros((*target.shape, 4), dtype=np.uint8)
    rgba[observed & (prediction == target)] = (45, 210, 150, 35)
    rgba[observed & (prediction != target)] = (255, 78, 80, 210)
    return rgba
