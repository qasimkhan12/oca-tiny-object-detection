"""
Attention-guided crop selection (thesis Section 3.4, Algorithm 1).

Steps for one image:
  1. Binarise each class heatmap at tau (default 0.6).
  2. Connected-component analysis -> one candidate region per component,
     scored by its peak activation.
  3. Place a fixed-size crop window centred on each peak.
  4. Keep only crops that contain at least one ground-truth box.
  5. Remove redundant crops (IoU > 0.7), keeping the higher-activation one.
  6. Keep the top-k crops for the image (k = 2 in the thesis).
  7. Project ground-truth boxes into crop coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Crop:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float      # peak Grad-CAM++ activation
    cls: int          # class whose heatmap produced the crop


def read_yolo_labels(path, img_w: int, img_h: int) -> np.ndarray:
    """Read a YOLO .txt label file -> array (n, 5) of [cls, x1, y1, x2, y2] in pixels."""
    try:
        rows = np.loadtxt(path, ndmin=2)
    except (OSError, ValueError):
        return np.zeros((0, 5))
    if rows.size == 0:
        return np.zeros((0, 5))
    cls, xc, yc, w, h = rows[:, 0], rows[:, 1] * img_w, rows[:, 2] * img_h, rows[:, 3] * img_w, rows[:, 4] * img_h
    return np.stack([cls, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def candidate_crops(heatmaps: dict[int, np.ndarray], img_w: int, img_h: int,
                    crop_size: int, tau: float = 0.6) -> list[Crop]:
    """Steps 1-3: one candidate crop per high-activation connected component, per class."""
    half = crop_size // 2
    cands = []
    for cls, hm in heatmaps.items():
        binary = (hm > tau).astype(np.uint8)
        n, labels = cv2.connectedComponents(binary)
        for comp in range(1, n):
            ys, xs = np.nonzero(labels == comp)
            i = np.argmax(hm[ys, xs])
            px, py, peak = int(xs[i]), int(ys[i]), float(hm[ys[i], xs[i]])
            # centre a crop on the peak, shifted to stay inside the image
            x1 = min(max(px - half, 0), max(img_w - crop_size, 0))
            y1 = min(max(py - half, 0), max(img_h - crop_size, 0))
            cands.append(Crop(x1, y1, min(x1 + crop_size, img_w), min(y1 + crop_size, img_h), peak, cls))
    return cands


def project_labels(gt: np.ndarray, crop: Crop, min_visible: float = 0.5) -> np.ndarray:
    """
    Step 7: clip GT boxes to the crop and convert to normalised YOLO format.
    Boxes with less than `min_visible` of their area inside the crop are dropped.
    Returns (n, 5) [cls, xc, yc, w, h] normalised to the crop.
    """
    if len(gt) == 0:
        return np.zeros((0, 5))
    cw, ch = crop.x2 - crop.x1, crop.y2 - crop.y1
    x1 = np.clip(gt[:, 1], crop.x1, crop.x2)
    y1 = np.clip(gt[:, 2], crop.y1, crop.y2)
    x2 = np.clip(gt[:, 3], crop.x1, crop.x2)
    y2 = np.clip(gt[:, 4], crop.y1, crop.y2)
    area_in = (x2 - x1) * (y2 - y1)
    area = (gt[:, 3] - gt[:, 1]) * (gt[:, 4] - gt[:, 2])
    keep = (area > 0) & (area_in / np.maximum(area, 1e-9) >= min_visible)
    if not keep.any():
        return np.zeros((0, 5))
    x1, y1, x2, y2 = x1[keep] - crop.x1, y1[keep] - crop.y1, x2[keep] - crop.x1, y2[keep] - crop.y1
    out = np.stack([gt[keep, 0], (x1 + x2) / 2 / cw, (y1 + y2) / 2 / ch, (x2 - x1) / cw, (y2 - y1) / ch], axis=1)
    return np.unique(out, axis=0)  # no duplicate annotations (thesis Section 3.4)


def select_crops(heatmaps: dict[int, np.ndarray], gt: np.ndarray, img_w: int, img_h: int,
                 k: int = 2, crop_size: int = 512, tau: float = 0.6, iou_thr: float = 0.7,
                 min_visible: float = 0.5) -> list[tuple[Crop, np.ndarray]]:
    """Run steps 1-7 and return up to k (crop, projected_labels) pairs."""
    cands = candidate_crops(heatmaps, img_w, img_h, crop_size, tau)
    cands.sort(key=lambda c: c.score, reverse=True)  # strongest activation first

    selected: list[tuple[Crop, np.ndarray]] = []
    for c in cands:
        labels = project_labels(gt, c, min_visible)
        if len(labels) == 0:
            continue                                      # step 4: no background-only crops
        if any(box_iou((c.x1, c.y1, c.x2, c.y2), (s.x1, s.y1, s.x2, s.y2)) > iou_thr for s, _ in selected):
            continue                                      # step 5: redundancy filter
        selected.append((c, labels))
        if len(selected) == k:
            break                                         # step 6: top-k
    return selected
