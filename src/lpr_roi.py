"""lpr_roi.py — helpers for vehicle-conditioned plate detection (Method 1 & 2).

Both methods reuse the same vehicle Detections (in full equirect pixel space,
already reprojected+NMS'd from the shared M8 crops) — Method 1 filters the
baseline plate detections by containment inside a vehicle box, Method 2 runs
the plate model directly on raw-resolution crops taken around each vehicle box.
"""
import numpy as np
from m8_preproc import Detection


def containment_ratio(inner: Detection, outer: Detection) -> float:
    """Fraction of `inner`'s own area that overlaps `outer` — not symmetric IoU,
    since a plate is always much smaller than the vehicle it sits on."""
    ix1, iy1 = max(inner.x1, outer.x1), max(inner.y1, outer.y1)
    ix2, iy2 = min(inner.x2, outer.x2), min(inner.y2, outer.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inner.area <= 0:
        return 0.0
    return inter / inner.area


def filter_by_vehicle_containment(plates: list[Detection], vehicles: list[Detection],
                                   thr: float = 0.3) -> list[Detection]:
    return [p for p in plates if any(containment_ratio(p, v) >= thr for v in vehicles)]


def iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    return inter / (a.area + b.area - inter + 1e-6)


def crop_with_margin(frame_bgr: np.ndarray, det: Detection, margin_frac: float = 0.2):
    """Axis-aligned crop of the ORIGINAL full-res frame around `det`, expanded by
    `margin_frac` of its own width/height — plates often sit right at a vehicle
    box's edge (bumper/hood), a tight crop can cut them off. Returns (crop, x_off, y_off)."""
    H, W = frame_bgr.shape[:2]
    w, h = det.x2 - det.x1, det.y2 - det.y1
    mx, my = w * margin_frac, h * margin_frac
    x1 = max(0, int(det.x1 - mx))
    y1 = max(0, int(det.y1 - my))
    x2 = min(W, int(det.x2 + mx))
    y2 = min(H, int(det.y2 + my))
    return frame_bgr[y1:y2, x1:x2], x1, y1
