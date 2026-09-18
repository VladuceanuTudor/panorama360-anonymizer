"""test_plate_rotation.py -- studiu: cenzura placutelor cu dreptunghi ROTIT
(orientat pe forma reala a placutei, gasita prin segmentare clasica CV in jurul
bbox-ului detectat) in loc de dreptunghi axis-aligned, care pe placute inclinate
(unghi de camera, distorsiune fisheye) trebuie sa fie mai mare decat placuta
insasi ca sa o acopere complet -- deci acopera si caroserie in jur.

Metoda: crop in jurul bbox-ului (Metoda 2, ca in censor_plates.py) -> Canny
edges -> contur cu aria/aspect-ratio plauzibile pentru o placuta -> minAreaRect
pe acel contur -> patrulater rotit, umplut alb. Fallback pe dreptunghiul
axis-aligned original daca nu se gaseste un contur plauzibil.

Ruleaza pe data/360testIMG, scrie comparatie in outputs/plate_rotation_test/
(varianta veche langa cea noua, pentru inspectie vizuala) -- nu e inca
integrat in censor_plates.py.
"""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import m8_preproc as m8
import detectors as det
import lpr_roi as roi

VEHICLE_CLASSES = {2, 3, 5, 7}
EQ_TILES = 6
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def detect_plates_method2(img, vehicle_model, plate_model, margin_frac=0.2):
    crops, meta = m8.generate_m8_crops(img, include_polar=False, eq_tiles=EQ_TILES)
    raw_v = vehicle_model(crops)
    all_v = []
    for i, d640 in enumerate(raw_v):
        all_v.extend(m8.reproject_to_equirect(d640, i, meta))
    all_v = [d for d in all_v if d.class_id in VEHICLE_CLASSES]
    vehicles = m8.global_nms(all_v, W=img.shape[1], n_tiles=meta['n_tiles'])

    crops2, offs = [], []
    for v in vehicles:
        c, xo, yo = roi.crop_with_margin(img, v, margin_frac=margin_frac)
        if c.size == 0:
            continue
        crops2.append(c)
        offs.append((xo, yo))

    plates = []
    if crops2:
        raw_p = plate_model(crops2)
        for dets_local, (xo, yo) in zip(raw_p, offs):
            for (x1, y1, x2, y2, cid, cf) in dets_local:
                plates.append(m8.Detection(x1 + xo, y1 + yo, x2 + xo, y2 + yo, cid, cf))
    return m8.global_nms(plates, W=0)


def find_oriented_plate_box(img_bgr: np.ndarray, det_: m8.Detection, margin_frac: float = 0.3):
    """Cauta un patrulater rotit care aproximeaza forma reala a placutei, in
    jurul bbox-ului detectat. Returneaza (4,2) puncte in coordonate imagine
    completa, sau None daca nu gaseste un contur plauzibil (fallback in caller)."""
    H, W = img_bgr.shape[:2]
    w, h = det_.x2 - det_.x1, det_.y2 - det_.y1
    mx, my = w * margin_frac, h * margin_frac
    x1 = max(0, int(det_.x1 - mx))
    y1 = max(0, int(det_.y1 - my))
    x2 = min(W, int(det_.x2 + mx))
    y2 = min(H, int(det_.y2 + my))
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    crop_area = crop.shape[0] * crop.shape[1]
    best_rect, best_area = None, -1.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.15 * crop_area or area > 0.95 * crop_area:
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if rw < 1 or rh < 1:
            continue
        ar = max(rw, rh) / min(rw, rh)
        if not (1.5 <= ar <= 6.0):  # aspect ratio plauzibil pentru o placuta
            continue
        if area > best_area:
            best_area = area
            best_rect = rect

    if best_rect is None:
        return None

    box = cv2.boxPoints(best_rect)
    box[:, 0] += x1
    box[:, 1] += y1
    return box


def censor_axis_aligned(img, plates, pad_frac=0.1):
    out = img.copy()
    H, W = out.shape[:2]
    for d in plates:
        w, h = d.x2 - d.x1, d.y2 - d.y1
        px, py = w * pad_frac, h * pad_frac
        x1 = max(0, int(d.x1 - px)); y1 = max(0, int(d.y1 - py))
        x2 = min(W, int(d.x2 + px)); y2 = min(H, int(d.y2 + py))
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
    return out


def censor_oriented(img, plates, pad_frac=0.12):
    out = img.copy()
    n_oriented = 0
    for d in plates:
        box = find_oriented_plate_box(img, d)
        if box is None:
            w, h = d.x2 - d.x1, d.y2 - d.y1
            px, py = w * pad_frac, h * pad_frac
            x1 = max(0, int(d.x1 - px)); y1 = max(0, int(d.y1 - py))
            x2 = min(img.shape[1], int(d.x2 + px)); y2 = min(img.shape[0], int(d.y2 + py))
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
            continue
        n_oriented += 1
        center = box.mean(axis=0)
        box_padded = center + (box - center) * (1.0 + pad_frac)
        cv2.fillPoly(out, [box_padded.astype(np.int32)], (255, 255, 255))
    return out, n_oriented


def main():
    in_dir = Path(__file__).resolve().parent.parent / "data" / "360testIMG"
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plate_rotation_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Incarc modelele pe {DEVICE}...")
    vehicle_model = det.YoloAdapter(str(WEIGHTS_DIR / "yolov8n_coco.pt"), device=DEVICE, conf=0.25)
    plate_model = det.YoloAdapter(str(WEIGHTS_DIR / "yolov8_plate.pt"), device=DEVICE, conf=0.25)
    warm = [np.zeros((640, 640, 3), dtype=np.uint8)] * 7
    vehicle_model(warm)
    plate_model(warm)

    frames = sorted(glob.glob(str(in_dir / "*.jpg")))
    total_plates = total_oriented = 0
    for fp in frames:
        img = cv2.imread(fp)
        plates = detect_plates_method2(img, vehicle_model, plate_model)
        if not plates:
            continue
        old = censor_axis_aligned(img, plates)
        new, n_or = censor_oriented(img, plates)
        total_plates += len(plates)
        total_oriented += n_or

        name = Path(fp).stem
        cv2.imwrite(str(out_dir / f"{name}_axis_aligned.jpg"), old)
        cv2.imwrite(str(out_dir / f"{name}_oriented.jpg"), new)
        print(f"  {Path(fp).name}: {len(plates)} placute, {n_or} cu contur orientat gasit")

    print(f"\nTotal: {total_plates} placute, {total_oriented} cu dreptunghi orientat "
          f"({total_plates - total_oriented} pe fallback axis-aligned).")
    print(f"Rezultat in {out_dir}/")


if __name__ == "__main__":
    main()
