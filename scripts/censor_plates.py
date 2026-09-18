#!/usr/bin/env python3.14
"""censor_plates.py -- inlocuieste placutele de inmatriculare cu un dreptunghi alb.

Foloseste Metoda 2 (cascada), validata empiric in lpr_vehicle_roi_study.ipynb:
detectie vehicul pe crop-urile M8 -> crop pe bbox-ul fiecarui vehicul (+ marja) ->
detectie placuta direct pe acel crop de rezolutie mare -> remapare in imaginea originala.

Input: folder cu imagini .jpg echirectangulare 360 grade.
Output: folder cu aceleasi imagini, placutele inlocuite cu dreptunghi alb.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import m8_preproc as m8
import detectors as det
import lpr_roi as roi

VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck (COCO)
EQ_TILES = 6
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_models(device: str, vehicle_conf: float, plate_conf: float):
    vehicle_model = det.YoloAdapter(str(WEIGHTS_DIR / "yolov8n_coco.pt"), device=device, conf=vehicle_conf)
    plate_model = det.YoloAdapter(str(WEIGHTS_DIR / "yolov8_plate.pt"), device=device, conf=plate_conf)
    warm = [np.zeros((640, 640, 3), dtype=np.uint8)] * 7
    vehicle_model(warm)
    plate_model(warm)
    return vehicle_model, plate_model


def detect_plates_method2(img: np.ndarray, vehicle_model, plate_model, margin_frac: float) -> list:
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


def censor(img: np.ndarray, plates: list, pad_frac: float) -> np.ndarray:
    out = img.copy()
    H, W = out.shape[:2]
    for d in plates:
        w, h = d.x2 - d.x1, d.y2 - d.y1
        px, py = w * pad_frac, h * pad_frac
        x1 = max(0, int(d.x1 - px))
        y1 = max(0, int(d.y1 - py))
        x2 = min(W, int(d.x2 + px))
        y2 = min(H, int(d.y2 + py))
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--vehicle-conf", type=float, default=0.25)
    ap.add_argument("--plate-conf", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=0.2,
                     help="marja crop in jurul vehiculului, ca fractie din latimea/inaltimea lui")
    ap.add_argument("--pad", type=float, default=0.1,
                     help="marja de siguranta pe dreptunghiul alb, ca fractie din latimea/inaltimea placutei")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted({p for pat in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG") for p in in_dir.glob(pat)})
    if not paths:
        print(f"Nicio imagine .jpg gasita in {in_dir}")
        return

    print(f"Incarc modelele pe {args.device}...")
    vehicle_model, plate_model = load_models(args.device, args.vehicle_conf, args.plate_conf)

    total_plates = 0
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [SKIP] nu pot citi {p.name}")
            continue
        t0 = time.perf_counter()
        plates = detect_plates_method2(img, vehicle_model, plate_model, args.margin)
        out = censor(img, plates, args.pad)
        cv2.imwrite(str(out_dir / p.name), out)
        dt = (time.perf_counter() - t0) * 1000
        total_plates += len(plates)
        print(f"  {p.name}: {len(plates)} placute cenzurate ({dt:.0f} ms)")

    print(f"\n{len(paths)} imagini procesate, {total_plates} placute cenzurate in total.")
    print(f"Rezultat in {out_dir}/")


if __name__ == "__main__":
    main()
