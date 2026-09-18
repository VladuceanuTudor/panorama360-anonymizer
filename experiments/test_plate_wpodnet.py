"""test_plate_wpodnet.py -- studiu: cenzura placutelor folosind WPOD-NET
(https://github.com/Pandede/WPODNet-Pytorch), o retea dedicata regresiei celor
4 colturi ale placutei (gestioneaza corect perspectiva/inclinarea), in loc de
euristica CV clasica din test_plate_rotation.py (care confunda uneori elemente
decorative -- ex. capace de roata de rezerva -- cu placuta reala).

Metoda: acelasi crop in jurul bbox-ului (Metoda 2, reutilizat din
test_plate_rotation.py) -> WPOD-NET pe crop (convertit BGR->PIL RGB) ->
prediction.bounds (4 colturi in spatiul crop-ului) remapate in imaginea
completa. Sub prag de confidenta: fallback pe dreptunghiul axis-aligned
original, ca sa nu ramana nicio placuta necenzurata.

Ruleaza pe data/360testIMG, scrie comparatie in outputs/plate_wpodnet_test/
-- nu e inca integrat in censor_plates.py.
"""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "wpodnet_repo"))

import detectors as det  # noqa: E402
from test_plate_rotation import detect_plates_method2, censor_axis_aligned  # noqa: E402
from wpodnet import Predictor, load_wpodnet_from_checkpoint  # noqa: E402

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CONF_THRESHOLD = 0.5


def find_wpodnet_box(predictor: Predictor, img_bgr: np.ndarray, det_, margin_frac: float = 0.3):
    """Ruleaza WPOD-NET pe crop-ul din jurul bbox-ului detectat. Returneaza
    (box (4,2) in coordonate imagine completa, confidence) sau (None, confidence)
    daca predictia e sub prag -- fallback in caller."""
    H, W = img_bgr.shape[:2]
    w, h = det_.x2 - det_.x1, det_.y2 - det_.y1
    mx, my = w * margin_frac, h * margin_frac
    x1 = max(0, int(det_.x1 - mx))
    y1 = max(0, int(det_.y1 - my))
    x2 = min(W, int(det_.x2 + mx))
    y2 = min(H, int(det_.y2 + my))
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0.0

    pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    pred = predictor.predict(pil_crop, scaling_ratio=1.0)

    if pred.confidence < CONF_THRESHOLD:
        return None, pred.confidence

    box = np.array(pred.bounds, dtype=np.float32)
    box[:, 0] += x1
    box[:, 1] += y1
    return box, pred.confidence


def censor_wpodnet(img, plates, predictor, pad_frac=0.15):
    out = img.copy()
    n_wpod = 0
    confidences = []
    for d in plates:
        box, conf = find_wpodnet_box(predictor, img, d)
        confidences.append(conf)
        if box is None:
            w, h = d.x2 - d.x1, d.y2 - d.y1
            px, py = w * 0.1, h * 0.1
            x1 = max(0, int(d.x1 - px)); y1 = max(0, int(d.y1 - py))
            x2 = min(img.shape[1], int(d.x2 + px)); y2 = min(img.shape[0], int(d.y2 + py))
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
            continue
        n_wpod += 1
        center = box.mean(axis=0)
        box_padded = center + (box - center) * (1.0 + pad_frac)
        cv2.fillPoly(out, [box_padded.astype(np.int32)], (255, 255, 255))
    return out, n_wpod, confidences


def main():
    in_dir = Path(__file__).resolve().parent.parent / "data" / "360testIMG"
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "plate_wpodnet_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Incarc modelele pe {DEVICE}...")
    vehicle_model = det.YoloAdapter(str(WEIGHTS_DIR / "yolov8n_coco.pt"), device=DEVICE, conf=0.25)
    plate_model = det.YoloAdapter(str(WEIGHTS_DIR / "yolov8_plate.pt"), device=DEVICE, conf=0.25)
    warm = [np.zeros((640, 640, 3), dtype=np.uint8)] * 7
    vehicle_model(warm)
    plate_model(warm)

    wpodnet_model = load_wpodnet_from_checkpoint(str(WEIGHTS_DIR / "wpodnet.pth")).to(DEVICE)
    predictor = Predictor(wpodnet_model)

    frames = sorted(glob.glob(str(in_dir / "*.jpg")))
    total_plates = total_wpod = 0
    all_confidences = []
    for fp in frames:
        img = cv2.imread(fp)
        plates = detect_plates_method2(img, vehicle_model, plate_model)
        if not plates:
            continue
        old = censor_axis_aligned(img, plates)
        new, n_wpod, confs = censor_wpodnet(img, plates, predictor)
        total_plates += len(plates)
        total_wpod += n_wpod
        all_confidences.extend(confs)

        name = Path(fp).stem
        cv2.imwrite(str(out_dir / f"{name}_axis_aligned.jpg"), old)
        cv2.imwrite(str(out_dir / f"{name}_wpodnet.jpg"), new)
        print(f"  {Path(fp).name}: {len(plates)} placute, {n_wpod} peste prag WPOD-NET "
              f"(conf: {[round(c, 3) for c in confs]})")

    print(f"\nTotal: {total_plates} placute, {total_wpod} peste prag {CONF_THRESHOLD} "
          f"({total_plates - total_wpod} pe fallback axis-aligned).")
    print(f"Toate confidentele: {sorted(round(c, 3) for c in all_confidences)}")
    print(f"Rezultat in {out_dir}/")


if __name__ == "__main__":
    main()
