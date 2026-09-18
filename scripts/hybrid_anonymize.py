#!/usr/bin/env python
"""hybrid_anonymize.py -- anonimizare hibrida a fetelor pe un folder de imagini
echirectangulare 360 grade:
  - fete cu aria bbox >= --area-threshold: NullFace (schimbare de identitate
    via Stable Diffusion + IP-Adapter FaceID + DDPM inversion)
  - fete mai mici: segmentare eliptica orientata (landmark-uri SCRFD, cu fallback
    la elipsa axis-aligned pe fete din profil) + umplere cu culoarea de piele
    estimata din fata respectiva

Detectia foloseste pipeline-ul M8 al autorului (model_comparison_study/m8_preproc.py)
-- fara el fetele mici sunt irecunoscibile la rezolutia unui panoramic 3840x1920.

Ruleaza cu interpretorul din nullface/conda_env/ (Python 3.12) -- singurul care are
simultan torch+diffusers+insightface (pentru NullFace) si onnxruntime+opencv
(pentru SCRFD/M8).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import face_skin_fill as fsf  # noqa: E402
import nullface_cached as nf  # noqa: E402

AREA_THRESHOLD_PX2 = 20000.0
CROP_MARGIN_FRAC = 0.6  # crop = bbox expandat cu 60% pe fiecare parte, context suficient pentru NullFace
SCRFD_WEIGHTS = ROOT / "weights" / "scrfd_10g.onnx"


def load512_square_region(w: int, h: int):
    """Replica exacta a crop-ului la patrat din repo/prompt_to_prompt/ptp_classes.py::load_512
    (fara top/left/right/bottom, care sunt 0 in apelul nostru) -- NullFace nu proceseaza
    niciodata dreptunghiul intreg, doar patratul central. Returneaza (sq_x, sq_y, sq_size)
    relativ la crop, ca sa lipim rezultatul anonimizat inapoi exact unde a fost procesat,
    nu intins peste tot dreptunghiul (asta cauza deraparea vizuala observata initial)."""
    if h < w:
        offset = (w - h) // 2
        return offset, 0, h
    elif w < h:
        offset = (h - w) // 2
        return 0, offset, w
    return 0, 0, w


def crop_face_with_margin(img: np.ndarray, det, margin_frac: float):
    H, W = img.shape[:2]
    w, h = det.x2 - det.x1, det.y2 - det.y1
    mx, my = w * margin_frac, h * margin_frac
    x1 = max(0, int(det.x1 - mx))
    y1 = max(0, int(det.y1 - my))
    x2 = min(W, int(det.x2 + mx))
    y2 = min(H, int(det.y2 + my))
    return img[y1:y2, x1:x2], x1, y1, x2, y2


def process_image(fp: Path, out_dir: Path, tmp_dir: Path, model, ldm_stable, extractor,
                   log_file: Path, area_threshold: float):
    img = cv2.imread(str(fp))
    results = fsf.detect_faces_with_landmarks(img, model)
    out = img.copy()

    n_big = n_small = n_fallback = 0
    for i, (det, landmarks) in enumerate(results):
        if det.area >= area_threshold:
            crop, x1, y1, x2, y2 = crop_face_with_margin(img, det, CROP_MARGIN_FRAC)
            if crop.size == 0:
                continue
            tmp_path = tmp_dir / f"crop_{fp.stem}_{i}.png"
            cv2.imwrite(str(tmp_path), crop)
            anon = nf.anonymize_face_cached(
                ldm_stable, extractor, image_path=str(tmp_path), mask_image_path="",
                output_log_file=str(log_file),
            )
            tmp_path.unlink(missing_ok=True)
            if anon is not None:
                sq_x, sq_y, sq_size = load512_square_region(x2 - x1, y2 - y1)
                anon_bgr = cv2.cvtColor(np.array(anon), cv2.COLOR_RGB2BGR)
                anon_resized = cv2.resize(anon_bgr, (sq_size, sq_size), interpolation=cv2.INTER_LANCZOS4)
                py1, py2 = y1 + sq_y, y1 + sq_y + sq_size
                px1, px2 = x1 + sq_x, x1 + sq_x + sq_size
                out[py1:py2, px1:px2] = anon_resized
                n_big += 1
            else:
                # NullFace nu a gasit fata in crop (rar) -- nu lasam fata neprocesata,
                # cadem pe segmentare+skin-fill ca plasa de siguranta pentru confidentialitate.
                mask = fsf.build_face_mask(out.shape[:2], det, landmarks)
                color = fsf.estimate_skin_color(img, mask)
                fsf.fill_face(out, mask, color)
                n_fallback += 1
        else:
            mask = fsf.build_face_mask(out.shape[:2], det, landmarks)
            color = fsf.estimate_skin_color(img, mask)
            fsf.fill_face(out, mask, color)
            n_small += 1

    cv2.imwrite(str(out_dir / fp.name), out)
    return len(results), n_big, n_small, n_fallback


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--area-threshold", type=float, default=AREA_THRESHOLD_PX2,
                     help="arie bbox (px^2) peste care o fata merge la NullFace in loc de skin-fill")
    ap.add_argument("--device-num", type=int, default=0)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_tmp_crops"
    tmp_dir.mkdir(exist_ok=True)
    log_file = out_dir / "nullface_log.txt"
    log_file.write_text("")

    frames = sorted(in_dir.glob("*.jpg"))
    if not frames:
        print(f"Nicio imagine .jpg gasita in {in_dir}")
        return

    print("Incarc SCRFD (M8)...")
    model = fsf.ScrfdLandmarkAdapter(str(SCRFD_WEIGHTS),
                                      providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                      conf=0.4)

    print("Incarc pipeline-ul NullFace (Stable Diffusion + IP-Adapter + InsightFace)...")
    t0 = time.perf_counter()
    ldm_stable, extractor = nf.load_pipeline(device_num=args.device_num)
    print(f"  incarcat in {time.perf_counter() - t0:.1f}s")

    totals = {'faces': 0, 'big': 0, 'small': 0, 'fallback': 0}
    for fp in frames:
        t0 = time.perf_counter()
        n, n_big, n_small, n_fb = process_image(fp, out_dir, tmp_dir, model, ldm_stable, extractor,
                                                  log_file, args.area_threshold)
        dt = time.perf_counter() - t0
        totals['faces'] += n
        totals['big'] += n_big
        totals['small'] += n_small
        totals['fallback'] += n_fb
        print(f"  {fp.name}: {n} fete ({n_big} NullFace, {n_small} skin-fill, {n_fb} fallback) -- {dt:.1f}s")

    tmp_dir.rmdir()
    print(f"\n{len(frames)} imagini, {totals['faces']} fete "
          f"({totals['big']} NullFace, {totals['small']} skin-fill, {totals['fallback']} fallback).")
    print(f"Rezultat in {out_dir}/")


if __name__ == "__main__":
    main()
