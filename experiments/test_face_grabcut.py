"""test_face_grabcut.py -- studiu: in loc sa umplem toata elipsa geometrica cu
culoarea de piele (acopera si fundal/par care cad accidental in interiorul
elipsei), rafinam masca prin GrabCut, initializat cu elipsa ca "foreground
probabil" -- rezultatul ar trebui sa urmareasca mai fidel conturul real al
fetei, nu doar o forma geometrica aproximativa.

Fallback pe elipsa originala daca GrabCut da un rezultat degenerat (masca goala
sau brusc mult mai mare/mica decat elipsa initiala -- semn ca a "scapat" in
fundal, plauzibil pe fete foarte mici unde nu sunt suficienti pixeli pentru un
model de culoare stabil).

Ruleaza pe data/360testIMG, scrie comparatie (elipsa vs GrabCut) in
outputs/face_grabcut_test/ -- nu e inca integrat in face_skin_fill.py/hybrid.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import face_skin_fill as fsf


def refine_mask_grabcut(img_bgr: np.ndarray, ellipse_mask: np.ndarray):
    """Rafineaza elipsa printr-un pas de GrabCut local. Returneaza
    (masca_rafinata, a_reusit: bool) -- a_reusit=False inseamna fallback."""
    ys, xs = np.where(ellipse_mask > 0)
    if len(xs) == 0:
        return ellipse_mask, False

    H, W = ellipse_mask.shape
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    w, h = x2 - x1, y2 - y1
    pad = int(0.35 * max(w, h))
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(W, x2 + pad), min(H, y2 + pad)
    roi = img_bgr[ry1:ry2, rx1:rx2]
    roi_ellipse = ellipse_mask[ry1:ry2, rx1:rx2]

    if roi.shape[0] < 15 or roi.shape[1] < 15:
        return ellipse_mask, False  # prea mic pentru un model de culoare stabil

    gc_mask = np.full(roi.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[roi_ellipse > 0] = cv2.GC_PR_FGD
    kernel = np.ones((3, 3), np.uint8)
    sure_fg = cv2.erode(roi_ellipse, kernel, iterations=3)
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return ellipse_mask, False

    refined_roi = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    ellipse_area = int((roi_ellipse > 0).sum())
    refined_area = int((refined_roi > 0).sum())
    if refined_area == 0 or refined_area > 2.5 * ellipse_area or refined_area < 0.25 * ellipse_area:
        return ellipse_mask, False  # degenerat -- a "scapat" in fundal sau a disparut

    full_refined = np.zeros_like(ellipse_mask)
    full_refined[ry1:ry2, rx1:rx2] = refined_roi
    return full_refined, True


def main():
    in_dir = Path(__file__).resolve().parent.parent / "data" / "360testIMG"
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "face_grabcut_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = fsf.ScrfdLandmarkAdapter(str(fsf.WEIGHTS),
                                      providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                      conf=0.4)

    import glob
    frames = sorted(glob.glob(str(in_dir / "*.jpg")))
    total_faces = total_refined = 0
    for fp in frames:
        img = cv2.imread(fp)
        results = fsf.detect_faces_with_landmarks(img, model)
        out_ellipse = img.copy()
        out_grabcut = img.copy()
        n_refined = 0
        for det_, landmarks in results:
            mask = fsf.build_face_mask(img.shape[:2], det_, landmarks)
            color = fsf.estimate_skin_color(img, mask)
            fsf.fill_face(out_ellipse, mask, color)

            refined_mask, ok = refine_mask_grabcut(img, mask)
            if ok:
                n_refined += 1
            fsf.fill_face(out_grabcut, refined_mask, color)

        total_faces += len(results)
        total_refined += n_refined
        name = Path(fp).stem
        cv2.imwrite(str(out_dir / f"{name}_ellipse.jpg"), out_ellipse)
        cv2.imwrite(str(out_dir / f"{name}_grabcut.jpg"), out_grabcut)
        print(f"  {Path(fp).name}: {len(results)} fete, {n_refined} rafinate cu GrabCut "
              f"({len(results) - n_refined} pe fallback elipsa)")

    print(f"\nTotal: {total_faces} fete, {total_refined} rafinate cu succes.")
    print(f"Rezultat in {out_dir}/")


if __name__ == "__main__":
    main()
