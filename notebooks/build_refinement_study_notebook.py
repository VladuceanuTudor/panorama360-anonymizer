from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(r"""# Studiu de rafinare: orientare precisa la placute + segmentare precisa la fete

**Scop:** doua rafinari experimentale ale pipeline-ului final, testate pe cele 9 cadre din `data/360testIMG`, inca neintegrate in `scripts/censor_plates.py` / `src/face_skin_fill.py`:

1. **Placute -- 3 metode comparate:** dreptunghi axis-aligned (baseline curent) vs. contur orientat gasit prin CV clasic (Canny+minAreaRect) vs. **WPOD-NET** (retea dedicata regresiei celor 4 colturi ale placutei, ECCV 2018 -- [Pandede/WPODNet-Pytorch](https://github.com/Pandede/WPODNet-Pytorch), GPL-3.0, compatibil cu AGPL-3.0-ul acestui repo).
2. **Fete -- elipsa vs. GrabCut:** elipsa orientata curenta (`face_skin_fill.py`) vs. rafinare prin OpenCV GrabCut, initializat cu elipsa ca "foreground probabil".

Notebook-ul e local-only (contine imagini reale, neanonimizate, din `data/360testIMG` -- exclus explicit din git, vezi `.gitignore`)."""))

cells.append(nbf.v4.new_code_cell(r"""import sys, glob
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd().parent / "src"))
sys.path.insert(0, str(Path.cwd().parent / "experiments"))
sys.path.insert(0, str(Path.cwd().parent / "experiments" / "wpodnet_repo"))

import detectors as det
import face_skin_fill as fsf
from test_plate_rotation import detect_plates_method2, find_oriented_plate_box
from test_plate_wpodnet import find_wpodnet_box
from test_face_grabcut import refine_mask_grabcut
from wpodnet import Predictor, load_wpodnet_from_checkpoint

DATA_DIR = Path.cwd().parent / "data" / "360testIMG"
WEIGHTS_DIR = Path.cwd().parent / "weights"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def zoom_pair(images, box_xyxy, pad_frac=1.5, min_pad=20, scale=None):
    x1, y1, x2, y2 = box_xyxy
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    pad = int(max(x2 - x1, y2 - y1) * pad_frac) + min_pad
    crops = []
    for img in images:
        H, W = img.shape[:2]
        cx1, cy1 = max(0, cx - pad), max(0, cy - pad)
        cx2, cy2 = min(W, cx + pad), min(H, cy + pad)
        crops.append(img[cy1:cy2, cx1:cx2])
    h0 = min(c.shape[0] for c in crops)
    if h0 == 0:
        return None
    s = scale or max(1, 260 // h0)
    resized = [cv2.resize(c, (c.shape[1] * s, c.shape[0] * s), interpolation=cv2.INTER_NEAREST) for c in crops]
    return np.hstack(resized)"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 1. Placute -- baseline vs. contur CV vs. WPOD-NET

Toate 3 metodele pornesc din aceleasi detectii (Metoda 2 -- cascada vehicul-placuta, `detect_plates_method2`), aplicate pe toate cele 7 placute detectate in cele 4 cadre care contin masini vizibile."""))

cells.append(nbf.v4.new_code_cell(r"""vehicle_model = det.YoloAdapter(str(WEIGHTS_DIR / "yolov8n_coco.pt"), device=DEVICE, conf=0.25)
plate_model = det.YoloAdapter(str(WEIGHTS_DIR / "yolov8_plate.pt"), device=DEVICE, conf=0.25)
warm = [np.zeros((640, 640, 3), dtype=np.uint8)] * 7
vehicle_model(warm); plate_model(warm)

wpodnet_model = load_wpodnet_from_checkpoint(str(WEIGHTS_DIR / "wpodnet.pth")).to(DEVICE)
wpodnet_predictor = Predictor(wpodnet_model)

print("Modele incarcate.")"""))

cells.append(nbf.v4.new_code_cell(r"""from PIL import Image

def censor_axis_aligned_single(img, d, pad_frac=0.1):
    out = img.copy()
    w, h = d.x2 - d.x1, d.y2 - d.y1
    px, py = w * pad_frac, h * pad_frac
    x1 = max(0, int(d.x1 - px)); y1 = max(0, int(d.y1 - py))
    x2 = min(img.shape[1], int(d.x2 + px)); y2 = min(img.shape[0], int(d.y2 + py))
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
    return out

def censor_box(img, box, pad_frac=0.12):
    out = img.copy()
    center = box.mean(axis=0)
    box_padded = center + (box - center) * (1.0 + pad_frac)
    cv2.fillPoly(out, [box_padded.astype(np.int32)], (255, 255, 255))
    return out

frames = sorted(glob.glob(str(DATA_DIR / "*.jpg")))
plate_rows = []
for fp in frames:
    img = cv2.imread(fp)
    plates = detect_plates_method2(img, vehicle_model, plate_model)
    for d in plates:
        baseline = censor_axis_aligned_single(img, d)

        cv_box = find_oriented_plate_box(img, d)
        cv_img = censor_box(img, cv_box) if cv_box is not None else baseline.copy()

        wpod_box, conf = find_wpodnet_box(wpodnet_predictor, img, d)
        wpod_img = censor_box(img, wpod_box) if wpod_box is not None else baseline.copy()

        pair = zoom_pair([img, baseline, cv_img, wpod_img], (d.x1, d.y1, d.x2, d.y2))
        plate_rows.append((Path(fp).name, pair, conf, cv_box is not None))

print(f"{len(plate_rows)} placute comparate.")"""))

cells.append(nbf.v4.new_code_cell(r"""fig, axes = plt.subplots(len(plate_rows), 1, figsize=(14, 3 * len(plate_rows)))
if len(plate_rows) == 1:
    axes = [axes]
for ax, (name, pair, conf, cv_found) in zip(axes, plate_rows):
    ax.imshow(cv2.cvtColor(pair, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{name} -- original | baseline | contur CV{' (gasit)' if cv_found else ' (fallback)'} | "
                 f"WPOD-NET (conf={conf:.3f})", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell(r"""### Concluzie -- placute

Rezultatele reale de mai sus arata:

- **Contur CV (Canny + minAreaRect):** functioneaza bine pe cazuri clare (placuta cu margine bine contrastata), dar pe cateva SUV-uri cu roata de rezerva pe spate confunda conturul decorativ circular de pe capacul rotii cu placuta reala -- rezultat inconsistent.
- **WPOD-NET:** confidenta a ramas sub pragul de 0.5 pe **toate** cele 7 placute (fallback total pe baseline). Cauza diagnosticata direct, nu presupusa: crop-urile din jurul bbox-ului de placuta sunt minuscule (ex. 17x47px), iar chiar rulat pe crop-ul intreg de vehicul (pana la 879x456px pentru masina cea mai apropiata), confidenta maxima obtinuta a fost doar 0.34 -- fata de ~0.99 pe imaginile lor proprii de test (fotografii apropiate de masina). WPOD-NET e antrenat pe fotografii de masini relativ apropiate; pe un panoramic 360 unde majoritatea masinilor sunt mici/distante, nu are suficienta rezolutie sursa ca sa functioneze.
- **Recomandare:** niciuna din cele doua rafinari nu e gata de integrare in forma testata. Contur CV are potential daca se adauga un filtru mai strict (ex. uniformitate de culoare in interior, nu doar arie+aspect-ratio). WPOD-NET ar avea nevoie fie de o sursa cu rezolutie mai mare, fie de fine-tuning specific pe placute mici/distante -- nu e o solutie "out of the box" pentru acest dataset."""))

cells.append(nbf.v4.new_markdown_cell(r"""## 2. Fete -- elipsa vs. GrabCut

Comparatie pe fete reprezentative: cea mai mare fata din dataset (cazul cel mai clar) plus cateva mai mici."""))

cells.append(nbf.v4.new_code_cell(r"""scrfd_model = fsf.ScrfdLandmarkAdapter(str(fsf.WEIGHTS), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], conf=0.4)

all_faces = []
for fp in frames:
    img = cv2.imread(fp)
    results = fsf.detect_faces_with_landmarks(img, scrfd_model)
    for d, lm in results:
        all_faces.append((fp, img, d, lm))

all_faces.sort(key=lambda r: -r[2].area)
selected = [all_faces[0]] + all_faces[1::7][:5]
print(f"{len(all_faces)} fete detectate total, {len(selected)} selectate pentru comparatie vizuala.")"""))

cells.append(nbf.v4.new_code_cell(r"""face_rows = []
for fp, img, d, lm in selected:
    ellipse_mask = fsf.build_face_mask(img.shape[:2], d, lm)
    color = fsf.estimate_skin_color(img, ellipse_mask)

    ellipse_img = img.copy()
    fsf.fill_face(ellipse_img, ellipse_mask, color)

    refined_mask, ok = refine_mask_grabcut(img, ellipse_mask)
    grabcut_img = img.copy()
    fsf.fill_face(grabcut_img, refined_mask, color)

    pair = zoom_pair([img, ellipse_img, grabcut_img], (d.x1, d.y1, d.x2, d.y2), pad_frac=0.8)
    face_rows.append((Path(fp).name, pair, d.area, ok))

fig, axes = plt.subplots(len(face_rows), 1, figsize=(12, 3.2 * len(face_rows)))
if len(face_rows) == 1:
    axes = [axes]
for ax, (name, pair, area, ok) in zip(axes, face_rows):
    ax.imshow(cv2.cvtColor(pair, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{name} (arie={area:.0f}px^2) -- original | elipsa | GrabCut{' (rafinat)' if ok else ' (fallback pe elipsa)'}",
                 fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell(r"""### Concluzie -- fete

Pe cea mai mare fata din dataset (cazul cu cei mai multi pixeli utili, deci cel mai favorabil pentru GrabCut), rezultatul e clar negativ: masca GrabCut se extinde masiv in par si cer, inclusiv un petic deconectat de fundal langa un copac -- mult mai rau decat elipsa simpla. Pe fetele mici diferenta e neglijabila (majoritatea cad pe fallback din cauza rezolutiei insuficiente pentru un model de culoare stabil).

**Recomandare:** GrabCut, asa cum a fost testat, nu aduce niciun beneficiu si strica rezultatul pe cel mai bun caz posibil. Ramane elipsa orientata curenta din `face_skin_fill.py`, fara modificari."""))

cells.append(nbf.v4.new_markdown_cell(r"""## Recomandare finala

Nicio rafinare testata aici nu e pregatita pentru integrare in `scripts/censor_plates.py` sau `src/face_skin_fill.py`:

- **Placute:** baseline-ul axis-aligned actual ramane cel mai sigur -- ambele alternative (contur CV, WPOD-NET) au eșuat sau au fost inconsistente pe acest dataset specific de placute mici/distante.
- **Fete:** elipsa orientata actuala ramane neschimbata -- GrabCut a produs un rezultat clar mai prost pe cazul cel mai favorabil testat.

Daca se doreste imbunatatire ulterioara: contur CV cu filtrare mai stricta (uniformitate culoare), sau un detector de placute cu 4 colturi antrenat specific pe date la rezolutie mica/distanta (WPOD-NET fine-tuned), pentru placute; nicio directie clara de urmat pentru fete in acest moment."""))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'M8 venv (Python 3.14)', 'language': 'python', 'name': 'm8venv'},
    'language_info': {'name': 'python', 'version': '3.14.4'},
}

out_path = Path(__file__).resolve().parent / "studiu_rafinare_placute_fete.ipynb"
with open(out_path, 'w') as f:
    nbf.write(nb, f)
print("notebook written, cells:", len(cells))
