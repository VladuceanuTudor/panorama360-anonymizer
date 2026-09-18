import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(r"""# Comparatie SCRFD vs YOLOv8-Face + detector placute (YOLOv8) pe pipeline M8

**Scop:** studiu comparativ pentru alegerea modelului de detectie faciala (SCRFD vs YOLOv8-face) si evaluarea unui detector de placute (YOLOv8), toate rulate prin preprocesarea M8 adaptata a autorului (tiling echirectangular multi-scala + reproiectie inversa).

**Dataset:** 9 frame-uri reale, 3840x1920, echirectangular, camera 360 grade montata pe piept/ghidon, filmare stradala in Bucuresti (`360testIMG/`).

**Modele:**
- `weights/yolov8_face.pt` -- YOLOv8n fine-tuned pe WiderFace ([arnabdhar/YOLOv8-Face-Detection](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection))
- `weights/scrfd_10g.onnx` -- SCRFD det_10g oficial insightface, via mirror ONNX [immich-app/buffalo_l](https://huggingface.co/immich-app/buffalo_l)
- `weights/yolov8_plate.pt` -- YOLOv8n fine-tuned pentru detectie placute, 1 clasa `license_plate` ([Koushim/yolov8-license-plate-detection](https://huggingface.co/Koushim/yolov8-license-plate-detection)) -- **atentie**: README-ul nu specifica tara/regiunea dataset-ului de training; validam generalizarea pe placute romanesti empiric mai jos, doar la nivel de bbox (nu OCR).

**Metodologie cheie -- crop-urile M8 se genereaza O SINGURA DATA per frame si sunt partajate intre toate cele 3 modele.** Partea costisitoare geometric (tiling + reproiectie) e amortizata o data; doar normalizarea/decodarea proprie fiecarui model (letterbox+NMS la YOLOv8 via ultralytics, decode multi-stride la SCRFD) difera si e ieftina per-crop. Astfel comparam modelele corect, fara sa triplam costul de preprocesare."""))

cells.append(nbf.v4.new_code_cell(r"""import sys, os, time, glob
import numpy as np
import cv2
import torch
import onnxruntime as ort

print("Python", sys.version.split()[0])
print("torch", torch.__version__, "| cuda build:", torch.version.cuda)
print("torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  GPU:", torch.cuda.get_device_name(0))

print("onnxruntime", ort.__version__)
print("onnxruntime providers available:", ort.get_available_providers())

has_cv_cuda = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
print("opencv", cv2.__version__, "| cv2.cuda build:", has_cv_cuda)
if not has_cv_cuda:
    print("  -> pip opencv nu are build CUDA; M8 cade automat pe cv2.resize/remap CPU (comportament corect, nu bug).")"""))

cells.append(nbf.v4.new_markdown_cell(r"""## Adaptarea M8 pentru fete/placute

Pe baza inspectiei vizuale a frame-urilor reale, am aplicat 2 modificari in `m8_preproc.py`:

1. **Calotele polare dezactivate** (`include_polar=False`) -- polul nord e cer gol, polul sud e echipamentul operatorului; niciuna nu contine fete/placute pe acest dataset.
2. **6 tile-uri ecuatoriale** (`eq_tiles=6`, fata de 4 in varianta originala) -- fetele/placutele sunt mult mai mici decat obiectele pentru care a fost gandit M8 initial; mai multe tile-uri mai inguste pastreaza mai multa rezolutie dupa downscale la 640x640."""))

cells.append(nbf.v4.new_code_cell(r"""import m8_preproc as m8

EQ_TILES_STUDY = 6
crops, meta = m8.generate_m8_crops(np.zeros((1920, 3840, 3), dtype=np.uint8),
                                    include_polar=False, eq_tiles=EQ_TILES_STUDY)
print("n crop-uri generate:", len(crops), "| meta n_tiles:", meta['n_tiles'], "| include_polar:", meta['include_polar'])"""))

cells.append(nbf.v4.new_markdown_cell(r"""## Incarcare modele

Prag de confidence: YOLOv8 (face/plate) foloseste 0.25 (default ultralytics), SCRFD foloseste 0.4 -- fiecare model isi pastreaza conventiile proprii de calibrare a scorului, nu fortam un threshold artificial comun intre arhitecturi diferite."""))

cells.append(nbf.v4.new_code_cell(r"""import detectors as det

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
ORT_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
    if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']

t0 = time.perf_counter()
face_model = det.YoloAdapter('../weights/yolov8_face.pt', device=DEVICE, conf=0.25)
plate_model = det.YoloAdapter('../weights/yolov8_plate.pt', device=DEVICE, conf=0.25)
scrfd_model = det.ScrfdAdapter('../weights/scrfd_10g.onnx', providers=ORT_PROVIDERS, conf=0.4)
print(f"Modele incarcate in {time.perf_counter()-t0:.2f}s")
print("YOLO device:", DEVICE)
print("SCRFD onnxruntime providers efectiv folosite:", scrfd_model.sess.get_providers())

MODELS = {'YOLOv8-face': face_model, 'SCRFD': scrfd_model, 'YOLOv8-plate': plate_model}

_warm = [np.zeros((640, 640, 3), dtype=np.uint8)] * 7
for _m in MODELS.values():
    _m(_warm)
print("Warm-up complet (exclude cold-start CUDA din masurarea de latenta).")"""))

cells.append(nbf.v4.new_markdown_cell("## Rulare pe toate cele 9 frame-uri: preprocesare partajata + 3 modele + reproiectie + NMS global"))

cells.append(nbf.v4.new_code_cell(r"""import pandas as pd

IMG_DIR = '../data/360testIMG'
frames = sorted(glob.glob(f'{IMG_DIR}/*.jpg'))
print(f"{len(frames)} frame-uri gasite")

rows = []
frame_dets = {}

for fp in frames:
    img = cv2.imread(fp)
    H, W = img.shape[:2]

    t0 = time.perf_counter()
    crops, meta = m8.generate_m8_crops(img, include_polar=False, eq_tiles=EQ_TILES_STUDY)
    t_prep = (time.perf_counter() - t0) * 1000

    frame_dets[fp] = {}
    for name, model in MODELS.items():
        t0 = time.perf_counter()
        raw = model(crops)
        t_inf = (time.perf_counter() - t0) * 1000

        all_dets = []
        for crop_idx, dets_640 in enumerate(raw):
            all_dets.extend(m8.reproject_to_equirect(dets_640, crop_idx, meta))
        n_raw = len(all_dets)
        final = m8.global_nms(all_dets, W=W, n_tiles=meta['n_tiles'])

        frame_dets[fp][name] = final
        areas = [d.area for d in final]
        confs = [d.confidence for d in final]
        rows.append({
            'frame': os.path.basename(fp), 'model': name,
            'n_crops': len(crops), 'n_detections_raw': n_raw,
            'n_detections_final': len(final),
            'mean_confidence': float(np.mean(confs)) if confs else 0.0,
            'mean_bbox_area_px': float(np.mean(areas)) if areas else 0.0,
            'inference_ms': t_inf, 'preprocessing_ms_shared': t_prep,
        })

df = pd.DataFrame(rows)
df"""))

cells.append(nbf.v4.new_markdown_cell("## Vizualizare: toate 3 modelele suprapuse pe fiecare frame"))

cells.append(nbf.v4.new_code_cell(r"""import matplotlib.pyplot as plt
import matplotlib.patches as patches

COLORS = {'YOLOv8-face': '#2ecc71', 'SCRFD': '#e74c3c', 'YOLOv8-plate': '#3498db'}

for fp in frames:
    img_rgb = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.imshow(img_rgb)
    for name, dets in frame_dets[fp].items():
        for d in dets:
            rect = patches.Rectangle((d.x1, d.y1), d.x2 - d.x1, d.y2 - d.y1,
                                      linewidth=1.6, edgecolor=COLORS[name], facecolor='none')
            ax.add_patch(rect)
    handles = [patches.Patch(color=c, label=n) for n, c in COLORS.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=9)
    ax.set_title(os.path.basename(fp))
    ax.axis('off')
    plt.tight_layout()
    plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## Latenta per model"))

cells.append(nbf.v4.new_code_cell(r"""lat = df.groupby('model')[['inference_ms', 'preprocessing_ms_shared']].agg(['mean', 'std'])
lat"""))

cells.append(nbf.v4.new_code_cell(r"""fig, ax = plt.subplots(figsize=(7, 4))
means = df.groupby('model')['inference_ms'].mean()
stds = df.groupby('model')['inference_ms'].std()
ax.bar(means.index, means.values, yerr=stds.values, capsize=4, color=[COLORS[m] for m in means.index])
prep_mean = df['preprocessing_ms_shared'].mean()
ax.axhline(prep_mean, color='gray', linestyle='--', label=f'preprocesare partajata (medie {prep_mean:.1f} ms)')
ax.set_ylabel('ms / frame (7 crop-uri)')
ax.set_title('Latenta medie de inferenta per model (peste 9 frame-uri)')
ax.legend()
plt.tight_layout()
plt.show()

print(f"Cost total pipeline (preproc + toate 3 modele) per frame: {prep_mean + means.sum():.1f} ms")
print("In productie ruleaza doar 1 model de fata + 1 de placuta (nu toate 3), deci costul real va fi sub suma de mai sus.")"""))

cells.append(nbf.v4.new_markdown_cell("## Numar de detectii finale per frame/model"))

cells.append(nbf.v4.new_code_cell(r"""counts = df.pivot_table(index='frame', columns='model', values='n_detections_final')
counts"""))

cells.append(nbf.v4.new_markdown_cell("## SCRFD vs YOLOv8-face: cat de mult se suprapun detectiile lor (IoU >= 0.4)"))

cells.append(nbf.v4.new_code_cell(r"""def iou(a, b):
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / (a.area + b.area - inter + 1e-6)

agreement_rows = []
for fp in frames:
    fy = frame_dets[fp]['YOLOv8-face']
    fs = frame_dets[fp]['SCRFD']
    matched_y, matched_s = set(), set()
    for i, dy in enumerate(fy):
        best_j, best_iou = -1, 0.0
        for j, ds in enumerate(fs):
            v = iou(dy, ds)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= 0.4:
            matched_y.add(i)
            matched_s.add(best_j)
    agreement_rows.append({
        'frame': os.path.basename(fp),
        'n_yolo_face': len(fy), 'n_scrfd': len(fs),
        'n_matched_iou_ge_0.4': len(matched_y),
        'yolo_only': len(fy) - len(matched_y),
        'scrfd_only': len(fs) - len(matched_s),
    })
agree_df = pd.DataFrame(agreement_rows)
agree_df"""))

cells.append(nbf.v4.new_code_cell(r"""total_y = agree_df['n_yolo_face'].sum()
total_s = agree_df['n_scrfd'].sum()
total_m = agree_df['n_matched_iou_ge_0.4'].sum()
print(f"Total detectii YOLOv8-face: {total_y}, SCRFD: {total_s}, potrivite (IoU>=0.4): {total_m}")
print(f"% din YOLOv8-face confirmate si de SCRFD: {100*total_m/max(total_y,1):.1f}%")
print(f"% din SCRFD confirmate si de YOLOv8-face: {100*total_m/max(total_s,1):.1f}%")"""))

cells.append(nbf.v4.new_markdown_cell("## Detector placute -- raport calitativ (fara al doilea model de comparat)"))

cells.append(nbf.v4.new_code_cell(r"""plate_rows = df[df['model'] == 'YOLOv8-plate'][['frame', 'n_detections_final', 'mean_confidence', 'mean_bbox_area_px']]
plate_rows"""))

cells.append(nbf.v4.new_markdown_cell(r"""## Concluzii si limitari

**Limitare metodologica importanta:** acest dataset nu are adnotari ground-truth (bbox-uri reale desenate manual), deci nu putem calcula precision/recall/mAP formal. Comparatia de mai sus e **calitativa (vizual) + de viteza**, nu un benchmark formal. Daca sunt necesare metrici formale, urmatorul pas ar fi adnotarea manuala a unui subset mic (10-20 frame-uri) cu bbox-uri reale pentru fete/placute.

**Pasul de productie:** din cele 2 modele de fata comparate aici (SCRFD, YOLOv8-face), doar **unul singur** va fi migrat impreuna cu detectorul de placute in pipeline-ul M8 adaptat pentru integrarea finala TensorRT/DeepStream discutata separat -- nu se ruleaza toate 3 modelele simultan in productie, asta a fost doar pentru acest studiu comparativ."""))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'M8 venv (Python 3.14)',
        'language': 'python',
        'name': 'm8venv',
    },
    'language_info': {'name': 'python', 'version': '3.14.4'},
}

with open('/home/adminlinux/Documents/detectie-fete-lp-360/model_comparison.ipynb', 'w') as f:
    nbf.write(nb, f)

print("notebook written, cells:", len(cells))
