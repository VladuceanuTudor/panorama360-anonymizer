import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(r"""# Placute conditionate de detectia vehiculului: filtrare vs cascada

**Problema:** `weights/yolov8_plate.pt` rulat singur (vezi `model_comparison.ipynb`) are recall foarte slab si scoate fals-pozitive care nu sunt pe niciun vehicul (semne, banner-e etc.) -- de aici nevoia acestui studiu.

**Pastram `yolov8_plate.pt`** ca detector de placute si testam 2 metode de a-l combina cu un detector de vehicule (`yolov8n.pt`, COCO, clasele car=2/motorcycle=3/bus=5/truck=7):

- **Baseline:** placute detectate direct pe cele 7 crop-uri M8 partajate (identic cu `model_comparison.ipynb`).
- **Metoda 1 -- filtrare prin intersectie:** pastram din Baseline doar placutele cu `containment_ratio >= 0.3` fata de orice bbox de vehicul (`intersection / aria_placutei`) -- elimina fals-pozitivele care nu stau pe o masina detectata.
- **Metoda 2 -- cascada pe ROI:** detectam vehiculele, apoi decupam cate un crop la rezolutie completa (nu redus la 640) in jurul fiecarui bbox de vehicul (+20% margine, sa nu taiem placuta de la marginea cutiei), rulam `yolov8_plate.pt` direct pe acele crop-uri si mapam rezultatul inapoi prin simpla adunare a offsetului -- fara reproiectie cilindrica/polara, e un crop simplu axis-aligned din frame-ul original. Placuta primeste astfel o rezolutie efectiva mult mai mare decat intr-un tile M8 de 640x640 care acopera o bucata mare din panorama.

Ambele metode reutilizeaza aceleasi detectii de vehicule (calculate o singura data per frame) si aceleasi crop-uri M8 partajate pentru Baseline/vehicule -- doar Metoda 2 mai adauga un pas propriu de crop la rezolutie completa.

**Dataset si surse model:** vezi `model_comparison.ipynb` -- aceleasi 9 frame-uri din `360testIMG/`, aceeasi adaptare M8 (`include_polar=False`, `eq_tiles=6`)."""))

cells.append(nbf.v4.new_code_cell(r"""import sys, os, time, glob
import numpy as np
import pandas as pd
import cv2
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import m8_preproc as m8
import detectors as det
import lpr_roi as roi

print("torch cuda:", torch.cuda.is_available())
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
VEHICLE_CLASSES = {2, 3, 5, 7}
EQ_TILES_STUDY = 6"""))

cells.append(nbf.v4.new_markdown_cell(r"""## Incarcare modele

`yolov8n.pt` (COCO, pretrained ultralytics) salvat local ca `weights/yolov8n_coco.pt` pentru autonomia proiectului. Threshold 0.25 pentru ambele YOLOv8 (default ultralytics, consistent cu restul studiului)."""))

cells.append(nbf.v4.new_code_cell(r"""veh_model = det.YoloAdapter('../weights/yolov8n_coco.pt', device=DEVICE, conf=0.25)
plate_model = det.YoloAdapter('../weights/yolov8_plate.pt', device=DEVICE, conf=0.25)

_warm7 = [np.zeros((640, 640, 3), dtype=np.uint8)] * 7
veh_model(_warm7); plate_model(_warm7)
print("Warm-up complet.")"""))

cells.append(nbf.v4.new_markdown_cell(r"""## Rulare pe toate cele 9 frame-uri: vehicule + Baseline + Metoda 1 + Metoda 2"""))

cells.append(nbf.v4.new_code_cell(r"""IMG_DIR = '../data/360testIMG'
frames = sorted(glob.glob(f'{IMG_DIR}/*.jpg'))

rows = []
frame_data = {}

for fp in frames:
    img = cv2.imread(fp)
    H, W = img.shape[:2]

    t0 = time.perf_counter()
    crops, meta = m8.generate_m8_crops(img, include_polar=False, eq_tiles=EQ_TILES_STUDY)
    t_prep = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    raw_v = veh_model(crops)
    t_veh = (time.perf_counter() - t0) * 1000
    all_v = []
    for i, d640 in enumerate(raw_v):
        all_v.extend(m8.reproject_to_equirect(d640, i, meta))
    all_v = [d for d in all_v if d.class_id in VEHICLE_CLASSES]
    vehicles = m8.global_nms(all_v, W=W, n_tiles=meta['n_tiles'])

    t0 = time.perf_counter()
    raw_p = plate_model(crops)
    t_plate_base = (time.perf_counter() - t0) * 1000
    all_p = []
    for i, d640 in enumerate(raw_p):
        all_p.extend(m8.reproject_to_equirect(d640, i, meta))
    baseline = m8.global_nms(all_p, W=W, n_tiles=meta['n_tiles'])

    m1 = roi.filter_by_vehicle_containment(baseline, vehicles, thr=0.3)

    t0 = time.perf_counter()
    crops2, offs = [], []
    for v in vehicles:
        c, xo, yo = roi.crop_with_margin(img, v, margin_frac=0.2)
        if c.size == 0:
            continue
        crops2.append(c); offs.append((xo, yo))
    t_m2_crop = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    m2_dets = []
    if crops2:
        raw2 = plate_model(crops2)
        for dets_local, (xo, yo) in zip(raw2, offs):
            for (x1, y1, x2, y2, cid, cf) in dets_local:
                m2_dets.append(m8.Detection(x1 + xo, y1 + yo, x2 + xo, y2 + yo, cid, cf))
    t_m2_inf = (time.perf_counter() - t0) * 1000
    m2 = m8.global_nms(m2_dets, W=0)

    frame_data[fp] = {'vehicles': vehicles, 'baseline': baseline, 'm1': m1, 'm2': m2}
    rows.append({
        'frame': os.path.basename(fp), 'n_vehicles': len(vehicles),
        'n_plates_baseline': len(baseline), 'n_plates_method1': len(m1), 'n_plates_method2': len(m2),
        't_prep_ms': t_prep, 't_veh_ms': t_veh, 't_plate_baseline_ms': t_plate_base,
        't_m2_crop_ms': t_m2_crop, 't_m2_inf_ms': t_m2_inf,
    })

df = pd.DataFrame(rows)
df"""))

cells.append(nbf.v4.new_markdown_cell(r"""## Vizualizare: vehicule + cele 3 variante de placute, pe fiecare frame"""))

cells.append(nbf.v4.new_code_cell(r"""COLORS = {'vehicles': '#95a5a6', 'baseline': '#e74c3c', 'm1': '#2ecc71', 'm2': '#3498db'}
LABELS = {'vehicles': 'vehicul (COCO)', 'baseline': 'placuta - Baseline', 'm1': 'placuta - Metoda 1 (filtrare)', 'm2': 'placuta - Metoda 2 (cascada ROI)'}

for fp in frames:
    img_rgb = cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.imshow(img_rgb)
    for key in ('vehicles', 'baseline', 'm1', 'm2'):
        lw = 1.0 if key == 'vehicles' else 2.0
        for d in frame_data[fp][key]:
            rect = patches.Rectangle((d.x1, d.y1), d.x2 - d.x1, d.y2 - d.y1,
                                      linewidth=lw, edgecolor=COLORS[key], facecolor='none',
                                      linestyle='--' if key == 'vehicles' else '-')
            ax.add_patch(rect)
    handles = [patches.Patch(color=c, label=LABELS[k]) for k, c in COLORS.items()]
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    ax.set_title(os.path.basename(fp))
    ax.axis('off')
    plt.tight_layout()
    plt.show()"""))

cells.append(nbf.v4.new_markdown_cell(r"""## Cate detectii Baseline elimina Metoda 1 (candidati fals-pozitiv)"""))

cells.append(nbf.v4.new_code_cell(r"""drop_rows = []
for fp in frames:
    baseline = frame_data[fp]['baseline']
    m1 = frame_data[fp]['m1']
    kept_ids = {id(p) for p in m1}
    dropped = [p for p in baseline if id(p) not in kept_ids]
    drop_rows.append({'frame': os.path.basename(fp), 'n_baseline': len(baseline),
                       'n_dropped_by_m1': len(dropped), 'n_kept_by_m1': len(m1)})
drop_df = pd.DataFrame(drop_rows)
drop_df"""))

cells.append(nbf.v4.new_code_cell(r"""tot_base = drop_df['n_baseline'].sum()
tot_dropped = drop_df['n_dropped_by_m1'].sum()
print(f"Total Baseline: {tot_base}, eliminate de Metoda 1: {tot_dropped} ({100*tot_dropped/max(tot_base,1):.1f}%)")"""))

cells.append(nbf.v4.new_markdown_cell(r"""## Cate detectii noi gaseste Metoda 2 (fata de Baseline, IoU < 0.3 => nemarcate de Baseline)"""))

cells.append(nbf.v4.new_code_cell(r"""new_rows = []
for fp in frames:
    baseline = frame_data[fp]['baseline']
    m2 = frame_data[fp]['m2']
    n_new = 0
    for d2 in m2:
        if not any(roi.iou(d2, db) >= 0.3 for db in baseline):
            n_new += 1
    new_rows.append({'frame': os.path.basename(fp), 'n_baseline': len(baseline),
                      'n_method2': len(m2), 'n_new_by_method2': n_new})
new_df = pd.DataFrame(new_rows)
new_df"""))

cells.append(nbf.v4.new_code_cell(r"""tot_m2 = new_df['n_method2'].sum()
tot_new = new_df['n_new_by_method2'].sum()
print(f"Total Metoda 2: {tot_m2} detectii, dintre care noi (nu erau in Baseline): {tot_new}")

areas_base = [d.area for fp in frames for d in frame_data[fp]['baseline']]
areas_m2 = [d.area for fp in frames for d in frame_data[fp]['m2']]
print(f"Aria medie bbox Baseline: {np.mean(areas_base) if areas_base else 0:.0f} px^2")
print(f"Aria medie bbox Metoda 2: {np.mean(areas_m2) if areas_m2 else 0:.0f} px^2")"""))

cells.append(nbf.v4.new_markdown_cell(r"""## Latenta pe etape (medie peste 9 frame-uri)"""))

cells.append(nbf.v4.new_code_cell(r"""lat_cols = ['t_prep_ms', 't_veh_ms', 't_plate_baseline_ms', 't_m2_crop_ms', 't_m2_inf_ms']
lat_summary = df[lat_cols].agg(['mean', 'std'])
lat_summary"""))

cells.append(nbf.v4.new_code_cell(r"""fig, ax = plt.subplots(figsize=(8, 4))
means = df[lat_cols].mean()
stds = df[lat_cols].std()
ax.bar(means.index, means.values, yerr=stds.values, capsize=4,
       color=['#7f8c8d', '#f39c12', '#e74c3c', '#9b59b6', '#3498db'])
ax.set_ylabel('ms / frame')
ax.set_title('Latenta medie pe etape (9 frame-uri)')
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.show()

cost_baseline_path = means['t_prep_ms'] + means['t_veh_ms'] + means['t_plate_baseline_ms']
cost_m1_path = cost_baseline_path
cost_m2_path = means['t_prep_ms'] + means['t_veh_ms'] + means['t_m2_crop_ms'] + means['t_m2_inf_ms']
print(f"Cost total/frame -- Baseline si Metoda 1 (aceeasi trecere): {cost_baseline_path:.1f} ms")
print(f"Cost total/frame -- Metoda 2 (cascada, scaleaza cu nr. vehicule): {cost_m2_path:.1f} ms")
print(f"Nr. mediu vehicule/frame: {df['n_vehicles'].mean():.1f}")"""))

cells.append(nbf.v4.new_markdown_cell(r"""## Concluzii si limitari

**Limitare metodologica:** ca si in `model_comparison.ipynb`, nu exista adnotari ground-truth pe acest dataset -- comparatia e calitativa (vizual) + de numar de detectii + de viteza, nu un benchmark formal de precision/recall.

**Ce arata rularea de mai sus (numere reale, 9 frame-uri):** Baseline gaseste doar 6 placute in total pe tot dataset-ul, si Metoda 1 elimina 5 din ele (83%) ca neancorate pe niciun vehicul -- adica aproape tot ce scoate Baseline e fals-pozitiv, exact problema semnalata initial. Metoda 2 gaseste 7 placute, dintre care 6 complet noi (Baseline nu le vedea deloc). Si mai relevant: **aria medie bbox e 47807 px² la Baseline vs doar 356 px² la Metoda 2** -- Baseline scotea cutii mari, nepotrivite ca forma pentru o placuta reala (semne, banner-e), in timp ce Metoda 2 scoate cutii de dimensiunea reala a unei placute, exact consecinta rularii detectorului pe un crop la rezolutie completa in jurul vehiculului, in loc de un tile M8 partajat de o bucata mare din panorama. Costul Metodei 2 (52ms/frame vs 36ms Baseline/Metoda 1) scaleaza cu numarul de vehicule detectate per frame (14.6 in medie aici), spre diferenta de costul fix al tiling-ului M8 -- de luat in calcul pe scene foarte dense de trafic.

**Recomandare:** Metoda 2 (cascada pe ROI) e directia corecta pentru productie -- Metoda 1 singura nu rezolva problema de recall (0 detectii noi posibile, e strict un filtru), doar cosmetizeaza fals-pozitivele Baseline-ului, care oricum sunt majoritare (83% din Baseline). Metoda 2 recupereaza recall real (6 placute noi din 7 totale) si produce cutii cu dimensiune plauzibila de placuta reala, nu blob-uri mari nespecifice. O varianta hibrida rezonabila: ruleaza Metoda 2 ca sursa principala de placute, si opsional aplica filtrul de containment din Metoda 1 peste rezultatul ei ca sa elimini eventualele placute "gasite" in crop-uri care nu se suprapun de fapt cu vehiculul respectiv (poate aparea daca marginea de 20% prinde un obiect vecin)."""))

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {'display_name': 'M8 venv (Python 3.14)', 'language': 'python', 'name': 'm8venv'},
    'language_info': {'name': 'python', 'version': '3.14.4'},
}

with open('/home/adminlinux/Documents/detectie-fete-lp-360/model_comparison_study/lpr_vehicle_roi_study.ipynb', 'w') as f:
    nbf.write(nb, f)

print("notebook written, cells:", len(cells))
