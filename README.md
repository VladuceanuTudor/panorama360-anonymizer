# detectie-fete-lp-360

Anonimizare de fețe și plăcuțe de înmatriculare în dataset-uri panoramice 360°
(echirectangulare, tip cameră montată pe piept/ghidon). Pipeline construit pe
metoda de preprocesare M8 a autorului (tiling multi-scală + reproiecție inversă),
adaptată pentru obiecte mici (fețe, plăcuțe) în loc de obiectele mari pentru care
a fost gândită inițial.

Pentru o scriere completă, cu exemple vizuale before/after pentru fiecare etapă,
vezi `docs/documentatie_proiect.docx`. **Nu e inclus in git** (contine exemple
cu fete/placute reale, chiar daca majoritatea anonimizate) -- ramane doar local.

## Structură

```
detectie-fete-lp-360/
├── README.md
├── .gitignore
├── docs/
│   ├── documentatie_proiect.docx     # documentatia completa, cu exemple vizuale
│   ├── instruct_notes_originale.txt  # notele originale ale autorului + codul M8 initial
│   └── lista_fete_nullface.txt       # lista fisierelor cu fete procesate via NullFace
├── src/                     # module reutilizabile (nu se ruleaza direct)
│   ├── m8_preproc.py
│   ├── detectors.py
│   ├── lpr_roi.py
│   ├── face_skin_fill.py
│   └── nullface_cached.py
├── scripts/                 # utilitare rulabile direct
│   ├── censor_plates.py
│   ├── hybrid_anonymize.py
│   └── rename_jpgs.py
├── notebooks/                # studiile comparative, deja executate
│   ├── model_comparison.ipynb
│   ├── build_notebook.py
│   ├── lpr_vehicle_roi_study.ipynb
│   └── build_lpr_roi_notebook.py
├── weights/                  # checkpoint-urile modelelor (nu in git, vezi tabelul de mai jos)
├── data/
│   ├── 360testIMG/           # 9 cadre de test (nu in git -- neanonimizate, date sensibile)
│   └── personal_run/         # dataset personal, 892 poze (nu in git -- date sensibile)
├── outputs/                  # output-uri demo pe cele 9 cadre de test (nu in git)
├── .venv/                    # Python 3.14 -- torch+cuda, ultralytics, onnxruntime, opencv
└── nullface/                 # tot ce ruleaza pe Python 3.12 -- NullFace
    ├── repo/                 # clone github.com/hanweikung/nullface (read-only)
    └── conda_env/            # torch+diffusers+insightface+onnxruntime+opencv
```

**De ce doua medii separate:** NullFace (anonimizare de fete prin Stable Diffusion)
cere Python 3.12 + diffusers/insightface cu versiuni fixe, incompatibile cu restul
pipeline-ului (Python 3.14). `nullface/conda_env` e singurul interpretor care are
simultan tot ce trebuie (torch+diffusers+insightface **si** onnxruntime+opencv),
asa ca scriptul hibrid ruleaza cu el, desi fisierele lui (`scripts/hybrid_anonymize.py`,
`src/nullface_cached.py`) stau langa restul codului.

## Pipeline-ul de detecție (M8)

`src/m8_preproc.py` -- metoda originala a autorului, cu doua adaptari pentru
fete/placute (obiecte mult mai mici decat ce se detecta initial):

- **Calotele polare dezactivate** (`include_polar=False`, default) -- pe acest
  dataset (camera montata pe piept/ghidon), polul nord e cer gol si polul sud e
  echipamentul operatorului; nu contin niciodata fete/placute.
- **6 tile-uri ecuatoriale** in loc de 4 (`eq_tiles=6`) -- mai multa rezolutie
  pastrata dupa downscale la 640x640, necesara pentru obiecte mici.

Restul (seam handling la cusatura 0°/360°, NMS cilindric, reproiectie polara) e
neschimbat.

## Modele folosite

| Model | Fisier | Sursa |
|---|---|---|
| YOLOv8-face | `weights/yolov8_face.pt` | [arnabdhar/YOLOv8-Face-Detection](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection) |
| SCRFD (det_10g) | `weights/scrfd_10g.onnx` | insightface, mirror [immich-app/buffalo_l](https://huggingface.co/immich-app/buffalo_l) |
| YOLOv8-plate | `weights/yolov8_plate.pt` | [Koushim/yolov8-license-plate-detection](https://huggingface.co/Koushim/yolov8-license-plate-detection) |
| YOLOv8n COCO (vehicule) | `weights/yolov8n_coco.pt` | ultralytics (pretrained standard) |
| NullFace (SD1.5 + IP-Adapter-FaceID) | descarcat automat, cache HF | [hanweikung/nullface](https://github.com/hanweikung/nullface) |

`weights/` nu e in git (binare mari, usor de reobtinut din sursele de mai sus) --
regenereaza-l descarcand fiecare fisier la calea indicata inainte de a rula scripturile.

## Studii comparative (notebook-uri, doar pentru decizie -- nu productie)

- **`notebooks/model_comparison.ipynb`** -- SCRFD vs YOLOv8-face vs YOLOv8-plate, aceleasi
  crop-uri M8 partajate intre toate 3 modelele. Concluzie: SCRFD gaseste mai
  multe fete mici/distante (46 vs 39 pe 9 frame-uri), YOLOv8-face e mai rapid
  la batching. **SCRFD a fost ales** pentru fete in scriptele finale.
- **`notebooks/lpr_vehicle_roi_study.ipynb`** -- detectorul de placute singur avea multe
  fals-pozitive (bannere/panouri confundate cu placute). Compara filtrare prin
  intersectie cu vehicul detectat (Metoda 1) vs cascada -- crop pe bbox-ul
  vehiculului + placuta rulata acolo, la rezolutie mult mai mare (Metoda 2).
  **Metoda 2 a fost aleasa** -- Metoda 1 doar curata, nu recupereaza recall.

Ambele notebook-uri nu au adnotari ground-truth -- comparatiile sunt calitative
(vizual) + de viteza, nu precision/recall/mAP formal.

## Scripturi de productie

### `scripts/censor_plates.py` -- cenzura placute cu dreptunghi alb

Foloseste Metoda 2 (cascada vehicul -> placuta), validata in studiul de mai sus.

```bash
.venv/bin/python3.14 scripts/censor_plates.py <folder_input> <folder_output>
```
Opțiuni: `--device`, `--vehicle-conf`, `--plate-conf`, `--margin` (marja crop
vehicul, default 0.2), `--pad` (marja de siguranta pe dreptunghiul alb, default 0.1).

### `scripts/hybrid_anonymize.py` -- anonimizare hibrida de fete

Fete mari -> NullFace (schimbare de identitate prin diffusion). Fete mici ->
segmentare eliptica (landmark-uri SCRFD) + umplere cu culoarea de piele estimata.
Plasa de siguranta: daca NullFace nu gaseste fata in crop, cade automat pe
skin-fill -- nicio fata detectata nu ramane neprocesata.

```bash
# ruleaza cu interpretorul din nullface/, nu cu .venv-ul principal
nullface/conda_env/bin/python scripts/hybrid_anonymize.py <folder_input> <folder_output>
```
Opțiuni: `--area-threshold` (arie bbox px^2, default 20000 -- peste asta merge
la NullFace), `--device-num`.

**De retinut:**
- Primul apel descarca automat SD1.5 + IP-Adapter-FaceID de pe HuggingFace
  (~4-5GB, o singura data, cache in `~/.cache/huggingface/`).
- Fetele foarte apropiate/mari (NullFace) dureaza ~7-10s/fata (dupa incarcarea
  initiala a pipeline-ului, ~2-10s). Fetele mici (skin-fill) sunt aproape
  instant. Timpul total pentru un folder mare depinde de cate fete mari are.
- Nu are resume/checkpoint -- o intrerupere la mijloc inseamna reluare de la
  capat (nu e distructiv, doar rescrie fisierele, dar consuma timp in plus).
- NullFace are un artefact recurent (adauga o "sapca" care nu exista in
  original) -- comportament inerent al modelului, nu bug de integrare.
- Validat pe o rulare reala de 892 fotografii: 3604 fete, 0 erori. Detalii in
  `docs/documentatie_proiect.docx`.

### `src/face_skin_fill.py` -- testul izolat al segmentarii+skin-fill

Aplica segmentare+skin-fill la **toate** fetele detectate, indiferent de
marime -- folosit doar ca test de calitate inainte de integrarea in hibrid.
Nu apeleaza NullFace deloc.

```bash
.venv/bin/python3.14 src/face_skin_fill.py   # ruleaza pe data/360testIMG/, scrie in outputs/skin_fill_test/
```

### `scripts/rename_jpgs.py` -- utilitar generic (nu are legatura cu detectia)

Copiaza (nu muta) doar fisierele `.jpg`/`.JPG` dintr-un director cu tipuri
mixte de fisiere intr-un folder nou, redenumindu-le `img1.jpg`, `img2.jpg`, ...

```bash
python3 scripts/rename_jpgs.py <folder_input> <folder_output>
```

## Module interne (nu se ruleaza direct)

- `src/m8_preproc.py` -- preprocesarea M8 (crop-uri, reproiectie, NMS global).
- `src/detectors.py` -- adaptoare YOLOv8 (ultralytics) si SCRFD (decode manual onnxruntime).
- `src/lpr_roi.py` -- helpers pentru filtrare/cascada placute-vehicule.
- `src/nullface_cached.py` -- reimplementare a `anonymize_face()` din repo-ul NullFace,
  cu pipeline-ul SD incarcat o singura data (originalul il reincarca la fiecare
  apel, ~80s -- inacceptabil pentru batch).

## Praguri si parametri cheie (de ajustat empiric daca e nevoie)

| Parametru | Valoare | Unde |
|---|---|---|
| `eq_tiles` | 6 | `m8_preproc.generate_m8_crops` |
| `include_polar` | False | idem |
| Prag containment placuta-vehicul | 0.3 | `lpr_roi.filter_by_vehicle_containment` |
| Marja crop vehicul (placute) | 0.2 | `censor_plates.py`, `lpr_roi.crop_with_margin` |
| Prag arie NullFace vs skin-fill | 20000 px² | `hybrid_anonymize.py` |
| Marja crop fata (NullFace) | 0.6 | `hybrid_anonymize.CROP_MARGIN_FRAC` |
| Prag unghi/centroid landmark-uri nesigure | 35°/0.35×bbox | `face_skin_fill.build_face_mask` |

## Dataset de test

`data/360testIMG/` -- 9 frame-uri reale, 3840×1920, cameră 360° montată pe
piept/ghidon, filmare stradală în București, folosite pentru toate testele
descrise in acest README si in `docs/documentatie_proiect.docx`. **Nu e in
git** -- sunt frame-uri brute, neanonimizate, cu oameni si placute reale
identificabile, la fel ca `data/personal_run/` (892 fotografii proprii).
Ambele raman doar local, excluse explicit prin `.gitignore`. Singurele
imagini care ajung efectiv in git sunt cele cateva exemple deja anonimizate,
incorporate direct in `docs/documentatie_proiect.docx`.

## Licență

Acest repo e licențiat **[AGPL-3.0](LICENSE)** -- nu dintr-o preferință
generică, ci pentru că `src/nullface_cached.py` conține cod adaptat direct din
`anonymize_face.py` al proiectului [NullFace](https://github.com/hanweikung/nullface),
care e la rândul lui AGPL-3.0. O lucrare derivată din cod AGPL trebuie
distribuită tot sub AGPL-3.0 (sau o licență compatibilă) -- de asta primează
în fața oricărei alte alegeri pentru restul codului din acest repo.

### NOTICE -- dependințe terțe și restricțiile lor

Codul din acest repo (module proprii, scripturile, notebook-urile) e AGPL-3.0.
Modelele/checkpoint-urile pe care le folosește **nu sunt incluse în git** (vezi
`.gitignore`) și au propriile licențe, separate de codul acestui repo -- cine
le descarcă și le rulează trebuie să respecte termenii lor:

| Dependință | Licență / restricție | Observații |
|---|---|---|
| [NullFace](https://github.com/hanweikung/nullface) | AGPL-3.0 | sursa codului adaptat în `src/nullface_cached.py` |
| [ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8) | AGPL-3.0 | folosit ca dependință pip, nu vendorizat |
| [InsightFace](https://github.com/deepinsight/insightface) / `buffalo_l` | cod MIT, **dar modelele pretrained au istoric restricții de uz non-comercial** | verifică explicit înainte de orice uz comercial |
| Stable Diffusion v1.5 | [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license) | restricții de utilizare (nu genereaza anumite categorii de conținut), nu impune AGPL asupra codului tău |
| Checkpoint-uri YOLOv8-face / YOLOv8-plate ([arnabdhar](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection), [Koushim](https://huggingface.co/Koushim/yolov8-license-plate-detection)) | vezi paginile HuggingFace respective | nu verificate exhaustiv aici -- confirmă termenii înainte de uz comercial |

Dacă la un moment dat vrei o licență mai permisivă (MIT/Apache) pentru restul
codului, singura cale curată e să elimini/izolezi `nullface_cached.py` din
acest repo (de ex. ca submodul separat, licențiat propriu AGPL) -- cât timp
rămâne inclus direct aici, tot repo-ul moștenește AGPL-3.0.
