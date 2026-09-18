"""
m8_preproc.py — M8 Adaptive Zonal Multi-Scale: crop generation + reprojection
==============================================================================

Genereaza 7 crop-uri din un frame echirectangular 360° si implementeaza
reproiectia inversa a detectiilor inapoi in spatiul echirectangular original.

Structura M8:
  Crop 0   : full frame downscaledla 640×640 (prinde obiecte mari)
  Crops 1-4: banda ecuatoriala ±50° cu seam fix circular (15%) + 4 tile-uri
             cu 25% overlap (detectie precisa in zona de interes principala)
  Crop 5   : calota polara Nord prin proiectie stereografica (>45° lat)
  Crop 6   : calota polara Sud prin proiectie stereografica (<-45° lat)
  ─────────────────────────────────────────────────────────────────────────
  Total: 7 crop-uri / frame  →  un singur batch nvinfer(batch_size=7)
"""

import math
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

# ── Parametri M8 (identici cu studiul de preprocesare) ───────────────────────
NET_SIZE   = 640    # dimensiunea input YOLOv8
EQ_LAT     = 50     # banda ecuatoriala: ±50° latitudine
EQ_TILES   = 4      # numar tile-uri ecuatoriale
EQ_OVERLAP = 0.25   # overlap intre tile-uri adiacente (25%)
SEAM_PAD   = 0.15   # padding circular pentru cusatura 0°/360° (15% din latime)
POLAR_LAT  = 45     # prag latitudine pentru calotele polare (>45°)
NMS_IOU     = 0.35   # prag IoU pentru NMS global
CYL_NMS_IOU = 0.25   # prag IoU cilindric (mai permisiv, pentru detectii la cusatura 0°/360°)
SEAM_MARGIN = 0.08   # box-urile din primele/ultimele 8% din latime sunt candidate la dedup seam


@dataclass
class Detection:
    """O detectie in spatiul echirectangular."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float
    crop_id: int  = -1    # indexul crop-ului sursa (0-6); -1 = necunoscut
    seam:    bool = False # True daca obiectul a fost detectat la cusatura 0°/360°

    @property
    def area(self):
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


# ══════════════════════════════════════════════════════════════════════════════
# FUNCTII DE PROIECTIE SFERICA
# ══════════════════════════════════════════════════════════════════════════════

# Remap maps depend only on (H, W, north, out_sz, lat_limit_deg) — cache them.
_polar_map_cache: dict = {}
_polar_map_cache_gpu: dict = {}   # GpuMat versions, populated on first GPU use

_CUDA_REMAP   = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
_CUDA_RESIZE  = _CUDA_REMAP


def _cuda_resize(src, size: tuple) -> np.ndarray:
    """GPU resize via cv2.cuda (INTER_LINEAR). src can be ndarray or cuda_GpuMat."""
    if _CUDA_RESIZE:
        gpu = src if isinstance(src, cv2.cuda_GpuMat) else cv2.cuda_GpuMat(src)
        return cv2.cuda.resize(gpu, size, interpolation=cv2.INTER_LINEAR).download()
    arr = src.download() if isinstance(src, cv2.cuda_GpuMat) else src
    return cv2.resize(arr, size, interpolation=cv2.INTER_LINEAR)

def _equirect_to_polar_stereo(img_bgr: np.ndarray,
                               north: bool = True,
                               out_sz: int = NET_SIZE,
                               lat_limit_deg: float = POLAR_LAT,
                               _gpu_src=None):
    """
    Proiectie stereografica polara din echirectangular.
    Transforma calota polara (lat > lat_limit_deg pentru nord) intr-o
    imagine circulara rectilineara fara distorsiune.
    """
    H, W = img_bgr.shape[:2]
    cache_key = (H, W, north, out_sz, lat_limit_deg)

    if cache_key not in _polar_map_cache:
        lat_lim = math.radians(lat_limit_deg)
        half = out_sz // 2

        coords = (np.arange(out_sz) - half + 0.5) / half  # [-1, 1]
        xg, yg = np.meshgrid(coords, coords)
        r = np.sqrt(xg**2 + yg**2)
        valid = r <= 1.0

        c_arr = 2 * np.arctan(r * math.tan((math.pi / 2 - lat_lim) / 2))
        c_arr = np.where(valid, c_arr, 0)

        if north:
            lat_m = math.pi / 2 - c_arr
            lon_m = np.arctan2(xg, -yg)
        else:
            lat_m = -math.pi / 2 + c_arr
            lon_m = np.arctan2(xg, yg)

        map_u = ((lon_m + math.pi) % (2 * math.pi)) / (2 * math.pi) * W
        map_v = (math.pi / 2 - lat_m) / math.pi * H

        _polar_map_cache[cache_key] = (
            map_u.astype(np.float32),
            map_v.astype(np.float32),
            r > 1.0,   # circular mask
        )

    map_u, map_v, mask = _polar_map_cache[cache_key]

    if _CUDA_REMAP:
        if cache_key not in _polar_map_cache_gpu:
            _polar_map_cache_gpu[cache_key] = (
                cv2.cuda_GpuMat(map_u),
                cv2.cuda_GpuMat(map_v),
            )
        gpu_map_u, gpu_map_v = _polar_map_cache_gpu[cache_key]
        # Use caller-supplied GpuMat to avoid re-uploading the full frame
        gpu_src = _gpu_src if _gpu_src is not None else cv2.cuda_GpuMat(img_bgr)
        gpu_out = cv2.cuda.remap(gpu_src, gpu_map_u, gpu_map_v,
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        out = gpu_out.download()
    else:
        out = cv2.remap(img_bgr, map_u, map_v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    out[mask] = 0  # masca circulara (fast CPU boolean mask)
    return out


def _polar_px_to_equirect(px: float, py: float,
                           out_sz: int,
                           north: bool,
                           lat_limit_deg: float,
                           W_eq: int, H_eq: int
                           ) -> Optional[Tuple[float, float]]:
    """
    Pixel in proiectia polara stereografica → (u, v) in spatiul echirectangular.
    Returneaza None daca punctul e in afara cercului valid.
    """
    lat_lim = math.radians(lat_limit_deg)
    half = out_sz / 2
    xn = (px - half + 0.5) / half
    yn = (py - half + 0.5) / half
    r = math.sqrt(xn * xn + yn * yn)
    if r > 1.0:
        return None
    c = 2 * math.atan(r * math.tan((math.pi / 2 - lat_lim) / 2))
    if north:
        lat = math.pi / 2 - c
        lon = math.atan2(xn, -yn)
    else:
        lat = -math.pi / 2 + c
        lon = math.atan2(xn, yn)
    u = ((lon + math.pi) % (2 * math.pi)) / (2 * math.pi) * W_eq
    v = (math.pi / 2 - lat) / math.pi * H_eq
    return float(u), float(v)


def _reproject_bbox_polar(x1: float, y1: float, x2: float, y2: float,
                           north: bool, crop_sz: int,
                           lat_limit_deg: float,
                           W_eq: int, H_eq: int,
                           n: int = 12) -> Optional[Tuple[float, float, float, float]]:
    """
    Reproiecteaza un bbox din spatiul proiectiei polare in spatiul echirectangular.
    Esantioneaza n puncte pe fiecare latura a bbox-ului si calculeaza AABB-ul lor.
    """
    t = np.linspace(0, 1, n)
    pts = []
    for tt in t:
        corners = [
            (x1 + tt * (x2 - x1), y1),
            (x1 + tt * (x2 - x1), y2),
            (x1, y1 + tt * (y2 - y1)),
            (x2, y1 + tt * (y2 - y1)),
        ]
        for px, py in corners:
            r = _polar_px_to_equirect(px, py, crop_sz, north, lat_limit_deg, W_eq, H_eq)
            if r is not None:
                pts.append(r)

    if len(pts) < 4:
        return None
    us = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    return float(min(us)), float(min(vs)), float(max(us)), float(max(vs))


# ══════════════════════════════════════════════════════════════════════════════
# GENERARE CROP-URI M8
# ══════════════════════════════════════════════════════════════════════════════

def generate_m8_crops(frame_bgr: np.ndarray,
                       include_polar: bool = False,
                       eq_tiles: int = EQ_TILES) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Genereaza crop-urile M8 din frame-ul echirectangular.

    include_polar : daca False (default), omite calotele polare — pentru camere
                     montate pe piept/ghidon polul nord e cer gol si polul sud e
                     echipamentul propriu al operatorului, deci nu contin fete/placute.
    eq_tiles       : numar de tile-uri pe banda ecuatoriala (default marit de la 4 la
                     valoarea din EQ_TILES, pentru a pastra mai multa rezolutie pe
                     obiecte mici precum fete/placute).

    Returns
    -------
    crops : list of numpy arrays (BGR, 640×640) — 1 + eq_tiles (+2 daca include_polar)
    meta  : dict cu informatii necesare pentru reproiectia inversa
    """
    H, W = frame_bgr.shape[:2]
    crops: List[np.ndarray] = []
    meta: Dict[str, Any] = {'H': H, 'W': W, 'n_tiles': eq_tiles, 'include_polar': include_polar}

    # Upload frame to GPU once — reused by crop 0 and both polar caps.
    # Avoids 3 separate PCIe uploads of the same 5.5 MB frame.
    gpu_frame = cv2.cuda_GpuMat(frame_bgr) if _CUDA_RESIZE else None

    # ── Crop 0: Full frame downscaled ────────────────────────────────────────
    full_small = _cuda_resize(gpu_frame if gpu_frame is not None else frame_bgr,
                              (NET_SIZE, NET_SIZE))
    crops.append(full_small)
    meta['full_scale_x'] = W / NET_SIZE  # factor pentru a rescala inapoi
    meta['full_scale_y'] = H / NET_SIZE

    # ── Crops 1-4: Banda ecuatoriala cu seam fix + tile-uri cu overlap ────────
    y_top = int((90 - EQ_LAT) / 180 * H)
    y_bot = int((90 + EQ_LAT) / 180 * H)
    eq_crop = frame_bgr[y_top:y_bot, :]

    pad = int(W * SEAM_PAD)
    # Padding circular: adaugam sfarsitul imaginii la stanga si inceputul la dreapta
    eq_padded = np.hstack([eq_crop[:, W - pad:], eq_crop, eq_crop[:, :pad]])
    Wp = eq_padded.shape[1]
    Hp = eq_padded.shape[0]  # inaltime banda ecuatoriala

    tile_meta = []
    tile_w = Wp / eq_tiles
    for ti in range(eq_tiles):
        tx1 = max(0, int(ti * tile_w - tile_w * EQ_OVERLAP))
        tx2 = min(Wp, int((ti + 1) * tile_w + tile_w * EQ_OVERLAP))
        tile = eq_padded[:, tx1:tx2]
        tile_resized = _cuda_resize(tile, (NET_SIZE, NET_SIZE))
        crops.append(tile_resized)
        tile_meta.append({
            'tx1': tx1,
            'tile_w_orig': tx2 - tx1,  # latimea tile-ului in spatiul padded
            'tile_h_orig': Hp,          # inaltimea benzii ecuatoriale
            'pad': pad,
            'y_top': y_top,
        })
    meta['tile_meta'] = tile_meta

    if include_polar:
        # ── Crop N+1: Calota polara Nord (proiectie stereografica) ────────────
        polar_n = _equirect_to_polar_stereo(frame_bgr, north=True,
                                             out_sz=NET_SIZE, lat_limit_deg=POLAR_LAT,
                                             _gpu_src=gpu_frame)
        crops.append(polar_n)

        # ── Crop N+2: Calota polara Sud (proiectie stereografica) ─────────────
        polar_s = _equirect_to_polar_stereo(frame_bgr, north=False,
                                             out_sz=NET_SIZE, lat_limit_deg=POLAR_LAT,
                                             _gpu_src=gpu_frame)
        crops.append(polar_s)

    expected = 1 + eq_tiles + (2 if include_polar else 0)
    assert len(crops) == expected, f"M8 trebuie sa produca {expected} crop-uri, nu {len(crops)}"
    return crops, meta


# ══════════════════════════════════════════════════════════════════════════════
# REPROIECTIE INVERSA
# ══════════════════════════════════════════════════════════════════════════════

def reproject_to_equirect(bboxes_640: List[Tuple],
                           crop_index: int,
                           meta: Dict[str, Any]) -> List[Detection]:
    """
    Reproiecteaza detectii din spatiul crop-ului (640×640) in spatiul
    echirectangular original.

    Parameters
    ----------
    bboxes_640 : list of (x1, y1, x2, y2, class_id, confidence)
                 coordonate in spatiul 640×640 al crop-ului
    crop_index : 0–6
    meta       : dict returnat de generate_m8_crops

    Returns
    -------
    list of Detection in spatiul echirectangular
    """
    W = meta['W']
    H = meta['H']
    n_tiles = meta['n_tiles']
    polar_north_id = n_tiles + 1
    polar_south_id = n_tiles + 2
    result: List[Detection] = []

    for x1, y1, x2, y2, cls_id, conf in bboxes_640:

        if crop_index == 0:
            # ── Crop 0: Full frame – scala simpla ────────────────────────────
            sx = meta['full_scale_x']
            sy = meta['full_scale_y']
            result.append(Detection(
                x1=max(0.0, x1 * sx), y1=max(0.0, y1 * sy),
                x2=min(W - 1.0, x2 * sx), y2=min(H - 1.0, y2 * sy),
                class_id=cls_id, confidence=conf, crop_id=crop_index
            ))

        elif 1 <= crop_index <= n_tiles:
            # ── Crops 1..n_tiles: Banda ecuatoriala ───────────────────────────
            ti = crop_index - 1
            tm = meta['tile_meta'][ti]
            tx1_orig = tm['tx1']
            tw_orig  = tm['tile_w_orig']
            th_orig  = tm['tile_h_orig']
            pad      = tm['pad']
            y_top    = tm['y_top']

            # Scala din spatiul 640 in spatiul tile-ului padded
            px1 = x1 / NET_SIZE * tw_orig + tx1_orig
            px2 = x2 / NET_SIZE * tw_orig + tx1_orig
            py1 = y1 / NET_SIZE * th_orig
            py2 = y2 / NET_SIZE * th_orig

            cx_p = (px1 + px2) / 2

            # Elimina offset-ul seam padding
            if pad <= cx_p <= pad + W:
                ox1, ox2 = px1 - pad, px2 - pad
            elif cx_p < pad:
                ox1, ox2 = px1 - pad + W, px2 - pad + W
            else:
                ox1, ox2 = px1 - pad - W, px2 - pad - W

            ey1 = y_top + py1
            ey2 = y_top + py2

            # Nu clipam x la [0,W-1]: lasam ox1<0 sau ox2>W pentru detectii la cusatura.
            # NMS-ul cilindric din global_nms va gestiona suprimarea duplicatelor.
            result.append(Detection(
                x1=float(ox1),
                y1=float(max(0, min(H - 1, ey1))),
                x2=float(ox2),
                y2=float(max(0, min(H - 1, ey2))),
                class_id=cls_id, confidence=conf, crop_id=crop_index
            ))

        elif crop_index in (polar_north_id, polar_south_id):
            # ── Crops polare: Proiectie polara stereografica ──────────────────
            north = (crop_index == polar_north_id)
            bbox_eq = _reproject_bbox_polar(
                x1, y1, x2, y2,
                north=north,
                crop_sz=NET_SIZE,
                lat_limit_deg=POLAR_LAT,
                W_eq=W, H_eq=H
            )
            if bbox_eq is not None:
                ex1, ey1, ex2, ey2 = bbox_eq
                if ex2 - ex1 > W * 0.40:
                    continue  # artefact reprojection polar: bbox prea lat
                result.append(Detection(
                    x1=max(0.0, ex1), y1=max(0.0, ey1),
                    x2=min(W - 1.0, ex2), y2=min(H - 1.0, ey2),
                    class_id=cls_id, confidence=conf, crop_id=crop_index
                ))

    return result


# ══════════════════════════════════════════════════════════════════════════════
# NMS GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

def _merge_into(keeper: Detection, other: Detection, W: float = 0.0) -> None:
    """Extinde bbox-ul keeper sa acopere si zona lui other (union bbox).

    In spatiul cilindric (W > 0), cauta cel mai bun offset {0, +W, -W} pentru
    a alinia other cu keeper inainte de a calcula union-ul — necesar pentru
    detectii care se suprapun in jurul cusaturii 0°/360°.
    """
    if W > 0:
        best_overlap = -1.0
        best_off = 0.0
        for off in (0.0, W, -W):
            ov = max(0.0, min(keeper.x2, other.x2 + off) - max(keeper.x1, other.x1 + off))
            if ov > best_overlap:
                best_overlap = ov
                best_off = off
        ox1, ox2 = other.x1 + best_off, other.x2 + best_off
    else:
        ox1, ox2 = other.x1, other.x2
    keeper.x1 = min(keeper.x1, ox1)
    keeper.y1 = min(keeper.y1, other.y1)
    keeper.x2 = max(keeper.x2, ox2)
    keeper.y2 = max(keeper.y2, other.y2)


def _iou(a: Detection, b: Detection) -> float:
    ix1 = max(a.x1, b.x1); iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2); iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    return inter / (a.area + b.area - inter + 1e-6)


def _iou_cyl(a: Detection, b: Detection, W: float) -> float:
    """IoU cu wrap-around cilindric pe axa X.
    Incearca a shiftat cu {0, +W, -W} fata de b pentru a detecta overlap-ul
    in jurul cusaturii 0°/360°. Foloseste si fractia din aria minima ca fallback
    pentru cazul in care o detectie e partial clipped."""
    best_iou = 0.0
    for ax_off in (0.0, W, -W):
        ax1, ax2 = a.x1 + ax_off, a.x2 + ax_off
        ix1 = max(ax1, b.x1); ix2 = min(ax2, b.x2)
        iy1 = max(a.y1, b.y1); iy2 = min(a.y2, b.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0.0:
            continue
        union = a.area + b.area - inter + 1e-6
        # IoU standard + fractia din aria minima (pentru detectii partial clipped la seam)
        iou_std = inter / union
        iou_min = inter / (min(a.area, b.area) + 1e-6)
        best_iou = max(best_iou, iou_std, iou_min * 0.5)
    return best_iou


def _normalize_seam_det(d: Detection, W: int) -> Detection:
    """Aduce coordonatele x ale unei detectii la cusatura in [0, W-1]."""
    if d.x1 < 0 or d.x2 >= W:
        # Box straddles seam: plasam centrul in jumatatea cu latime mai mare
        if d.x1 < 0 and d.x2 >= W:
            # Acopera toata latimea - clip normal
            d.x1 = 0.0; d.x2 = float(W - 1)
        elif d.x1 < 0:
            # Centrul e la dreapta seamului (ox1<0 inseamna ca vine din dreapta)
            right_w = -d.x1        # latime pe dreapta (x de la W+ox1 la W)
            left_w  = d.x2         # latime pe stanga (x de la 0 la ox2)
            if right_w >= left_w:
                d.x1 = float(W + d.x1)   # = W - abs(ox1)
                d.x2 = float(W - 1)
            else:
                d.x1 = 0.0
                # d.x2 ramane
        else:  # d.x2 >= W
            d.x2 = float(W - 1)
    return d


def _seam_dedup(detections: List[Detection], W: int) -> List[Detection]:
    """A doua trecere post-NMS: elimina perechile de box-uri care reprezinta
    aceeasi detectie taiata in doua de cusatura 0°/360°.

    Conditii pentru suprimare:
      - Aceeasi clasa
      - Un box langa marginea stanga (x1 ≤ 8%W) si altul langa cea dreapta (x2 ≥ 92%W)
      - Range-ul Y se suprapune semnificativ (IoU pe axa Y ≥ 40%)
    Pastreaza box-ul cu confidence mai mare.
    """
    margin = W * SEAM_MARGIN
    remove: set = set()
    for i, a in enumerate(detections):
        if i in remove:
            continue
        for j, b in enumerate(detections):
            if j <= i or j in remove or a.class_id != b.class_id:
                continue
            a_near_left  = a.x1 <= margin  and a.x2 <= W * 0.35
            a_near_right = a.x2 >= W - margin and a.x1 >= W * 0.65
            b_near_left  = b.x1 <= margin  and b.x2 <= W * 0.35
            b_near_right = b.x2 >= W - margin and b.x1 >= W * 0.65

            if not ((a_near_left and b_near_right) or (a_near_right and b_near_left)):
                continue

            # Overlap Y: inter / min_height >= 0.50
            # (cel mai scurt bbox e cel putin 50% continut in Y-ul celuilalt)
            yi1 = max(a.y1, b.y1); yi2 = min(a.y2, b.y2)
            if yi2 <= yi1:
                continue
            y_inter = yi2 - yi1
            y_min_h = min(a.y2 - a.y1, b.y2 - b.y1)
            if y_min_h <= 0 or y_inter / y_min_h < 0.50:
                continue

            # Acelasi obiect la cusatura: keeper-ul (confidence mai mare) este extins
            # sa acopere si Y-ul celeilalte jumatati si se ancoreaza la marginea seam-ului.
            if a.confidence >= b.confidence:
                a.y1 = min(a.y1, b.y1)
                a.y2 = max(a.y2, b.y2)
                if a_near_right:
                    a.x2 = float(W - 1)
                else:
                    a.x1 = 0.0
                a.seam = True
                remove.add(j)
            else:
                b.y1 = min(b.y1, a.y1)
                b.y2 = max(b.y2, a.y2)
                if b_near_right:
                    b.x2 = float(W - 1)
                else:
                    b.x1 = 0.0
                b.seam = True
                remove.add(i)
                break

    return [d for i, d in enumerate(detections) if i not in remove]


def _polar_equator_merge(detections: List[Detection], W: int = 0, n_tiles: int = EQ_TILES) -> List[Detection]:
    """Mergeaza detectii polare cu detectii ecuatoriale (crops 1..n_tiles) care
    sunt vertical continue (acelasi obiect care se extinde de la ecuator spre pol).

    Conditii pentru merge:
      - Aceeasi clasa
      - O detectie polara, cealalta din banda ecuatoriala
      - inter / aria_ecuatoriala > 0.08 SAU inter / aria_polara > 0.08
    Suporta si cazul in care cei doi sunt pe laturi opuse ale cusaturii 0°/360° (W > 0):
    incearca offset-urile {0, +W, -W} pe ecuatorial pentru a gasi suprapunerea cilindrica.
    Pastreaza confidence mai mare; bbox-ul devine union-ul celor doua.

    Fara efect daca nu exista detectii polare (include_polar=False) — polar_ids
    nu se va potrivi niciodata cu vreun crop_id.
    """
    polar_ids = (n_tiles + 1, n_tiles + 2)
    equat_ids = tuple(range(1, n_tiles + 1))
    remove: set = set()
    offsets = (0.0, float(W), -float(W)) if W > 0 else (0.0,)

    for i, a in enumerate(detections):
        if i in remove:
            continue
        for j, b in enumerate(detections):
            if j <= i or j in remove or a.class_id != b.class_id:
                continue

            # Identifica care e polar si care e ecuatorial
            if a.crop_id in polar_ids and b.crop_id in equat_ids:
                polar_d, equat_d, pi, ei = a, b, i, j
            elif b.crop_id in polar_ids and a.crop_id in equat_ids:
                polar_d, equat_d, pi, ei = b, a, j, i
            else:
                continue

            equat_area = equat_d.area
            polar_area = polar_d.area
            if equat_area <= 0 and polar_area <= 0:
                continue

            # Cauta cel mai bun offset cilindric pentru ecuatorial
            best_ratio = 0.0
            best_off   = 0.0
            for off in offsets:
                ix1 = max(polar_d.x1, equat_d.x1 + off)
                iy1 = max(polar_d.y1, equat_d.y1)
                ix2 = min(polar_d.x2, equat_d.x2 + off)
                iy2 = min(polar_d.y2, equat_d.y2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                if inter == 0.0:
                    continue
                ratio = max(
                    inter / equat_area if equat_area > 0 else 0.0,
                    inter / polar_area if polar_area > 0 else 0.0,
                )
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_off   = off

            if best_ratio < 0.15:
                continue

            # Merge: keeper primeste union bbox si confidence mai mare
            if equat_d.confidence >= polar_d.confidence:
                keeper, gone = equat_d, pi
            else:
                keeper, gone = polar_d, ei
            keeper.x1 = min(polar_d.x1, equat_d.x1 + best_off)
            keeper.y1 = min(polar_d.y1, equat_d.y1)
            keeper.x2 = max(polar_d.x2, equat_d.x2 + best_off)
            keeper.y2 = max(polar_d.y2, equat_d.y2)
            keeper.confidence = max(polar_d.confidence, equat_d.confidence)
            if best_off != 0.0:
                keeper.seam = True
            remove.add(gone)

    return [d for i, d in enumerate(detections) if i not in remove]


def global_nms(detections: List[Detection], iou_thr: float = NMS_IOU, W: int = 0,
               n_tiles: int = EQ_TILES) -> List[Detection]:
    """
    NMS global pe detectiile din toate crop-urile, dupa reproiectie in
    spatiul echirectangular comun.

    Daca W > 0, foloseste IoU cilindric (CYL_NMS_IOU) pentru a suprima
    corect detectiile duble la cusatura 0°/360°.
    """
    if not detections:
        return []
    equat_ids = tuple(range(1, n_tiles + 1))
    detections = sorted(detections, key=lambda d: (d.confidence, d.area), reverse=True)
    keep, suppressed = [], set()
    cyl_thr = CYL_NMS_IOU if W > 0 else iou_thr
    for i, d in enumerate(detections):
        if i in suppressed:
            continue
        keep.append(d)
        for j, d2 in enumerate(detections[i + 1:], i + 1):
            if j in suppressed or d2.class_id != d.class_id:
                continue
            score = _iou_cyl(d, d2, float(W)) if W > 0 else _iou(d, d2)
            thr   = cyl_thr if W > 0 else iou_thr
            if score > thr:
                suppressed.add(j)
                if (d.crop_id in equat_ids and d2.crop_id in equat_ids) or \
                   (d.crop_id == 0 and d2.crop_id in equat_ids):
                    _merge_into(d, d2, float(W))
                # Union-merge doar intre tile-urile ecuatoriale care se suprapun 25%.
                # Crop 0 (full-frame, rezolutie mai slaba) si polari sunt exclusi.
    # Merge polar ↔ ecuatorial vertical-continuu (acelasi obiect in doua zone)
    keep = _polar_equator_merge(keep, W=W, n_tiles=n_tiles)
    # Trecere suplimentara: elimina perechile de jumatati la cusatura ce au scapat IoU-NMS
    if W > 0:
        keep = _seam_dedup(keep, W)
        keep = [_normalize_seam_det(d, W) for d in keep]
    return keep
