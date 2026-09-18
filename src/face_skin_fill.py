"""face_skin_fill.py -- test standalone: pentru fete prea mici ca sa fie procesate
cu NullFace, segmenteaza fata printr-o elipsa orientata (din landmark-urile SCRFD:
ochi + nas + colturi gura) si umple regiunea cu culoarea de piele estimata direct
din fata respectiva (median RGB in interiorul elipsei).

Nu e inca integrat in hibridul final (NullFace pentru fete mari + asta pentru fete
mici) -- e doar testul cerut, rulat pe cele 9 imagini din 360testIMG, cu rezultatul
intr-un folder nou pentru inspectie vizuala.

Foloseste pipeline-ul M8 al autorului (generate_m8_crops / reproject_to_equirect /
global_nms) pentru detectie -- fara el, fetele mici sunt irecunoscibile la
rezolutia unui panoramic 3840x1920 intreg.
"""
import glob
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m8_preproc as m8

EQ_TILES = 6
WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "scrfd_10g.onnx"


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


class ScrfdLandmarkAdapter:
    """Ca detectors.ScrfdAdapter, dar decodeaza si cele 5 landmark-uri (kps) --
    branch-ul outs[6:9], ignorat de adaptorul original (nu era nevoie de el pentru
    comparatia SCRFD vs YOLOv8-face din studiul precedent)."""
    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2

    def __init__(self, onnx_path: str, providers: list, conf: float = 0.4,
                 nms_iou: float = 0.4, net_size: int = 640):
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.conf = conf
        self.nms_iou = nms_iou
        self.net_size = net_size
        self._center_cache = {}
        self._sigmoid_needed = None

    def _anchor_centers(self, stride: int) -> np.ndarray:
        if stride in self._center_cache:
            return self._center_cache[stride]
        h = w = self.net_size // stride
        grid = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
        centers = (grid * stride).reshape(-1, 2)
        centers = np.stack([centers] * self.NUM_ANCHORS, axis=1).reshape(-1, 2)
        self._center_cache[stride] = centers
        return centers

    def _decode_one(self, crop_bgr: np.ndarray):
        blob = crop_bgr.astype(np.float32)[:, :, ::-1]
        blob = (blob - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[None].astype(np.float32)
        outs = self.sess.run(None, {self.input_name: blob})
        scores_all, bboxes_all, kps_all = outs[0:3], outs[3:6], outs[6:9]

        if self._sigmoid_needed is None:
            raw_max = max(float(s.max()) for s in scores_all)
            raw_min = min(float(s.min()) for s in scores_all)
            self._sigmoid_needed = raw_max > 1.0 or raw_min < 0.0

        all_boxes, all_scores, all_kps = [], [], []
        for stride, scores, bbox_preds, kps_preds in zip(self.STRIDES, scores_all, bboxes_all, kps_all):
            scores = scores.reshape(-1)
            if self._sigmoid_needed:
                scores = 1.0 / (1.0 + np.exp(-scores))
            keep_idx = np.where(scores >= self.conf)[0]
            if keep_idx.size == 0:
                continue
            centers = self._anchor_centers(stride)

            deltas = bbox_preds.reshape(-1, 4) * stride
            x1 = centers[:, 0] - deltas[:, 0]
            y1 = centers[:, 1] - deltas[:, 1]
            x2 = centers[:, 0] + deltas[:, 2]
            y2 = centers[:, 1] + deltas[:, 3]
            boxes = np.stack([x1, y1, x2, y2], axis=-1)[keep_idx]

            # SCRFD kps decode: 5 puncte (x,y) intercalate, offset simetric fata de centru
            # (nu ltrb ca la bbox) -- vezi insightface/model_zoo/scrfd.py::distance2kps.
            kd = kps_preds.reshape(-1, 10) * stride
            kx = centers[:, 0:1] + kd[:, 0::2]
            ky = centers[:, 1:2] + kd[:, 1::2]
            kps = np.stack([kx, ky], axis=-1)[keep_idx]  # (n, 5, 2)

            all_boxes.append(boxes)
            all_scores.append(scores[keep_idx])
            all_kps.append(kps)

        if not all_boxes:
            return [], np.zeros((0, 5, 2), dtype=np.float32)
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        kps = np.concatenate(all_kps, axis=0)
        keep = _nms_xyxy(boxes, scores, self.nms_iou)
        dets = [(float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]), float(boxes[i, 3]),
                 0, float(scores[i])) for i in keep]
        return dets, kps[keep]

    def __call__(self, crops: list[np.ndarray]):
        out_dets, out_kps = [], []
        for c in crops:
            d, k = self._decode_one(c)
            out_dets.append(d)
            out_kps.append(k)
        return out_dets, out_kps


def _reproject_point(x: float, y: float, crop_index: int, meta: dict):
    """Reproiecteaza un singur punct reutilizand exact matematica din
    m8.reproject_to_equirect -- tratam punctul ca un bbox degenerat (x,y,x,y)."""
    d = m8.reproject_to_equirect([(x, y, x, y, -1, 1.0)], crop_index, meta)
    return (d[0].x1, d[0].y1) if d else None


def detect_faces_with_landmarks(img: np.ndarray, model: ScrfdLandmarkAdapter):
    crops, meta = m8.generate_m8_crops(img, include_polar=False, eq_tiles=EQ_TILES)
    raw_dets, raw_kps = model(crops)

    all_dets = []
    landmarks_by_id = {}
    for crop_idx, (dets_640, kps_640) in enumerate(zip(raw_dets, raw_kps)):
        if not dets_640:
            continue
        # Fara poli (include_polar=False), reproject_to_equirect pastreaza ordinea
        # 1:1 intre input si output pentru crop_index 0..eq_tiles -- zip-ul e sigur.
        reproj = m8.reproject_to_equirect(dets_640, crop_idx, meta)
        for det, kp5 in zip(reproj, kps_640):
            pts = []
            for (lx, ly) in kp5:
                p = _reproject_point(float(lx), float(ly), crop_idx, meta)
                if p is not None:
                    pts.append(p)
            landmarks_by_id[id(det)] = np.array(pts, dtype=np.float32) if len(pts) == 5 else None
            all_dets.append(det)

    # global_nms muta/filtreaza in-place aceleasi obiecte Detection (nu creeaza altele
    # noi la merge/seam-dedup) -- lookup-ul prin id() de mai jos e sigur dupa NMS.
    final = m8.global_nms(all_dets, W=img.shape[1], n_tiles=meta['n_tiles'])
    return [(det, landmarks_by_id.get(id(det))) for det in final]


def build_face_mask(shape_hw: tuple, det: m8.Detection, landmarks) -> np.ndarray:
    """Elipsa orientata (unghi din linia ochi-ochi) care aproximeaza forma fetei.
    Fallback pe elipsa axis-aligned centrata pe bbox daca landmark-urile lipsesc
    (rar: pot lipsi doar cand un punct reproiectat cade complet in afara benzii
    valide -- practic niciodata pentru fete, dar tratat explicit, nu presupus)."""
    H, W = shape_hw
    cx_bbox = (det.x1 + det.x2) / 2
    cy_bbox = (det.y1 + det.y2) / 2
    w_bbox = det.x2 - det.x1
    h_bbox = det.y2 - det.y1

    angle = 0.0
    cx, cy = cx_bbox, cy_bbox
    if landmarks is not None and len(landmarks) == 5:
        left_eye, right_eye = landmarks[0], landmarks[1]
        dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
        cand_angle = 0.0
        if dx * dx + dy * dy > 1.0:
            cand_angle = math.degrees(math.atan2(dy, dx))
            # normalizeaza la (-90, 90] -- o fata inclinata nu trebuie sa dea vreodata
            # un unghi de ~180 doar pentru ca "ochiul stang/drept" e ambiguu pe profil
            if cand_angle > 90:
                cand_angle -= 180
            elif cand_angle <= -90:
                cand_angle += 180
        lcx, lcy = landmarks[:, 0].mean(), landmarks[:, 1].mean()
        centroid_dist = math.hypot(lcx - cx_bbox, lcy - cy_bbox)

        # Pe fete din profil, un ochi e ocultat/extrapolat gresit de SCRFD si
        # landmark-urile devin nesigure -- centroidul se deplaseaza mult fata de
        # bbox si/sau unghiul devine extrem, "tragand" elipsa in afara fetei
        # (observat empiric pe acest dataset). In acel caz cadem pe elipsa
        # axis-aligned centrata pe bbox, in loc sa incercam sa o rotim gresit.
        landmarks_reliable = (abs(cand_angle) <= 35.0
                               and centroid_dist <= 0.35 * max(w_bbox, h_bbox))
        if landmarks_reliable:
            angle = cand_angle
            # Centroidul celor 5 puncte e tras spre partea de sus a fetei (ochi+nas+gura,
            # fara frunte/barbie) -- combinat cu centrul bbox-ului ca elipsa sa acopere
            # frunte si barbie in mod echilibrat.
            cx = 0.5 * cx_bbox + 0.5 * lcx
            cy = 0.45 * cy_bbox + 0.55 * lcy

    axis_w = max(2.0, w_bbox * 0.42)
    axis_h = max(2.0, h_bbox * 0.48)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))),
                (int(round(axis_w)), int(round(axis_h))),
                angle, 0, 360, 255, thickness=-1)
    return mask


def estimate_skin_color(img_bgr: np.ndarray, mask: np.ndarray) -> tuple:
    """Median BGR peste toti pixelii din masca -- simplu si robust la outlieri,
    dar include ochi/sprancene/gura (mai intunecate), deci tinde usor spre un ton
    puțin mai inchis decat pielea 'curata'; suficient pentru acest test."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (128, 128, 128)
    pixels = img_bgr[ys, xs].astype(np.float32)
    return tuple(float(v) for v in np.median(pixels, axis=0))


def fill_face(img_bgr: np.ndarray, mask: np.ndarray, color: tuple) -> None:
    img_bgr[mask > 0] = color


def main():
    in_dir = Path(__file__).resolve().parent.parent / "data" / "360testIMG"
    out_dir = Path(__file__).resolve().parent.parent / "outputs" / "skin_fill_test"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = ScrfdLandmarkAdapter(str(WEIGHTS),
                                  providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                  conf=0.4)

    frames = sorted(glob.glob(str(in_dir / "*.jpg")))
    total_faces = 0
    for fp in frames:
        img = cv2.imread(fp)
        results = detect_faces_with_landmarks(img, model)
        out = img.copy()
        for det, landmarks in results:
            mask = build_face_mask(out.shape[:2], det, landmarks)
            color = estimate_skin_color(img, mask)
            fill_face(out, mask, color)
        total_faces += len(results)
        cv2.imwrite(str(out_dir / Path(fp).name), out)
        print(f"  {Path(fp).name}: {len(results)} fete umplute cu culoare de piele")

    print(f"\n{len(frames)} imagini procesate, {total_faces} fete in total. Rezultat in {out_dir}/")


if __name__ == "__main__":
    main()
