"""detectors.py — adaptoare pentru cele 3 modele, interfata comuna:
   given list[np.ndarray BGR 640x640] -> list[list[(x1,y1,x2,y2,class_id,confidence)]]
   (o lista de detectii per crop, in spatiul 640x640, gata pentru reproject_to_equirect)
"""
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


class YoloAdapter:
    """Wrapper generic pentru YOLOv8 (face sau plate) — decode/NMS interne ultralytics."""

    def __init__(self, weight_path: str, device: str, conf: float = 0.25, imgsz: int = 640):
        self.model = YOLO(weight_path)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def __call__(self, crops: list[np.ndarray]) -> list[list[tuple]]:
        results = self.model.predict(crops, imgsz=self.imgsz, conf=self.conf,
                                      device=self.device, verbose=False)
        out = []
        for r in results:
            dets = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                conf = r.boxes.conf.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy()
                for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf):
                    dets.append((float(x1), float(y1), float(x2), float(y2), int(c), float(cf)))
            out.append(dets)
        return out


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


class ScrfdAdapter:
    """SCRFD (det_10g, insightface) — decode multi-stride manual via onnxruntime.

    Arhitectura fixa a checkpoint-ului: strides (8, 16, 32), 2 anchors/locatie,
    input 640x640 (batch=1, deci ruleaza cate un crop pe rand — 7 crop-uri e ieftin).
    Preprocesare identica cu insightface/model_zoo/scrfd.py: BGR->RGB, (x-127.5)/128.
    """
    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2

    def __init__(self, onnx_path: str, providers: list, conf: float = 0.4, nms_iou: float = 0.4,
                 net_size: int = 640):
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.conf = conf
        self.nms_iou = nms_iou
        self.net_size = net_size
        self._center_cache = {}
        self._sigmoid_needed = None  # determinat empiric la prima rulare

    def _anchor_centers(self, stride: int) -> np.ndarray:
        key = stride
        if key in self._center_cache:
            return self._center_cache[key]
        h = w = self.net_size // stride
        grid = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
        centers = (grid * stride).reshape(-1, 2)
        centers = np.stack([centers] * self.NUM_ANCHORS, axis=1).reshape(-1, 2)
        self._center_cache[key] = centers
        return centers

    def _decode_one(self, crop_bgr: np.ndarray) -> list[tuple]:
        blob = crop_bgr.astype(np.float32)[:, :, ::-1]
        blob = (blob - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[None].astype(np.float32)
        outs = self.sess.run(None, {self.input_name: blob})
        scores_all = outs[0:3]
        bboxes_all = outs[3:6]

        if self._sigmoid_needed is None:
            raw_max = max(float(s.max()) for s in scores_all)
            raw_min = min(float(s.min()) for s in scores_all)
            self._sigmoid_needed = raw_max > 1.0 or raw_min < 0.0

        all_boxes, all_scores = [], []
        for stride, scores, bbox_preds in zip(self.STRIDES, scores_all, bboxes_all):
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
            all_boxes.append(boxes)
            all_scores.append(scores[keep_idx])

        if not all_boxes:
            return []
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        keep = _nms_xyxy(boxes, scores, self.nms_iou)
        return [(float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]), float(boxes[i, 3]),
                  0, float(scores[i])) for i in keep]

    def __call__(self, crops: list[np.ndarray]) -> list[list[tuple]]:
        return [self._decode_one(c) for c in crops]
