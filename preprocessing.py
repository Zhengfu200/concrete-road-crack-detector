"""One image geometry and prediction path for evaluation and video inference."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

STRIDE = 32
BORDER = 16
COLOR = (114, 114, 114)
MAX_DET = 300
EVALUATION_CONF = .001


def preprocessing_spec(imgsz, nms_iou=None, tta=None):
    if not isinstance(imgsz, int) or isinstance(imgsz, bool) or imgsz <= 0 or imgsz % STRIDE:
        raise ValueError("imgsz 必须是正的32倍数（内容区域尺寸，例如960）")
    if nms_iou is not None and (isinstance(nms_iou, bool) or not 0 < nms_iou < 1):
        raise ValueError("重复框抑制IoU必须在0和1之间")
    spec = {"version": "letterbox-border-v1", "content_size": imgsz,
            "input_size": imgsz + 2 * BORDER, "border": BORDER, "stride": STRIDE,
            "color": list(COLOR), "resize": "opencv-linear-ceil", "preserve_aspect_ratio": True,
            "center": True, "end2end": True, "max_det": MAX_DET}
    if nms_iou is not None:
        spec.update(version="letterbox-border-nms-v1", postprocess={"type": "greedy_class_nms", "iou": nms_iou})
    if tta is not None:
        if tta != 'flip4' or nms_iou is None:
            raise ValueError('flip4多视图必须显式设置重复框抑制IoU')
        spec.update(version='letterbox-border-flip4-nms-v1', test_time_augmentation={
            'views':['identity','horizontal','vertical','both'],
            'merge':'per_view_nms_then_combined_class_nms', 'max_candidates':4*MAX_DET,
            'prediction_min_conf':EVALUATION_CONF})
    return spec


@dataclass(frozen=True)
class FrameGeometry:
    height: int
    width: int
    resized_height: int
    resized_width: int
    top: int
    left: int

    def restore(self, boxes):
        """Map canvas xyxy to original pixels, clipping boxes to the original image."""
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - self.left) * self.width / self.resized_width
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - self.top) * self.height / self.resized_height
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, self.width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, self.height)
        return boxes


def prepare_frame(frame, imgsz=960):
    """Resize the long side, center-letterbox, then add a fixed 16px gray border."""
    spec = preprocessing_spec(imgsz)
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
            or frame.shape[2] != 3 or not frame.shape[0] or not frame.shape[1]):
        raise ValueError("输入必须是非空的 uint8 BGR 三通道图片")
    h, w = frame.shape[:2]
    ratio = imgsz / max(h, w)
    rh, rw = min(math.ceil(h * ratio), imgsz), min(math.ceil(w * ratio), imgsz)
    resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR) if (rh, rw) != (h, w) else frame
    top, left = BORDER + (imgsz - rh) // 2, BORDER + (imgsz - rw) // 2
    size = spec["input_size"]
    canvas = cv2.copyMakeBorder(resized, top, size - rh - top, left, size - rw - left,
                               cv2.BORDER_CONSTANT, value=COLOR)
    return canvas, FrameGeometry(h, w, rh, rw, top, left)


def suppress_duplicates(predictions, nms_iou=None):
    """Optional per-class greedy NMS; the default remains the native NMS-free output."""
    if nms_iou is None:
        return predictions
    from metrics import iou
    kept = []
    for p in sorted(predictions, key=lambda p: p[5], reverse=True):
        if all(p[0] != other[0] or iou(p[1:5], other[1:5]) <= nms_iou for other in kept):
            kept.append(p)
    return kept


def predict_frame(model, frame, imgsz=960, device="0", conf=EVALUATION_CONF, normalized=False, nms_iou=None, tta=None):
    """Return [class, x1, y1, x2, y2, confidence] through the shared geometry."""
    preprocessing_spec(imgsz, nms_iou, tta)
    if not 0 <= conf <= 1:
        raise ValueError("置信度必须在0和1之间")
    if tta is not None:
        from multi_view import predict_flip_ensemble
        return predict_flip_ensemble(model,frame,imgsz,device,conf,normalized,nms_iou)
    canvas, geometry = prepare_frame(frame, imgsz)
    # Ultralytics uses score > conf. Our operating metrics use >=, including exact breakpoints.
    backend_conf = max(0., float(np.nextafter(np.float32(conf), np.float32(-np.inf))))
    result = model.predict(canvas, conf=backend_conf, imgsz=canvas.shape[0], device=device,
                           end2end=True, rect=False, max_det=MAX_DET, verbose=False)[0]
    raw = result.boxes.data.cpu().numpy().reshape(-1, 6)
    restored = geometry.restore(raw[:, :4])
    if normalized:
        restored /= [geometry.width, geometry.height, geometry.width, geometry.height]
    # Boxes wholly within gray padding collapse after clipping and are not image detections.
    predictions = [[int(row[5]), *box.tolist(), float(row[4])] for row, box in zip(raw, restored)
                   if row[4] >= conf and box[2] > box[0] and box[3] > box[1]]
    return suppress_duplicates(predictions, nms_iou)


def check_threshold_preprocessing(info, imgsz, nms_iou=None, tta=None):
    if info.get("preprocessing") != preprocessing_spec(imgsz, nms_iou, tta):
        raise ValueError("阈值文件的预处理版本不匹配，请使用当前 evaluate.py 在验证集重新校准。")
