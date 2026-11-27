"""Shared paths, labels and release checks. No model is loaded on import."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
# Set before importing Ultralytics. Avoid OneDrive configuration hydration on startup.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.home()/".cache/concrete-yolov10"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_OFFLINE", "true")
NAMES = {0: "crack"}
COLORS = {0: (0, 0, 255)}  # OpenCV BGR: red
MANIFEST = ROOT / "data/manifest-crack.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def relative(path):
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def read_json(path, default=None):
    p = resolve(path)
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else default


def write_json(path, value):
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(p)


def digest(path):
    h = hashlib.sha256()
    with resolve(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path=MANIFEST):
    return read_json(path, {"version": 1, "classes": NAMES, "records": []})


def validate_boxes(boxes):
    for b in boxes:
        if len(b) != 5:
            raise ValueError("每个框必须为 [class, cx, cy, width, height]")
        c, x, y, w, h = b
        if c not in NAMES or int(c) != c:
            raise ValueError(f"非法类别: {c}")
        if not all(__import__("math").isfinite(v) for v in b):
            raise ValueError("框包含非有限数")
        if not (0 < w <= 1 and 0 < h <= 1 and 0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"框坐标越界: {b}")
        if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6:
            raise ValueError(f"框超出图片边界: {b}")


def read_image(path):
    import cv2
    import numpy as np
    img = cv2.imdecode(np.fromfile(str(resolve(path)), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def save_image(path, image):
    import cv2
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix or ".jpg", image)
    if not ok:
        raise ValueError(f"无法编码图片: {path}")
    buf.tofile(str(p))


def check_model(model, weights, allow_unvalidated=False):
    names = {int(k): v for k, v in model.names.items()}
    if names != NAMES:
        raise ValueError(f"权重类别不符: {names}；需要 {NAMES}。不能使用 COCO 通用权重。")
    sidecar = resolve(weights).with_suffix(".release.json")
    meta = read_json(sidecar, {})
    if meta.get("weights_sha256") != digest(weights):
        raise ValueError("模型说明缺失或与权重不匹配；请用 train.py 生成权重说明。")
    validated = meta.get("status") == "accepted"
    if not validated and not allow_unvalidated:
        raise ValueError("此权重尚未通过裂缝检测验收。开发预览需明确加 --allow-unvalidated；流程测试不能用于项目效果结论。")
    return meta, validated
