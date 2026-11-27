"""Calibrate thresholds on validation only, then evaluate an untouched test set."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import (ROOT, NAMES, check_model, digest, now, read_image, read_json,
                    resolve, save_image, write_json)
from metrics import calibrate, calibration_summary, class_metrics, confusion, xywh_to_xyxy
from prepare_data import dataset_readiness
from preprocessing import (EVALUATION_CONF, check_threshold_preprocessing,
                           predict_frame, preprocessing_spec)


def evaluation_context(args, weights, training_sha256, evaluation_sha256):
    """Cross-version comparisons are validation-only and cannot overwrite deployment assets."""
    reference = getattr(args, 'reference_evaluation', False)
    if reference:
        if args.split != 'val':
            raise ValueError('参考评估仅允许验证集，禁止对test跨版本评估或验收')
        if not args.output or not args.thresholds:
            raise ValueError('参考评估须显式指定独立 --output 和 --thresholds')
        output, thresholds = resolve(args.output).resolve(), resolve(args.thresholds).resolve()
        if output == weights.parent.resolve() or output.is_relative_to(weights.parent.resolve()):
            raise ValueError('参考评估输出不能位于权重目录')
        if not thresholds.is_relative_to(output) or thresholds.name in {'report.json', 'predictions.json'}:
            raise ValueError('参考阈值必须在独立输出目录内，且不能与报告重名')
        if args.calibrate and (thresholds.exists() or (output.exists() and any(output.iterdir()))):
            raise ValueError('参考评估校准输出已存在，请使用新的报告目录以保留历史')
    else:
        if training_sha256 != evaluation_sha256:
            raise ValueError('评估数据与训练时的分组记录不一致；跨版本基线须使用 --reference-evaluation --split val')
        output = resolve(args.output or f'reports/{weights.parent.parent.name}/{args.split}')
        thresholds = resolve(args.thresholds) if args.thresholds else weights.with_suffix('.thresholds.json')
    return reference, output, thresholds


def collect(model, directory, split, imgsz, device, nms_iou=None, tta=None):
    info = read_json(directory/"provenance.json")
    samples = []
    for row in info["splits"][split]:
        image = directory/"images"/split/f"{row['id']}{resolve(row['image']).suffix.lower()}"
        img = read_image(image)
        preds = predict_frame(model, img, imgsz, device, conf=EVALUATION_CONF, normalized=True, nms_iou=nms_iou, tta=tta)
        samples.append({"id": row["id"], "image": str(image),
                        "gt": [xywh_to_xyxy(b) for b in row["boxes"]], "pred": preds})
    return samples


def average_precision(samples, output):
    """Use Ultralytics IoU matching/AP on the SAME predictions as operating-point metrics."""
    import numpy as np
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator

    output.mkdir(parents=True, exist_ok=True)
    validator = DetectionValidator(save_dir=output, args={"plots": True})
    validator.metrics.names = NAMES
    for sample in samples:
        gt = torch.tensor(sample["gt"], dtype=torch.float32).reshape(-1, 5)
        pred = torch.tensor(sample["pred"], dtype=torch.float32).reshape(-1, 6)
        batch = {"cls": gt[:, 0], "bboxes": gt[:, 1:5]}
        detections = {"cls": pred[:, 0], "bboxes": pred[:, 1:5]}
        validator.metrics.update_stats({
            **validator._process_batch(detections, batch),
            "conf": pred[:, 5].numpy(), "pred_cls": pred[:, 0].numpy(),
            "target_cls": gt[:, 0].numpy(), "target_img": np.unique(gt[:, 0].numpy()),
            "im_name": sample["id"]})
    if not samples:
        raise ValueError("评估集合为空")
    validator.metrics.process(save_dir=output, plot=True)
    return {str(c): {"map50": float(validator.metrics.box.ap50[i]),
                     "map50_95": float(validator.metrics.box.ap[i])}
            for i, c in enumerate(validator.metrics.box.ap_class_index.tolist())}


def run(args):
    from ultralytics import YOLO
    weights = resolve(args.weights)
    directory = resolve(args.data)
    readiness = dataset_readiness(directory)
    nms_iou = getattr(args, "nms_iou", None)
    tta = getattr(args, 'tta', None)
    preprocessing = preprocessing_spec(args.imgsz, nms_iou, tta)
    model = YOLO(str(weights))
    meta, _ = check_model(model, weights, allow_unvalidated=True)
    reference, output, thresholds_path = evaluation_context(
        args, weights, meta.get('provenance_sha256'), readiness['provenance_sha256'])
    if not readiness["ready"] and not args.smoke:
        raise ValueError("正式评估数据尚未就绪：\n"+"\n".join(readiness["gaps"]))
    if args.calibrate and args.split != "val":
        raise ValueError("只允许在验证集 (--split val) 校准阈值；禁止在测试集调参。")
    output.mkdir(parents=True, exist_ok=True)
    if args.calibrate:
        samples = collect(model, directory, args.split, args.imgsz, args.device, nms_iou, tta)
        thresholds, curve = calibrate(samples)
        threshold_info = {"thresholds": thresholds, "created_at": now(), "split": "val",
                          "weights_sha256": digest(weights), "provenance_sha256": readiness["provenance_sha256"],
                          "imgsz": args.imgsz, "preprocessing": preprocessing,
                          "curve": curve, "calibration": calibration_summary(curve), "smoke_only": args.smoke,
                          "reference_evaluation": reference,
                          "training_provenance_sha256": meta.get('provenance_sha256'),
                          "evaluation_provenance_sha256": readiness['provenance_sha256']}
        write_json(thresholds_path, threshold_info)
    else:
        threshold_info = read_json(thresholds_path)
        if not threshold_info or threshold_info.get("split") != "val":
            raise ValueError("缺少验证集阈值文件，先使用 --split val --calibrate。")
        if bool(threshold_info.get('reference_evaluation', False)) != reference:
            raise ValueError('参考评估阈值不能用于常规评估或正式验收')
        if reference and threshold_info.get('training_provenance_sha256') != meta.get('provenance_sha256'):
            raise ValueError('参考评估阈值的训练数据摘要不匹配')
        if threshold_info.get("weights_sha256") != digest(weights) or threshold_info.get("provenance_sha256") != readiness["provenance_sha256"]:
            raise ValueError("阈值文件与模型/数据不匹配")
        if threshold_info.get("imgsz") != args.imgsz:
            raise ValueError("推理尺寸与阈值校准时不同，请保持 imgsz 一致。")
        check_threshold_preprocessing(threshold_info, args.imgsz, nms_iou, tta)
        if threshold_info.get("smoke_only") and not args.smoke:
            raise ValueError("不能使用流程测试的阈值进行正式评估")
        thresholds = {int(k): v for k, v in threshold_info["thresholds"].items()}
        samples = collect(model, directory, args.split, args.imgsz, args.device, nms_iou, tta)
    measured = class_metrics(samples, thresholds)
    ap = average_precision(samples, output/"ultralytics")
    scenarios = measured["scenarios"]
    coverage_ok = all(scenarios[k] > 0 for k in scenarios)
    crack = measured["per_class"]["0"]
    checks = {"crack_recall_ge_90": crack["recall"] >= .9,
              "crack_precision_ge_80": crack["precision"] >= .8,
              "crack_and_background_present": coverage_ok,
              "formal_data_ready": readiness["ready"],
              "concrete_domain_reviewed": readiness.get("domain_reviewed", True)}
    accepted = not reference and args.split == "test" and not args.smoke and all(checks.values())
    report = {"created_at": now(), "split": args.split, "smoke_only": args.smoke,
              "thresholds": thresholds, "operating_point_iou": .5, "metrics": measured,
              "average_precision": ap, "confusion_true_rows_pred_columns": confusion(samples, thresholds),
              "confusion_classes": ["crack", "background"], "checks": checks,
              "accepted": accepted, "weights_sha256": digest(weights),
              "provenance_sha256": readiness["provenance_sha256"], "imgsz": args.imgsz,
              "reference_evaluation": reference,
              "training_provenance_sha256": meta.get('provenance_sha256'),
              "evaluation_provenance_sha256": readiness['provenance_sha256'],
              "preprocessing": preprocessing, "prediction_min_conf": EVALUATION_CONF,
              "calibration": threshold_info.get("calibration"),
              "ap_backend": "Ultralytics DetectionValidator matching + DetMetrics; shared predictions"}
    write_json(output/"report.json", report)
    write_json(output/"predictions.json", samples)
    # Plot the very same operating-point matrix as report.json (native plots use pred rows).
    import numpy as np
    from ultralytics.utils.metrics import ConfusionMatrix
    matrix = ConfusionMatrix(names=NAMES)
    matrix.matrix = np.asarray(report["confusion_true_rows_pred_columns"]).T
    for normalize in (True, False):
        matrix.plot(normalize=normalize, save_dir=output/"ultralytics")
    # Save actual false positives/negatives (not random successful examples).
    from inference import draw_detections
    saved = 0
    for s in samples:
        single = class_metrics([s], thresholds)
        if any(m["fp"] or m["fn"] for m in single["per_class"].values()):
            img = read_image(s["image"])
            img = draw_detections(img, s["pred"], thresholds, normalized=True)[0]
            import cv2
            h, w = img.shape[:2]
            for g in s["gt"]:
                cv2.rectangle(img, (int(g[1]*w), int(g[2]*h)), (int(g[3]*w), int(g[4]*h)), (0,255,0), 1)
            save_image(output/"errors"/f"{s['id']}.jpg", img)
            saved += 1
            if saved >= 20:
                break
    lines = ["# 模型评估报告", "", f"集合：{args.split}；流程测试：{args.smoke}",
             f"跨版本验证集参考评估：{reference}（参考评估不改变模型验收状态）。",
             f"检测框匹配 IoU：0.5；内容区域：{args.imgsz}；实际输入：{preprocessing['input_size']}",
             f"预处理：{preprocessing['version']}；保持比例、居中灰色填充、四周16像素边框。",
             f"固定工作阈值：{thresholds}；预测采集最低置信度：{EVALUATION_CONF}。",
             f"验证集联合目标存在可行阈值：{threshold_info.get('calibration', {}).get('joint_target_feasible', '旧阈值未记录')}。",
             "P/R、mAP和视频检测共用预测路径；mAP使用Ultralytics匹配与积分算法。", "",
             "| 类别 | Precision | Recall | mAP50 | mAP50-95 |", "|---|---:|---:|---:|---:|"]
    for c, name in NAMES.items():
        m = measured["per_class"][str(c)]; a = ap.get(str(c), {})
        lines.append(f"| {name} | {m['precision']:.2%} | {m['recall']:.2%} | {a.get('map50',0):.2%} | {a.get('map50_95',0):.2%} |")
    lines.extend(["", f"裂缝/无裂缝场景图片数：{scenarios}", f"验收检查：{checks}",
                  f"模型精度验收：{'通过' if accepted else '未通过/未进行正式验收'}", "",
                  "绿色细框为真实标注；红框为裂缝预测。摄像头帧率、延迟与10分钟稳定性需另测。",
                  "流程测试的数据量和标签未经正式复核，其指标不能作为课题性能结论。"])
    (output/"report.md").write_text("\n".join(lines), encoding="utf-8")
    if not reference and args.split == "test" and not args.smoke:
        meta.update(status="accepted" if accepted else "evaluation_failed", evaluation_report=str(output/"report.json"),
                    thresholds_path=str(thresholds_path), evaluation_at=now(), evaluated_imgsz=args.imgsz,
                    evaluated_preprocessing=preprocessing)
        write_json(weights.with_suffix(".release.json"), meta)
    print(f"评估报告: {output/'report.md'}\n正式精度验收通过: {accepted}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="data/prepared/formal-crack-v1")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--thresholds")
    p.add_argument("--nms-iou", type=float, help="可选重复框抑制；默认保留YOLOv10原生无NMS输出")
    p.add_argument('--tta', choices=['flip4'], help='四方向翻转集成；需要额外推理时间')
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default="0")
    p.add_argument("--output")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--reference-evaluation", action="store_true", help="仅val跨数据版本参考评估；独立报告和阈值，不更改模型验收状态")
    args = p.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
