"""Deployment-aligned fine-tuning and validation-only checkpoint selection."""
from __future__ import annotations

import gc
from pathlib import Path
import shutil

from ultralytics.models.yolo.detect import DetectionTrainer

from common import digest, now, read_json, write_json
from metrics import calibration_summary, threshold_curve
from preprocessing import BORDER, COLOR, preprocessing_spec


def pad_training_batch(batch):
    """Augmented square images are already resized; shift and renormalize labels with padding."""
    import torch.nn.functional as functional
    height, width = batch["img"].shape[-2:]
    if height != width:
        raise ValueError("部署对齐训练要求方形内容区域")
    batch["img"] = functional.pad(batch["img"], (BORDER,)*4, value=COLOR[0]/255)
    boxes = batch["bboxes"].clone()
    boxes[:, :2] = (boxes[:, :2]*width + BORDER)/(width+2*BORDER)
    boxes[:, 2:] *= width/(width+2*BORDER)
    batch["bboxes"] = boxes
    return batch


class AlignedDetectionTrainer(DetectionTrainer):
    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)
        return pad_training_batch(batch)


class DeploymentSelection:
    """Inspect every epoch with the unchanged val set; never read test images."""
    def __init__(self, directory, imgsz, device, metadata, nms_iou=None):
        self.directory, self.imgsz, self.device, self.metadata = directory, imgsz, device, metadata
        self.best_key = None
        self.nms_iou = nms_iou
        self.rows = []
        self.seen = set()

    def restore(self, run_dir, completed_epochs):
        """Resume without forgetting the best earlier deployment checkpoint."""
        run_dir = Path(run_dir)
        previous = read_json(run_dir/'deployment-validation/progress.json')
        if not previous:
            return
        if previous.get('preprocessing') != preprocessing_spec(self.imgsz, self.nms_iou) or previous.get('selection_split') != 'val':
            raise ValueError('恢复训练的部署验证协议不一致')
        rows = previous['epochs']
        epochs = [r['epoch'] for r in rows]
        if len(set(epochs)) != len(epochs) or any(e < 1 or e > completed_epochs for e in epochs):
            raise ValueError('部署验证记录超出恢复检查点轮次')
        if rows:
            chosen = run_dir/'weights/best_deployment.pt'
            meta = read_json(chosen.with_suffix('.release.json'), {})
            if meta.get('provenance_sha256') != self.metadata.get('provenance_sha256') or meta.get('weights_sha256') != digest(chosen):
                raise ValueError('恢复部署候选的来源或权重摘要不一致')
            def key(row):
                summary = row['calibration']
                point = summary['recall_at_precision80']
                return (summary['joint_target_feasible'], point['metrics']['0']['recall'] if point else 0.,
                        summary['selected']['metrics']['0']['f1'])
            self.best_key = max(map(key, rows))
        self.rows, self.seen = rows, set(epochs)

    def __call__(self, trainer):
        # final_eval invokes on_fit_epoch_end again with epoch+1, but no training happened there.
        if getattr(getattr(trainer, "validator", None), "training", True) is False:
            return
        from ultralytics import YOLO
        from evaluate import collect
        import torch
        epoch = trainer.epoch + 1
        if epoch in self.seen:
            return
        self.seen.add(epoch)
        checkpoint = Path(trainer.save_dir)/"weights/last.pt"
        # Loading a separate checkpoint avoids fusing/mutating the training EMA during inference.
        model = YOLO(str(checkpoint))
        samples = collect(model, self.directory, "val", self.imgsz, self.device, self.nms_iou)
        summary = calibration_summary(threshold_curve(samples))
        point = summary["recall_at_precision80"]
        recall = point["metrics"]["0"]["recall"] if point else 0.
        f1 = summary["selected"]["metrics"]["0"]["f1"]
        key = (summary["joint_target_feasible"], recall, f1)
        improved = self.best_key is None or key > self.best_key
        row = {"epoch": epoch, "at": now(), "calibration": summary, "selected_checkpoint": improved}
        self.rows.append(row)
        output = Path(trainer.save_dir)/"deployment-validation"
        write_json(output/"progress.json", {"status": "training", "epochs": self.rows,
                   "preprocessing": preprocessing_spec(self.imgsz, self.nms_iou), "selection_split": "val"})
        if improved:
            self.best_key = key
            chosen = checkpoint.with_name("best_deployment.pt")
            shutil.copy2(checkpoint, chosen)
            write_json(chosen.with_suffix(".release.json"), {
                **self.metadata, "weights_sha256": digest(chosen), "selected_epoch": epoch,
                "selection_split": "val", "selection_calibration": summary})
            write_json(output/"best_predictions.json", samples)
        print(f"\n部署验证 epoch={epoch}: R@P80={recall:.4f}, preview_F1={f1:.4f}, "
              f"joint_feasible={summary['joint_target_feasible']}, selected={improved}", flush=True)
        del model, samples
        gc.collect()
        torch.cuda.empty_cache()
