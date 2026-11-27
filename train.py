"""Train YOLOv10s; enforce reviewed crack data before a formal training run."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import yaml

from common import ROOT, NAMES, digest, now, read_json, resolve, write_json
from prepare_data import dataset_readiness


def gpu_check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("没有可用的 CUDA PyTorch。请运行 setup.ps1 配置独立 GPU 环境。")
    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    (x @ x.T).square().mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError("GPU 反向传播检查失败")
    info = {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "architectures": torch.cuda.get_arch_list(),
            "backward_pass": True, "created_at": now()}
    write_json(ROOT/"reports/environment.json", info)
    del x
    torch.cuda.empty_cache()
    return info


def train(args):
    directory = resolve(args.data).parent if str(args.data).endswith(".yaml") else resolve(args.data)
    readiness = dataset_readiness(directory)
    write_json(ROOT/"reports/training_readiness.json", readiness)
    if not args.smoke and not readiness["ready"]:
        raise ValueError("正式训练尚未开始，缺少以下数据条件：\n" + "\n".join(readiness["gaps"]))
    if args.smoke and readiness["purpose"] != "pipeline_smoke":
        raise ValueError("--smoke 仅允许使用明确标记的流程测试数据集")
    import torch
    from ultralytics import YOLO

    environment = gpu_check() if args.device != "cpu" else {"device": "cpu"}
    config_path = resolve(getattr(args, "config", None) or "configs/train.yaml")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pretrained = resolve(args.model) if args.model else ROOT/"weights/yolov10s.pt"
    if not args.resume and not pretrained.is_file():
        raise ValueError(f"预训练权重不存在: {pretrained}。先运行 setup.ps1 或将官方 yolov10s.pt 放入 weights/。")
    cfg.pop("model")
    cfg.update(data=str(directory/"data.yaml"), device=args.device, project=str(ROOT/"runs"),
               name=args.name, exist_ok=False)
    if args.smoke:
        cfg.update(epochs=1, imgsz=320, batch=2, patience=1, mosaic=0.0, close_mosaic=0)
    for key in ["epochs", "imgsz", "batch"]:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.resume:
        resume = resolve(args.resume)
        meta = read_json(resume.with_suffix(".release.json"), {})
        if meta.get("provenance_sha256") != readiness["provenance_sha256"] or meta.get("purpose") != readiness["purpose"]:
            raise ValueError("恢复权重的数据来源记录不一致，拒绝混用实验。")
        if not args.smoke and meta.get("status") == "smoke_only":
            raise ValueError("不能把流程测试训练恢复为正式训练")
    batch = int(cfg["batch"])
    attempts = []
    while True:
        model = YOLO(str(resolve(args.resume)) if args.resume else str(pretrained))
        if model.model.model[-1].__class__.__name__ != "v10Detect":
            raise ValueError("需要 YOLOv10 的 v10Detect 检测头")
        run_meta = {"status": "smoke_only" if args.smoke else "trained_unvalidated",
                    "purpose": readiness["purpose"], "classes": NAMES, "created_at": now(),
                    "data_directory": str(directory), "provenance_sha256": readiness["provenance_sha256"],
                    "environment": environment, "training_args": cfg, "attempts": attempts,
                    "initial_weights": str(pretrained), "initial_weights_sha256": digest(pretrained),
                    "config_path": str(config_path), "config_sha256": digest(config_path)}
        if getattr(args, "trainer_class", None) is not None:
            run_meta["trainer_class"] = f"{args.trainer_class.__module__}.{args.trainer_class.__name__}"
            if (directory/"sampling.json").exists():
                run_meta["sampling_configuration_sha256"] = digest(directory/"sampling.json")

        if getattr(args, "align_preprocessing", False):
            from preprocessing import preprocessing_spec
            run_meta["training_preprocessing"] = preprocessing_spec(cfg["imgsz"])
        if getattr(args, "deployment_selection", False):
            from training_improvements import DeploymentSelection
            monitor = DeploymentSelection(directory, cfg["imgsz"], args.device, run_meta,
                                          getattr(args, "selection_nms_iou", None))
            if args.resume:
                monitor.restore(resolve(args.resume).parent.parent, model.ckpt['epoch'] + 1)
            model.add_callback("on_fit_epoch_end", monitor)

        def save_metadata(trainer):
            for filename in ["last.pt", "best.pt"]:
                weights = Path(trainer.save_dir)/"weights"/filename
                if weights.exists():
                    write_json(weights.with_suffix(".release.json"), {**run_meta, "weights_sha256": digest(weights)})

        model.add_callback("on_model_save", save_metadata)
        try:
            overrides = dict(cfg, batch=batch)
            if args.resume:
                # The checkpoint supplies original optimizer/data settings; batch is the safe OOM override.
                overrides = {"resume": True, "batch": batch, "device": args.device}
            if getattr(args, "align_preprocessing", False):
                from training_improvements import AlignedDetectionTrainer
                overrides["trainer"] = AlignedDetectionTrainer
            if getattr(args, "trainer_class", None) is not None:
                overrides["trainer"] = args.trainer_class
            model.train(**overrides)
            save_metadata(model.trainer)
            run_dir = Path(model.trainer.save_dir)
            write_json(run_dir/"run_summary.json", {**run_meta, "save_dir": str(run_dir),
                       "batch_used": model.trainer.batch_size})
            print(f"\n训练输出: {run_dir}\n状态: {run_meta['status']}（尚未通过正式裂缝检测验收）")
            return
        except torch.OutOfMemoryError:
            attempts.append({"batch": batch, "reason": "CUDA OOM", "at": now()})
            del model
            gc.collect(); torch.cuda.empty_cache()
            if batch <= 1:
                raise RuntimeError("batch=1 仍显存不足；训练未完成。请关闭占用 GPU 的程序后恢复训练。")
            batch = 2 if batch > 2 else 1
            print(f"显存不足，重启本次训练并将 batch 调至 {batch}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/prepared/crack-seg-user-20261021")
    p.add_argument("--model")
    p.add_argument("--config", help="独立实验配置；默认configs/train.yaml")
    p.add_argument("--align-preprocessing", action="store_true", help="训练方图增加与部署一致的16像素边框")
    p.add_argument("--deployment-selection", action="store_true", help="仅用val选择P≥80%时召回率最高的候选")
    p.add_argument("--smoke", action="store_true", help="仅验证训练管线，不证明识别效果")
    p.add_argument("--resume", help="last.pt；支持恢复正式训练的优化器状态")
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="concrete-crack-v10s")
    for name in ["epochs", "imgsz", "batch"]:
        p.add_argument(f"--{name}", type=int)
    args = p.parse_args()
    if any(getattr(args, k) is not None and getattr(args, k) <= 0 for k in ["epochs", "imgsz", "batch"]):
        p.error("epochs/imgsz/batch 必须为正数")
    try:
        train(args)
    except (ValueError, RuntimeError, OSError) as e:
        write_json(ROOT/"reports/training_blockers.json", {"created_at": now(), "error": str(e)})
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
