"""Run a bounded fine-tuning experiment; compare on val without consuming test data."""
from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from common import ROOT, digest, now, read_json, write_json
from metrics import calibrate, calibration_summary
from preprocessing import preprocessing_spec


def main():
    from evaluate import run as evaluate
    from train import train
    name = 'concrete-crack-finetune-aligned-20261024-v1'
    run_dir = ROOT/'runs'/name
    output = ROOT/'reports'/name
    if run_dir.exists() or output.exists():
        raise ValueError('本实验目录已存在，拒绝覆盖；需要显式创建新的实验名称。')
    source = ROOT/'runs/concrete-crack-20261021-212347/weights/best.pt'
    baseline_dir = ROOT/'reports/concrete-crack-20261021-212347/unified-preprocessing-20261024'
    original_paths = [source, source.with_suffix('.release.json'), source.with_suffix('.thresholds.json'),
                      ROOT/'reports/concrete-crack-20261021-212347/test/report.json']
    originals = {str(p): digest(p) for p in original_paths}
    baseline_report = read_json(baseline_dir/'report.json')
    if (baseline_report['weights_sha256'] != digest(source)
            or baseline_report['preprocessing'] != preprocessing_spec(960)):
        raise ValueError('基线权重或预处理版本不匹配')
    _, baseline_curve = calibrate(read_json(baseline_dir/'predictions.json'))
    baseline = calibration_summary(baseline_curve)
    status = {'status': 'training', 'name': name, 'started_at': now(), 'source_weights': str(source),
              'source_weights_sha256': digest(source), 'baseline_calibration': baseline,
              'baseline_ap': baseline_report['average_precision'], 'selection_split': 'val',
              'test_used': False, 'preprocessing': preprocessing_spec(960),
              'limitations': ['No new background images', 'Concrete domain/source groups not reviewed',
                              'One seed; recipe comparison does not isolate individual factor effects']}
    write_json(output/'experiment.json', status)
    try:
        train(argparse.Namespace(data='data/prepared/crack-seg-user-20261021', model=str(source),
              config='configs/finetune-aligned.yaml', smoke=False, resume=None, device='0', name=name,
              epochs=None, imgsz=None, batch=None, align_preprocessing=True, deployment_selection=True))
        weights = run_dir/'weights/best_deployment.pt'
        evaluate(argparse.Namespace(weights=str(weights), data='data/prepared/crack-seg-user-20261021',
                 split='val', calibrate=True, thresholds=None, imgsz=960, device='0',
                 output=str(output/'val'), smoke=False))
        report = read_json(output/'val/report.json')
        candidate = report['calibration']
        def r80(summary):
            point = summary['recall_at_precision80']
            return point['metrics']['0']['recall'] if point else 0.
        baseline_f1 = baseline['selected']['metrics']['0']['f1']
        candidate_f1 = candidate['selected']['metrics']['0']['f1']
        key = lambda s: (s['joint_target_feasible'], r80(s), s['selected']['metrics']['0']['f1'])
        unchanged = originals == {str(p): digest(p) for p in original_paths}
        if not unchanged:
            raise RuntimeError('原始模型或验收文件意外变化')
        status.update(status='completed', completed_at=now(), candidate_weights=str(weights),
                      candidate_calibration=candidate, candidate_ap=report['average_precision'],
                      validation_objective_improved=key(candidate) > key(baseline),
                      recall_at_precision80_delta=r80(candidate)-r80(baseline),
                      preview_f1_delta=candidate_f1-baseline_f1,
                      original_artifacts_unchanged=unchanged,
                      formal_acceptance_passed=False, test_used=False)
        write_json(output/'experiment.json', status)
        deployment = read_json(run_dir/'deployment-validation/progress.json')
        deployment['status'] = 'completed'
        write_json(run_dir/'deployment-validation/progress.json', deployment)
        text = (f"# 部署对齐微调实验\n\n状态：完成；仅使用验证集比较，尚未正式验收。\n\n"
                f"| 指标 | 原模型 | 新候选 |\n|---|---:|---:|\n"
                f"| Recall @ Precision≥80% | {r80(baseline):.2%} | {r80(candidate):.2%} |\n"
                f"| 预览工作点F1 | {baseline_f1:.4f} | {candidate_f1:.4f} |\n"
                f"| mAP50 | {baseline_report['average_precision']['0']['map50']:.2%} | {report['average_precision']['0']['map50']:.2%} |\n"
                f"| mAP50-95 | {baseline_report['average_precision']['0']['map50_95']:.2%} | {report['average_precision']['0']['map50_95']:.2%} |\n\n"
                f"验证目标改善：{status['validation_objective_improved']}。联合P≥80%、R≥90%存在可行阈值：{candidate['joint_target_feasible']}。\n\n"
                f"候选：{weights}\n\n原模型、原阈值和原测试验收文件哈希未变。未增加或伪造无裂缝样本，未读取test图像调参。"
                "现有材质、来源独立性和背景覆盖限制仍然存在；这是一组单种子微调方案对照，不能归因于单个参数。\n\n"
                "方法参考：使用experimental-design技能组织对照与预先声明的选择标准。Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). "
                "[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。\n")
        (output/'comparison.md').write_text(text, encoding='utf-8')
        print(text, flush=True)
    except BaseException:
        status.update(status='failed', failed_at=now(), error=traceback.format_exc())
        write_json(output/'experiment.json', status)
        raise


if __name__ == '__main__':
    main()
