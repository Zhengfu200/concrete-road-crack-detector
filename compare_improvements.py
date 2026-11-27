"""Compare completed val-only candidates and publish a reproducible preview configuration."""
from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, digest, now, read_json, write_json
from metrics import calibrate, calibration_summary, class_metrics, confusion
from preprocessing import preprocessing_spec, suppress_duplicates


def main():
    experiment_name = 'concrete-crack-finetune-aligned-20261024-v1'
    fine_dir = ROOT/'reports'/experiment_name
    experiment = read_json(fine_dir/'experiment.json')
    if not experiment or experiment['status'] != 'completed':
        raise ValueError('微调实验尚未完成，不能发布完整对照')
    original_dir = ROOT/'reports/concrete-crack-20261021-212347'
    baseline_dir = original_dir/'unified-preprocessing-20261024'
    nms_dir = original_dir/'duplicate-suppression-20261024'
    output = ROOT/'reports/model-improvement-20261024'
    output.mkdir(exist_ok=True)
    from evaluate import average_precision
    fine_report = read_json(fine_dir/'val/report.json')
    fine_samples = read_json(fine_dir/'val/predictions.json')
    filtered = [{**s, 'pred': suppress_duplicates(s['pred'], .3)} for s in fine_samples]
    thresholds, curve = calibrate(filtered)
    calibration = calibration_summary(curve)
    filtered_output = fine_dir/'val-nms03'
    filtered_ap = average_precision(filtered, filtered_output/'ultralytics')
    threshold_info = {**read_json(Path(experiment['candidate_weights']).with_suffix('.thresholds.json')),
                      'thresholds': thresholds, 'curve': curve, 'calibration': calibration,
                      'preprocessing': preprocessing_spec(960, .3), 'created_at': now()}
    write_json(filtered_output/'val.thresholds.json', threshold_info)
    filtered_report = {**fine_report, 'metrics': class_metrics(filtered, thresholds),
                       'thresholds': thresholds, 'average_precision': filtered_ap,
                       'calibration': calibration, 'preprocessing': preprocessing_spec(960, .3),
                       'confusion_true_rows_pred_columns': confusion(filtered, thresholds), 'created_at': now()}
    filtered_report['checks'] = {**fine_report['checks'],
        'crack_recall_ge_90': filtered_report['metrics']['per_class']['0']['recall'] >= .9,
        'crack_precision_ge_80': filtered_report['metrics']['per_class']['0']['precision'] >= .8}
    write_json(filtered_output/'report.json', filtered_report)
    write_json(filtered_output/'predictions.json', filtered)
    baseline = read_json(baseline_dir/'report.json')
    baseline['calibration'] = read_json(baseline_dir/'exact-calibration.json')['calibration']
    baseline['metrics'] = class_metrics(read_json(baseline_dir/'predictions.json'),
        {0: baseline['calibration']['selected']['threshold']})
    weights = ROOT/'runs/concrete-crack-20261021-212347/weights/best.pt'
    rows = [
        ('原模型，共用预处理', baseline, weights, None, None),
        ('原模型＋重复框抑制', read_json(nms_dir/'report.json'), weights, .3, nms_dir/'val.thresholds.json'),
        ('微调候选', fine_report, Path(experiment['candidate_weights']), None,
         Path(experiment['candidate_weights']).with_suffix('.thresholds.json')),
        ('微调候选＋重复框抑制', filtered_report, Path(experiment['candidate_weights']), .3,
         filtered_output/'val.thresholds.json')]
    def objective(row):
        c = row[1]['calibration'];point = c['recall_at_precision80']
        return c['joint_target_feasible'], point['metrics']['0']['recall'] if point else 0., c['selected']['metrics']['0']['f1']
    chosen = max(rows[1:], key=objective)
    records = []
    for name, report, model_path, nms_iou, threshold_path in rows:
        c = report['calibration'];point = c['recall_at_precision80'];m = report['metrics']['per_class']['0']
        records.append({'name': name, 'weights': str(model_path), 'nms_iou': nms_iou,
                        'thresholds': str(threshold_path) if threshold_path else None,
                        'precision': m['precision'], 'recall': m['recall'], 'f1': m['f1'],
                        'recall_at_precision80': point['metrics']['0']['recall'] if point else 0.,
                        'map50': report['average_precision']['0']['map50'],
                        'map50_95': report['average_precision']['0']['map50_95'],
                        'joint_target_feasible': c['joint_target_feasible']})
    selected = next(row for row in records if row['name'] == chosen[0])
    recommended = read_json(chosen[4])
    point = recommended['calibration']['recall_at_precision80']
    if not recommended['calibration']['joint_target_feasible'] and point:
        recommended['thresholds'] = {'0': point['threshold']}
        recommended['calibration'] = {**recommended['calibration'], 'selected': point,
            'selection_policy': 'preview_max_recall_at_precision80_no_joint_threshold'}
    recommended_path = output/'recommended.val.thresholds.json'
    write_json(recommended_path, recommended)
    command = ['detect.py', '--weights', str(chosen[2]), '--thresholds', str(recommended_path), '--allow-unvalidated']
    if chosen[3] is not None:
        command += ['--nms-iou', str(chosen[3])]
    config = {'created_at': now(), 'selection_split': 'val', 'test_used': False,
              'formal_acceptance_passed': False, 'selected': selected, 'candidates': records,
              'preprocessing': chosen[1]['preprocessing'], 'preview_args': command,
              'recommended_operating_point': recommended['calibration']['selected'],
              'method_reference': 'https://doi.org/10.48550/arXiv.2609.00065'}
    write_json(output/'summary.json', config)
    write_json(ROOT/'configs/preview-improved.json', config)
    lines = ['# 现有数据模型优化对照', '', '所有候选仅在同一200张验证集上比较；未作独立测试验收。', '',
             '| 方案 | 精确率 | 召回率 | F1 | Recall@Precision≥80% | mAP50 | mAP50-95 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['name']} | {r['precision']:.2%} | {r['recall']:.2%} | {r['f1']:.4f} | "
                     f"{r['recall_at_precision80']:.2%} | {r['map50']:.2%} | {r['map50_95']:.2%} |")
    lines += ['', f"推荐未验收预览候选：{chosen[0]}。", '',
              '每项精确率/召回率均为该候选在验证集最大F1工作点，R@P80为完整曲线上精确率至少80%时的最高召回率。',
              '默认预览按项目减少漏检的偏好，在无联合可行阈值时选取Precision≥80%下最高召回点，详见recommended.val.thresholds.json。',
              '选择顺序：联合目标是否可行、R@P80、F1。无独立测试通过证据，不能称为最终达标模型。', '',
              '本次训练不增加新数据；无裂缝场景、混凝土道路材质和独立来源组仍缺少复核。',
              '六个NMS参数的开发搜索和单种子训练均使用val，后续正式验收需另留独立数据。', '',
              '模型仍为YOLOv10s；开启重复框抑制的候选增加NMS后处理，不是整条流程无NMS。', '',
              '复核清单：同名微调报告目录的baseline-error-review，未自动更改标签。', '',
              '方法参考：experimental-design技能；Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). '
              '[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。']
    (output/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
