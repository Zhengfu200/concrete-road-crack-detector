"""Complete the reviewed hard-example pilot with uniform and weighted sampling arms."""
from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from common import ROOT,digest,now,read_json,write_json
from preprocessing import preprocessing_spec

OUT=ROOT/'reports/review-ablation-20261024'
DATA=ROOT/'data/prepared/crack-seg-hardreview-20261024'
SOURCE=ROOT/'runs/concrete-crack-20261021-212347/weights/best.pt'


def main():
    from train import train
    from evaluate import run as evaluate
    from review_training import UniformReviewTrainer,HardReviewTrainer
    from metrics import class_metrics
    if OUT.exists():
        raise ValueError('对照目录已存在，拒绝覆盖')
    sampling=read_json(DATA/'sampling.json')
    if not all(sampling['val_test_byte_identical'].values()):
        raise ValueError('验证/测试内容必须与基线一致')
    protected=[SOURCE,SOURCE.with_suffix('.release.json'),SOURCE.with_suffix('.thresholds.json'),
               ROOT/'data/prepared/crack-seg-user-20261021/provenance.json',ROOT/'configs/preview-improved.json']
    before={str(p):digest(p) for p in protected}
    status={'status':'running','started_at':now(),'arms':[],'factor':'reviewed hard sample weight 1 versus3',
            'unique_train_images':640,'agent_inspected_hard_images':len(sampling['hard_ids']),
            'epochs_per_arm':3,'seed':42,'selection_split':'val','nms_iou':.3,
            'same_train_roster':True,'same_config':True,'same_initial_weights':True,'same_steps':True,
            'sampler_replacement_in_both_arms':True,'bn_statistics_frozen_in_both_arms':True,
            'labels_modified':0,'val_test_unchanged':True,'test_used_for_selection':False,
            'one_seed_pilot_only':True,'baseline_weights_sha256':digest(SOURCE)}
    write_json(OUT/'experiment.json',status)
    try:
        for label,trainer in [('uniform',UniformReviewTrainer),('hard3',HardReviewTrainer)]:
            name=f'concrete-crack-review-{label}-20261024-v1'
            if (ROOT/'runs'/name).exists():
                raise ValueError(f'训练目录已存在: {name}')
            status.update(current_arm=label)
            write_json(OUT/'experiment.json',status)
            train(argparse.Namespace(data=str(DATA),model=str(SOURCE),config='configs/review-ablation.yaml',
                smoke=False,resume=None,device='0',name=name,epochs=None,imgsz=None,batch=None,
                align_preprocessing=True,deployment_selection=True,selection_nms_iou=.3,trainer_class=trainer))
            weights=ROOT/'runs'/name/'weights/best_deployment.pt'
            evaluate(argparse.Namespace(weights=str(weights),data=str(DATA),split='val',calibrate=True,
                thresholds=None,imgsz=960,device='0',output=str(OUT/label),smoke=False,nms_iou=.3))
            report=read_json(OUT/label/'report.json')
            status['arms'].append({'label':label,'weights':str(weights),'report':str(OUT/label/'report.json'),
                'calibration':report['calibration'],'ap':report['average_precision'],
                'epoch_selected':read_json(weights.with_suffix('.release.json')).get('selected_epoch'),
                'sampling':read_json(ROOT/'runs'/name/'sampling-configuration.json')})
            progress=read_json(ROOT/'runs'/name/'deployment-validation/progress.json')
            progress['status']='completed'
            write_json(ROOT/'runs'/name/'deployment-validation/progress.json',progress)
            write_json(OUT/'experiment.json',status)
        baseline=read_json(ROOT/'reports/concrete-crack-20261021-212347/duplicate-suppression-20261024/report.json')
        records=[{'label':'原模型+相同NMS','calibration':baseline['calibration'],'ap':baseline['average_precision']}]
        records+=status['arms']
        def recall80(row):
            p=row['calibration']['recall_at_precision80']
            return p['metrics']['0']['recall'] if p else 0.
        def key(row):
            return row['calibration']['joint_target_feasible'],recall80(row),row['calibration']['selected']['metrics']['0']['f1']
        winner=max(records,key=key)
        unchanged=before=={str(p):digest(p) for p in protected}
        if not unchanged:
            raise RuntimeError('受保护原模型/数据/推荐配置发生变化')
        status.update(status='completed',completed_at=now(),winner=winner['label'],
                      hard_minus_uniform_r80=recall80(records[2])-recall80(records[1]),
                      original_artifacts_unchanged=unchanged,formal_acceptance_passed=False)
        write_json(OUT/'experiment.json',status)
        lines=['# 复核困难样本采样对照','',
            '48张验证错误图与48张训练困难图完成Codex图像复核，非独立人工材质/来源验收。',
            '同640张训练图、相同原权重初始化、同配置/种子/3轮；两组只改变34张复核困难样本的权重1或3。',
            '两组均采用有放回抽样、每轮640次；BN统计冻结和低学习率1e-5为两组共有设置。',
            'val/test图片及标签与原版本字节一致。保留每分割多边形对应框的既有单位，本轮不自动合框；实际改标0张。','',
            '| 方案 | P≥80%时最高Recall | 最大F1工作点P | 最大F1工作点R | F1 | mAP50 | mAP50-95 |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for row in records:
            m=row['calibration']['selected']['metrics']['0'];ap=row['ap']['0']
            lines.append(f"| {row['label']} | {recall80(row):.2%} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.4f} | {ap['map50']:.2%} | {ap['map50_95']:.2%} |")
        lines+=['',f"本轮验证选择：{winner['label']}。困难加权相对均匀采样的R@P80差：{status['hard_minus_uniform_r80']:+.2%}。",'',
            '单种子小样本先导结果，只支持此轮方案比较；不证明稳定泛化或正式精度验收通过。原推荐配置不会因试验完成自动替换。',
            '主要瓶颈及逐张观察见reports/hard-example-review-20261024/review-report.md及两个visual-review.csv。','',
            '方法参考：scientific-critical-thinking、experimental-design技能；Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). '
            '[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。']
        (OUT/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
        print('\n'.join(lines),flush=True)
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),failed_at=now())
        write_json(OUT/'experiment.json',status)
        raise


if __name__=='__main__':
    main()
