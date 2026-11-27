"""Conditional second stage: preserve background improvement and recover crack recall."""
from argparse import Namespace
from common import ROOT,write_json,read_json,digest,now
from review_training import ReviewedSamplerTrainer
from train import train
from experiment_road_expansion import evaluate,mark_completed

class RoadRecoveryTrainer(ReviewedSamplerTrainer):
    hard_multiplier=1.

def main():
    report=ROOT/'reports/data-expansion-20261029/recovery'
    data=ROOT/'data/prepared/crack-road-expanded-20261029'
    source=ROOT/'runs/concrete-crack-road-expanded-20261029/weights/best_deployment.pt'
    config=ROOT/'configs/road-recovery-20261029.yaml'
    if (report/'plan.json').exists():raise ValueError('Preserve previous experiment')
    first=read_json(ROOT/'reports/data-expansion-20261029/comparison.json')
    write_json(report/'plan.json',dict(at=now(),rationale='First stage reduced negative alarms 94 to4 but selected R88.17%; no joint P80/R90 operating point. Reduce negative sampling2x to1x and lr3e-4 to1e-4 to attempt recall recovery.',
        source_sha256=digest(source),data_sha256=digest(data/'provenance.json'),config_sha256=digest(config),epochs=6,seed=42,
        prior_experiment=first['candidate']['at_selected'],selection='Same fixed mixed val and same DeploymentSelection policy; no test evaluation.',promotion_rule=read_json(ROOT/'reports/data-expansion-20261029/plan.json')['promotion_rule'],
        scope='Conditional development iteration; stage1 val already observed; no independent acceptance claim. Combined source/sampling/lr change is not isolated causal evidence.',formal_acceptance=False))
    train(Namespace(data=str(data),model=str(source),config=str(config),device='0',name='concrete-crack-road-recovery-20261029',smoke=False,resume=None,epochs=None,imgsz=None,batch=None,trainer_class=RoadRecoveryTrainer,align_preprocessing=True,deployment_selection=True,selection_nms_iou=.3))
    mark_completed(ROOT/'runs/concrete-crack-road-recovery-20261029',6)
    evaluate(ROOT/'runs/concrete-crack-road-recovery-20261029/weights/best_deployment.pt',report/'candidate-val')
if __name__=='__main__':main()
