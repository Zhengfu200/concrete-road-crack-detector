"""Exploratory train-negative experiment; keep existing validation/test labels fixed."""
from argparse import Namespace
from common import ROOT,now,write_json,digest,read_json
from review_training import ReviewedSamplerTrainer
from train import train

class BackgroundHeadTrainer(ReviewedSamplerTrainer):
    hard_multiplier=8.

def main():
    data=ROOT/'data/prepared/crack-connected-v2-background-20261027'
    source=ROOT/'runs/concrete-crack-connected-v2-head-20261027/weights/best_deployment.pt'
    config=ROOT/'configs/connected-v2-head-20261027.yaml'
    report=ROOT/'reports/optimization-20261027/background-head'
    if (report/'plan.json').exists():raise ValueError('Experiment already exists')
    write_json(report/'plan.json',dict(at=now(),epochs=6,seed=42,selection_split='val',
         data_sha256=digest(data/'provenance.json'),source_sha256=digest(source),config_sha256=digest(config),
         hypothesis='train-derived verified negative supervision may reduce background false alarms',
         changes='62 derived crops, negative weight8, weighted replacement sampler; entire scheme comparison',
         prior_test_observed=True,prior_test_used_as_development_diagnosis=True,
         new_test_to_be_used_for_selection=False,repeat_test_not_planned=True,
         scope='exploratory validation; no new independent acceptance set available',
         baseline=read_json(ROOT/'reports/optimization-20261027/head-single/val/report.json')['metrics']))
    train(Namespace(data=str(data),model=str(source),config=str(config),device='0',
          name='concrete-crack-connected-v2-background-20261027',smoke=False,resume=None,
          epochs=None,imgsz=None,batch=None,trainer_class=BackgroundHeadTrainer,
          align_preprocessing=True,deployment_selection=True,selection_nms_iou=.3))

if __name__=='__main__':main()
