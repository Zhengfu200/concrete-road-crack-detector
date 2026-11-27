"""2026-10-29: explicit public-data expansion experiment, val-only selection."""
from argparse import Namespace
from common import ROOT,write_json,read_json,digest,now
from review_training import ReviewedSamplerTrainer
from train import train
from evaluate import run

class RoadExpansionTrainer(ReviewedSamplerTrainer):
    hard_multiplier=2.

def mark_completed(run,epochs):
    progress=read_json(run/'deployment-validation/progress.json')
    if len(progress.get('epochs',[]))!=epochs or not (run/'run_summary.json').is_file():
        raise ValueError('Training completion records are incomplete')
    progress.update(status='completed',completed_at=now())
    write_json(run/'deployment-validation/progress.json',progress)

def evaluate(weights,report,reference=False):
    return run(Namespace(weights=str(weights),data=str(ROOT/'data/prepared/crack-road-expanded-20261029'),split='val',imgsz=960,device='0',nms_iou=.3,tta=None,smoke=False,calibrate=True,thresholds=str(report/'val.thresholds.json'),output=str(report),reference_evaluation=reference))

def main():
    report=ROOT/'reports/data-expansion-20261029'
    data=ROOT/'data/prepared/crack-road-expanded-20261029'
    source=ROOT/'runs/concrete-crack-connected-v2-head-20261027/weights/best_deployment.pt'
    config=ROOT/'configs/road-expansion-20261029.yaml'
    if (report/'plan.json').exists():raise ValueError('Experiment already exists; preserve records')
    write_json(report/'plan.json',dict(at=now(),seed=42,epochs=8,source_sha256=digest(source),data_sha256=digest(data/'provenance.json'),config_sha256=digest(config),
        hypothesis='Real pavement non-cracked supervision plus localized concrete cracks can reduce background false positives while maintaining crack recall.',
        selection='8-epoch head-only fine-tune; existing DeploymentSelection lexicographic joint P80/R90, R@P80, F1 on mixed val only; frozen BN; negatives 2x, positives 1x.',
        promotion_rule='Candidate mixed-val P>=.80 and R>=.90 at calibrated threshold; original 164 positive-val R>=.90 at same threshold; fewer negative-image FPs than baseline at its existing deployment threshold. Report also equal-recall comparison. If gates fail keep existing preview.',
        primary_evaluation='Mixed development val: original 164 positive images plus screened SDNET negative prefix-held groups; IoU .5, preprocessing 960+16border, NMS .3.',
        secondary='Same previous 24 evenly spaced target video frames: number of boxes area>=.8 with confidence>=.5; unlabeled video diagnosis, not precision/recall or promotion gate.',
        old_test_observed=True,repeat_test_for_selection=False,formal_acceptance=False,
        limitations='No independent road-scene acceptance set; val used for model selection, grouped negatives rely on filename proxies; DamSegment train-only. This is a combined-scheme experiment, not isolated causal proof.'))
    evaluate(source,report/'baseline-val',reference=True)
    train(Namespace(data=str(data),model=str(source),config=str(config),device='0',name='concrete-crack-road-expanded-20261029',smoke=False,resume=None,epochs=None,imgsz=None,batch=None,trainer_class=RoadExpansionTrainer,align_preprocessing=True,deployment_selection=True,selection_nms_iou=.3))
    mark_completed(ROOT/'runs/concrete-crack-road-expanded-20261029',8)
    candidate=ROOT/'runs/concrete-crack-road-expanded-20261029/weights/best_deployment.pt'
    evaluate(candidate,report/'candidate-val')
if __name__=='__main__':main()
