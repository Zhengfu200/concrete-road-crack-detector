"""Complete the predeclared negative experiment using val only; never repeat test tuning."""
from argparse import Namespace
from pathlib import Path
from common import ROOT,now,read_json,write_json,digest
from evaluate import run

def main():
    base=ROOT/'reports/optimization-20261027'
    run_dir=ROOT/'runs/concrete-crack-connected-v2-background-20261027'
    assert (run_dir/'run_summary.json').is_file()
    output=base/'background-head/val'
    if (output/'report.json').exists():raise ValueError('Do not replace completed validation')
    weights=run_dir/'weights/best_deployment.pt'
    data=ROOT/'data/prepared/crack-connected-v2-background-20261027'
    thresholds=output/'val.thresholds.json'
    run(Namespace(weights=str(weights),data=str(data),split='val',calibrate=True,thresholds=str(thresholds),
        nms_iou=.3,tta=None,imgsz=960,device='0',output=str(output),smoke=False,reference_evaluation=False))
    report=read_json(output/'report.json')
    previous=read_json(base/'head-single/val/report.json')
    def key(r):
        c=r['calibration'];p=c['recall_at_precision80']
        return(c['joint_target_feasible'],p['metrics']['0']['recall'] if p else 0.,
               c['selected']['metrics']['0']['f1'])
    improved=key(report)>key(previous)
    decision=dict(at=now(),improved=improved,previous=previous['metrics']['per_class']['0'],
          candidate=report['metrics']['per_class']['0'],ap=report['average_precision']['0'],
          selected_epoch=read_json(weights.with_suffix('.release.json'))['selected_epoch'],
          identical_val_labels=True,selection_split='val',test_repeated=False,
          source_sha256=digest(weights),report=str(output/'report.json'))
    # Keep the candidate separate: its test has not been performed and the available
    # holdout has already informed development. A val gain is not acceptance evidence.
    decision['preview_replaced']=False
    decision['reason']='exploratory candidate; keep the separately tested head candidate as default preview'
    write_json(base/'background-head/decision.json',decision)
    print(decision,flush=True)

if __name__=='__main__':main()
