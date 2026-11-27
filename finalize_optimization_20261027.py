"""Select on validation, freeze once, run one dataset test, then check actual inference."""
from argparse import Namespace
from pathlib import Path
import gc,time
import numpy as np
from common import ROOT,now,read_json,write_json,digest,read_image
from evaluate import run

OUT=ROOT/'reports/optimization-20261027'
RUN=ROOT/'runs/concrete-crack-connected-v2-head-20261027'
WEIGHTS=RUN/'weights/best_deployment.pt'
DATA=ROOT/'data/prepared/crack-connected-v2'

def evaluate(name,tta=None,split='val',thresholds=None):
    output=OUT/name/split
    if (output/'report.json').exists():
        raise ValueError(f'Refuse to overwrite {output}')
    run(Namespace(weights=str(WEIGHTS),data=str(DATA),split=split,calibrate=split=='val',
          thresholds=str(thresholds or output/'val.thresholds.json'),nms_iou=.3,tta=tta,
          imgsz=960,device='0',output=str(output),smoke=False,reference_evaluation=False))
    report=read_json(output/'report.json')
    return dict(name=name,tta=tta,report=str(output/'report.json'),
        thresholds=str(thresholds or output/'val.thresholds.json'),metrics=report['metrics']['per_class']['0'],
        average_precision=report['average_precision']['0'],calibration=report['calibration'])

def main():
    assert (RUN/'run_summary.json').is_file(),'Wait for training completion'
    history=read_json(RUN/'deployment-validation/progress.json')['epochs']
    assert [row['epoch'] for row in history]==list(range(1,7))
    if (OUT/'selection-freeze.json').exists():
        raise ValueError('Selection/test already frozen; do not repeat adaptive testing')
    candidates=[evaluate('head-single'),evaluate('head-flip4','flip4')]
    def key(candidate):
        point=candidate['calibration']['recall_at_precision80']
        return (candidate['calibration']['joint_target_feasible'],
                point['metrics']['0']['recall'] if point else 0.,
                candidate['metrics']['f1'],candidate['tta'] is None)
    chosen=max(candidates,key=key)
    selected_meta=read_json(WEIGHTS.with_suffix('.release.json'))
    freeze=dict(at=now(),weights=str(WEIGHTS),weights_sha256=digest(WEIGHTS),
                selected_epoch=selected_meta['selected_epoch'],data_sha256=digest(DATA/'provenance.json'),
                selected=chosen,candidates=candidates,thresholds_sha256=digest(Path(chosen['thresholds'])),
                selection_split='val',test_used_for_selection=False,
                test_scope='existing dataset holdout, not new independent concrete road acceptance',
                independent_road_data_available=False)
    write_json(OUT/'selection-freeze.json',freeze)
    print('FROZEN CANDIDATE',chosen['name'],chosen['metrics'],flush=True)
    test=evaluate(chosen['name'],chosen['tta'],'test',chosen['thresholds'])
    write_json(OUT/'test-result.json',test)
    # The test result is recorded, with no subsequent selection/threshold change.
    from inference import Detector
    import torch
    gc.collect();torch.cuda.empty_cache()
    detector=Detector(WEIGHTS,allow_unvalidated=True,nms_iou=.3,thresholds=chosen['thresholds'],tta=chosen['tta'])
    validation=read_json(Path(chosen['report']).with_name('predictions.json'))
    timings=[]
    max_difference=0.
    threshold=detector.thresholds[0]
    for i,row in enumerate(validation,1):
        image=read_image(row['image'])
        start=time.perf_counter()
        _,_,actual=detector.predict(image)
        timings.append(time.perf_counter()-start)
        h,w=image.shape[:2]
        actual=[[p[0],p[1]/w,p[2]/h,p[3]/w,p[4]/h,p[5]] for p in actual]
        expected=[p for p in row['pred'] if p[5]>=threshold]
        assert len(actual)==len(expected),(row['id'],len(actual),len(expected))
        if expected:
            difference=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
            assert difference<=1e-6,(row['id'],difference)
            max_difference=max(max_difference,difference)
        if i%40==0: print('live consistency',i,len(validation),flush=True)
    runtime=dict(at=now(),images=len(validation),all_matched=True,max_abs_difference=max_difference,
                 mean_prediction_render_ms=float(np.mean(timings)*1000),
                 p95_prediction_render_ms=float(np.percentile(timings,95)*1000),
                 throughput_fps=len(timings)/sum(timings),cold_start_ms=detector.cold_start_ms,
                 dimensions='dataset 416x416; content960/input992',
                 scope='offline Detector.predict including drawing; not camera/window end-to-end FPS',
                 preprocessing=detector.preprocessing,validated=detector.validated)
    write_json(OUT/'runtime-verification.json',runtime)
    config=ROOT/'configs/preview-improved.json'
    write_json(OUT/'previous-preview-config.json',read_json(config))
    preview_args=['detect.py','--weights',str(WEIGHTS),'--thresholds',chosen['thresholds'],
                  '--allow-unvalidated','--nms-iou','0.3']
    if chosen['tta']:preview_args+=['--tta',chosen['tta']]
    write_json(config,dict(created_at=now(),selection_split='val',test_used=False,
          formal_acceptance_passed=False,selected=chosen,preview_args=preview_args,
          selection_manifest=str(OUT/'selection-freeze.json'),test_report=test['report'],
          previous_configuration=str(OUT/'previous-preview-config.json')))
    print('DONE',chosen['name'],'val',chosen['metrics'],'test',test['metrics'],'runtime',runtime,flush=True)

if __name__=='__main__':main()
