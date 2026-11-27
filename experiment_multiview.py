"""Predeclared flip-ensemble screening, validation only; does not publish/accept a model."""
from pathlib import Path
import time
from common import ROOT, digest, now, read_image, read_json, write_json
from metrics import calibration_summary, threshold_curve, xywh_to_xyxy
from multi_view import VIEWS, collect_flip_views, fuse_views
from preprocessing import suppress_duplicates

def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-cache',action='store_true')
    args=parser.parse_args()
    out=ROOT/'reports/optimization-20261027/multiview'
    out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/prepared/crack-connected-v2'
    weights=ROOT/'runs/concrete-crack-20261021-212347/weights/best.pt'
    candidates={
        'baseline': ('identity',),
        'horizontal_nms': ('identity','horizontal'),
        'four_nms': VIEWS,
        'horizontal_fusion': ('identity','horizontal'),
        'four_fusion': VIEWS,
    }
    plan=dict(at=now(),selection_split='val',test_used=False,candidates=candidates,
              fusion_iou=.55,fusion_missing_penalty=.5,nms_iou=.3,imgsz=960,
              weights_sha256=digest(weights),data_sha256=digest(data/'provenance.json'),
              primary='recall_at_precision80',secondary='maximum_F1',
              acceptance='development only; no independent road domain/background data available')
    if args.reuse_cache:
        old=read_json(out/'plan.json')
        assert old['weights_sha256']==plan['weights_sha256'] and old['data_sha256']==plan['data_sha256']
        cache=read_json(out/'views.json')
    else:
        if (out/'views.json').exists():
            raise ValueError('Use --reuse-cache to keep recorded predictions')
        write_json(out/'plan.json',plan)
        from ultralytics import YOLO
        model=YOLO(str(weights))
        cache=[]
        for index,row in enumerate(read_json(data/'provenance.json')['splits']['val'],1):
            image=data/'images/val'/f"{row['id']}.jpg"
            start=time.perf_counter()
            views=collect_flip_views(model,read_image(image))
            cache.append(dict(id=row['id'],image=str(image),gt=[xywh_to_xyxy(b) for b in row['boxes']],
                              views=views,seconds=time.perf_counter()-start))
            if index%20==0: print(f'views {index}/164',flush=True)
        write_json(out/'views.json',cache)
    results={}
    for name,views in candidates.items():
        samples=[]
        for row in cache:
            groups=[row['views'][v] for v in views]
            pred=fuse_views(groups) if name.endswith('fusion') else suppress_duplicates([p for g in groups for p in g],.3)
            samples.append(dict(id=row['id'],image=row['image'],gt=row['gt'],pred=pred))
        summary=calibration_summary(threshold_curve(samples))
        write_json(out/f'{name}.predictions.json',samples)
        results[name]=summary
        point=summary['recall_at_precision80']
        print(name,'joint=',summary['joint_target_feasible'],'R@P80=',point['metrics']['0'] if point else None,flush=True)
    write_json(out/'screening.json',dict(plan=plan,results=results))

if __name__=='__main__': main()
