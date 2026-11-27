import sys
from pathlib import Path
import cv2,numpy as np
sys.path.insert(0,str(Path.cwd()))
from common import ROOT,read_json,write_json,digest,now
from metrics import class_metrics,calibration_summary,threshold_curve
from preprocessing import predict_frame
from ultralytics import YOLO

REPORT=ROOT/'reports/data-expansion-20261029/recovery'
def point(samples,t):
    bg=[s for s in samples if not s['gt']];pos=[s for s in samples if s['gt']]
    return dict(threshold=t,mixed=class_metrics(samples,{0:t})['per_class']['0'],positive_only=class_metrics(pos,{0:t})['per_class']['0'],
        background_images=len(bg),background_false_alarm_images=sum(any(p[5]>=t for p in s['pred']) for s in bg),background_false_boxes=sum(sum(p[5]>=t for p in s['pred']) for s in bg))
def main():
    output={};source=ROOT/'runs/concrete-crack-connected-v2-head-20261027/weights/best_deployment.pt'
    candidate=ROOT/'runs/concrete-crack-road-recovery-20261029/weights/best_deployment.pt'
    for name,folder,weights in [('baseline','baseline-val',source),('candidate','candidate-val',candidate)]:
        samples=read_json((REPORT.parent if name=='baseline' else REPORT)/folder/'predictions.json');r=read_json((REPORT.parent if name=='baseline' else REPORT)/folder/'report.json');summary=r['calibration']
        output[name]=dict(weights=str(weights),weights_sha256=digest(weights),selected_epoch=read_json(weights.with_suffix('.release.json'),{}).get('selected_epoch'),
            at_selected=point(samples,float(r['thresholds']['0'])),average_precision=r['average_precision'],joint_target_feasible=summary['joint_target_feasible'])
        if name=='baseline':output[name]['at_existing_deployment']=point(samples,.03675317019224167)
        p=summary.get('precision_at_recall90');output[name]['at_recall90']=point(samples,p['threshold']) if p else None
    cap=cv2.VideoCapture(str(ROOT/'outputs/road-video-prepared-20261027/实际路面视频_检测适配版.mp4'))
    previous=read_json(ROOT/'reports/real-video-diagnosis-20261029/model-video-comparison.json')
    indices=[s['frame'] for s in previous['results'][0]['samples']];frames=[]
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
        if not ok:raise ValueError(f'Unreadable target frame {index}')
        frames.append(frame)
    cap.release();sheets={}
    for name,weights in [('baseline',source),('candidate',candidate)]:
        model=YOLO(str(weights));t=output[name]['at_existing_deployment']['threshold'] if name=='baseline' else output[name]['at_selected']['threshold'];rows=[];vis=[]
        for index,frame in zip(indices,frames):
            pred=predict_frame(model,frame,960,'0',conf=.001,normalized=True,nms_iou=.3)
            large=[p for p in pred if (p[3]-p[1])*(p[4]-p[2])>=.8]
            rows.append(dict(frame=index,predictions=pred,large_box_max_conf=max((p[5] for p in large),default=0),alarm=any(p[5]>=t for p in pred),large_at_deployment=any(p[5]>=t for p in large)))
            im=frame.copy();h,w=im.shape[:2]
            for p in pred:
                if p[5]<t:continue
                cv2.rectangle(im,(int(p[1]*w),int(p[2]*h)),(int(p[3]*w),int(p[4]*h)),(0,0,255),2)
                cv2.putText(im,f'{p[5]:.2f}',(max(0,int(p[1]*w)),max(16,int(p[2]*h))),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
            im=cv2.resize(im,(256,256));im=cv2.copyMakeBorder(im,24,0,0,0,cv2.BORDER_CONSTANT,value=(30,30,30));cv2.putText(im,f'{name} frame {index}',(3,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);vis.append(im)
        sheet=np.concatenate([np.concatenate(vis[i:i+6],axis=1) for i in range(0,24,6)],axis=0)
        cv2.imencode('.jpg',sheet)[1].tofile(str(REPORT/f'video-{name}.jpg'))
        output[name]['video']=dict(frames=24,threshold=t,alarm_frames=sum(r['alarm'] for r in rows),large_at_deployment=sum(r['large_at_deployment'] for r in rows),large_conf_ge_05=sum(r['large_box_max_conf']>=.5 for r in rows),samples=rows)
        del model
    old=output['baseline']['at_existing_deployment'];new=output['candidate']['at_selected']
    gates=dict(mixed_precision80=new['mixed']['precision']>=.8,mixed_recall90=new['mixed']['recall']>=.9,
        positive_recall90=new['positive_only']['recall']>=.9,background_false_boxes_reduced=new['background_false_boxes']<old['background_false_boxes'])
    output.update(at=now(),promotion_gates=gates,promotion_eligible=all(gates.values()),formal_acceptance=False,video_note='Unlabelled development diagnosis; large predicted area is not automatically a false positive, and reduced detections alone do not establish recall or precision.')
    write_json(REPORT/'comparison.json',output)
    print({k:{'val':v['at_selected'],'video':{kk:vv for kk,vv in v['video'].items() if kk!='samples'}} for k,v in output.items() if k in ['baseline','candidate']});print(gates)
if __name__=='__main__':main()
