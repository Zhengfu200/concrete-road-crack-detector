"""Validation-development candidates: flip views restored before prediction fusion."""
from __future__ import annotations

import numpy as np
from metrics import iou
from preprocessing import EVALUATION_CONF, predict_frame, suppress_duplicates

VIEWS = ('identity', 'horizontal', 'vertical', 'both')


def flip_image(image, view):
    if view not in VIEWS:
        raise ValueError('Unknown flip view')
    if view in ('vertical', 'both'):
        image = image[::-1]
    if view in ('horizontal', 'both'):
        image = image[:, ::-1]
    return np.ascontiguousarray(image)


def restore_flip(predictions, view):
    """Coordinates are normalized xyxy; flipping is its own inverse."""
    if view not in VIEWS:
        raise ValueError('Unknown flip view')
    result=[]
    for c, x1, y1, x2, y2, score in predictions:
        if view in ('horizontal', 'both'):
            x1, x2 = 1-x2, 1-x1
        if view in ('vertical', 'both'):
            y1, y2 = 1-y2, 1-y1
        result.append([c,x1,y1,x2,y2,score])
    return result


def fuse_views(view_predictions, overlap=.55, missing_penalty=.5):
    """Score-weighted boxes, at most one vote per view per cluster. No labels are read."""
    if not view_predictions or not 0 < overlap < 1 or not 0 <= missing_penalty <= 1:
        raise ValueError('Invalid fusion configuration')
    events=sorted([(p,view) for view,preds in enumerate(view_predictions) for p in preds],
                  key=lambda event:event[0][5],reverse=True)
    clusters=[]
    for prediction,view in events:
        choices=[(iou(prediction[1:5],cluster['box']),i) for i,cluster in enumerate(clusters)
                 if prediction[0]==cluster['class'] and view not in cluster['views']]
        best,index=max(choices,default=(0,-1))
        if best >= overlap:
            cluster=clusters[index]
            cluster['members'].append(prediction)
            cluster['views'].add(view)
            weights=np.array([p[5] for p in cluster['members']])
            cluster['box']=np.average([p[1:5] for p in cluster['members']],axis=0,weights=weights).tolist()
        else:
            clusters.append(dict(box=prediction[1:5],members=[prediction],views={view},
                                 **{'class':prediction[0]}))
    result=[]
    for cluster in clusters:
        support=len(cluster['views'])/len(view_predictions)
        score=np.mean([p[5] for p in cluster['members']])*(1-missing_penalty+missing_penalty*support)
        result.append([cluster['class'],*cluster['box'],float(score)])
    return suppress_duplicates(result,.3)


def collect_flip_views(model,image,imgsz=960,device='0'):
    return {view:restore_flip(predict_frame(model,flip_image(image,view),imgsz,device,
                          conf=EVALUATION_CONF,normalized=True,nms_iou=.3),view) for view in VIEWS}


def predict_flip_ensemble(model,image,imgsz,device,conf,normalized,nms_iou):
    # Each view uses the low evaluation floor BEFORE the final NMS. Filtering a view
    # at the working threshold earlier could change which box suppresses another.
    groups=[restore_flip(predict_frame(model,flip_image(image,view),imgsz,device,
             conf=min(conf,EVALUATION_CONF),normalized=True,nms_iou=nms_iou),view) for view in VIEWS]
    combined=suppress_duplicates([p for group in groups for p in group],nms_iou)
    result=[p for p in combined if p[5]>=conf]
    if not normalized:
        h,w=image.shape[:2]
        result=[[p[0],p[1]*w,p[2]*h,p[3]*w,p[4]*h,p[5]] for p in result]
    return result
