import numpy as np
import pytest
from multi_view import flip_image, restore_flip, fuse_views, VIEWS


def test_flip_restoration_is_involution_on_rectangular_image():
    img=np.arange(3*5*3,dtype=np.uint8).reshape(3,5,3)
    predictions=[[0,.1,.2,.4,.8,.7]]
    for view in VIEWS:
        np.testing.assert_array_equal(flip_image(flip_image(img,view),view),img)
        np.testing.assert_allclose(restore_flip(restore_flip(predictions,view),view),predictions)


def test_fusion_uses_one_vote_per_view_and_preserves_isolated_targets():
    p=[0,.1,.2,.3,.8,.8]
    q=[0,.7,.1,.9,.8,.8]
    predictions=fuse_views([[p,p,q],[p,q]])
    assert len(predictions)==2
    assert predictions[0][5]==pytest.approx(.8)
    assert predictions[1][5]==pytest.approx(.8)


def test_missing_view_penalty_does_not_turn_repeated_same_view_into_support():
    p=[0,.1,.2,.3,.8,.8]
    result=fuse_views([[p,p],[]])
    assert len(result)==1
    assert result[0][5]==pytest.approx(.6)


def test_live_flip_prediction_restores_rectangular_coordinates_and_threshold_boundary(monkeypatch):
    import multi_view
    from preprocessing import predict_frame
    canonical=[[0,.1,.2,.4,.8,.4]]
    views=iter(VIEWS)
    def backend(model,image,imgsz,device,**kwargs):
        assert kwargs['conf']==.001
        return restore_flip(canonical,next(views))
    monkeypatch.setattr(multi_view,'predict_frame',backend)
    result=predict_frame(object(),np.zeros((20,40,3),np.uint8),conf=.4,nms_iou=.3,tta='flip4')
    np.testing.assert_allclose(result,[[0,4,4,16,16,.4]])


def test_flip_protocol_rejects_single_view_thresholds():
    from preprocessing import preprocessing_spec, check_threshold_preprocessing
    with pytest.raises(ValueError):
        preprocessing_spec(960,None,'flip4')
    with pytest.raises(ValueError,match='预处理版本'):
        check_threshold_preprocessing({'preprocessing':preprocessing_spec(960,.3)},960,.3,'flip4')


def test_reference_threshold_is_rejected_even_in_unvalidated_preview(tmp_path,monkeypatch):
    import ultralytics,inference
    from types import SimpleNamespace
    from common import NAMES,digest,write_json
    from preprocessing import preprocessing_spec
    p=tmp_path/'best.pt'
    p.write_bytes(b'fixture')
    monkeypatch.setattr(ultralytics,'YOLO',lambda _:SimpleNamespace(names=NAMES))
    monkeypatch.setattr(inference,'predict_frame',lambda *a,**kw:[])
    write_json(p.with_suffix('.release.json'),dict(status='evaluation_failed',weights_sha256=digest(p),
               provenance_sha256='train-v1'))
    threshold=p.with_suffix('.reference.json')
    t=dict(weights_sha256=digest(p),imgsz=960,thresholds={'0':.3},reference_evaluation=True,
           training_provenance_sha256='train-v1',evaluation_provenance_sha256='eval-v2',
           provenance_sha256='eval-v2',preprocessing=preprocessing_spec(960,.3,'flip4'))
    write_json(threshold,t)
    kwargs=dict(allow_unvalidated=True,nms_iou=.3,thresholds=threshold,tta='flip4')
    with pytest.raises(ValueError,match='参考阈值'):
        inference.Detector(p,**kwargs)
