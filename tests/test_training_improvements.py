import numpy as np
import torch

from preprocessing import prepare_frame
from training_improvements import pad_training_batch


def test_training_canvas_and_labels_agree_with_shared_inference_geometry():
    original = np.full((960, 960, 3), 23, dtype=np.uint8)
    canvas, geometry = prepare_frame(original)
    batch = {'img': torch.from_numpy(original.transpose(2, 0, 1)).unsqueeze(0).float()/255,
             'bboxes': torch.tensor([[.5, .4, .8, .2], [.5, .5, 1, 1]], dtype=torch.float32)}
    before = batch['bboxes'].clone()
    result = pad_training_batch(batch)
    np.testing.assert_allclose(result['img'][0].permute(1, 2, 0).numpy(), canvas/255, atol=1e-7)
    boxes = result['bboxes'].numpy()
    xyxy = np.column_stack([boxes[:, 0]-boxes[:, 2]/2, boxes[:, 1]-boxes[:, 3]/2,
                            boxes[:, 0]+boxes[:, 2]/2, boxes[:, 1]+boxes[:, 3]/2])*992
    restored = geometry.restore(xyxy)/960
    expected = before.numpy()
    expected = np.column_stack([expected[:, 0]-expected[:, 2]/2, expected[:, 1]-expected[:, 3]/2,
                                expected[:, 0]+expected[:, 2]/2, expected[:, 1]+expected[:, 3]/2])
    np.testing.assert_allclose(restored, expected, atol=1e-7)


def test_background_batch_with_no_boxes_is_valid():
    batch = {'img': torch.zeros(1, 3, 320, 320), 'bboxes': torch.empty(0, 4)}
    padded = pad_training_batch(batch)
    assert padded['img'].shape == (1, 3, 352, 352)
    assert padded['bboxes'].shape == (0, 4)


def test_final_validation_does_not_create_a_fake_extra_training_epoch(tmp_path):
    from types import SimpleNamespace
    from training_improvements import DeploymentSelection
    monitor=DeploymentSelection(tmp_path,960,'cpu',{})
    monitor(SimpleNamespace(validator=SimpleNamespace(training=False)))
    assert monitor.rows==[] and monitor.seen==set()


def test_resume_preserves_previous_best_and_rejects_changed_protocol(tmp_path):
    import pytest
    from common import digest, write_json
    from preprocessing import preprocessing_spec
    from training_improvements import DeploymentSelection
    chosen = tmp_path/'weights/best_deployment.pt'
    chosen.parent.mkdir()
    chosen.write_bytes(b'checkpoint')
    write_json(chosen.with_suffix('.release.json'), {'provenance_sha256':'data-v2', 'weights_sha256':digest(chosen)})
    row = {'epoch':1, 'calibration':{'joint_target_feasible':False,
           'recall_at_precision80':{'metrics':{'0':{'recall':.85}}},
           'selected':{'metrics':{'0':{'f1':.86}}}}}
    progress = {'epochs':[row], 'preprocessing':preprocessing_spec(960,.3), 'selection_split':'val'}
    write_json(tmp_path/'deployment-validation/progress.json', progress)
    monitor = DeploymentSelection(tmp_path,960,'cpu',{'provenance_sha256':'data-v2'},.3)
    monitor.restore(tmp_path,1)
    assert monitor.seen=={1} and monitor.rows==[row] and monitor.best_key==(False,.85,.86)
    with pytest.raises(ValueError,match='协议不一致'):
        DeploymentSelection(tmp_path,640,'cpu',{'provenance_sha256':'data-v2'},.3).restore(tmp_path,1)
    with pytest.raises(ValueError,match='检查点轮次'):
        monitor.restore(tmp_path,0)
    chosen.write_bytes(b'changed')
    with pytest.raises(ValueError,match='摘要不一致'):
        monitor.restore(tmp_path,1)
