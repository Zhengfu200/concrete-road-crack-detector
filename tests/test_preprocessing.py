from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from common import save_image, write_json
from evaluate import collect
from inference import Detector
from preprocessing import (check_threshold_preprocessing, predict_frame,
                           prepare_frame, preprocessing_spec, suppress_duplicates)


@pytest.mark.parametrize("shape", [(416, 416), (720, 1280), (1280, 720), (233, 617), (1, 100)])
def test_image_geometry_preserves_ratio_and_restores_original_boxes(shape):
    h, w = shape
    frame = np.full((h, w, 3), 23, np.uint8)
    canvas, geometry = prepare_frame(frame)
    assert canvas.shape == (992, 992, 3)
    assert abs(geometry.resized_height / geometry.resized_width - h / w) <= 1 / geometry.resized_width
    assert np.all(canvas[:16] == 114) and np.all(canvas[-16:] == 114)
    assert np.all(canvas[:, :16] == 114) and np.all(canvas[:, -16:] == 114)
    expected = np.array([[.1*w, .2*h, .8*w, .9*h], [0, 0, w, h]])
    projected = expected.copy()
    projected[:, [0, 2]] = projected[:, [0, 2]] * geometry.resized_width / w + geometry.left
    projected[:, [1, 3]] = projected[:, [1, 3]] * geometry.resized_height / h + geometry.top
    np.testing.assert_allclose(geometry.restore(projected), expected, atol=1e-10)


def test_square_input_reproduces_diagnosed_resize_and_border_exactly():
    frame = np.random.default_rng(42).integers(0, 256, (416, 416, 3), dtype=np.uint8)
    canvas, geometry = prepare_frame(frame)
    expected = cv2.copyMakeBorder(cv2.resize(frame, (960, 960)), 16, 16, 16, 16,
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114))
    np.testing.assert_array_equal(canvas, expected)
    assert (geometry.top, geometry.left) == (16, 16)


class FakeModel:
    def __init__(self):
        self.inputs = []

    def predict(self, frame, **kwargs):
        self.inputs.append((frame.copy(), kwargs))
        # A real image box, a partially clipped box, and a padding-only box.
        raw = np.array([[112, 322, 784, 658, .8, 0], [-10, -10, 1008, 1008, .6, 0],
                        [0, 0, 8, 8, .9, 0]], dtype=np.float32)
        return [SimpleNamespace(boxes=SimpleNamespace(data=SimpleNamespace(
            cpu=lambda: SimpleNamespace(numpy=lambda: raw))))]


def test_padding_only_boxes_removed_and_clipped_to_original_image():
    frame = np.zeros((720, 1280, 3), np.uint8)
    result = predict_frame(FakeModel(), frame)
    assert len(result) == 2
    assert result[0][1:5] == pytest.approx([128, 128, 1024, 576])
    assert result[1][1:5] == pytest.approx([0, 0, 1280, 720])
    assert predict_frame(FakeModel(), frame, normalized=True)[1][1:5] == [0, 0, 1, 1]


def test_evaluation_and_video_entrypoints_use_identical_pixels_options_and_boxes(tmp_path):
    frame = np.full((720, 1280, 3), 23, np.uint8)
    image = tmp_path / 'images/val/frame.png'
    save_image(image, frame)
    write_json(tmp_path/'provenance.json', {"splits": {"val": [
        {"id": "frame", "image": str(image), "boxes": []}]}})
    model = FakeModel()
    samples = collect(model, tmp_path, 'val', 960, 'cpu')
    detector = Detector.__new__(Detector)
    detector.model, detector.imgsz, detector.device = model, 960, 'cpu'
    detector.thresholds, detector.validated = {0: .001}, False
    rendered, counts, video_predictions = detector.predict(frame)
    np.testing.assert_array_equal(model.inputs[0][0], model.inputs[1][0])
    assert model.inputs[0][1] == model.inputs[1][1]
    assert model.inputs[0][1]['imgsz'] == 992
    normalized = np.array(video_predictions)
    normalized[:, [1, 3]] /= 1280
    normalized[:, [2, 4]] /= 720
    np.testing.assert_allclose(samples[0]['pred'], normalized, atol=1e-12)
    assert counts == {0: 2} and rendered.shape == frame.shape


@pytest.mark.parametrize('imgsz', [0, -32, 961, 960., True])
def test_invalid_content_sizes_rejected(imgsz):
    with pytest.raises(ValueError, match='32倍数'):
        preprocessing_spec(imgsz)


@pytest.mark.parametrize('frame', [np.zeros((0, 4, 3), np.uint8), np.zeros((4, 4), np.uint8),
                                   np.zeros((4, 4, 3), np.float32)])
def test_invalid_frames_rejected(frame):
    with pytest.raises(ValueError, match='BGR'):
        prepare_frame(frame)


def test_old_or_different_preprocessing_thresholds_rejected():
    with pytest.raises(ValueError, match='重新校准'):
        check_threshold_preprocessing({'thresholds': {0: .01}}, 960)
    with pytest.raises(ValueError, match='重新校准'):
        check_threshold_preprocessing({'preprocessing': preprocessing_spec(640)}, 960)
    check_threshold_preprocessing({'preprocessing': preprocessing_spec(960)}, 960)


@pytest.mark.parametrize('status', ['evaluation_failed', 'accepted'])
def test_detector_does_not_reuse_legacy_threshold_or_acceptance(tmp_path, monkeypatch, status):
    import ultralytics
    from common import NAMES, digest
    weights = tmp_path/'best.pt'
    weights.write_bytes(b'fake model for release checks')
    model = FakeModel()
    model.names = NAMES
    monkeypatch.setattr(ultralytics, 'YOLO', lambda path: model)
    write_json(weights.with_suffix('.release.json'), {
        'status': status, 'weights_sha256': digest(weights), 'provenance_sha256': 'fixture'})
    write_json(weights.with_suffix('.thresholds.json'), {
        'weights_sha256': digest(weights), 'provenance_sha256': 'fixture',
        'imgsz': 960, 'thresholds': {'0': .01}})
    if status == 'accepted':
        with pytest.raises(ValueError, match='验收预处理版本'):
            Detector(weights)
    else:
        with pytest.warns(UserWarning, match='旧阈值'):
            detector = Detector(weights, allow_unvalidated=True)
        assert detector.thresholds == {0: .2} and detector.validated is False


def test_detector_loads_thresholds_bound_to_current_preprocessing(tmp_path, monkeypatch):
    import ultralytics
    from common import NAMES, digest
    weights = tmp_path/'best.pt'
    weights.write_bytes(b'fake model for release checks')
    model = FakeModel()
    model.names = NAMES
    monkeypatch.setattr(ultralytics, 'YOLO', lambda path: model)
    spec = preprocessing_spec(960)
    write_json(weights.with_suffix('.release.json'), {
        'status': 'accepted', 'weights_sha256': digest(weights),
        'provenance_sha256': 'fixture', 'evaluated_preprocessing': spec})
    write_json(weights.with_suffix('.thresholds.json'), {
        'weights_sha256': digest(weights), 'provenance_sha256': 'fixture',
        'imgsz': 960, 'thresholds': {'0': .35}, 'preprocessing': spec})
    detector = Detector(weights)
    assert detector.thresholds == {0: .35} and detector.validated is True


def test_optional_nms_keeps_distinct_cracks_and_highest_score():
    predictions = [[0, 0, 0, .4, .4, .5], [0, 0, 0, .4, .4, .9],
                   [0, .6, .6, 1, 1, .8]]
    assert suppress_duplicates(predictions) is predictions
    assert suppress_duplicates(predictions, .3) == [predictions[1], predictions[2]]
    with pytest.raises(ValueError, match='预处理版本'):
        check_threshold_preprocessing({'preprocessing': preprocessing_spec(960)}, 960, .3)
    check_threshold_preprocessing({'preprocessing': preprocessing_spec(960, .3)}, 960, .3)


@pytest.mark.parametrize('overlap', [0, 1, -1, float('nan'), True])
def test_bad_suppression_options_rejected(overlap):
    with pytest.raises(ValueError, match='IoU'):
        preprocessing_spec(960, overlap)


def test_exact_calibrated_score_is_included_despite_backend_strict_comparison():
    class StrictBackend:
        def predict(self, frame, **kwargs):
            raw = np.array([[16, 16, 976, 976, .8, 0], [16, 16, 976, 976, .6, 0]], dtype=np.float32)
            raw = raw[raw[:, 4] > np.float32(kwargs['conf'])]
            return [SimpleNamespace(boxes=SimpleNamespace(data=SimpleNamespace(
                cpu=lambda: SimpleNamespace(numpy=lambda: raw))))]
    threshold = float(np.float32(.8))
    predictions = predict_frame(StrictBackend(), np.zeros((416, 416, 3), np.uint8), conf=threshold)
    assert len(predictions) == 1 and predictions[0][5] == threshold
