import pytest

from metrics import calibrate, calibration_summary, class_metrics, confusion, threshold_curve


def test_duplicate_predictions_are_false_positives():
    rows = [{"gt": [[0,0,0,1,1]], "pred": [[0,0,0,1,1,.9],[0,0,0,1,1,.8]]}]
    metrics = class_metrics(rows, {0:.2})
    assert metrics["per_class"]["0"]["precision"] == .5
    assert metrics["per_class"]["0"]["recall"] == 1


def test_background_false_positive_counted():
    rows = [{"gt": [], "pred": [[0,0,0,1,1,.9]]}]
    metrics = class_metrics(rows, {0:.2})
    assert metrics["per_class"]["0"]["fp"] == 1
    assert metrics["scenarios"] == {"crack":0,"background":1}


def test_missed_crack_and_single_class_schema():
    row = {"gt": [[0,0,0,.4,.4]], "pred": []}
    metrics = class_metrics([row], {0:.2})
    assert metrics["scenarios"]["crack"] == 1
    assert list(metrics["per_class"]) == ["0"]
    assert metrics["per_class"]["0"]["fn"] == 1


def test_confusion_has_crack_and_background_only():
    matrix = confusion([{"gt": [[0,0,0,1,1]], "pred": []},
                        {"gt": [], "pred": [[0,0,0,1,1,.9]]}], {0:.2})
    assert matrix == [[0,1],[1,0]]


def test_threshold_calibration_prioritizes_recall():
    samples = [{"gt": [[0,0,0,1,1]], "pred": [[0,0,0,1,1,score]]} for score in [.15,.3,.9]]
    samples += [{"gt": [], "pred": [[0,0,0,1,1,.6]]}]
    thresholds, _ = calibrate(samples)
    assert set(thresholds) == {0}
    assert thresholds[0] <= .15
    assert class_metrics(samples,thresholds)["per_class"]["0"]["recall"] == 1


def test_unknown_class_is_rejected():
    with pytest.raises(ValueError,match="单类别"):
        class_metrics([{"gt":[[1,0,0,1,1]],"pred":[]}],{0:.2})


def test_exact_curve_matches_operating_metrics_including_tied_scores_and_duplicates():
    samples = [
        {'gt': [[0, 0, 0, .4, .4], [0, .6, .6, 1, 1]],
         'pred': [[0, 0, 0, .4, .4, .83], [0, 0, 0, .4, .4, .83],
                  [0, .6, .6, 1, 1, .371], [0, .2, .2, .5, .5, .15]]},
        {'gt': [], 'pred': [[0, 0, 0, .4, .4, .83]]},
        {'gt': [[0, 0, 0, 1, 1]], 'pred': []}]
    curve = threshold_curve(samples)
    assert {row['threshold'] for row in curve} == {.83, .371, .15}
    for row in curve:
        assert row['metrics'] == class_metrics(samples, {0: row['threshold']})['per_class']


def test_no_feasible_threshold_is_explicit_and_preview_uses_f1():
    samples = [{'gt': [[0, 0, 0, 1, 1]], 'pred': [[0, 0, 0, 1, 1, .9]]},
               {'gt': [[0, 0, 0, 1, 1]], 'pred': []},
               {'gt': [], 'pred': [[0, 0, 0, 1, 1, .1]]}]
    thresholds, curve = calibrate(samples)
    summary = calibration_summary(curve)
    assert thresholds == {0: .9}
    assert summary['joint_target_feasible'] is False
    assert summary['precision_at_recall90'] is None
    assert summary['selection_policy'] == 'preview_max_f1_no_feasible_threshold'


def test_joint_target_threshold_is_not_lost_between_coarse_grid_points():
    samples = [{'gt': [[0, 0, 0, 1, 1]], 'pred': [[0, 0, 0, 1, 1, .371]]}
               for _ in range(10)]
    samples += [{'gt': [], 'pred': [[0, 0, 0, 1, 1, .369]]} for _ in range(5)]
    thresholds, curve = calibrate(samples)
    assert thresholds[0] == .371
    assert calibration_summary(curve)['joint_target_feasible'] is True


def test_empty_predictions_have_explicit_nonfeasible_calibration():
    thresholds, curve = calibrate([{'gt': [[0, 0, 0, 1, 1]], 'pred': []}])
    assert thresholds == {0: .2}
    assert calibration_summary(curve)['joint_target_feasible'] is False
