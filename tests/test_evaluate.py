import pytest

from evaluate import average_precision


def context_args(tmp_path, **kwargs):
    from types import SimpleNamespace
    return SimpleNamespace(reference_evaluation=True, split='val', calibrate=True,
        output=str(tmp_path/'reports/reference'), thresholds=str(tmp_path/'reports/reference/val.thresholds.json'), **kwargs)


def test_cross_version_reference_is_val_only_with_separate_outputs(tmp_path):
    from evaluate import evaluation_context
    args = context_args(tmp_path)
    weights = tmp_path/'weights/best.pt'
    assert evaluation_context(args, weights, 'old', 'new')[0]
    args.split = 'test'
    with pytest.raises(ValueError, match='仅允许验证集'):
        evaluation_context(args, weights, 'old', 'new')
    args.split = 'val';args.reference_evaluation = False
    with pytest.raises(ValueError, match='不一致'):
        evaluation_context(args, weights, 'old', 'new')


def test_reference_never_overwrites_weight_thresholds_or_prior_report(tmp_path):
    from evaluate import evaluation_context
    args = context_args(tmp_path)
    weights = tmp_path/'weights/best.pt'
    args.thresholds = str(weights.with_suffix('.thresholds.json'))
    with pytest.raises(ValueError, match='独立输出目录'):
        evaluation_context(args, weights, 'old', 'new')
    args = context_args(tmp_path)
    output = __import__('pathlib').Path(args.output);output.mkdir(parents=True)
    (output/'report.json').write_text('old result')
    with pytest.raises(ValueError, match='已存在'):
        evaluation_context(args, weights, 'old', 'new')


def test_reference_run_keeps_release_and_deployment_thresholds_unchanged(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import evaluate
    import ultralytics
    from common import digest, read_json, write_json
    weights = tmp_path/'weights/best.pt';weights.parent.mkdir()
    weights.write_bytes(b'test model')
    release = weights.with_suffix('.release.json')
    deploy_threshold = weights.with_suffix('.thresholds.json')
    write_json(release, {'weights_sha256':digest(weights),'provenance_sha256':'v1','status':'evaluation_failed'})
    write_json(deploy_threshold, {'historical':'preserve'})
    protected = {p:digest(p) for p in [weights,release,deploy_threshold]}
    fake = SimpleNamespace(names={0:'crack'})
    monkeypatch.setattr(ultralytics, 'YOLO', lambda *a: fake)
    monkeypatch.setattr(evaluate, 'dataset_readiness', lambda *a: {'ready':True,'domain_reviewed':False,
                        'provenance_sha256':'v2','gaps':[]})
    monkeypatch.setattr(evaluate, 'collect', lambda *a: [{'id':'target','gt':[[0,.1,.1,.5,.5]],
                        'pred':[[0,.1,.1,.5,.5,.9]]}])
    def ap_stub(samples, output):
        output.mkdir(parents=True, exist_ok=True)
        return {'0':{'map50':.99,'map50_95':.99}}
    monkeypatch.setattr(evaluate, 'average_precision', ap_stub)
    args = context_args(tmp_path)
    args.weights=str(weights);args.data=str(tmp_path/'newdata');args.imgsz=960
    args.device='cpu';args.smoke=False;args.nms_iou=.3
    evaluate.run(args)
    report = read_json(__import__('pathlib').Path(args.output)/'report.json')
    assert report['reference_evaluation'] and not report['accepted']
    assert report['training_provenance_sha256']=='v1' and report['evaluation_provenance_sha256']=='v2'
    assert all(digest(p)==sha for p,sha in protected.items())


def test_shared_ap_counts_missing_targets_and_background_predictions(tmp_path):
    # Perfect boxes at all 10 IoUs. A high-confidence background FP lowers AP.
    good = [{'id': 'one', 'gt': [[0, .1, .1, .4, .4]],
             'pred': [[0, .1, .1, .4, .4, .8]]}]
    perfect = average_precision(good, tmp_path/'perfect')['0']
    bad = average_precision([
        {'id': 'background', 'gt': [], 'pred': [[0, .2, .2, .5, .5, .95]]},
        *good, {'id': 'missed', 'gt': [[0, .1, .1, .4, .4]], 'pred': []}
    ], tmp_path/'bad')['0']
    assert perfect['map50'] > .99
    assert perfect['map50_95'] == pytest.approx(perfect['map50'])
    assert bad['map50'] < .5
    assert (tmp_path/'bad/BoxPR_curve.png').exists()


def test_shared_ap_handles_zero_predictions(tmp_path):
    ap = average_precision([{'id': 'miss', 'gt': [[0, .1, .1, .4, .4]], 'pred': []}], tmp_path)
    assert ap['0']['map50'] == 0 and ap['0']['map50_95'] == 0
