"""Bounded V2 warm-start experiment; compare against unchanged weights on the same V2 val."""
from argparse import Namespace
from torch.nn.modules.batchnorm import _BatchNorm
from common import ROOT, digest, read_json, now, write_json
from connected_annotations import WORK, frozen
from training_improvements import AlignedDetectionTrainer
from train import train


class ConnectedTrialTrainer(AlignedDetectionTrainer):
    def _model_train(self):
        super()._model_train()
        for module in self.model.modules():
            if isinstance(module, _BatchNorm):
                module.eval()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='恢复本次实验最后保存的轮次')
    cli = parser.parse_args()
    directory = ROOT/'data/prepared/crack-connected-v2'
    version = read_json(directory/'version.json')
    assert frozen(WORK) and version['source_files_verified_unchanged']
    assert version['missing'] == []
    report = ROOT/'reports/connected-v2-full-review-20261025'
    baseline = read_json(ROOT/'reports/connected-v2-original-baseline-20261025/report.json')
    assert baseline['evaluation_provenance_sha256'] == digest(directory/'provenance.json')
    spec = dict(created_at=now(), name='concrete-crack-connected-v2-trial-20261025',
                comparator='unchanged original checkpoint evaluated on identical V2 val images',
                design='baseline versus warm-start; not a label-only causal ablation',
                epochs=3, seed=42, selection_split='val', test_used_for_model_selection=False,
                primary_metric='recall at precision >= 0.8', selection_nms_iou=0.3,
                training_bn_running_statistics='frozen', candidate_status='unvalidated',
                data_provenance_sha256=digest(directory/'provenance.json'),
                config_sha256=digest(ROOT/'configs/connected-v2-trial.yaml'))
    if cli.resume:
        previous = read_json(report/'trial-plan.json')
        assert previous['config_sha256'] == spec['config_sha256']
        assert previous['data_provenance_sha256'] == spec['data_provenance_sha256']
        write_json(report/'trial-resume.json', dict(resumed_at=now(), reason='user turn interrupted process during epoch 2; resume saved epoch 1',
                   checkpoint=str(ROOT/'runs'/spec['name']/'weights/last.pt')))
    else:
        write_json(report/'trial-plan.json', spec)
    args = Namespace(data=str(directory), model='runs/concrete-crack-20261021-212347/weights/best.pt',
                     config='configs/connected-v2-trial.yaml', device='0', name=spec['name'],
                     smoke=False, resume=str(ROOT/'runs'/spec['name']/'weights/last.pt') if cli.resume else None,
                     epochs=None, imgsz=None, batch=None,
                     trainer_class=ConnectedTrialTrainer, align_preprocessing=True,
                     deployment_selection=True, selection_nms_iou=0.3)
    train(args)
