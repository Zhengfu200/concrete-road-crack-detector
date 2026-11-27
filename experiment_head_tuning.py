"""Conservative V2 head adaptation after the earlier full-model degradation."""
from argparse import Namespace
from copy import deepcopy
from torch.nn.modules.batchnorm import _BatchNorm
from common import ROOT, now, read_json, write_json, digest
from train import train
from training_improvements import AlignedDetectionTrainer

OUT=ROOT/'reports/optimization-20261027/head-tuning'
SOURCE=ROOT/'runs/concrete-crack-20261021-212347/weights/best.pt'

class HeadTrainer(AlignedDetectionTrainer):
    def _model_train(self):
        super()._model_train()
        for module in self.model.modules():
            if isinstance(module,_BatchNorm):
                module.eval()

    def _setup_train(self):
        super()._setup_train()
        if self.resume:
            return
        from ultralytics import YOLO
        from evaluate import collect
        from metrics import calibration_summary, threshold_curve
        import torch
        expected=YOLO(str(SOURCE))
        reference=expected.model.state_dict()
        current=self.model.state_dict()
        changed=[key for key in current if key not in reference or not torch.equal(current[key].cpu(),reference[key].cpu())]
        assert not changed, f'Warm-start altered before first step: {changed}'
        wrapper=YOLO(str(SOURCE))
        wrapper.model=deepcopy(self.model).float().eval()
        samples=collect(wrapper,ROOT/'data/prepared/crack-connected-v2','val',960,'0',.3)
        result=calibration_summary(threshold_curve(samples))
        write_json(OUT/'zero-step.json',dict(at=now(),tensor_exact=True,calibration=result,
                  trainable_parameters=sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
        print('ZERO STEP',result['recall_at_precision80']['metrics']['0'],flush=True)
        del expected,wrapper,samples
        torch.cuda.empty_cache()

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume',action='store_true')
    cli=p.parse_args()
    data=ROOT/'data/prepared/crack-connected-v2'
    cfg=ROOT/'configs/connected-v2-head-20261027.yaml'
    name='concrete-crack-connected-v2-head-20261027'
    OUT.mkdir(parents=True,exist_ok=True)
    plan=dict(at=now(),name=name,comparator='unaltered source checkpoint on identical V2 val',
              hypothesis='freeze feature extractor and BN, adapt detection head without affine/color distortions',
              data_provenance_sha256=digest(data/'provenance.json'),config_sha256=digest(cfg),
              source_sha256=digest(SOURCE),epochs=6,seed=42,selection_split='val',test_used=False,
              primary='recall_at_precision80',baseline_single_view_recall=152/169,
              criteria='Only adopt if better than unchanged baseline; retain failure results',
              candidate_status='unvalidated',training_only_data_changes='none; pending samples remain excluded')
    if cli.resume:
        old=read_json(OUT/'plan.json')
        assert old['config_sha256']==plan['config_sha256'] and old['source_sha256']==plan['source_sha256']
    else:
        if (OUT/'plan.json').exists(): raise ValueError('Experiment already exists')
        write_json(OUT/'plan.json',plan)
    train(Namespace(data=str(data),model=str(SOURCE),config=str(cfg),device='0',name=name,
                    smoke=False,resume=str(ROOT/'runs'/name/'weights/last.pt') if cli.resume else None,
                    epochs=None,imgsz=None,batch=None,trainer_class=HeadTrainer,align_preprocessing=True,
                    deployment_selection=True,selection_nms_iou=.3))

if __name__=='__main__':main()
