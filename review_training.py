"""Single-GPU sampling ablation; both arms share geometry and fixed BN statistics."""
from pathlib import Path

import torch
from torch.nn.modules.batchnorm import _BatchNorm

from common import read_json, write_json
from training_improvements import AlignedDetectionTrainer


def make_review_sampler(image_files, hard_ids, multiplier, seed=42):
    hard_ids = set(hard_ids)
    ids = [Path(p).stem for p in image_files]
    if not hard_ids <= set(ids):
        raise ValueError('采样名单包含非训练集图片')
    if multiplier < 1:
        raise ValueError('困难采样倍率至少为1')
    weights = torch.tensor([multiplier if i in hard_ids else 1. for i in ids],dtype=torch.double)
    generator = torch.Generator().manual_seed(seed+173)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(ids), replacement=True, generator=generator)
    return sampler, weights


class ReviewedSamplerTrainer(AlignedDetectionTrainer):
    hard_multiplier = 1.

    def get_dataloader(self, dataset_path, batch_size=16, rank=-1, mode='train'):
        loader = super().get_dataloader(dataset_path,batch_size,rank,mode)
        if mode != 'train':
            return loader
        if rank != -1:
            raise ValueError('本次小规模对照只支持单GPU')
        spec = read_json(Path(self.args.data).parent/'sampling.json')
        sampler, weights = make_review_sampler(loader.dataset.im_files, spec['hard_ids'], self.hard_multiplier,self.args.seed)
        loader.batch_sampler.sampler.sampler = sampler
        object.__setattr__(loader,'sampler',sampler)
        loader.reset()
        hard_set = set(spec['hard_ids'])
        hard_weight = sum(float(w) for p,w in zip(loader.dataset.im_files,weights) if Path(p).stem in hard_set)
        write_json(Path(self.save_dir)/'sampling-configuration.json',{
            'multiplier':self.hard_multiplier,'weights':weights.tolist(),'hard_ids':spec['hard_ids'],
            'unique_images':len(weights),'draws_per_epoch':len(sampler),'replacement':True,
            'expected_hard_probability':hard_weight/float(weights.sum()),
            'bn_running_statistics':'frozen in both arms','data_fingerprint':spec['dataset_provenance_sha256']})
        return loader

    def _model_train(self):
        super()._model_train()
        # Preserve the source checkpoint's running statistics in BOTH small-sample arms.
        for module in self.model.modules():
            if isinstance(module,_BatchNorm):
                module.eval()


class UniformReviewTrainer(ReviewedSamplerTrainer):
    hard_multiplier = 1.


class HardReviewTrainer(ReviewedSamplerTrainer):
    hard_multiplier = 3.
