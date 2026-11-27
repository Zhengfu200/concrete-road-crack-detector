from pathlib import Path
import torch
import pytest

from review_training import make_review_sampler


def test_review_sampler_uses_only_train_ids_and_expected_probability():
    images=[f'/train/{i}.jpg' for i in range(10)]
    _, uniform=make_review_sampler(images,['0','1'],1)
    _, hard=make_review_sampler(images,['0','1'],3)
    assert uniform.tolist()==[1.]*10
    assert hard.tolist()==[3.,3.]+[1.]*8
    assert float(hard[:2].sum()/hard.sum())==pytest.approx(6/14)
    with pytest.raises(ValueError,match='非训练集'):
        make_review_sampler(images,['val-image'],3)


def test_sampler_is_reproducible_and_does_not_change_dataset_length():
    a,_=make_review_sampler([f'/train/{i}.jpg' for i in range(100)],['0','1'],3)
    b,_=make_review_sampler([f'/train/{i}.jpg' for i in range(100)],['0','1'],3)
    assert list(a)==list(b)
    assert len(a)==100


def test_infinite_loader_replacement_actually_uses_review_sampler():
    from ultralytics.data.build import InfiniteDataLoader
    dataset=torch.arange(10)
    loader=InfiniteDataLoader(dataset,batch_size=2,shuffle=True,num_workers=0)
    sampler,_=make_review_sampler([f'/train/{i}.jpg' for i in range(10)],['0'],100000)
    loader.batch_sampler.sampler.sampler=sampler
    object.__setattr__(loader,'sampler',sampler)
    loader.reset()
    actual=torch.cat(list(loader)).tolist()
    assert actual==[0]*10


def test_bn_statistics_stay_frozen_but_model_can_train():
    from review_training import ReviewedSamplerTrainer
    trainer=ReviewedSamplerTrainer.__new__(ReviewedSamplerTrainer)
    trainer.model=torch.nn.Sequential(torch.nn.Conv2d(3,3,1),torch.nn.BatchNorm2d(3))
    trainer.freeze_layer_names=[]
    trainer._model_train()
    before=trainer.model[1].running_mean.clone()
    trainer.model(torch.randn(2,3,8,8)).sum().backward()
    assert torch.equal(before,trainer.model[1].running_mean)
    assert trainer.model[0].training is True and trainer.model[0].weight.grad is not None
