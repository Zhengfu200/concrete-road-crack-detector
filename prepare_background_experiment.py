"""Record 80 actually inspected crops and export a separate train-only augmentation version."""
from copy import deepcopy
from pathlib import Path
from common import ROOT,now,read_json,write_json,digest
from prepare_data import materialize

REJECTED={
 2:'表面短线和细碎线条疑似细裂缝，不能确定为背景。',
 3:'底部剥落边界和细线不明确。',8:'上部横线存在接缝或裂缝歧义。',
 10:'斜向线状剥落边界可能包含细裂缝。',17:'金属构件左侧有裂纹状网络。',
 23:'白色表面可见斜向细线，不能排除裂缝。',32:'右上区域可见短竖向细裂缝。',
 36:'左侧粗糙区域的细线连通关系不清。',41:'中部有近竖向细线，不能排除裂缝。',
 43:'上边缘粗糙缺口与细线不明确。',47:'颗粒团块及右边界存在不明确的细暗线。',
 49:'左侧纵向边界可能含裂隙。',68:'纵向曲折细沟可能为细裂缝。',
 72:'右侧存在细线状龟裂。',75:'横纵接缝及剥落区有裂纹状边界。',
 77:'污迹边界及孔洞附近不能可靠排除细裂隙。',78:'可见明确分叉细裂缝。',
 79:'密集暗色纹理模糊，无法可靠排除裂缝。'}

def main():
    review=ROOT/'reports/optimization-20261027/background-review'
    selection=read_json(review/'selection.json')
    decisions=[]
    for i,row in enumerate(selection['samples'],1):
        assert digest(Path(row['image']))==row['image_sha256']
        accepted=i not in REJECTED
        decisions.append(dict(row,status='agent_background_reviewed' if accepted else 'excluded_uncertain',
            decision_at=now(),reviewer='Codex native-size crop visual inspection',reviewer_type='agent',
            observation=REJECTED.get(i,'已查看200×200原始像素裁剪：可见颗粒、平滑底面或污渍纹理，未见可辨认的线状裂缝。'),
            evidence_sha256=digest(Path(row['evidence_sheet'])),
            material_certified=False,source_independence_certified=False))
    assert len(decisions)==80 and sum(r['status']=='agent_background_reviewed' for r in decisions)==62
    write_json(review/'decisions.json',dict(at=now(),reviewed=80,accepted=62,excluded=18,rows=decisions,
         scope='AI visual negatives from original training pixels, not human/domain certification'))
    parent=ROOT/'data/prepared/crack-connected-v2'
    dest=ROOT/'data/prepared/crack-connected-v2-background-20261027'
    splits=deepcopy(read_json(parent/'provenance.json')['splits'])
    for split,rows in splits.items():
        for row in rows:
            row['historical_source_image']=row['image']
            row['image']=str(parent/'images'/split/f"{row['id']}{Path(row['image']).suffix}")
    ids=[]
    for row in decisions:
        if row['status']!='agent_background_reviewed':continue
        ident='background-'+row['review_id']+'-'+row['parent_id']
        ids.append(ident)
        splits['train'].append(dict(id=ident,image=row['image'],sha256=row['image_sha256'],boxes=[],
            group='parent-image:'+row['parent_sha256'],group_confirmed=False,reviewed=False,surface='unverified',
            family=row['family'],split='train',parent_id=row['parent_id'],parent_sha256=row['parent_sha256'],
            crop_xyxy=row['crop_xyxy'],annotation_source='AI inspected train-derived negative crop',
            agent_background_review=row,independent_new_observation=False))
    materialize(splits,dest,'user_labeled')
    verified={}
    for split in ('val','test'):
        before={p.relative_to(parent).as_posix():digest(p) for folder in ('images','labels') for p in (parent/folder/split).iterdir()}
        after={p.relative_to(dest).as_posix():digest(p) for folder in ('images','labels') for p in (dest/folder/split).iterdir()}
        assert before==after
        verified[split]=dict(files=len(before),byte_identical=True)
    sampling=dict(hard_ids=ids,role='background negatives',multiplier=8.,replacement=True,
        draws_per_epoch=len(splits['train']),seed=42,unique_original_train_images=3073,
        unique_derived_crops=62,source_provenance_sha256=digest(parent/'provenance.json'),
        dataset_provenance_sha256=digest(dest/'provenance.json'),
        review_record_sha256=digest(review/'decisions.json'),val_test_byte_identical=verified,
        default_training_dataset_changed=False,raw_source_and_frozen_pilot_changed=False)
    write_json(dest/'sampling.json',sampling)
    write_json(review/'export-verification.json',dict(at=now(),directory=str(dest),**sampling))

if __name__=='__main__':main()
