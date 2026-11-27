"""Export the same small training roster for both arms, with untouched val/test labels."""
from copy import deepcopy
from pathlib import Path
import random

from common import ROOT, digest, read_json, write_json
from prepare_data import materialize

SOURCE = ROOT/'data/prepared/crack-seg-user-20261021'
DESTINATION = ROOT/'data/prepared/crack-seg-hardreview-20261024'
REVIEW = ROOT/'reports/hard-example-review-20261024'


def main():
    original = read_json(SOURCE/'provenance.json')
    chosen = read_json(REVIEW/'hard-selection.json')
    hard_ids = set(chosen['ids'])
    train = original['splits']['train']
    lookup = {r['id']:r for r in train}
    if not hard_ids <= lookup.keys():
        raise ValueError('困难样本必须全部来自原train')
    hard = [lookup[i] for i in chosen['ids']]
    hard_families = {r['id'].split('.rf.')[0] for r in hard}
    families = {}
    for row in train:
        family = row['id'].split('.rf.')[0]
        if family not in hard_families:
            families.setdefault(family, row)
    normal = list(families.values())
    random.Random(42).shuffle(normal)
    selected = hard + normal[:640-len(hard)]
    if len(selected) != 640:
        raise ValueError('需要640张不同训练图片')
    splits = {'train': selected, 'val': original['splits']['val'], 'test': original['splits']['test']}
    prepared = deepcopy(splits)
    for split, rows in prepared.items():
        for r in rows:
            r['historical_source_image'] = r['image']
            r['image'] = str(SOURCE/'images'/split/f"{r['id']}{Path(r['image']).suffix.lower()}")
            r['agent_visual_review'] = 'hard-example-review-20261024' if r['id'] in hard_ids else None
            # Original reviewed/material/source status remains untouched: this is not formal domain certification.
    materialize(prepared, DESTINATION, 'user_labeled')
    unchanged = {}
    for split in ('val','test'):
        for r in splits[split]:
            for part, suffix in [('images','.jpg'), ('labels','.txt')]:
                before = SOURCE/part/split/(r['id']+suffix)
                after = DESTINATION/part/split/(r['id']+suffix)
                if digest(before) != digest(after):
                    raise ValueError(f'对照的{split}内容改变: {before}')
        unchanged[split] = True
    sampling = {'hard_ids':chosen['ids'],'multiplier':3.,'replacement':True,'draws_per_epoch':640,
                'seed':42,'source_provenance_sha256':digest(SOURCE/'provenance.json'),
                'review_record_sha256':digest(REVIEW/'train-visual-review.json'),
                'dataset_provenance_sha256':digest(DESTINATION/'provenance.json'),
                'val_test_byte_identical':unchanged,'labels_modified':0,
                'unique_train_images':640,'hard_images':len(hard),
                'selection':'one variant per filename augmentation family; same roster for both arms'}
    write_json(DESTINATION/'sampling.json',sampling)
    write_json(REVIEW/'prepared-data.json',{'directory':str(DESTINATION),**sampling})
    print(sampling,flush=True)


if __name__ == '__main__':
    main()
