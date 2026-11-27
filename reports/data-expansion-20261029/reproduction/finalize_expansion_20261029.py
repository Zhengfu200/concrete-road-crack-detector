import sys, copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from common import ROOT,read_json,write_json,now,digest
from prepare_data import materialize

report=ROOT/'reports/data-expansion-20261029'
c=read_json(report/'candidates.json')
negative_excluded={5,8,24,27,32,54,68,112,118,121,128,130,132,142,145,149,151,157,180,190,192,203,218,226,242,258,260,268,270,271,277,279,288,295,311,312,313,317,346,347,362,371,378,384,393,395,402,413,426,441,449,450,453,454,467,483,497,500,510,515,523,526,532,554,558,564,568,572,578,580,589,598,599}
positive_excluded={26,37,84,85,93,96,110,112,117,123,135,136,146,159,169,178,181,197,198}
merged={105,106,107,109,113,116,118,120,124,128,129,132,138,142,147,149,150,151,155,156,160,164,166,167,168,170,171,172,174,175,180,184,185,186,189,190,192,194,195,199,202,204,207}
decisions=[];accepted=[]
for kind,excluded in [('negative',negative_excluded),('positive',positive_excluded)]:
    for i,row in enumerate(c[kind]):
        r=copy.deepcopy(row)
        action='exclude_ambiguous' if i in excluded else ('merge_visible_connected_polygons' if kind=='positive' and i in merged else 'retain_upstream')
        note=('Visible seam/crack/edge or ambiguous discontinuity; not used as clean negative.' if kind=='negative' else 'Incomplete/ambiguous annotation or connectivity at sheet resolution; exclude rather than guess.') if i in excluded else ('One visually connected crack; source fragments united, not based on bbox overlap.' if action.startswith('merge') else 'AI contact-sheet screen; source label retained; not a human/pixel-exhaustive certification.')
        decisions.append(dict(kind=kind,index=i,id=r['id'],action=action,note=note))
        if i in excluded:continue
        if kind=='positive' and i in merged:
            xy=np.concatenate([np.array(p[1:]).reshape(-1,2) for p in r['source_polygons']]);lo=xy.min(0);hi=xy.max(0)
            r['boxes']=[[0,*((lo+hi)/2).tolist(),*(hi-lo).tolist()]]
        r['ai_screen']={'status':'contact_sheet_screened','at':now(),'decision':action,'note':note,'human_reviewed':False}
        r['annotation_origin']='upstream_noncracked_with_ai_screen' if kind=='negative' else 'upstream_crack_polygons_with_ai_connectivity_screen'
        r.pop('source_polygons',None);accepted.append(r)
write_json(report/'screening-decisions.json',dict(at=now(),method='AI visual screen of all 24 native-256 negative sheets and 13 downsampled-320 positive polygon-overlay sheets; uncertain excluded. Not expert or exhaustive pixel review.',decisions=decisions))
splits=read_json(ROOT/'data/prepared/crack-connected-v2/provenance.json')['splits']
for s,rows in splits.items():
    for r in rows:
        r['image']=str(ROOT/'data/prepared/crack-connected-v2/images'/s/(r['id']+Path(r['image']).suffix))
for r in accepted:splits[r['split']].append(r)
dest=ROOT/'data/prepared/crack-road-expanded-20261029'
materialize(splits,dest,'user_labeled')
write_json(dest/'sampling.json',dict(hard_ids=[r['id'] for r in accepted if r['split']=='train' and not r['boxes']],dataset_provenance_sha256=digest(dest/'provenance.json'),description='2x SDNET train-negative sampling; all positives 1x, replacement sampler; frozen BN.'))
stats={s:dict(images=len(rows),crack=sum(bool(r['boxes']) for r in rows),background=sum(not r['boxes'] for r in rows)) for s,rows in splits.items()}
write_json(report/'intake-summary.json',dict(at=now(),dataset=str(dest),stats=stats,added_positive=sum(bool(r['boxes']) for r in accepted),added_negative=sum(not r['boxes'] for r in accepted),negative_excluded=len(negative_excluded),positive_excluded=len(positive_excluded),connected_merges=len(merged),exact_and_near_exclusions=c['excluded'],provenance_sha256=digest(dest/'provenance.json'),limitations=['AI sheet screening is not human ground-truth certification','SDNET groups are filename-prefix proxies, not confirmed capture metadata','DamSegment is concrete dam, train only; not proof of road-domain accuracy','Old V2 test retained unchanged but will not be evaluated in this experiment','pHash<=6 screen does not rule out partial crops or all visual overlap']))
print(stats)
