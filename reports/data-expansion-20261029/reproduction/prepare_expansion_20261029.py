"""Explicit, reproducible candidate intake. Does not edit original sources/defaults."""
import sys, json, random, hashlib, zipfile, math
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path.cwd()))
from common import ROOT, digest, read_json, write_json, now

REPORT=ROOT/'reports/data-expansion-20261029'
RAW=ROOT/'data/raw/damsegment-2025/selected-source'
def phash(im):
    v=cv2.dct(cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(32,32)).astype(np.float32))[:8,:8].flatten()
    return sum(int(x>np.median(v[1:]))<<i for i,x in enumerate(v))
def main():
    REPORT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    old=read_json(ROOT/'data/prepared/crack-connected-v2/provenance.json')['splits']
    hashes={r['sha256'] for rows in old.values() for r in rows}; near=[]
    for s,rows in old.items():
        for r in rows:
            p=ROOT/'data/prepared/crack-connected-v2/images'/s/(r['id']+Path(r['image']).suffix)
            near.append((phash(cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),1)),r['id'],s))
    rng=random.Random(20261029)
    negatives=sorted((ROOT/'data/raw/sdnet2018/Pavements/Non-cracked').glob('*.jpg'))
    groups=sorted({p.stem.split('-')[0] for p in negatives});rng.shuffle(groups)
    devgroups=set(groups[:21]);rng.shuffle(negatives)
    chosen=[];counts=Counter();excluded=[]
    for p in negatives:
        split='val' if p.stem.split('-')[0] in devgroups else 'train'
        if counts[split]>=(100 if split=='val' else 500):continue
        raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest();im=cv2.imdecode(np.frombuffer(raw,np.uint8),1)
        if im is None:raise ValueError(p)
        h=phash(im);similar=[(rid,s) for hh,rid,s in near if (h^hh).bit_count()<=6]
        if sha in hashes or similar:
            excluded.append(dict(source=str(p),reason='sha_or_phash_distance_le_6',matches=similar));continue
        rid='sdnet-pu-'+p.stem
        chosen.append(dict(id=rid,image=str(p),sha256=sha,boxes=[],group='sdnet-pavement-prefix:'+p.stem.split('-')[0],group_confirmed=False,reviewed=False,surface='concrete_upstream',source='SDNET2018 pavement non-cracked',source_url='https://digitalcommons.usu.edu/all_datasets/48/',license='CC-BY-4.0',annotation_origin='upstream_noncracked_classification_empty_detection_label',split=split,phash=f'{h:016x}'))
        counts[split]+=1;hashes.add(sha);near.append((h,rid,split))
        if sum(counts.values())==600:break
    z=zipfile.ZipFile(ROOT/'data/raw/damsegment-2025/Damage Segmentaion.zip')
    candidates=[]
    for n in z.namelist():
        if '/Labels/Yolo/' not in n or not n.endswith('.txt'):continue
        polys=[list(map(float,line.split())) for line in z.read(n).decode().splitlines() if line.strip()]
        if 1<=len(polys)<=2 and all(p[0]==0 for p in polys):candidates.append((n,polys))
    rng.shuffle(candidates);candidates.sort(key=lambda x:len(x[1]))
    pos=[]
    for n,polys in candidates:
        stem=Path(n).stem;group=n.split('/')[1];member=f'Damage Segmentaion/{group}/Images/{stem}.jpg'
        raw=z.read(member);sha=hashlib.sha256(raw).hexdigest();im=cv2.imdecode(np.frombuffer(raw,np.uint8),1)
        if im is None:raise ValueError(member)
        h=phash(im);similar=[(rid,s) for hh,rid,s in near if (h^hh).bit_count()<=6]
        if sha in hashes or similar:
            excluded.append(dict(source=member,reason='sha_or_phash_distance_le_6',matches=similar));continue
        rid='dam-'+stem.replace(' (','-').replace(')','');p=RAW/(rid+'.jpg');p.write_bytes(raw)
        (RAW/(rid+'.polygons.txt')).write_bytes(z.read(n))
        boxes=[]
        for a in polys:
            xy=np.array(a[1:]).reshape(-1,2);lo=xy.min(0);hi=xy.max(0);boxes.append([0,*((lo+hi)/2).tolist(),*(hi-lo).tolist()])
        pos.append(dict(id=rid,image=str(p),sha256=sha,boxes=boxes,source_polygons=polys,group='damsegment-unconfirmed-capture',group_confirmed=False,reviewed=False,surface='concrete_dam_upstream',source='DamSegment',source_url='https://data.mendeley.com/datasets/z5z6gtt5t4/1',license='CC-BY-4.0',archive_member=member,annotation_origin='upstream_crack_polygons_to_boxes_pending_connectivity_screen',split='train',phash=f'{h:016x}'))
        hashes.add(sha);near.append((h,rid,'train'))
        if len(pos)==220:break
    write_json(REPORT/'candidates.json',dict(created_at=now(),seed=20261029,source_group_note='SDNET filename-prefix proxy; not independently confirmed capture metadata. DamSegment train only.',negative=chosen,positive=pos,excluded=excluded))
    for kind,rows,side,cols,per in [('negative',chosen,256,5,25),('positive',pos,320,4,16)]:
        folder=REPORT/'review-sheets'/kind;folder.mkdir(parents=True,exist_ok=True)
        for page in range(math.ceil(len(rows)/per)):
            subset=rows[page*per:(page+1)*per];canvas=Image.new('RGB',(cols*side,math.ceil(len(subset)/cols)*(side+24)),(30,30,30));d=ImageDraw.Draw(canvas)
            for j,r in enumerate(subset):
                im=cv2.imdecode(np.frombuffer(Path(r['image']).read_bytes(),np.uint8),1)
                if kind=='positive':
                    for k,a in enumerate(r['source_polygons']):
                        pts=np.round(np.array(a[1:]).reshape(-1,2)*[im.shape[1],im.shape[0]]).astype('int32');cv2.polylines(im,[pts],True,(0,255,0) if k==0 else (255,0,255),1)
                im=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((side,side));x=j%cols*side;y=j//cols*(side+24)
                canvas.paste(im,(x,y));d.text((x+3,y+side+3),f'{page*per+j:03d} {r["id"]} {r["split"]}',fill='white')
            canvas.save(folder/f'{page:02d}.jpg',quality=94)
    print(json.dumps(dict(negative=counts,positive=len(pos),excluded=len(excluded)),ensure_ascii=False))
if __name__=='__main__':main()
