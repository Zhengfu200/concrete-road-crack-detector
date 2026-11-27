"""Propose only train-derived negative crops; every released crop needs visual evidence."""
import random
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from common import ROOT,read_json,write_json,digest,now
from metrics import xywh_to_xyxy

def main():
    out=ROOT/'reports/optimization-20261027/background-review'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'selection.json').exists():raise ValueError('Do not replace the review selection')
    data=ROOT/'data/prepared/crack-connected-v2'
    rows=read_json(data/'provenance.json')['splits']['train']
    random.Random(20261027).shuffle(rows)
    selected=[]
    families=set()
    for row in rows:
        family=row['id'].split('.rf.')[0]
        if family in families:continue
        source=data/'images/train'/f"{row['id']}.jpg"
        image=Image.open(source).convert('RGB')
        assert image.size==(416,416)
        boxes=[xywh_to_xyxy(b)[1:5] for b in row['boxes']]
        possibilities=[]
        for left,top in [(0,0),(216,0),(0,216),(216,216)]:
            crop=(left,top,left+200,top+200)
            # Entire crop must be separated from every annotated overall box by 8px.
            if all(crop[2]+8<=b[0]*416 or crop[0]-8>=b[2]*416 or
                   crop[3]+8<=b[1]*416 or crop[1]-8>=b[3]*416 for b in boxes):
                patch=image.crop(crop)
                gray=np.asarray(patch.convert('L'),dtype=float)
                if gray.std()>8 and np.mean(gray<8)<.03:
                    possibilities.append((float(gray.std()),crop,patch))
        if not possibilities:continue
        _,crop,patch=max(possibilities,key=lambda p:p[0])
        n=len(selected)+1
        filename=out/f'BG{n:03d}.png'
        patch.save(filename)
        selected.append(dict(review_id=f'BG{n:03d}',parent_id=row['id'],family=family,split='train',
            parent_image=str(source),parent_sha256=digest(source),crop_xyxy=list(crop),
            image=str(filename),image_sha256=digest(filename),model_predictions_visible=False,
            status='unreviewed',annotation_source='candidate only; no background label yet'))
        families.add(family)
        if len(selected)==80:break
    assert len(selected)==80
    font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',16)
    for page in range(5):
        sheet=Image.new('RGB',(4*224,4*232),'white')
        d=ImageDraw.Draw(sheet)
        for j,row in enumerate(selected[page*16:(page+1)*16]):
            x,y=(j%4)*224,(j//4)*232
            d.text((x,y),row['review_id']+' '+row['family'][:14],fill='black',font=font)
            sheet.paste(Image.open(row['image']),(x,y+28))
            row['evidence_sheet']=str(out/f'sheet-{page+1}.png')
        sheet.save(out/f'sheet-{page+1}.png')
    write_json(out/'selection.json',dict(at=now(),seed=20261027,count=len(selected),parent_split='train',
        crop_pixels=[200,200],display_scale=1,gt_exclusion_margin_pixels=8,
        validation_or_test_crops=False,samples=selected))
    print(out)

if __name__=='__main__':main()
