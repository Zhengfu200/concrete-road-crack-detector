import sys,json,requests,zipfile,collections
from pathlib import Path
sys.path.insert(0,str(Path(r'D:\Project\专业导论项目\crack_detector')))
from common import ROOT,digest,write_json,now
from concurrent.futures import ThreadPoolExecutor
out=ROOT/'data/raw/damsegment-2025';out.mkdir(exist_ok=True)
meta=json.loads((ROOT/'reports/dataset-search-20261029/mendeley-z5z6gtt5t4-files.json').read_text(encoding='utf-8'))
def fetch(row):
 p=out/row['filename']
 if not p.exists():
  with requests.get(row['content_details']['download_url'],stream=True,timeout=(20,90)) as r:
   r.raise_for_status()
   with p.with_suffix('.partial').open('wb') as f:
    for chunk in r.iter_content(1024*1024): f.write(chunk)
  p.with_suffix('.partial').replace(p)
 assert digest(p)==row['content_details']['sha256_hash']
 with zipfile.ZipFile(p) as z:
  names=z.namelist();print(row['filename'],len(names),names[:15],flush=True)
  write_json(out/(p.stem+'.inventory.json'),names)
 return dict(file=str(p),sha256=digest(p),bytes=p.stat().st_size)
with ThreadPoolExecutor(2) as pool: results=list(pool.map(fetch,[r for r in meta if r['filename']!='Damage Classification.zip']))
write_json(out/'source.json',dict(at=now(),page='https://data.mendeley.com/datasets/z5z6gtt5t4/1',license='CC BY 4.0',files=results))
np=ROOT/'data/raw/sdnet2018/Pavements/Non-cracked'
groups=collections.Counter(p.stem.split('-')[0] for p in np.glob('*.jpg'))
print('SDNET groups',len(groups),groups.most_common(5),flush=True)
