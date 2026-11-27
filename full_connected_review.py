"""Contact-sheet-assisted, explicit per-image AI review; no automatic box merging."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from common import ROOT, digest, now, read_json, write_json
from connected_annotations import WORK, annotation_path, default_objects, frozen, union_box, validate_annotation

OUT = ROOT/'reports/connected-v2-full-review-20261025'
SIZE, PER_SHEET = 320, 40


def build():
    if not frozen(WORK):
        raise ValueError('规则尚未冻结')
    if (OUT/'order.json').exists():
        raise ValueError('已存在全量复核清单，不覆盖')
    source = read_json(WORK/'source-index.json')
    pilot = {x['id'] for x in read_json(WORK/'pilot.json')['items']}
    rows = sorted((r for r in source['rows'] if r['id'] not in pilot),
                  key=lambda r: ({'val':0,'test':1,'train':2}[r['split']], int(r['family']) if r['family'].isdigit() else 100000, r['id']))
    OUT.mkdir(parents=True, exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    order=[]
    for start in range(0,len(rows),PER_SHEET):
        sheet=Image.new('RGB',(SIZE*8,(SIZE+28)*5),'white');d=ImageDraw.Draw(sheet)
        for j,row in enumerate(rows[start:start+PER_SHEET]):
            x=(j%8)*SIZE;y=(j//8)*(SIZE+28)
            im=Image.open(row['image']).convert('RGB').resize((SIZE,SIZE))
            draw=ImageDraw.Draw(im)
            for k,ps in enumerate(row['polygons']):
                pts=[(p[0]*SIZE,p[1]*SIZE) for p in ps]
                draw.line(pts+[pts[0]], fill='#00e7e7',width=1)
                px=max(0,min(SIZE-28,pts[len(pts)//2][0]));py=max(0,min(SIZE-16,pts[len(pts)//2][1]))
                draw.rectangle((px,py,px+25,py+16),fill='black');draw.text((px+1,py),'p'+str(k),fill='white',font=font)
            sheet.paste(im,(x,y+28))
            n=start+j+1
            d.text((x+3,y+5),f'{n:04d} {row["split"]} {row["family"]} n={len(row["polygons"])}',fill='black',font=font)
            order.append({'index':n,'id':row['id'],'split':row['split'],'family':row['family'],
                          'polygons':len(row['polygons']),'sheet':start//PER_SHEET+1})
        (OUT/'sheets').mkdir(exist_ok=True)
        sheet.save(OUT/'sheets'/f'{start//PER_SHEET+1:03d}.jpg',quality=96)
    write_json(OUT/'order.json',{'created_at':now(),'source_index_sha256':digest(WORK/'source-index.json'),
        'rule_freeze_sha256':digest(WORK/'rule-freeze.json'),'tile_pixels':SIZE,
        'initial_view':'original with source polygon outlines; no model predictions', 'items':order})
    print(f'{len(rows)} images / {(len(rows)+PER_SHEET-1)//PER_SHEET} sheets',flush=True)


def record(decisions_path):
    """Accept only explicit indices for sheets actually inspected by Codex."""
    if not frozen(WORK):
        raise ValueError('冻结记录变化，停止')
    source=read_json(WORK/'source-index.json');rows={r['id']:r for r in source['rows']}
    order_info=read_json(OUT/'order.json');order={r['index']:r for r in order_info['items']}
    if digest(WORK/'rule-freeze.json')!=order_info['rule_freeze_sha256'] or digest(WORK/'source-index.json')!=order_info['source_index_sha256']:
        raise ValueError('来源或规则冻结记录变化')
    decisions=read_json(decisions_path)
    current_rules=digest(WORK/'rules.md')
    for sheet in decisions:
        expected={x['index'] for x in order.values() if x['sheet']==sheet['sheet']}
        whole=set(sheet.get('whole',[]));separate=set(sheet.get('separate',[]))
        grouped={int(k):v for k,v in sheet.get('groups',{}).items()}
        pending={int(k):v for k,v in sheet.get('pending',{}).items()}
        categories=[whole,separate,set(grouped),set(pending)]
        flattened=[i for c in categories for i in c]
        if set(flattened)!=expected or len(flattened)!=len(set(flattened)):
            raise ValueError(f'第{sheet["sheet"]}页必须逐张明确记录且无重复：缺{expected-set(flattened)}')
        evidence=OUT/'sheets'/f'{sheet["sheet"]:03d}.jpg'
        evidence_sha=digest(evidence)
        for n in sorted(expected):
            item=order[n];row=rows[item['id']];path=annotation_path(WORK,row['id'])
            if path.exists():
                continue  # Never replace the user's work or an earlier completed review.
            if n in whole:
                groups=[list(range(len(row['polygons'])))]
                why='逐图查看原图与源多边形：可见主裂缝/连通分支按一个整体；原多边形覆盖可确认范围。'
            elif n in separate:
                groups=[[i] for i in range(len(row['polygons']))]
                why='逐图查看原图与源多边形：多个可见裂缝在画面内未见连接，分别保留；不依框重叠合并。'
            elif n in grouped:
                groups=grouped[n]
                why='逐图查看原图与源多边形：同一可见连通整体的片段按记录分组，独立目标保持分开。'
            else:
                groups=[[i] for i in range(len(row['polygons']))]
                why='逐图图像检查待定：'+pending[n]+'；整图不导出，不作为背景。'
            objects=[{'polygon_indices':g,'box':union_box(row['polygons'],g),'reason':why} for g in groups if g]
            value={'status':'pending' if n in pending else 'agent_reviewed','reason':why,'objects':objects,
                   'excluded':[],'background_confirmed':False}
            validate_annotation(row,value)
            value.update(id=row['id'],split=row['split'],rules_sha256=current_rules,
                source_image_sha256=row['sha256'],source_annotation_sha256=row['annotation_sha256'],
                updated_at=now(),revision=1,reviewer='Codex contact-sheet image inspection; not human/domain certification',
                reviewer_type='agent',visual_evidence={'sheet':str(evidence),'sha256':evidence_sha,
                    'image_index':n,'tile_pixels':SIZE,'model_predictions_visible':False})
            history=WORK/'history'/row['id']/'00001.json'
            history.parent.mkdir(parents=True,exist_ok=True)
            with history.open('x',encoding='utf-8') as f:
                __import__('json').dump(value,f,ensure_ascii=False,indent=2)
            write_json(path,value)
    if not frozen(WORK):
        raise ValueError('复核期间冻结记录发生变化')
    all_records=[read_json(p) for p in (WORK/'annotations').glob('*.json')]
    stats={'updated_at':now(),'source_total':len(rows),'recorded':len(all_records),
           'remaining':len(rows)-len(all_records),'states':dict(Counter(r['status'] for r in all_records))}
    write_json(OUT/'progress.json',stats)
    print(stats,flush=True)


def audit():
    """Verify decision coverage, saved revisions, evidence and exported membership."""
    import csv
    from prepare_data import dataset_readiness
    assert frozen(WORK), 'Frozen human pilot changed'
    source = read_json(WORK/'source-index.json')
    order_info = read_json(OUT/'order.json')
    order = {r['index']: r for r in order_info['items']}
    assert digest(WORK/'source-index.json') == order_info['source_index_sha256']
    assert digest(WORK/'rule-freeze.json') == order_info['rule_freeze_sha256']
    seen, expected = set(), {}
    for path in sorted(OUT.glob('decisions-*.json')):
        for page in read_json(path):
            page_seen = set()
            for category in ('whole', 'separate', 'groups', 'pending'):
                values = page.get(category, [])
                for raw in values:
                    n = int(raw)
                    assert n not in seen and n not in page_seen, ('duplicate', n)
                    assert order[n]['sheet'] == page['sheet'], ('wrong page', n)
                    page_seen.add(n)
                    expected[order[n]['id']] = (category, values[str(n)] if category == 'groups' else None)
            assert page_seen == {n for n, r in order.items() if r['sheet'] == page['sheet']}
            seen.update(page_seen)
    assert seen == set(order) and len(seen) == 3978
    published = ROOT/'data/prepared/crack-connected-v2'
    version = read_json(published/'version.json')
    provenance = read_json(published/'provenance.json')
    exported = {r['id']: r for rows in provenance['splits'].values() for r in rows}
    export_audit = {r['id']: r for r in read_json(published/'annotation-audit.json')}
    evidence_hashes = {s: digest(OUT/'sheets'/f'{s:03d}.jpg') for s in range(1,101)}
    states, stats, rows = Counter(), {}, []
    for row in source['rows']:
        record = read_json(annotation_path(WORK, row['id']))
        validate_annotation(row, record)
        assert export_audit[row['id']] == record
        history_name = f'{record["revision"]:05d}.json'
        assert read_json(WORK/'history'/row['id']/history_name) == record
        assert digest(WORK/'history'/row['id']/history_name) == digest(published/'review-history'/row['id']/history_name)
        states[record['status']] += 1
        objects = record['objects']
        stat = stats.setdefault(row['split'], Counter(parent=0, retained=0, pending=0, old_boxes_retained=0, new_boxes=0, merged_images=0, multi_target_images=0))
        stat['parent'] += 1
        evidence = record.get('visual_evidence', {})
        if row['id'] in expected:
            category, group_spec = expected[row['id']]
            assert record['reviewer_type'] == 'agent'
            item = order[evidence['image_index']]
            assert item['id'] == row['id'] and evidence['sha256'] == evidence_hashes[item['sheet']]
            assert evidence['model_predictions_visible'] is False
            assert (record['status'] == 'pending') == (category == 'pending')
            if category != 'pending':
                groups = [o['polygon_indices'] for o in objects]
                reference = ([list(range(len(row['polygons']))) ] if category == 'whole' else
                             [[i] for i in range(len(row['polygons']))] if category == 'separate' else group_spec)
                assert groups == reference, row['id']
        else:
            assert record['status'] == 'confirmed' and record['reviewer_type'] == 'human'
        if record['status'] == 'pending':
            assert row['id'] not in exported
            stat['pending'] += 1
            assert not (published/'labels'/row['split']/f'{row["id"]}.txt').exists()
        else:
            assert exported[row['id']]['split'] == row['split']
            assert exported[row['id']]['boxes'] == [o['box'] for o in objects]
            label = published/'labels'/row['split']/f'{row["id"]}.txt'
            actual = [[float(v) for v in line.split()] for line in label.read_text().splitlines() if line.strip()]
            assert len(actual) == len(objects)
            assert all(abs(a-b) < 1e-6 for arow, obj in zip(actual, objects) for a,b in zip(arow,obj['box']))
            stat.update(retained=1, old_boxes_retained=len(row['polygons']), new_boxes=len(objects),
                        merged_images=int(len(objects)<len(row['polygons'])), multi_target_images=int(len(objects)>1))
        rows.append(dict(id=row['id'], split=row['split'], family=row['family'], status=record['status'],
                         reviewer_type=record['reviewer_type'], source_polygons=len(row['polygons']),
                         output_objects=len(objects) if record['status'] != 'pending' else '',
                         image_index=evidence.get('image_index',''), sheet=evidence.get('sheet',''),
                         reason=record['reason'], source_image=row['image']))
    assert len(exported) == 3326 and len(rows) == 4028
    assert digest(published/'annotation-audit.json') == version['annotation_audit_sha256']
    for s in stats:
        assert stats[s]['retained'] == version['counts'][s]['retained']
        stats[s]['retention_rate'] = stats[s]['retained']/stats[s]['parent']
    for name, selected in [('per-image-review.csv', rows), ('pending-review.csv', [r for r in rows if r['status']=='pending'])]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(selected)
    result = {'created_at':now(), 'coverage':len(rows), 'states':dict(states), 'splits':stats,
              'decision_indices_verified':len(seen), 'evidence_sheets_verified':len(evidence_hashes),
              'frozen_pilot_unchanged':True, 'source_verified_during_export':version['source_files_verified_unchanged'],
              'pending_excluded_not_background':True, 'export_membership_labels_and_history_verified':True,
              'readiness':dataset_readiness(published), 'review_resolution':'320px tiles, displayed resized contact sheets; screening, not pixel-level certification',
              'domain_certification':False, 'selection_bias':'Filtered subset; scores cannot be compared directly with V1 or interpreted as training improvement.'}
    write_json(OUT/'verification.json',result)
    print({k:v for k,v in result.items() if k not in ('readiness',)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['build','record','audit'])
    p.add_argument('--decisions',type=Path);a=p.parse_args()
    if a.action=='build': build()
    elif a.action=='record': record(a.decisions)
    else: audit()
