"""Versioned connected-crack review; never modifies source labels or retired manifest."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

from common import ROOT, digest, now, read_json, resolve, validate_boxes, write_json
from prepare_data import materialize, polygon_to_box, dataset_readiness
from prepare_external import SOURCE, DEFAULT_DATA

WORK = ROOT/'data/annotation/crack-connected-v2'
RULES = ROOT/'docs/crack-connected-v2-rules.md'
CATEGORIES = {'segmented': 20, 'branched': 10, 'independent': 10, 'faint': 5, 'edge_occlusion': 5}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def union_box(polygons, indices):
    pts = [p for i in indices for p in polygons[i]]
    if not pts:
        raise ValueError('实例必须关联多边形，或提供手动补框')
    xs, ys = zip(*pts)
    box = [0, (min(xs)+max(xs))/2, (min(ys)+max(ys))/2, max(xs)-min(xs), max(ys)-min(ys)]
    validate_boxes([box])
    return box


def default_objects(row):
    return [{'polygon_indices': [i], 'box': union_box(row['polygons'], [i]),
             'reason': '原多边形，尚待判断连通关系'} for i in range(len(row['polygons']))]


def initialize(work=WORK, parent=ROOT/DEFAULT_DATA, source=SOURCE):
    work, parent, source = map(lambda p: resolve(p).resolve(), (work, parent, source))
    if (work/'source-index.json').exists():
        raise ValueError('工作区已存在，不覆盖现有复核记录')
    info = read_json(parent/'provenance.json')
    if not info:
        raise ValueError('缺少父版本')
    rows = []
    for split in ('train', 'val', 'test'):
        for old in info['splits'][split]:
            row = deepcopy(old)
            image = parent/'images'/split/(row['id']+Path(row['image']).suffix.lower())
            label = source/'labels'/split/(row['id']+'.txt')
            if digest(image) != row['sha256'] or digest(label) != row['annotation_sha256']:
                raise ValueError(f'原图/多边形摘要变化：{row["id"]}')
            polygons = []
            for line in label.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():
                    continue
                vals = [float(v) for v in line.split()]
                box = polygon_to_box(line.split())
                validate_boxes([box])
                if len(vals) < 7 or (len(vals)-1) % 2 or vals[0] != 0:
                    raise ValueError('此流程要求原始单类多边形')
                if any(not 0 <= v <= 1 for v in vals[1:]):
                    raise ValueError('多边形坐标越界')
                polygons.append(list(map(list, zip(vals[1::2], vals[2::2]))))
            if len(polygons) != len(row['boxes']):
                raise ValueError('源多边形数量与父版本不一致')
            row.update(split=split, image=str(image), original_annotation=str(label),
                       parent_label=str(parent/'labels'/split/(row['id']+'.txt')),
                       parent_label_sha256=digest(parent/'labels'/split/(row['id']+'.txt')),
                       polygons=polygons, family=row['id'].split('.rf.')[0])
            rows.append(row)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('重复图片ID')
    work.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RULES, work/'rules.md')
    write_json(work/'source-index.json', {'created_at': now(), 'parent': str(parent),
               'parent_provenance_sha256': digest(parent/'provenance.json'),
               'source': str(source), 'rows': rows})
    return rows


def validate_annotation(row, value):
    if value.get('status') not in ('draft', 'confirmed', 'agent_reviewed', 'pending'):
        raise ValueError('无效复核状态')
    if not str(value.get('reason', '')).strip():
        raise ValueError('必须记录本图的判断依据或待定原因')
    objects, excluded = value.get('objects', []), value.get('excluded', [])
    if not isinstance(objects, list) or not isinstance(excluded, list):
        raise ValueError('标注格式错误')
    used = Counter()
    for obj in objects:
        validate_boxes([obj['box']])
        indices = obj.get('polygon_indices', [])
        if not isinstance(indices, list) or len(indices) != len(set(indices)):
            raise ValueError('多边形关联重复')
        for i in indices:
            if type(i) is not int or not 0 <= i < len(row['polygons']):
                raise ValueError('无效原多边形索引')
            used[i] += 1
        if not indices or obj['box'] != union_box(row['polygons'], indices):
            if not obj.get('reason', '').strip():
                raise ValueError('补框/修正边界必须记录理由')
        if len(indices) > 1 and not obj.get('reason', '').strip():
            raise ValueError('合并必须说明原图中连通证据')
    ignored = set()
    for e in excluded:
        i = e['polygon_index']
        if type(i) is not int or not 0 <= i < len(row['polygons']) or i in ignored or i in used:
            raise ValueError('排除的多边形无效、重复或仍在实例中')
        if not e.get('reason', '').strip():
            raise ValueError('排除必须记录原因')
        ignored.add(i)
    for i, count in used.items():
        if count > 1 and any(not o.get('split_reason', '').strip() for o in objects if i in o.get('polygon_indices', [])):
            raise ValueError('同一原多边形拆成多个实例必须逐一记录拆分依据')
    if value['status'] in ('confirmed', 'agent_reviewed'):
        if set(used) | ignored != set(range(len(row['polygons']))):
            raise ValueError('仍有原多边形未关联或未说明排除原因')
        if not objects and value.get('background_confirmed') is not True:
            raise ValueError('空标注必须显式确认原图确实无裂缝；疑似裂缝请标待定')
    # NaNs and non-JSON values are not accepted anywhere in the review record.
    fingerprint(value)


def annotation_path(work, image_id):
    return Path(work)/'annotations'/(image_id+'.json')


def save_annotation(work, image_id, payload, reviewer, reviewer_type='human'):
    work = Path(work)
    rows = {r['id']: r for r in read_json(work/'source-index.json')['rows']}
    if image_id not in rows:
        raise ValueError('图片不在此数据版本')
    row = rows[image_id]
    pilot = read_json(work/'pilot.json', {}).get('items', [])
    if image_id not in {x['id'] for x in pilot} and not frozen(work):
        raise ValueError('先完成50张试标确认并冻结规则')
    if not reviewer.strip() or reviewer_type not in ('human', 'agent'):
        raise ValueError('必须填写复核者与复核者类型')
    old = read_json(annotation_path(work, image_id), {})
    if payload.get('expected_revision', 0) != old.get('revision', 0):
        raise ValueError('此图已被更新，请重新加载，避免覆盖他人的修改')
    value = {k: deepcopy(payload[k]) for k in ('status', 'reason', 'objects', 'excluded', 'background_confirmed') if k in payload}
    if reviewer_type == 'agent' and value.get('status') == 'confirmed':
        raise ValueError('AI提案不能记作用户确认')
    if value.get('status') == 'agent_reviewed' and (reviewer_type != 'agent' or image_id in {x['id'] for x in pilot}):
        raise ValueError('试标须用户确认；全量AI复核需独立记录为agent_reviewed')
    validate_annotation(row, value)
    value.update(id=image_id, split=row['split'], rules_sha256=digest(work/'rules.md'),
                 source_image_sha256=row['sha256'], source_annotation_sha256=row['annotation_sha256'],
                 updated_at=now(), revision=old.get('revision', 0)+1,
                 reviewer=reviewer.strip(), reviewer_type=reviewer_type)
    path = annotation_path(work, image_id)
    # Archive every revision, including the initial proposal; immutable once written.
    history = work/'history'/image_id/f'{value["revision"]:05d}.json'
    if history.exists():
        raise ValueError('修订历史冲突')
    write_json(history, value)
    write_json(path, value)
    return value


def pilot_digest(work):
    pilot = read_json(Path(work)/'pilot.json')
    if not pilot or len(pilot['items']) != 50:
        raise ValueError('缺少50张试标清单')
    return fingerprint({'pilot': pilot, 'annotations': [read_json(annotation_path(work, x['id'])) for x in pilot['items']]})


def freeze_rules(work, reviewer):
    work = Path(work)
    rows = {r['id']: r for r in read_json(work/'source-index.json')['rows']}
    pilot = read_json(work/'pilot.json', {}).get('items', [])
    if Counter(x['category'] for x in pilot) != CATEGORIES or len({rows[x['id']]['family'] for x in pilot}) != 50:
        raise ValueError('试标须符合50张分类配额且不同增强家族')
    for item in pilot:
        row = rows[item['id']]
        record = read_json(annotation_path(work, item['id']), {})
        if row['split'] != 'train' or record.get('status') != 'confirmed' or record.get('reviewer_type') != 'human':
            raise ValueError('50张试标须由用户逐张确认，待定图需先解决')
        if record.get('rules_sha256') != digest(work/'rules.md'):
            raise ValueError('规则已变更，请重新确认试标')
        validate_annotation(row, record)
    if not reviewer.strip():
        raise ValueError('必须填写确认者')
    stamp = {'frozen_at': now(), 'reviewer': reviewer.strip(), 'rules_sha256': digest(work/'rules.md'),
             'pilot_sha256': pilot_digest(work), 'source_index_sha256': digest(work/'source-index.json')}
    write_json(work/'rule-freeze.json', stamp)
    return stamp


def frozen(work):
    work = Path(work)
    stamp = read_json(work/'rule-freeze.json', {})
    return bool(stamp and stamp.get('rules_sha256') == digest(work/'rules.md')
                and stamp.get('source_index_sha256') == digest(work/'source-index.json')
                and stamp.get('pilot_sha256') == pilot_digest(work))


def verify_sources(work):
    work = Path(work)
    index = read_json(work/'source-index.json')
    if digest(Path(index['parent'])/'provenance.json') != index['parent_provenance_sha256']:
        raise ValueError('父版本记录已变化')
    for row in index['rows']:
        for path, sha in ((row['image'], row['sha256']), (row['original_annotation'], row['annotation_sha256']),
                          (row['parent_label'], row['parent_label_sha256']),
                          (Path(index['source'])/'images'/row['split']/Path(row['image']).name, row['sha256'])):
            if digest(path) != sha:
                raise ValueError(f'源文件已变化：{path}')
    return index


def export_version(work, output):
    work, output = resolve(work), resolve(output)
    if output.exists():
        raise ValueError('输出目录已存在，不覆盖旧版本')
    if not frozen(work):
        raise ValueError('50张试标尚未确认并冻结，禁止导出训练版本')
    index = verify_sources(work)
    splits = {s: [] for s in ('train', 'val', 'test')}
    pending, missing, audit = [], [], []
    for row in index['rows']:
        record = read_json(annotation_path(work, row['id']), {})
        status, actor = record.get('status'), record.get('reviewer_type')
        completed = ((status == 'confirmed' and actor == 'human') or
                     (status == 'agent_reviewed' and actor == 'agent') or
                     (status == 'pending' and actor in ('human', 'agent')))
        if not completed:
            missing.append(row['id'])
            continue
        validate_annotation(row, record)
        if record['rules_sha256'] != digest(work/'rules.md') or record['source_image_sha256'] != row['sha256'] or record['source_annotation_sha256'] != row['annotation_sha256']:
            raise ValueError('标注来源/规则版本不匹配')
        audit.append(record)
        if record['status'] == 'pending':
            pending.append({'id': row['id'], 'split': row['split'], 'reason': record['reason']})
            continue
        new = {k: deepcopy(v) for k, v in row.items() if k not in ('polygons', 'parent_label', 'parent_label_sha256')}
        new['boxes'] = [o['box'] for o in record['objects']]
        # Annotation confirmation does not establish material or acquisition-group independence.
        new.update(annotation_review=record, annotation_unit='visible_connected_crack',
                   reviewed=row.get('reviewed', False), surface=row.get('surface', 'unverified'))
        splits[row['split']].append(new)
    if missing:
        raise ValueError(f'仍有{len(missing)}张未形成逐图复核/待定结论，禁止发布全量V2')
    if any(not any(r['boxes'] for r in splits[s]) for s in splits) or sum(bool(r['boxes']) for r in splits['test']) < 50:
        raise ValueError('有效数据数量不足（各划分非空，test至少50张裂缝图）')
    materialize(splits, output, 'user_labeled')
    shutil.copy2(work/'rules.md', output/'annotation-rules.md')
    for name in ('source-index.json', 'pilot.json', 'rule-freeze.json'):
        shutil.copy2(work/name, output/name)
    shutil.copytree(work/'history', output/'review-history')
    write_json(output/'annotation-audit.json', audit)
    manifest = {'created_at': now(), 'annotation_unit': 'visible_connected_crack',
                'parent_provenance_sha256': index['parent_provenance_sha256'],
                'source_index_sha256': digest(work/'source-index.json'),
                'rules_sha256': digest(work/'rules.md'), 'freeze': read_json(work/'rule-freeze.json'),
                'annotation_audit_sha256': digest(output/'annotation-audit.json'),
                'review_history_sha256': fingerprint({p.relative_to(output/'review-history').as_posix(): digest(p)
                                                      for p in sorted((output/'review-history').rglob('*.json'))}),
                'pending': pending, 'missing': missing,
                'counts': {s: {'retained': len(splits[s]), 'pending': sum(p['split'] == s for p in pending),
                               'parent': sum(r['split'] == s for r in index['rows'])} for s in splits},
                'source_files_verified_unchanged': True, 'original_splits_preserved': True,
                'domain_reviewed': False, 'automatic_training_started': False}
    manifest['reviewer_counts'] = dict(Counter(r['reviewer_type'] for r in audit))
    write_json(output/'version.json', manifest)
    checks = dataset_readiness(output)
    if not checks['ready']:
        raise ValueError('V2未通过就绪检查：'+str(checks['gaps']))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('init', 'verify', 'export'))
    p.add_argument('--work', type=Path, default=WORK)
    p.add_argument('--output', type=Path, default=ROOT/'data/prepared/crack-connected-v2')
    args = p.parse_args()
    if args.action == 'init':
        print(f'已索引{len(initialize(args.work))}张；尚未导出训练数据')
    elif args.action == 'verify':
        print(f'原文件摘要一致：{len(verify_sources(args.work)["rows"])}张')
    else:
        print(json.dumps(export_version(args.work, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
