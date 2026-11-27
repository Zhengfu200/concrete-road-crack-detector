from copy import deepcopy

import pytest

from common import digest, read_json, save_image, write_json
from connected_annotations import (CATEGORIES, annotation_path, default_objects, export_version,
    freeze_rules, frozen, initialize, save_annotation, union_box, validate_annotation, verify_sources)
from prepare_data import materialize, dataset_readiness


@pytest.fixture
def workspace(tmp_path):
    import numpy as np
    source, parent, work = (tmp_path/n for n in ('source', 'parent', 'work'))
    splits = {}
    polygon = '0 .1 .1 .5 .1 .5 .5 .1 .5'
    for split, count in [('train', 50), ('val', 1), ('test', 51)]:
        splits[split] = []
        for i in range(count):
            image_id = f'{split}-{i}'
            image = source/'images'/split/(image_id+'.png')
            label = source/'labels'/split/(image_id+'.txt')
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(polygon)
            save_image(image, np.full((8, 8, 3), [i+1, len(splits)*30, 10], dtype=np.uint8))
            sha = digest(image)
            splits[split].append({'id': image_id, 'image': str(image), 'boxes': [[0, .3, .3, .4, .4]],
                'sha256': sha, 'annotation_sha256': digest(label), 'group': sha,
                'group_confirmed': False, 'reviewed': False, 'surface': 'unverified'})
    materialize(splits, parent, 'user_labeled')
    rows = initialize(work, parent, source)
    categories = [c for c, n in CATEGORIES.items() for _ in range(n)]
    write_json(work/'pilot.json', {'items': [{'id': r['id'], 'category': c}
               for r, c in zip(rows[:50], categories)]})
    return work, rows


def confirmation(row, **kwargs):
    return {'status': 'confirmed', 'reason': '核对原图：一条可见连通裂缝',
            'objects': default_objects(row), 'excluded': [], 'background_confirmed': False,
            'expected_revision': 0, **kwargs}


def confirm_pilot(work, rows):
    for r in rows[:50]:
        save_annotation(work, r['id'], confirmation(r), 'test human')
    freeze_rules(work, 'test human')


def test_connected_units_cover_segments_forks_and_independent_parallel():
    # Deliberate groups express image-review decisions, never distance-based auto-merge.
    ps = [[[.1,.1],[.2,.1],[.2,.3]], [[.2,.3],[.3,.3],[.3,.5]],
          [[.3,.5],[.5,.5],[.5,.6]], [[.7,.1],[.8,.1],[.8,.5]]]
    row = {'polygons': ps}
    value = {'status':'confirmed','reason':'首三段图像确认连通，右侧独立',
             'objects':[{'polygon_indices':[0,1,2],'box':union_box(ps,[0,1,2]),'reason':'交点可见连通'},
                        {'polygon_indices':[3],'box':union_box(ps,[3])}], 'excluded':[]}
    validate_annotation(row, value)
    assert len(value['objects']) == 2
    # All nine pieces or a Y/T/cross network are the same reviewed-group operation.
    ps9 = ps[:3]*3
    assert union_box(ps9, list(range(9))) == union_box(ps, [0,1,2])


def test_empty_and_pending_are_distinct_and_all_polygons_accounted_for():
    row = {'polygons': [[[.1,.1],[.3,.1],[.3,.3]]]}
    v = {'status':'confirmed','reason':'疑似接缝','objects':[],
         'excluded':[{'polygon_index':0,'reason':'接缝'}]}
    with pytest.raises(ValueError, match='空标注'):
        validate_annotation(row, v)
    v['status']='pending'
    validate_annotation(row, v)
    v['status']='confirmed';v['background_confirmed']=True
    validate_annotation(row, v)
    v['excluded']=[]
    with pytest.raises(ValueError, match='未关联'):
        validate_annotation(row, v)


def test_agent_proposal_cannot_confirm_or_unlock_full_data(workspace):
    work, rows = workspace
    with pytest.raises(ValueError, match='AI'):
        save_annotation(work, rows[0]['id'], confirmation(rows[0]), 'Codex', 'agent')
    with pytest.raises(ValueError, match='试标须用户确认'):
        save_annotation(work, rows[0]['id'], confirmation(rows[0], status='agent_reviewed'), 'Codex', 'agent')
    with pytest.raises(ValueError, match='先完成'):
        save_annotation(work, rows[50]['id'], confirmation(rows[50]), 'human')
    with pytest.raises(ValueError, match='逐张确认'):
        freeze_rules(work, 'human')
    with pytest.raises(ValueError, match='禁止导出'):
        export_version(work, work.parent/'v2')


def test_revision_history_and_freeze_invalidation(workspace):
    work, rows = workspace
    confirm_pilot(work, rows)
    assert frozen(work)
    first = rows[0]
    with pytest.raises(ValueError, match='已被更新'):
        save_annotation(work, first['id'], confirmation(first), 'human')
    save_annotation(work, first['id'], confirmation(first, expected_revision=1), 'human')
    assert not frozen(work)
    assert (work/'history'/first['id']/'00001.json').exists()
    assert (work/'history'/first['id']/'00002.json').exists()


def test_full_export_preserves_splits_sources_and_excludes_pending(workspace):
    work, rows = workspace
    source_before = digest(rows[0]['original_annotation'])
    confirm_pilot(work, rows)
    with pytest.raises(ValueError, match='仍有'):
        export_version(work, work.parent/'v2')
    for r in rows[50:]:
        # One test image is undecidable and must never become a negative sample.
        save_annotation(work, r['id'], confirmation(r, status='pending' if r==rows[-1] else 'agent_reviewed'), 'test agent', 'agent')
    output = work.parent/'v2'
    manifest = export_version(work, output)
    assert manifest['counts']['test'] == {'retained':50,'pending':1,'parent':51}
    assert manifest['reviewer_counts'] == {'human':50,'agent':52}
    assert not (output/'labels/test'/f'{rows[-1]["id"]}.txt').exists()
    assert digest(rows[0]['original_annotation']) == source_before
    info = read_json(output/'provenance.json')
    assert all(r['annotation_unit']=='visible_connected_crack' for s in info['splits'].values() for r in s)
    ready = dataset_readiness(output)
    assert ready['ready'] and not ready['domain_reviewed']
    assert set(r['id'] for r in info['splits']['train']) == set(r['id'] for r in rows[:50])
    with pytest.raises(ValueError, match='已存在'):
        export_version(work, output)


def test_original_label_change_is_detected(workspace):
    work, rows = workspace
    path = __import__('pathlib').Path(rows[0]['original_annotation'])
    path.write_text('')
    with pytest.raises(ValueError, match='源文件已变化'):
        verify_sources(work)
