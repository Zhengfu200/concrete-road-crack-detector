"""Record the 50 individually inspected pilot proposals, without human certification."""
from collections import Counter

from common import ROOT, digest, now, read_image, read_json, save_image, write_json
from connected_annotations import WORK, annotation_path, save_annotation, union_box

# Each entry corresponds to the frozen selection order in reports/.../pilot-selection.json.
# None of these group choices is inferred from box overlap or model predictions.
# None means all supplied polygons, after inspecting that particular original and polygon sheet.
PROPOSALS = [
    (None, False, '黄色表面同一条竖向裂缝被拆为4段，原图各段沿可见主干连续。'),
    (None, True, '主横裂缝连续，左侧细支也接入；原多边形止于图像中部，右半段漏标。需重绘整体框覆盖可见右端。'),
    (None, False, '黄色表面横向弯折主裂缝的3个首尾段连续，合为整体。'),
    ([[0,1,2],[3]], True, '竖向主干3段连续；P3大幅向右伸出的多边形在原图中缺少清楚依据，暂不并入，需判断排除还是补全弱支。'),
    (None, False, '3个纵向多边形属于同一条从上到下的弯曲裂缝，合为整体。'),
    (None, False, '两个原多边形几乎重复覆盖同一条横裂缝，按同一实例合并。'),
    (None, False, '黄色表面的主横裂缝与很浅下支在中部接合，原图连续，保留整个可见网络。'),
    (None, False, '左端与右侧较长多边形沿同一条折线裂缝连续，合为整体。'),
    (None, False, '白色表面左段和右段沿同一条浅横裂缝连续，合为整体。'),
    (None, False, '主竖裂缝与底端短段连续，合为整体。'),
    (None, False, '左端较粗短段与向右延伸的主裂缝相接，合为整体。'),
    (None, False, '浅灰表面的两段曲折横裂缝在原图相接，合为整体。'),
    ([[1],[0]], True, '主裂缝P1可见，右下角孤立P0缺少清楚图像依据；不能因同图有裂缝就合并该角落。需复核P0及主干遗漏范围。'),
    (None, False, '上端短段与全长纵向裂缝连续；极细斜向白痕不作为裂缝补入。'),
    ([[1],[0]], True, '竖主干跨水平表面接合带后的位置有偏移，原标签下端短段是否属于同一可见裂缝不明确，保留两组待判。'),
    (None, False, '浅色表面的长斜裂缝两段首尾连接，合为整体。'),
    (None, False, '同一条斜裂缝右半段被拆成多个极短段，5个多边形合为整体。'),
    (None, False, '上段和下段属于同一条竖向弯折裂缝，原图中间连续。'),
    (None, False, '绿色表面顶端极短段与下方长裂缝连续，合为整体。'),
    (None, False, '绿色表面一条长斜裂缝的4个首尾段在图像中连续，合为整体。'),
    (None, False, '主横裂缝和细竖裂缝交点可见连通，4个多边形按十字网络整体标。'),
    ([[0,1,2],[3]], True, '主竖干与横支可见接合；左上独立短竖段P3与横支之间存在缺口，需确认是否真连通及右上主干覆盖。'),
    (None, True, '粗横裂缝的3段可合并，但原图还可见向下的细支，原多边形未覆盖；需补全后重绘整个网络框。'),
    (None, True, 'T形交点清楚，主横支与下竖支属于整体；原横支多边形只到中部，右半段未覆盖，需扩大到可见右端。'),
    ([[0]], True, '原标签仅横裂缝，但原图下方细竖支与主干相交；需补全该支后更新整体框，不能直接沿用原横向框。'),
    (None, False, 'Y形三条支路交点清晰，顶部短段接入主干，按整个Y形网络一个框。'),
    (None, False, '原单多边形已包括主横裂缝和下竖支，按可见T形整体保留一个框。'),
    (None, True, '原多边形均沿横裂缝；原图明显向上的弯曲竖支未覆盖，需补全并重绘T形整体框。'),
    (None, False, '主横支与下竖支交点清晰，底端短多边形与竖支连续，合为T形整体。'),
    (None, False, '主竖裂缝和横向细支在中心相接，左侧短段接入横支，整体一个框。'),
    ([[0],[1]], True, '两条平行裂缝在画面内独立，保持两个实例；上方裂缝原多边形止于中部，需补足可见右端，不能只保留半条。'),
    ([[0],[1]], False, '色带上方两条平行裂缝之间无可见连接，保留两个独立实例。'),
    ([[0,1,2,3],[4,5,6]], False, '黄色表面两条独立竖裂缝，右侧4段、左侧3段各自连续；合为两个实例，不能合为一个大框。'),
    ([[1],[2],[0]], True, '两条竖向裂缝独立，分别保留；底边P0很短的横向标签在原图依据不清，需核对是否裂缝及是否漏标。'),
    ([[0],[1]], False, '两条弯曲纵向裂缝在当前画面可见范围内不相接，分别保留。'),
    ([[0],[1]], True, '两条斜向近平行裂缝应保持两个实例；上方裂缝原标注只覆盖右侧一截，需重绘覆盖完整可见部分。'),
    ([[0],[1,2,3,4]], False, '上方粗裂缝与下方细裂缝之间无可见连接；下方4段沿同一条线连续，合成第二个整体。'),
    ([[0],[1]], False, '主竖裂缝含下方侧支；右上短裂缝与主干之间留有明显间隔，暂按两个独立可见整体保留。'),
    ([[0,1],[2]], True, '上下主竖段原图可跟踪连续；左下孤立斜段P2图像依据较弱，不能据距离与主竖裂缝合并，需确认。'),
    ([[0,1],[2,3]], True, '上部主裂缝分段可合并，下方裂纹网络另分一组；两组之间连通关系及左端遗漏支路需人工核对，不先推断连接。'),
    ([[1,2,3],[0]], True, '浅色表面有两条极弱线状裂缝候选；左线的3个首尾段、右线P0暂分别分组。左线间断处连通和污痕干扰需确认。'),
    ([[0,1,2],[3]], False, '左侧全长弱裂缝与右侧较短弱裂缝可分别追踪，未见清晰交点；右侧3段合并，保持两个可见整体。'),
    ([[0,1,2],[3]], True, '浅横裂缝3段在原图可跟踪连续；左上角P3短标签无清楚依据，需判定排除或补标，暂不合并。'),
    (None, False, '一条细弱竖裂缝的上下两个多边形连续，合为整体。'),
    ([[0]], True, '原横向细裂缝可见，但原多边形右端提前终止；需确定最细右端与表面纹理分界，重绘到可确认端点。'),
    ([[3,2,1],[0]], True, '左右两条可见段之间存在较大断口，需判断纹理中是否仍可跟踪裂缝；暂不跨缺口推断，保留两组待定。'),
    ([[1,2],[0],[3]], True, '线缆遮挡主裂缝，左侧主干与下支相接；遮挡两侧是否按同一可见整体及右侧细线P3的连接关系不清，需判定。'),
    ([[0]], True, '粗横裂缝伸出左右边缘，原标签已覆盖主干；下方还有未标弱竖裂缝候选，需判定并补框，不能当作完整标注。'),
    (None, True, '横裂缝两段在瓶盖左侧连续；上下纵向线状结构被瓶盖遮挡，连接和裂缝属性无法确认，整图待定。'),
    ([[0],[1]], True, '主竖裂缝穿过明暗带；左上短支与主干的接合处位于深阴影，需确认连通，并核对下边缘细支是否漏标。'),
]
EXPECTED_CANDIDATES = [5,7,8,14,64,24,25,28,32,33,37,38,39,41,43,44,45,48,51,59,
    4,12,15,21,26,27,29,47,61,67,16,17,49,54,79,135,148,19,11,1,2,10,13,20,30,3,6,35,63,80]


def main():
    import cv2
    import numpy as np
    pilot = read_json(WORK/'pilot.json')
    rows = {r['id']:r for r in read_json(WORK/'source-index.json')['rows']}
    if len(pilot['items']) != len(PROPOSALS) or [x['candidate_index'] for x in pilot['items']] != EXPECTED_CANDIDATES:
        raise ValueError('提案与试标清单不一致')
    out = ROOT/'reports/connected-v2-pilot-20261025'
    snapshots = []
    for item, (groups, pending, why) in zip(pilot['items'], PROPOSALS):
        row = rows[item['id']]
        if annotation_path(WORK, row['id']).exists():
            raise ValueError('已有复核记录，不覆盖')
        groups = groups if groups is not None else [list(range(len(row['polygons'])))]
        record = save_annotation(WORK, row['id'], {'status':'pending' if pending else 'draft',
            'reason':why, 'objects':[{'polygon_indices':ids,'box':union_box(row['polygons'],ids),
                'reason':why} for ids in groups], 'excluded':[], 'background_confirmed':False,
            'expected_revision':0}, 'Codex image inspection; not human confirmation', 'agent')
        snapshots.append({**item, 'record':record, 'before_boxes':row['boxes']})
        image=read_image(row['image']);before=image.copy();after=image.copy()
        for boxes,canvas,color in ((row['boxes'],before,(0,200,255)),
                    ([o['box'] for o in record['objects']],after,(0,170,255) if pending else (80,240,90))):
            for i,b in enumerate(boxes):
                x,y,w,h=[v*416 for v in b[1:]]
                cv2.rectangle(canvas,(round(x-w/2),round(y-h/2)),(round(x+w/2),round(y+h/2)),color,1)
                cv2.putText(canvas,str(i+1),(max(0,round(x-w/2)),max(14,round(y-h/2))),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
        joined=np.concatenate([before,after],axis=1)
        save_image(out/'before-after'/f'{item["pilot_index"]}.jpg',joined)
    write_json(out/'agent-proposals.json', snapshots)
    write_json(out/'pilot-status.json', {'created_at':now(), 'images':50, 'unique_train_families':50,
        'categories':dict(Counter(x['category'] for x in pilot['items'])),
        'draft_proposals':sum(r['record']['status']=='draft' for r in snapshots),
        'pending_cases':[{'pilot_index':r['pilot_index'],'id':r['id'],'reason':r['record']['reason']}
                         for r in snapshots if r['record']['status']=='pending'],
        'human_confirmed':0,'rules_frozen':False,'v2_exported':False,'training_started':False,
        'model_predictions_used_for_first_pass':False,'test_images_visually_reviewed':0,
        'source_index_sha256':digest(WORK/'source-index.json'),'pilot_sha256':digest(WORK/'pilot.json')})
    print('50 image-specific AI proposals saved, zero human confirmations')


if __name__ == '__main__':
    main()
