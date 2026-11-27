"""Record Codex image inspections of the saved original/GT/prediction sheets."""
from collections import Counter
import csv

from common import ROOT, now, read_json, write_json

OUT = ROOT/'reports/hard-example-review-20261024'

VAL = [
('granularity','同一条斜向裂缝被分为9框，整体和局部预测与分段单位不一致。'),
('granularity','连续弯折裂缝分为6框，模型整体框覆盖裂缝但不能匹配各小段。'),
('granularity','粗糙面裂缝分为长框和端部小框，预测采用不同分段。'),
('granularity','右侧细裂缝分为3段，左侧主裂缝可检出，右侧还有低对比问题。'),
('localization','两条纵向裂缝的预测仅覆盖局部；右支较细，原框还包含下方分支。'),
('localization','交叉裂缝的大框与模型细长局部框不一致，右下细支未检出。'),
('granularity','一条连续竖向细裂缝两段标注，模型整体框覆盖两段。'),
('granularity','墙面接合处连续裂缝仅标局部小段，模型检测为整条长框。'),
('localization','纵向GT很窄，预测同时覆盖横向支裂缝，框宽明显偏大。'),
('localization','纵向裂缝分段与模型预测起止位置不同，下端很细。'),
('granularity','斜向连续裂缝两个重叠大框与局部预测尺度不同。'),
('localization','粗纹理中的纵向裂缝预测比细长GT更宽、更长。'),
('granularity','主竖裂缝及右支被两个分段大框覆盖，预测单位不同。'),
('localization','明显竖裂缝已被看到，预测宽度大于极窄GT，导致IoU不足。'),
('localization','竖裂缝与横向细支造成GT大框，预测仅覆盖竖裂缝。'),
('localization','明显粗竖裂缝预测与GT均沿同一路径，主要为框宽偏差。'),
('granularity','Y形连通裂缝存在整体与局部GT重叠，模型仅保留整体框。'),
('low_contrast_or_small_miss','右侧明显裂缝检出，左侧低对比细裂缝未输出。'),
('granularity','主竖裂缝检出，下段小分支与主体的实例划分不同。'),
('low_contrast_or_small_miss','浅色弥散网状细裂缝未输出，存在真实低对比检测困难。'),
('annotation_coverage_review','除已标主裂缝外，左侧剥落边界和细裂纹也有预测，需要判定标注覆盖范围。'),
('partial_duplicate','主裂缝和右侧裂纹已有匹配，额外局部框仍覆盖裂缝的一部分。'),
('partial_duplicate','整体V形裂缝已匹配，左右端部的局部预测计为额外框。'),
('annotation_coverage_review','GT为空，但原图中存在明显横向及纵向裂缝；标注覆盖异常，暂不修改val。'),
('joint_or_texture_fp','右侧目标裂缝检出，左侧深色接缝及其边缘产生额外框。'),
('joint_or_texture_fp','左侧细裂缝已检出，右边水平色痕/纹理产生额外框。'),
('partial_duplicate','主竖裂缝和横支已检出，额外大框覆盖主裂缝与背景。'),
('partial_duplicate','竖裂缝已有整体框，额外下半段框造成重复计数。'),
('partial_duplicate','T形裂缝整体框已匹配，竖向子框仍保留。'),
('partial_duplicate','已标横裂缝部分匹配，额外局部框嵌于整体范围。'),
('partial_duplicate','长斜裂缝已有整体框，右侧端部再次输出小框。'),
('partial_duplicate','长斜裂缝已有匹配，右上端部额外局部框。'),
('annotation_coverage_review','横裂缝有标注，弱纵向线状结构及整体大框需确认是否为未标裂缝或纹理。'),
('low_contrast_or_small_miss','大横裂缝检出，下方短小竖裂缝/分支未检出。'),
('partial_duplicate','分叉裂缝整体框匹配，右上支裂缝又输出局部框。'),
('low_contrast_or_small_miss','明显横裂缝检出，向上的浅细支裂缝未检出。'),
('partial_duplicate','竖裂缝整体检出，下端局部框重复。'),
('partial_duplicate','T形裂缝整体框检出，竖向子框重复。'),
('granularity','粗斜裂缝主要段检出，下边缘的独立极小GT段未匹配。'),
('granularity','大分叉裂缝检出，顶部细竖裂缝存在极小分段标注。'),
('partial_duplicate','长折线裂缝和右端检出，边缘还存在低分局部框。'),
('partial_duplicate','十字裂缝整体GT匹配，额外竖向子框重复。'),
('joint_or_texture_fp','主竖裂缝检出，左侧短水平纹理/接合痕形成额外框。'),
('granularity','粗竖裂缝主段检出，下端独立极小段未匹配。'),
('low_contrast_or_small_miss','两条平行裂缝中上方粗裂缝检出，下方细裂缝漏检。'),
('partial_duplicate','长斜裂缝整体匹配，左上端局部框重复。'),
('granularity','两条主支检出，交叉处短小GT段单独计数。'),
('low_contrast_or_small_miss','粗横裂缝检出，右上细分支未单独检出。'),
]

TRAIN_NOTES = [
'粗纹理分叉裂缝4个局部GT与模型整体框不一致。',
'同一165来源的另一增强，粗纹理和分段尺度问题相同。',
'细长低对比裂缝分4段，原框有图像依据。',
'同一1875来源另一增强，局部GT与长框不一致。',
'横向裂缝分4个局部GT，预测整体框。',
'同一2221来源旋转增强，连续横裂缝分段。',
'同一2221来源另一增强，连续横裂缝分段。',
'交叉裂缝中粗横支与极细竖支并存，细支低置信度。',
'暗色背景下竖裂缝分4小段，整体预测覆盖实际裂缝。',
'线缆遮挡粗裂缝，局部真裂缝与线缆外观混淆。',
'两条相近细裂缝，分段GT与局部预测不一致。',
'黄色表面斜裂缝分段，整体框加局部框造成多计。',
'粗细分叉裂缝并存，上部细支低置信度，局部框边界不同。',
'165来源第三种增强，与前两个属于相近增强家族。',
'细长主裂缝和上部极细段，部分低对比标签难检。',
'1651来源的另一增强，沿长裂缝分出很小局部段。',
'粗纹理主竖裂缝加短细侧支，存在真小支漏检和分段问题。',
'1729来源另一增强，主竖裂缝被整体预测。',
'1875来源第三种增强，浅横裂缝分4段。',
'多条竖向及横向裂缝，主干可见，细支和分段困难。',
'低对比弯曲裂缝和极小边缘GT，局部框与整体框不一致。',
'1985来源另一增强，同一弯曲线多段GT。',
'主竖裂缝与较浅右支交叉，子框和主框的粒度不一致。',
'粗糙表面横裂缝分3段，整体预测未匹配局部GT。',
'两条平行裂缝，较粗下线出现多个局部预测。',
'墙面色带旁两条浅平行裂缝，长框与局部预测不匹配。',
'1858来源另一增强，平行裂缝及色带使定位困难。',
'剥落/接合边缘与细横支并存，局部预测边界偏差。',
'竖主裂缝与右上支裂缝，有图像依据但预测起止偏差。',
'2265来源另一增强，细横支分段及低置信度。',
'细长裂缝两个首尾GT，模型长框和局部框混合。',
'T形细裂缝，整体大框与细长分支标注尺度不同。',
'主竖裂缝已匹配，上下子框及边缘线状结构产生多框。',
'1981来源另一增强，分叉处和竖段定位不准。',
'细裂缝分叉结构的两个GT，模型用过大整体框覆盖。',
'2150来源另一增强，相邻细线分段定位困难。',
'粗纹理单横裂缝检出，另有极小局部段未匹配。',
'两条相近浅横裂缝，左侧长细目标置信度不足。',
'主横裂缝检出，下方弱竖线可能是未标裂缝或接合痕，需确认，暂不加标。',
'Y形裂缝整体检出，顶部边缘极小GT未匹配。',
'粗斜裂缝主体检出，左端局部GT未匹配。',
'T形裂缝整体匹配，右侧竖支重复框。',
'长浅横裂缝已匹配，右端额外局部框。',
'图像存在清晰斜裂缝，完整GT有依据，但当前工作点没有检出。',
'浅横裂缝被标为两段，模型输出一条长框。',
'复杂条纹背景中的细竖裂缝，主段匹配，末端小段漏检。',
'横向细裂缝与粗纹理相似，下方目标漏检、上方另有弱线候选。',
'明显横裂缝匹配，下方竖向表面色痕产生额外候选。',
]


def main():
    results = {}
    for split, prefix, notes in [('val', 'V', VAL), ('train', 'T', TRAIN_NOTES)]:
        order = read_json(OUT/f'{split}-review/review-order.json')
        if len(order) != len(notes):
            raise ValueError('Review notes do not match viewed sheets')
        rows = []
        for row, note in zip(order, notes):
            category, explanation = note if split == 'val' else ('training_hard_case', note)
            rows.append({**row, 'visual_review_status': 'agent_inspected',
                'reviewer': 'Codex image inspection (not independent human review)',
                'reviewed_at': now(), 'primary_category': category, 'visual_observation': explanation,
                'labels_modified': False, 'formal_material_review': False})
        write_json(OUT/f'{split}-visual-review.json', rows)
        with (OUT/f'{split}-visual-review.csv').open('w',encoding='utf-8-sig',newline='') as f:
            keys = ['review_index','id','tp','fp','fn','primary_category','visual_observation','labels_modified','reviewer']
            writer = csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
        results[split] = rows
    val_audit = read_json(OUT/'val-geometric-audit.json')['images']
    counts = Counter(r['primary_category'] for r in results['val'])
    geometric = Counter(f['geometric_candidate'] for r in val_audit for f in r['fn_candidates'])
    # Keep just one reviewed augmentation per source prefix; do not call this a verified acquisition group.
    hard, families = [], set()
    for row in results['train']:
        family = row['id'].split('.rf.')[0]
        if family not in families and row['review_index'] != 'T39':
            hard.append(row['id']);families.add(family)
    write_json(OUT/'hard-selection.json', {'ids':hard,'weight':3., 'split':'train',
        'selection':'one agent-inspected example per filename augmentation prefix; original GT units retained',
        'augmentation_prefix_is_not_independent_source_certification':True,
        'excluded_pending_label_case':'T39', 'num_reviewed_images':48,'num_selected_images':len(hard)})
    summary = {'val_images_inspected':48,'train_images_inspected':48,
        'val_total_error_images':sum(bool(r['fp'] or r['fn']) for r in val_audit),
        'targeted_sample_not_random_population_estimate':True,'val_primary_categories':dict(counts),
        'val_all_fn_geometric_candidates':dict(geometric),'hard_train_selected':len(hard),
        'labels_changed':0,'validation_labels_frozen':True,'test_images_reviewed':0,
        'formal_domain_review_complete':False}
    write_json(OUT/'review-summary.json',summary)
    lines=['# 错误样例与训练困难样本复核','',
        '已逐张查看48张验证错误图、48张训练困难图的原图与GT/预测叠加图。复核者为Codex图像检查，非独立人工验收。',
        '验证集共有52张在当前工作点出现FP或FN，本次覆盖其中48张；定向选样不是总体随机抽样。','',
        '| 验证样例主类 | 图片数 |','|---|---:|']
    chinese={'granularity':'实例/分段尺度不一致','localization':'检测框定位与边界偏差',
             'low_contrast_or_small_miss':'低对比/细小裂缝漏检','annotation_coverage_review':'标注覆盖需复核',
             'partial_duplicate':'整体与局部框重复','joint_or_texture_fp':'接缝/纹理误报'}
    lines += [f'| {chinese[key]} | {value} |' for key,value in counts.items()]
    lines += ['',f'全部200张验证图的51个FN几何候选原因：{dict(geometric)}。这些自动几何标签不是独立视觉因果诊断。','',
        'V24原标注为空但原图有明显交叉裂缝；V21/V33覆盖范围仍需确认。分段GT与整体预测不一致，不等于标注错误。',
        '当前原数据按每个分割多边形取外接框；本实验保留该检测单位，不将相邻框自动合并。',
        '没有足够证据支持批量改写训练GT，故实际改标0张。T39弱竖线属于未决项目，不进入困难加权池。',
        f'从训练48张中按文件名前缀避免重复增强家族，选出{len(hard)}张有原标注依据的困难样本，提高抽样权重；不伪增独立数据量。',
        '本轮对照在同一640张训练图片、相同初始化/配置/种子下，只比较均匀权重与困难样本3倍权重。val/test图片、标注保持原样。',
        '测试图片未参与视觉复核或调参；正式混凝土道路材质、拍摄来源和正常路面覆盖仍未完成。','',
        '方法参考：scientific-critical-thinking、experimental-design技能；Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). '
        '[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。']
    (OUT/'review-report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(summary,flush=True)


if __name__ == '__main__':
    main()
