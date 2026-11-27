"""Build auditable image-review sheets and mine ONLY the current training split."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import random

import cv2
import numpy as np

from common import ROOT, digest, now, read_image, read_json, save_image, write_json
from metrics import class_metrics, iou, xywh_to_xyxy
from preprocessing import predict_frame, suppress_duplicates

OUTPUT = ROOT/'reports/hard-example-review-20261024'
DATA = ROOT/'data/prepared/crack-seg-user-20261021'
WEIGHTS = ROOT/'runs/concrete-crack-20261021-212347/weights/best.pt'
THRESHOLD = .1438644826412201
NMS_IOU = .3


def inspect_sample(sample):
    kept = suppress_duplicates(sample['pred'], NMS_IOU)
    selected = [p for p in kept if p[5] >= THRESHOLD]
    used, matched, false_predictions = set(), [], []
    for pred in sorted(selected, key=lambda p: p[5], reverse=True):
        score, j = max(((iou(pred[1:5], g[1:5]), j) for j, g in enumerate(sample['gt']) if j not in used),
                       default=(0, -1))
        if score >= .5:
            used.add(j); matched.append(j)
        else:
            false_predictions.append(pred)
    fn = []
    for index, gt in enumerate(sample['gt']):
        if index in used:
            continue
        before = max((p[5] for p in sample['pred'] if iou(p[1:5], gt[1:5]) >= .5), default=0)
        after = max((p[5] for p in kept if iou(p[1:5], gt[1:5]) >= .5), default=0)
        overlap = max((iou(p[1:5], gt[1:5]) for p in selected), default=0)
        cause = ('nms_removed_candidate' if before >= THRESHOLD and after < THRESHOLD else
                 'below_threshold_candidate' if 0 < after < THRESHOLD else
                 'localization_or_instance_granularity_candidate' if overlap >= .1 else
                 'no_matching_box_candidate')
        fn.append({'gt_index': index, 'geometric_candidate': cause, 'max_score_before_nms': before,
                   'max_score_after_nms': after, 'max_iou_at_operating_point': overlap})
    return {'id': sample['id'], 'image': sample['image'], 'tp': len(matched), 'fp': len(false_predictions),
            'fn': len(fn), 'gt_boxes': len(sample['gt']), 'fn_candidates': fn,
            'reviewer': None, 'visual_review_status': 'pending'}


def make_sheets(samples, ids, folder, prefix):
    folder.mkdir(parents=True, exist_ok=True)
    lookup = {s['id']: s for s in samples}
    order = []
    for index, sample_id in enumerate(ids, 1):
        sample = lookup[sample_id]
        image = read_image(sample['image'])
        if image.shape[:2] != (416, 416):
            raise ValueError('Review sheets expect original416 images')
        overlay = image.copy()
        for gi, gt in enumerate(sample['gt']):
            _, x1, y1, x2, y2 = gt
            a, b = (round(x1*416), round(y1*416)), (round(x2*416), round(y2*416))
            cv2.rectangle(overlay, a, b, (0, 220, 0), 1)
            cv2.putText(overlay, f'G{gi}', (a[0], max(12, a[1])), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 180, 0), 1)
        for pred in suppress_duplicates(sample['pred'], NMS_IOU):
            if pred[5] < THRESHOLD:
                continue
            _, x1, y1, x2, y2, score = pred
            cv2.rectangle(overlay, (round(x1*416), round(y1*416)), (round(x2*416), round(y2*416)), (0, 0, 255), 1)
            cv2.putText(overlay, f'{score:.2f}', (round(x1*416), max(12, round(y1*416)-2)), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 255), 1)
        card = np.full((448, 832, 3), 245, np.uint8)
        card[32:, :416] = image;card[32:, 416:] = overlay
        short = sample_id.split('.rf.')[0]
        counts = inspect_sample(sample)
        caption = f'{prefix}{index:02d} id={short}  TP={counts["tp"]} FP={counts["fp"]} FN={counts["fn"]}  left=raw right=GTgreen/predRed'
        cv2.putText(card, caption, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1)
        order.append({'review_index': f'{prefix}{index:02d}', **counts})
        sheet_number, slot = divmod(index-1, 6)
        if slot == 0:
            sheet = np.full((448*3, 832*2, 3), 255, np.uint8)
        y, x = divmod(slot, 2)
        sheet[y*448:(y+1)*448, x*832:(x+1)*832] = card
        if slot == 5 or index == len(ids):
            save_image(folder/f'{prefix}-sheet-{sheet_number+1:02d}.jpg', sheet)
    write_json(folder/'review-order.json', order)
    return order


def prepare_val():
    samples = read_json(ROOT/'reports/concrete-crack-20261021-212347/unified-preprocessing-20261024/predictions.json')
    rows = [inspect_sample(s) for s in samples]
    errors = [r for r in rows if r['fp'] or r['fn']]
    selected = []
    def take(pool, count):
        for row in pool:
            if row['id'] not in selected:
                selected.append(row['id']);count -= 1
                if count == 0:
                    break
    take(sorted(errors, key=lambda r: (r['fn'], r['fp']), reverse=True), 20)
    take(sorted(errors, key=lambda r: (r['fp'], r['fn']), reverse=True), 12)
    remaining = [r for r in errors if r['id'] not in selected]
    random.Random(42).shuffle(remaining)
    take(remaining, 16)
    selected = selected[:48]
    make_sheets(samples, selected, OUTPUT/'val-review', 'V')
    write_json(OUTPUT/'val-geometric-audit.json', {'split': 'val', 'threshold': THRESHOLD, 'nms_iou': NMS_IOU,
              'images': rows, 'selected_ids': selected, 'geometric_tags_are_not_visual_diagnoses': True})
    print('VAL_REVIEW_SHEETS', len(selected), flush=True)


def mine_train():
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    provenance = read_json(DATA/'provenance.json')
    samples, rows = [], []
    for index, row in enumerate(provenance['splits']['train'], 1):
        path = DATA/'images/train'/f"{row['id']}{Path(row['image']).suffix.lower()}"
        sample = {'id': row['id'], 'image': str(path), 'gt': [xywh_to_xyxy(b) for b in row['boxes']],
                  'pred': predict_frame(model, read_image(path), 960, '0', conf=.001, normalized=True)}
        samples.append(sample);rows.append(inspect_sample(sample))
        if index % 200 == 0:
            write_json(OUTPUT/'train-mining-progress.json', {'status': 'running', 'processed': index,
                       'total': len(provenance['splits']['train']), 'at': now()})
            print('TRAIN_MINED', index, flush=True)
    write_json(OUTPUT/'train-predictions.json', samples)
    write_json(OUTPUT/'train-geometric-audit.json', rows)
    # Review train failures spanning thin/small, fragmented, multiple-crack, and spurious boxes.
    lookup = {s['id']: s for s in samples}
    selected = []
    def take(pool, n):
        for r in pool:
            if r['id'] not in selected:
                selected.append(r['id']);n -= 1
                if n == 0:
                    break
    # At most4GTs prioritizes interpretable label units for train enrichment, not ambiguous fragmentation.
    eligible = [r for r in rows if 1 <= r['gt_boxes'] <= 4 and (r['fn'] or r['fp'])]
    take(sorted(eligible, key=lambda r: (r['fn'], r['fp']), reverse=True), 24)
    take(sorted(eligible, key=lambda r: (r['fp'], r['fn']), reverse=True), 12)
    remaining = [r for r in eligible if r['id'] not in selected]
    random.Random(43).shuffle(remaining)
    take(remaining, 12)
    make_sheets(samples, selected, OUTPUT/'train-review', 'T')
    write_json(OUTPUT/'train-mining-progress.json', {'status': 'completed', 'processed': len(rows),
              'total': len(rows), 'at': now(), 'weights_sha256': digest(WEIGHTS),
              'provenance_sha256': digest(DATA/'provenance.json'), 'split': 'train'})
    print('TRAIN_REVIEW_SHEETS',len(selected),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['val', 'train'])
    args = parser.parse_args()
    prepare_val() if args.action == 'val' else mine_train()
