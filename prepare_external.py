"""Prepare the user-selected labels without importing the retired annotation manifest."""
import argparse
import math
from PIL import Image
from common import ROOT, digest, now, write_json
from prepare_data import materialize, polygon_to_box

SOURCE = ROOT.parent / 'dataset_crack-seg/crack-seg'
DEFAULT_DATA = 'data/prepared/crack-seg-user-20261021'


def prepare(source, output):
    splits = {}
    seen = {}
    duplicates = []
    counts = {}
    # Keep held-out images when exact duplicates occur in multiple splits.
    for split in ('test', 'val', 'train'):
        rows = []
        image_dir = source / 'images' / split
        label_dir = source / 'labels' / split
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
        if not images or len({p.stem for p in images}) != len(images):
            raise ValueError(f'Missing images or duplicate stems: {image_dir}')
        if {p.stem for p in images} != {p.stem for p in label_dir.glob('*.txt')}:
            raise ValueError(f'Image/label pairing mismatch: {split}')
        counts[split] = len(images)
        for image in images:
            with Image.open(image) as im:
                im.verify()
            label = label_dir / (image.stem + '.txt')
            boxes = []
            for line in label.read_text(encoding='utf-8-sig').splitlines():
                if not line.strip():
                    continue
                tokens = line.split()
                values = [float(t) for t in tokens]
                if not all(math.isfinite(v) for v in values) or any(v < 0 or v > 1 for v in values[1:]):
                    raise ValueError(f'Invalid coordinates: {label}')
                boxes.append(polygon_to_box(tokens))
            sha = digest(image)
            if sha in seen:
                duplicates.append({'image': str(image), 'kept': seen[sha]})
                continue
            seen[sha] = str(image)
            rows.append({'id': image.stem, 'image': str(image), 'sha256': sha,
                         'boxes': boxes, 'group': sha, 'group_confirmed': False,
                         'reviewed': False, 'surface': 'unverified',
                         'annotation_source': str(label), 'annotation_sha256': digest(label)})
        splits[split] = rows
    materialize(splits, output, 'user_labeled')
    write_json(output / 'import_report.json', {
        'created_at': now(), 'source': str(source), 'original_counts': counts,
        'duplicates_removed': duplicates, 'conversion': 'one bounding box per supplied polygon',
        'notes': 'User-selected labels; upstream split retained except exact duplicates. Material and sequence independence are not manually verified.'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=__import__('pathlib').Path, default=SOURCE)
    p.add_argument('--output', type=__import__('pathlib').Path, default=ROOT / DEFAULT_DATA)
    args = p.parse_args()
    prepare(args.source.resolve(), args.output.resolve())
