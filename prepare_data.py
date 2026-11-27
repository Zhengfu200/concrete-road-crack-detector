"""Download, import, audit and split data. Candidates never become reviewed automatically."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import random
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml
from PIL import Image

from common import (ROOT, MANIFEST, NAMES, digest, load_manifest, now, read_json,
                    relative, resolve, validate_boxes, write_json)

SOURCES = {
    "crack-seg": {
        "page": "https://docs.ultralytics.com/datasets/segment/crack-seg/",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/crack-seg.zip",
        "license": "See upstream dataset license; retain original attribution",
    },
    "sdnet2018": {
        "page": "https://digitalcommons.usu.edu/all_datasets/48/",
        "url": "https://digitalcommons.usu.edu/cgi/viewcontent.cgi?filename=2&article=1047&context=all_datasets&type=additional",
        "license": "CC-BY-4.0; Maguire, Dorafshan, Thomas (2018), doi:10.15142/T3TD19",
        "mirror": "https://www.kaggle.com/api/v1/datasets/download/aniruddhsharma/structural-defects-network-concrete-crack-images",
        "mirror_page": "https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images",
    },
}


def extract_zip(archive, destination):
    destination = resolve(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(resolve(archive)) as z:
        for entry in z.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"压缩包存在不安全路径: {entry.filename}")
        z.extractall(destination)


def download(source):
    info = SOURCES[source]
    archive = ROOT / "data/raw" / f"{source}.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or not zipfile.is_zipfile(archive):
        part = archive.with_suffix(".zip.part")
        failure = None
        for url in [info["url"], *([info["mirror"]] if info.get("mirror") else [])]:
            print(f"下载 {source}: {url}", flush=True)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as response, part.open("wb") as f:
                    shutil.copyfileobj(response, f, 1024 * 1024)
                if not zipfile.is_zipfile(part):
                    raise ValueError("下载内容不是 ZIP（可能是访问限制页面），未导入。")
                info = {**info, "download_url": url}
                failure = None
                break
            except (OSError, ValueError) as exc:
                failure = exc
                print(f"入口失败: {exc}", flush=True)
        if failure:
            raise failure
        part.replace(archive)
    write_json(archive.with_suffix(".source.json"), {**info, "sha256": digest(archive), "downloaded_at": now()})
    extract_zip(archive, ROOT / "data/raw" / source)
    print(f"已解压: data/raw/{source}")


def polygon_to_box(tokens):
    vals = [float(x) for x in tokens]
    if len(vals) == 5:
        box = vals
    else:
        if len(vals) < 7 or (len(vals)-1) % 2:
            raise ValueError("不是合法的 YOLO 检测框或多边形")
        xs, ys = vals[1::2], vals[2::2]
        box = [vals[0], (min(xs)+max(xs))/2, (min(ys)+max(ys))/2,
               max(xs)-min(xs), max(ys)-min(ys)]
    validate_boxes([box])
    box[0] = int(box[0])
    return box


def source_group(path, source):
    """Only a suggestion: reviewer must confirm original-image/sequence identity."""
    stem = path.stem.split(".rf.")[0]
    stem = re.sub(r"_(?:jpg|png|jpeg)$", "", stem, flags=re.I)
    if source == "sdnet2018":
        stem = re.sub(r"[_-]\d+$", "", stem)
    return f"{source}:{stem}"


def import_images(source, directory, limit=None, source_info=None):
    info = source_info or SOURCES[source]
    manifest = load_manifest()
    known = {r["id"] for r in manifest["records"]}
    images = sorted(p for p in resolve(directory).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if source == "sdnet2018":
        # SDNET's P/PC/PU folders and P-prefixed file names designate pavement.
        images = [p for p in images if p.stem.upper().startswith("P") or
                  any(s.upper() in {"P", "PC", "PU", "PAVEMENTS"} for s in p.parts)]
    added = 0
    for p in images:
        rid = hashlib.sha256(relative(p).encode()).hexdigest()[:20]
        if rid in known:
            continue
        boxes = []
        if source == "crack-seg":
            parts = list(p.parts)
            if "images" not in parts:
                continue
            parts[parts.index("images")] = "labels"
            label = Path(*parts).with_suffix(".txt")
            if not label.exists():
                continue  # A missing label is not automatically a verified negative.
            for line in label.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    box = polygon_to_box(line.split())
                    if box[0] != 0:
                        raise ValueError(f"上游出现未知类别: {label}")
                    boxes.append(box)
        with Image.open(p) as im:
            width, height = im.size
        manifest["records"].append({
            "id": rid, "image": relative(p), "source": source,
            "source_url": info["page"], "license": info["license"],
            "group": source_group(p, source), "group_confirmed": False,
            "reviewed": False, "reviewer": "", "surface": "unknown", "excluded": False,
            "boxes": boxes, "width": width, "height": height, "sha256": digest(p),
            "annotation_origin": "upstream_polygon_to_bbox" if source == "crack-seg" else "unlabelled",
        })
        added += 1
        if limit and added >= limit:
            break
    write_json(MANIFEST, manifest)
    print(f"导入 {added} 张候选；未经复核不进入正式训练。")


def reviewed_records(manifest):
    return [r for r in manifest["records"] if r.get("reviewed") and not r.get("excluded")
            and r.get("surface") == "concrete" and r.get("reviewer", "").strip()]


def audit(manifest):
    errors = []
    ids = set()
    for r in manifest["records"]:
        if r["id"] in ids:
            errors.append(f"重复 ID: {r['id']}")
        ids.add(r["id"])
        try:
            validate_boxes(r["boxes"])
            if digest(r["image"]) != r["sha256"]:
                errors.append(f"图片内容已更改: {r['image']}")
        except (ValueError, OSError) as e:
            errors.append(f"{r['id']}: {e}")
        if r.get("group_confirmed") and not r.get("group", "").strip():
            errors.append(f"确认的来源组为空: {r['id']}")
        if r.get("reviewed") and not r.get("excluded"):
            if r.get("surface") != "concrete" or not r.get("reviewer", "").strip():
                errors.append(f"复核缺少材质或复核人: {r['id']}")
    records = reviewed_records(manifest)
    seen_labels = {}
    for r in records:
        normalized = sorted(tuple(round(v,6) for v in b) for b in r["boxes"])
        if r["sha256"] in seen_labels and seen_labels[r["sha256"]] != normalized:
            errors.append(f"同一图片存在相互冲突的复核标注: {r['id']}。请保留一份一致的标注或排除重复项。")
        seen_labels[r["sha256"]] = normalized
    counts = Counter()
    source_classes = defaultdict(Counter)
    for r in records:
        for c in {b[0] for b in r["boxes"]}:
            counts[NAMES[c]] += 1
            source_classes[r["source"]][NAMES[c]] += 1
    return {"created_at": now(), "candidate_images": len(manifest["records"]),
            "candidate_images_by_source": dict(Counter(r["source"] for r in manifest["records"])),
            "reviewed_concrete_images": len(records), "images_by_class": dict(counts),
            "reviewed_images_by_source_and_class": {s:dict(c) for s,c in source_classes.items()},
            "group_confirmed_images": sum(bool(r.get("group_confirmed")) for r in records),
            "errors": errors}


def split_records(records, seed=42):
    """Union exact duplicates and original groups before splitting; unknown provenance -> train."""
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    seen_group, seen_hash = {}, {}
    for i, r in enumerate(records):
        for key, seen in [(r.get("group") or r["id"], seen_group), (r["sha256"], seen_hash)]:
            if key in seen:
                union(i, seen[key])
            seen[key] = i
    components = defaultdict(list)
    for i, r in enumerate(records):
        components[find(i)].append(r)
    groups = list(components.values())
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    totals = [len(records), *[sum(any(b[0] == c for b in r["boxes"]) for r in records) for c in NAMES]]
    ratios = {"train": .7, "val": .15, "test": .15}
    counts = {s: [0] * (1+len(NAMES)) for s in ratios}
    output = {s: [] for s in ratios}
    for group in groups:
        features = [len(group), *[sum(any(b[0] == c for b in r["boxes"]) for r in group) for c in NAMES]]
        if not all(r.get("group_confirmed") for r in group):
            target = "train"
        else:
            # Select split with largest normalized unmet demand for this group's features.
            target = max(ratios, key=lambda s: sum(
                features[k] * (ratios[s]*totals[k]-counts[s][k]) / max(1, ratios[s]*totals[k])
                for k in range(1+len(NAMES))))
        output[target].extend(group)
        counts[target] = [a+b for a, b in zip(counts[target], features)]
    return output


def assert_split_integrity(splits, require_confirmed=True):
    hashes, groups = {}, {}
    for split, rows in splits.items():
        for r in rows:
            for key, seen in [(r["sha256"], hashes), (r.get("group") or r["id"], groups)]:
                if key in seen and seen[key] != split:
                    raise ValueError(f"来源组或重复图片跨集合: {key}")
                seen[key] = split
            if require_confirmed and split != "train" and not r.get("group_confirmed"):
                raise ValueError("未确认来源组的样本不得用于独立验证/测试")


def materialize(splits, destination, purpose):
    destination = resolve(destination)
    if destination.exists():
        raise ValueError(f"输出目录已存在，避免覆盖旧实验，请使用新的 --output: {destination}")
    assert_split_integrity(splits, require_confirmed=purpose != "user_labeled")
    # A byte-identical image is one observation, even if it arrived through multiple sources.
    splits = {s: list({r["sha256"]: r for r in rows}.values()) for s, rows in splits.items()}
    provenance = {}
    for split, rows in splits.items():
        for folder in [destination/"images"/split, destination/"labels"/split]:
            folder.mkdir(parents=True, exist_ok=True)
        for r in rows:
            validate_boxes(r["boxes"])
            target = destination / "images" / split / f"{r['id']}{resolve(r['image']).suffix.lower()}"
            shutil.copy2(resolve(r["image"]), target)
            label = destination / "labels" / split / f"{r['id']}.txt"
            label.write_text("\n".join(f"{int(b[0])} " + " ".join(f"{x:.8f}" for x in b[1:]) for b in r["boxes"]), encoding="utf-8")
        provenance[split] = rows
    cfg = {"path": str(destination.resolve()), "train": "images/train", "val": "images/val",
           "test": "images/test", "names": NAMES}
    (destination/"data.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(destination/"provenance.json", {"purpose": purpose, "created_at": now(), "splits": provenance})
    readiness = dataset_readiness(destination)
    write_json(destination/"readiness.json", readiness)
    print(f"输出: {destination}\n正式训练就绪: {readiness['ready']}\n" + "\n".join(readiness["gaps"]))


def dataset_readiness(directory):
    directory = resolve(directory)
    info = read_json(directory/"provenance.json")
    if not info:
        raise ValueError("缺少 provenance.json，请用 prepare_data.py export 生成数据集。")
    splits = info["splits"]
    user_labeled = info.get("purpose") == "user_labeled"
    assert_split_integrity(splits, require_confirmed=not user_labeled)
    gaps, stats = [], {}
    config_path = directory/"data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    expected_config = {"path": str(directory.resolve()), "train": "images/train", "val": "images/val",
                       "test": "images/test", "names": NAMES}
    if config != expected_config:
        gaps.append("data.yaml 与复核导出记录不一致，请重新导出，不能替换类别/路径/下载脚本。")
    for s in ["train", "val", "test"]:
        rows = splits.get(s, [])
        unique = list({r["sha256"]: r for r in rows}.values())
        c = [sum(any(b[0] == k for b in r["boxes"]) for r in unique) for k in NAMES]
        stats[s] = {"images": len(rows), "crack_images": c[0]}
        for k in NAMES:
            if not c[k]:
                gaps.append(f"{s} 缺少 {NAMES[k]} 样本")
        unreviewed = 0
        for r in rows:
            validate_boxes(r["boxes"])
            if not r.get("reviewed") or r.get("surface") != "concrete" or not r.get("reviewer", "").strip():
                unreviewed += 1
            img = directory/"images"/s/f"{r['id']}{resolve(r['image']).suffix.lower()}"
            label = directory/"labels"/s/f"{r['id']}.txt"
            if not img.exists() or digest(img) != r["sha256"]:
                gaps.append(f"导出图片缺失/内容更改: {r['id']}")
            if not label.exists():
                gaps.append(f"导出标注缺失: {r['id']}")
            else:
                actual = [polygon_to_box(line.split()) for line in label.read_text().splitlines() if line.strip()]
                if len(actual) != len(r["boxes"]) or any(any(abs(x-y)>1e-6 for x,y in zip(a,b)) for a,b in zip(actual,r["boxes"])):
                    gaps.append(f"导出标注与复核记录不符: {r['id']}")
        if unreviewed and not user_labeled:
            gaps.append(f"{s} 存在 {unreviewed} 张未经复核的混凝土样本")
    for k in ["crack_images"]:
        if stats["test"][k] < 50:
            gaps.append(f"测试集 {k} {stats['test'][k]}/50")
    if info.get("purpose") not in {"formal", "user_labeled"}:
        gaps.append("此数据集仅用于流程测试")
    return {"ready": not gaps, "purpose": info.get("purpose"), "stats": stats,
            "domain_reviewed": not user_labeled,
            "gaps": list(dict.fromkeys(gaps)), "provenance_sha256": digest(directory/"provenance.json")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    dl = sub.add_parser("download"); dl.add_argument("source", choices=SOURCES)
    imp = sub.add_parser("import"); imp.add_argument("source", choices=SOURCES)
    imp.add_argument("--directory"); imp.add_argument("--limit", type=int)
    add = sub.add_parser("add", help="导入其他有来源说明的图片，默认未标注/未复核")
    add.add_argument("--directory", required=True)
    add.add_argument("--source-name", required=True)
    add.add_argument("--source-url", required=True)
    add.add_argument("--license", required=True)
    sub.add_parser("audit")
    ex = sub.add_parser("export"); ex.add_argument("--output", default="data/prepared/formal-crack-v1")
    ex.add_argument("--seed", type=int, default=42)
    smoke = sub.add_parser("smoke"); smoke.add_argument("--output", default="data/prepared/smoke-crack-v1")
    args = ap.parse_args()
    if args.action == "download":
        download(args.source)
    elif args.action == "import":
        import_images(args.source, args.directory or f"data/raw/{args.source}", args.limit)
    elif args.action == "add":
        if args.source_name in SOURCES or not re.fullmatch(r"[a-zA-Z0-9_-]+",args.source_name):
            raise ValueError("来源名称须为新的英文字母/数字/连字符组合，不能冒充已有来源。")
        import_images(args.source_name,args.directory,source_info={"page":args.source_url,"license":args.license})
    elif args.action == "audit":
        report = audit(load_manifest())
        write_json(ROOT/"reports/data_audit.json", report)
        print(__import__("json").dumps(report, ensure_ascii=False, indent=2))
        if report["errors"]:
            raise SystemExit(1)
    else:
        manifest = load_manifest()
        report = audit(manifest)
        if report["errors"]:
            raise ValueError("数据检查失败，请先运行 audit。")
        if args.action == "export":
            rows = reviewed_records(manifest)
            if not rows:
                raise ValueError("尚无已复核混凝土样本；先运行 annotate.py。")
            materialize(split_records(rows, args.seed), args.output, "formal")
        else:
            # Pure pipeline check on real upstream crack labels; no semantic validation claim.
            rows = [r for r in manifest["records"] if r["source"] == "crack-seg" and r["boxes"]][:64]
            if len(rows) < 8:
                raise ValueError("流程测试至少需要 8 张公开裂缝标注图；请先下载并导入。")
            for r in rows:
                r["group_confirmed"] = True  # only within this explicitly non-evaluation smoke set
            groups = defaultdict(list)
            for r in rows:
                groups[r["group"]].append(r)
            if len(groups) < 3:
                raise ValueError("流程测试至少需要三个来源组")
            chunks = list(groups.values())
            splits = {"train": [r for g in chunks[2:] for r in g], "val": chunks[0], "test": chunks[1]}
            # Original groups are kept, and duplicate integrity is still enforced.
            materialize(splits, args.output, "pipeline_smoke")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc))
