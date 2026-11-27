from copy import deepcopy

import pytest
from PIL import Image

from common import digest, read_json, validate_boxes, write_json
from prepare_data import (assert_split_integrity, dataset_readiness, materialize,
                          polygon_to_box, reviewed_records, split_records, extract_zip)


def record(i, **kwargs):
    return {"id": str(i), "group": f"original-{i}", "sha256": f"hash-{i}",
            "group_confirmed": True, "reviewed": True, "surface": "concrete", "reviewer": "test-reviewer",
            "boxes": [[0, .5, .5, .4, .4]], **kwargs}


def test_polygon_conversion_retains_localization():
    assert polygon_to_box("0 .1 .2 .5 .2 .5 .8 .1 .8".split()) == pytest.approx([0,.3,.5,.4,.6])


@pytest.mark.parametrize("box", [[0,.1,.5,.4,.1], [2,.5,.5,.2,.2], [0,.5,.5,0,.2],
                                    [0,float('nan'),.5,.1,.1], [0,.5,.5,1.1,.1]])
def test_invalid_labels_rejected(box):
    with pytest.raises(ValueError):
        validate_boxes([box])


def test_group_and_duplicate_transitive_union():
    rows = [record(i) for i in range(60)]
    rows[1]["group"] = rows[0]["group"]
    rows[2]["sha256"] = rows[1]["sha256"]
    rows[3]["group_confirmed"] = False
    splits = split_records(rows)
    assert_split_integrity(splits)
    membership = {r["id"]: s for s, group in splits.items() for r in group}
    assert membership["0"] == membership["1"] == membership["2"]
    assert membership["3"] == "train"
    assert splits == split_records(rows)
    assert all({b[0] for r in group for b in r["boxes"]} == {0} for group in splits.values())


def test_cross_split_leakage_rejected():
    a = record(0)
    b = record(1, sha256=a["sha256"])
    with pytest.raises(ValueError, match="跨集合"):
        assert_split_integrity({"train": [a], "test": [b]})


def test_unreviewed_or_wrong_material_excluded():
    records = [record(0), record(1, reviewed=False), record(2, surface="asphalt"), record(3, excluded=True)]
    assert [r["id"] for r in reviewed_records({"records": records})] == ["0"]


def test_export_checks_tampering_and_smoke_cannot_pass(tmp_path):
    p = tmp_path/"中文图片.png"
    Image.new("RGB", (30,30), "gray").save(p)
    r = record(0, image=str(p), sha256=digest(p))
    out = tmp_path/"dataset"
    materialize({"train": [r], "val": [], "test": []}, out, "pipeline_smoke")
    result = dataset_readiness(out)
    assert not result["ready"]
    assert "此数据集仅用于流程测试" in result["gaps"]
    (out/"labels/train/0.txt").write_text("0 .5 .5 .2 .2")
    assert any("不符" in s for s in dataset_readiness(out)["gaps"])


def test_zip_cannot_escape_destination(tmp_path):
    import zipfile
    archive = tmp_path/"bad.zip"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("../escaped.txt", "bad")
    with pytest.raises(ValueError):
        extract_zip(archive, tmp_path/"out")
    assert not (tmp_path/"escaped.txt").exists()


def test_formal_ready_needs_only_crack_and_untampered_yaml(tmp_path):
    splits={"train":[],"val":[],"test":[]}
    index=0
    for split,count in [("train",140),("val",30),("test",50)]:
        for c in [0]:
            for _ in range(count):
                p=tmp_path/f"image-{index}.png"
                Image.new("RGB",(8,8),(index%256,index//256,100)).save(p)
                r=record(index,image=str(p),sha256=digest(p),boxes=[[c,.5,.5,.4,.4]])
                splits[split].append(r);index+=1
    out=tmp_path/"formal"
    materialize(splits,out,"formal")
    assert dataset_readiness(out)["ready"]
    import yaml
    p=out/"data.yaml"
    cfg=yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["names"]={0:"person",1:"car"}
    p.write_text(yaml.safe_dump(cfg),encoding="utf-8")
    assert any("data.yaml" in gap for gap in dataset_readiness(out)["gaps"])


def test_removed_class_rejected_by_annotation_schema():
    with pytest.raises(ValueError,match="非法类别"):
        validate_boxes([[1,.5,.5,.4,.4]])


def test_formal_training_stops_before_gpu_when_labels_unready(tmp_path,monkeypatch):
    import argparse
    import train
    monkeypatch.setattr(train,"ROOT",tmp_path)
    monkeypatch.setattr(train,"dataset_readiness",lambda p:{"ready":False,"gaps":["缺少已复核裂缝"]})
    with pytest.raises(ValueError,match="缺少已复核裂缝"):
        train.train(argparse.Namespace(data=str(tmp_path),smoke=False))
    assert (tmp_path/"reports/training_readiness.json").exists()
