import argparse
import tkinter as tk
import pytest

from PIL import Image

import annotate
from detect import DetectionWindow


@pytest.fixture(scope="module")
def tk_session():
    root=tk.Tk();root.withdraw()
    yield root
    root.destroy()


def test_annotation_review_requires_explicit_confirmation(tmp_path,monkeypatch,tk_session):
    image=tmp_path/"图像.png"; Image.new("RGB",(80,80),"gray").save(image)
    record={"id":"fixture","image":str(image),"source":"sdnet2018","boxes":[],"reviewed":False,
            "group":"original-1","group_confirmed":False,"surface":"unknown"}
    monkeypatch.setattr(annotate,"load_manifest",lambda:{"records":[record]})
    monkeypatch.setattr(annotate,"MANIFEST",tmp_path/"manifest.json")
    monkeypatch.setattr(annotate.messagebox,"askyesno",lambda *a:True)
    monkeypatch.setattr(annotate.messagebox,"showwarning",lambda *a:None)
    root=tk.Toplevel(tk_session);root.withdraw()
    try:
        ui=annotate.Annotator(root)
        ui.approve()
        assert record["reviewed"] is False
        ui.concrete.set(True);ui.reviewer.set("fixture-reviewer");ui.confirmed.set(True)
        ui.complete.set(True);ui.approve()
        assert record["reviewed"] is True
        assert record["group_confirmed"] is True
        ui.boxes.append([0,.5,.5,.3,.3]);ui.modified();ui.save_draft()
        assert record["reviewed"] is False
    finally:
        root.destroy()


def test_detection_window_constructs_and_stops_without_model(tmp_path,tk_session):
    args=argparse.Namespace(weights=None,source="中文视频.mp4",device="cpu",imgsz=960,
                            allow_unvalidated=False,output=None,max_frames=None,duration=None)
    root=tk.Toplevel(tk_session);root.withdraw()
    try:
        ui=DetectionWindow(root,args)
        root.update_idletasks()
        assert ui.video.get()=="中文视频.mp4"
        assert ui.thresholds=={0:.2}
        ui.stop.set()
        assert ui.stop.is_set()
    finally:
        root.destroy()
