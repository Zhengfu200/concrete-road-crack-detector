"""Exercise actual checkpoint -> worker -> Tk renderer on a public-image engineering clip."""
import argparse
from pathlib import Path
import sys
import time
import tkinter as tk

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,write_json
from detect import DetectionWindow


def main():
    args=argparse.Namespace(weights="runs/pipeline-smoke-crack/weights/last.pt",
        source=str(ROOT/"outputs/engineering-input-720p.mp4"),device="0",imgsz=960,
        allow_unvalidated=True,output=None,max_frames=8,duration=None)
    root=tk.Tk();root.withdraw()
    app=DetectionWindow(root,args)
    app.start()
    deadline=time.perf_counter()+45
    try:
        while time.perf_counter() < deadline:
            root.update()
            if app.thread and not app.thread.is_alive():
                root.update();break
            time.sleep(.02)
        assert app.thread and not app.thread.is_alive(),"Worker did not exit"
        assert app.frame is not None,"No frame rendered"
        assert app.latest_info["frame"] > 0
        write_json(ROOT/"reports/gui_integration.json",{"passed":True,
            "rendered_frame":app.latest_info["frame"],"source":"public-image engineering clip",
            "window":"Tk rendering tested with window withdrawn", "model":"smoke_only"})
        print("Actual model -> GUI rendering: passed")
    finally:
        app.stop.set()
        if app.thread:
            app.thread.join(timeout=5)
        root.destroy()


if __name__ == "__main__":
    main()
