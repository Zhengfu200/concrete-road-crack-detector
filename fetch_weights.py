"""Download the original authors' YOLOv10s pretraining checkpoint, with provenance."""
import shutil
import urllib.request

from common import ROOT, digest, now, write_json


def main():
    url = "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt"
    target = ROOT/"weights/yolov10s.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        part = target.with_suffix(".pt.part")
        with urllib.request.urlopen(req, timeout=60) as response, part.open("wb") as f:
            shutil.copyfileobj(response, f, 1024*1024)
        part.replace(target)
    write_json(target.with_suffix(".source.json"), {"url": url, "sha256": digest(target), "at": now(),
               "purpose": "COCO pretrained initialization, NOT a trained crack detector"})
    print(target)


if __name__ == "__main__":
    main()
