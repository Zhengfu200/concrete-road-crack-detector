"""Engineering test clip made from public dataset images; not an independent accuracy test."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from common import ROOT, load_manifest, read_image, write_json

path=ROOT/"outputs/engineering-input-720p.mp4"
path.parent.mkdir(parents=True,exist_ok=True)
writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),15,(1280,720))
if not writer.isOpened():
    raise SystemExit("Cannot create fixture video")
rows=[r for r in load_manifest()["records"] if r["source"]=="crack-seg"][:12]
for row in rows:
    frame=cv2.resize(read_image(row["image"]),(1280,720))
    for _ in range(10):
        writer.write(frame)
writer.release()
write_json(path.with_suffix(".json"),{"purpose":"engineering video I/O and throughput only", "frames":120,
           "width":1280,"height":720,"fps":15,"source_ids":[r["id"] for r in rows]})
print(path)
