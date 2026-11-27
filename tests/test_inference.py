import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from common import NAMES, check_model, digest, read_image, save_image, write_json
from inference import VideoRecorder, draw_detections, run_stream


def test_release_guard_rejects_untrained_or_mismatched_weights(tmp_path):
    p = tmp_path/"best.pt"; p.write_bytes(b"fixture-not-real-model")
    model = SimpleNamespace(names=NAMES)
    with pytest.raises(ValueError,match="说明"):
        check_model(model,p)
    write_json(p.with_suffix(".release.json"), {"status":"smoke_only","weights_sha256":digest(p)})
    with pytest.raises(ValueError,match="验收"):
        check_model(model,p)
    assert check_model(model,p,True)[1] is False
    with pytest.raises(ValueError,match="类别"):
        check_model(SimpleNamespace(names={0:"person"}),p,True)
    with pytest.raises(ValueError,match="类别"):
        check_model(SimpleNamespace(names={0:"crack",1:"other"}),p,True)


def test_unicode_image_roundtrip(tmp_path):
    p = tmp_path/"中文图片.png"
    img = np.full((30,40,3),100,np.uint8)
    save_image(p,img)
    assert np.array_equal(read_image(p),img)


def test_only_crack_red_boxes_are_rendered():
    frame = np.zeros((120,160,3), np.uint8)
    image, counts = draw_detections(frame,[[0,10,30,40,80,.4],[1,80,30,120,80,.6]],{0:.3})
    assert counts == {0:1}
    assert tuple(image[40,10]) == (0,0,255)
    assert tuple(image[40,80]) == (0,0,0)
    assert not np.any(np.all(image==[255,0,0],axis=2))


def test_video_all_frames_saved_unicode_paths_and_input_protected(tmp_path):
    source = tmp_path/"输入视频.mp4"
    writer = cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*"mp4v"),10,(160,120))
    assert writer.isOpened()
    for i in range(12):
        writer.write(np.full((120,160,3),i*10,np.uint8))
    writer.release()
    weights = tmp_path/"fake.pt"; weights.write_bytes(b"fixture")
    detector = SimpleNamespace(validated=False, weights=weights, imgsz=960,thresholds={0:.2},
        predict=lambda frame,t:(frame,{0:0},[]))
    with pytest.raises(ValueError,match="覆盖输入"):
        run_stream(detector,str(source),threading.Event(),lambda f,i:None,output=str(source))
    output = tmp_path/"输出视频.mp4"
    report = run_stream(detector,str(source),threading.Event(),lambda f,i:None,
                        output=str(output),log_path=tmp_path/"timing.csv")
    assert report["frames"] == 12 and report["error"] is None
    video = cv2.VideoCapture(str(output))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    video.release()
    assert (tmp_path/"timing.json").exists()
    with pytest.raises(ValueError,match="已存在"):
        VideoRecorder(output,(160,120),10)


def test_empty_or_corrupt_video_error(tmp_path):
    path = tmp_path/"broken.mp4"; path.write_bytes(b"not a video")
    weights=tmp_path/"model.pt"; weights.write_bytes(b"fixture")
    detector=SimpleNamespace(validated=False,weights=weights,imgsz=960)
    with pytest.raises(ValueError,match="无法打开视频"):
        run_stream(detector,str(path),threading.Event(),lambda f,i:None)
