"""Run one selected video with an explicit, hash-checked frozen inference contract."""
from __future__ import annotations

import argparse
from importlib.metadata import version
from pathlib import Path
import threading
import uuid

from common import ROOT, digest, now, read_json, resolve, write_json

CONTRACT = ROOT/'configs/frozen-road-20261029.json'

def validate_contract(contract):
    for name, expected in contract['files'].items():
        path=resolve(name)
        if not path.is_file() or digest(path)!=expected:
            raise ValueError(f'冻结文件已改变或缺失，停止检测：{path}')
    for package, expected in contract['packages'].items():
        if version(package)!=expected:
            raise ValueError(f'运行环境已变化：{package}；请核对后建立新的冻结版本。')
    weights=resolve(contract['weights']);thresholds=resolve(contract['thresholds'])
    t=read_json(thresholds);meta=read_json(weights.with_suffix('.release.json'))
    if (t['thresholds']!={'0':contract['confidence']} or
        t['weights_sha256']!=digest(weights) or
        t['provenance_sha256']!=meta['provenance_sha256'] or
        t.get('reference_evaluation',False) or t['imgsz']!=contract['imgsz']):
        raise ValueError('冻结模型、阈值和数据摘要不匹配')
    from preprocessing import check_threshold_preprocessing
    check_threshold_preprocessing(t,contract['imgsz'],contract['nms_iou'],None)
    return weights,thresholds

def choose_video():
    import tkinter as tk
    from tkinter import filedialog
    window=tk.Tk();window.withdraw()
    try:
        return filedialog.askopenfilename(title='选择道路视频：固定模型与阈值检测',filetypes=[('视频','*.mp4 *.avi *.mov *.mkv'),('所有文件','*')])
    finally:
        window.destroy()

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',help='不填写则打开视频选择窗口')
    parser.add_argument('--check-only',action='store_true',help='只核对冻结记录，不读取视频或加载GPU')
    parser.add_argument('--max-frames',type=int,help='仅功能检查时截断；使用此参数的结果不是完整视频检测')
    args=parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames<=0:parser.error('max-frames必须大于0')
    contract=read_json(CONTRACT)
    if not contract:raise ValueError('缺少固定参数配置')
    weights,thresholds=validate_contract(contract)
    print(f'冻结模型：{weights}\n置信度阈值：{contract["confidence"]}\nNMS IoU：{contract["nms_iou"]}；内容尺寸：{contract["imgsz"]}',flush=True)
    if args.check_only:
        print('冻结文件、环境与参数校验通过。');return
    source=args.video or choose_video()
    if not source:
        print('未选择视频，本次未检测。');return
    source=Path(source).resolve()
    if not source.is_file():raise ValueError(f'视频不存在：{source}')
    folder=ROOT/'outputs/frozen-road-20261029'/uuid.uuid4().hex
    folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'contract.json',contract)
    manifest=dict(created_at=now(),status='started',source=str(source),source_sha256=digest(source),
        contract_sha256=digest(CONTRACT),contract=contract,max_frames=args.max_frames,
        independent_road_test_confirmed=False,formal_acceptance=False,
        input_policy='Whole selected video, no additional crop/filter; shared frozen letterbox preprocessing.',
        purpose='Frozen inference output only; reference annotations and independent sampling are still required for accuracy evaluation.')
    write_json(folder/'run.json',manifest)
    print(f'输出目录：{folder}\n正在处理；原视频保持不变。',flush=True)
    try:
        from inference import Detector,run_stream
        detector=Detector(weights,device=contract['device'],imgsz=contract['imgsz'],allow_unvalidated=True,nms_iou=contract['nms_iou'],thresholds=thresholds)
        if detector.thresholds!={0:contract['confidence']}:raise ValueError('实际加载的阈值不一致')
        def progress(frame,info):
            if info['frame']%300==0:print(f'已处理 {info["frame"]} 帧',flush=True)
        summary=run_stream(detector,str(source),threading.Event(),progress,output=folder/'detections.mp4',max_frames=args.max_frames,log_path=folder/'frames.csv')
        if digest(source)!=manifest['source_sha256']:raise ValueError('运行期间输入视频发生变化')
        validate_contract(contract)
        manifest.update(status='completed',completed_at=now(),summary=summary,
            full_video_processing_requested=args.max_frames is None)
        write_json(folder/'run.json',manifest)
        print(f'检测完成：{summary["frames"]}帧\n带框视频：{folder/"detections.mp4"}\n尚未计算精确率/召回率；带框输出为无音轨MP4。',flush=True)
    except Exception as exc:
        manifest.update(status='failed',failed_at=now(),error=str(exc));write_json(folder/'run.json',manifest);raise

if __name__=='__main__':
    try:main()
    except (ValueError,OSError) as exc:raise SystemExit(str(exc))
