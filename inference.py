"""Inference, latest-frame camera capture, rendering and timestamp-aware recording."""
from __future__ import annotations

from collections import deque
import csv
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time
import warnings

import cv2

from common import COLORS, NAMES, check_model, digest, read_json, resolve
from preprocessing import check_threshold_preprocessing, predict_frame, preprocessing_spec


def draw_detections(frame, predictions, thresholds, normalized=False):
    image = frame.copy()
    h, w = image.shape[:2]
    counts = {0: 0}
    for p in predictions:
        c, x1, y1, x2, y2, score = p
        c = int(c)
        if c not in NAMES or score < thresholds[c]:
            continue
        if normalized:
            x1, x2, y1, y2 = x1*w, x2*w, y1*h, y2*h
        x1, x2 = sorted((max(0, min(w-1, round(x1))), max(0, min(w-1, round(x2)))))
        y1, y2 = sorted((max(0, min(h-1, round(y1))), max(0, min(h-1, round(y2)))))
        color = COLORS[c]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{NAMES[c]} {score:.2f}"
        cv2.putText(image, label, (x1, max(18, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
        counts[c] += 1
    return image, counts


class Detector:
    def __init__(self, weights, device="0", imgsz=960, allow_unvalidated=False, nms_iou=None, thresholds=None,
                 tta=None):
        from ultralytics import YOLO
        self.weights = resolve(weights)
        if not self.weights.is_file():
            raise ValueError(f"模型权重不存在: {self.weights}")
        self.model = YOLO(str(self.weights))
        self.meta, self.validated = check_model(self.model, self.weights, allow_unvalidated)
        self.device, self.imgsz = device, imgsz
        self.nms_iou = nms_iou
        self.tta = tta
        self.preprocessing = preprocessing_spec(imgsz, nms_iou, tta)
        if self.validated and self.meta.get("evaluated_preprocessing") != self.preprocessing:
            raise ValueError("此模型的验收预处理版本与当前代码不一致，请重新评估。")
        self.thresholds = {0: .2}
        threshold_file = resolve(thresholds) if thresholds else self.weights.with_suffix(".thresholds.json")
        if thresholds and not threshold_file.is_file():
            raise ValueError(f"指定的阈值文件不存在: {threshold_file}")
        if not thresholds and not threshold_file.exists() and self.meta.get("thresholds_path") and resolve(self.meta["thresholds_path"]).exists():
            threshold_file = resolve(self.meta["thresholds_path"])
        t = read_json(threshold_file, {})
        if t:
            if t.get('reference_evaluation', False):
                raise ValueError('跨版本参考阈值不能用于正式检测；请使用与训练数据一致的阈值')
            if (t.get("weights_sha256") != digest(self.weights) or t.get("imgsz") != imgsz
                    or t.get("provenance_sha256") != self.meta.get("provenance_sha256")):
                raise ValueError("阈值文件与权重或输入尺寸不匹配")
            try:
                check_threshold_preprocessing(t, imgsz, nms_iou, tta)
            except ValueError:
                if self.validated or thresholds:
                    raise
                warnings.warn("旧阈值未用于新的预处理流程；未验收预览暂用0.20，请在验证集重新校准。",
                              UserWarning, stacklevel=2)
            else:
                self.thresholds = {int(k): v for k, v in t["thresholds"].items()}
        elif self.validated:
            raise ValueError("已验收模型缺少验证集阈值文件")
        # Initialize CUDA, kernels and postprocessing before acquiring live frames.
        # Cold-start time is recorded separately and is not silently discarded.
        import numpy as np
        warmup_start = time.perf_counter()
        for _ in range(2):
            predict_frame(self.model, np.zeros((720,1280,3),dtype=np.uint8), self.imgsz,
                          self.device, conf=min(self.thresholds.values()), nms_iou=nms_iou, tta=tta)
        self.cold_start_ms = (time.perf_counter()-warmup_start)*1000

    def predict(self, frame, thresholds=None):
        thresholds = thresholds or self.thresholds
        predictions = predict_frame(self.model, frame, self.imgsz, self.device,
                                    conf=min(thresholds.values()), nms_iou=getattr(self, "nms_iou", None),
                                    tta=getattr(self,'tta',None))
        image, counts = draw_detections(frame, predictions, thresholds)
        if not self.validated:
            cv2.putText(image, "UNVALIDATED / DEMO ONLY", (12, image.shape[0]-18),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 200, 255), 2, cv2.LINE_AA)
        return image, counts, predictions


def _camera_worker(index, frames, errors, stopped):
    """Native driver calls may block indefinitely; confine all of them to this child."""
    capture = None
    frames.cancel_join_thread()
    try:
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release(); capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            raise ValueError(f"无法打开摄像头 {index}，请检查连接、权限和设备占用。")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        capture.set(cv2.CAP_PROP_BUFFERSIZE,1)
        sequence = 0
        while not stopped.is_set():
            ok, frame = capture.read()
            if not ok:
                raise ValueError("摄像头读取失败或已断开")
            sequence += 1
            packet = (sequence,time.perf_counter(),frame)
            try:
                frames.put_nowait(packet)
            except queue.Full:
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass
                try:
                    frames.put_nowait(packet)
                except queue.Full:
                    pass
    except Exception as exc:
        errors.put(str(exc))
    finally:
        if capture:
            capture.release()


class LatestCamera:
    """One queued frame and an isolated driver process with bounded shutdown."""
    def __init__(self,index,worker=None):
        context=mp.get_context("spawn")
        self.frames=context.Queue(maxsize=1)
        self.errors=context.Queue()
        self.stopped=context.Event()
        self.process=context.Process(target=worker or _camera_worker,
            args=(index,self.frames,self.errors,self.stopped),daemon=True)
        self.process.start()

    def read(self,after=0,timeout=10,stop=None):
        deadline=time.perf_counter()+timeout
        while not self.stopped.is_set() and not (stop and stop.is_set()):
            try:
                error=self.errors.get_nowait()
                raise ValueError(error)
            except queue.Empty:
                pass
            remaining=deadline-time.perf_counter()
            if remaining <= 0:
                raise ValueError("摄像头取帧超时：设备虽能打开，但驱动未返回画面。请检查隐私开关、相机权限或更换摄像头。")
            try:
                packet=self.frames.get(timeout=min(.1,remaining))
                if packet[0] > after:
                    return packet
            except queue.Empty:
                if not self.process.is_alive():
                    raise ValueError("摄像头采集进程已退出，未获得可用画面。")
        return None

    def close(self):
        self.stopped.set()
        self.process.join(timeout=1)
        if self.process.is_alive():
            self.process.terminate();self.process.join(timeout=2)
        self.frames.cancel_join_thread();self.frames.close()
        self.errors.cancel_join_thread();self.errors.close()


class VideoRecorder:
    def __init__(self, path, size, fps, live=False):
        self.path = resolve(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise ValueError(f"输出视频已存在，请使用新文件名: {self.path}")
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("结果视频请使用 .mp4 扩展名")
        self.fps = fps if 0 < fps <= 120 else 25.
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
        if not self.writer.isOpened():
            self.writer.release()
            raise ValueError(f"无法保存视频: {self.path}，请检查路径和编码器。")
        self.live = live
        self.first = None
        self.last = None
        self.written = 0

    def write(self, frame, timestamp):
        if self.first is None:
            self.first = timestamp
        if self.live:
            target = int((timestamp-self.first)*self.fps)
            # Duplicate prior rendered frame for time gaps, preserving wall-clock duration.
            while self.last is not None and self.written < target:
                self.writer.write(self.last); self.written += 1
            if self.written <= target:
                self.writer.write(frame); self.written += 1
        else:
            self.writer.write(frame); self.written += 1
        self.last = frame.copy()

    def close(self):
        self.writer.release()


def run_stream(detector, source, stop, on_frame, get_thresholds=None, output=None,
               max_frames=None, duration=None, realtime=False, log_path=None):
    camera = isinstance(source, int)
    if not camera and output and resolve(output).resolve() == resolve(source).resolve():
        raise ValueError("输出视频不能覆盖输入视频")
    capture, recorder, log_file = None, None, None
    previous = 0
    frames = 0
    latencies, intervals = [], deque(maxlen=60)
    started = time.perf_counter()
    error = None
    try:
        if camera:
            capture = LatestCamera(source)
            source_fps = 25.
        else:
            path = resolve(source)
            if not path.is_file():
                raise ValueError(f"视频文件不存在: {path}")
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise ValueError(f"无法打开视频: {path}")
            source_fps = capture.get(cv2.CAP_PROP_FPS)
            source_fps = source_fps if 0 < source_fps <= 240 else 25.
            expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if log_path:
            log_path = resolve(log_path); log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", newline="", encoding="utf-8")
            log = csv.writer(log_file); log.writerow(["frame", "elapsed_s", "capture_to_render_ms", "fps", "crack"])
        while not stop.is_set():
            if duration and time.perf_counter()-started >= duration:
                break
            if camera:
                packet = capture.read(previous,stop=stop)
                if packet is None:
                    break
                previous, captured_at, frame = packet
            else:
                ok, frame = capture.read()
                captured_at = time.perf_counter()
                if not ok:
                    if frames == 0:
                        raise ValueError("视频没有可解码的画面")
                    if expected_frames > 0 and frames < expected_frames-1:
                        raise ValueError(f"视频提前解码失败: {frames}/{expected_frames} 帧")
                    break
            thresholds = dict(get_thresholds() if get_thresholds else detector.thresholds)
            rendered, counts, predictions = detector.predict(frame, thresholds)
            finished = time.perf_counter()
            intervals.append(finished)
            fps = (len(intervals)-1)/(intervals[-1]-intervals[0]) if len(intervals)>1 else 0.
            latency = (finished-captured_at)*1000
            latencies.append(latency)
            cv2.putText(rendered, f"FPS {fps:.1f}  crack {counts[0]}  {latency:.0f}ms",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 2, cv2.LINE_AA)
            if output and recorder is None:
                recorder = VideoRecorder(output, (frame.shape[1], frame.shape[0]), source_fps, live=camera)
            if recorder:
                recorder.write(rendered, captured_at)
            frames += 1
            if log_file:
                log.writerow([frames, finished-started, latency, fps, counts[0]])
            # GUI latency includes waiting for presentation, measured by the receiver separately.
            on_frame(rendered, {"frame": frames, "fps": fps, "latency_ms": latency,
                                   "captured_at": captured_at, "counts": counts})
            if max_frames and frames >= max_frames:
                break
            if realtime and not camera:
                stop.wait(max(0, frames/source_fps-(time.perf_counter()-started)))
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        if camera and capture:
            capture.close()
        elif capture:
            capture.release()
        if recorder:
            recorder.close()
        if log_file:
            log_file.close()
        from common import write_json, now
        elapsed = time.perf_counter()-started
        import numpy as np
        summary = {"at": now(), "source": source, "frames": frames, "elapsed_s": elapsed,
                   "average_processed_fps": frames/max(elapsed,1e-9),
                   "mean_capture_to_render_ms": float(np.mean(latencies)) if latencies else None,
                   "p95_capture_to_render_ms": float(np.percentile(latencies,95)) if latencies else None,
                   "error": error, "stopped_by_user": stop.is_set(),
                   "model_validated": detector.validated, "weights_sha256": digest(detector.weights),
                   "inference_warmup_ms": getattr(detector,"cold_start_ms",None),
                   "imgsz": detector.imgsz, "preprocessing": getattr(detector, "preprocessing", None),
                   "output": str(resolve(output)) if output else None,
                   "camera_10min_completed": bool(camera and elapsed >= 600 and frames > 0 and not error and not stop.is_set())}
        if log_path:
            write_json(Path(log_path).with_suffix(".json"), summary)
    return summary
