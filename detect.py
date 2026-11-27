"""Desktop detection window; also supports headless video/camera benchmarks."""
from __future__ import annotations

import argparse
from datetime import datetime
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

from common import ROOT, resolve, save_image
from inference import Detector, run_stream


class DetectionWindow:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.thread = None
        self.stop = threading.Event()
        self.events = queue.Queue(maxsize=2)
        self.frame = None
        self.thresholds = {0: .2}
        self.latest_info = None
        self.display_latencies = []
        self.override_thresholds = False
        root.title("YOLOv10 混凝土裂缝检测")
        root.geometry("1280x850")
        panel = ttk.Frame(root, padding=10); panel.pack(fill="x")
        self.weights = tk.StringVar(value=str(resolve(args.weights)) if args.weights else "")
        ttk.Label(panel, text="裂缝权重").grid(row=0, column=0, sticky="w")
        ttk.Entry(panel, textvariable=self.weights, width=85).grid(row=0, column=1, columnspan=4, sticky="ew")
        ttk.Button(panel, text="选择权重", command=self.choose_weights).grid(row=0, column=5)
        self.mode = tk.StringVar(value="camera" if args.source.isdigit() else "file")
        ttk.Radiobutton(panel, text="摄像头编号", variable=self.mode, value="camera").grid(row=1, column=0)
        self.camera = tk.StringVar(value=args.source if args.source.isdigit() else "0")
        ttk.Spinbox(panel, from_=0, to=9, width=4, textvariable=self.camera).grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(panel, text="视频文件", variable=self.mode, value="file").grid(row=1, column=2)
        self.video = tk.StringVar(value="" if args.source.isdigit() else args.source)
        ttk.Entry(panel, textvariable=self.video, width=42).grid(row=1, column=3, columnspan=2, sticky="ew")
        ttk.Button(panel, text="选择视频", command=self.choose_video).grid(row=1, column=5)
        self.crack_conf = tk.DoubleVar(value=.2)
        for row, name, var in [(2, "裂缝阈值", self.crack_conf)]:
            ttk.Label(panel, text=name).grid(row=row, column=0, sticky="w")
            ttk.Scale(panel, from_=.01, to=.95, variable=var, command=self.set_thresholds).grid(row=row, column=1, columnspan=4, sticky="ew")
        self.conf_label = ttk.Label(panel, text="0.20"); self.conf_label.grid(row=2, column=5, rowspan=2)
        self.record = tk.BooleanVar(value=bool(args.output))
        ttk.Checkbutton(panel, text="保存标注视频（无音轨）", variable=self.record).grid(row=4, column=0, columnspan=2, sticky="w")
        self.start_button = ttk.Button(panel, text="开始检测", command=self.start); self.start_button.grid(row=4, column=2)
        ttk.Button(panel, text="停止", command=self.stop.set).grid(row=4, column=3)
        ttk.Button(panel, text="保存截图", command=self.screenshot).grid(row=4, column=4)
        panel.columnconfigure(3, weight=1)
        self.status = ttk.Label(root, text="红框标注裂缝。请选择单类裂缝检测权重。", padding=8)
        self.status.pack(fill="x")
        self.canvas = tk.Canvas(root, background="#171717", highlightthickness=0); self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.render())
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(30, self.poll)

    def choose_weights(self):
        path = filedialog.askopenfilename(title="选择训练得到的 best.pt", filetypes=[("PyTorch 模型", "*.pt")])
        if path:
            self.weights.set(path); self.override_thresholds = False

    def choose_video(self):
        path = filedialog.askopenfilename(title="选择视频", filetypes=[("视频", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*")])
        if path:
            self.video.set(path); self.mode.set("file")

    def set_thresholds(self, _=None):
        self.thresholds = {0: self.crack_conf.get()}
        self.override_thresholds = True
        self.conf_label.configure(text=f"{self.thresholds[0]:.2f}")

    def post(self, kind, data):
        # Frame events may be dropped for responsive rendering; terminal events are retained.
        while True:
            try:
                self.events.put_nowait((kind, data)); break
            except queue.Full:
                try:
                    self.events.get_nowait()
                except queue.Empty:
                    pass

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        try:
            source = int(self.camera.get()) if self.mode.get() == "camera" else self.video.get()
            if not self.weights.get().strip() or source == "":
                raise ValueError("请选择训练权重和视频来源。")
            if isinstance(source, int) and source < 0:
                raise ValueError("摄像头编号必须非负")
        except ValueError as e:
            messagebox.showerror("输入有误", str(e)); return
        output = self.args.output
        if self.record.get() and not output:
            output = filedialog.asksaveasfilename(title="保存标注视频", defaultextension=".mp4", filetypes=[("MP4 视频", "*.mp4")])
            if not output:
                return
        if not self.record.get():
            output = None
        weights = self.weights.get()
        self.stop.clear(); self.display_latencies = []
        self.start_button.configure(state="disabled")
        self.status.configure(text="正在加载模型…")
        log_path = ROOT/"outputs"/f"session-{datetime.now():%Y%m%d-%H%M%S}.csv"

        def work():
            try:
                detector = Detector(weights, self.args.device, self.args.imgsz, self.args.allow_unvalidated,
                                    nms_iou=getattr(self.args, "nms_iou", None),
                                    thresholds=getattr(self.args, "thresholds", None),
                                    tta=getattr(self.args,'tta',None))
                if not self.override_thresholds:
                    self.thresholds = dict(detector.thresholds)
                    self.post("thresholds", detector.thresholds)
                summary = run_stream(detector, source, self.stop,
                    lambda frame, info: self.post("frame", (frame, info)),
                    get_thresholds=lambda: dict(self.thresholds), output=output,
                    max_frames=self.args.max_frames, duration=self.args.duration, realtime=True, log_path=log_path)
                self.post("done", summary)
            except Exception as e:
                self.post("error", str(e))
        self.thread = threading.Thread(target=work, daemon=True); self.thread.start()

    def poll(self):
        try:
            while True:
                kind, data = self.events.get_nowait()
                if kind == "frame":
                    self.frame, self.latest_info = data
                    latency = (time.perf_counter()-self.latest_info["captured_at"])*1000
                    self.display_latencies.append(latency)
                    self.status.configure(text=f"检测中 | FPS {self.latest_info['fps']:.1f} | 采集到窗口 {latency:.0f} ms | 裂缝 {self.latest_info['counts'][0]}")
                    self.render()
                elif kind == "thresholds":
                    self.crack_conf.set(data[0])
                    self.conf_label.configure(text=f"{data[0]:.2f}")
                elif kind == "done":
                    self.start_button.configure(state="normal")
                    self.status.configure(text=f"检测已结束：{data['frames']} 帧，平均 {data['average_processed_fps']:.1f} FPS。")
                    if self.display_latencies:
                        import numpy as np
                        from common import write_json
                        write_json(ROOT/"reports/last_display_latency.json", {
                            "frames_presented": len(self.display_latencies),
                            "mean_capture_to_window_ms": float(np.mean(self.display_latencies)),
                            "p95_capture_to_window_ms": float(np.percentile(self.display_latencies,95)),
                            "note": "应用取得画面到 Tk 画布更新，不含相机传感器内部延迟；桌面合成延迟未测量。"})
                else:
                    self.start_button.configure(state="normal")
                    self.status.configure(text=data); messagebox.showerror("检测失败", data)
        except queue.Empty:
            pass
        self.root.after(30, self.poll)

    def render(self):
        if self.frame is None:
            return
        image = Image.fromarray(self.frame[:, :, ::-1])
        image.thumbnail((max(1,self.canvas.winfo_width()), max(1,self.canvas.winfo_height())))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width()/2, self.canvas.winfo_height()/2, image=self.photo)

    def screenshot(self):
        if self.frame is None:
            return
        path = filedialog.asksaveasfilename(title="保存截图", defaultextension=".png", filetypes=[("PNG 图片", "*.png")])
        if path:
            save_image(path, self.frame)

    def close(self):
        self.stop.set()
        if self.thread and self.thread.is_alive():
            self.status.configure(text="正在停止并释放摄像头与视频文件…")
            self.root.after(100, self.close)
            return
        self.root.destroy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights")
    p.add_argument("--source", default="0", help="摄像头编号，或视频文件路径")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--device", default="0")
    p.add_argument("--thresholds", help="显式选择匹配当前预处理和后处理的验证阈值文件")
    p.add_argument("--nms-iou", type=float, help="可选重复框抑制IoU；默认关闭")
    p.add_argument('--tta',choices=['flip4'],help='四方向翻转集成，优先精度，速度较慢')
    p.add_argument("--allow-unvalidated", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--duration", type=float, help="摄像头10分钟测试使用600秒")
    p.add_argument("--output")
    args = p.parse_args()
    if args.imgsz <= 0 or (args.max_frames is not None and args.max_frames <= 0) or (args.duration is not None and args.duration <= 0):
        p.error("imgsz/max-frames/duration 必须为正数")
    try:
        if args.headless:
            if not args.weights:
                p.error("--headless 需要 --weights")
            detector = Detector(args.weights, args.device, args.imgsz, args.allow_unvalidated,
                                nms_iou=args.nms_iou, thresholds=args.thresholds,
                                tta=args.tta)
            source = int(args.source) if args.source.isdigit() else args.source
            summary = run_stream(detector, source, threading.Event(), lambda f,i: None,
                                 output=args.output, max_frames=args.max_frames, duration=args.duration,
                                 log_path=ROOT/"outputs"/f"headless-{datetime.now():%Y%m%d-%H%M%S}.csv")
            print(summary)
        else:
            root = tk.Tk(); DetectionWindow(root, args); root.mainloop()
    except (ValueError, OSError) as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
