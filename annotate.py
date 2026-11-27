"""Local human review: draw/edit boxes and explicitly confirm material and provenance."""
from __future__ import annotations

import argparse
import copy
import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk

from common import ROOT, MANIFEST, digest, load_manifest, now, resolve, validate_boxes, write_json


def unique_review_rows(records, source=None):
    """Show each image content once, reusing reviewed records across source aliases.

    This only selects what to display. It does not merge boxes, alter review flags,
    or suppress conflicting annotations during the dataset audit.
    """
    groups = {}
    for record in records:
        key = ("sha256", record["sha256"]) if record.get("sha256") else ("id", record["id"])
        groups.setdefault(key, []).append(record)
    rows = []
    for aliases in groups.values():
        if source and not any(r["source"] == source for r in aliases):
            continue
        representative = min(aliases, key=lambda r: (
            bool(r.get("excluded")),
            not bool(r.get("reviewed")),
            r.get("annotation_origin") != "human_review",
        ))
        rows.append(representative)
    return rows


class Annotator:
    def __init__(self, root, source=None):
        self.root = root
        self.manifest = load_manifest()
        self.manifest_hash = digest(MANIFEST) if resolve(MANIFEST).exists() else None
        self.rows = unique_review_rows(self.manifest["records"], source)
        if not self.rows:
            raise ValueError("没有候选图片。先用 prepare_data.py 下载并导入。")
        self.rows.sort(key=lambda r: (r.get("reviewed", False) or r.get("excluded", False), -r.get("candidate_score", 0)))
        self.index = 0
        self.boxes = []
        self.dirty = False
        self.start = None
        self.preview = None
        root.title("混凝土裂缝：标注与复核")
        root.geometry("1280x850")
        controls = ttk.Frame(root, padding=8); controls.pack(fill="x")
        self.status = ttk.Label(controls); self.status.pack(side="left", fill="x", expand=True)
        ttk.Button(controls, text="上一张", command=lambda: self.move(-1)).pack(side="left")
        ttk.Button(controls, text="下一张", command=lambda: self.move(1)).pack(side="left")
        ttk.Button(controls, text="跳至未复核", command=self.jump_pending).pack(side="left")
        body = ttk.Panedwindow(root, orient="horizontal"); body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, bg="#222", highlightthickness=0)
        body.add(self.canvas, weight=4)
        panel = ttk.Frame(body, padding=10); body.add(panel, weight=1)
        ttk.Label(panel, text="拖动绘制裂缝红框；列表选择后可删除错误框。\n重复图片仅显示一份，已复核记录优先。", wraplength=300).pack(anchor="w")
        ttk.Label(panel, text="类别：裂缝（红框）").pack(anchor="w", pady=4)
        self.boxlist = tk.Listbox(panel, height=12, exportselection=False); self.boxlist.pack(fill="x", pady=8)
        ttk.Button(panel, text="删除选中框", command=self.delete).pack(fill="x", pady=4)
        self.concrete = tk.BooleanVar()
        ttk.Checkbutton(panel, text="已确认是混凝土表面", variable=self.concrete).pack(anchor="w", pady=4)
        ttk.Label(panel, text="原照片/路段/拍摄序列组（相邻图填同组）").pack(anchor="w")
        self.group = tk.StringVar(); ttk.Entry(panel, textvariable=self.group, width=40).pack(fill="x")
        self.confirmed = tk.BooleanVar()
        ttk.Checkbutton(panel, text="已核实来源组（非凭空编组）", variable=self.confirmed).pack(anchor="w", pady=4)
        ttk.Label(panel, text="复核人").pack(anchor="w")
        self.reviewer = tk.StringVar(); ttk.Entry(panel, textvariable=self.reviewer).pack(fill="x")
        self.complete = tk.BooleanVar()
        ttk.Checkbutton(panel, text="已检查整图，裂缝均无漏标", variable=self.complete).pack(anchor="w", pady=8)
        ttk.Button(panel, text="保存草稿", command=self.save_draft).pack(fill="x", pady=3)
        ttk.Button(panel, text="确认复核并下一张", command=self.approve).pack(fill="x", pady=3)
        ttk.Button(panel, text="排除：非混凝土/无法辨认", command=self.exclude).pack(fill="x", pady=3)
        ttk.Label(panel, text="没有框可作为负样本，但须确认整图\n没有裂缝。\n来源组不明的复核图仅用于训练。", wraplength=300).pack(anchor="w", pady=10)
        self.canvas.bind("<Configure>", lambda e: self.render())
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.boxlist.bind("<<ListboxSelect>>", lambda e: self.render())
        root.bind("<Delete>", lambda e: self.delete() if root.focus_get() == self.boxlist else None)
        root.protocol("WM_DELETE_WINDOW", self.close)
        for var in [self.concrete, self.group, self.confirmed, self.reviewer]:
            var.trace_add("write", self.form_changed)
        self.load()

    def form_changed(self, *_):
        self.dirty = True
        self.complete.set(False)

    def load(self):
        r = self.rows[self.index]
        self.image = Image.open(resolve(r["image"])).convert("RGB")
        self.boxes = copy.deepcopy(r["boxes"])
        self.concrete.set(r.get("surface") == "concrete")
        self.group.set(r.get("group", "")); self.confirmed.set(r.get("group_confirmed", False))
        self.reviewer.set(r.get("reviewer") or self.reviewer.get())
        self.complete.set(r.get("reviewed", False))
        self.status.configure(text=f"{self.index+1}/{len(self.rows)} | {r['source']} | {resolve(r['image']).name} | {'已复核' if r.get('reviewed') else '候选'}")
        self.refresh_list(); self.render(); self.dirty = False

    def refresh_list(self):
        self.boxlist.delete(0, "end")
        for i, b in enumerate(self.boxes):
            self.boxlist.insert("end", f"{i+1}: 裂缝  ({b[1]:.3f},{b[2]:.3f})")

    def render(self):
        if not hasattr(self, "image"):
            return
        w, h = self.image.size
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.scale = min(cw/w, ch/h)
        dw, dh = max(1, int(w*self.scale)), max(1, int(h*self.scale))
        self.offset = ((cw-dw)//2, (ch-dh)//2)
        self.photo = ImageTk.PhotoImage(self.image.resize((dw, dh)))
        self.canvas.delete("all")
        ox, oy = self.offset
        self.canvas.create_image(ox, oy, image=self.photo, anchor="nw")
        selection = self.boxlist.curselection()
        for i, (c, x, y, bw, bh) in enumerate(self.boxes):
            color = "#ff4444"
            a, b = ox+(x-bw/2)*w*self.scale, oy+(y-bh/2)*h*self.scale
            self.canvas.create_rectangle(a, b, ox+(x+bw/2)*w*self.scale, oy+(y+bh/2)*h*self.scale,
                                         outline=color, width=4 if selection and selection[0] == i else 2)
            self.canvas.create_text(a+3, b+3, text=f"{i+1} crack", fill=color, anchor="nw")

    def point(self, event):
        w, h = self.image.size; ox, oy = self.offset
        return min(1., max(0., (event.x-ox)/(w*self.scale))), min(1., max(0., (event.y-oy)/(h*self.scale)))

    def press(self, event):
        self.start = self.point(event)

    def drag(self, event):
        if self.start:
            if self.preview:
                self.canvas.delete(self.preview)
            x, y = self.start; ex, ey = self.point(event)
            w, h = self.image.size; ox, oy = self.offset
            self.preview = self.canvas.create_rectangle(ox+x*w*self.scale, oy+y*h*self.scale,
                ox+ex*w*self.scale, oy+ey*h*self.scale, outline="yellow", width=2)

    def release(self, event):
        if not self.start:
            return
        x, y = self.start; ex, ey = self.point(event)
        self.start = None
        if abs(ex-x)*self.image.width >= 2 and abs(ey-y)*self.image.height >= 2:
            self.boxes.append([0, (x+ex)/2, (y+ey)/2, abs(ex-x), abs(ey-y)])
            self.modified()
        self.render()

    def modified(self):
        self.dirty = True
        self.complete.set(False)
        self.refresh_list()

    def delete(self):
        s = self.boxlist.curselection()
        if s:
            del self.boxes[s[0]]; self.modified(); self.render()

    def persist(self, reviewed=False, excluded=False):
        validate_boxes(self.boxes)
        if self.manifest_hash and digest(MANIFEST) != self.manifest_hash:
            raise ValueError("候选库已被另一个程序修改；为避免覆盖复核结果，请关闭并重新打开工具。")
        if self.confirmed.get() and not self.group.get().strip():
            raise ValueError("来源组不能为空。")
        r = self.rows[self.index]
        r.update(boxes=copy.deepcopy(self.boxes), surface="concrete" if self.concrete.get() else "unknown",
                 group=self.group.get().strip(), group_confirmed=self.confirmed.get(),
                 reviewer=self.reviewer.get().strip(), reviewed=reviewed, excluded=excluded,
                 reviewed_at=now() if reviewed else None, annotation_origin="human_review")
        write_json(MANIFEST, self.manifest)
        self.manifest_hash = digest(MANIFEST)
        self.dirty = False

    def save_draft(self):
        try:
            self.persist()
        except ValueError as e:
            messagebox.showerror("保存失败", str(e))

    def approve(self):
        if not self.concrete.get() or not self.complete.get() or not self.reviewer.get().strip():
            messagebox.showwarning("尚未完成复核", "请确认混凝土材质、整图无漏标，并填写复核人。")
            return
        if not self.boxes and not messagebox.askyesno("确认负样本", "确认整张图没有裂缝？"):
            return
        try:
            self.persist(reviewed=True); self.move(1)
        except ValueError as e:
            messagebox.showerror("保存失败", str(e))

    def exclude(self):
        self.persist(excluded=True); self.move(1)

    def move(self, delta):
        # Always save navigated-away edits as an unreviewed draft.
        if self.dirty:
            self.persist()
        self.index = min(len(self.rows)-1, max(0, self.index+delta)); self.load()

    def jump_pending(self):
        if self.dirty:
            self.persist()
        for offset in range(1, len(self.rows)+1):
            i = (self.index+offset) % len(self.rows)
            if not self.rows[i].get("reviewed") and not self.rows[i].get("excluded"):
                self.index = i; self.load(); return
        messagebox.showinfo("已完成", "全部候选已复核或排除。")

    def close(self):
        if self.dirty:
            self.persist()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="仅复核指定来源，例如 sdnet2018、crack-seg 或补充来源名称")
    args = parser.parse_args()
    root = tk.Tk()
    try:
        Annotator(root, args.source); root.mainloop()
    except ValueError as exc:
        messagebox.showerror("无法打开复核工具", str(exc)); root.destroy()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
