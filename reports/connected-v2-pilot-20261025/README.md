# 整体连通裂缝V2：试标交付与复核说明

后续状态（2026-10-25）：用户已确认全部50张试标并冻结规则，随后授权AI全量复核；V2现已导出。最新结果见 `../connected-v2-full-review-20261025/README.md`。以下保留试标交付时的历史快照，不作为当前进度。

日期：2026-10-25。当前阶段：试标交付，规则尚未冻结，全量V2尚未生成，新训练尚未启动。

已索引现有4028张图片，保持train=3717、val=200、test=111及原始身份。先检查156张训练候选原图，再逐张查看所选50张原图和原多边形，形成50份AI提案。首次判断未显示模型预测，未查看测试图像。

试标来自50个不同文件名前缀家族：连续拆段20、分叉交叉10、独立目标关系10、细弱5、贴边/遮挡5。类别表示选样复核重点，个别连接关系仍待确定；不视为最终类别结论。文件名前缀也不证明拍摄来源独立。

当前28份提案可供确认，另22张标为待定。待定包括漏标支路、半条裂缝、孤立短标签或遮挡处连通不明；这些图没有被当作背景，也未被强行合并。0张记作人工确认。

## 打开复核入口

双击程序目录内 `06_整体裂缝V2复核.cmd`，或打开 http://127.0.0.1:8766 。服务仅监听本机，不上传图片。当前已启动；页面不能访问时双击入口重新启动。

1. 左侧填写自己的姓名，选择图片。左图默认原图，可勾选显示原多边形与P序号；右图为待确认标注。没有模型预测。
2. 查看整张依据，核对是否遗漏裂缝。合并：勾选多个实例、填写连通依据、点合并。拆分：拆回各多边形；单个多边形覆盖多个独立目标时使用单多边形拆分并逐个拖绘。
3. 漏标可拖绘补框；同一连通裂缝补支后，应合并为同一整体，或用“替换所选框”重新框住全部可见范围。接缝/纹理标签可排除并记录原因。
4. 判断清楚后点“确认本张并下一张”。暂时无法决定时点“保存为待定”；草稿可续改。每次保存生成新修订，不覆盖源标签。
5. 50张全部确认后点“确认试标并冻结规则”。这是前一计划约定的人工确认节点，必须实际逐图核对；不能仅以“继续实施”代替已检查的记录。

## 已实现的数据版本保护

- 试标未完成确认时，验证/测试图片不会出现在复核入口，也不能导出训练V2。
- 冻结记录绑定规则、试标和源索引摘要。修改试标或规则会使冻结失效。
- 全量逐图复核后，只导出完成的标注；待定图单独列出，未处理/草稿会阻止发布。AI复核和用户确认分别计数，均不冒记为混凝土材质或拍摄来源复核。
- V2包含规则、父数据摘要、完整原多边形索引、实例映射、复核审计和全部修订历史；不能覆盖同名已发布版本。
- 发布前重新核对上游原图、上游原多边形、V1图片和检测标签的SHA-256；保持划分不变。

全量完成后执行（当前会明确阻止提前发布）：

```powershell
& 'USER_HOME\miniconda3\envs\concrete-yolov10\python.exe' connected_annotations.py export
```

## V2基线评估已准备，尚未运行

`evaluate.py --reference-evaluation` 只允许val，须显式使用独立目录与独立阈值；记录训练/评估数据的不同摘要，不修改旧权重的验收状态。以下命令须等V2正式发布后执行：

```powershell
& 'USER_HOME\miniconda3\envs\concrete-yolov10\python.exe' evaluate.py `
  --weights runs/concrete-crack-20261021-212347/weights/best.pt `
  --data data/prepared/crack-connected-v2 --split val --calibrate `
  --reference-evaluation --imgsz 960 --nms-iou 0.3 `
  --output reports/connected-v2-original-baseline `
  --thresholds reports/connected-v2-original-baseline/val.thresholds.json
```

这里的960内容区域+四周16像素、NMS=0.3与现有候选评估一致。旧版79.52%是旧单位、旧验证标签的R@P80，不能与V2直接比较或归因于模型学习。正常路面样本和正式混凝土道路域验收仍未补足。

## 文件与证据

- `pilot-selection.json`：最终50张试标清单；初筛清单另留存。
- `agent-proposals.json`：逐图依据、原框与提案，明确标为AI。
- `before-after/P01.jpg`至`P50.jpg`：左侧原检测框，右侧提案；绿色为草稿，橙色为待定。
- `pilot-status.json`：试标生成时快照，后续实际进度以复核工作区为准。
- `verification.json`：本轮测试和原文件摘要核对。

复核方法沿用scientific-critical-thinking技能，对实例单位、图像证据和域验收分别记录；引用已核对作者与当前arXiv v2：Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065)。
