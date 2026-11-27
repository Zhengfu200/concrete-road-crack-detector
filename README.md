# 混凝土道路裂缝检测

基于 YOLOv10s 的单类裂缝检测课程项目。支持本地视频和摄像头推理、红框预览、结果视频保存、标注复核、训练和固定阈值评估。项目另有巡检机器人的概念方案，机器人实物与路径规划实测仍待完成。

**当前权重是开发候选。** 在 259 张混合开发图上达到 Precision 85.47%、Recall 90.53%。独立道路精度验收和新版模型的摄像头长期稳定性测试待完成。

## 克隆与安装

```powershell
git clone https://github.com/Zhengfu200/concrete-road-crack-detector.git
cd concrete-road-crack-detector
powershell -ExecutionPolicy Bypass -File .\setup.ps1
conda activate concrete-yolov10
python -m pytest tests -q
```

原开发环境是 Windows、Conda、Python 3.12、RTX 5080 Laptop GPU、PyTorch 2.12.1+cu130 和 Ultralytics 8.4.86。安装脚本需要 PATH 上能调用 Conda，使用 CUDA 13.0 对应的 PyTorch wheel，下载 YOLOv10s 初始化权重并执行 GPU 自检。环境依赖见 [requirements.txt](requirements.txt)，[requirements.lock.txt](requirements.lock.txt) 是原环境快照，含 Conda 本地来源条目。对于其他系统或 GPU，按机器环境安装匹配的 PyTorch，仅用 Python 入口，并重新验证推理。Windows 双击脚本默认从用户目录下的 Miniconda 读取环境。

## 检测自己的视频

```powershell
python run_frozen_video.py --check-only
python run_frozen_video.py --video "D:/videos/road.mp4"
```

也可运行 `08_固定模型阈值_道路视频检测.cmd` 后选择视频。输出到 `outputs/frozen-road-20261029/<运行ID>/`，包含无音轨的 `detections.mp4`、来源和版本记录 `run.json`、冻结配置 `contract.json` 及计数日志 `frames.csv`。固定配置校验权重、阈值、推理文件及依赖版本。需要一致环境时使用 PyTorch 2.12.1+cu130、Ultralytics 8.4.86、NumPy 2.5.2、opencv-python 4.14.0.94。

`05_优化候选预览_未验收.cmd` 打开可调整阈值的预览。也可直接运行以下命令（需要时替换 `--source` 或加 `--device cpu`）：

```powershell
python detect.py --weights runs/concrete-crack-road-recovery-20261029/weights/best_deployment.pt --thresholds reports/data-expansion-20261029/recovery/candidate-val/val.thresholds.json --imgsz 960 --nms-iou 0.3 --allow-unvalidated --source "D:/videos/road.mp4"
```

`04` 使用流程测试权重。`07` 依赖原开发视频的裁剪产物，可用 `08` 检测自己的视频。冻结配置详情见 [固定视频检测说明](docs/frozen-road-20261029.md)。

## 数据和训练

| 版本 | train / val / test | 用途 |
| --- | --- | --- |
| Crack-Seg V1 | 3717 / 200 / 111 | 默认正式训练。用户提供的分割多边形转检测框 |
| 整体连通裂缝 V2 | 3073 / 164 / 89 | 依照整体裂缝标注规则的独立实验版本 |
| 道路扩充版 | 3694 / 259 / 89 | V2 加 432 张路面负图、189 张裂缝训练图、95 张路面开发负图 |

取得项目使用的原始 Crack-Seg 标注目录后，按 `images/{train,val,test}`、`labels/{train,val,test}` 放置并显式传入路径。导入保留上游划分，遇到跨划分完全重复的图像优先保留 test、val。准备脚本会生成数据及来源记录，避免用弃用的旧自标注数据。

```powershell
python prepare_external.py --source "D:/datasets/crack-seg"
python train.py --name crack-seg-v10s
```

默认设置见 [configs/train.yaml](configs/train.yaml)，内容尺寸 960，最多 100 轮、早停 20 轮、batch 4、seed 42。其他实验显式指定数据和独立运行名。历史 V2 及道路扩充数据需要从项目原始来源和复核结果恢复。SDNET2018 采用作者的正常路面类别并筛查，DamSegment 选用裂缝多边形且仅进入训练。来源、许可证、分组局限和筛选过程见 [数据扩充记录](reports/data-expansion-20261029/README.md)。[实验脚本快照](reports/data-expansion-20261029/reproduction) 用于追溯，内含原机器路径及数据前置条件，迁移前先核对。

## 评估与当前结果

部署评估与推理共用 `preprocessing.py`。输入保持比例并填充至 960×960 内容区，四边各加 16 像素后为 992×992。候选关闭翻转集成，预测后 NMS IoU 0.3，裂缝阈值 0.18819282948970795。检测框与标注匹配的评估 IoU 为 0.5。

| 工作点（同一混合开发集） | Precision | Recall | 正常路面图误报 |
| --- | ---: | ---: | ---: |
| 旧候选，同召回率对比 | 56.67% | 90.53% | 92/95 |
| 召回恢复第 3 轮候选 | 85.47% | 90.53% | 8/95 |

最终候选有 TP 153、FP 26、FN 16，漏检率 9.47%，mAP50 89.33%，mAP50–95 64.09%。逐轮训练指标和固定部署工作点评估采用各自的评估路径。图表见 [汇报 PPT](docs/slides/裂缝检测微调结果_逐轮训练版.pptx)，精确数值见 [开发集对照](reports/data-expansion-20261029/recovery/comparison.json)。

```powershell
python evaluate.py --weights runs/crack-seg-v10s/weights/best.pt --data data/prepared/crack-seg-user-20261021 --split val --calibrate
python evaluate.py --weights runs/crack-seg-v10s/weights/best.pt --data data/prepared/crack-seg-user-20261021 --split test
```

以上为自训 V1 的命令示例。只在 val 选阈值，保持推理协议一致。历史 test 已观察，项目验收需要按路段和拍摄批次隔离的新视频或图片，冻结模型、阈值、取样和标注方案后测试 P≥80%、R≥90% 和场景覆盖。

## 仓库导览

| 位置 | 内容 |
| --- | --- |
| `train.py`、`training_improvements.py` | 训练、预处理对齐及部署候选选择 |
| `evaluate.py`、`metrics.py` | 评估、阈值校准及统计 |
| `detect.py`、`inference.py`、`preprocessing.py` | 本地推理、摄像头、视频和几何变换 |
| `prepare_data.py`、`prepare_external.py`、`connected_*.py` | 数据准备和标注复核 |
| `configs/`、`reports/`、`runs/` | 配置、选录报告、关键训练曲线、候选和烟测权重 |
| `tests/`、`docs/` | 自动测试、标注规则、操作文档和汇报单页 |

仓库包含四份有来源记录的项目权重。完整原始数据、历史标注库、开发视频、逐帧图像和运行产物保存在原主机。旧报告中的 `PROJECT_ROOT` 与历史文件路径供追溯。涉及第三方数据、权重和依赖时遵循各自来源的许可。需要公开仓库或对外再分发前，先核对课程材料、项目权重和上游资源的授权。

后续开发请先读 [AGENTS.md](AGENTS.md) 和 [项目记忆](docs/PROJECT_MEMORY.md)。
