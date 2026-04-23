# WSI Inference For Breast Cancer

基于 `ViT-H + PatchBinaryHead` 的乳腺癌全切片（WSI）推理与热力图生成项目。

本仓库支持输入 `.svs` 切片，自动完成：

1. 组织区域提取（过滤背景）
2. 动态切块与批量推理
3. 肿瘤概率热力图生成

---

## 1. 仓库结构

```text
wsi-inference/
├── best_model.pt                 # 你训练得到的最佳分类头权重（需自行放置）
├── model.py                      # PatchBinaryHead 定义
└── inference/
    ├── main_inference.py         # 主入口
    ├── engine.py                 # 推理引擎（ViT + Head）
    ├── wsi_reader.py             # WSI读取与组织mask
    ├── wsi_dataset.py            # 动态Patch数据集
    ├── visualizer.py             # 热力图生成
    ├── requirements.inference.txt
    └── .gitignore
```

---

## 2. 环境要求

### 2.1 Python 版本

- 推荐：`Python 3.10+`

### 2.2 系统依赖（Linux）

`openslide-python` 依赖系统动态库，请先安装：

```bash
sudo apt-get update
sudo apt-get install -y openslide-tools libopenslide0
```

### 2.3 Python 依赖安装

在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r inference/requirements.inference.txt
```

---

## 3. 权重准备

你需要准备两类权重：

1. `ViT-H` 主干权重目录（本项目通过 `transformers.ViTModel.from_pretrained` 加载）
2. `best_model.pt`（你训练得到的 `PatchBinaryHead` 权重）

### 3.1 best_model.pt 放置位置

推荐放在仓库根目录：

```text
wsi-inference/best_model.pt
```

### 3.2 ViT-H 权重要求

- 参数 `--vit_weights` 传入的是“本地目录”，目录内应包含 HuggingFace 需要的模型文件（如 `config.json`、模型参数文件等）。
- 该目录必须可被 `ViTModel.from_pretrained(..., local_files_only=True)` 正常识别。

---

## 4. 快速开始（推理一张WSI）

在仓库根目录执行：

```bash
python inference/main_inference.py \
  --wsi_path /path/to/your_slide.svs \
  --vit_weights /path/to/your_vit_h_dir \
  --head_weights ./best_model.pt \
  --output_dir /path/to/save_results \
  --patch_size 512 \
  --tissue_thresh 0.1 \
  --batch_size 32 \
  --num_workers 8 \
  --hidden_dim 256 \
  --alpha 0.5
```

---

## 5. 参数说明

- `--wsi_path`：输入切片路径，当前重点支持 `.svs`
- `--vit_weights`：ViT-H 本地权重目录
- `--head_weights`：分类头权重路径（`best_model.pt`）
- `--output_dir`：输出目录，保存热力图
- `--patch_size`：Level 0 下切块大小，默认 `512`
- `--tissue_thresh`：组织占比阈值，默认 `0.1`
- `--batch_size`：推理批大小，按显存调节
- `--num_workers`：DataLoader worker 数
- `--hidden_dim`：分类头隐藏层维度，必须与训练时一致
- `--alpha`：热力图叠加透明度

---

## 6. 输出结果

运行成功后，将在 `--output_dir` 下生成：

- `{slide_name}_heatmap.png`：最终热力图

终端会打印：

- 总耗时
- Patch 总数
- 推理速度（Patch/s）

---

## 7. 常见问题

### Q1: 报错 `OpenSlide is required for .svs files`

说明系统缺少 OpenSlide 动态库，先安装：

```bash
sudo apt-get install -y openslide-tools libopenslide0
```

### Q2: `hidden_dim` 不匹配导致加载权重失败

请确保 `--hidden_dim` 与训练 `best_model.pt` 时一致，否则会出现维度不匹配。

### Q3: CUDA 显存不足

尝试：

- 降低 `--batch_size`
- 降低 `--num_workers`
- 或设置 `--device cpu`

### Q4: ViT权重加载失败

确认 `--vit_weights` 是可被 HuggingFace 正确识别的本地模型目录，而不是单个错误文件路径。

---

## 8. 发布建议

- 建议代码与文档开源。
- 若 `best_model.pt` 较大，建议通过 GitHub Releases 或 HuggingFace Hub 分发，并在 README 中放下载链接。
