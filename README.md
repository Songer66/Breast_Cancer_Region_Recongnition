# WSI Inference For Breast Cancer

基于 `ViT-H + PatchBinaryHead` 的乳腺癌 WSI 推理项目。输入 `.svs` 切片后，自动完成组织区域筛选、512 级别切块推理，并输出可浏览的热力图结果。

## 功能概览

- 支持 `SVS` 读取（`KFB` 目前为占位逻辑）
- 基于组织 mask 过滤背景区域，减少无效 patch
- 动态读取 `512x512` patch 并批量推理
- 自动从 `best_model.pt` 推断 `hidden_dim`，避免维度配置错误
- 输出三类结果：
  - `*_heatmap.png`（缩略图叠加预览）
  - `*_heatmap_pyramid.tif`（RGBA 彩色金字塔 TIF，`uint8`）
  - `*_probability_pyramid.tif`（单通道概率金字塔 TIF，`uint8`）
- CPU 模式支持显式线程控制，充分利用多核
## 仓库结构

```text
wsi-inference/
├── best_model.pt
├── model.py
├── README.md
└── inference/
    ├── main_inference.py
    ├── engine.py
    ├── wsi_reader.py
    ├── wsi_dataset.py
    ├── visualizer.py
    ├── requirements.inference.txt
    └── README.md
```

## 环境安装

### 1) 系统依赖（Linux）

```bash
sudo apt-get update
sudo apt-get install -y openslide-tools libopenslide0
sudo apt-get install -y libvips libvips-dev
```

### 2) Python 依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r inference/requirements.inference.txt
```

## 权重准备

- `--vit_weights`：本地 ViT-H 目录（需包含 `config.json` 与模型参数文件）
- `--head_weights`：你训练的 `best_model.pt`

示例：

```text
/path/to/vith_weight/
  ├── config.json
  └── pytorch_model.bin
```

## 运行示例

在仓库根目录执行：

```bash
python inference/main_inference.py \
  --wsi_path /path/to/slide.svs \
  --vit_weights /path/to/vith_weight \
  --head_weights /path/to/best_model.pt \
  --output_dir ./wsi_inference_results \
  --patch_size 512 \
  --tissue_thresh 0.1 \
  --batch_size 32 \
  --num_workers 8 \
  --output_format both \
  --tif_compression deflate \
  --tif_tile_size 256 \
  --heatmap_colormap jet
```

说明：

- `--hidden_dim` 可省略，程序会从 `--head_weights` 自动推断
- 若手动传 `--hidden_dim`，必须和权重一致

## CPU 多核加速（不降精度）

```bash
python inference/main_inference.py \
  --wsi_path /path/to/slide.svs \
  --vit_weights /path/to/vith_weight \
  --head_weights /path/to/best_model.pt \
  --output_dir ./wsi_inference_results \
  --device cpu \
  --cpu_num_threads 96 \
  --cpu_interop_threads 8 \
  --batch_size 16 \
  --num_workers 2 \
  --output_format both
```

参数建议：

- `cpu_num_threads`：先设为 `nproc` 或接近 `nproc`
- `cpu_interop_threads`：建议 `4~8`
- `num_workers`：CPU 模式建议 `1~4`，过高容易抢算力

## 关键参数说明

- `--wsi_path`：输入 WSI 路径（当前主支持 `.svs`）
- `--output_dir`：输出目录
- `--patch_size`：Level 0 切块大小，默认 `512`
- `--tissue_thresh`：组织占比阈值，默认 `0.1`
- `--batch_size`：推理 batch 大小
- `--num_workers`：DataLoader 线程数
- `--device`：`cuda` 或 `cpu`
- `--output_format`：`png` / `tif` / `both`
- `--tif_compression`：`jpeg` / `deflate` / `lzw` / `none`
- `--heatmap_colormap`：`jet` / `gray`（用于彩色 TIF）

## 输出文件说明

- `*_heatmap.png`：缩略图融合热力图，便于快速查看
- `*_heatmap_pyramid.tif`：RGBA 彩色金字塔 TIFF（4 通道 `uint8`）
- `*_probability_pyramid.tif`：单通道概率金字塔 TIFF（1 通道 `uint8`，背景为 0）

## 常见问题

### 1) `OpenSlide is required for .svs files`

未安装 OpenSlide 系统库，执行：

```bash
sudo apt-get install -y openslide-tools libopenslide0
```

### 2) `hidden_dim` mismatch

删掉 `--hidden_dim` 让程序自动推断，或手动改成与权重一致。

### 3) CPU 很慢

- 确认 `--device cpu --cpu_num_threads ...` 已设置
- 适当增大 `--batch_size`（如 8/16/24 对比）
- 降低 `--num_workers`（通常 1~4 更稳）
### 4) `pyvips` 相关报错

确认系统已安装 `libvips`，并在当前 Python 环境中安装 `pyvips`。
