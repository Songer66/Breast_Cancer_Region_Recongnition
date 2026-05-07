# inference/ 说明

本目录是 WSI 推理主模块，入口文件为 `main_inference.py`。

## 当前能力

- 读取 `.svs` 并构建组织 mask（`.kfb` 仍为占位逻辑）
- 组织区域按 `patch_size`（默认 512）提取有效坐标
- 动态读取 patch 并用 `ViT-H + PatchBinaryHead` 批量推理
- 自动从 `--head_weights` 推断 `hidden_dim`，避免权重维度不匹配
- 输出：
  - `*_heatmap.png`
  - `*_heatmap_pyramid.tif`（RGBA `uint8`）
  - `*_probability_pyramid.tif`（单通道 `uint8`）

## 依赖安装

```bash
sudo apt-get update
sudo apt-get install -y openslide-tools libopenslide0
sudo apt-get install -y libvips libvips-dev
pip install -r requirements.inference.txt
```

## 运行示例

```bash
python main_inference.py \
  --wsi_path /path/to/slide.svs \
  --vit_weights /path/to/vith_weight \
  --head_weights /path/to/best_model.pt \
  --output_dir ./wsi_inference_results \
  --patch_size 512 \
  --batch_size 32 \
  --num_workers 8 \
  --output_format both \
  --tif_compression deflate
```

## CPU 多核参数

- `--device cpu`
- `--cpu_num_threads`：算子内线程，`0` 表示自动
- `--cpu_interop_threads`：算子间线程，`0` 表示自动

例如：

```bash
python main_inference.py ... --device cpu --cpu_num_threads 96 --cpu_interop_threads 8
```

## 备注

- 若使用根目录运行方式，建议优先阅读 `../README.md`，包含更完整的端到端说明。
