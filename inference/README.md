# WSI 推理与热力图生成

这个目录用于乳腺癌 WSI 的推理与热力图可视化，主入口是 `main_inference.py`。

## 最小必需代码

推理最小闭环依赖以下文件：

- `inference/main_inference.py`
- `inference/engine.py`
- `inference/wsi_reader.py`
- `inference/wsi_dataset.py`
- `inference/visualizer.py`
- `../model.py`（其中的 `PatchBinaryHead`）

## 环境依赖

### 1) 系统依赖（Linux）

`openslide-python` 依赖系统库 `openslide`，先安装：

```bash
sudo apt-get update
sudo apt-get install -y openslide-tools libopenslide0
```

### 2) Python 依赖

建议 Python 3.10+，在虚拟环境中安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.inference.txt
```

## 运行示例

```bash
python main_inference.py \
  --wsi_path /path/to/slide.svs \
  --vit_weights /path/to/vit_h_local_dir \
  --head_weights /path/to/best_model.pt \
  --output_dir ./wsi_inference_results \
  --patch_size 512 \
  --tissue_thresh 0.1 \
  --batch_size 32 \
  --num_workers 8 \
  --hidden_dim 256 \
  --alpha 0.5
```

> 注意：`--hidden_dim` 必须和训练 `PatchBinaryHead` 时保持一致。

## 建议上传到 GitHub 的内容

建议只上传代码与文档，不上传大文件（权重/数据/输出）：

- `inference/*.py`
- `inference/README.md`
- `inference/requirements.inference.txt`
- `inference/.gitignore`
- `model.py`

`.gitignore` 已配置忽略常见大文件和中间结果目录。

## 可选：单独建立轻量仓库

如果你只想发布推理模块，建议新建一个独立仓库，把上面的最小文件集合复制过去，目录结构示例：

```text
wsi-inference/
├── inference/
│   ├── main_inference.py
│   ├── engine.py
│   ├── wsi_reader.py
│   ├── wsi_dataset.py
│   ├── visualizer.py
│   ├── requirements.inference.txt
│   ├── README.md
│   └── .gitignore
└── model.py
```
