import argparse
import os
import sys
import time
import torch
from torch.utils.data import DataLoader

# =====================================================================
# 🚀 核心修复：动态将项目的根目录加入系统路径，以便顺利导入 model.py
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 导入我们亲手打造的四大核心模块
from wsi_reader import WSIReader
from wsi_dataset import WSIDynamicPatchDataset
from engine import WSIInferenceEngine
from visualizer import HeatmapGenerator

def parse_args():
    parser = argparse.ArgumentParser(description="WSI 乳腺癌全切片级联推理与热力图生成平台")
    
    # 基础输入输出路径
    parser.add_argument("--wsi_path", type=str, required=True, help="输入 WSI 文件的路径 (.svs 或 .kfb)")
    parser.add_argument("--output_dir", type=str, default="./wsi_inference_results", help="热力图保存目录")
    
    # 模型权重路径
    parser.add_argument("--vit_weights", type=str, required=True, help="ViT-H 骨干网络权重路径")
    parser.add_argument("--head_weights", type=str, required=True, help="你训练的二分类头 (PatchBinaryHead) 权重路径")
    
    # 数据处理参数
    parser.add_argument("--patch_size", type=int, default=512, help="Level 0 切块大小")
    parser.add_argument("--tissue_thresh", type=float, default=0.1, help="组织面积占比阈值 (漫水填充算法下设为 0.1 即可)")
    
    # 硬件与 DataLoader 参数
    parser.add_argument("--batch_size", type=int, default=32, help="推理 Batch Size (根据显存大小调整)")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader 多线程数量")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="推理设备")
    
    # 模型架构与可视化参数
    parser.add_argument("--hidden_dim", type=int, default=256, help="分类头的隐藏层维度 (需与训练时保持绝对一致)")
    parser.add_argument("--alpha", type=float, default=0.5, help="热力图透明度 (0.0~1.0)")
    parser.add_argument(
        "--output_format",
        type=str,
        default="both",
        choices=["png", "tif", "both"],
        help="输出格式：png 预览图、tif 金字塔热力图，或二者都输出"
    )
    parser.add_argument(
        "--tif_compression",
        type=str,
        default="jpeg",
        choices=["jpeg", "deflate", "lzw", "none"],
        help="金字塔 TIF 压缩方式"
    )
    parser.add_argument("--tif_tile_size", type=int, default=256, help="金字塔 TIF 的 Tile 大小")
    parser.add_argument(
        "--heatmap_colormap",
        type=str,
        default="jet",
        choices=["jet", "gray"],
        help="热力图颜色映射"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    slide_name = os.path.splitext(os.path.basename(args.wsi_path))[0]
    heatmap_png_path = os.path.join(args.output_dir, f"{slide_name}_heatmap.png")
    heatmap_tif_path = os.path.join(args.output_dir, f"{slide_name}_heatmap_pyramid.tif")
    
    print("\n" + "="*60)
    print(f" 🌟 WSI 智能推理流水线启动: {slide_name}")
    print("="*60)
    
    total_start_time = time.time()

    # =====================================================================
    # 阶段 1：WSI 前端解析 (获取坐标与缩略图)
    # =====================================================================
    print("\n[1/4] 正在解析 WSI 并提取有效组织坐标...")
    try:
        reader = WSIReader(args.wsi_path, patch_size=args.patch_size, tissue_thresh=args.tissue_thresh)
        level_0_dimensions = reader.level_0_dimensions
        valid_coords, downsample = reader.get_valid_patch_coordinates()
        # 顺便再调一次 get_tissue_mask 拿到 rgb 缩略图给可视化用
        _, _, thumb_img = reader.get_tissue_mask()
        reader.close()
    except Exception as e:
        print(f"[致命错误] WSI 读取失败: {e}")
        return
        
    if not valid_coords:
        print("❌ 未提取到任何有效组织，推理终止。")
        return

    # =====================================================================
    # 阶段 2：构建动态数据加载器
    # =====================================================================
    print(f"\n[2/4] 初始化动态 DataLoader (Workers: {args.num_workers})...")
    dataset = WSIDynamicPatchDataset(args.wsi_path, valid_coords, patch_size=args.patch_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # =====================================================================
    # 阶段 3：级联引擎推理
    # =====================================================================
    print("\n[3/4] 启动 ViT-H + MLP 级联推理引擎...")
    try:
        engine = WSIInferenceEngine(
            vit_weights_path=args.vit_weights,
            head_weights_path=args.head_weights,
            hidden_dim=args.hidden_dim,
            device=args.device
        )
        results = engine.run_inference(dataloader)
    except Exception as e:
        print(f"[致命错误] 推理引擎崩溃: {e}")
        import traceback
        traceback.print_exc()
        return

    # =====================================================================
    # 阶段 4：热力图可视化
    # =====================================================================
    print("\n[4/4] 绘制并输出肿瘤热力图 (Heatmap)...")
    visualizer = HeatmapGenerator(patch_size=args.patch_size, alpha=args.alpha)
    if args.output_format in ["png", "both"]:
        visualizer.generate(
            results=results,
            thumb_img=thumb_img,
            downsample=downsample,
            save_path=heatmap_png_path
        )
    if args.output_format in ["tif", "both"]:
        visualizer.generate_pyramidal_tiff(
            results=results,
            level_0_dimensions=level_0_dimensions,
            save_path=heatmap_tif_path,
            compression=args.tif_compression,
            tile_size=args.tif_tile_size,
            colormap=args.heatmap_colormap,
        )

    # =====================================================================
    # 终点：性能统计
    # =====================================================================
    total_time = time.time() - total_start_time
    print("\n" + "="*60)
    print(f" 🎉 全流程执行完毕！")
    print(f" ⏱️  总耗时: {total_time:.2f} 秒")
    print(f" 📊  处理 Patch 数量: {len(valid_coords)}")
    print(f" 🚀  推理速度: {len(valid_coords) / max(total_time, 0.001):.2f} Patch/秒")
    if args.output_format in ["png", "both"]:
        print(f" 📁  PNG 热力图保存至: {heatmap_png_path}")
    if args.output_format in ["tif", "both"]:
        print(f" 📁  金字塔 TIF 热力图保存至: {heatmap_tif_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore") # 屏蔽第三方库的烦人警告
    main()
