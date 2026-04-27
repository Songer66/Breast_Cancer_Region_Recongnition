import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple

try:
    import pyvips
except ImportError:
    pyvips = None

class HeatmapGenerator:
    def __init__(self, patch_size: int = 512, alpha: float = 0.5):
        """
        初始化热力图生成器
        :param patch_size: Level 0 的切块大小
        :param alpha: 热力图的透明度 (0.0 完全透明, 1.0 完全不透明)
        """
        self.patch_size = patch_size
        self.alpha = alpha

    def generate(self, results: list, thumb_img: Image.Image, downsample: float, save_path: str):
        print(f"-> 正在生成热力图，共有 {len(results)} 个有效区块...")
        
        # 将 PIL 缩略图转为 OpenCV 格式 (BGR)
        thumb_cv = cv2.cvtColor(np.array(thumb_img), cv2.COLOR_RGB2BGR)
        h, w, _ = thumb_cv.shape
        
        # 将画布初始化为 -1.0
        heatmap_canvas = np.full((h, w), -1.0, dtype=np.float32)
        
        # 计算在缩略图上，一个 Patch 应该画多大
        box_size = max(1, int(self.patch_size / downsample))
        
        # 1. 填色
        for res in results:
            x_thumb = int(res['x'] / downsample)
            y_thumb = int(res['y'] / downsample)
            prob = res['prob']
            heatmap_canvas[y_thumb : y_thumb + box_size, x_thumb : x_thumb + box_size] = prob
            
        # 提取 Mask
        valid_tissue_mask = (heatmap_canvas >= 0).astype(np.float32)
        safe_canvas = np.maximum(heatmap_canvas, 0.0)
        
        # =======================================================
        # 🚀 升级版 2D 概率矩阵空间平滑算法 (彻底消除马赛克与椒盐噪声)
        # =======================================================
        # 核心：高斯核必须足够大，至少是单个 block 的 2.5 倍，才能让相邻的色块充分融合
        k_size = int(box_size * 2.5)
        
        # 确保核大小是奇数
        if k_size % 2 == 0:
            k_size += 1
        # 设定一个保底大小
        k_size = max(5, k_size)
        
        print(f"   * 动态平滑核大小: {k_size}x{k_size} (Block size: {box_size})")

        # 对概率矩阵和 Mask 执行大尺度高斯平滑
        blurred_canvas = cv2.GaussianBlur(safe_canvas, (k_size, k_size), 0)
        blurred_mask = cv2.GaussianBlur(valid_tissue_mask, (k_size, k_size), 0)
        
        # 归一化补偿 (防止边缘概率被稀释)
        smoothed_canvas = blurred_canvas / (blurred_mask + 1e-5)
        
        # 截断回真实组织区域内
        final_safe_canvas = smoothed_canvas * valid_tissue_mask
        # =======================================================
        
        # 转换为伪彩色 (Jet Colormap)
        heatmap_uint8 = np.clip(final_safe_canvas * 255, 0, 255).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # 融合
        blend_mask = (valid_tissue_mask > 0).astype(np.uint8)
        mask_3d = np.repeat(blend_mask[:, :, np.newaxis], 3, axis=2)
        blended = np.where(
            mask_3d == 1,
            cv2.addWeighted(thumb_cv, 1 - self.alpha, color_heatmap, self.alpha, 0),
            thumb_cv
        )
        
        # 保存图片
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, blended)
        print(f"✅ 柔和版热力图已成功保存至: {save_path}")

    def _build_patch_probability_grid(self, results: list, level_0_dimensions: Tuple[int, int]) -> np.ndarray:
        level_0_w, level_0_h = level_0_dimensions
        grid_w = int(np.ceil(level_0_w / self.patch_size))
        grid_h = int(np.ceil(level_0_h / self.patch_size))

        grid = np.full((grid_h, grid_w), -1.0, dtype=np.float32)
        count = np.zeros((grid_h, grid_w), dtype=np.uint16)

        for res in results:
            gx = int(res["x"]) // self.patch_size
            gy = int(res["y"]) // self.patch_size
            if 0 <= gx < grid_w and 0 <= gy < grid_h:
                prob = float(res["prob"])
                if grid[gy, gx] < 0:
                    grid[gy, gx] = prob
                    count[gy, gx] = 1
                else:
                    # 对重复命中的网格做在线平均，避免覆盖带来的随机性
                    cnt = int(count[gy, gx])
                    grid[gy, gx] = (grid[gy, gx] * cnt + prob) / (cnt + 1)
                    count[gy, gx] = cnt + 1

        return grid

    def generate_pyramidal_tiff(
        self,
        results: list,
        level_0_dimensions: Tuple[int, int],
        save_path: str,
        compression: str = "jpeg",
        tile_size: int = 256,
        colormap: str = "jet",
    ):
        if pyvips is None:
            raise RuntimeError(
                "pyvips 未安装，无法输出金字塔 TIF。请先安装 libvips 与 pyvips。"
            )

        print(f"-> 正在生成金字塔 TIF 热力图，共有 {len(results)} 个有效区块...")
        level_0_w, level_0_h = level_0_dimensions
        grid = self._build_patch_probability_grid(results, level_0_dimensions)

        valid_mask = (grid >= 0).astype(np.float32)
        safe_grid = np.maximum(grid, 0.0)

        k_size = 5
        smoothed_grid = cv2.GaussianBlur(safe_grid, (k_size, k_size), 0)
        smoothed_mask = cv2.GaussianBlur(valid_mask, (k_size, k_size), 0)
        final_grid = (smoothed_grid / (smoothed_mask + 1e-5)) * valid_mask

        heatmap_uint8 = np.clip(final_grid * 255, 0, 255).astype(np.uint8)
        if colormap == "gray":
            color_grid = cv2.cvtColor(heatmap_uint8, cv2.COLOR_GRAY2RGB)
        else:
            color_grid = cv2.cvtColor(
                cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET),
                cv2.COLOR_BGR2RGB
            )

        zero_mask = (valid_mask == 0)
        color_grid[zero_mask] = 0

        grid_h, grid_w = color_grid.shape[:2]
        vips_img = pyvips.Image.new_from_memory(
            color_grid.tobytes(),
            grid_w,
            grid_h,
            3,
            "uchar"
        )

        scale_x = max(1.0, level_0_w / float(grid_w))
        scale_y = max(1.0, level_0_h / float(grid_h))
        upscaled = vips_img.resize(scale_x, vscale=scale_y, kernel="nearest")
        upscaled = upscaled.crop(0, 0, level_0_w, level_0_h)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        upscaled.tiffsave(
            save_path,
            tile=True,
            tile_width=tile_size,
            tile_height=tile_size,
            pyramid=True,
            compression=compression,
            bigtiff=True,
            Q=90
        )
        print(f"✅ 金字塔 TIF 热力图已成功保存至: {save_path}")
