import os
import cv2
import numpy as np
from PIL import Image

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