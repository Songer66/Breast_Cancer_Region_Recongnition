import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple

try:
    import openslide
except ImportError:
    openslide = None
    print("[Warning] OpenSlide not installed. SVS reading will fail.")

class WSIReader:
    def __init__(self, wsi_path: str, patch_size: int = 512, tissue_thresh: float = 0.1): # 🟢 阈值降到 0.1
        self.wsi_path = wsi_path
        self.patch_size = patch_size
        self.tissue_thresh = tissue_thresh
        self.ext = os.path.splitext(wsi_path)[-1].lower()
        
        self.slide = None
        self.level_0_dimensions = (0, 0)
        self._init_slide()

    def _init_slide(self):
        if self.ext == '.svs':
            if openslide is None:
                raise RuntimeError("OpenSlide is required for .svs files.")
            self.slide = openslide.OpenSlide(self.wsi_path)
            self.level_0_dimensions = self.slide.dimensions
        elif self.ext == '.kfb':
            print(f"[Info] KFB file detected. Placeholder triggered for {self.wsi_path}")
            self.level_0_dimensions = (0, 0)
        else:
            raise ValueError(f"Unsupported WSI format: {self.ext}")

    def get_tissue_mask(self, mask_level: int = -1) -> Tuple[np.ndarray, float, Image.Image]:
        if self.ext == '.svs':
            if mask_level == -1:
                mask_level = max(0, self.slide.level_count - 2)
            
            thumb = self.slide.read_region((0, 0), mask_level, self.slide.level_dimensions[mask_level])
            thumb_rgb = np.array(thumb.convert('RGB'))
            downsample_factor = self.slide.level_downsamples[mask_level]
            
            # ========================================================
            # 🚀 升级版：灰度 (亮度) + HSV (色彩) 双通道解析
            # ========================================================
            gray = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1] # 提取饱和度通道
            
            # 轻微模糊去噪
            gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            sat_blurred = cv2.GaussianBlur(saturation, (5, 5), 0)
            
            # 1. 亮度条件：剔除纯白玻璃 (灰度 < 235 视为候选组织)
            _, mask_gray = cv2.threshold(gray_blurred, 235, 255, cv2.THRESH_BINARY_INV)
            
            # 2. 色彩条件：剔除灰色灰尘/阴影 (饱和度 > 10 视为候选组织)
            # 阈值 10 非常宽容，能保留极淡的粉色脂肪膜，但能完美杀掉纯灰色的灰尘
            _, mask_sat = cv2.threshold(sat_blurred, 10, 255, cv2.THRESH_BINARY)
            
            # 3. 双剑合璧：必须同时满足“不是纯白”且“有颜色”
            mask_combined = cv2.bitwise_and(mask_gray, mask_sat)
            
            # 动态计算形态学核大小
            max_dim = max(gray.shape[0], gray.shape[1])
            scale = max_dim / 2000.0
            
            # 4. 筑坝：闭运算连结脂肪膜
            k_close = max(5, int(15 * scale) | 1)
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
            mask_closed = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            
            # 5. 放水：漫水填充 (FloodFill) 提取被包围的脂肪气泡
            h, w = mask_closed.shape
            padded = np.zeros((h + 4, w + 4), dtype=np.uint8)
            padded[2:h+2, 2:w+2] = mask_closed
            
            ff_mask = np.zeros((h + 6, w + 6), dtype=np.uint8)
            cv2.floodFill(padded, ff_mask, (0, 0), 255)
            
            internal_holes = cv2.bitwise_not(padded)
            internal_holes = internal_holes[2:h+2, 2:w+2]
            
            # 6. 融合与清理
            final_mask = cv2.bitwise_or(mask_closed, internal_holes)
            
            k_open = max(3, int(7 * scale) | 1)
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
            # 增加一次开运算的力度，把落单的微小碎屑彻底抹掉
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
            
            return final_mask, downsample_factor, thumb.convert('RGB')
            
        elif self.ext == '.kfb':
            return np.zeros((100, 100)), 1.0, Image.new('RGB', (100, 100))

    def get_valid_patch_coordinates(self) -> Tuple[List[Tuple[int, int]], float]:
        if self.ext == '.kfb':
            return [], 1.0

        # 🟢 注意解包三个返回值
        mask, downsample, _ = self.get_tissue_mask()
        
        mask_patch_size = int(self.patch_size / downsample)
        if mask_patch_size <= 0: mask_patch_size = 1

        valid_coords = []
        mask_h, mask_w = mask.shape

        for y in range(0, mask_h, mask_patch_size):
            for x in range(0, mask_w, mask_patch_size):
                patch_mask = mask[y : y + mask_patch_size, x : x + mask_patch_size]
                tissue_ratio = np.count_nonzero(patch_mask) / (mask_patch_size * mask_patch_size)
                
                if tissue_ratio >= self.tissue_thresh:
                    level_0_x = int(x * downsample)
                    level_0_y = int(y * downsample)
                    valid_coords.append((level_0_x, level_0_y))

        print(f"[WSIReader] 找到 {len(valid_coords)} 个有效组织区域。")
        return valid_coords, downsample

    def read_patch(self, x: int, y: int) -> Image.Image:
        if self.ext == '.svs':
            patch = self.slide.read_region((x, y), 0, (self.patch_size, self.patch_size))
            return patch.convert('RGB')
        elif self.ext == '.kfb':
            return Image.new('RGB', (self.patch_size, self.patch_size))

    def close(self):
        if self.slide is not None:
            self.slide.close()