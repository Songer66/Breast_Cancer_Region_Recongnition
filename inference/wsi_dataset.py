import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Tuple

# 动态导入我们刚才写的 WSIReader
from wsi_reader import WSIReader

class WSIDynamicPatchDataset(Dataset):
    """
    针对 WSI 优化的动态切图 Dataset。
    支持多线程 (num_workers > 0) 安全读取。
    """
    def __init__(self, wsi_path: str, valid_coords: List[Tuple[int, int]], patch_size: int = 512):
        """
        :param wsi_path: WSI 切片路径
        :param valid_coords: WSIReader 算出来的有效组织坐标列表 [(x1, y1), (x2, y2), ...]
        :param patch_size: Level 0 切块大小
        """
        self.wsi_path = wsi_path
        self.coords = valid_coords
        self.patch_size = patch_size
        
        # ⚠️ 关键：这里初始化为 None，绝不能在这里实例化 WSIReader
        self.reader = None
        
        # ViT-H 的标准预处理：512x512 -> Resize 224x224 -> ToTensor -> Normalize
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            # 这里的均值方差需与你 ViT-H 训练/提取特征时保持绝对一致
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        # ⚠️ 进程隔离的安全实例化：
        # 当 PyTorch 的 Worker 第一次调用 __getitem__ 时，它会在自己的独立进程里创建 Reader
        if self.reader is None:
            # threshold 设为 0 即可，因为坐标已经是我们之前严格筛选过的了，这里只负责读图
            self.reader = WSIReader(self.wsi_path, patch_size=self.patch_size, tissue_thresh=0.0)

        # 1. 拿到当前索引的坐标
        x, y = self.coords[idx]
        
        # 2. 从高分辨率 Level 0 动态抠图 (非常快)
        patch_pil = self.reader.read_patch(x, y)
        
        # 3. 预处理转换为 Tensor [3, 224, 224]
        tensor = self.transform(patch_pil)
        
        # 4. 返回 Tensor 以及它对应的原始坐标 (供后续画热力图使用)
        return tensor, x, y

    def __del__(self):
        # 析构时安全关闭 slide
        if self.reader is not None:
            self.reader.close()