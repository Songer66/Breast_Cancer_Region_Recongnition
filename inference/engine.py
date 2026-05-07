import torch
import torch.nn as nn
from contextlib import nullcontext
from tqdm import tqdm
from transformers import ViTModel
from typing import Optional, Dict
# 这里假设你的分类头在 model.py 中，名为 PatchBinaryHead
from model import PatchBinaryHead

class WSIInferenceEngine:
    def __init__(self, vit_weights_path: str, head_weights_path: str, hidden_dim: Optional[int] = None, device: str = 'cuda'):
        """
        初始化级联推理引擎
        :param vit_weights_path: HuggingFace ViT-H 的本地权重路径
        :param head_weights_path: 你自己训练的 best_model.pt 的路径
        :param hidden_dim: 必须与你训练时设置的 --hidden_dim 保持一致
        :param device: 推理设备，默认为 cuda
        """
        self.device = torch.device(device)
        print("=========================================")
        print(f" 🚀 初始化 WSI 推理引擎 (Device: {self.device})")
        print("=========================================")
        
        # 1. 加载 ViT-H (特征提取器)
        print("-> 正在加载 ViT-Huge 骨干网络...")
        self.vit = ViTModel.from_pretrained(vit_weights_path, local_files_only=True)
        self.vit.to(self.device)
        self.vit.eval()
        
        # 解析并加载你训练好的权重字典
        state_dict = torch.load(head_weights_path, map_location=self.device)
        # 如果你训练时用了 DDP，保存的字典 key 会带有 'module.' 前缀，需要剔除
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_key = k.replace('module.', '') if k.startswith('module.') else k
            clean_state_dict[clean_key] = v

        inferred_hidden_dim = self._infer_hidden_dim_from_state_dict(clean_state_dict)
        if hidden_dim is None:
            hidden_dim = inferred_hidden_dim
            print(f"-> 自动识别 hidden_dim = {hidden_dim}")
        elif hidden_dim != inferred_hidden_dim:
            raise ValueError(
                f"hidden_dim 不匹配：参数为 {hidden_dim}，但权重推断为 {inferred_hidden_dim}。"
                " 请移除 --hidden_dim 或改为与权重一致。"
            )

        # 2. 加载 分类头 (MLP)
        print("-> 正在加载 Patch 分类头...")
        self.head = PatchBinaryHead(in_dim=1280, hidden_dim=hidden_dim, dropout=0.0)
        self.head.load_state_dict(clean_state_dict)
        self.head.to(self.device)
        self.head.eval()
        
        print("✅ 级联模型加载完毕！")

    def _infer_hidden_dim_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> int:
        fc_key = "net.0.weight"
        if fc_key not in state_dict:
            raise KeyError(f"在分类头权重中未找到 {fc_key}，无法自动推断 hidden_dim。")
        tensor = state_dict[fc_key]
        if tensor.ndim != 2:
            raise ValueError(f"{fc_key} 维度异常：期望2维，实际{tensor.ndim}维。")
        return int(tensor.shape[0])

    @torch.no_grad()
    def run_inference(self, dataloader) -> list:
        """
        执行整张 WSI 的批量推理
        :return: 返回一个列表，元素格式为字典: [{'x': 1024, 'y': 2048, 'prob': 0.98}, ...]
        """
        results = []
        total_batches = len(dataloader)
        
        print(f"-> 开始执行滑动窗口推理，共 {total_batches} 个 Batch...")
        
        # 使用 tqdm 显示进度条
        for tensors, xs, ys in tqdm(dataloader, desc="WSI 推理中", unit="batch"):
            tensors = tensors.to(self.device, non_blocking=True)

            # CPU 上禁用 autocast，保持 FP32 精度且通常更快；CUDA 保持 FP16 加速
            if self.device.type == "cuda":
                amp_context = torch.autocast(device_type="cuda", dtype=torch.float16)
            else:
                amp_context = nullcontext()

            with amp_context:
                # 1. 提特征
                vit_outputs = self.vit(pixel_values=tensors)
                if vit_outputs.pooler_output is not None:
                    feats = vit_outputs.pooler_output
                else:
                    feats = vit_outputs.last_hidden_state[:, 0]

                # 2. 过分类头
                logits = self.head(feats)
                # 3. Sigmoid 将 logits 转换为 0~1 的概率
                probs = torch.sigmoid(logits).squeeze(-1)
            
            # 转移到 CPU 并转为 python 原生数据类型
            probs_np = probs.cpu().numpy()
            xs_np = xs.numpy()
            ys_np = ys.numpy()
            
            # 记录结果
            for i in range(len(probs_np)):
                results.append({
                    'x': int(xs_np[i]),
                    'y': int(ys_np[i]),
                    'prob': float(probs_np[i])
                })
                
        print(f"✅ WSI 推理完成！共处理了 {len(results)} 个 Patch。")
        return results
