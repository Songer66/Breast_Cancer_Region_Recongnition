import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Baseline: 原版线性分类头
# ==========================================
class PatchBinaryHead(nn.Module):
    def __init__(self, in_dim: int = 1280, hidden_dim: int = 512, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


# ==========================================
# 2. Model A: 深度非线性残差头 (ResPatchHead)
# ==========================================
class ResPatchHead(nn.Module):
    def __init__(self, in_dim: int = 1280, hidden_dim: int = 512, dropout: float = 0.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.act = nn.GELU()
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.act(h + self.res_block(h))
        return self.classifier(h).squeeze(1)


# ==========================================
# 3. Model B: 特征通道校准头 (SEPatchHead)
# ==========================================
class SEPatchHead(nn.Module):
    def __init__(self, in_dim: int = 1280, hidden_dim: int = 512, dropout: float = 0.5):
        super().__init__()
        reduction_dim = in_dim // 16 
        self.se_block = nn.Sequential(
            nn.Linear(in_dim, reduction_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduction_dim, in_dim),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.se_block(x)
        x_calibrated = x * attention
        return self.classifier(x_calibrated).squeeze(1)


# ==========================================
# 4. Model C: 超球面余弦分类头 (CosinePatchHead)
# ==========================================
class CosinePatchHead(nn.Module):
    def __init__(self, in_dim: int = 1280, hidden_dim: int = 512, dropout: float = 0.5, scale: float = 15.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.weight = nn.Parameter(torch.randn(1, hidden_dim))
        self.scale = scale 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlp(x)
        h_norm = F.normalize(h, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine_sim = F.linear(h_norm, w_norm)
        return (cosine_sim * self.scale).squeeze(1)


# ==========================================
# 5. Model D: TabNet 注意力头 (TabNetHead)
# ==========================================
class GatedLinearUnit(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)
    def forward(self, x):
        x = self.fc(x)
        return x[:, :x.shape[1]//2] * torch.sigmoid(x[:, x.shape[1]//2:])

class TabNetHead(nn.Module):
    def __init__(self, in_dim: int = 1280, hidden_dim: int = 256, dropout: float = 0.2, n_steps: int = 3):
        super().__init__()
        self.in_dim = in_dim
        self.n_steps = n_steps
        self.decision_dim = hidden_dim
        
        self.feature_transform = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            GatedLinearUnit(hidden_dim * 2, hidden_dim)
        )
        
        self.attentive_transformer = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, in_dim),
                nn.BatchNorm1d(in_dim),
                nn.Sigmoid()
            ) for _ in range(n_steps)
        ])
        
        self.classifier = nn.Linear(hidden_dim, 1)
        self.bn = nn.BatchNorm1d(in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        B = x.shape[0]
        
        # 初始化
        prior_scales = torch.ones((B, self.in_dim)).to(x.device)
        total_decision_out = torch.zeros((B, self.decision_dim)).to(x.device)
        
        for step in range(self.n_steps):
            # 这里的 total_decision_out 作为输入传给下一步
            m = self.attentive_transformer[step](total_decision_out if step > 0 else torch.zeros_like(total_decision_out))
            
            masked_x = x * m * prior_scales
            step_out = self.feature_transform(masked_x)
            
            # ❌ 错误写法: total_decision_out += F.relu(step_out)
            # ✅ 正确写法: 使用普通的加法，避免原地操作修改梯度路径上的变量
            total_decision_out = total_decision_out + F.relu(step_out)
            
            # 更新先验权重 (同样避免使用 *=)
            prior_scales = prior_scales * (0.9 - m)

        logits = self.classifier(total_decision_out).squeeze(1)
        return logits