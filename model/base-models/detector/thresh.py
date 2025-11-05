import torch, torch.nn as nn
import torch, torch.nn as nn

class ThresholdNet(nn.Module):
    def __init__(self, feat_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden,   1),      nn.Sigmoid()
        )

    def forward(self, global_feat):
        # global_feat: [1, feat_dim]
        return self.net(global_feat)  # scalar ∈ (0,1)

class TransformerQualityHead(nn.Module):
    def __init__(self, roi_dim, ctx_dim, num_heads=4):
        super().__init__()
        self.ctx_proj = nn.Linear(roi_dim, ctx_dim)
        self.roi_proj = nn.Linear(roi_dim, ctx_dim)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=ctx_dim, nhead=num_heads, batch_first=True
        )
        self.out = nn.Linear(ctx_dim, 1)

    def forward(self, roi_feats, global_feat):
        # roi_feats: [N, roi_dim]; global_feat: [1, roi_dim]
        ctx = self.ctx_proj(global_feat)           # [1,ctx_dim]
        roi = self.roi_proj(roi_feats)             # [N,ctx_dim]
        tokens = torch.cat([ctx, roi], dim=0)      # [N+1,ctx_dim]
        tokens = tokens.unsqueeze(0)               # [1,N+1,ctx_dim]
        out = self.transformer(tokens)[0,1:]       # [N,ctx_dim]
        q   = self.out(out)                        # [N,1]
        return torch.sigmoid(q)
