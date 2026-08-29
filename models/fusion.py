import torch
import torch.nn as nn

class SimpleFusion(nn.Module):
    """Text embedding ko spatially broadcast karke visual features ke sath concatenate karta hai."""
    def __init__(self, visual_channels, text_dim=768, fused_channels=None):
        super().__init__()
        fused_channels = fused_channels or visual_channels
        # Pooled text embedding ko visual_channels dimension mein project karo
        self.text_proj = nn.Linear(text_dim, visual_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(visual_channels * 2, fused_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, visual_features, token_embeds, attn_mask):
        # token_embeds: [B, L, D] -> mean-pool over valid tokens (attn_mask se) -> [B, D]
        mask = attn_mask.unsqueeze(-1).float()
        pooled_text = (token_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        text_feat = self.text_proj(pooled_text)          # [B, C]
        B, C = text_feat.shape
        _, _, H, W = visual_features.shape
        text_feat = text_feat.view(B, C, 1, 1).expand(-1, -1, H, W)  # [B, C, H, W]

        fused = torch.cat([visual_features, text_feat], dim=1)       # [B, 2C, H, W]
        return self.fusion_conv(fused)                                # [B, C, H, W]
