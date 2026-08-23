import torch
import torch.nn as nn

class SimpleFusion(nn.Module):
    def __init__(self, visual_channels=256, text_dim=768):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, visual_channels)
        self.fusion_conv = nn.Conv2d(visual_channels * 2, visual_channels, kernel_size=1)

    def forward(self, visual_features, text_token_embeds, attn_mask):
        B, C, H, W = visual_features.shape
        mask_expanded = attn_mask.unsqueeze(-1).float()
        text_pooled = (text_token_embeds * mask_expanded).sum(1) / mask_expanded.sum(1)
        text_feat = self.text_proj(text_pooled)
        text_feat = text_feat.unsqueeze(-1).unsqueeze(-1)
        text_feat = text_feat.expand(-1, -1, H, W)
        fused = torch.cat([visual_features, text_feat], dim=1)
        fused = self.fusion_conv(fused)
        return fused
