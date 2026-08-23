import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """
    Visual features -> Query
    Text tokens     -> Key, Value
    Uses PyTorch's built-in MultiheadAttention.
    """
    def __init__(self, visual_channels=256, text_dim=768, attn_dim=256, num_heads=4):
        super().__init__()
        self.visual_proj = nn.Linear(visual_channels, attn_dim)
        self.text_proj_k = nn.Linear(text_dim, attn_dim)
        self.text_proj_v = nn.Linear(text_dim, attn_dim)

        self.attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(attn_dim, visual_channels)
        self.norm = nn.LayerNorm(visual_channels)

    def forward(self, visual_features, text_token_embeds, text_attn_mask):
        B, C, H, W = visual_features.shape

        # Visual: [B, C, H, W] -> [B, H*W, C] (sequence of spatial locations)
        visual_seq = visual_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        Q = self.visual_proj(visual_seq)                           # [B, H*W, attn_dim]

        K = self.text_proj_k(text_token_embeds)  # [B, L, attn_dim]
        V = self.text_proj_v(text_token_embeds)  # [B, L, attn_dim]

        # key_padding_mask: True = ignore (opposite convention of attention_mask)
        key_padding_mask = (text_attn_mask == 0)

        attn_out, _ = self.attn(Q, K, V, key_padding_mask=key_padding_mask)  # [B, H*W, attn_dim]
        attn_out = self.out_proj(attn_out)  # [B, H*W, C]

        # Reshape back to spatial: [B, H*W, C] -> [B, C, H, W]
        attn_out = attn_out.permute(0, 2, 1).reshape(B, C, H, W)

        # Residual connection + norm (stabilizes training)
        out = visual_features + attn_out
        out = out.permute(0, 2, 3, 1)  # [B, H, W, C] for LayerNorm over C
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # back to [B, C, H, W]

        return out
