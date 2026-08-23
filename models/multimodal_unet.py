import torch
import torch.nn as nn
from models.unet import UNetEncoder, UNetDecoder
from models.text_encoder import TextEncoder
from models.cross_attention import CrossAttentionFusion

class MultimodalUNet(nn.Module):
    """
    Full pipeline: Image + Prompt -> Segmentation Logits

    images -> UNetEncoder -> bottleneck + skips
    prompts -> TextEncoder (frozen) -> token embeddings
    bottleneck + token embeddings -> CrossAttentionFusion -> language-grounded features
    language-grounded features + skips -> UNetDecoder -> logits
    """
    def __init__(self, base_channels=32, text_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                 attn_dim=256, num_heads=4):
        super().__init__()
        self.visual_encoder = UNetEncoder(in_channels=1, base_channels=base_channels)
        self.text_encoder = TextEncoder(model_name=text_model_name, freeze=True)
        self.cross_attention = CrossAttentionFusion(
            visual_channels=base_channels * 8,  # bottleneck channels = base*8
            text_dim=768,
            attn_dim=attn_dim,
            num_heads=num_heads
        )
        self.decoder = UNetDecoder(base_channels=base_channels, out_channels=1)

    def forward(self, images, prompts, device):
        bottleneck, skips = self.visual_encoder(images)
        token_embeds, attn_mask = self.text_encoder(prompts, device)
        fused = self.cross_attention(bottleneck, token_embeds, attn_mask)
        logits = self.decoder(fused, skips)
        return logits  # raw logits, no sigmoid — for BCEWithLogitsLoss
