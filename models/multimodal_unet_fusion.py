import torch
import torch.nn as nn

class MultimodalUNetFusion(nn.Module):
    def __init__(self, unet_encoder_cls, unet_decoder_cls, text_encoder_cls, fusion_cls,
                 base_channels=32, text_model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
        super().__init__()
        self.visual_encoder = unet_encoder_cls(in_channels=1, base_channels=base_channels)
        self.text_encoder = text_encoder_cls(model_name=text_model_name, freeze=True)
        self.fusion = fusion_cls(visual_channels=base_channels * 8, text_dim=768)
        self.decoder = unet_decoder_cls(base_channels=base_channels, out_channels=1)

    def forward(self, images, prompts, device):
        bottleneck, skips = self.visual_encoder(images)
        token_embeds, attn_mask = self.text_encoder(prompts, device)
        fused = self.fusion(bottleneck, token_embeds, attn_mask)
        logits = self.decoder(fused, skips)
        return logits
