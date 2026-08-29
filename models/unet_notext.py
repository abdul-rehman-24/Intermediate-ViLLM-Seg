import torch.nn as nn

class UNetBaseline(nn.Module):
    def __init__(self, unet_encoder_cls, unet_decoder_cls, base_channels=32):
        super().__init__()
        self.encoder = unet_encoder_cls(in_channels=1, base_channels=base_channels)
        self.decoder = unet_decoder_cls(base_channels=base_channels, out_channels=1)

    def forward(self, images, prompts=None, device=None):
        # prompts/device ignore honge — signature training loop se compatible rakhne ke liye
        bottleneck, skips = self.encoder(images)
        return self.decoder(bottleneck, skips)
