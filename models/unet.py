import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class UNetEncoder(nn.Module):
    """Encoder-only, exposes skip connections + bottleneck for multimodal fusion."""
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c*2)
        self.enc3 = ConvBlock(c*2, c*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c*4, c*8)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        return b, (e1, e2, e3)  # bottleneck + skip connections


class UNetDecoder(nn.Module):
    def __init__(self, base_channels=32, out_channels=1):
        super().__init__()
        c = base_channels
        self.up3 = nn.ConvTranspose2d(c*8, c*4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c*8, c*4)
        self.up2 = nn.ConvTranspose2d(c*4, c*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c*4, c*2)
        self.up1 = nn.ConvTranspose2d(c*2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c*2, c)
        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, bottleneck_features, skips):
        e1, e2, e3 = skips
        d3 = self.up3(bottleneck_features)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out_conv(d1)
