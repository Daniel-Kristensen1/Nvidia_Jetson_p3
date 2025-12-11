import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    2x (Conv2d + BatchNorm + ReLU)
    Bruges både i encoder og decoder.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetInpainting(nn.Module):
    """
    Klassisk U-Net til inpainting.
    Input:  (B, 4, H, W)  (masked RGB + mask)
    Output: (B, 3, H, W)  (rekonstrueret billede)

    Virker fint for H,W delelig med 16 (64, 128, 256, ...).
    """

    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32) -> None:
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, base_channels)          # H
        self.enc2 = DoubleConv(base_channels, base_channels * 2)    # H/2
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4) # H/4
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8) # H/8

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (H/16)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8,
            kernel_size=2, stride=2
        )# H/16 -> H/8
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4,
            kernel_size=2, stride=2
        )# H/8 -> H/4
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2,
            kernel_size=2, stride=2
        )# H/4 -> H/2
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels,
            kernel_size=2, stride=2
        )# H/2 -> H
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Sidste lag: 3 kanaler, sigmoid for [0,1]
        self.out_conv = nn.Conv2d(base_channels, 3, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x) # (B, C, H,   W)
        x2 = self.enc2(self.pool(x1)) # (B, 2C, H/2, W/2)
        x3 = self.enc3(self.pool(x2)) # (B, 4C, H/4, W/4)
        x4 = self.enc4(self.pool(x3)) # (B, 8C, H/8, W/8)

        # Bottleneck
        x5 = self.bottleneck(self.pool(x4)) # (B, 16C, H/16, W/16)

        # Decoder + skip connections
        d4 = self.up4(x5) # (B, 8C, H/8, W/8)
        d4 = torch.cat([d4, x4], dim=1) # (B, 16C, H/8, W/8)
        d4 = self.dec4(d4)

        d3 = self.up3(d4) # (B, 4C, H/4, W/4)
        d3 = torch.cat([d3, x3], dim=1) # (B, 8C, H/4, W/4)
        d3 = self.dec3(d3)

        d2 = self.up2(d3) # (B, 2C, H/2, W/2)
        d2 = torch.cat([d2, x2], dim=1) # (B, 4C, H/2, W/2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2) # (B, C, H, W)
        d1 = torch.cat([d1, x1], dim=1) # (B, 2C, H, W)
        d1 = self.dec1(d1)

        out = self.out_conv(d1) # (B, 3, H, W)
        out = self.out_act(out)
        return out
