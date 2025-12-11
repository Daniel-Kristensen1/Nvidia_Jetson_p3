import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional


class PartialConv2d(nn.Module):
    """
    Partial Convolution som tager højde for en binær maske.
    Input:  x: (B, C_in, H, W)
            mask: (B, 1, H, W) med 1 = gyldig pixel, 0 = hul
    Output: out: (B, C_out, H_out, W_out)
            out_mask: (B, 1, H_out, W_out)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        if isinstance(kernel_size, int):
            kh, kw = kernel_size, kernel_size
        else:
            kh, kw = kernel_size

        # Kernel til at tælle antal gyldige pixels i masken
        self.register_buffer(
            "mask_kernel",
            torch.ones(1, 1, kh, kw)
        )
        self.kernel_size = kh * kw
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: (B, C_in, H, W)
        mask: (B, 1, H, W) eller None (så antager vi alt gyldigt)
        """
        if mask is None:
            mask = torch.ones(
                (x.size(0), 1, x.size(2), x.size(3)),
                device=x.device,
                dtype=x.dtype,
            )

        # Nulstil maskede pixels
        x_masked = x * mask

        # Convolution på masked input
        out = self.conv(x_masked)

        # Hvor mange gyldige pixels var der i hvert kernel-vindue?
        with torch.no_grad():
            mask_sum = F.conv2d(
                mask,
                self.mask_kernel,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

            # Hvor der ingen gyldige pixels er, skal output være 0
            # og masken forbliver 0
            mask_sum_clamped = mask_sum.clone()
            mask_sum_clamped[mask_sum_clamped == 0] = 1.0

        # Normaliser output så det ikke bliver for svagt
        out = out * (self.kernel_size / mask_sum_clamped)

        # Lav en "gyldig pixel"-maske i samme dtype som mask/input
        valid_mask = (mask_sum > 0).to(out.dtype)

        # Hvis mask_sum == 0, så skal out være 0 (ingen info)
        out = out * valid_mask

        # Ny maske 1 hvor der er mindst én gyldig pixel i vinduet
        new_mask = valid_mask

        return out, new_mask


class PConvDoubleConv(nn.Module):
    """
    2x (PartialConv2d + BatchNorm + ReLU)
    Arbejder både på features og maske.
    """
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.pconv1 = PartialConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.pconv2 = PartialConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x, mask = self.pconv1(x, mask)
        x = self.bn1(x)
        x = self.relu1(x)

        x, mask = self.pconv2(x, mask)
        x = self.bn2(x)
        x = self.relu2(x)

        return x, mask


class PConvUNetInpainting(nn.Module):
    """
    Partial Convolution-baseret U-Net til inpainting.

    Input til forward:
        x: (B, 4, H, W)  = masked RGB (3) + mask (1)
    Output:
        (B, 3, H, W)     = rekonstrueret billede i [0,1]
    """
    def __init__(self,
                 in_channels: int = 4,
                 base_channels: int = 32) -> None:
        super().__init__()

        # ----- Encoder -----
        # Første lag: 3 RGB-kanaler, masken bruges separat
        self.enc1 = PConvDoubleConv(3, base_channels)           # H
        self.enc2 = PConvDoubleConv(base_channels, base_channels * 2)   # H/2
        self.enc3 = PConvDoubleConv(base_channels * 2, base_channels * 4)  # H/4
        self.enc4 = PConvDoubleConv(base_channels * 4, base_channels * 8)  # H/8

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck (H/16)
        self.bottleneck = PConvDoubleConv(base_channels * 8, base_channels * 16)

        # ----- Decoder -----
        self.up4 = nn.ConvTranspose2d(
            base_channels * 16, base_channels * 8,
            kernel_size=2, stride=2
        )  # H/16 -> H/8
        self.dec4 = PConvDoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8, base_channels * 4,
            kernel_size=2, stride=2
        )  # H/8 -> H/4
        self.dec3 = PConvDoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2,
            kernel_size=2, stride=2
        )  # H/4 -> H/2
        self.dec2 = PConvDoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels,
            kernel_size=2, stride=2
        )  # H/2 -> H
        self.dec1 = PConvDoubleConv(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, 3, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 4, H, W) = [masked RGB (3 kanaler), mask (1 kanal)]
        """
        # Split input i billede og HUL-maske (1 = hul, 0 = baggrund)
        img = x[:, :3, :, :]        # (B,3,H,W), masked_image
        hole_mask = x[:, 3:4, :, :] # (B,1,H,W), 1 = hul

        # PartialConv skal have 1 = gyldig pixel, 0 = hul
        valid_mask = 1.0 - hole_mask

        # ===== ENCODER =====
        x1, m1 = self.enc1(img, valid_mask)
        x2_in = self.pool(x1)
        m2_in = F.max_pool2d(m1, kernel_size=2, stride=2)
        x2, m2 = self.enc2(x2_in, m2_in)

        x3_in = self.pool(x2)
        m3_in = F.max_pool2d(m2, kernel_size=2, stride=2)
        x3, m3 = self.enc3(x3_in, m3_in)

        x4_in = self.pool(x3)
        m4_in = F.max_pool2d(m3, kernel_size=2, stride=2)
        x4, m4 = self.enc4(x4_in, m4_in)

        x5_in = self.pool(x4)
        m5_in = F.max_pool2d(m4, kernel_size=2, stride=2)
        x5, m5 = self.bottleneck(x5_in, m5_in)


        # ===== DECODER =====
        # Niveau 4
        d4 = self.up4(x5)                           # (B, 8C, H/8, W/8)
        m_d4 = F.interpolate(m5, scale_factor=2, mode="nearest")
        # kombiner masken fra encoder og decoder
        m_d4 = torch.max(m_d4, m4)
        d4 = torch.cat([d4, x4], dim=1)             # (B,16C, H/8, W/8)
        d4, m_d4 = self.dec4(d4, m_d4)

        # Niveau 3
        d3 = self.up3(d4)                           # (B, 4C, H/4, W/4)
        m_d3 = F.interpolate(m_d4, scale_factor=2, mode="nearest")
        m_d3 = torch.max(m_d3, m3)
        d3 = torch.cat([d3, x3], dim=1)             # (B, 8C, H/4, W/4)
        d3, m_d3 = self.dec3(d3, m_d3)

        # Niveau 2
        d2 = self.up2(d3)                           # (B, 2C, H/2, W/2)
        m_d2 = F.interpolate(m_d3, scale_factor=2, mode="nearest")
        m_d2 = torch.max(m_d2, m2)
        d2 = torch.cat([d2, x2], dim=1)             # (B, 4C, H/2, W/2)
        d2, m_d2 = self.dec2(d2, m_d2)

        # Niveau 1
        d1 = self.up1(d2)                           # (B, C, H, W)
        m_d1 = F.interpolate(m_d2, scale_factor=2, mode="nearest")
        m_d1 = torch.max(m_d1, m1)
        d1 = torch.cat([d1, x1], dim=1)             # (B, 2C, H, W)
        d1, m_d1 = self.dec1(d1, m_d1)

        out = self.out_conv(d1)
        out = self.out_act(out)
        return out
