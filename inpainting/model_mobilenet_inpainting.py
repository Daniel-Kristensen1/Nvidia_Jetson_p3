import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class MobileNetInpainting(nn.Module):
    """
    Inpainting model - based om MobileNetV2 encoder + simpel decoder
    Input: (B, 4, H, W) 3 channels picture + 1 channel mask
    Output: (B, 3, H, W) Reconstructed image
    """

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()

        # Get the MobileNetV2 as an encoder
        base = mobilenet_v2(pretrained=None)
        self.encoder = base.features # convolutionelle lag

        # We will replace the first conv, so we can get 4 channels instead of 3
        first_conv = self.encoder[0][0] # Conv2d first block
        new_conv = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )

        #PyTorch starts the new_cov weights automatic
        self.encoder[0][0] = new_conv


        # Simple decoder, which upsamples back to the original size
        # Encoder-output from MobileNetV2 has typical 1280 channels

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1), # H/16
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # H/8
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # H/4
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # H/2
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # H
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 4, H, W) # Masked image + mask
        """

        feats = self.encoder(x)
        out = self.decoder(feats)
        return out
