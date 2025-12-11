from PIL import Image
import torch
import torchvision.transforms as T
from typing import Tuple

def preprocess_image(img: Image.Image, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    PIL RGB -> tensor i formen (3, H, W), værdier i [0, 1].
    image_size: (H, W), fx (64, 64), (128, 128) eller (256, 256)
    """
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),  # (C,H,W) med værdier [0,1]
    ])
    return transform(img)


def preprocess_mask(mask_img: Image.Image, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    PIL maske -> tensor i formen (1, H, W), værdier i {0.0, 1.0}.
    Vi antager at maske = 1 dér, hvor der er hul/menneske.
    """
    mask_img = mask_img.convert("L")

    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),  # (1,H,W) med værdier [0,1]
    ])

    mask_tensor = transform(mask_img)
    mask_tensor = (mask_tensor > 0.5).float()  # binær maske
    return mask_tensor


def make_masked_image(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    image: (3, H, W), mask: (1, H, W) med 1 i hullerne.
    Vi sætter hullerne til 0 (sort), resten beholdes.
    """
    return image * (1.0 - mask)


def build_model_input(masked_image: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    """
    masked_image: (3, H, W)
    mask:        (1, H, W)
    return:      (4, H, W) = concat(image, mask)
    """
    return torch.cat([masked_image, mask], dim=0)
