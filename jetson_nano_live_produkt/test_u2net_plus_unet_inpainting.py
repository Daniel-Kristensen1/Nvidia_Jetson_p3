from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T

from u2net import U2NET
from data_loader import RescaleT, ToTensorLab
from model_unet_inpainting import UNetInpainting
from dataset_manager import (
    preprocess_image,
    preprocess_mask,
    make_masked_image,
    build_model_input,
)
import select_model


# ---------------------------------------
# RET DISSE TO STIER
# ---------------------------------------
IMAGE_PATH = Path("test_image.png")
# vægtene findes automatisk via select_model:
U2NET_WEIGHTS = select_model.find_u2net_model()
UNET_WEIGHTS  = select_model.select_model()

# ---------------------------------------
# HARDKODET IMAGE SIZE
# vælg én af disse:
# IMAGE_SIZE = (64, 64)
# IMAGE_SIZE = (128, 128)
IMAGE_SIZE = (256, 256)
# ---------------------------------------


def normalize_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi + 1e-8)


def make_u2net_input(pil_img: Image.Image):
    """Samme input-format som realtime-pipelinen."""
    rgb = np.array(pil_img)
    label = np.zeros((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)
    sample = {"imidx": np.array([0]), "image": rgb, "label": label}

    transforms = T.Compose([RescaleT(320), ToTensorLab(flag=0)])
    sample_t = transforms(sample)
    return sample_t["image"].unsqueeze(0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("➡ Device:", device)

    # 1) Load image
    img = Image.open(IMAGE_PATH).convert("RGB")

    # 2) Load U2NET
    net_seg = U2NET(3, 1).to(device)
    net_seg.load_state_dict(torch.load(U2NET_WEIGHTS, map_location=device))
    net_seg.eval()

    # 3) Run U2NET → mask
    u2_input = make_u2net_input(img).float().to(device)
    with torch.no_grad():
        d1, *_ = net_seg(u2_input)
        pred = normalize_pred(d1[:, 0, :, :])

    pred_np = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
    mask_img = Image.fromarray(pred_np)

    # 4) Preprocess image & mask for UNet
    image_tensor = preprocess_image(img, IMAGE_SIZE)
    mask_tensor  = preprocess_mask(mask_img, IMAGE_SIZE)

    # 5) Dilate mask (same as realtime)
    mask_tensor = F.max_pool2d(
        mask_tensor.unsqueeze(0),
        kernel_size=7,
        stride=1,
        padding=3
    ).squeeze(0)

    masked_image = make_masked_image(image_tensor, mask_tensor)
    model_input  = build_model_input(masked_image, mask_tensor).unsqueeze(0).to(device)

    # 6) Load UNet inpainting model
    net_inpaint = UNetInpainting(in_channels=4, base_channels=32).to(device)
    state = torch.load(UNET_WEIGHTS, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    net_inpaint.load_state_dict(state)
    net_inpaint.eval()

    # 7) Run inpainting
    with torch.no_grad():
        output = net_inpaint(model_input)[0].cpu().clamp(0, 1)

    # 8) Convert for plotting
    orig_np   = np.array(img)
    mask_vis  = mask_tensor.squeeze().cpu().numpy()
    masked_np = (masked_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    out_np    = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # 9) Plot
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].imshow(orig_np)
    axs[0].set_title("Original")

    axs[1].imshow(mask_vis, cmap="gray")
    axs[1].set_title("Dilated U2NET Mask")

    axs[2].imshow(masked_np)
    axs[2].set_title("Masked Input")

    axs[3].imshow(out_np)
    axs[3].set_title("Inpainting Result")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print("\nTest complete.")


if __name__ == "__main__":
    main()
