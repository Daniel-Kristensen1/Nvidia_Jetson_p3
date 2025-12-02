import torch
import matplotlib.pyplot as plt
import random

from dataloader_mobilenetv2 import InpaintingDataset
from model_mobilenet_inpainting import MobileNetInpainting


# Hvor mange tilfældige samples du vil se
NUM_SAMPLES = 10

# Hvilken model du vil teste
BEST_MODEL_PATH = r"C:\Users\alext\Desktop\weights\256_udenstraf\mobilenet_inpaint_best_val.pth"   # skift til 64/128/256-varianten


def tensor_to_img(t: torch.Tensor):
    """
    t: (C, H, W) med værdier [0,1]
    return: (H, W, C) numpy-array klar til plt.imshow
    """
    t = t.detach().cpu().clamp(0, 1)
    return t.permute(1, 2, 0).numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset
    dataset = InpaintingDataset()
    print(f"Dataset size: {len(dataset)}")

    # Model
    model = MobileNetInpainting(in_channels=4).to(device)
    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Vælg tilfældige unikke indices
    num_samples = min(NUM_SAMPLES, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    # Lav figur: num_samples rækker, 3 kolonner
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    # Hvis num_samples == 1, sørg for at axes er 2D
    if num_samples == 1:
        axes = axes.reshape(1, 3)

    for row, idx in enumerate(indices):
        sample = dataset[idx]

        input_4c = sample["input"]      # (4, H, W)
        target   = sample["target"]     # (3, H, W)

        # Masked input (kun RGB)
        masked_rgb = input_4c[:3, :, :]

        # Kør model
        with torch.no_grad():
            x = input_4c.unsqueeze(0).to(device)   # (1, 4, H, W)
            out = model(x).squeeze(0)             # (3, H, W)
            out = out.cpu()

        # Konverter til billeder
        img_masked = tensor_to_img(masked_rgb)
        img_out    = tensor_to_img(out)
        img_target = tensor_to_img(target)

        # Plot i rækken
        ax0, ax1, ax2 = axes[row]

        ax0.imshow(img_masked)
        ax0.set_title("Masked Input")
        ax0.axis("off")

        ax1.imshow(img_out)
        ax1.set_title("Model Output")
        ax1.axis("off")

        ax2.imshow(img_target)
        ax2.set_title("Target Image")
        ax2.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

