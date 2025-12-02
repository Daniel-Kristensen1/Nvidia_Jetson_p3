import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataloader_mobilenetv2 import InpaintingDataset
from model_mobilenet_inpainting import MobileNetInpainting


# ==========================
# Konfiguration
# ==========================

EXTRA_EPOCHS = 20          # hvor mange ekstra epochs denne KØRSEL skal træne
BATCH_SIZE = 8             # justér til GPU / AI LAB
NUM_WORKERS = 4

CHECKPOINT_PATH = "mobilenet_inpaint_checkpoint.pth"
BEST_MODEL_PATH = "mobilenet_inpaint_best.pth"


# ==========================
# Masked L1 loss (samme som i train_inpainting.py)
# ==========================

def masked_l1_loss(pred: torch.Tensor,
                   target: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
    """
    pred, target: (B, 3, H, W)
    mask: (B, 1, H, W) med 1 = område der skal inpaintes
    """
    mask_3 = mask.expand_as(pred)  # (B, 3, H, W)

    diff = (pred - target) * mask_3

    loss_sum = diff.abs().sum()
    num_pixels = mask_3.sum()

    if num_pixels == 0:
        return torch.zeros((), device=pred.device)

    return loss_sum / num_pixels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) Tjek om checkpoint findes
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[RESUME] No checkpoint found at {CHECKPOINT_PATH}. Nothing to resume.")
        return

    print(f"[RESUME] Loading checkpoint from {CHECKPOINT_PATH} ...")

    # 2) Dataset + DataLoader
    train_ds = InpaintingDataset()
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"[RESUME] Dataset size: {len(train_ds)} samples")

    # 3) Model + optimizer
    model = MobileNetInpainting(in_channels=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 4) Load checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get("best_loss", float("inf"))

    print(f"[RESUME] Starting from epoch {start_epoch}")
    print(f"[RESUME] Previous best_loss: {best_loss:.4f}")

    end_epoch = start_epoch + EXTRA_EPOCHS

    # 5) Træningsloop (fortsættelse)
    for epoch in range(start_epoch, end_epoch):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            inputs = batch["input"].to(device)    # (B, 4, H, W)
            targets = batch["target"].to(device)  # (B, 3, H, W)
            mask = batch["mask"].to(device)       # (B, 1, H, W)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = masked_l1_loss(outputs, targets, mask)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # ---------- status og best-opdatering ----------
        if epoch_loss < best_loss:
            old_best = best_loss
            best_loss = epoch_loss

            print(
                f"[RESUME] ==> Epoch [{epoch+1}] - loss: {epoch_loss:.4f}  "
                f"<<< NEW BEST (prev best: {old_best:.4f} → new best: {best_loss:.4f})"
            )

            # overskriv bedste vægte
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"[RESUME] Saved NEW BEST model to: {BEST_MODEL_PATH}")
        else:
            print(
                f"[RESUME] ==> Epoch [{epoch+1}] - loss: {epoch_loss:.4f}  "
                f"(best so far: {best_loss:.4f})"
            )

        # Gem opdateret checkpoint (model + optimizer + best_loss)
        new_checkpoint = {
            "epoch": epoch + 1,               # næste epoch
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(new_checkpoint, CHECKPOINT_PATH)
        print(f"[RESUME] Checkpoint updated at {CHECKPOINT_PATH}")

    print("[RESUME] Finished extra training.")


if __name__ == "__main__":
    main()
