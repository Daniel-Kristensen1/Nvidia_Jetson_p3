import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms as T

from u2net import U2NET
from data_loader import RescaleT, ToTensorLab

from model_unet_inpainting import UNetInpainting
from model_mobilenet_inpainting import MobileNetInpainting
from model_pconv_unet_inpainting import PConvUNetInpainting

from dataset_manager import (
    preprocess_image,
    preprocess_mask,
    make_masked_image,
    build_model_input,
)

import select_model


def choose_image_size():
    """
    Lad brugeren vælge mellem 64x64, 128x128 eller 256x256.
    """
    sizes = {
        1: (64, 64),
        2: (128, 128),
        3: (256, 256),
    }

    print("Select image size:")
    print("1: 64 x 64")
    print("2: 128 x 128")
    print("3: 256 x 256")
    choice = int(input("Enter number (1-3): "))

    return sizes[choice]


def create_inpaint_model(model_type, device):
    """
    Opret den rigtige inpainting-model ud fra model_type:
      "mobilenet" -> MobileNetInpainting
      "unet"      -> UNetInpainting
      "pconv"     -> PConvUNetInpainting
    """
    if model_type == "mobilenet":
        model = MobileNetInpainting(in_channels=4).to(device)
    elif model_type == "unet":
        model = UNetInpainting(in_channels=4, base_channels=32).to(device)
    elif model_type == "pconv":
        model = PConvUNetInpainting(in_channels=4, base_channels=32).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def load_models(device, u2net_weights, inpaint_weights, inpaint_type, use_fp16=False):
    print("Loader modeller...")

    # --- U2NET (segmentering) ---
    net_seg = U2NET(3, 1).to(device)
    net_seg.load_state_dict(torch.load(u2net_weights, map_location=device))
    net_seg.eval()

    # --- Inpainting-model (mobilenet / unet / pconv) ---
    net_inpaint = create_inpaint_model(inpaint_type, device)
    state = torch.load(inpaint_weights, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    net_inpaint.load_state_dict(state)
    net_inpaint.eval()


    print("Modeller klar.")
    return net_seg, net_inpaint


def normalize_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    return (d - mi) / (ma - mi + 1e-8)


def frame_to_u2net_input(frame_rgb):
    label = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1], 1), dtype=np.float32)
    sample = {"imidx": np.array([0]), "image": frame_rgb, "label": label}
    u2_transforms = T.Compose([RescaleT(320), ToTensorLab(flag=0)])
    sample_t = u2_transforms(sample)
    return sample_t["image"].unsqueeze(0)  # (1,3,H,W)


def run_pipeline(frame, net_seg, net_inpaint, device, image_size, use_fp16=False):
    """
    frame (BGR) → U2NET maske → dilate → inpainting model → output (BGR)
    """
    # ----- BGR -> RGB -----
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # ----- U2NET (segmentering) -----
    u2_input = frame_to_u2net_input(rgb)  # (1,3,H,W), float32

    # ----- Fp16/fp32 handeling -----
    dtype = torch.float16 if use_fp16 else torch.float32

    u2_input = u2_input.to(device=device, dtype=dtype)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
        d1, *_ = net_seg(u2_input)     # U2NET output
        pred = normalize_pred(d1[:, 0, :, :])

    pred_np = (pred.squeeze().float().cpu().numpy() * 255.0).astype(np.uint8)
    mask_img = Image.fromarray(pred_np)

    # ----- Preprocess til inpainting (CPU, FP32) -----
    image_tensor = preprocess_image(pil_img, image_size)   # (3,H,W), float32 CPU
    mask_tensor  = preprocess_mask(mask_img, image_size)   # (1,H,W), float32 CPU

    # Dilate maske (stadig float32 CPU)
    mask_tensor = F.max_pool2d(
        mask_tensor.unsqueeze(0),
        kernel_size=7,
        stride=1,
        padding=3,
    ).squeeze(0)

    # Maskér billedet og byg model-input (4 kanaler: 3 RGB + 1 maske)
    masked_image = make_masked_image(image_tensor, mask_tensor)
    model_input  = build_model_input(masked_image, mask_tensor).unsqueeze(0)  # (1,4,H,W)

    model_input = model_input.to(device=device, dtype=dtype)

    # ----- Inpainting -----
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_fp16):
        output = net_inpaint(model_input)[0]


    # Tilbage til CPU + uint8 BGR for visning
    output = output.clamp(0, 1)
    out_np = (
        output.permute(1, 2, 0)
            .detach()
            .cpu()
            .float()
            .numpy() * 255.0
    ).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    return out_bgr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("➡ Device:", device)

    # Vælg modeller + FP
    u2net_weights = select_model.find_u2net_model()
    inpaint_weights, inpaint_type, fp_bits = select_model.select_model()
    use_fp16 = (fp_bits == 16 and device.type == "cuda")
    print(f"Precision: FP{fp_bits} (FP16 aktiv: {use_fp16})")

    # Vælg billedstørrelse
    image_size = choose_image_size()
    print("Using image size:", image_size)

    # Load modeller
    net_seg, net_inpaint = load_models(
        device,
        u2net_weights,
        inpaint_weights,
        inpaint_type,
        use_fp16=use_fp16,
    )

    # Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kunne ikke åbne webcam!")
        return

    print("Realtime pipeline kører. Tryk 'q' for stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = run_pipeline(
            frame,
            net_seg,
            net_inpaint,
            device,
            image_size,
            use_fp16=use_fp16,
        )


        display = cv2.resize(result_frame, (512, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Realtime Inpainting", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
