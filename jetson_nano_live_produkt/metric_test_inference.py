import cv2
import torch
import time
from jtop import jtop
import numpy as np
from pathlib import Path
from datetime import datetime


import select_model
import realtime_inpainting

import subprocess
import re


START_FILE = Path("START_LOGGING")
STOP_FILE = Path("STOP_LOGGING")

def load_images_from_dir(img_dir: Path):
    """
    ChatGPT wrote this function to load all images from a directory into memory.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".jfif"}
    image_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])

    if not image_paths:
        raise RuntimeError(f"No images found in {img_dir}")

    images = []
    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Warning: could not read {p}, skipping")
            continue
        images.append((p.name, img))

    return images



def test():
    
    # --------------
    # --- Device ---
    # --------------
    if not torch.cuda.is_available():
        print("CUDA ikke tilgængelig. Dette script skal køres på en Jetson-enhed.")
        return

    device = torch.device("cuda")
    print("➡ Device:", device)
    
    # -----------------------
    # --- Model selection ---
    # -----------------------
    u2net_weights = select_model.find_u2net_model()
    inpaint_weights, inpaint_type, fp_bits = select_model.select_model()
    use_fp16 = (fp_bits == 16 and device.type == "cuda")
    print(f"Precision: FP{fp_bits} (FP16 aktiv: {use_fp16})")
    
    
    # ------------------
    # --- Image size ---
    # ------------------
    image_size = realtime_inpainting.choose_image_size()
    print("Using image size:", image_size)
    
    # -------------------
    # --- Load models ---
    # -------------------
    net_seg, net_inpaint = realtime_inpainting.load_models(
        device,
        u2net_weights,
        inpaint_weights,
        inpaint_type,
        use_fp16=use_fp16,
    )

    # -----------------------
    # --- Load Test image ---
    # -----------------------
    image_dir = Path("/home/aaunano/Desktop/projekt_p3/jetson_nano_live_produkt/1000_images")
    images = load_images_from_dir(image_dir)

    print(f"Loaded {len(images)} images for benchmark.")

    # ---------------------------------------------
    # --- Test loop (WARMUP and BENCHMARK test) ---
    # ---------------------------------------------
    

    WARMUP_ITERS = 10
    BENCH_ITERS = 100
    print(f"Running {WARMUP_ITERS} warmup iterations...")
    for i in range(WARMUP_ITERS):
        _, img = images[i]
        realtime_inpainting.run_pipeline(
            img,
            net_seg,
            net_inpaint,
            device,
            image_size,
            use_fp16=use_fp16,
        )
    torch.cuda.synchronize()
    print("Warmup complete.")
    

    # --- BENCHMARK ---

    # Metrics storage
    latencies = []



    # --- BENCHMARK Loop---

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latency_path = results_dir / f"latency_{timestamp}.csv"

    latency_f = open(latency_path, "w")
    latency_f.write("image,latency_ms\n")


    print("Starting benchmark...")
    START_FILE.touch()

    for idx, (name, img) in enumerate(images):
        t0 = time.perf_counter()

        realtime_inpainting.run_pipeline(
            img,
            net_seg,
            net_inpaint,
            device,
            image_size,
            use_fp16=use_fp16,
        )
        torch.cuda.synchronize()

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        latencies.append(latency_ms)
        latency_f.write(f"{name},{latency_ms:.3f}\n")
        latency_f.flush()

        print(f"[{idx+1}/{len(images)}] {name} | latency={latency_ms:.2f} ms")

    STOP_FILE.touch()
    latency_f.close()

if __name__ == "__main__":
    test()
