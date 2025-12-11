import cv2
import torch
import time
from jtop import jtop

import select_model
import realtime_inpainting


def main():
    if not torch.cuda.is_available():
        print("❌ CUDA ikke tilgængelig. Dette script skal køres på en Jetson-enhed.")
        return

    device = torch.device("cuda")
    print("➡ Device:", device)

    # --- Model selection ---
    u2net_weights = select_model.find_u2net_model()
    inpaint_weights, inpaint_type, fp_bits = select_model.select_model()
    use_fp16 = (fp_bits == 16 and device.type == "cuda")
    print(f"Precision: FP{fp_bits} (FP16 aktiv: {use_fp16})")

    # --- Image size selection ---
    image_size = realtime_inpainting.choose_image_size()
    print("Using image size:", image_size)

    # --- Load models ---
    net_seg, net_inpaint = realtime_inpainting.load_models(
        device,
        u2net_weights,
        inpaint_weights,
        inpaint_type,
        use_fp16=use_fp16,
    )

    # --- Webcam ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kunne ikke åbne webcam!")
        return

    print("🎥 Realtime pipeline kører. Tryk 'q' for stop.")
    print("Timestamp, FPS, GPU_Load(%), GPU_Clock(MHz), Temp(C), RAM(%)")

    prev_time = time.time()

    # --- Start jtop stats ---
    with jtop() as jetson:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame-fejl")
                break

            # === Run your pipeline ===
            result_frame = realtime_inpainting.run_pipeline(
                frame,
                net_seg,
                net_inpaint,
                device,
                image_size,
                use_fp16=use_fp16,
            )

            # === FPS ===
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # === Jetson stats ===
            stats = jetson.stats

            gpu_load = stats.get("GPU", None)        # GPU usage % (Jetson Nano uses "GPU" key)
            gpu_clock = stats.get("GPU", None)       # GPU frequency - same as load on Nano
            temp_gpu = stats.get("Temp GPU", None)
            temp_cpu = stats.get("Temp CPU", None)

            cpu1 = stats.get("CPU1", None)
            cpu2 = stats.get("CPU2", None)
            cpu3 = stats.get("CPU3", None)
            cpu4 = stats.get("CPU4", None)

            ram_gb = stats["RAM"] / (1024 * 1024)    # convert KB → GB
            emc_gb = stats["EMC"] / (1024 * 1024)
            power = stats.get("power cur", None)

            # === PRINT METRICS ===
            print(
                f"time: {time.time():.2f}, "
                f"fps: {fps:.2f}, "
                f"gpu_load: {gpu_load} %, "
                f"gpu_clock: {gpu_clock} MHz, "
                f"temp_gpu: {temp_gpu} C, "
                f"temp_cpu: {temp_cpu} C, "
                f"cpu: [{cpu1}%, {cpu2}%, {cpu3}%, {cpu4}%], "
                f"ram: {ram_gb:.2f} GB, "
                f"emc: {emc_gb:.2f} GB, "
                f"power: {power} mW"
            )

            # === Show image ===
            display = cv2.resize(result_frame, (512, 512), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Realtime Inpainting", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
