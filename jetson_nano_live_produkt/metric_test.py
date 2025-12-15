import cv2
import torch
import time
from jtop import jtop
import numpy as np
from pathlib import Path

import select_model
import realtime_inpainting

import subprocess
import re

def read_tegrastats():
    """
    Reads one line from tegrastats and extracts GPU usage.
    Returns GPU percentage (0-100).

    Written by ChatGPT!
    """
    try:
        # Run tegrastats for one line
        proc = subprocess.Popen(
            ["tegrastats", "--interval", "100", "--logfile", "/dev/stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        line = proc.stdout.readline()
        proc.kill()
    except Exception as e:
        print("Error reading tegrastats:", e)
        return 0

    # Extract GPU usage
    m = re.search(r"GR3D_FREQ\s+(\d+)%@", line, re.IGNORECASE)
    gpu_load = int(m.group(1)) if m else 0
    return gpu_load

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
    test_image_path = Path("/home/aaunano/Desktop/projekt_p3/jetson_nano_live_produkt/test_images/linkedin_billede.jfif")
    test_image = cv2.imread(str(test_image_path))
    
    if test_image is None:
        raise FileNotFoundError(f"Could not load image: {test_image_path}")
    
    print("Loaded test image.")

    # ---------------------------------------------
    # --- Test loop (WARMUP and BENCHMARK test) ---
    # ---------------------------------------------
    WARMUP_ITERS = 10
    BENCH_ITERS = 100

    with jtop() as jetson:
        # --- WARMUP Loop---    
        print(f"Running {WARMUP_ITERS} warmup iterations...")
        for i in range(WARMUP_ITERS):

            result_frame = realtime_inpainting.run_pipeline(
                test_image,
                net_seg,
                net_inpaint,
                device,
                image_size,
                use_fp16=use_fp16,
            )
        print("Warmup complete.")
        # --- BENCHMARK ---

        # Metrics storage
        latencies = []
        gpu_loads = []
        temps_gpu = []
        temps_cpu = []
        powers = []
        ram_usages = []
        cpu_usages = {1: [], 2: [], 3: [], 4: []}
        peak_ram = 0


        # --- BENCHMARK Loop---
        for i in range(BENCH_ITERS):

            time_0 = time.time()
            realtime_inpainting.run_pipeline(
                test_image, net_seg, net_inpaint, device, image_size, use_fp16=use_fp16
            )
            torch.cuda.synchronize()


            time_1 = time.time()
            latency = time_1 - time_0
            


            stats = jetson.stats

            # GPU and CPU usage
            gpu_load = stats.get("GPU", None)


            
            cpu1 = stats.get("CPU1", None)
            cpu2 = stats.get("CPU2", None)
            cpu3 = stats.get("CPU3", None)
            cpu4 = stats.get("CPU4", None)

            # RAM usage
            ram_gb = stats["RAM"] / (1024 * 1024)    # convert KB → GB
            peak_ram = max(peak_ram, ram_gb)

            # Temp and power 
            temp_gpu = stats.get("Temp GPU", None)
            temp_cpu = stats.get("Temp CPU", None)
            power = stats.get("power cur", None)

            # Append metrics
            latencies.append(latency)
            gpu_loads.append(gpu_load) 

            cpu_usages[1].append(cpu1)
            cpu_usages[2].append(cpu2)
            cpu_usages[3].append(cpu3)
            cpu_usages[4].append(cpu4)
            temps_gpu.append(temp_gpu)
            temps_cpu.append(temp_cpu)
            powers.append(power)
            ram_usages.append(ram_gb)

            print(f"[{i+1}/{BENCH_ITERS}] latency={latency*1000:.2f} ms | GPU={gpu_load}% | RAM={ram_gb:.2f} GB")
        torch.cuda.synchronize()
        # --- RESULTS ---
        print("\n--- BENCHMARK RESULTS ---")
        avg_latency = np.mean(latencies)
        avg_gpu_load = np.mean(gpu_loads)

        avg_temp_gpu = np.mean(temps_gpu)
        avg_temp_cpu = np.mean(temps_cpu)
        avg_power = np.mean(powers)
        avg_ram = np.mean(ram_usages)
        avg_cpu1 = np.mean(cpu_usages[1])
        avg_cpu2 = np.mean(cpu_usages[2])
        avg_cpu3 = np.mean(cpu_usages[3])
        avg_cpu4 = np.mean(cpu_usages[4])
        print("\n========= BENCHMARK RESULTS (AFTER WARMUP) =========")
        print(f"Average latency:                {avg_latency*1000:.2f} ms")
        print(f"Average FPS:                    {1/avg_latency:.2f}")
        print(f"Average GPU load:               {avg_gpu_load:.2f} %")
        print(f"Average CPU load:               [{avg_cpu1:.1f}%, {avg_cpu2:.1f}%, {avg_cpu3:.1f}%, {avg_cpu4:.1f}%]")
        print(f"Average GPU temp:               {avg_temp_gpu:.1f} °C")
        print(f"Average CPU temp:               {avg_temp_cpu:.1f} °C")
        print(f"Average power draw:             {avg_power/1000:.3f} W")
        print(f"Average RAM usage:              {avg_ram:.2f} GB")
        print(f"Peak RAM usage:                 {peak_ram:.2f} GB")
        print("============================================================\n")
if __name__ == "__main__":
    test()
