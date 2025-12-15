from jtop import jtop
import time
from pathlib import Path
import csv
import sys

# --- Check filename argument ---
if len(sys.argv) < 2:
    print("Usage: python hardware_logger.py <output_csv_name>")
    sys.exit(1)

output_file = Path(sys.argv[1])

START_FILE = Path("START_LOGGING")
STOP_FILE = Path("STOP_LOGGING")

# --- Clean up any leftover signal files ---
for f in [START_FILE, STOP_FILE]:
    if f.exists():
        f.unlink()

print("Waiting for START_LOGGING signal...")
while not START_FILE.exists():
    time.sleep(0.05)

print(f"Started logging hardware metrics to {output_file}")
with jtop() as jetson, open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "ts", "gpu", "cpu1", "cpu2", "cpu3", "cpu4",
        "ram", "temp_gpu", "temp_cpu", "power"
    ])

    try:
        while not STOP_FILE.exists():
            s = jetson.stats
            writer.writerow([
                time.time(),
                s.get("GPU"),
                s.get("CPU1"),
                s.get("CPU2"),
                s.get("CPU3"),
                s.get("CPU4"),
                s["RAM"] / (1024 * 1024),  # KB -> GB
                s.get("Temp GPU"),
                s.get("Temp CPU"),
                s.get("power cur", 0) / 1000  # mW -> W
            ])
            f.flush()
            time.sleep(0.2)
    finally:
        # --- Clean up signal files ---
        if START_FILE.exists():
            START_FILE.unlink()
        if STOP_FILE.exists():
            STOP_FILE.unlink()

print("Stopped logging.")
