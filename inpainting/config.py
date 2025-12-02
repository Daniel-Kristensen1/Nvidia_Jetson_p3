from pathlib import Path
import torch

IMAGE_FILE_PATH = []
MASK_FILE_PATH = []
WEIGHTS = Path(r"C:\Users\Daniel K\Desktop\DAKI\3. Semester\P3\Nvidia_Jetson_p3\weights\mobilenet_inpaint_64.pth")

TEST_IMAGE = Path(r"C:\Users\Daniel K\Desktop\DAKI\3. Semester\P3\Nvidia_Jetson_p3\images\bezos.webp")


IMAGE_SIZE = (256, 256) # (128, 128), (256, 256)
BATCH_SIZE = 4
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

