from pathlib import Path


IMAGE_FILE_PATH = r"D:\Program Files (x86)\Programmer programmering\3 Semester python work\3_semester\P3_projekt\ILSVRC\Data\CLS-LOC\data_for_AI_lab\images"
MASK_FILE_PATH = r"D:\Program Files (x86)\Programmer programmering\3 Semester python work\3_semester\P3_projekt\ILSVRC\Data\CLS-LOC\data_for_AI_lab\masks"
WEIGHTS = Path(r"C:\Users\Daniel K\Desktop\DAKI\3. Semester\P3\Nvidia_Jetson_p3\weights\mobilenet_inpaint_64.pth")

TEST_IMAGE = Path(r"C:\Users\Daniel K\Desktop\DAKI\3. Semester\P3\Nvidia_Jetson_p3\images\my_name_is_jeff.png")


IMAGE_SIZE = (256, 256) # (128, 128), (256, 256)
BATCH_SIZE = 4
NUM_WORKERS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")