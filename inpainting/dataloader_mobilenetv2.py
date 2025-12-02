from torch.utils.data import Dataset
from pathlib import Path
import config
from dataset_manager import get_files, make_sample

class InpaintingDataset(Dataset):
        
        def __init__(self) -> None:
            # Get all files out from the config
            self.image_paths = get_files(config.IMAGE_FILE_PATH)
            self.mask_paths = get_files(config.MASK_FILE_PATH)
        


        def __len__(self) -> int:
            # Define the ammount of pictues
            return len(self.image_paths)
        

        def __getitem__(self, idx: int) -> dict:
            # one picture for every idx
            # One random mask
            # preproces + masked image + build_model_input

            sample = make_sample(self.image_paths, self.mask_paths, idx)
            return sample