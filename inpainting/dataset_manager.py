# Finde filer
from pathlib import Path

#Håndtere billede
from PIL import Image

# Import config python program for variable changes
import config

# imports torch and transforms for getting the data ready as tensors
import torch
import torchvision.transforms as T

import random
from typing import Sequence

# Vi skal have filhåndtering (stier).
# Skal finde alle billedfiler i en mappe
# Skal returnere en liste med filstierne
def get_files(path):
    "Makes list with all file paths"
    path = Path(path)
    return sorted([p for p in path.iterdir() if p.is_file()])



# Vi skal have indlæsning af et billede / en maske
# åben et billede med PIL og sæt farvemode (RGB)
# Output skal være farvebillede
def load_image(path):
    "Opens picture with PIL and converts it to RGB"
    image = Image.open(path)
    rgb_image = image.convert("RGB")

    return rgb_image


# Vi skal have load_mask. 
# Formålet er at åbne en maske som laver grayscale-billede
# Output et gråt billede, en kanal
def load_mask(path):
    "Loads mask and makes sure it is grayscaled"
    mask = Image.open(path)
    return mask.convert("L")





# Vi skal bruge en funktion som laver PIL -> til tensor, og sikrer størrelse og formater
# Resize til fx (image_size)
# Konverter til tensor
# Sørge for shape (3, H, W)
# Output bliver en torch.Tensor med shape (3, H, W)
def preprocess_image(image: Image.Image):
    """
    Takes a PIL RGB picture and makes it ready for the model:
    - It will resize to (image_size, image_size)
    - It converts to tensor values between [0,1]
    - Shapes (3, H, W)
    """
    transform = T.Compose([T.Resize(config.IMAGE_SIZE), T.ToTensor(),])

    tensor = transform(image)
    return tensor


# Vi skal bruge funktion som laver maskerne om til tensor
# Resize til samme størrelse som billeder
# grayscale
# konverter til tensor [0, 1]
# Binarisere -> så de er 0 og 1
# Shacpe skal være (1, H, W)
def preprocess_mask(mask_image: Image.Image):
    """
    Takes a PIL mask
    - Resizes it to (image_size, image_size)
    - Makes it into a 1 channel tensor
    - Makes it binary (0 or 1)
    - Shape (1, H, W)
    """

    transform = T.Compose([T.Resize(config.IMAGE_SIZE), T.ToTensor(),])

    mask = transform(mask_image)

    mask = (mask > 0.5).float()

    return mask


# Der skal bygges en funktion som kan lave de "ødelagte data"
# Formålet er at lave et input-billede med hul baseret på masken. 
# Konventionen skal være mask == 1 -> område der skal fjernes (hul) / mask == 0 -> behold orginal pixel
# Output skal være masked_image med shape (3, H, W)
def make_masked_image(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Makes a "destoryed" image with the orginial image and a binary mask

    We can assume with the other functions that:
    - An image has the shape (3, H, W), with values in [0, 1]
    - A mask has the shape (1, H, W), values with 0 or 1

    Where 1 is an area we want removed (inpainting area)
    And 0 is an area we want to preserve

    It returns a masked_image with the shape (3, H, W)
    - pixels where mask == 1 become 0 (black)
    - pixels where mask == 0 remain unchanged
    
    Returns a masked_image as a torch.Tensor with shape (3, H, W)
    """

    inverted_mask = 1.0 - mask # 1 where we want to keep pixels
    masked_image = image * inverted_mask

    return masked_image


# Lave den tensor modellen skal have som input
# Vi skal give både det ødelagte billede og masken som kanal.
# Noget i den her stil (4, H, W)
def build_model_input(masked_image: torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
    """
    Combines the masked image and the mask into a single input tensor for the model

    masked_image tenser = (3, H, W)
    
    mask tensor = (1, H, W)

    returns an input tensor as (4, H, W)
    """

    input_tensor = torch.cat([masked_image, mask], dim=0)

    return input_tensor




# Lave sample. 
# Vælge billeder idx
# Vælg en random maske
# indlæs + preprocess begge
# Lav masked_image
# Byg input og target
# pak det i et dictionary
# Output som
#{
#    "input":  input_tensor,   # (4, H, W)
#    "target": target_tensor,  # (3, H, W)
#    "mask":   mask_tensor,    # (1, H, W)
#    "image_path": str(...),
#    "mask_path":  str(...),
#}

def make_sample(image_paths: Sequence[str], mask_paths: Sequence[str], idx: int,) -> dict:
    """
    Builds one training sample for the inpainting model.

    Steps:
    - Pick an image from image_paths using idx
    - pick a random mask from mask_paths
    - load the preprocess both
    - create the masked (destroyed) image
    - build the model input (masked image + mask as ekstra channel)
    - Use the orginal image as target
    """
    
    # Get an image from the index and a random mask
    image_path = image_paths[idx]
    mask_path = random.choice(mask_paths)

    # load it in with PIL
    pil_image = load_image(image_path) # RGB
    pil_mask = load_mask(mask_path) # Grayscale

    # preprocess to make tensors
    image_tensor = preprocess_image(pil_image) # (3, H, W)
    mask_tensor = preprocess_mask(pil_mask) # (1, H, W)


    # create destoryed image (masked image)
    masked_image = make_masked_image(image_tensor, mask_tensor) # (3, H, W)

    # Build the input
    input_tensor = build_model_input(masked_image, mask_tensor) # (4, H, W)

    # Target tensor is simply the original image
    target_tensor = image_tensor

    # Return it all in a dictionary
    return {
        "input": input_tensor,
        "target": target_tensor,
        "mask": mask_tensor, 
        "image_path": str(image_path),
        "mask_path": str(mask_path),
    }




# eksempel på brug
#image_paths = get_files(config.IMAGE_FILE_PATH)
#mask_paths  = get_files(config.MASK_FILE_PATH)

#sample = make_sample(image_paths, mask_paths, idx=0)

#print(sample["input"].shape)   # (4, H, W)
#print(sample["target"].shape)  # (3, H, W)
#print(sample["mask"].shape)    # (1, H, W)
#print(sample["image_path"])
#print(sample["mask_path"])


test_img_path = r"C:\Users\Daniel K\Desktop\test.png"

# create a valid image
Image.new("RGB", (64, 64), "blue").save(test_img_path)

# try your load function

img = load_image(test_img_path)
print("Loaded:", img)
