import random

import torch

import config
import utils
import dataloader_mobilenetv2 as dataloader
import model_mobilenet_inpainting as m
import dataset_manager as dm

from PIL import Image
import numpy as np

weight_path = config.WEIGHTS

image_path = config.TEST_IMAGE

# Might need these later
##############
#training_val = torch.load(weight_path, map_location=config.DEVICE)
#weights = training_val["model_state_dict"]
##############

##############
### Model ####
##############

print("Using device:", config.DEVICE)

# Model
model = m.MobileNetInpainting(in_channels=4).to(config.DEVICE)
state_dict = torch.load(weight_path, map_location=config.DEVICE)
model.load_state_dict(state_dict)
model.eval()

##############
# Inference ##
##############
print("\nSTEP 2: Run inference on image:" )

with torch.no_grad():
    image = dm.load_image(image_path)
    #image = image.unsqueeze(0)   # add batch dimension -> [1, 3, H, W]
    image = dm.preprocess_image(image)      
    
    ############################# Delete this part when running real model
    # Create mask: everything known = 1
    mask = torch.ones(1, image.shape[1], image.shape[2])  # [1,H,W]

    # Make a big black square in the center
    h, w = image.shape[1], image.shape[2]
    square_size = min(h, w) // 2  # half of image size
    start_h = h // 4
    start_w = w // 4
    mask[:, start_h:start_h + square_size, start_w:start_w + square_size] = 0  # masked region


    input_tensor = torch.cat([image, mask], dim=0)
    input_tensor = input_tensor.unsqueeze(0)  # add batch dimension -> [1, 4, H, W]
    # Remove batch dimension and convert [3,H,W] -> [H,W,C]
    masked_input_tensor = image * mask              # [3,H,W], multiply mask to black out region
    masked_input_img = masked_input_tensor.permute(1, 2, 0).cpu().numpy()  # [H,W,C]

    # Convert to uint8 and save
    masked_pil = Image.fromarray((masked_input_img * 255).astype(np.uint8))
    masked_pil.save("masked_input.png")

    #############################

    outputs = model(input_tensor.to(config.DEVICE))

    ############ DELETE ###################
        # Remove batch dimension and convert [C,H,W] -> [H,W,C]
    output_img = outputs.squeeze(0).permute(1,2,0).cpu().numpy()  # [H,W,C]

    # Clamp values to [0,1] just in case
    output_img = np.clip(output_img, 0, 1)

    # Convert to uint8
    output_pil = Image.fromarray((output_img * 255).astype(np.uint8))
    output_pil.save("inpainted_output.png")
    ####################################
print(" Inference sucessful." )


    

