import random

import torch

import config

import dataloader_mobilenetv2 as dataloader
import model_mobilenet_inpainting as m
import dataset_manager as dm
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

# Dataset
dataset = dataloader.InpaintingDataset()
print(f"Dataset size: {len(dataset)}")

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
    outputs = model(dm.preprocess_image(image_path))
print(" Inference sucessful." )

