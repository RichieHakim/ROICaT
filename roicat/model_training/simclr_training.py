# Imports
import sys
import os
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import time
import copy
import json
import random
import pandas as pd
import math
import argparse
import pickle
import roicat

# Pull from sys.argv tuple dir_data (numpy file), filepath_params (JSON), and dir_save from sys.argv
path_script = sys.argv[0]
dir_data = sys.argv[1]
filepath_params = sys.argv[2]
dir_save = sys.argv[3]

# Load parameters from JSON
with open(filepath_params) as f:
    dict_params = json.load(f)

# Load data from dir_data into Data object... or load from saved Data object
ROI_images = [roicat.helpers.load_ROI_images(filepath_ROI_images) for filepath_ROI_images in dict_params['list_filepaths_ROI_images']]
data = roicat.data_importing.Data_roicat();
data.set_ROI_images(
    ROI_images=ROI_images,
    um_per_pixel=2.5,
);
### Alternatively: Load in premade / serializeable dict version of Data object

assert np.all(~np.isnan(data.ROI_images[0])), "JZ Error: NaNs Exist in ROI images"
assert np.all(np.sum(np.abs(data.ROI_images[0]), axis=[-1, -2]) != 0), "JZ Error: All zero images exist in ROI images"

# Create torch sequential of Data Augmentations

### TODO: Replace with actually used augmentations
### TODO: Replace with the automatic getattribute dunder expansion of the params dict
list_transforms = [
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
    torchvision.transforms.RandomRotation(degrees=180),
    torchvision.transforms.RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10, resample=False, fillcolor=0),
    torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    torchvision.transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
];

# Create dataset / dataloader

ROI_images = np.concatenate(data.ROI_images, axis=0)
roi_resizer = roicat.ROInet.Resizer_ROI_images(ROI_images,
                                               dict_params['um_per_pixel'],
                                               dict_params['nan_to_num'],
                                               dict_params['nan_to_num_val'],
                                               verbose=dict_params['verbose'])
ROI_images_rs = roi_resizer.ROI_images_rs
dataloader_generator = roicat.ROInet.Dataloader_ROInet(
    ROI_images_rs,
    dict_params['batchSize_dataloader'],
    dict_params['pinMemory_dataloader'],
    dict_params['numWorkers_dataloader'],
    dict_params['persistentWorkers_dataloader'],
    dict_params['prefetchFactor_dataloader'],
    torch.nn.Sequential(*list_transforms), # TODO: Replace with actual transforms / unpacking the *args list comprehension version
    dict_params['img_size_out'],
    dict_params['jit_script_transforms'],
    dict_params['verbose'],
)

##### ***** Bundle Together *****
[[[[[
    # Load pretrained weights, freeze all layers

    ### TODO: JZ: Download convnext from online source
    ### Freeze untrained layers
    ### Freeze untrained layers

    # Chop model off at layer _, pool output, add linear layer unfrozen, flatten

    # Loop through parameters and freeze/unfreeze relevant layers

    # Model to device, prep_contrast, define forward
]]]]]

# Save relevant pre-training parameters to JSON
##### TODO


# Specify criterion, optimizer, scheduler, learning rate, etc.
##### TODO


# Loop through epochs, batches, etc. if loss becomes NaNs, don't save the network and stop training. Otherwise, save the network as an onnx file.
##### TODO


# Save model, optimizer, scheduler, etc. to dir_save
##### TODO


# Save training loss to dir_save
##### TODO
