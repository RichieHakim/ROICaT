# Imports
import argparse
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
import scipy.sparse
import simclr_training_helpers as sth

path_script = sys.argv[0]

## Argparse --directory_data, --path_params, --directory_save
parser = argparse.ArgumentParser(
    prog='ROICaT SimCLR Training',
    description='This script runs the basic fit pipeline for a self-supervised ROI model using a json file containing the parameters.',
)
parser.add_argument(
    '--directory_data',
    '-d',
    required=True,
    metavar='',
    type=str,
    default='/Users/josh/analysis/data/classification/stat_s2p_backup',
    help='Path to raw ROI data to be used to train the model.',
)
parser.add_argument(
    '--path_params',
    '-p',
    required=True,
    metavar='',
    type=str,
    default='/Users/josh/analysis/outputs/ROICaT/simclr_training/simclr_params.json',
    help='Path to json file containing parameters.',
)
parser.add_argument(
    '--directory_save',
    '-s',
    required=False,
    metavar='',
    type=str,
    default='/Users/josh/analysis/outputs/ROICaT/simclr_training',
    help="Directory into which final model and evaluations should be saved.",
)
args = parser.parse_args()
directory_data = args.directory_data
filepath_params = args.path_params
directory_save = args.directory_save

# Load parameters from JSON
with open(filepath_params) as f:
    dict_params = json.load(f)

list_filepaths_data = [os.path.join(directory_data, filename) for filename in os.listdir(directory_data)]

# Load data from dir_data into Data object... or load from saved Data object
ROI_sparse_all = [scipy.sparse.load_npz(filepath_ROI_images) for filepath_ROI_images in list_filepaths_data]
ROI_images = [torch.as_tensor(sf_sparse.toarray().reshape(sf_sparse.shape[0], 36,36), dtype=torch.float32) for sf_sparse in ROI_sparse_all]

data = roicat.data_importing.Data_roicat();
data.set_ROI_images(
    ROI_images=ROI_images,
    um_per_pixel=dict_params['data']['um_per_pixel'],
)

# Create dataset / dataloader
ROI_images_rs = roicat.ROInet.Resizer_ROI_images(
    np.concatenate(data.ROI_images, axis=0),
    dict_params['data']['um_per_pixel'],
    dict_params['data']['nan_to_num'],
    dict_params['data']['nan_to_num_val'],
    dict_params['data']['verbose']
).ROI_images_rs

dataloader = roicat.ROInet.Dataloader_ROInet(
    ROI_images_rs,
    dict_params['dataloader']['batchSize_dataloader'],
    dict_params['dataloader']['pinMemory_dataloader'],
    dict_params['dataloader']['numWorkers_dataloader'],
    dict_params['dataloader']['persistentWorkers_dataloader'],
    dict_params['dataloader']['prefetchFactor_dataloader'],
    torch.nn.Sequential(*dict_params['dataloader']['list_transforms']), # TODO: Replace with actual transforms / unpacking the *args list comprehension version
    dict_params['dataloader']['img_size_out'],
    dict_params['dataloader']['jit_script_transforms'],
    dict_params['dataloader']['verbose'],
).dataloader

# Create Model
model = sth.Simclr_Model(
    dict_params['model']['hyperparameters'],
    dict_params['model']['filepath_model'],
)

# Specify criterion, optimizer, scheduler, learning rate, etc.
trainer = sth.Simclr_Trainer(
    data,
    model,
    dict_params['trainer']['hyperparameters'],
    directory_save
)

# Loop through epochs, batches, etc. if loss becomes NaNs, don't save the network and stop training. Otherwise, save the network as an onnx file.
##### TODO
trainer.train()
