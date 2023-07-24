# Imports
import argparse
import sys
import os
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import json
import pandas as pd
import argparse
import scipy.sparse
from pathlib import Path

import roicat.data_importing
import roicat.ROInet
import roicat.model_training.augmentation
import roicat.model_training.simclr_training_helpers as sth

path_script = sys.argv[0]

# Parse arguments for data directory, parameters, and save directory
parser = argparse.ArgumentParser(
    prog='ROICaT SimCLR Training',
    description='This script runs the basic fit pipeline for a self-supervised ROI model using a json file containing the parameters.',
)
parser.add_argument(
    '--directory_data',
    '-d',
    required=False,
    metavar='',
    type=str,
    default='/Users/josh/analysis/data/ROICaT/simclr_training',
    help='Path to raw ROI data to be used to train the model.',
)
parser.add_argument(
    '--path_params',
    '-p',
    required=False,
    metavar='',
    type=str,
    default='/Users/josh/analysis/github_repos/ROICaT/roicat/model_training/simclr_params_base.json',
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
parser.add_argument(
    '--test_option',
    '-t',
    required=False,
    metavar='',
    type=bool,
    default=True,
    help="Whether to run the script in test mode (True) or not (False). Reduces number of training examples to 3000.",
)
args = parser.parse_args()
directory_data = args.directory_data
filepath_params = args.path_params
directory_save = args.directory_save
test_option = args.test_option


# Load paths from directory_data and parameters from JSON
list_filepaths_data = [Path(os.path.join(directory_data, filename)) for filename in os.listdir(directory_data)]
with open(filepath_params) as f:
    dict_params = json.load(f)

# Load data from dir_data into Data object
ROI_sparse_all = [scipy.sparse.load_npz(filepath_ROI_images) for filepath_ROI_images in list_filepaths_data if filepath_ROI_images.suffix == '.npz']
ROI_images = [sf_sparse.toarray().reshape(sf_sparse.shape[0], 36,36).astype(np.float32) for sf_sparse in ROI_sparse_all]
data = roicat.data_importing.Data_roicat();

if test_option:
    ROI_images = [ROI_images[0][:3000]]
    n_epochs = 2
else:
    n_epochs = dict_params['trainer']['n_epochs']

data.set_ROI_images(
    ROI_images=ROI_images,
    um_per_pixel=dict_params['data']['um_per_pixel'],
)

# Create dataset / dataloader for SimCLR training
ROI_images_rs = roicat.ROInet.Resizer_ROI_images(
    ROI_images=np.concatenate(data.ROI_images, axis=0),
    um_per_pixel=dict_params['data']['um_per_pixel'],
    nan_to_num=dict_params['data']['nan_to_num'],
    nan_to_num_val=dict_params['data']['nan_to_num_val'],
    verbose=dict_params['data']['verbose']
).ROI_images_rs
dataloader_generator = roicat.ROInet.Dataloader_ROInet(
    ROI_images_rs,
    batchSize_dataloader=dict_params['dataloader']['batchSize_dataloader'],
    pinMemory_dataloader=dict_params['dataloader']['pinMemory_dataloader'],
    numWorkers_dataloader=dict_params['dataloader']['numWorkers_dataloader'],
    persistentWorkers_dataloader=dict_params['dataloader']['persistentWorkers_dataloader'],
    prefetchFactor_dataloader=dict_params['dataloader']['prefetchFactor_dataloader'],
    transforms=torch.nn.Sequential(
        *[roicat.model_training.augmentation.__dict__[key](**params) for key,params in dict_params['dataloader']['transforms_invariant'].items()]
    ), # Converting dictionary of transforms to torch.nn.Sequential object
    n_transforms=2,
    img_size_out=tuple(dict_params['dataloader']['img_size_out']),
    jit_script_transforms=dict_params['dataloader']['jit_script_transforms'],
    shuffle_dataloader=dict_params['dataloader']['shuffle_dataloader'],
    drop_last_dataloader=dict_params['dataloader']['drop_last_dataloader'],
    verbose=dict_params['dataloader']['verbose'],
)
dataloader = dataloader_generator.dataloader

# Create Model
model_container = sth.Simclr_Model(
    filepath_model=dict_params['model']['filepath_model'], # Set filepath to/from which to save/load model
    base_model=torchvision.models.__dict__[dict_params['model']['torchvision_model']](pretrained=True),
    head_pool_method=dict_params['model']['head_pool_method'],
    head_pool_method_kwargs=dict_params['model']['head_pool_method_kwargs'],
    pre_head_fc_sizes=dict_params['model']['pre_head_fc_sizes'],
    post_head_fc_sizes=dict_params['model']['post_head_fc_sizes'],
    head_nonlinearity=dict_params['model']['head_nonlinearity'],
    head_nonlinearity_kwargs=dict_params['model']['head_nonlinearity_kwargs'],
    block_to_unfreeze=dict_params['model']['block_to_unfreeze'],
    n_block_toInclude=dict_params['model']['n_block_toInclude'],
    image_out_size=list(dataloader_generator.dataset[0][0][0].shape),
    load_model=False,
)

# Specify criterion, optimizer, scheduler, learning rate, etc.
trainer = sth.Simclr_Trainer(
    dataloader=dataloader,
    model_container=model_container,
    n_epochs=n_epochs,
    device_train=dict_params['trainer']['device_train'],
    inner_batch_size=dict_params['trainer']['inner_batch_size'],
    learning_rate=dict_params['trainer']['learning_rate'],
    penalty_orthogonality=dict_params['trainer']['penalty_orthogonality'],
    weight_decay=dict_params['trainer']['weight_decay'],
    gamma=dict_params['trainer']['gamma'],
    temperature=dict_params['trainer']['temperature'],
    l2_alpha=dict_params['trainer']['l2_alpha'],
)

# Loop through epochs, batches, etc. if loss becomes NaNs, don't save the network and stop training.
# Otherwise, save the network as an onnx file.
# trainer.train()

if test_option:
    pass
    # trainer.test(torch.ones((1, 3, 224, 224)))

    # trainer.train_pca(torch.rand((260, 3, 224, 224)))
    # trainer.test_pca(torch.ones((1, 3, 224, 224)))

    trainer.train_pca(torch.cat([torch.cat(x[0], dim=0) for x in trainer.dataloader], axis=0), check_pca_layer_valid=True)
