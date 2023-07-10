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
    default='/Users/josh/analysis/data/ROICaT/simclr_training/sf_sparse_36x36_20220503.npz',
    help='Path to raw ROI data to be used to train the model.',
)
parser.add_argument(
    '--path_params',
    '-p',
    required=True,
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
args = parser.parse_args()
directory_data = args.directory_data
filepath_params = args.path_params
directory_save = args.directory_save


### Global preferences
# device_train = torch_helpers.set_device(use_GPU=params['useGPU_dataloader'])

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
### Data import preferences
# params['paths']['path_data_training']


# Create dataset / dataloader
ROI_images_rs = roicat.ROInet.Resizer_ROI_images(
    np.concatenate(data.ROI_images, axis=0),
    dict_params['data']['um_per_pixel'],
    dict_params['data']['nan_to_num'],
    dict_params['data']['nan_to_num_val'],
    dict_params['data']['verbose']
).ROI_images_rs
### Resizing preferences

dataloader = roicat.ROInet.Dataloader_ROInet(
    ROI_images_rs,
    dict_params['dataloader']['batchSize_dataloader'],
    dict_params['dataloader']['pinMemory_dataloader'],
    dict_params['dataloader']['numWorkers_dataloader'],
    dict_params['dataloader']['persistentWorkers_dataloader'],
    dict_params['dataloader']['prefetchFactor_dataloader'],
    torch.nn.Sequential(
        *[roicat.model_training.augmentation.__dict__[key](**params) for key,params in dict_params['transforms_invariant'].items()]
    ), # Converting dictionary of transforms to torch.nn.Sequential object
    tuple(dict_params['dataloader']['img_size_out']),
    dict_params['dataloader']['jit_script_transforms'],
    dict_params['dataloader']['verbose'],
).dataloader
### DataLoader preferences
# params['useGPU_training']
# transforms = torch.nn.Sequential(
#     *[augmentation.__dict__[key](**params) for key,params in params['augmentation'].items()]
# )
# scripted_transforms = torch.jit.script(transforms)
# dataloader_train = torch.utils.data.DataLoader(
#     dataset_train,
#     **params['dataloader_kwargs']
# )
# image_out_size = list(dataset_train[0][0][0].shape)
# data_dim = tuple([1] + list(image_out_size))



# Create Model
model = sth.Simclr_Model(
    dict_params['model']['hyperparameters'],
    dict_params['model']['filepath_model'],
)
### Model preferences

# base_model_frozen = torchvision.models.__dict__[params['torchvision_model']](pretrained=True)
# for param in base_model_frozen.parameters():
#     param.requires_grad = False

# model_chopped = torch.nn.Sequential(list(base_model_frozen.children())[0][:params['n_block_toInclude']])  ## 0.
# model_chopped_pooled = torch.nn.Sequential(model_chopped, torch.nn.__dict__[params['head_pool_method']](**params['head_pool_method_kwargs']), torch.nn.Flatten())  ## 1.

# model = ModelTackOn(
#     model_chopped_pooled.to('cpu'),
#     base_model_frozen.to('cpu'),
#     data_dim=data_dim,
#     pre_head_fc_sizes=params['pre_head_fc_sizes'], 
#     post_head_fc_sizes=params['post_head_fc_sizes'], 
#     classifier_fc_sizes=None,
#     nonlinearity=params['head_nonlinearity'],
#     kwargs_nonlinearity=params['head_nonlinearity_kwargs'],
# )

# mnp = [name for name, param in model.named_parameters()]  ## 'model named parameters'
# mnp_blockNums = [name[name.find('.'):name.find('.')+8] for name in mnp]  ## pulls out the numbers just after the model name
# mnp_nums = [path_helpers.get_nums_from_string(name) for name in mnp_blockNums]  ## converts them to numbers
# block_to_freeze_nums = path_helpers.get_nums_from_string(params['block_to_unfreeze'])  ## converts the input parameter specifying the block to freeze into a number for comparison

# m_baseName = mnp[0][:mnp[0].find('.')]

# for ii, (name, param) in enumerate(model.named_parameters()):
#     if m_baseName in name:
# #         print(name)
#         if mnp_nums[ii] < block_to_freeze_nums:
#             param.requires_grad = False
#         elif mnp_nums[ii] >= block_to_freeze_nums:
#             param.requires_grad = True

# names_layers_requiresGrad = [( param.requires_grad , name ) for name,param in list(model.named_parameters())]

# model.forward = model.forward_latent


# Specify criterion, optimizer, scheduler, learning rate, etc.
trainer = sth.Simclr_Trainer(
    data,
    model,
    dict_params['trainer']['hyperparameters'],
    directory_save
)
### Trainer preferences

# from torch.nn import CrossEntropyLoss
# from torch.optim import Adam

# model.train();
# model.to(device_train)
# model.prep_contrast()

# criterion = [CrossEntropyLoss()]
# criterion = [_.to(device_train) for _ in criterion]
# optimizer = Adam(
#     model.parameters(), 
#     lr=params['lr'],
# )
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
#                                                    gamma=params['gamma'],
#                                                   )

# losses_train, losses_val = [], [np.nan]
# for epoch in tqdm(range(params['n_epochs'])):

#     print(f'epoch: {epoch}')
    
#     losses_train = training.epoch_step(
#         dataloader_train, 
#         model, 
#         optimizer, 
#         criterion,
#         scheduler=scheduler,
#         temperature=params['temperature'],
#         # l2_alpha,
#         penalty_orthogonality=params['penalty_orthogonality'],
#         mode='semi-supervised',
#         loss_rolling_train=losses_train, 
#         loss_rolling_val=losses_val,
#         device=device_train, 
#         inner_batch_size=params['inner_batch_size'],
#         verbose=2,
#         verbose_update_period=1,
#         log_function=partial(write_to_log, path_log=path_saveLog),
# )
    
#     ## save loss stuff
#     if params['prefs']['saveLogs']:
#         np.save(path_saveLoss, losses_train)
    
#     ## if loss becomes NaNs, don't save the network and stop training
#     if torch.isnan(torch.as_tensor(losses_train[-1])):
#         break
        
#     ## save model
#     if params['prefs']['saveModelIteratively']:
#         torch.save(model.state_dict(), path_saveModel)



# Loop through epochs, batches, etc. if loss becomes NaNs, don't save the network and stop training. Otherwise, save the network as an onnx file.
##### TODO
trainer.train()

