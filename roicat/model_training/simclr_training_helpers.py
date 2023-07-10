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

from torch.nn import CrossEntropyLoss
from torch.optim import Adam


class Attachment_Blocks(torch.nn.Module):
    def __init__(self, attachment_block, slice_point, base_model, n_block_toInclude):
        super(Attachment_Blocks, self).__init__()
        self.attachment_block = attachment_block
        self.slice_point = slice_point
        self.base_model = base_model
        self.n_block_toInclude



class Simclr_Model():
    # Load pretrained weights, freeze all layers

    ### TODO: JZ: Download convnext from online source
    ### Freeze untrained layers

    # Chop model off at layer _, pool output, add linear layer unfrozen, flatten
    # Loop through parameters and freeze/unfreeze relevant layers
    # Model to device, prep_contrast, define forward

    def __init__(
            self,
            filepath_model, # Set filepath to save model
            base_model=None, # Freeze base_model
            slice_point=None, # Slice off the model at the slice_point and only keep the prior blocks
            attachment_block=None, # Add the attachment blocks to the end of the base_model
            block_to_unfreeze=None, # Unfreeze the model at and beyond the unfreeze_point
            n_block_toInclude=None, # Unfreeze the model at and beyond the unfreeze_point
            forward_version=None, # Set version of the forward pass to use
            load_model=False, # Whether the model should be loaded from the filepath_model (or otherwise saved to it)
            ):
        # If loading model, load it from onnx, otherwise create one from scratch using the other parameters
        if load_model:
            self.load_onnx(filepath_model)
        else:
            self.create_model(
                base_model=base_model,
                slice_point=slice_point,
                attachment_block=attachment_block,
                block_to_unfreeze=block_to_unfreeze,
                n_block_toInclude=n_block_toInclude,
                forward_version=forward_version,
                )
            
    def create_model(
            self,
            base_model=None, # Freeze base_model
            slice_point=None, # Slice off the model at the slice_point and only keep the prior blocks
            attachment_block=None, # Add the attachment blocks to the end of the base_model
            block_to_unfreeze=None, # Unfreeze the model at and beyond the unfreeze_point
            n_block_toInclude=None, # Unfreeze the model at and beyond the unfreeze_point
            forward_version=None, # Set version of the forward pass to use
            ):
        # Load base model
        if base_model is None:
            base_model = torchvision.models.resnet18(pretrained=True)
        self.base_model = base_model

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Slice off base model
        if slice_point is None:
            slice_point = 0
        self.slice_point = slice_point
        self.base_model = torch.nn.Sequential(list(self.base_model.children())[0][:self.slice_point])

        # Add attachment blocks
        if attachment_block is None:
            attachment_block = torch.nn.Sequential()
        self.attachment_block = attachment_block
        self.base_model = torch.nn.Sequential(self.base_model, self.attachment_block)

        # Unfreeze model
        if block_to_unfreeze is None:
            block_to_unfreeze = 0
        self.block_to_unfreeze = block_to_unfreeze
        self.n_block_toInclude = n_block_toInclude
        self.base_model = torch.nn.Sequential(list(self.base_model.children())[0][:self.block_to_unfreeze])

        # Set forward version
        if forward_version is None:
            forward_version = 0
        self.forward_version = forward_version

        # Set model to device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.base_model = self.base_model.to(self.device)

        # Prep contrast
        self.contrast = roicat.model_training.contrastive_learning.ContrastiveLearning(
            base_model=self.base_model,
            device=self.device,
            )
        
    def forward(self, x):
        if self.forward_version == 0:
            return self.forward_v0(x)
        else:
            raise ValueError('forward_version not recognized')
        
    def forward_v0(self, x):
        return self.contrast.forward(x)
    

    def save_onnx(self, filepath_model):
        pass
    def load_onnx(self, filepath_model):
        pass
    
    ### Model preferences

    # base_model_frozen = base_model
    # for param in base_model_frozen.parameters():
    #     param.requires_grad = False

    # model_chopped = torch.nn.Sequential(list(base_model_frozen.children())[0][:params['n_block_toInclude']])  ## 0.
    # model_chopped_pooled = torch.nn.Sequential(model_chopped, torch.nn.__dict__[params['head_pool_method']](**params['head_pool_method_kwargs']), torch.nn.Flatten())  ## 1.

    # data_dim = tuple([1] + list(image_out_size))
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


class Simclr_Trainer():
    def __init__(
            self,
            data,
            model,
            
            learning_rate,

            ):
        pass


    # def train(self, train_loader, val_loader, params):
    #     # Set model to training mode
    #     self.base_model.train()

    #     # Set optimizer
    #     optimizer = Adam(self.base_model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    #     # Set loss function
    #     criterion = CrossEntropyLoss()

    #     # Set scheduler
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'], eta_min=params['lr_min'])

    #     # Set best model
    #     best_model_wts = copy.deepcopy(self.base_model.state_dict())
    #     best_acc = 0.0

    #     # Set training history
    #     history = {
    #         'train_loss': [],
    #         'train_acc': [],
    #         'val_loss': [],
    #         'val_acc': [],
    #         }
        
    #     # Loop through epochs
    #     for epoch in range(params['epochs']):

    # Save model, optimizer, scheduler, etc. to dir_save
    # Save training loss to dir_save


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
