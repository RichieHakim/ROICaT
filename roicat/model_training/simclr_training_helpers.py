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
import tqdm
from functools import partial
from roicat.model_training import training
from typing import Optional, List, Tuple, Union, Dict, Any

from torch.nn import CrossEntropyLoss
from torch.optim import Adam




def log_fn(log_str, log_file):
    """
    Write a string to a log file
    
    Args:
        log_str (str):
            String to be written to the log file
        log_file (str):
            Path to the log file
    """
    with open(log_file, 'a') as f:
        f.write(log_str + '\n')

# class Attachment_Blocks(torch.nn.Module):
#     def __init__(self, attachment_block, slice_point, base_model, n_block_toInclude):
#         super(Attachment_Blocks, self).__init__()
#         self.attachment_block = attachment_block
#         self.slice_point = slice_point
#         self.base_model = base_model
#         self.n_block_toInclude

def get_nums_from_string(string_with_nums):
    """
    Return the numbers from a string as an int
    RH 2021-2022

    Args:
        string_with_nums (str):
            String with numbers in it
    
    Returns:
        nums (int):
            The numbers from the string    
            If there are no numbers, return None.        
    """
    idx_nums = [ii in str(np.arange(10)) for ii in string_with_nums]
    
    nums = []
    for jj, val in enumerate(idx_nums):
        if val:
            nums.append(string_with_nums[jj])
    if not nums:
        return None
    nums = int(''.join(nums))
    return nums

class ModelTackOn(torch.nn.Module):
    def __init__(
        self, 
        base_model, 
        un_modified_model,
        data_dim=(1,3,36,36), 
        pre_head_fc_sizes=[100], 
        post_head_fc_sizes=[100], 
        classifier_fc_sizes=None, 
        nonlinearity='relu', 
        kwargs_nonlinearity={},
    ):
            super(ModelTackOn, self).__init__()
            self.base_model = base_model
            final_base_layer = list(un_modified_model.children())[-1]
            
            self.data_dim = data_dim

            self.pre_head_fc_lst = []
            self.post_head_fc_lst = []
            self.classifier_fc_lst = []
                
            self.nonlinearity = nonlinearity
            self.kwargs_nonlinearity = kwargs_nonlinearity

            self.init_prehead(final_base_layer, pre_head_fc_sizes)
            self.init_posthead(pre_head_fc_sizes[-1], post_head_fc_sizes)
            if classifier_fc_sizes is not None:
                self.init_classifier(pre_head_fc_sizes[-1], classifier_fc_sizes)
            
    def init_prehead(self, prv_layer, pre_head_fc_sizes):
        for i, pre_head_fc in enumerate(pre_head_fc_sizes):
            if i == 0:
                in_features = self.base_model(torch.rand(*(self.data_dim))).data.squeeze().shape[0]  ## RH EDIT
            else:
                in_features = pre_head_fc_sizes[i - 1]
            fc_layer = torch.nn.Linear(in_features=in_features, out_features=pre_head_fc)
            self.add_module(f'PreHead_{i}', fc_layer)
            self.pre_head_fc_lst.append(fc_layer)

            non_linearity = torch.nn.__dict__[self.nonlinearity](**self.kwargs_nonlinearity)
            self.add_module(f'PreHead_{i}_NonLinearity', non_linearity)
            self.pre_head_fc_lst.append(non_linearity)

    def init_posthead(self, prv_size, post_head_fc_sizes):
        for i, post_head_fc in enumerate(post_head_fc_sizes):
            if i == 0:
                in_features = prv_size
            else:
                in_features = post_head_fc_sizes[i - 1]
            fc_layer = torch.nn.Linear(in_features=in_features, out_features=post_head_fc)
            self.add_module(f'PostHead_{i}', fc_layer)
            self.post_head_fc_lst.append(fc_layer)

            non_linearity = torch.nn.__dict__[self.nonlinearity](**self.kwargs_nonlinearity)    
            self.add_module(f'PostHead_{i}_NonLinearity', non_linearity)
            self.pre_head_fc_lst.append(non_linearity)
    
    def init_classifier(self, prv_size, classifier_fc_sizes):
            for i, classifier_fc in enumerate(classifier_fc_sizes):
                if i == 0:
                    in_features = prv_size
                else:
                    in_features = classifier_fc_sizes[i - 1]
            fc_layer = torch.nn.Linear(in_features=in_features, out_features=classifier_fc)
            self.add_module(f'Classifier_{i}', fc_layer)
            self.classifier_fc_lst.append(fc_layer)

    def reinit_classifier(self):
        for i_layer, layer in enumerate(self.classifier_fc_lst):
            layer.reset_parameters()
    
    def forward_classifier(self, X):
        interim = self.base_model(X)
        interim = self.get_head(interim)
        interim = self.classify(interim)
        return interim

    def forward_latent(self, X):
        interim = self.base_model(X)
        interim = self.get_head(interim)
        interim = self.get_latent(interim)
        return interim


    def get_head(self, base_out):
        head = base_out
        for pre_head_layer in self.pre_head_fc_lst:
          head = pre_head_layer(head)
        return head

    def get_latent(self, head):
        latent = head
        for post_head_layer in self.post_head_fc_lst:
            latent = post_head_layer(latent)
        return latent

    def classify(self, head):
        logit = head
        for classifier_layer in self.classifier_fc_lst:
            logit = classifier_layer(logit)
        return logit

    def set_pre_head_grad(self, requires_grad=True):
        for layer in self.pre_head_fc_lst:
            for param in layer.parameters():
                param.requires_grad = requires_grad
                
    def set_post_head_grad(self, requires_grad=True):
        for layer in self.post_head_fc_lst:
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def set_classifier_grad(self, requires_grad=True):
        for layer in self.classifier_fc_lst:
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def prep_contrast(self):
        self.set_pre_head_grad(requires_grad=True)
        self.set_post_head_grad(requires_grad=True)
        self.set_classifier_grad(requires_grad=False)

    def prep_classifier(self):
        self.set_pre_head_grad(requires_grad=False)
        self.set_post_head_grad(requires_grad=False)
        self.set_classifier_grad(requires_grad=True)




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
            # slice_point=None, # Slice off the model at the slice_point and only keep the prior blocks

            head_pool_method=None,
            head_pool_method_kwargs=None,
            pre_head_fc_sizes=None,
            post_head_fc_sizes=None,
            head_nonlinearity=None,
            head_nonlinearity_kwargs=None,

            # attachment_block=None, # Add the attachment blocks to the end of the base_model
            block_to_unfreeze=None, # Unfreeze the model at and beyond the unfreeze_point
            n_block_toInclude=None, # Unfreeze the model at and beyond the unfreeze_point
            image_out_size=None, # Set the size of the output image

            load_model=False, # Whether the model should be loaded from the filepath_model (or otherwise saved to it)
            ):
        # If loading model, load it from onnx, otherwise create one from scratch using the other parameters
        if load_model:
            self.load_onnx(filepath_model)
        else:
            self.create_model(
                base_model=base_model,
                # slice_point=slice_point,

                head_pool_method=head_pool_method,
                head_pool_method_kwargs=head_pool_method_kwargs,
                pre_head_fc_sizes=pre_head_fc_sizes,
                post_head_fc_sizes=post_head_fc_sizes,
                head_nonlinearity=head_nonlinearity,
                head_nonlinearity_kwargs=head_nonlinearity_kwargs,
                
                # attachment_block=attachment_block,
                block_to_unfreeze=block_to_unfreeze,
                n_block_toInclude=n_block_toInclude,

                image_out_size=image_out_size,
                )
            self.filepath_model = filepath_model
            
    def create_model(
            self,
            base_model=None, # Freeze base_model
            # slice_point=None, # Slice off the model at the slice_point and only keep the prior blocks
            
            head_pool_method=None,
            head_pool_method_kwargs=None,
            pre_head_fc_sizes=None,
            post_head_fc_sizes=None,
            head_nonlinearity=None,
            head_nonlinearity_kwargs=None,
            
            # attachment_block=None, # Add the attachment blocks to the end of the base_model
            block_to_unfreeze=None, # Unfreeze the model at and beyond the unfreeze_point
            n_block_toInclude=None, # Unfreeze the model at and beyond the unfreeze_point
            image_out_size=None,
            ):
            
        ## Model preferences

        base_model_frozen = base_model
        for param in base_model_frozen.parameters():
            param.requires_grad = False

        model_chopped = torch.nn.Sequential(list(base_model_frozen.children())[0][:n_block_toInclude])  ## 0.
        model_chopped_pooled = torch.nn.Sequential(model_chopped, torch.nn.__dict__[head_pool_method](**head_pool_method_kwargs), torch.nn.Flatten())  ## 1.

        data_dim = tuple([1] + list(image_out_size))
        self.model = ModelTackOn(
            model_chopped_pooled.to('cpu'),
            base_model_frozen.to('cpu'),
            data_dim=data_dim,
            pre_head_fc_sizes=pre_head_fc_sizes,
            post_head_fc_sizes=post_head_fc_sizes,
            classifier_fc_sizes=None,
            nonlinearity=head_nonlinearity,
            kwargs_nonlinearity=head_nonlinearity_kwargs,
        )

        mnp = [name for name, param in self.model.named_parameters()]  ## 'model named parameters'
        mnp_blockNums = [name[name.find('.'):name.find('.')+8] for name in mnp]  ## pulls out the numbers just after the model name
        mnp_nums = [get_nums_from_string(name) for name in mnp_blockNums]  ## converts them to numbers
        block_to_freeze_nums = get_nums_from_string(block_to_unfreeze)  ## converts the input parameter specifying the block to freeze into a number for comparison

        m_baseName = mnp[0][:mnp[0].find('.')]

        for ii, (name, param) in enumerate(self.model.named_parameters()):
            if m_baseName in name:
        #         print(name)
                if mnp_nums[ii] < block_to_freeze_nums:
                    param.requires_grad = False
                elif mnp_nums[ii] >= block_to_freeze_nums:
                    param.requires_grad = True

        names_layers_requiresGrad = [( param.requires_grad , name ) for name,param in list(self.model.named_parameters())]

        self.model.forward = self.model.forward_latent
    
    def save_onnx(
        self,
        # filepath_model: Optional[str]=None,
        allow_overwrite: bool=False,
        check_load_onnx_valid: bool=False,
    ):
        """
        Uses ONNX to save the best model as a binary file.

        Args:
            filepath (str): 
                The path to save the model to.
                If None, then the model will not be saved.
            allow_overwrite (bool):
                Whether to allow overwriting of existing files.

        Returns:
            (onnx.ModelProto):
                The ONNX model.
        """
        import datetime
        try:
            import onnx
        except ImportError as e:
            raise ImportError(f'You need to (pip) install onnx to use this method. {e}')
        
        ## Make sure we have what we need
        assert self.model is not None, 'You need to fit the model first.'

        # Convert the model to ONNX format
        ## Prepare initial types

        # torch.onnx.export
        torch.onnx.export(
            self.model,
            (torch.ones(8, 3, 224, 224),),
            self.filepath_model, # "onnx.pb",
            input_names=["x"],
            output_names=["latents"],
            dynamic_axes={
                # dict value: manually named axes
                "x": {0: "batch"},
                # list value: automatic names
                "latents": [0],
            }
        )
        
        if check_load_onnx_valid:
            print('Checking ONNX model...')
            import onnxruntime as ort
            # Create example data
            x = torch.ones((1, 3, 224, 224))

            out_torch_original = self.model(x)
            
            model_loaded = self.load_onnx(self.filepath_model, inplace=False)
            out_torch_loaded = model_loaded(x)

            # Check the Onnx output against PyTorch
            print(torch.max(torch.abs(out_torch_original - out_torch_loaded.detach().numpy())))
            assert np.allclose(out_torch_original, out_torch_loaded.detach().numpy(), atol=1.e-7), "The outputs from the saved and loaded models are different."
            print('Saved ONNX model is valid.')


    def load_onnx(
            self,
            filepath_model=None,
            inplace=True,
            ):
        
        try:
            import onnx
            import torch
            import onnx2torch
        except ImportError as e:
            raise ImportError(f'You need to (pip) install onnx and skl2onnx to use this method. {e}')
        
        """
        Initializes the runtime session.
        """

        ## load ONNX model first
        if isinstance(filepath_model, str):
            model = onnx2torch.convert(filepath_model)
        else:
            raise ValueError(f'path_or_bytes must be either a string or bytes. This error should never be raised.')
        
        if inplace:
            self.model = model
        else:
            return model



class Simclr_Trainer():
    def __init__(
            self,
            dataloader,
            model_container,
                    
            n_epochs=9999999,
            device_train='cuda:0',
            inner_batch_size=256,
            learning_rate=0.01,
            penalty_orthogonality=1.00,
            weight_decay=0.1,
            gamma=1.0000,
            temperature=0.03,
            l2_alpha=0.0000,

            path_saveLog=None
            ):
        self.dataloader = dataloader
        self.model_container = model_container
        self.n_epochs = n_epochs
        self.device_train = device_train
        self.inner_batch_size = inner_batch_size
        self.learning_rate = learning_rate
        self.penalty_orthogonality = penalty_orthogonality
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.temperature = temperature
        self.l2_alpha = l2_alpha
        self.path_saveLog = path_saveLog

    # def train(self, train_loader, val_loader, params):
    #     # Set model to training mode
    #     self.model_container.model.train()

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

    def train(self):
        self.model_container.model.train();
        self.model_container.model.to(self.device_train)
        self.model_container.model.prep_contrast()

        criterion = [CrossEntropyLoss()]
        criterion = [_.to(self.device_train) for _ in criterion]
        optimizer = Adam(
            self.model_container.model.parameters(), 
            lr=self.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=self.gamma,
                                                        )
        
        self.dataloader
        self.model_container
        self.n_epochs
        self.device_train
        self.inner_batch_size
        self.learning_rate
        self.penalty_orthogonality
        self.weight_decay
        self.gamma
        self.temperature
        self.l2_alpha
        self.path_saveLog

        losses_train, losses_val = [], [np.nan]
        for epoch in tqdm.tqdm(range(self.n_epochs)):

            print(f'epoch: {epoch}')
            
            log_function = partial(log_fn, path_log=self.path_saveLog) if self.path_saveLog is not None else lambda x: None

            losses_train = training.epoch_step(
                self.dataloader, 
                self.model_container.model, 
                optimizer, 
                criterion,
                scheduler=scheduler,
                temperature=self.temperature,
                # l2_alpha,
                penalty_orthogonality=self.penalty_orthogonality,
                mode='semi-supervised',
                loss_rolling_train=losses_train, 
                loss_rolling_val=losses_val,
                device=self.device_train, 
                inner_batch_size=self.inner_batch_size,
                verbose=2,
                verbose_update_period=1,
                log_function=log_function,
            )
            
            ## save loss stuff
            if self.path_saveLog is not None:
                np.save(self.path_saveLoss, losses_train)
            
            ## if loss becomes NaNs, don't save the network and stop training
            if torch.isnan(torch.as_tensor(losses_train[-1])):
                break
            
            ## save model
            self.model_container.save_onnx(allow_overwrite=True, check_load_onnx_valid=True)
