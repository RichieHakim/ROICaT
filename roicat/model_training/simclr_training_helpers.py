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
    """
    Class to attach fully connected layers to the end of a pretrained
    network to create a SimCLR model with "head" and "latent" outputs.
    JZ / RH 2021-2023

    Args:
        base_model (torch.nn.Module):
            Pretrained model to which fully connected layers will be attached
        un_modified_model (torch.nn.Module):
            Pretrained model that has not been modified
        data_dim (tuple):
            Dimensions of the data to be passed through the model
        pre_head_fc_sizes (list):
            List of fully connected layer sizes to be attached before the head
        post_head_fc_sizes (list):
            List of fully connected layer sizes to be attached after the head
        classifier_fc_sizes (list):
            List of fully connected layer sizes to be attached to the head
        nonlinearity (str):
            Nonlinearity to be used in the fully connected layers
        kwargs_nonlinearity (dict):
            Keyword arguments to be passed to the nonlinearity function
            
    Returns:
        model (torch.nn.Module):
            Model with fully connected layers attached
    """

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
        """
        Initialize the fully connected layers to be attached before the head
        
        Args:
            prv_layer (torch.nn.Module):
                Final layer of the base model
            pre_head_fc_sizes (list):
                List of fully connected layer sizes to be attached before the head
        """
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
        """
        Initialize the fully connected layers to be attached after the head

        Args:
            prv_size (int):
                Size of the final layer of the base model
            post_head_fc_sizes (list):
                List of fully connected layer sizes to be attached after the head
        """
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
    
    def forward_latent(self, X):
        """
        Run the model forward to get the latent representation of the data
        (final output of model—used for similarity calculations in SimCLR training)

        Args:
            X (torch.Tensor):
                Input data to be run through the model

        Returns:
            latent (torch.Tensor):
                Latent representation of the input data
        """
        interim = self.base_model(X)
        interim = self.get_head(interim)
        interim = self.get_latent(interim)
        return interim

    def get_head(self, base_out):
        """
        Run the model forward through the FC layers between the base model
        and the head output

        Args:
            base_out (torch.Tensor):
                Output of the base model

        Returns:
            head (torch.Tensor):
                Output of the FC layers (the head output used for classification)
        """
        head = base_out
        for pre_head_layer in self.pre_head_fc_lst:
          head = pre_head_layer(head)
        return head

    def get_latent(self, head):
        """
        Run the model forward through the FC layers between the head output
        and the latent representation

        Args:
            head (torch.Tensor):
                Output of the FC layers (the head output used for classification)

        Returns:
            latent (torch.Tensor):
                Latent representation of the input data (used for SimCLR similarity training)
        """
        latent = head
        for post_head_layer in self.post_head_fc_lst:
            latent = post_head_layer(latent)
        return latent

    def set_pre_head_grad(self, requires_grad=True):
        """
        Set the gradient requirements for the FC layers between the base model
        and the head output

        Args:
            requires_grad (bool):
                Whether or not to require gradients for the FC layers
        """
        for layer in self.pre_head_fc_lst:
            for param in layer.parameters():
                param.requires_grad = requires_grad
                
    def set_post_head_grad(self, requires_grad=True):
        """
        Set the gradient requirements for the FC layers between the head output
        and the latent representation

        Args:
            requires_grad (bool):
                Whether or not to require gradients for the FC layers
        """
        for layer in self.post_head_fc_lst:
            for param in layer.parameters():
                param.requires_grad = requires_grad

    def prep_contrast(self):
        """
        Set the gradient requirements for the FC layers between the base model
        and the head output and the FC layers between the head output
        and the latent representation to True

        Args:
            requires_grad (bool):
                Whether or not to require gradients for the FC layers
        """
        self.set_pre_head_grad(requires_grad=True)
        self.set_post_head_grad(requires_grad=True)




class Simclr_Model():
    """
    SimCLR model class

    Args:
        filepath_model (str):
            Filepath to/from which to save/load the model
        base_model (torch.nn.Module):
            Base torchvision model (or otherwise) to use for the SimCLR model
        head_pool_method (str):
            Pooling method to use for the head
        head_pool_method_kwargs (dict):
            Pooling method kwargs to use for the head  
        pre_head_fc_sizes (list):
            List of fully connected layer sizes to be attached before the head
        post_head_fc_sizes (list):
            List of fully connected layer sizes to be attached after the head
        head_nonlinearity (str):
            Nonlinearity to use after the FC layers
        head_nonlinearity_kwargs (dict):
            Nonlinearity kwargs to use after the FC layers
        block_to_unfreeze (str):
            Name of the block to unfreeze for training
        n_block_toInclude (int):
            Number of blocks to include in the base model
        image_out_size (int):
            Size of the output image (for resizing)
        load_model (bool):
            Whether or not to load the model from the filepath (if not, will initialize from scratch)
    """

    # Load pretrained weights, freeze all layers

    ### TODO: JZ: Download convnext from online source
    ### Freeze untrained layers

    # Chop model off at layer _, pool output, add linear layer unfrozen, flatten
    # Loop through parameters and freeze/unfreeze relevant layers
    # Model to device, prep_contrast, define forward

    def __init__(
            self,
            filepath_model, # Set filepath to save model
            base_model: Optional[torch.nn.Module]=None, # Set base model to use
            head_pool_method: Optional[str]=None, # Set pooling method to use for the head
            head_pool_method_kwargs: Optional[dict]=None, # Set pooling method kwargs to use for the head
            pre_head_fc_sizes: Optional[list]=None, # Set the sizes of the FC layers to be attached before the head
            post_head_fc_sizes: Optional[int]=None, # Set the size of the FC layer to be attached after the head
            head_nonlinearity: Optional[str]=None, # Set the nonlinearity to use after the head
            head_nonlinearity_kwargs: Optional[dict]=None, # Set the nonlinearity kwargs to use after the head
            block_to_unfreeze: Optional[str]=None, # Unfreeze the model at and beyond the unfreeze_point
            n_block_toInclude: Optional[int]=None, # Set the number of blocks to include in the model
            image_out_size: Optional[int]=None, # Set the size of the output image
            load_model: Optional[bool]=False, # Whether or not to load the onnx model from filepath_model (if not, model will be saved to filepath_model)
            ):
        # If loading model, load it from onnx, otherwise create one from scratch using the other parameters
        if load_model:
            self.load_onnx(filepath_model)
        else:
            self.create_model(
                base_model=base_model,
                head_pool_method=head_pool_method,
                head_pool_method_kwargs=head_pool_method_kwargs,
                pre_head_fc_sizes=pre_head_fc_sizes,
                post_head_fc_sizes=post_head_fc_sizes,
                head_nonlinearity=head_nonlinearity,
                head_nonlinearity_kwargs=head_nonlinearity_kwargs,
                block_to_unfreeze=block_to_unfreeze,
                n_block_toInclude=n_block_toInclude,
                image_out_size=image_out_size,
                )
            self.filepath_model = filepath_model
            
    def create_model(
            self,
            base_model=None, # Freeze base_model
            head_pool_method=None,
            head_pool_method_kwargs=None,
            pre_head_fc_sizes=None,
            post_head_fc_sizes=None,
            head_nonlinearity=None,
            head_nonlinearity_kwargs=None,
            block_to_unfreeze=None, # Unfreeze the model at and beyond the unfreeze_point
            n_block_toInclude=None, # Unfreeze the model at and beyond the unfreeze_point
            image_out_size=None,
            ):
        """
        Create the model from scratch using the parameters

        Args:
            base_model (torch.nn.Module):
                Base torchvision model (or otherwise) to use for the SimCLR model
            head_pool_method (str):
                Pooling method to use for the head
            head_pool_method_kwargs (dict):
                Pooling method kwargs to use for the head
            pre_head_fc_sizes (list):
                List of fully connected layer sizes to be attached before the head  
            post_head_fc_sizes (list):
                List of fully connected layer sizes to be attached after the head
            head_nonlinearity (str):
                Nonlinearity to use after the FC layers
            head_nonlinearity_kwargs (dict):
                Nonlinearity kwargs to use after the FC layers
            block_to_unfreeze (str):
                Name of the block to unfreeze for training
            n_block_toInclude (int):
                Number of blocks to include in the base model
            image_out_size (int):
                Size of the output image (for resizing)
        """

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
                if mnp_nums[ii] < block_to_freeze_nums:
                    param.requires_grad = False
                elif mnp_nums[ii] >= block_to_freeze_nums:
                    param.requires_grad = True

        self.model.forward = self.model.forward_latent
    
    def save_onnx(
        self,
        allow_overwrite: bool=False,
        check_load_onnx_valid: bool=False,
    ):
        """
        Uses ONNX to save the current model as a binary file.

        Args:
            allow_overwrite (bool):
                Whether to allow overwriting of existing files.
            check_load_onnx_valid (bool):
                Whether to check that the saved model is valid by loading onnx back in and
                comparing outputs
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
            self.model.prep_contrast()
            self.model.eval()
            out_torch_original = self.model(x).detach().numpy()
            
            model_loaded = self.load_onnx(self.filepath_model, inplace=False)
            # model_loaded.prep_contrast()
            model_loaded.eval()
            out_torch_loaded = model_loaded(x).detach().numpy()

            # Check the Onnx output against PyTorch
            print(np.max(np.abs(out_torch_original - out_torch_loaded)))
            assert np.allclose(out_torch_original, out_torch_loaded, atol=1.e-5), "The outputs from the saved and loaded models are different."
            print('Saved ONNX model is valid.')


            self.model.prep_contrast()
            self.model.train()

            # model_loaded.prep_contrast()
            model_loaded.train()


    def load_onnx(
            self,
            filepath_model=None,
            inplace=True,
            ):
        """
        Loads the ONNX model from a file.

        Args:
            filepath_model (str):
                Path to the ONNX model file.
            inplace (bool):
                Whether to load the model as an attribute or return it.

        Returns:
            model (ModelTackOn):
                The loaded model.
        """
        
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
            
            training_stop_revert_atNan=True,
                    
            n_epochs=9999999,
            device_train='cuda:0',
            inner_batch_size=256,
            learning_rate=0.01,
            penalty_orthogonality=1.00,
            weight_decay=0.1,
            gamma=1.0000,
            temperature=0.03,
            l2_alpha=0.0000,

            path_saveLog=None,
            ):
        """
        Training module to train a SimCLR model from scratch using the provided parameters.

        Args:
            dataloader (torch.utils.data.DataLoader):
                The dataloader to use for training.
            model_container (ModelContainer):
                The model container to use for training.
            training_stop_revert_atNan (bool):
                Whether to revert to the previous model if the loss becomes NaN and stop training.
            n_epochs (int):
                The number of epochs to train for.
            device_train (str):
                The device to train on.
            inner_batch_size (int):
                The batch size to use for training.
            learning_rate (float):
                The learning rate to use for training.
            penalty_orthogonality (float):
                The penalty to apply to the orthogonality of the latent space.
            weight_decay (float):
                The weight decay to use for training.
            gamma (float):
                The gamma to use for training.
            temperature (float):
                The temperature to use for training.
            l2_alpha (float):
                The alpha to use for L2 regularization.
            path_saveLog (str):
                The path to which to save the training log.
        """

        self.dataloader = dataloader
        self.training_stop_revert_atNan = training_stop_revert_atNan
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

    def train(self):
        """
        Trains the model using the saved attributes.
        """
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
            if torch.isnan(torch.as_tensor(losses_train[-1])) and self.training_stop_revert_atNan:
                break
            
            ## save model
            self.model_container.save_onnx(allow_overwrite=True, check_load_onnx_valid=True)
