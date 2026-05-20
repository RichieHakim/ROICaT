# Imports
import sys
import os
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from functools import partial
from typing import Optional, List, Tuple, Union, Dict, Any

from torch.nn import CrossEntropyLoss
from torch.optim import Adam


## Inlined from roicat.helpers so this module can be loaded standalone (e.g. inside a distributed bundle via sys.path.append + bare `import model`).
def get_nums_from_string(string_with_nums):
    """Return the digits in a string concatenated as an int, or None if no digits are present."""
    _digits = set('0123456789')
    nums = [ch for ch in string_with_nums if ch in _digits]
    if not nums:
        return None
    return int(''.join(nums))


class ModelTackOn(torch.nn.Module):
    """
    Class to attach fully connected layers to the end of a pretrained
    network to create a SimCLR model with "head" and "latent" outputs.
    JZ, RH 2021-2023

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
        nonlinearity (str):
            Nonlinearity to be used in the fully connected layers
        kwargs_nonlinearity (dict):
            Keyword arguments to be passed to the nonlinearity function
        non_singular_pca_size (int):
            Size of the PCA to be used to create the latent space
            NOT YET IMPLEMENTED
            
    Returns:
        model (torch.nn.Module):
            Model with fully connected layers attached
    """

    def __init__(
        self, 
        base_model: torch.nn.Module, 
        un_modified_model: torch.nn.Module,
        data_dim: Tuple[int, int, int, int]=(1,3,36,36),
        pre_head_fc_sizes: List[int]=[100],
        post_head_fc_sizes: List[int]=[100], 
        nonlinearity: str='relu', 
        kwargs_nonlinearity={},
        non_singular_pca_size=None,
    ):
        super(ModelTackOn, self).__init__()
        self.base_model = base_model
        final_base_layer = list(un_modified_model.children())[-1]
        
        self.data_dim = data_dim
        self.non_singular_pca_size = non_singular_pca_size

        self.pre_head_fc_lst = []
        self.post_head_fc_lst = []
            
        self.nonlinearity = nonlinearity
        self.kwargs_nonlinearity = kwargs_nonlinearity

        self.init_prehead(pre_head_fc_sizes)
        self.init_posthead(pre_head_fc_sizes[-1], post_head_fc_sizes)
        # self.init_pca_layer(pre_head_fc_sizes[-1])
    
    def init_prehead(self, pre_head_fc_sizes):
        """
        Initialize the fully connected layers to be attached before the head
        
        Args:
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
    
    # def init_pca_layer(self, pca_size):
    #     """
    #     Initialize the PCA layer with identity weights and biases

    #     Args:
    #         pca_size (int):
    #             Size of the PCA to be used to create the latent space
    #     """
    #     self.pca_layer = torch.nn.Sequential(
    #         torch.nn.Linear(pca_size, pca_size),
    #         torch.nn.Linear(pca_size, pca_size, bias=False)
    #     )
    #     self.pca_layer[0].weight = torch.nn.Parameter(torch.tensor(np.eye(pca_size,),dtype=torch.float32))
    #     self.pca_layer[0].bias = torch.nn.Parameter(torch.tensor(np.zeros(pca_size,),dtype=torch.float32))
    #     self.pca_layer[1].weight = torch.nn.Parameter(torch.tensor(np.eye(pca_size,),dtype=torch.float32))
    #     self.pca_layer[1].bias = torch.nn.Parameter(torch.tensor(np.zeros(pca_size,),dtype=torch.float32))

    #     self.add_module(f'PCA_Layer', self.pca_layer)
    
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
        head = self.get_head(interim)
        latent = self.get_latent(head)
        return latent

    def forward_head(self, X):
        """
        Run the model forward to get the head output of the model (should be used for training PCA)

        Args:
            X (torch.Tensor):
                Input data to be run through the model

        Returns:
            latent (torch.Tensor):
                Latent representation of the input data
        """
        interim = self.base_model(X)
        head = self.get_head(interim)
        return head

    # def forward_head_pca(self, X):
    #     """
    #     Run the model forward to get the head output of the model, passed through a pre-fit PCA layer
    #     (used for classification)

    #     Args:
    #         X (torch.Tensor):
    #             Input data to be run through the model

    #     Returns:
    #         head_pca (torch.Tensor):
    #             Head output of the model, passed through a pre-fit PCA layer
    #     """
    #     interim = self.base_model(X)
    #     head = self.get_head(interim)
    #     head_pca = self.pca_layer(head)[...,:self.non_singular_pca_size] if self.non_singular_pca_size is not None else self.pca_layer(head)
    #     return head_pca

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
        interim = base_out
        for pre_head_layer in self.pre_head_fc_lst:
            interim = pre_head_layer(interim)
        head = interim
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
        interim = head
        for post_head_layer in self.post_head_fc_lst:
            interim = post_head_layer(interim)
        latent = interim
        return latent

    # def set_pca_head_grad(self, requires_grad=False):
    #     """
    #     Set the gradient requirements for the PCA output head layers built on the head

    #     Args:
    #         requires_grad (bool):
    #             Whether or not to require gradients for the FC layers
    #     """
    #     for param in self.pca_layer.parameters():
    #         param.requires_grad = requires_grad
    
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
        # self.set_pca_head_grad(requires_grad=False)

    @property
    def device(self):
        """
        Get the device of the model

        Returns:
            device (torch.device):
                Device of the model
        """
        return next(self.parameters()).device

class Simclr_Model_with_PCA(torch.nn.Module):
    """
    Backbone + head + frozen PCA-whitening layer, fused into a single
    ``nn.Module`` for serialisation as a ``.pth`` state dict.
    RH 2026

    The PCA layer implements centering + orthogonal whitening of the
    256-d head embeddings via a single fused ``nn.Linear``::

        weight = pca.components_                      (pca_size × pca_size)
        bias   = -pca.components_ @ pca_mean          (pca_size,)

    Weights and bias are stored as **buffers** (not Parameters) so they
    travel with ``state_dict`` but are excluded from ``parameters()`` and
    never updated by an optimizer.

    Args:
        backbone (torch.nn.Module):
            Trained SimCLR backbone+head (a ``ModelTackOn`` instance).
            Must accept ``(N, C, H, W)`` and return ``(N, pca_size)``.
        pca_weight (torch.Tensor):
            PCA rotation matrix.  Shape: ``(pca_size, pca_size)``.
        pca_bias (torch.Tensor):
            Fused centering bias ``-components @ mean``.
            Shape: ``(pca_size,)``.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        pca_weight: torch.Tensor,
        pca_bias: torch.Tensor,
    ):
        super().__init__()
        self.backbone = backbone
        ## Register as buffers: saved/loaded with state_dict, not optimised.
        self.register_buffer('pca_weight', pca_weight.clone().detach().float())
        self.register_buffer('pca_bias', pca_bias.clone().detach().float())

    @classmethod
    def from_simclr_and_sklearn_pca(
        cls,
        model_container: 'Simclr_Model',
        pca_sklearn,
        pca_mean: np.ndarray,
        trainer_scale: bool = False,
    ) -> 'Simclr_Model_with_PCA':
        """
        Construct from a trained ``Simclr_Model`` container and a fitted
        ``sklearn.decomposition.PCA``.

        Args:
            model_container (Simclr_Model):
                Trained container; its ``.model`` attribute is used as the
                backbone.
            pca_sklearn:
                A fitted ``sklearn.PCA`` instance.  Uses
                ``pca_sklearn.components_`` (shape: ``(pca_size, pca_size)``).
            pca_mean (np.ndarray):
                The centering mean to bake into the fused bias.  Must be
                supplied explicitly: ``Simclr_PCA_Trainer`` zeros
                ``pca_sklearn.mean_`` before fitting, so
                ``pca_sklearn.mean_`` cannot be used.  Use
                ``trainer.pca_mean_fitted`` (see ``simclr_training_helpers.py``).
            trainer_scale (bool):
                Must be False.  ``scale=True`` is not yet supported.

        Returns:
            (Simclr_Model_with_PCA):
                Ready-to-save module with frozen PCA buffers.
        """
        if pca_mean is None:
            raise ValueError(
                "pca_mean must be supplied explicitly. "
                "Simclr_PCA_Trainer zeros pca_sklearn.mean_ before fitting, "
                "so pca_sklearn.mean_ cannot be used for centering. "
                "Pass trainer.pca_mean_fitted instead."
            )
        if trainer_scale:
            raise ValueError(
                "scale=True is not yet supported in Simclr_Model_with_PCA."
                " Use scale=False (the default) in Simclr_PCA_Trainer."
            )
        components = torch.as_tensor(pca_sklearn.components_, dtype=torch.float32)  # (pca_size, pca_size)
        mean = torch.as_tensor(pca_mean, dtype=torch.float32)  # (pca_size,)
        ## Fused bias: -components @ mean
        bias = -(components @ mean)  # (pca_size,)
        return cls(
            backbone=model_container.model,
            pca_weight=components,
            pca_bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: backbone → (flatten if needed) → fused PCA linear.

        Args:
            x (torch.Tensor):
                Input images.  Shape: ``(N, C, H, W)``.

        Returns:
            (torch.Tensor):
                PCA-whitened embeddings.  Shape: ``(N, pca_size)``.
        """
        h = self.backbone(x)  # (N, pca_size)
        if h.dim() > 2:
            h = h.flatten(start_dim=1)
        ## torch.nn.functional.linear: y = x @ weight.T + bias
        return torch.nn.functional.linear(h, self.pca_weight, self.pca_bias)

    @property
    def device(self) -> torch.device:
        """Device of the first registered buffer."""
        return self.pca_weight.device


def build_backbone(
    base_model: torch.nn.Module,
    head_pool_method: str,
    head_pool_method_kwargs: dict,
    n_block_toInclude: int,
    image_out_size: Tuple[int, int, int],
    pre_head_fc_sizes: List[int],
    post_head_fc_sizes: List[int],
    head_nonlinearity: str,
    head_nonlinearity_kwargs: dict,
) -> ModelTackOn:
    """
    Build the chopped + pooled + flattened backbone and wrap it in a ``ModelTackOn``.

    Freeze-agnostic: caller is responsible for freezing ``base_model`` parameters
    before calling this function if required.

    Args:
        base_model (torch.nn.Module):
            Torchvision base model (full architecture; will be chopped internally).
        head_pool_method (str):
            Name of a ``torch.nn`` pooling class, e.g. ``'AdaptiveAvgPool2d'``.
        head_pool_method_kwargs (dict):
            Kwargs passed to the pooling constructor.
        n_block_toInclude (int):
            Number of children blocks to retain from ``base_model``.
        image_out_size (Tuple[int, int, int]):
            ``(C, H, W)`` input tensor shape used to infer FC layer sizes.
        pre_head_fc_sizes (List[int]):
            Sizes of the fully-connected layers before the head.
        post_head_fc_sizes (List[int]):
            Sizes of the fully-connected layers after the head.
        head_nonlinearity (str):
            Name of a ``torch.nn`` nonlinearity class, e.g. ``'ReLU'``.
        head_nonlinearity_kwargs (dict):
            Kwargs passed to the nonlinearity constructor.

    Returns:
        (ModelTackOn):
            Ready-to-use backbone module on CPU.
    """
    model_chopped = torch.nn.Sequential(list(base_model.children())[0][:n_block_toInclude])
    model_chopped_pooled = torch.nn.Sequential(
        model_chopped,
        torch.nn.__dict__[head_pool_method](**head_pool_method_kwargs),
        torch.nn.Flatten(),
    )
    data_dim = (1, *image_out_size)
    return ModelTackOn(
        base_model=model_chopped_pooled.to('cpu'),
        un_modified_model=base_model.to('cpu'),
        data_dim=data_dim,
        pre_head_fc_sizes=pre_head_fc_sizes,
        post_head_fc_sizes=post_head_fc_sizes,
        nonlinearity=head_nonlinearity,
        kwargs_nonlinearity=head_nonlinearity_kwargs,
    )


class Simclr_Model():
    """
    SimCLR model class

    Args:
        filepath_model_load (str):
            Filepath from which to load a pretrained model
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
        forward_version (str):
            Version of the forward pass to use
    """
    def __init__(
            self,
            filepath_model_load: Optional[torch.nn.Module]=None, # Set filepath to save model
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
            forward_version: Optional[str]=None, # Set the version of the forward pass to use
            ):

        assert filepath_model_load is not None or (
            base_model is not None and
            head_pool_method is not None and
            head_pool_method_kwargs is not None and
            pre_head_fc_sizes is not None and
            post_head_fc_sizes is not None and
            head_nonlinearity is not None and
            head_nonlinearity_kwargs is not None and
            block_to_unfreeze is not None and
            n_block_toInclude is not None and
            image_out_size is not None and
            forward_version is not None
            ), "Either filepath_model_load or every other parameter must be set"

        # If loading model, load it from a checkpoint, otherwise create one from scratch using the other parameters
        if filepath_model_load is not None:
            raise NotImplementedError(
                "Loading a model via filepath_model_load is no longer supported in Simclr_Model.__init__. "
                "Build the architecture with from_dict_params(), then load weights with torch.load() + model.load_state_dict()."
            )
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
            forward_version=forward_version
            )
        self.filepath_model_load = filepath_model_load

    @classmethod
    def from_dict_params(
        cls,
        dict_params_model: dict,
        base_model: torch.nn.Module,
        image_out_size: Tuple[int, int, int],
        forward_version: str,
    ) -> 'Simclr_Model':
        """
        Convenience constructor: pulls standard model kwargs out of ``dict_params['model']``
        and instantiates ``Simclr_Model``.

        Args:
            dict_params_model (dict):
                The ``dict_params['model']`` sub-dict. Expected keys:
                ``head_pool_method``, ``head_pool_method_kwargs``,
                ``pre_head_fc_sizes``, ``post_head_fc_sizes``,
                ``head_nonlinearity``, ``head_nonlinearity_kwargs``,
                ``block_to_unfreeze``, ``n_block_toInclude``.
            base_model (torch.nn.Module):
                Pretrained torchvision backbone instance.
            image_out_size (Tuple[int, int, int]):
                ``(C, H, W)`` input tensor shape (e.g. ``[3, 224, 224]``).
            forward_version (str):
                Forward-pass binding, e.g. ``'forward_latent'`` or ``'forward_head'``.

        Returns:
            (Simclr_Model):
                Newly constructed model container.
        """
        return cls(
            base_model=base_model,
            head_pool_method=dict_params_model['head_pool_method'],
            head_pool_method_kwargs=dict_params_model['head_pool_method_kwargs'],
            pre_head_fc_sizes=dict_params_model['pre_head_fc_sizes'],
            post_head_fc_sizes=dict_params_model['post_head_fc_sizes'],
            head_nonlinearity=dict_params_model['head_nonlinearity'],
            head_nonlinearity_kwargs=dict_params_model['head_nonlinearity_kwargs'],
            block_to_unfreeze=dict_params_model['block_to_unfreeze'],
            n_block_toInclude=dict_params_model['n_block_toInclude'],
            image_out_size=image_out_size,
            forward_version=forward_version,
        )

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
            forward_version=None
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
            forward_version (str):
                Version of the forward pass to use
        """

        for param in base_model.parameters():
            param.requires_grad = False

        self.model = build_backbone(
            base_model=base_model,
            head_pool_method=head_pool_method,
            head_pool_method_kwargs=head_pool_method_kwargs,
            n_block_toInclude=n_block_toInclude,
            image_out_size=tuple(image_out_size),
            pre_head_fc_sizes=pre_head_fc_sizes,
            post_head_fc_sizes=post_head_fc_sizes,
            head_nonlinearity=head_nonlinearity,
            head_nonlinearity_kwargs=head_nonlinearity_kwargs,
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

        # self.model.forward = self.model.forward_latent if forward_version == 'forward_latent' else self.model.forward_head_pca
        self.model.forward = self.model.forward_latent if forward_version == 'forward_latent' else self.model.forward_head
    


def make_model(fwd_version: str, **params) -> 'Simclr_Model_with_PCA':
    """
    Factory called by ``ROInet_embedder`` after loading ``params.json`` from a distributed bundle.
    Rebuilds the chopped + pooled backbone, wraps it with the frozen PCA-whitening layer, and returns a ``Simclr_Model_with_PCA`` whose state_dict the caller loads next.

    ``fwd_version`` is accepted for ROInet_embedder API compatibility but ignored: the wPCA bundle always returns PCA-whitened head output regardless of value.

    Args:
        fwd_version (str): Forward-pass version from ROInet_embedder (``'latent'`` / ``'head'`` / ``'base'``). Ignored.
        **params: All keys from ``params.json`` — ``torchvision_model``, ``head_pool_method``, ``head_pool_method_kwargs``, ``pre_head_fc_sizes``, ``post_head_fc_sizes``, ``head_nonlinearity``, ``head_nonlinearity_kwargs``, ``n_block_toInclude``, ``pca_size``.
    """
    base_model = torchvision.models.__dict__[params['torchvision_model']](weights=None)
    ## ROInet always feeds 224x224 RGB
    backbone = build_backbone(
        base_model=base_model,
        head_pool_method=params['head_pool_method'],
        head_pool_method_kwargs=params['head_pool_method_kwargs'],
        n_block_toInclude=params['n_block_toInclude'],
        image_out_size=(3, 224, 224),
        pre_head_fc_sizes=params['pre_head_fc_sizes'],
        post_head_fc_sizes=params['post_head_fc_sizes'],
        head_nonlinearity=params['head_nonlinearity'],
        head_nonlinearity_kwargs=params['head_nonlinearity_kwargs'],
    )
    ## wPCA bundle wraps the 256-d head output regardless of fwd_version
    backbone.forward = backbone.forward_head

    pca_size = params['pca_size']
    return Simclr_Model_with_PCA(
        backbone=backbone,
        pca_weight=torch.zeros(pca_size, pca_size),
        pca_bias=torch.zeros(pca_size),
    )

