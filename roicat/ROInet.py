# Documentation style guide:
## - Class init docstrings should be in the class definition docstring.
## - Use a style guide similar to Google's Python style guide except that argument definitions should start on a new indented line after the argument name.
## - If there is more than one argument, use multiple lines for the argument definition code.
## - Example parameters should start on a new line ('\n' should be used before the first one), should start with a dash, and the parameter definition should start on a new indented line.
## - All arguments should have type hints, accurately reflecting the expected type of the argument.
## - Special inputs or conditions related to the arguments should be highlighted using bold for emphasis, italic for optional aspects, and code for specific values or code-related inputs.
## - Keep the return variable name in the docstring for clarity.
## - Keep a consistent line length to improve readability of the docstring.
## - Ensure the clarity of argument descriptions through the use of clear sentence structure and punctuation.

"""
OSF.io links to ROInet versions:

* ROInet_tracking:
    * Info: This version does not include occlusions or large
      affine transformations.
    * Link: https://osf.io/x3fd2/download
    * Hash (MD5 hex): 7a5fb8ad94b110037785a46b9463ea94
* ROInet_classification:
    * Info: This version includes occlusions and large affine
      transformations.
    * Link: https://osf.io/c8m3b/download
    * Hash (MD5 hex): 357a8d9b630ec79f3e015d0056a4c2d5
"""


import sys
from pathlib import Path
import json
import os
import hashlib
import importlib.util
import PIL
import multiprocessing as mp
from functools import partial
import gc
from typing import List, Tuple, Union, Optional, Dict, Any, Callable

import numpy as np
import torch
import torchvision
from torch.nn import Module
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import scipy.signal
import warnings

from . import util, helpers, data_importing


def check_ROI_images(ROI_images: np.ndarray, nan_to_num: bool = True) -> None:
    """
    Warns about ROI images that will pass through the network but give odd
    results: NaNs, Infs, and all-zero images.

    Args:
        ROI_images (np.ndarray):
            The ROI images to check. Shape: *(n_rois, height, width)*.
        nan_to_num (bool):
            If ``False``, NaNs raise instead of warning.

    Raises:
        ValueError: If NaNs are present and ``nan_to_num`` is ``False``.
    """
    if np.any(np.isnan(ROI_images)):
        if nan_to_num:
            warnings.warn('ROICaT WARNING: NaNs detected. You should consider removing these before passing to the network. Using nan_to_num arguments.')
        else:
            raise ValueError('ROICaT ERROR: NaNs detected. You should consider removing these before passing to the network. Use nan_to_num=True to ignore this error.')
    if np.any(np.isinf(ROI_images)):
        warnings.warn('ROICaT WARNING: Infs detected. You should consider removing these before passing to the network.')
    ## Check if any images in any of the sessions are all zeros
    if np.any(np.all(ROI_images == 0, axis=(1, 2))):
        warnings.warn('ROICaT WARNING: Image(s) with all zeros detected. These can pass through the network, but may give weird results.')


def scale_normalize_ROI_images(
    ROI_images: np.ndarray,
    scale: float,
    nan_to_num: bool = True,
    nan_to_num_val: float = 0.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Rescales ROI images by an affine transform, so that ROIs occupy a standard
    fraction of the image frame. **The single implementation of the scale
    normalization step**; :class:`Resizer_ROI_images` and
    :class:`Preprocessor_ROI_images` differ only in how they arrive at ``scale``.

    Args:
        ROI_images (np.ndarray):
            The ROI images to rescale. Shape: *(n_rois, height, width)*.
        scale (float):
            The scale factor. Typically ``1.2 * um_per_pixel * (size_im / 36)``.
        nan_to_num (bool):
            Whether to replace NaNs with ``nan_to_num_val``.
        nan_to_num_val (float):
            The value to replace NaNs with.
        verbose (bool):
            If ``True``, print progress.

    Returns:
        (np.ndarray):
            ROI_images_rs (np.ndarray):
                The rescaled images. Height and width are unchanged.
    """
    check_ROI_images(ROI_images=ROI_images, nan_to_num=nan_to_num)

    if nan_to_num:
        print(f'ROICaT: replacing NaNs with {nan_to_num_val}') if verbose else None
        ROI_images = np.nan_to_num(ROI_images, nan=nan_to_num_val)

    ## np.stack of an empty list raises; the affine preserves height and width
    if ROI_images.shape[0] == 0:
        return ROI_images

    print(f'ROICaT: resizing ROIs') if verbose else None
    return np.stack([resize_affine(img, scale=scale, clamp_range=True) for img in tqdm(ROI_images, mininterval=5, disable=not verbose)], axis=0)
    ## Faster but slightly different results
    # return np.concatenate(
    #     [resize_images(
    #         batch,
    #         scale=scale,
    #         clamp_range=True,
    #     ) for batch in tqdm(
    #         helpers.make_batches(ROI_images, batch_size=10000),
    #         total=np.ceil(len(ROI_images)/10000),
    #         mininterval=5,
    #         unit='images',
    #         unit_scale=10000,
    #         disable=not verbose,
    #     )], axis=0)


def plot_resized_comparison(ROI_images_cat: np.ndarray, ROI_images_rs: np.ndarray) -> None:
    """
    Plots a comparison of the ROI sizes before and after scale normalization.

    Args:
        ROI_images_cat (np.ndarray):
            Array of ROIs before resizing. Shape: *(n_rois, height, width)*.
        ROI_images_rs (np.ndarray):
            Array of resized ROIs. Shape: *(n_rois, height, width)*.
    """
    fig, axs = plt.subplots(2, 1, figsize=(7, 10))
    for ax, images, title in zip(axs, (ROI_images_cat, ROI_images_rs), ('ROI sizes raw', 'ROI sizes resized')):
        ax.plot(np.mean(images > 0, axis=(1, 2)))
        ax.plot(scipy.signal.savgol_filter(np.mean(images > 0, axis=(1, 2)), 501, 3))
        ax.set_xlabel('ROI number')
        ax.set_ylabel('mean npix')
        ax.set_title(title)


class Resizer_ROI_images(util.ROICaT_Module):
    """
    Class for resizing ROIs.
    RH 2023-2024

    Args:
        function_scaleFactor (Callable):
            The function used to convert ``um_per_pixel`` to a scale factor.
            (Default is ``lambda um_per_pixel, size_im: 1.2 * um_per_pixel * (size_im / 36)``)
            Where ``um_per_pixel`` is the number of microns per pixel and
            size_im is the edge length of the image.
        nan_to_num (bool): 
            Whether to replace NaNs with a specific value. (Default is
            ``True``)
        nan_to_num_val (float): 
            The value to replace NaNs with. (Default is *0.0*)
        verbose (bool): 
            If True, print out extra information. (Default is ``False``)
    """
    def __init__(
        self, 
        function_scaleFactor: Callable[[float, int], float]=lambda um_per_pixel, size_im: 1.2 * um_per_pixel * (size_im / 36),
        nan_to_num: bool=True, 
        nan_to_num_val: float=0.0, verbose: bool=True,
        batch_size: int=10000,
    ):
        super().__init__()
        self.nan_to_num = nan_to_num
        self.nan_to_num_val = nan_to_num_val
        self.batch_size = batch_size
        self._verbose = verbose

        ## Store parameter (but not data) args as attributes
        self.params['__init__'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'nan_to_num',
                'nan_to_num_val',
            ],
        )

        self.function_scaleFactor = function_scaleFactor
        
    def _check_ROI_images(self, ROI_images: np.ndarray):
        check_ROI_images(ROI_images=ROI_images, nan_to_num=self.nan_to_num)

    def plot_resized_comparison(self, ROI_images_cat: np.ndarray, ROI_images_rs: np.ndarray):
        """
        Plot a comparison of the ROI sizes before and after resizing.

        Args:
            ROI_images_cat (np.ndarray):
                Array of ROIs to resize. Shape should be (nROIs, height,
                width).
            ROI_images_rs (np.ndarray):
                Array of resized ROIs. Shape should be (nROIs, height, width).
        """
        plot_resized_comparison(ROI_images_cat=ROI_images_cat, ROI_images_rs=ROI_images_rs)

    def resize_ROIs(
        self,
        ROI_images: np.ndarray,  # Array of shape (n_rois, height, width)
        um_per_pixel: float,
    ) -> np.ndarray:
        """
        Resizes the ROI (Region of Interest) images to prepare them for pass
        through network.

        Args:
            ROI_images (np.ndarray): 
                The ROI images to resize. Array of shape *(n_rois, height,
                width)*.
            um_per_pixel (float): 
                The number of microns per pixel. This value is used to rescale
                the ROI images so that they occupy a standard region of the
                image frame.

        Returns:
            (np.ndarray): 
                ROI_images_rs (np.ndarray): 
                    The resized ROI images.
        """
        ## Store parameter (but not data) args as attributes
        self.params['resize_ROIs'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'um_per_pixel',
            ],
        )

        assert isinstance(um_per_pixel, (int, float)), f'um_per_pixel should be an int or float, but is {type(um_per_pixel)}'

        return scale_normalize_ROI_images(
            ROI_images=ROI_images,
            scale=self.function_scaleFactor(um_per_pixel=float(um_per_pixel), size_im=ROI_images.shape[1]),
            nan_to_num=self.nan_to_num,
            nan_to_num_val=self.nan_to_num_val,
            verbose=self._verbose,
        )


class Preprocessor_ROI_images(util.ROICaT_Module):
    """
    The complete preprocessing chain that takes raw ROI images to network-ready
    tensors. **This class is the single definition of that chain.** Every path
    that feeds ROInet — the DataLoader used by tracking and by the training
    notebooks, ``ROInet_embedder.embed``, and
    ``roicat.classification.ClassifierPackage.predict`` — goes through this
    object, so predict-time preprocessing cannot drift from train-time
    preprocessing.
    RH 2026

    The chain has two stages:

    1. **Scale normalization** (numpy, per image): an affine rescale by
       ``factor_scaleFactor * um_per_pixel * (size_im / size_im_reference)`` so
       that an ROI occupies a standard fraction of the frame regardless of the
       optics it was acquired with. Implemented by
       :func:`scale_normalize_ROI_images`, which :class:`Resizer_ROI_images` also
       calls. Note that ``um_per_pixel`` is a property of the *data*, not of the
       model, and so is passed per call — never stored on this object.
    2. **Tensor transforms** (torch, per image): min-max scale the dynamic range
       to [0, 1], resize to ``img_size_out``, tile to ``n_channels_out``
       channels. Exposed as ``self.transforms``, a
       ``torch.nn.Sequential`` that is safe to apply either to one image
       *(1, height, width)* or to a whole batch *(n_images, 1, height, width)*
       with identical per-image results.

    Args:
        scale_normalize (bool):
            If ``True``, apply stage 1. Set to ``False`` only to reproduce a
            pipeline that was trained without scale normalization. (Default is
            ``True``)
        factor_scaleFactor (float):
            Multiplier in the stage-1 scale factor. (Default is *1.2*)
        size_im_reference (int):
            Reference image edge length in the stage-1 scale factor. (Default is
            *36*)
        img_size_out (Tuple[int, int]):
            Height and width of the images the network expects. (Default is
            *(224, 224)*)
        n_channels_out (int):
            Number of channels the network expects. (Default is *3*)
        nan_to_num (bool):
            Whether to replace NaNs. (Default is ``True``)
        nan_to_num_val (float):
            The value to replace NaNs with. (Default is *0.0*)
        verbose (bool):
            If ``True``, print out extra information. (Default is ``True``)
    """
    def __init__(
        self,
        scale_normalize: bool = True,
        factor_scaleFactor: float = 1.2,
        size_im_reference: int = 36,
        img_size_out: Tuple[int, int] = (224, 224),
        n_channels_out: int = 3,
        nan_to_num: bool = True,
        nan_to_num_val: float = 0.0,
        verbose: bool = True,
    ):
        super().__init__()

        if not isinstance(img_size_out, (tuple, list)):
            raise TypeError(f'img_size_out should be a tuple or list of two ints, but is {type(img_size_out)}')
        img_size_out = tuple(int(s) for s in img_size_out)
        if len(img_size_out) != 2:
            raise ValueError(f'img_size_out should have length 2, but has length {len(img_size_out)}')

        self._verbose = verbose
        self.scale_normalize = scale_normalize
        self.factor_scaleFactor = factor_scaleFactor
        self.size_im_reference = size_im_reference
        self.img_size_out = img_size_out
        self.n_channels_out = n_channels_out
        self.nan_to_num = nan_to_num
        self.nan_to_num_val = nan_to_num_val

        ## Store parameter (but not data) args as attributes. This dict IS the
        ## serialized config: see to_dict / from_dict.
        self.params['__init__'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'scale_normalize',
                'factor_scaleFactor',
                'size_im_reference',
                'img_size_out',
                'n_channels_out',
                'nan_to_num',
                'nan_to_num_val',
            ],
        )

        ## Stage 2. Applied per-sample by dataset_simCLR and batched by
        ## transform_images; both give identical per-image results.
        self.transforms = torch.nn.Sequential(
            ScaleDynamicRange(scaler_bounds=(0, 1)),
            torchvision.transforms.Resize(
                size=img_size_out,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            TileChannels(dim=-3, n_channels=n_channels_out),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns the JSON-safe config of this preprocessor.

        Returns:
            (Dict[str, Any]):
                config (Dict[str, Any]):
                    Keyword arguments sufficient to rebuild this object via
                    :meth:`from_dict`.
        """
        out = dict(self.params['__init__'])
        out['img_size_out'] = list(out['img_size_out'])  ## JSON has no tuples
        return out

    @classmethod
    def from_dict(cls, config: Dict[str, Any], verbose: bool = False) -> 'Preprocessor_ROI_images':
        """
        Rebuilds a preprocessor from the output of :meth:`to_dict`.

        Args:
            config (Dict[str, Any]):
                Config dict from :meth:`to_dict`.
            verbose (bool):
                If ``True``, print out extra information. Not part of the
                config, since it does not affect the output. (Default is
                ``False``)

        Returns:
            (Preprocessor_ROI_images):
                preprocessor (Preprocessor_ROI_images):
                    A preprocessor with the given config.

        Raises:
            TypeError: If ``config`` contains unexpected keys.
        """
        return cls(**config, verbose=verbose)

    def get_scaleFactor(self, um_per_pixel: float, size_im: int) -> float:
        """
        The stage-1 scale factor: ``factor_scaleFactor * um_per_pixel * (size_im
        / size_im_reference)``. Parameterized by two numbers rather than held as
        a ``Callable``, so that this object stays JSON-serializable (see
        :meth:`to_dict`) and picklable.

        Args:
            um_per_pixel (float):
                Micrometers per pixel of the images being passed in.
            size_im (int):
                Edge length of the images being passed in.

        Returns:
            (float):
                scale (float):
                    The factor to rescale the images by.
        """
        return self.factor_scaleFactor * float(um_per_pixel) * (size_im / self.size_im_reference)

    def scale_normalize_images(
        self,
        ROI_images: Union[np.ndarray, List[np.ndarray]],
        um_per_pixel: Union[float, List[float]],
    ) -> np.ndarray:
        """
        Stage 1: rescale ROI images so that ROIs occupy a standard fraction of
        the frame. Sessions are resized with their own ``um_per_pixel`` and then
        concatenated.

        Args:
            ROI_images (Union[np.ndarray, List[np.ndarray]]):
                Either one array of shape *(n_rois, height, width)* or a list of
                such arrays, one per session.
            um_per_pixel (Union[float, List[float]]):
                Micrometers per pixel of the images being passed in. A float, or
                one float per session.

        Returns:
            (np.ndarray):
                ROI_images_rs (np.ndarray):
                    The rescaled images, concatenated over sessions. Shape:
                    *(n_rois_total, height, width)*.
        """
        ## Accept a single array (one session) without the list-wrapping warning
        ROI_images = [ROI_images,] if isinstance(ROI_images, np.ndarray) else ROI_images
        ROI_images = data_importing.Data_roicat._fix_ROI_images(ROI_images=ROI_images)
        ## Cast to float first: _fix_um_per_pixel rejects ints
        um_per_pixel = float(um_per_pixel) if isinstance(um_per_pixel, (int, float)) else [float(u) for u in um_per_pixel]
        um_per_pixel = data_importing.Data_roicat._fix_um_per_pixel(um_per_pixel=um_per_pixel, n_sessions=len(ROI_images))

        if not self.scale_normalize:
            ## Preserve the NaN handling that stage 1 would otherwise have done
            images_cat = np.concatenate(ROI_images, axis=0)
            check_ROI_images(ROI_images=images_cat, nan_to_num=self.nan_to_num)
            return np.nan_to_num(images_cat, nan=self.nan_to_num_val) if self.nan_to_num else images_cat

        print(f'Starting Image Resizer') if self._verbose else None
        ## Each session gets its own scale factor, from its own um_per_pixel
        return np.concatenate([
            scale_normalize_ROI_images(
                ROI_images=ROI_images[ii],
                scale=self.get_scaleFactor(um_per_pixel=um_per_pixel[ii], size_im=ROI_images[ii].shape[1]),
                nan_to_num=self.nan_to_num,
                nan_to_num_val=self.nan_to_num_val,
                verbose=self._verbose,
            ) for ii in range(len(ROI_images))
        ], axis=0)

    def transform_images(self, ROI_images: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Stage 2: min-max scale, resize, and tile channels for a batch of images.

        Args:
            ROI_images (Union[np.ndarray, torch.Tensor]):
                Images of shape *(n_rois, height, width)*.

        Returns:
            (torch.Tensor):
                images_transformed (torch.Tensor):
                    Shape: *(n_rois, n_channels_out, *img_size_out)*. dtype:
                    ``torch.float32``.
        """
        x = torch.as_tensor(ROI_images, dtype=torch.float32)  # (n_rois, height, width)
        if x.ndim != 3:
            raise ValueError(f'ROI_images should be 3-D (n_rois, height, width), but has ndim={x.ndim}')
        if x.shape[0] == 0:
            ## Skip the transforms: torchvision resize of an empty batch is not well defined
            return x.reshape(0, self.n_channels_out, *self.img_size_out)
        return self.transforms(x[:, None, ...])  # (n_rois, 1, h, w) -> (n_rois, n_channels_out, *img_size_out)

    def preprocess(
        self,
        ROI_images: Union[np.ndarray, List[np.ndarray]],
        um_per_pixel: Union[float, List[float]],
    ) -> torch.Tensor:
        """
        Runs the full chain: stage 1 then stage 2.

        Args:
            ROI_images (Union[np.ndarray, List[np.ndarray]]):
                Either one array of shape *(n_rois, height, width)* or a list of
                such arrays, one per session.
            um_per_pixel (Union[float, List[float]]):
                Micrometers per pixel of the images being passed in. A float, or
                one float per session.

        Returns:
            (torch.Tensor):
                images_preprocessed (torch.Tensor):
                    Network-ready images. Shape: *(n_rois_total,
                    n_channels_out, *img_size_out)*.
        """
        return self.transform_images(ROI_images=self.scale_normalize_images(
            ROI_images=ROI_images,
            um_per_pixel=um_per_pixel,
        ))

    def __repr__(self):
        return f"Preprocessor_ROI_images({self.to_dict()})"


class Dataloader_ROInet(util.ROICaT_Module):
    """
    Class for creating a dataloader for the ROInet network.
    JZ, RH 2023
        
    Args:
        ROI_images (np.ndarray):
            Array of ROIs to resize. Shape should be (nROIs, height,
            width).
        pref_plot (bool): 
            If ``True``, plots the sizes of the ROI images before and after
            normalization. (Default is ``False``)
        batchSize_dataloader (int): 
            The batch size to use for the DataLoader. (Default is *8*)
        pinMemory_dataloader (bool): 
            If ``True``, pins the memory of the DataLoader, as per PyTorch's
            best practices. (Default is ``True``)
        numWorkers_dataloader (int): 
            The number of worker processes for data loading. (Default is
            *-1*)
        persistentWorkers_dataloader (bool): 
            If ``True``, uses persistent worker processes. (Default is
            ``True``)
        prefetchFactor_dataloader (int): 
            The prefetch factor for data loading. (Default is *2*)
        transforms (Optional[Callable]): 
            The transforms to use for the DataLoader. If ``None``, the
            function will only scale dynamic range (to 0-1), resize (to
            img_size_out dimensions), and tile channels (to 3) as a minimum
            to pass images through the network. (Default is ``None``)
        n_transforms (int):
            The number of times to apply the transforms to each image. Should
            be 1 for inference and 2 for training. (Default is *1*)
        img_size_out (Tuple[int, int]): 
            The image output dimensions of DataLoader if transforms is
            ``None``. (Default is *(224, 224)*)
        jit_script_transforms (bool): 
            If ``True``, converts the transforms pipeline into a TorchScript
            pipeline, potentially improving calculation speed but can cause
            problems with multiprocessing. (Default is ``False``)
        shuffle (bool):
            If ``True``, shuffles the data. Should be set to ``True`` for
            SimCLR training. (Default is ``False``)
        drop_last (bool):
            If ``True``, drops the last batch if it is not full. Should be
            set to ``True`` for SimCLR training. (Default is ``False``)
        verbose (bool):
            If ``True``, print out extra information. (Default is ``True``)    """
    def __init__(
            self,
            ROI_images: np.ndarray,
            batchSize_dataloader: int = 8,
            pinMemory_dataloader: bool = True,
            numWorkers_dataloader: int = -1,
            persistentWorkers_dataloader: bool = True,
            prefetchFactor_dataloader: int = 2,
            transforms: Optional[Callable] = None,
            n_transforms: int = 1,
            img_size_out: Tuple[int, int] = (224, 224),
            jit_script_transforms: bool = False,
            shuffle_dataloader: bool = False,
            drop_last_dataloader: bool = False,
            verbose: bool = True,
        ):
        super().__init__()

        self._verbose = verbose
        numWorkers_dataloader = mp.cpu_count() if numWorkers_dataloader == -1 else numWorkers_dataloader

        ## Store parameter (but not data) args as attributes
        self.params['__init__'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'batchSize_dataloader',
                'pinMemory_dataloader',
                'numWorkers_dataloader',
                'persistentWorkers_dataloader',
                'prefetchFactor_dataloader',
                'n_transforms',
                'img_size_out',
                'jit_script_transforms',
                'shuffle_dataloader',
                'drop_last_dataloader',
                'verbose',
            ],
        )

        ## Type checking / correction
        if not isinstance(img_size_out, (tuple, list)):
            assert isinstance(img_size_out, int), f'img_size_out should be a tuple or list, but is {type(img_size_out)}'
            img_size_out = (img_size_out, img_size_out)

        ## Default transforms come from Preprocessor_ROI_images so that the chain is
        ## defined in exactly one place (see Preprocessor_ROI_images stage 2).
        transforms = Preprocessor_ROI_images(
            img_size_out=img_size_out,
            verbose=False,
        ).transforms if transforms is None else transforms

        if jit_script_transforms:
            if numWorkers_dataloader > 0:
                warnings.warn("\n\nWarning: Converting transforms to a jit-based script has been known to cause issues on Windows when numWorkers_dataloader > 0. If self.generate_latents() raises an Exception similar to 'Tried to serialize object __torch__.torch.nn.modules.container.Sequential which does not have a __getstate__ method defined!' consider setting numWorkers_dataloader=0 or jit_script_transforms=False.\n")
            self.transforms = torch.jit.script(transforms)
        else:
            self.transforms = transforms
        
        print(f'Defined image transformations: {transforms}') if self._verbose else None
        self.dataset = dataset_simCLR(
                X=torch.as_tensor(ROI_images, device='cpu', dtype=torch.float32),
                y=torch.as_tensor(torch.zeros(ROI_images.shape[0]), device='cpu', dtype=torch.float32),
                n_transforms=n_transforms,
                transform=self.transforms,
                DEVICE='cpu',
                dtype_X=torch.float32,
            )
        print(f'Defined dataset') if self._verbose else None
        ## torch's DataLoader rejects prefetch_factor and persistent_workers when
        ## num_workers==0, so only pass them when there are workers.
        kwargs_workers = {
            'persistent_workers': persistentWorkers_dataloader,
            'prefetch_factor': prefetchFactor_dataloader,
        } if numWorkers_dataloader > 0 else {}
        self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batchSize_dataloader,
                shuffle=shuffle_dataloader,
                drop_last=drop_last_dataloader,
                pin_memory=pinMemory_dataloader,
                num_workers=numWorkers_dataloader,
                **kwargs_workers,
        )
        print(f'Defined dataloader') if self._verbose else None


def import_model_py_from_bundle(filepath_model_py: Union[str, Path]):
    """
    Imports a network bundle's ``model.py`` by path, under a name derived from its
    own bytes.

    A plain ``import model`` would let the **first** bundle loaded in a process
    serve every later one out of ``sys.modules['model']``, so a second embedder
    would build its network from the first bundle's code while carrying its own
    ``params.json`` and weights. Nothing raises when that happens: ``load_state_dict``
    succeeds whenever the two files produce matching parameter names, which is the
    case for successive releases of the same architecture. The bare name also
    collides with any user module already registered as ``model``. This is the same
    technique :meth:`roicat.classification.package.ClassifierPackage._unpack_embedder`
    uses to load a packet's bundled ``model.py``, differing in one respect: the name
    keys on the bytes rather than on a tempdir's lifetime, so the same bundle
    imported twice returns the same module object, and a process holding one embedder
    sees exactly what it saw before this function existed.

    Args:
        filepath_model_py (Union[str, Path]):
            Path to the bundle's ``model.py``.

    Returns:
        (module):
            module (module):
                The imported module, also registered in ``sys.modules`` under
                ``roicat_roinet_<sha256[:12]>``.
    """
    filepath_model_py = str(filepath_model_py)
    sha = hashlib.sha256(Path(filepath_model_py).read_bytes()).hexdigest()[:12]
    name_module = f"roicat_roinet_{sha}"
    ## The name is derived from the file's bytes, so a cache hit is the same code by
    ## construction and two distinct bundles can never land in the same slot.
    if name_module in sys.modules:
        return sys.modules[name_module]

    spec_import = importlib.util.spec_from_file_location(name_module, filepath_model_py)
    module_model = importlib.util.module_from_spec(spec_import)
    ## Insert before exec so any internal pickle/torch references resolve.
    sys.modules[name_module] = module_model
    try:
        spec_import.loader.exec_module(module_model)
    except Exception:
        sys.modules.pop(name_module, None)
        raise
    return module_model


class ROInet_embedder(util.ROICaT_Module):
    """
    Class for loading the ROInet model, preparing data for it, and running it.
    RH, JZ 2022
    
    OSF.io links to ROInet versions:

    * ROInet_tracking:
        * Info: This version does not include occlusions or large affine
          transformations.
        * Link: https://osf.io/x3fd2/download
        * Hash (MD5 hex): 7a5fb8ad94b110037785a46b9463ea94
    * ROInet_classification:
        * Info: This version includes occlusions and large affine
          transformations.
        * Link: https://osf.io/c8m3b/download
        * Hash (MD5 hex): 357a8d9b630ec79f3e015d0056a4c2d5
    
    Args:
        dir_networkFiles (str): 
            Directory to find an existing ROInet.zip file or download and
            extract a new one into.
        device (str): 
            Device to use for the model and data. (Default is ``'cpu'``)
        download_method (str): 
            Approach to downloading the network files. Options are: \n
            * ``'check_local_first'``: Check if the network files are already in
              dir_networkFiles, if so, use them.
            * ``'force_download'``: Download an ROInet.zip file from
              download_url.
            * ``'force_local'``: Use an existing local copy of an ROInet.zip
              file, if they don't exist, raise an error. Hash checking is done
              and download_hash must be specified. \n
            (Default is ``'check_local_first'``)
        download_url (str): 
            URL to download the ROInet.zip file from.
            (Default is https://osf.io/x3fd2/download)
        download_hash (dict): 
            MD5 hash of the ROInet.zip file. This can be obtained from
            ROICaT documentation. If you don't have one, use
            download_method='force_download' and determine the hash using
            helpers.hash_file(). (Default is ``None``)
        names_networkFiles (dict): 
            Names of the files in the ROInet.zip file. If uncertain, leave
            as None. The dictionary should have the form: \n
            ``{'params': 'params.json', 'model': 'model.py', 'state_dict':
            'ConvNext_tiny__1_0_unfrozen__simCLR.pth',}`` \n
            Where 'params' is the parameters used to train the network
            (usually a .json file), 'model' is the model definition (usually
            a .py file), and 'state_dict' are the weights of the network
            (usually a .pth file). (Default is ``None``)
        forward_pass_version (str): 
            Version of the forward pass to use. Options are 'latent' (return
            the post-head output latents, use this for tracking), 'head'
            (return the output of the head layers, use this for
            classification), and 'base' (return the output of the base
            model). (Default is ``'latent'``)
        verbose (bool): 
            If True, print out extra information. (Default is ``True``)
    """
    def __init__(
        self,
        dir_networkFiles: str,
        device: str = 'cpu',
        download_method: str = 'check_local_first',
        download_url: str = 'https://osf.io/x3fd2/download',
        download_hash: dict = None,
        names_networkFiles: dict = None,
        forward_pass_version: str = 'latent',
        verbose: bool = True,
    ):
        ## Imports
        super().__init__()

        ## Store parameter (but not data) args as attributes
        self.params['__init__'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'dir_networkFiles',
                'device',
                'download_method',
                'download_url',
                'download_hash',
                'names_networkFiles',
                'forward_pass_version',
                'verbose',
            ],
        )

        self._device = device
        self._verbose = verbose
        self._dir_networkFiles = dir_networkFiles
        self._download_url = download_url

        self._download_path_save = str(Path(self._dir_networkFiles).resolve() / 'ROInet.zip')

        fn_download = partial(
            helpers.download_file,
            path_save=self._download_path_save,
            hash_type='MD5',
            hash_hex=download_hash,
            mkdir=True,
            allow_overwrite=True,
            write_mode='wb',
            verbose=self._verbose,
            chunk_size=1024,
        )

        ## Find or download network files
        if download_method == 'force_download':
            fn_download(url=self._download_url, check_local_first=False, check_hash=False)

        if download_method == 'check_local_first':
            # assert download_hash is not None, "if using download_method='check_local_first' download_hash cannot be None. Either determine the hash of the zip file or use download_method='force_download'."
            fn_download(url=self._download_url, check_local_first=True, check_hash=True)

        if download_method == 'force_local':
            # assert download_hash is not None, "if using download_method='force_local' download_hash cannot be None"
            assert Path(self._download_path_save).exists(), f"if using download_method='force_local' the network files must exist in {self._download_path_save}"
            fn_download(url=None, check_local_first=True, check_hash=True)

        ## Extract network files from zip
        paths_extracted = helpers.extract_zip(
            path_zip=self._download_path_save,
            path_extract=self._dir_networkFiles,
            verbose=self._verbose,
        )

        ## Find network files
        if names_networkFiles is None:
            names_networkFiles = {
                'params': 'params.json',
                'model': 'model.py',
                'state_dict': '.pth',
            }
        paths_networkFiles = {}
        paths_networkFiles['params'] = [p for p in paths_extracted if names_networkFiles['params'] in str(Path(p).name)][0]
        paths_networkFiles['model'] = [p for p in paths_extracted if names_networkFiles['model'] in str(Path(p).name)][0]
        paths_networkFiles['state_dict'] = [p for p in paths_extracted if names_networkFiles['state_dict'] in str(Path(p).name)][0]

        ## Import network files. By path under a bytes-derived module name, so a second
        ## embedder built in this process does not get handed this bundle's code.
        model = import_model_py_from_bundle(filepath_model_py=paths_networkFiles['model'])
        print(f"Imported model from {paths_networkFiles['model']}") if self._verbose else None

        ## Everything needed to rebuild this network from the bundle alone. Read by
        ## roicat.classification.ClassifierPackage when packing this embedder.
        self.filepath_model_py = str(paths_networkFiles['model'])
        self.forward_pass_version = forward_pass_version

        with open(paths_networkFiles['params']) as f:
            self.params_model = json.load(f)
            print(f"Loaded params_model from {paths_networkFiles['params']}") if self._verbose else None
            self.net = model.make_model(fwd_version=forward_pass_version, **self.params_model)
            print(f"Generated network using params_model") if self._verbose else None

        ## Prep network and load state_dict
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.eval()

        self.net.load_state_dict(torch.load(
            f=paths_networkFiles['state_dict'],
            map_location=torch.device(self._device),
            weights_only=True,
        ))
        print(f'Loaded state_dict into network from {paths_networkFiles["state_dict"]}') if self._verbose else None

        self.net = self.net.to(self._device)
        print(f'Loaded network onto device {self._device}') if self._verbose else None

    @property
    def arch_kwargs(self) -> Dict[str, Any]:
        """
        The bundle's ``params.json`` contents, verbatim. Passed back to the
        bundled ``model.py``'s ``make_model`` to rebuild this architecture.
        """
        return dict(self.params_model)

    def __repr__(self):
        device = self._device if hasattr(self, '_device') else '?'
        has_latents = hasattr(self, 'latents')
        n_latents = self.latents.shape[0] if has_latents else 0
        return (
            f"ROInet_embedder(device='{device}', "
            f"n_latents={n_latents if has_latents else 'not generated'})"
        )

    def generate_dataloader(
        self,
        ROI_images: List[np.ndarray],
        um_per_pixel: Union[float, List[float]],
        resize_ROI_images: bool = True,
        nan_to_num: bool = True,
        nan_to_num_val: float = 0.0,
        pref_plot: bool = False,
        batchSize_dataloader: int = 8,
        pinMemory_dataloader: bool = True,
        numWorkers_dataloader: int = -1,
        persistentWorkers_dataloader: bool = True,
        prefetchFactor_dataloader: int = 2,
        transforms: Optional[Callable] = None,
        img_size_out: Tuple[int, int] = (224, 224),
        jit_script_transforms: bool = False,
    ):
        """
        Generates a PyTorch DataLoader for a list of Region of Interest (ROI)
        images. Performs preprocessing such as rescaling, normalization, and
        resizing.

        Args:
            ROI_images (List[np.ndarray]): 
                The ROI images to use for the dataloader. List of arrays, each
                array corresponds to a session and is of shape *(n_rois, height,
                width)*.
            um_per_pixel (Union[float, List[float]]):
                The conversion factor from pixels to microns. This is used to scale
                the ROI_images to a common size. Should either be a float or a list
                of floats, one for each session.
            resize_ROI_images (bool):
                If ``True``, resizes the ROI images to a common size. (Default is
                ``True``)
            nan_to_num (bool): 
                Whether to replace NaNs with a specific value. (Default is
                ``True``)
            nan_to_num_val (float): 
                The value to replace NaNs with. (Default is *0.0*)
            pref_plot (bool): 
                If ``True``, plots the sizes of the ROI images before and after
                normalization. (Default is ``False``)
            batchSize_dataloader (int): 
                The batch size to use for the DataLoader. (Default is *8*)
            pinMemory_dataloader (bool): 
                If ``True``, pins the memory of the DataLoader, as per PyTorch's
                best practices. (Default is ``True``)
            numWorkers_dataloader (int): 
                The number of worker processes for data loading. (Default is
                *-1*)
            persistentWorkers_dataloader (bool): 
                If ``True``, uses persistent worker processes. (Default is
                ``True``)
            prefetchFactor_dataloader (int): 
                The prefetch factor for data loading. (Default is *2*)
            transforms (Optional[Callable]): 
                The transforms to use for the DataLoader. If ``None``, the
                function will only scale dynamic range (to 0-1), resize (to
                img_size_out dimensions), and tile channels (to 3) as a minimum
                to pass images through the network. A non-``None`` value replaces
                ``self.preprocessor``'s chain for this DataLoader only, which makes
                the resulting latents unpackable: see
                :class:`~roicat.classification.package.ClassifierPackage`. (Default is
                ``None``)
            img_size_out (Tuple[int, int]): 
                The image output dimensions of DataLoader if transforms is
                ``None``. (Default is *(224, 224)*)
            jit_script_transforms (bool): 
                If ``True``, converts the transforms pipeline into a TorchScript
                pipeline, potentially improving calculation speed but can cause
                problems with multiprocessing. (Default is ``False``)

        Returns:
            (np.ndarray): 
                ROI_images (np.ndarray): 
                    The ROI images after normalization and resizing. Shape is
                    *(n_sessions, n_rois, n_channels, height, width)*.

        Example:
            .. highlight:: python
            .. code-block:: python

                dataloader = generate_dataloader(ROI_images)
        """
        um_per_pixel = data_importing.Data_roicat._fix_um_per_pixel(um_per_pixel=um_per_pixel, n_sessions=len(ROI_images))
        ROI_images = data_importing.Data_roicat._fix_ROI_images(ROI_images=ROI_images)

        ## Store parameter (but not data) args as attributes
        self.params['generate_dataloader'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'um_per_pixel',
                'nan_to_num',
                'nan_to_num_val',
                'pref_plot',
                'batchSize_dataloader',
                'pinMemory_dataloader',
                'numWorkers_dataloader',
                'persistentWorkers_dataloader',
                'prefetchFactor_dataloader',
                'img_size_out',
                'jit_script_transforms',
            ],
        )    

        ## The preprocessor owns the whole chain and is the object to hand to
        ## ClassifierPackage so that inference reuses this exact configuration.
        self.preprocessor = Preprocessor_ROI_images(
            scale_normalize=resize_ROI_images,
            img_size_out=img_size_out,
            nan_to_num=nan_to_num,
            nan_to_num_val=nan_to_num_val,
            verbose=self._verbose,
        )
        self.ROI_images_rs = self.preprocessor.scale_normalize_images(
            ROI_images=ROI_images,
            um_per_pixel=um_per_pixel,
        )
        plot_resized_comparison(
            ROI_images_cat=np.concatenate(ROI_images, axis=0),
            ROI_images_rs=self.ROI_images_rs,
        ) if pref_plot else None

        print(f'Creating dataloader') if self._verbose else None
        dataloader_generator = Dataloader_ROInet(
            ROI_images=self.ROI_images_rs,
            batchSize_dataloader=batchSize_dataloader,
            pinMemory_dataloader=pinMemory_dataloader,
            numWorkers_dataloader=numWorkers_dataloader,
            persistentWorkers_dataloader=persistentWorkers_dataloader,
            prefetchFactor_dataloader=prefetchFactor_dataloader,
            transforms=self.preprocessor.transforms if transforms is None else transforms,
            n_transforms=1,
            img_size_out=img_size_out,
            jit_script_transforms=jit_script_transforms,
            shuffle_dataloader=False,
            drop_last_dataloader=False,
            verbose=self._verbose,
        )

        self.transforms = dataloader_generator.transforms
        self.dataset = dataloader_generator.dataset
        self.dataloader = dataloader_generator.dataloader
        ## A caller-supplied chain bypasses self.preprocessor, which is the only
        ## thing a ClassifierPackage can store. Recorded so that packing refuses
        ## rather than silently preprocessing inference images a different way.
        self._transforms_custom = transforms is not None
        return self.ROI_images_rs

    def generate_latents(self) -> torch.Tensor:
        """
        Passes the data in the dataloader through the network and generates latents.

        Returns:
            (torch.Tensor): 
                latents (torch.Tensor): 
                    Latents for each ROI (Region of Interest).
        """
        if hasattr(self, 'dataloader') == False:
            raise Exception('dataloader not defined. Call generate_dataloader() first.')

        print(f'starting: running data through network')
        self.latents = torch.cat([self.net(data[0][0].to(self._device)).detach() for data in tqdm(self.dataloader, mininterval=5)], dim=0).cpu()
        print(f'completed: running data through network')

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        return self.latents

    def embed(
        self,
        ROI_images: Union[np.ndarray, List[np.ndarray]],
        um_per_pixel: Union[float, List[float]],
        batch_size: int = 256,
        device: Optional[str] = None,
        preprocessor: Optional[Preprocessor_ROI_images] = None,
    ) -> torch.Tensor:
        """
        Preprocesses ROI images and runs them through the network in one call,
        returning latents. Equivalent to ``generate_dataloader()`` followed by
        ``generate_latents()``, but batched in memory rather than through a
        DataLoader, so it needs no worker processes and never materializes the
        whole *(n_rois, 3, 224, 224)* tensor.

        Args:
            ROI_images (Union[np.ndarray, List[np.ndarray]]):
                Either one array of shape *(n_rois, height, width)* or a list of
                such arrays, one per session.
            um_per_pixel (Union[float, List[float]]):
                Micrometers per pixel **of the images being passed in**. A float,
                or one float per session. This is a property of the data, so it
                must be given for every call.
            batch_size (int):
                Number of ROIs preprocessed and passed through the network at a
                time. (Default is *256*)
            device (Optional[str]):
                Torch device to run on. If ``None``, the device given at
                initialization is used.
            preprocessor (Optional[Preprocessor_ROI_images]):
                Preprocessor to use. If ``None``, the one built by a previous
                call to ``generate_dataloader`` is reused, or a default one is
                built.

        Returns:
            (torch.Tensor):
                latents (torch.Tensor):
                    Latents for each ROI, on the CPU. Shape: *(n_rois_total,
                    n_features)*, or *(0, 0)* if no ROIs were given. Also stored
                    as ``self.latents``, as ``generate_latents`` does.
        """
        if preprocessor is None:
            preprocessor = getattr(self, 'preprocessor', None)
        if preprocessor is None:
            preprocessor = Preprocessor_ROI_images(verbose=self._verbose)
        self.preprocessor = preprocessor

        device = self._device if device is None else device
        if batch_size < 1:
            raise ValueError(f'batch_size should be >= 1, but is {batch_size}')

        ## Stage 1 for all images at once (it is a numpy op on small images), then
        ## stage 1 output is chunked through stage 2 + the network.
        ROI_images_rs = preprocessor.scale_normalize_images(
            ROI_images=ROI_images,
            um_per_pixel=um_per_pixel,
        )  # (n_rois_total, height, width)

        self.net.to(device).eval()
        latents = []
        with torch.no_grad():
            for i_start in tqdm(
                range(0, ROI_images_rs.shape[0], batch_size),
                mininterval=5,
                disable=not self._verbose,
                unit='batch',
            ):
                images = preprocessor.transform_images(
                    ROI_images=ROI_images_rs[i_start : i_start + batch_size],
                )  # (n_batch, n_channels_out, *img_size_out)
                latents.append(self.net(images.to(device)).detach().cpu())
                del images

        self.latents = torch.cat(latents, dim=0) if len(latents) > 0 else torch.empty((0, 0))
        helpers.clear_gpu_cache()
        return self.latents


class ROInet_embedder_original(util.ROICaT_Module):
    """
    Class for loading the ROInet model, preparing data for it, and running it.
    RH, JZ 2022
    
    OSF.io links to ROInet versions:

    * ROInet_tracking:
        * Info: This version does not include occlusions or large affine
          transformations.
        * Link: https://osf.io/x3fd2/download
        * Hash (MD5 hex): 7a5fb8ad94b110037785a46b9463ea94
    * ROInet_classification:
        * Info: This version includes occlusions and large affine
          transformations.
        * Link: https://osf.io/c8m3b/download
        * Hash (MD5 hex): 357a8d9b630ec79f3e015d0056a4c2d5
    
    Args:
        dir_networkFiles (str): 
            Directory to find an existing ROInet.zip file or download and
            extract a new one into.
        device (str): 
            Device to use for the model and data. (Default is ``'cpu'``)
        download_method (str): 
            Approach to downloading the network files. Options are: \n
            * ``'check_local_first'``: Check if the network files are already in
              dir_networkFiles, if so, use them.
            * ``'force_download'``: Download an ROInet.zip file from
              download_url.
            * ``'force_local'``: Use an existing local copy of an ROInet.zip
              file, if they don't exist, raise an error. Hash checking is done
              and download_hash must be specified. \n
            (Default is ``'check_local_first'``)
        download_url (str): 
            URL to download the ROInet.zip file from.
            (Default is https://osf.io/x3fd2/download)
        download_hash (dict): 
            MD5 hash of the ROInet.zip file. This can be obtained from
            ROICaT documentation. If you don't have one, use
            download_method='force_download' and determine the hash using
            helpers.hash_file(). (Default is ``None``)
        names_networkFiles (dict): 
            Names of the files in the ROInet.zip file. If uncertain, leave
            as None. The dictionary should have the form: \n
            ``{'params': 'params.json', 'model': 'model.py', 'state_dict':
            'ConvNext_tiny__1_0_unfrozen__simCLR.pth',}`` \n
            Where 'params' is the parameters used to train the network
            (usually a .json file), 'model' is the model definition (usually
            a .py file), and 'state_dict' are the weights of the network
            (usually a .pth file). (Default is ``None``)
        forward_pass_version (str): 
            Version of the forward pass to use. Options are 'latent' (return
            the post-head output latents, use this for tracking), 'head'
            (return the output of the head layers, use this for
            classification), and 'base' (return the output of the base
            model). (Default is ``'latent'``)
        verbose (bool): 
            If True, print out extra information. (Default is ``True``)
    """
    def __init__(
        self,
        dir_networkFiles: str,
        device: str = 'cpu',
        download_method: str = 'check_local_first',
        download_url: str = 'https://osf.io/x3fd2/download',
        download_hash: dict = None,
        names_networkFiles: dict = None,
        forward_pass_version: str = 'latent',
        verbose: bool = True,
    ):
        ## Imports
        super().__init__()

        self._device = device
        self._verbose = verbose


        self._dir_networkFiles = dir_networkFiles
        self._download_url = download_url

        self._download_path_save = str(Path(self._dir_networkFiles).resolve() / 'ROInet.zip')

        fn_download = partial(
            helpers.download_file,
            path_save=self._download_path_save,
            hash_type='MD5',
            hash_hex=download_hash,
            mkdir=True,
            allow_overwrite=True,
            write_mode='wb',
            verbose=self._verbose,
            chunk_size=1024,
        )

        ## Find or download network files
        if download_method == 'force_download':
            fn_download(url=self._download_url, check_local_first=False, check_hash=False)

        if download_method == 'check_local_first':
            # assert download_hash is not None, "if using download_method='check_local_first' download_hash cannot be None. Either determine the hash of the zip file or use download_method='force_download'."
            fn_download(url=self._download_url, check_local_first=True, check_hash=True)

        if download_method == 'force_local':
            # assert download_hash is not None, "if using download_method='force_local' download_hash cannot be None"
            assert Path(self._download_path_save).exists(), f"if using download_method='force_local' the network files must exist in {self._download_path_save}"
            fn_download(url=None, check_local_first=True, check_hash=True)

        ## Extract network files from zip
        paths_extracted = helpers.extract_zip(
            path_zip=self._download_path_save,
            path_extract=self._dir_networkFiles,
            verbose=self._verbose,
        )

        ## Find network files
        if names_networkFiles is None:
            names_networkFiles = {
                'params': 'params.json',
                'model': 'model.py',
                'state_dict': '.pth',
            }
        paths_networkFiles = {}
        paths_networkFiles['params'] = [p for p in paths_extracted if names_networkFiles['params'] in str(Path(p).name)][0]
        paths_networkFiles['model'] = [p for p in paths_extracted if names_networkFiles['model'] in str(Path(p).name)][0]
        paths_networkFiles['state_dict'] = [p for p in paths_extracted if names_networkFiles['state_dict'] in str(Path(p).name)][0]

        ## Import network files
        sys.path.append(str(Path(paths_networkFiles['model']).parent.resolve()))
        import model
        print(f"Imported model from {paths_networkFiles['model']}") if self._verbose else None

        with open(paths_networkFiles['params']) as f:
            self.params_model = json.load(f)
            print(f"Loaded params_model from {paths_networkFiles['params']}") if self._verbose else None
            self.net = model.make_model(fwd_version=forward_pass_version, **self.params_model)
            print(f"Generated network using params_model") if self._verbose else None

        ## Prep network and load state_dict
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.eval()

        self.net.load_state_dict(torch.load(paths_networkFiles['state_dict'], map_location=torch.device(self._device)))
        print(f'Loaded state_dict into network from {paths_networkFiles["state_dict"]}') if self._verbose else None

        self.net = self.net.to(self._device)
        print(f'Loaded network onto device {self._device}') if self._verbose else None

    def generate_dataloader(
        self,
        ROI_images: List[np.ndarray],
        um_per_pixel: float = 1.0,
        nan_to_num: bool = True,
        nan_to_num_val: float = 0.0,
        pref_plot: bool = False,
        batchSize_dataloader: int = 8,
        pinMemory_dataloader: bool = True,
        numWorkers_dataloader: int = -1,
        persistentWorkers_dataloader: bool = True,
        prefetchFactor_dataloader: int = 2,
        transforms: Optional[Callable] = None,
        img_size_out: Tuple[int, int] = (224, 224),
        jit_script_transforms: bool = False,
    ):
        """
        Generates a PyTorch DataLoader for a list of Region of Interest (ROI)
        images. Performs preprocessing such as rescaling, normalization, and
        resizing.

        Args:
            ROI_images (List[np.ndarray]): 
                The ROI images to use for the dataloader. List of arrays, each
                array corresponds to a session and is of shape *(n_rois, height,
                width)*.
            um_per_pixel (float): 
                The number of microns per pixel. Used to rescale the ROI images
                to the same size as the network input. (Default is *1.0*)
            nan_to_num (bool): 
                Whether to replace NaNs with a specific value. (Default is
                ``True``)
            nan_to_num_val (float): 
                The value to replace NaNs with. (Default is *0.0*)
            pref_plot (bool): 
                If ``True``, plots the sizes of the ROI images before and after
                normalization. (Default is ``False``)
            batchSize_dataloader (int): 
                The batch size to use for the DataLoader. (Default is *8*)
            pinMemory_dataloader (bool): 
                If ``True``, pins the memory of the DataLoader, as per PyTorch's
                best practices. (Default is ``True``)
            numWorkers_dataloader (int): 
                The number of worker processes for data loading. (Default is
                *-1*)
            persistentWorkers_dataloader (bool): 
                If ``True``, uses persistent worker processes. (Default is
                ``True``)
            prefetchFactor_dataloader (int): 
                The prefetch factor for data loading. (Default is *2*)
            transforms (Optional[Callable]): 
                The transforms to use for the DataLoader. If ``None``, the
                function will only scale dynamic range (to 0-1), resize (to
                img_size_out dimensions), and tile channels (to 3) as a minimum
                to pass images through the network. (Default is ``None``)
            img_size_out (Tuple[int, int]): 
                The image output dimensions of DataLoader if transforms is
                ``None``. (Default is *(224, 224)*)
            jit_script_transforms (bool): 
                If ``True``, converts the transforms pipeline into a TorchScript
                pipeline, potentially improving calculation speed but can cause
                problems with multiprocessing. (Default is ``False``)

        Returns:
            (np.ndarray): 
                ROI_images (np.ndarray): 
                    The ROI images after normalization and resizing. Shape is
                    *(n_sessions, n_rois, n_channels, height, width)*.

        Example:
            .. highlight:: python
            .. code-block:: python

                dataloader = generate_dataloader(ROI_images)
        """
        ## Remove NaNs
        ### Check if any NaNs
        if np.any([np.any(np.isnan(roi)) for roi in ROI_images]):
            warnings.warn('ROICaT WARNING: NaNs detected. You should consider removing remove these before passing to the network. Using nan_to_num arguments.')
        if np.any([np.any(np.isinf(roi)) for roi in ROI_images]):
            warnings.warn('ROICaT WARNING: Infs detected. You should consider removing these before passing to the network.')
        ## Check if any images in any of the sessions are all zeros
        if np.any([np.any(np.all(rois==0, axis=(1,2))) for rois in ROI_images]):
            warnings.warn('ROICaT WARNING: Image(s) with all zeros detected. These can pass through the network, but may give weird results.')
        if nan_to_num:
            ROI_images = [np.nan_to_num(rois, nan=nan_to_num_val) for rois in ROI_images]

        if numWorkers_dataloader == -1:
            numWorkers_dataloader = mp.cpu_count()

        print('Starting: resizing ROIs') if self._verbose else None
        
        sf_rs = [self.resize_ROIs(rois, um_per_pixel) for rois in ROI_images]
        
        ROI_images_cat = np.concatenate(ROI_images, axis=0)
        ROI_images_rs = np.concatenate(sf_rs, axis=0)

        print('Completed: resizing ROIs') if self._verbose else None

        if pref_plot:
            fig, axs = plt.subplots(2,1, figsize=(7,10))
            axs[0].plot(np.mean(ROI_images_cat > 0, axis=(1,2)))
            axs[0].plot(scipy.signal.savgol_filter(np.mean(ROI_images_cat > 0, axis=(1,2)), 501, 3))
            axs[0].set_xlabel('ROI number');
            axs[0].set_ylabel('mean npix');
            axs[0].set_title('ROI sizes raw')

            axs[1].plot(np.mean(ROI_images_rs > 0, axis=(1,2)))
            axs[1].plot(scipy.signal.savgol_filter(np.mean(ROI_images_rs > 0, axis=(1,2)), 501, 3))
            axs[1].set_xlabel('ROI number');
            axs[1].set_ylabel('mean npix');
            axs[1].set_title('ROI sizes resized')

        if transforms is None:
            transforms = torch.nn.Sequential(
                ScaleDynamicRange(scaler_bounds=(0,1)),
                torchvision.transforms.Resize(
                    size=img_size_out,
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                TileChannels(dim=0, n_channels=3),
            )

        if jit_script_transforms:
            if numWorkers_dataloader > 0:
                warnings.warn("\n\nWarning: Converting transforms to a jit-based script has been known to cause issues on Windows when numWorkers_dataloader > 0. If self.generate_latents() raises an Exception similar to 'Tried to serialize object __torch__.torch.nn.modules.container.Sequential which does not have a __getstate__ method defined!' consider setting numWorkers_dataloader=0 or jit_script_transforms=False.\n")

            self.transforms = torch.jit.script(transforms)
        else:
            self.transforms = transforms
        
        print(f'Defined image transformations: {transforms}') if self._verbose else None

        self.dataset = dataset_simCLR(
                X=torch.as_tensor(ROI_images_rs, device='cpu', dtype=torch.float32),
                y=torch.as_tensor(torch.zeros(ROI_images_rs.shape[0]), device='cpu', dtype=torch.float32),
                n_transforms=1,
                transform=self.transforms,
                DEVICE='cpu',
                dtype_X=torch.float32,
            )
        print(f'Defined dataset') if self._verbose else None

        self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batchSize_dataloader,
                shuffle=False,
                drop_last=False,
                pin_memory=pinMemory_dataloader,
                num_workers=numWorkers_dataloader,
                persistent_workers=persistentWorkers_dataloader,
                prefetch_factor=prefetchFactor_dataloader,
        )

        print(f'Defined dataloader') if self._verbose else None

        self.ROI_images_rs = ROI_images_rs
        return ROI_images_rs

    @classmethod
    def resize_ROIs(
        cls,
        ROI_images: np.ndarray,  # Array of shape (n_rois, height, width)
        um_per_pixel: float,
    ) -> np.ndarray:
        """
        Resizes the ROI (Region of Interest) images to prepare them for pass
        through network.

        Args:
            ROI_images (np.ndarray): 
                The ROI images to resize. Array of shape *(n_rois, height,
                width)*.
            um_per_pixel (float): 
                The number of microns per pixel. This value is used to rescale
                the ROI images so that they occupy a standard region of the
                image frame.

        Returns:
            (np.ndarray): 
                ROI_images_rs (np.ndarray): 
                    The resized ROI images.
        """        
        scale_forRS = 0.7 * um_per_pixel  ## hardcoded for now sorry
        return np.stack([resize_affine(img, scale=scale_forRS, clamp_range=True) for img in ROI_images], axis=0)


    def generate_latents(self) -> torch.Tensor:
        """
        Passes the data in the dataloader through the network and generates latents.

        Returns:
            (torch.Tensor): 
                latents (torch.Tensor): 
                    Latents for each ROI (Region of Interest).
        """
        if hasattr(self, 'dataloader') == False:
            raise Exception('dataloader not defined. Call generate_dataloader() first.')

        print(f'starting: running data through network')
        self.latents = torch.cat([self.net(data[0][0].to(self._device)).detach() for data in tqdm(self.dataloader, mininterval=5)], dim=0).cpu()
        print(f'completed: running data through network')

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        
        return self.latents


###################################
########### RESIZING ##############
###################################


def resize_affine(
    img: np.ndarray, 
    scale: float, 
    clamp_range: bool = False,
) -> np.ndarray:
    """
    Resizes an image using an affine transformation, scaled by a factor.

    Args:
        img (np.ndarray): 
            The input image to resize. Shape: *(H, W)*
        scale (float): 
            The scale factor to apply for resizing.
        clamp_range (bool): 
            If ``True``, the image will be clamped to the range [min(img),
            max(img)] to prevent interpolation from extending outside of the
            image's range. (Default is ``False``)

    Returns:
        (np.ndarray): 
            resized_image (np.ndarray): 
                The resized image.
    """
    img_rs = np.array(torchvision.transforms.functional.affine(
        img=PIL.Image.fromarray(img),
        angle=0, translate=[0,0], shear=0,
        scale=scale,
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC
    ))

    if clamp_range:
        clamp_high = img.max()
        clamp_low = img.min()

        img_rs[img_rs>clamp_high] = clamp_high
        img_rs[img_rs<clamp_low] = clamp_low

    return img_rs

def resize_images(
    imgs: np.ndarray,
    scale: float,
    clamp_range: bool = False,
) -> np.ndarray:
    """
    Resizes images using an affine transformation, scaled by a factor.
    Uses torch.nn.functional.grid_sample to perform the resizing.
    
    Args:
        imgs (np.ndarray): 
            The input images to resize. Shape: *(N, H, W)*
        scale (float): 
            The scale factor to apply for resizing.
        clamp_range (bool): 
            If ``True``, the image will be clamped to the range [min(img),
            max(img)] to prevent interpolation from extending outside of the
            image's range. (Default is ``False``)

    Returns:
        (np.ndarray): 
            resized_images (np.ndarray): 
                The resized images. Shape: *(N, H, W)*
    """
    imgs = imgs[None, ...] if imgs.ndim == 2 else imgs
    imgs_rs = img_size = imgs.shape[1:]

    meshgrid_out = torch.stack(torch.meshgrid(torch.linspace(-1, 1, img_size[0]), torch.linspace(-1, 1, img_size[1]), indexing='xy'), dim=-1)

    imgs_rs = torch.nn.functional.grid_sample(
        input=torch.as_tensor(imgs)[None, ...],
        grid=meshgrid_out[None, ...] / scale,
        mode='bicubic',
        padding_mode='zeros',
        align_corners=True,
    )[0].numpy()

    if clamp_range:
        imgs_rs = np.clip(imgs_rs, a_min=imgs.min(axis=(1,2), keepdims=True), a_max=imgs.max(axis=(1,2), keepdims=True))

    return imgs_rs

def resize_affine2(
    imgs: np.ndarray, 
    scale: float, 
    clamp_range: bool = False,
) -> np.ndarray:
    """
    Resizes an image using an affine transformation, scaled by a factor.

    Args:
        img (np.ndarray): 
            The input images to resize. Shape: *(N, H, W)*
        scale (float): 
            The scale factor to apply for resizing.
        clamp_range (bool): 
            If ``True``, the image will be clamped to the range [min(img),
            max(img)] to prevent interpolation from extending outside of the
            image's range. (Default is ``False``)

    Returns:
        (np.ndarray): 
            resized_image (np.ndarray): 
                The resized image.
    """
    img_rs = np.array(torchvision.transforms.functional.affine(
        img=PIL.Image.fromarray(imgs.transpose(1,2,0)),
        angle=0, translate=[0,0], shear=0,
        scale=scale,
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC
    )).transpose(2,0,1)

    if clamp_range:
        imgs_rs = np.clip(imgs_rs, a_min=imgs.min(axis=(1,2), keepdims=True), a_max=imgs.max(axis=(1,2), keepdims=True))

    return img_rs


###################################
########### FROM GRC ##############
###################################

class TileChannels(Module):
    """
    Expand dimension dim in X_in and tile to be N channels.
    RH 2021
    """
    def __init__(self, dim=0, n_channels=3):
        """
        Initializes the class.
        Args:
            dim (int):
                The dimension to tile.
            n_channels (int):
                The number of channels to tile to.
        """
        super().__init__()
        self.dim = dim
        self.n_channels = n_channels

    def forward(self, tensor):
        dims = [1]*len(tensor.shape)
        dims[self.dim] = self.n_channels
        return torch.tile(tensor, dims)
    def __repr__(self):
        return f"TileChannels(dim={self.dim})"

class Unsqueeze(Module):
    """
    Expand dimension dim in X_in and tile to be N channels.
    JZ 2023
    """
    def __init__(self, dim=0):
        """
        Initializes the class.
        Args:
            dim (int):
                The dimension to tile.
            n_channels (int):
                The number of channels to tile to.
        """
        super().__init__()
        self.dim = dim

    def forward(self, tensor):
        return torch.unsqueeze(tensor, self.dim)
    def __repr__(self):
        return f"Unsqueeze(dim={self.dim})"

class ScaleDynamicRange(Module):
    """
    Min-max scaling of the input tensor, independently per image.
    RH 2021

    Reduces over the trailing three dimensions *(n_channels, height, width)*, so
    the same module gives identical results whether it is applied to a single
    image *(n_channels, height, width)* or to a batch *(n_images, n_channels,
    height, width)*. **This is what allows one transform chain to serve both the
    per-sample DataLoader path and the batched
    ``Preprocessor_ROI_images.transform_images`` path.**
    """
    def __init__(self, scaler_bounds=(0,1), epsilon=1e-9):
        """
        Initializes the class.
        Args:
            scaler_bounds (tuple):
                The bounds to scale the dynamic range of each image to.
             epsilon (float):
                 Value to add to the denominator when normalizing.
        """
        super().__init__()

        self.bounds = scaler_bounds
        self.range = scaler_bounds[1] - scaler_bounds[0]

        self.epsilon = epsilon
        ## List (not tuple) of dims for torch.jit.script compatibility
        self.dims_reduce = [-3, -2, -1]

    def forward(self, tensor):
        ## Reduce over (n_channels, height, width); leading dims (if any) are batch dims
        tensor_minSub = tensor - torch.amin(tensor, dim=self.dims_reduce, keepdim=True)
        return tensor_minSub * (self.range / (torch.amax(tensor_minSub, dim=self.dims_reduce, keepdim=True) + self.epsilon))
    def __repr__(self):
        return f"ScaleDynamicRange(scaler_bounds={self.bounds})"


class dataset_simCLR(Dataset):
    """
    Args:
        X (Union[torch.Tensor, np.array, List[float]]): 
            Images. Expected shape: *(n_samples, height, width)*. Currently
            expects no channel dimension. If/when it exists, then shape should
            be *(n_samples, n_channels, height, width)*.
        y (Union[torch.Tensor, np.array, List[int]]): 
            Labels. Shape: *(n_samples)*.
        n_transforms (int): 
            Number of transformations to apply to each image. Should be >= 1.
            (Default is ``2``)
        transform (Optional[Callable]): 
            Optional transform to be applied on a sample. See
            torchvision.transforms for more information. Can use
            torch.nn.Sequential(a, bunch, of, transforms,) or other methods
            from torchvision.transforms. \n
            * If not ``None``: Transform(s) are applied to each image and the
              output shape of X_sample_transformed for __getitem__ will be
              *(n_samples, n_transforms, n_channels, height, width)*.
            * If ``None``: No transform is applied and output shape of
              X_sample_trasformed for __getitem__ will be *(n_samples,
              n_channels, height, width)* (which is missing the n_transforms
              dimension). \n
            (Default is ``None``)
        DEVICE (str): 
            Device on which the data will be stored and transformed. Best to
            leave this as 'cpu' and do .to(DEVICE) on the data for the training
            loop. (Default is ``'cpu'``)
        dtype_X (torch.dtype): 
            Data type of X. (Default is ``torch.float32``)
        dtype_y (torch.dtype): 
            Data type of y. (Default is ``torch.int64``)
        temp_uncetainty (float):
            Temperture term applied to the CrossEntropyLoss input. (Default is
            ``1.0`` for no change)

    Example:
        .. highlight:: python
        .. code-block:: python

            transforms = torch.nn.Sequential(
                torchvision.transforms.RandomHorizontalFlip(p=0.5),

                torchvision.transforms.GaussianBlur(
                    5,
                    sigma=(0.01, 1.)
                ),
                torchvision.transforms.RandomPerspective(
                    distortion_scale=0.6,
                    p=1,
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    fill=0
                ),
                torchvision.transforms.RandomAffine(
                    degrees=(-180,180),
                    translate=(0.4, 0.4),
                    scale=(0.7, 1.7),
                    shear=(-20, 20, -20, 20),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                    fill=0,
                    fillcolor=None,
                    resample=None
                ),
            )
            scripted_transforms = torch.jit.script(transforms)

            dataset = dataset_simCLR(  torch.tensor(images),
                                        labels,
                                        n_transforms=2,
                                        transform=scripted_transforms,
                                        DEVICE='cpu',
                                        dtype_X=torch.float32,
                                        dtype_y=torch.int64)

            dataloader = torch.utils.data.DataLoader(   dataset,
                                                    batch_size=64,
                                                    shuffle=True,
                                                    drop_last=True,
                                                    pin_memory=False,
                                                    num_workers=0)
    """
    def __init__(   
        self,
        X: Union[torch.Tensor, np.array, List[float]],
        y: Union[torch.Tensor, np.array, List[int]],
        n_transforms: int = 2,
        transform: Optional[Callable] = None,
        DEVICE: str = 'cpu',
        dtype_X: torch.dtype = torch.float32,
        dtype_y: torch.dtype = torch.int64,
    ):
        """
        Initializes the dataset_simCLR object with the given images, labels, and
        optional settings.
        """

        self.X = torch.as_tensor(X, dtype=dtype_X, device=DEVICE) # first dim will be subsampled from. Shape: (n_samples, n_channels, height, width)
        self.X = self.X[:,None,...]
        self.y = torch.as_tensor(y, dtype=dtype_y, device=DEVICE) # first dim will be subsampled from.

        self.idx = torch.arange(self.X.shape[0], device=DEVICE)
        self.n_samples = self.X.shape[0]

        self.transform = transform
        self.n_transforms = n_transforms

        if X.shape[0] != y.shape[0]:
            raise ValueError('RH Error: X and y must have same first dimension shape')

    def tile_channels(
        self,
        X_in: Union[torch.Tensor, np.ndarray],
        dim: int = -3,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Expand dimension dim in X_in and tile to be 3 channels.

        Args:
            X_in (torch.Tensor or np.ndarray): 
                Input image with shape: *(n_channels==1, height, width)*
            dim (int): 
                Dimension to expand. (Default is ``-3``)

        Returns:
            (torch.Tensor or np.ndarray): 
                X_out (torch.Tensor or np.ndarray):
                    Output image with shape: *(n_channels==3, height, width)*
        """
        dims = [1]*len(X_in.shape)
        dims[dim] = 3
        return torch.tile(X_in, dims)
    
    def __len__(self):
        """
        Get the total number of samples in the dataset.

        Returns:
            (int): 
                n_samples (int): 
                    The total number of samples.
        """
        return self.n_samples
    
    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[Union[torch.Tensor, np.ndarray], int, int, int]:
        """
        Retrieves and transforms a sample.

        Args:
            idx (int): 
                Index of the sample to retrieve.

        Returns:
            (Tuple): tuple containing:
                X_sample_transformed (torch.Tensor or np.ndarray):
                    Transformed sample(s). Shape: 
                        * If transform is ``None``: *(batch_size, n_channels, height, width)*
                        * If transform is not ``None``: *(n_transforms, batch_size, n_channels, height, width)*
                y_sample (int):
                    Label of the sample.
                idx_sample (int):
                    Index of the sample.
                sample_weight (int):
                    Weight of the sample. Always 1.
        """
        y_sample = self.y[idx]
        idx_sample = self.idx[idx]

        sample_weight = 1

        X_sample_transformed = []
        if self.transform is not None:
            for ii in range(self.n_transforms):
                X_transformed = self.transform(self.X[idx_sample])
                X_sample_transformed.append(X_transformed)
        else:
            X_sample_transformed = self.tile_channels(self.X[idx_sample], dim=-3)

        return X_sample_transformed, y_sample, idx_sample, sample_weight
    