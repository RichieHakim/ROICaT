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
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import scipy.signal
import warnings

from . import util, helpers

class Resizer_ROI_images(util.ROICaT_Module):
    """
    Class for resizing ROIs.
    JZ, RH 2023

    Args:
        ROI_images (np.ndarray): 
            Array of ROIs to resize. Shape should be (nROIs, height,
            width).
        um_per_pixel (float): 
            Size of a pixel in microns.
        nan_to_num (bool): 
            Whether to replace NaNs with a specific value. (Default is
            ``True``)
        nan_to_num_val (float): 
            The value to replace NaNs with. (Default is *0.0*)
        verbose (bool): 
            If True, print out extra information. (Default is ``False``)
    """
    def __init__(self, ROI_images: np.ndarray, um_per_pixel: float, nan_to_num: bool=True, nan_to_num_val: float=0.0, verbose: bool=True):
        self._verbose = verbose

        ### Check if any NaNs
        if np.any(np.isnan(ROI_images)):
            if nan_to_num:
                warnings.warn('ROICaT WARNING: NaNs detected. You should consider removing these before passing to the network. Using nan_to_num arguments.')
            else:
                raise ValueError('ROICaT ERROR: NaNs detected. You should consider removing these before passing to the network. Use nan_to_num=True to ignore this error.')
        if np.any(np.isinf(ROI_images)):
            warnings.warn('ROICaT WARNING: Infs detected. You should consider removing these before passing to the network.')
        ## Check if any images in any of the sessions are all zeros
        if np.any(np.all(ROI_images==0, axis=(1,2))):
            warnings.warn('ROICaT WARNING: Image(s) with all zeros detected. These can pass through the network, but may give weird results.')
        
        if nan_to_num:
            ROI_images = np.nan_to_num(ROI_images, nan=nan_to_num_val)

        print('Starting: resizing ROIs') if self._verbose else None
        self.ROI_images_rs = self.resize_ROIs(ROI_images, um_per_pixel)
        print('Completed: resizing ROIs') if self._verbose else None

    def plot_resized_comparison(self, ROI_images_cat: np.ndarray):
        """
        Plot a comparison of the ROI sizes before and after resizing.

        Args:
            ROI_images_cat (np.ndarray):
                Array of ROIs to resize. Shape should be (nROIs, height,
                width).
        """
        fig, axs = plt.subplots(2,1, figsize=(7,10))
        axs[0].plot(np.mean(ROI_images_cat > 0, axis=(1,2)))
        axs[0].plot(scipy.signal.savgol_filter(np.mean(ROI_images_cat > 0, axis=(1,2)), 501, 3))
        axs[0].set_xlabel('ROI number');
        axs[0].set_ylabel('mean npix');
        axs[0].set_title('ROI sizes raw')

        axs[1].plot(np.mean(self.ROI_images_rs > 0, axis=(1,2)))
        axs[1].plot(scipy.signal.savgol_filter(np.mean(self.ROI_images_rs > 0, axis=(1,2)), 501, 3))
        axs[1].set_xlabel('ROI number');
        axs[1].set_ylabel('mean npix');
        axs[1].set_title('ROI sizes resized')

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
        scale_forRS = 1.2 * um_per_pixel * (ROI_images.shape[1] / 36)  ## hardcoded for now sorry
        return np.stack([resize_affine(img, scale=scale_forRS, clamp_range=True) for img in ROI_images], axis=0)


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
        img_size_out (Tuple[int, int]): 
            The image output dimensions of DataLoader if transforms is
            ``None``. (Default is *(224, 224)*)
        jit_script_transforms (bool): 
            If ``True``, converts the transforms pipeline into a TorchScript
            pipeline, potentially improving calculation speed but can cause
            problems with multiprocessing. (Default is ``False``)
        verbose (bool):
            If ``True``, print out extra information. (Default is ``True``)
    """
    def __init__(
            self,
            ROI_images: np.ndarray,
            batchSize_dataloader: int = 8,
            pinMemory_dataloader: bool = True,
            numWorkers_dataloader: int = -1,
            persistentWorkers_dataloader: bool = True,
            prefetchFactor_dataloader: int = 2,
            transforms: Optional[Callable] = None,
            img_size_out: Tuple[int, int] = (224, 224),
            jit_script_transforms: bool = False,
            verbose: bool = True,
        ):
        self._verbose = verbose
        numWorkers_dataloader = mp.cpu_count() if numWorkers_dataloader == -1 else numWorkers_dataloader

        transforms = torch.nn.Sequential(
            ScaleDynamicRange(scaler_bounds=(0,1)),
            torchvision.transforms.Resize(
                size=img_size_out,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            TileChannels(dim=0, n_channels=3),
        ) if transforms is None else transforms

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
        ROI_images = np.concatenate(ROI_images, axis=0)
        roi_resizer = Resizer_ROI_images(ROI_images,
                                         um_per_pixel,
                                         nan_to_num,
                                         nan_to_num_val,
                                         verbose=self._verbose)
        roi_resizer.plot_resized_comparison(ROI_images) if pref_plot else None
        self.ROI_images_rs = roi_resizer.ROI_images_rs

        dataloader_generator = Dataloader_ROInet(
            self.ROI_images_rs,
            batchSize_dataloader,
            pinMemory_dataloader,
            numWorkers_dataloader,
            persistentWorkers_dataloader,
            prefetchFactor_dataloader,
            transforms,
            img_size_out,
            jit_script_transforms,
            self._verbose,
        )

        self.transforms = dataloader_generator.transforms
        self.dataset = dataloader_generator.dataset
        self.dataloader = dataloader_generator.dataloader
        self.ROI_images_rs = roi_resizer.ROI_images_rs
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
    Min-max scaling of the input tensor.
    RH 2021
    """
    def __init__(self, scaler_bounds=(0,1), epsilon=1e-9):
        """
        Initializes the class.
        Args:
            scaler_bounds (tuple):
                The bounds of how much to multiply the image by
                 prior to adding the Poisson noise.
             epsilon (float):
                 Value to add to the denominator when normalizing.
        """
        super().__init__()

        self.bounds = scaler_bounds
        self.range = scaler_bounds[1] - scaler_bounds[0]

        self.epsilon = epsilon

    def forward(self, tensor):
        tensor_minSub = tensor - tensor.min()
        return tensor_minSub * (self.range / (tensor_minSub.max()+self.epsilon))
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
    