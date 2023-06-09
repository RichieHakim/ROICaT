"""
OSF.io links to ROInet versions:

ROInet_tracking:
    Info:
        This version does not includde occlusions or large
        affine transformations.
    Link:
        https://osf.io/x3fd2/download
    Hash (MD5 hex):
        7a5fb8ad94b110037785a46b9463ea94

ROInet_classification:
    Info:
        This version includes occlusions and large affine
        transformations.
    Link:
        https://osf.io/c8m3b/download
    Hash (MD5 hex):
        357a8d9b630ec79f3e015d0056a4c2d5
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

import numpy as np
# import gdown
import torch
import torchvision
from torch.nn import Module
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import scipy.signal
import warnings

from . import util

class ROInet_embedder(util.ROICaT_Module):
    """
    Class for loading the ROInet model, preparing data for it,
     and running it.

    OSF.io links to ROInet versions:

    ROInet_tracking:
        Info:
            This version does not includde occlusions or large
             affine transformations.
        Link:
            https://osf.io/x3fd2/download
        Hash (MD5 hex):
            7a5fb8ad94b110037785a46b9463ea94

    ROInet_classification:
        Info:
            This version includes occlusions and large affine
             transformations.
        Link:
            https://osf.io/c8m3b/download
        Hash (MD5 hex):
            357a8d9b630ec79f3e015d0056a4c2d5


    RH, JZ 2022

    Initialization of the class.
    This will look for a local copy of the network files, and
        if they don't exist, it will download them from Google Drive
        using a user specified download_url.
    There is some hash checking to make sure the files are the same.

    Args:
        dir_networkFiles (str):
            The directory to find an existing ROInet.zip file
                or download and extract a new one into.
        device (str):
            The device to use for the model and data.
        download_method (str):
            Approach to downloading the network files.
            'check_local_first':
                Check if the network files are already in
                    dir_networkFiles. If so, use them.
            'force_download':
                Download an ROInet.zip file from download_url.
            'force_local':
                Use an existing local copy of an ROInet.zip file.
                If they don't exist, raise an error.
                Hash checking is done and download_hash must be
                    specified.
        download_url (str):
            The url to download the ROInet.zip file from.
        download_hash (dict):
            MD5 hash of the ROInet.zip file. This can be obtained
                from ROICaT documentation. If you don't have one,
                use download_method='force_download' and determine
                the hash using helpers.hash_file().
        names_networkFiles (dict):
            Optional. The names of the files in the ROInet.zip
                file.
            If uncertain, leave as None.
            Should be of form (example):
            {
                'params': 'params.json',
                'model': 'model.py',
                'state_dict': 'ConvNext_tiny__1_0_unfrozen__simCLR.pth'
            }
            'params':
                The parameters used to train the network.
                    Usually a .json file.
            'model':
                The model definition. Usually a .py file.
            'state_dict':
                The weights of the network. Usually a .pth
                    file.
        forward_pass_version (str):
            The version of the forward pass to use.
            'latent':
                Return the post-head output latents.
                Use this for tracking.
            'head':
                Return the output of the head layers.
                Use this for classification.
            'base':
                Return the output of the base model.
        verbose (bool):
            Whether to print out extra information.
    """
    def __init__(
        self,
        dir_networkFiles,
        device='cpu',
        download_method='check_local_first',
        download_url='https://osf.io/x3fd2/download',
        download_hash=None,
        names_networkFiles=None,
        forward_pass_version='latent',
        verbose=True,
    ):
        ## Imports
        super().__init__()

        self._device = device
        self._verbose = verbose


        self._dir_networkFiles = dir_networkFiles
        self._download_url = download_url

        self._download_path_save = str(Path(self._dir_networkFiles).resolve() / 'ROInet.zip')

        fn_download = partial(
            download_file,
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
        paths_extracted = extract_zip(
            path_zip=self._download_path_save,
            path_extract=self._dir_networkFiles,
            verbose=self._verbose,
        )

        print(paths_extracted)

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

        ## Prepare dataloader

    def generate_dataloader(
        self,
        ROI_images,
        # goal_frac=0.1929,
        um_per_pixel=1.0,
        nan_to_num=True,
        nan_to_num_val=0.0,
        pref_plot=False,
        batchSize_dataloader=8,
        pinMemory_dataloader=True,
        numWorkers_dataloader=-1,
        persistentWorkers_dataloader=True,
        prefetchFactor_dataloader=2,
        transforms=None,
        img_size_out=(224, 224),
        jit_script_transforms=False,
    ):
        """
        Generate a dataloader for the given ROI_images.
        This will be used to pass the data through the network

        Args:
            ROI_images (list of np.ndarray of images):
                The ROI images to use for the dataloader.
                List of arrays where each array is from a session,
                 and each array is of shape (n_rois, height, width)
                This can be derived using the data_importing module.
            um_per_pixel (float):
                The number of microns per pixel. Used to rescale the
                 ROI images to the same size as the network input.
            nan_to_num (bool):
                Whether to replace NaNs with a value.
            nan_to_num_val (float):
                The value to replace NaNs with.
            pref_plot (bool):
                Whether to plot the sizes of the ROI images before
                 and after normalization.
            batchSize_dataloader (int):
                The batch size to use for the dataloader.
            pinMemory_dataloader (bool):
                Whether to pin the memory of the dataloader.
                See pytorch documentation on dataloaders.
            numWorkers_dataloader (int):
                The number of workers to use for the dataloader.
                See pytorch documentation on dataloaders.
            persistentWorkers_dataloader (bool):
                Whether to use persistent workers for the dataloader.
                See pytorch documentation on dataloaders.
            prefetchFactor_dataloader (int):
                The prefetch factor to use for the dataloader.
                See pytorch documentation on dataloaders.
            transforms (torchvision.transforms):
                (Optional) The transforms to use for the dataloader.
                If None, will only scale dynamic range (to 0-1),
                 resize (to img_size_out dimensions), and tile channels (to 3).
                 These are the minimum transforms required to pass
                 images through the network.
            img_size_out (tuple of ints):
                (Optional) Image output dimensions of dataloader if transforms is None.
            jit_script_transforms (bool):
                (Optional) Whether or not to convert the transforms pipeline into a
                TorchScript pipeline. (Should improve calculation speed but can cause
                problems with multiprocessing.)
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
            ROI_images =  [np.nan_to_num(rois, nan=nan_to_num_val) for rois in ROI_images]

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
                class_weights=np.array([1]),
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
    def resize_ROIs(self, ROI_images, um_per_pixel):
        """
        Resize the ROI images to prepare for pass through network.
        RH 2022

        Args:
            ROI_images (np.ndarray):
                The ROI images to resize.
                Array of shape (n_rois, height, width)
            um_per_pixel (float):
                The number of microns per pixel. Used to rescale the
                 ROI images so that they occupy a standard region of
                 the image frame.

        Returns:
            ROI_images_rs (np.ndarray):
                The resized ROI images.
        """
        scale_forRS = 0.7 * um_per_pixel  ## hardcoded for now sorry
        return np.stack([resize_affine(img, scale=scale_forRS, clamp_range=True) for img in ROI_images], axis=0)


    def generate_latents(self):
        """
        Pass the data in the dataloader (see self.generate_dataloader)
         through the network and generate latents.

        Returns:
            latents (torch.Tensor):
                latents for each ROI
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
    
    def dump_latents(self, latent_folder_out, file_prefix='latent_dump', num_copies=1, start_copy_num=0):
        """
        Run data from the dataloader (see self.generate_dataloader)
         through the network and dump resulting latents to file.
        """
        print(f'starting: dumping latents')
        
        for icopy in trange(start_copy_num, start_copy_num+num_copies):
            augmented_lst = []
            for data in tqdm(self.dataloader, mininterval=5):
                aug = self.net(data[0][0].to(self._device)).detach().cpu()
                augmented_lst.append(aug)
            
            augmented_latents = (torch.cat(augmented_lst, dim=0)).detach()
            dump = np.save(Path(latent_folder_out) / f'{file_prefix}-{icopy}.npy', augmented_latents.numpy())

    # def _download_network_files(self):
    #     if self._download_url is None or self._dir_networkFiles is None:
    #         raise ValueError('download_url and dir_networkFiles must be specified if download_method is True')

    #     print(f'Downloading network files from Google Drive to {self._dir_networkFiles}') if self._verbose else None
    #     download_file(
    #         url=self._download_url,
    #         path_save=self._dir_networkFiles,
    #         check_local_first=True,
    #         check_hash=True,
    #         hash_type='MD5',
    #         hash_hex='1e62893d8e944819516e793656afc31d',
    #         mkdir=True,
    #         allow_overwrite=True,
    #         write_mode='wb',
    #         verbose=True,
    #         chunk_size=1024,
    #     )
    #     paths_files = gdown.download_folder(id=self._download_url, output=self._dir_networkFiles, quiet=False, use_cookies=False)
    #     print('Downloaded network files') if self._verbose else None
    #     return paths_files

    def show_rescaled_rois(self, rows=10, cols=10, figsize=(7,7)):
        """
        Show post-rescaling ROIs for um_per_pixel sanity checking.

        JZ 2022

        Note: rows & cols both must be > 1.
        """
        fig, ax = plt.subplots(rows,cols,figsize=figsize)
        for ir,roi in enumerate(self.ROI_images_rs[:(rows*cols)]):
            ax[ir//cols,ir%cols].imshow(roi)
            ax[ir//cols,ir%cols].axis('off')

    def show_augmentations(self, rows=10, cols=10, figsize=(7,7)):
        """
        Pass the data through the dataloader (see self.generate_dataloader)
         to show augmented ROIs for sanity checking.
        """
        print(f'starting: running data through network')
        augmented_lst = []
        for data in (self.dataloader):
            aug = data[0][0].detach()
            augmented_lst.append(aug)


        augmented = torch.cat(augmented_lst, dim=0).cpu()
        print(f'completed: running data through network')

        fig, ax = plt.subplots(rows,cols,figsize=figsize)
        for ir,roi in enumerate(augmented[:(rows*cols)]):
            ax[ir//cols,ir%cols].imshow(roi[0])
            ax[ir//cols,ir%cols].axis('off')


def resize_affine(img, scale, clamp_range=False):
    """
    Wrapper for torchvision.transforms.Resize.
    Useful for resizing images to match the size of the images
     used in the training of the neural network.
    RH 2022

    Args:
        img (np.ndarray):
            Image to resize
            shape: (H,W)
        scale (float):
            Scale factor to use for resizing
        clamp_range (bool):
            If True, will clamp the image to the range [min(img), max(img)]
            This is to prevent the interpolation from going outside of the
             range of the image.

    Returns:
        np.ndarray:
            Resized image
    """
    img_rs = np.array(torchvision.transforms.functional.affine(
#         img=torch.as_tensor(img[None,...]),
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
########### FROM BNPM #############
###################################

def hash_file(path, type_hash='MD5', buffer_size=65536):
    """
    Gets hash of a file.
    Based on: https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
    RH 2022

    Args:
        path (str):
            Path to file to be hashed.
        type_hash (str):
            Type of hash to use. Can be:
                'MD5'
                'SHA1'
                'SHA256'
                'SHA512'
        buffer_size (int):
            Buffer size for reading file.
            65536 corresponds to 64KB.

    Returns:
        hash_val (str):
            Hash of file.
    """

    if type_hash == 'MD5':
        hasher = hashlib.md5()
    elif type_hash == 'SHA1':
        hasher = hashlib.sha1()
    elif type_hash == 'SHA256':
        hasher = hashlib.sha256()
    elif type_hash == 'SHA512':
        hasher = hashlib.sha512()
    else:
        raise ValueError(f'{type_hash} is not a valid hash type.')

    with open(path, 'rb') as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            hasher.update(data)

    hash_val = hasher.hexdigest()

    return hash_val


def download_file(
    url,
    path_save,
    check_local_first=True,
    check_hash=False,
    hash_type='MD5',
    hash_hex=None,
    mkdir=False,
    allow_overwrite=True,
    write_mode='wb',
    verbose=True,
    chunk_size=1024,
):
    """
    Download a file from a URL to a local path using requests.
    Allows for checking if file already exists locally and
    checking the hash of the downloaded file against a provided hash.
    RH 2022

    Args:
        url (str):
            URL of file to download.
            If url is None, then no download is attempted.
        path_save (str):
            Path to save file to.
        check_local_first (bool):
            If True, checks if file already exists locally.
            If True and file exists locally, plans to skip download.
            If True and check_hash is True, checks hash of local file.
             If hash matches, skips download. If hash does not match,
             downloads file.
        check_hash (bool):
            If True, checks hash of local or downloaded file against
             hash_hex.
        hash_type (str):
            Type of hash to use. Can be:
                'MD5', 'SHA1', 'SHA256', 'SHA512'
        hash_hex (str):
            Hash to compare to. In hex format (e.g. 'a1b2c3d4e5f6...').
            Can be generated using hash_file() or hashlib and .hexdigest().
            If check_hash is True, hash_hex must be provided.
        mkdir (bool):
            If True, creates parent directory of path_save if it does not exist.
        write_mode (str):
            Write mode for saving file. Should be one of:
                'wb' (write binary)
                'ab' (append binary)
                'xb' (write binary, fail if file exists)
        verbose (bool):
            If True, prints status messages.
        chunk_size (int):
            Size of chunks to download file in.
    """
    import os
    import requests

    # Check if file already exists locally
    if check_local_first:
        if os.path.isfile(path_save):
            print(f'File already exists locally: {path_save}') if verbose else None
            # Check hash of local file
            if check_hash:
                hash_local = hash_file(path_save, type_hash=hash_type)
                if hash_local == hash_hex:
                    print('Hash of local file matches provided hash_hex.') if verbose else None
                    return True
                else:
                    print('Hash of local file does not match provided hash_hex.') if verbose else None
                    print(f'Hash of local file: {hash_local}') if verbose else None
                    print(f'Hash provided in hash_hex: {hash_hex}') if verbose else None
                    print('Downloading file...') if verbose else None
            else:
                return True
        else:
            print(f'File does not exist locally: {path_save}. Will attempt download from {url}') if verbose else None

    # Download file
    if url is None:
        print('No URL provided. No download attempted.') if verbose else None
        return None
    try:
        response = requests.get(url, stream=True)
    except requests.exceptions.RequestException as e:
        print(f'Error downloading file: {e}') if verbose else None
        return False
    # Check response
    if response.status_code != 200:
        print(f'Error downloading file. Response status code: {response.status_code}') if verbose else None
        return False
    # Create parent directory if it does not exist
    prepare_filepath_for_saving(path_save, mkdir=mkdir, allow_overwrite=allow_overwrite)
    # Download file with progress bar
    total_size = int(response.headers.get('content-length', 0))
    wrote = 0
    with open(path_save, write_mode) as f:
        with tqdm(total=total_size, disable=(verbose==False), unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for data in response.iter_content(chunk_size):
                wrote = wrote + len(data)
                f.write(data)
                pbar.update(len(data))
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")
        return False
    # Check hash
    hash_local = hash_file(path_save, type_hash=hash_type)
    if check_hash:
        if hash_local == hash_hex:
            print('Hash of downloaded file matches hash_hex.') if verbose else None
            return True
        else:
            print('Hash of downloaded file does not match hash_hex.') if verbose else None
            print(f'Hash of downloaded file: {hash_local}') if verbose else None
            print(f'Hash provided in hash_hex: {hash_hex}') if verbose else None
            return False
    else:
        print(f'Hash of downloaded file: {hash_local}') if verbose else None
        return True


def extract_zip(
    path_zip,
    path_extract=None,
    verbose=True,
):
    """
    Extracts a zip file.
    RH 2022

    Args:
        path_zip (str):
            Path to zip file.
        path_extract (str):
            Path (directory) to extract zip file to.
            If None, extracts to the same directory as the zip file.
        verbose (int):
            Whether to print progress.

    Returns:
        paths_extracted (list):
            List of paths to extracted files.
    """
    import zipfile

    if path_extract is None:
        path_extract = str(Path(path_zip).resolve().parent)
    path_extract = str(Path(path_extract).resolve().absolute())

    print(f'Extracting {path_zip} to {path_extract}.') if verbose else None

    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(path_extract)
        paths_extracted = [str(Path(path_extract) / p) for p in zip_ref.namelist()]

    print('Completed zip extraction.') if verbose else None

    return paths_extracted


def prepare_filepath_for_saving(path, mkdir=False, allow_overwrite=True):
    """
    Checks if a file path is valid.
    RH 2022

    Args:
        path (str):
            Path to check.
        mkdir (bool):
            If True, creates parent directory if it does not exist.
        allow_overwrite (bool):
            If True, allows overwriting of existing file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True) if mkdir else None
    assert allow_overwrite or not Path(path).exists(), f'{path} already exists.'
    assert Path(path).parent.exists(), f'{Path(path).parent} does not exist.'
    assert Path(path).parent.is_dir(), f'{Path(path).parent} is not a directory.'



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
    demo:

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

    dataset = util.dataset_simCLR(  torch.tensor(images),
                                labels,
                                n_transforms=2,
                                transform=scripted_transforms,
                                DEVICE='cpu',
                                dtype_X=torch.float32,
                                dtype_y=torch.int64 )

    dataloader = torch.utils.data.DataLoader(   dataset,
                                            batch_size=64,
        #                                     sampler=sampler,
                                            shuffle=True,
                                            drop_last=True,
                                            pin_memory=False,
                                            num_workers=0,
                                            )
    """
    def __init__(   self,
                    X,
                    y,
                    n_transforms=2,
                    class_weights=None,
                    transform=None,
                    DEVICE='cpu',
                    dtype_X=torch.float32,
                    dtype_y=torch.int64,
                    temp_uncertainty=1,
                    expand_dim=True
                    ):

        """
        Make a dataset from a list / numpy array / torch tensor
        of images and labels.
        RH 2021 / JZ 2021

        Args:
            X (torch.Tensor / np.array / list of float32):
                Images.
                Shape: (n_samples, height, width)
                Currently expects no channel dimension. If/when
                 it exists, then shape should be
                (n_samples, n_channels, height, width)
            y (torch.Tensor / np.array / list of ints):
                Labels.
                Shape: (n_samples)
            n_transforms (int):
                Number of transformations to apply to each image.
                Should be >= 1.
            transform (callable, optional):
                Optional transform to be applied on a sample.
                See torchvision.transforms for more information.
                Can use torch.nn.Sequential( a bunch of transforms )
                 or other methods from torchvision.transforms. Try
                 to use torch.jit.script(transform) if possible.
                If not None:
                 Transform(s) are applied to each image and the
                 output shape of X_sample_transformed for
                 __getitem__ will be
                 (n_samples, n_transforms, n_channels, height, width)
                If None:
                 No transform is applied and output shape
                 of X_sample_trasformed for __getitem__ will be
                 (n_samples, n_channels, height, width)
                 (which is missing the n_transforms dimension).
            DEVICE (str):
                Device on which the data will be stored and
                 transformed. Best to leave this as 'cpu' and do
                 .to(DEVICE) on the data for the training loop.
            dtype_X (torch.dtype):
                Data type of X.
            dtype_y (torch.dtype):
                Data type of y.

        Returns:
            torch.utils.data.Dataset:
                torch.utils.data.Dataset object.
        """

        self.expand_dim = expand_dim

        self.X = torch.as_tensor(X, dtype=dtype_X, device=DEVICE) # first dim will be subsampled from. Shape: (n_samples, n_channels, height, width)
        self.X = self.X[:,None,...] if expand_dim else self.X
        self.y = torch.as_tensor(y, dtype=dtype_y, device=DEVICE) # first dim will be subsampled from.

        self.idx = torch.arange(self.X.shape[0], device=DEVICE)
        self.n_samples = self.X.shape[0]

        self.transform = transform
        self.n_transforms = n_transforms

        self.temp_uncertainty = temp_uncertainty

        self.headmodel = None

        self.net_model = None
        self.classification_model = None


        # self.class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=DEVICE)

        # self.classModelParams_coef_ = mp.Array(np.ctypeslib.as_array(mp.Array(ctypes.c_float, feature)))

        if X.shape[0] != y.shape[0]:
            raise ValueError('RH Error: X and y must have same first dimension shape')

    def tile_channels(X_in, dim=-3):
        """
        Expand dimension dim in X_in and tile to be 3 channels.

        JZ 2021 / RH 2021

        Args:
            X_in (torch.Tensor or np.ndarray):
                Input image.
                Shape: [n_channels==1, height, width]

        Returns:
            X_out (torch.Tensor or np.ndarray):
                Output image.
                Shape: [n_channels==3, height, width]
        """
        dims = [1]*len(X_in.shape)
        dims[dim] = 3
        return torch.tile(X_in, dims)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Retrieves and transforms a sample.
        RH 2021 / JZ 2021

        Args:
            idx (int):
                Index / indices of the sample to retrieve.

        Returns:
            X_sample_transformed (torch.Tensor):
                Transformed sample(s).
                Shape:
                    If transform is None:
                        X_sample_transformed[batch_size, n_channels, height, width]
                    If transform is not None:
                        X_sample_transformed[n_transforms][batch_size, n_channels, height, width]
            y_sample (int):
                Label(s) of the sample(s).
            idx_sample (int):
                Index of the sample(s).
        """

        y_sample = self.y[idx]
        idx_sample = self.idx[idx]

        if self.classification_model is not None:
            # features = self.net_model(tile_channels(self.X[idx][:,None,...], dim=1))
            # proba = self.classification_model.predict_proba(features.cpu().detach())[0]
            proba = self.classification_model.predict_proba(self.tile_channels(self.X[idx_sample][:,None,...], dim=-3))[0]

            # sample_weight = loss_uncertainty(torch.as_tensor(proba, dtype=torch.float32), temperature=self.temp_uncertainty)
            sample_weight = 1
        else:
            sample_weight = 1

        X_sample_transformed = []
        if self.transform is not None:
            for ii in range(self.n_transforms):

                # X_sample_transformed.append(tile_channels(self.transform(self.X[idx_sample]), dim=0))
                X_transformed = self.transform(self.X[idx_sample])
                X_sample_transformed.append(X_transformed)
        else:
            X_sample_transformed = self.tile_channels(self.X[idx_sample], dim=-3)

        return X_sample_transformed, y_sample, idx_sample, sample_weight
    