"""
hashes = {'baseline': ('1FCcPZUuOR7xG-hdO6Ei6mx8YnKysVsa8',
                       {'params': ('params.json', '877e17df8fa511a03bc99cd507a54403'),
                        'model': ('model.py', '55e1dd233989753fe0719c8238d0345e'),
                        'state_dict': ('ConvNext_tiny__1_0_unfrozen__simCLR.pth',
                                       'a5fae4c9ea95f2c78b4690222b2928a5')}),
          'occlusion': ('1D2Qa-YUNX176Q-wgboGflW0K6un7KYeN',
                        {'params': ('params.json', '68cf1bd47130f9b6d4f9913f86f0ccaa'),
                         'model': ('model.py', '61c85529b7aa33e0dfadb31ee253a7e1'),
                         'state_dict': ('ConvNext_tiny__1_0_best__simCLR.pth',
                                        '3287e001ff28d07ada2ae70aa7d0a4da')}),
          'minaffine': ('1Xh02nfw_Fgb9uih1WCrsFNI-WIYXDVDn',
                        {'params': ('params.json', '9399a311d47e6966c1201defde4c6c34'),
                        'model': ('model.py', '8789a7b27e41b39ee94c9f732f38eafc'),
                        'state_dict': ('ConvNext_tiny__1_0_unfrozen__simCLR.pth',
                                       '172a992fed5e4bbabc5503e19630b621')})}
"""


import sys
from pathlib import Path
import json
import os
import hashlib
import PIL
import multiprocessing as mp

import numpy as np
import gdown
import torch
import torchvision
from torch.nn import Module
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.signal


class ROInet_embedder:
    """
    Class for loading the ROInet model, preparing data for it, 
     and running it.

    RH 2022
    """
    def __init__(
        self,
        device='cpu',
        dir_networkFiles=None,
        download_from_gDrive='check_local_first',
        gDriveID='1D2Qa-YUNX176Q-wgboGflW0K6un7KYeN',
        hash_dict_networkFiles={
            'params': ('params.json', '68cf1bd47130f9b6d4f9913f86f0ccaa'),
            'model': ('model.py', '61c85529b7aa33e0dfadb31ee253a7e1'),
            'state_dict': ('ConvNext_tiny__1_0_best__simCLR.pth', '3287e001ff28d07ada2ae70aa7d0a4da'),
        },
        forward_pass_version='latent',
        verbose=True,
    ):
        """
        Initialize the class.
        This will look for a local copy of the network files, and
         if they don't exist, it will download them from Google Drive
         using a user specified gDriveID.
        There is some hash checking to make sure the files are the same.

        Args:
            device (str):
                The device to use for the model and data.
            dir_networkFiles (str):
                The directory to find or download the network files into
            download_from_gDrive (str):
                Approach to downloading the network files.
                'check_local_first':
                    Check if the network files are already in 
                     dir_networkFiles. If so, use them.
                'force_download':
                    Download the network files from Google Drive.
                'force_local':
                    Use the network files in dir_networkFiles.
                    If they don't exist, raise an error.
                    Hash checking is not done.
            hash_dict_networkFiles (dict):
                A dictionary of the hash values of the network files.
                Each item is {key: (filename, hash_value)}
                The (filename, hash_value) pairs can be made using:
                 paths_networkFiles = [(Path(dir_networkFiles).resolve() / name).as_posix() for name in get_dir_contents(dir_networkFiles)[1]]
                 {Path(path).name: hash_file(path) for path in paths_networkFiles}
            forward_pass_version (str):
                The version of the forward pass to use.
                'latent':
                    Return the post-head output latents.
                'head':
                    Return the output of the head layers.
                'base':
                    Return the output of the base model.
            verbose (bool):
                Whether to print out extra information.
        """
        self._device = device
        self._verbose = verbose


        self._dir_networkFiles = dir_networkFiles
        self._gDriveID = gDriveID

        ## Find or download network files
        if download_from_gDrive == 'force_download':
            paths_downloaded = self._download_network_files()
            print(paths_downloaded) if self._verbose else None
            if hash_dict_networkFiles is None:
                print('Skipping hash check because hash_dict_networkFiles is None')
                paths_matching = {
                    'params': np.array(paths_downloaded)[[('params.json' in t) for t in paths_downloaded]][0],
                    'model': np.array(paths_downloaded)[[('model.py' in t) for t in paths_downloaded]][0],
                    'state_dict': np.array(paths_downloaded)[[('.pth' in t) for t in paths_downloaded]][0],
                }
                print(paths_matching) if self._verbose else None
            else:
                results_all, results, paths_matching = compare_file_hashes(  
                    hash_dict_true=hash_dict_networkFiles,
                    dir_files_test=dir_networkFiles,
                    verbose=True,
                )
                print(paths_matching) if self._verbose else None
                if results_all == False:
                    print(f'WARNING: Hash comparison failed. Could not match downloaded files to hash_dict_networkFiles.')

        if download_from_gDrive == 'check_local_first':
            assert hash_dict_networkFiles is not None, "if using download_from_gDrive='check_local_first' hash_dict_networkFiles cannot be None"
            results_all, results, paths_matching = compare_file_hashes(  
                hash_dict_true=hash_dict_networkFiles,
                dir_files_test=dir_networkFiles,
                verbose=True,
            )
            print(f'Successful hash comparison. Found matching files: {paths_matching}') if results_all and self._verbose else None
            if results_all == False:
                print(f'Hash comparison failed. Downloading from Google Drive.') if self._verbose else None
                self._download_network_files()
                results_all, results, paths_matching = compare_file_hashes(  
                    hash_dict_true=hash_dict_networkFiles,
                    dir_files_test=dir_networkFiles,
                    verbose=True,
                )
                if results_all:
                    print(f'Successful hash comparison. Found matching files: {paths_matching}')  if self._verbose else None
                else:
                    raise Exception(f'Downloaded network files do not match expected hashes. Results: {results}')
        
        if download_from_gDrive == 'force_local':
            assert hash_dict_networkFiles is not None, "if using download_from_gDrive='force_local' hash_dict_networkFiles cannot be None"
            results_all, results, paths_matching = compare_file_hashes(  
                hash_dict_true=hash_dict_networkFiles,
                dir_files_test=dir_networkFiles,
                verbose=True,
            )
            if results_all == False:
                print(f'WARNING: Hash comparison failed. Could not match local files to hash_dict_networkFiles.')

        ## Import network files
        sys.path.append(dir_networkFiles)
        import model
        print(f"Imported model from {dir_networkFiles}/model.py") if self._verbose else None

        with open(paths_matching['params']) as f:
            self.params_model = json.load(f)
            print(f"Loaded params_model from {paths_matching['params']}") if self._verbose else None
            self.net = model.make_model(fwd_version=forward_pass_version, **self.params_model)
            print(f"Generated network using params_model") if self._verbose else None
            
        ## Prep network and load state_dict
        for param in self.net.parameters():
            param.requires_grad = False
        self.net.eval()

        self.net.load_state_dict(torch.load(paths_matching['state_dict'], map_location=torch.device(self._device)))
        print(f'Loaded state_dict into network from {paths_matching["state_dict"]}') if self._verbose else None

        self.net = self.net.to(self._device)
        print(f'Loaded network onto device {self._device}') if self._verbose else None

        ## Prepare dataloader
        
    def generate_dataloader(
        self,
        ROI_images,
        # goal_frac=0.1929,
        um_per_pixel=1.0,
        pref_plot=False,
        batchSize_dataloader=8,
        pinMemory_dataloader=True,
        numWorkers_dataloader=-1,
        persistentWorkers_dataloader=True,
        prefetchFactor_dataloader=2,
        transforms=None,
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
                 resize (to 224x224), and tile channels (to 3).
                 These are the minimum transforms required to pass
                 images through the network.
        """
        if numWorkers_dataloader == -1:
            numWorkers_dataloader = mp.cpu_count()

        if self._device == 'cpu':
            pinMemory_dataloader = False
            numWorkers_dataloader = 0
            persistentWorkers_dataloader = False

        print('Starting: resizing ROIs') if self._verbose else None
        # sf_ptile = np.array([np.percentile(np.mean(sf>0, axis=(1,2)), ptile_norm) for sf in tqdm(ROI_images)]).mean()
        # scale_forRS = (goal_frac/sf_ptile)**scale_norm
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
                    size=(224, 224),
                    interpolation=torchvision.transforms.InterpolationMode.BILINEAR
                ), 
                TileChannels(dim=0, n_channels=3),
            )
        transforms_scripted = torch.jit.script(transforms)
        print(f'Defined image transformations: {transforms}') if self._verbose else None

        self.dataset = dataset_simCLR(
                X=torch.as_tensor(ROI_images_rs, device='cpu', dtype=torch.float32),
                y=torch.as_tensor(torch.zeros(ROI_images_rs.shape[0]), device='cpu', dtype=torch.float32),
                n_transforms=1,
                class_weights=np.array([1]),
                transform=transforms_scripted,
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
        self.latents = torch.cat([self.net(data[0][0].to(self._device)).detach() for data in tqdm(self.dataloader, mininterval=60)], dim=0).cpu()
        print(f'completed: running data through network')
        return self.latents


    def _download_network_files(self):
        if self._gDriveID is None or self._dir_networkFiles is None:
            raise ValueError('gDriveID and dir_networkFiles must be specified if download_from_gDrive is True')

        print(f'Downloading network files from Google Drive to {self._dir_networkFiles}') if self._verbose else None
        paths_files = gdown.download_folder(id=self._gDriveID, output=self._dir_networkFiles, quiet=False, use_cookies=False)
        print('Downloaded network files') if self._verbose else None
        return paths_files
    
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

def get_dir_contents(directory):
    '''
    Get the contents of a directory (does not
     include subdirectories).
    RH 2021

    Args:
        directory (str):
            path to directory
    
    Returns:
        folders (List):
            list of folder names
        files (List):
            list of file names
    '''
    walk = os.walk(directory, followlinks=False)
    folders = []
    files = []
    for ii,level in enumerate(walk):
        folders, files = level[1:]
        if ii==0:
            break
    return folders, files



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


def compare_file_hashes(
    hash_dict_true,
    dir_files_test=None,
    paths_files_test=None,
    verbose=True,
):
    """
    Compares hashes of files in a directory or list of paths
     to user provided hashes.
    RH 2022

    Args:
        hash_dict_true (dict):
            Dictionary of hashes to compare to.
            Each entry should be:
                {'key': ('filename', 'hash')}
        dir_files_test (str):
            Path to directory to compare hashes of files in.
            Unused if paths_files_test is not None.
        paths_files_test (list of str):
            List of paths to files to compare hashes of.
            Optional. dir_files_test is used if None.
        verbose (bool):
            Whether or not to print out failed comparisons.

    Returns:
        total_result (bool):
            Whether or not all hashes were matched.
        individual_results (list of bool):
            Whether or not each hash was matched.
        paths_matching (dict):
            Dictionary of paths that matched.
            Each entry is:
                {'key': 'path'}
    """
    if paths_files_test is None:
        if dir_files_test is None:
            raise ValueError('Must provide either dir_files_test or path_files_test.')
        
        ## make a dict of {filename: path} for each file in dir_files_test
        files_test = {filename: (Path(dir_files_test).resolve() / filename).as_posix() for filename in get_dir_contents(dir_files_test)[1]} 
    else:
        files_test = {Path(path).name: path for path in paths_files_test}

    paths_matching = {}
    results_matching = {}
    for key, (filename, hash_true) in hash_dict_true.items():
        match = True
        if filename not in files_test:
            print(f'{filename} not found in test directory: {dir_files_test}.') if verbose else None
            match = False
        elif hash_true != hash_file(files_test[filename]):
            print(f'{filename} hash mismatch with {key, filename}.') if verbose else None
            match = False
        if match:
            paths_matching[key] = files_test[filename]
        results_matching[key] = match

    return all(results_matching.values()), results_matching, paths_matching



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


