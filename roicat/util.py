from pathlib import Path
import warnings
import copy
from typing import Dict, Any, Optional, Union, List, Tuple, Callable, Iterable, Iterator, Type
import datetime
import collections
import importlib

import numpy as np
import scipy.sparse
from tqdm import tqdm
import torch

import richfile as rf

from . import helpers


def get_roicat_version() -> str:
    """
    Retrieves the version of the roicat package.

    Returns:
        (str): 
            version (str):
                The version of the roicat package.
    """
    return importlib.metadata.version('roicat')


def get_default_parameters(
    pipeline='tracking', 
    path_defaults=None
):
    """
    This function returns a dictionary of parameters that can be used to run
    different pipelines. RH 2023

    Args:
        pipeline (str):
            The name of the pipeline to use. Options: \n
                * 'tracking': Tracking pipeline. \n
                * 'classification_inference': Classification inference pipeline
                  (TODO). \n
                * 'classification_training': Classification training pipeline
                  (TODO). \n
                * 'model_training': Model training pipeline (TODO). \n
        path_defaults (str):
            A path to a yaml file containing a parameters dictionary. The
            parameters from the file will be loaded as is. If None, the default
            parameters will be used.

    Returns:
        (dict):
            params (dict):
                A dictionary containing the default parameters.
    """

    if path_defaults is not None:
        defaults = helpers.yaml_load(path_defaults)
    else:
        defaults = {
            'general': {
                'use_GPU': True,
                'verbose': True,
                'random_seed': None,
            },
            'data_loading': {
                'data_kind': 'data_suite2p',  ## Can be 'suite2p', 'roiextractors', or 'roicat'. See documentation and/or notebook on custom data loading for more details.
                'dir_outer': None,  ## directory where directories containing below 'pathSuffixTo...' are
                'common': {
                    'um_per_pixel': 1.0,  ## Number of microns per pixel for the imaging dataset. Doesn't need to be exact. Used for resizing the ROIs. Check the images of the resized ROIs to tweak.
                    'centroid_method': 'centerOfMass', ## Can be 'centerOfMass' or 'median'.
                    'out_height_width': [36,36],  ## Height and width of the small ROI_images. Should generally be tuned slightly bigger than the largest ROIs. Leave if uncertain or if ROIs are small enough to fit in the default size.
                },
                'data_suite2p': {              
                    'new_or_old_suite2p': 'new',  ## Can be 'new' or 'old'. 'new' is for the Python version of Suite2p, 'old' is for the MATLAB version.
                    'type_meanImg': 'meanImgE',  ## Can be 'meanImg' or 'meanImgE'. 'meanImg' is the mean image of the dataset, 'meanImgE' is the mean image of the dataset after contrast enhancement.
                },
                'data_roicat': {
                    'filename_search': r'data_roicat.richfile',  ## Name stem of the single file (as a regex search string) in 'dir_outer' to look for. The files should be saved Data_roicat object.
                },
            },
            'alignment': {
                'initialization': {
                    'use_match_search': True,  ## Whether or not to use our match search algorithm to initialize the alignment.
                    'all_to_all': False,  ## Force the use of our match search algorithm for all-pairs matching. Much slower (False: O(N) vs. True: O(N^2)), but more accurate.
                    'radius_in': 4.0,  ## Value in micrometers used to define the maximum shift/offset between two images that are considered to be aligned. Larger means more lenient alignment.
                    'radius_out': 20.0,  ## Value in micrometers used to define the minimum shift/offset between two images that are considered to be misaligned.
                    'z_threshold': 4.0,  ## Z-score required to define two images as aligned. Larger values results in more stringent alignment requirements.
                },
                'augment': {
                    'normalize_FOV_intensities': True,  ## Whether or not to normalize the FOV_images to the max value across all FOV images.
                    'roi_FOV_mixing_factor': 0.5,  ## default: 0.5. Fraction of the max intensity projection of ROIs that is added to the FOV image. 0.0 means only the FOV_images, larger values mean more of the ROIs are added.
                    'use_CLAHE': True,  ## Whether or not to use 'Contrast Limited Adaptive Histogram Equalization'. Useful if params['importing']['type_meanImg'] is not a contrast enhanced image (like 'meanImgE' in Suite2p)
                    'CLAHE_grid_block_size': 10,  ## Size of the block size for the grid for CLAHE. Smaller values means more local contrast enhancement.
                    'CLAHE_clipLimit': 1.0,  ## Clipping limit for CLAHE. Higher values mean more contrast.
                    'CLAHE_normalize': True,  ## Whether or not to normalize the CLAHE image.
                },
                'fit_geometric': {
                    'template': 0.5,  ## Which session to use as a registration template. If input is float (ie 0.0, 0.5, 1.0, etc.), then it is the fractional position of the session to use; if input is int (ie 1, 2, 3), then it is the index of the session to use (0-indexed)
                    'template_method': 'image',  ## Can be 'sequential' or 'image'. If 'sequential', then the template is the FOV_image of the previous session. If 'image', then the template is the FOV_image of the session specified by 'template'.
                    'mask_borders': [0, 0, 0, 0],  ## Number of pixels to mask from the borders of the FOV_image. Useful for removing artifacts from the edges of the FOV_image.
                    'method': 'RoMa',  ## Accuracy order (best to worst): RoMa (by far, but slow without a GPU), LoFTR, DISK_LightGlue, ECC_cv2, (the following are not recommended) SIFT, ORB
                    'kwargs_method': {
                        'RoMa': {
                            'model_type': 'outdoor',
                            'n_points': 10000,  ## Higher values mean more points are used for the registration. Useful for larger FOV_images. Larger means slower.
                            'batch_size': 1000,
                        },
                        'DISK_LightGlue': {
                            'num_features': 3000,  ## Number of features to extract and match. I've seen best results around 2048 despite higher values typically being better.
                            'threshold_confidence': 0.2,  ## Higher values means fewer but better matches.
                        },
                        'LoFTR': {
                            'model_type': 'indoor_new',
                            'threshold_confidence': 0.2,  ## Higher values means fewer but better matches.
                        },
                        'ECC_cv2': {
                            'mode_transform': 'euclidean',  ## Must be one of {'translation', 'affine', 'euclidean', 'homography'}. See cv2 documentation on findTransformECC for more details.
                            'n_iter': 200,
                            'termination_eps': 1e-09,  ## Termination criteria for the registration algorithm. See documentation for more details.
                            'gaussFiltSize': 1,  ## Size of the gaussian filter used to smooth the FOV_image before registration. Larger values mean more smoothing.
                            'auto_fix_gaussFilt_step': 10,  ## If the registration fails, then the gaussian filter size is reduced by this amount and the registration is tried again.
                        },
                        'PhaseCorrelation': {
                            'bandpass_freqs': [1, 30],
                            'order': 5,
                        },
                        'SIFT': {
                            'nfeatures': 10000,
                            'contrastThreshold': 0.04,
                            'edgeThreshold': 10,
                            'sigma': 1.6,
                        },
                        'ORB': {
                            'nfeatures': 1000,
                            'scaleFactor': 1.2,
                            'nlevels': 8,
                            'edgeThreshold': 31,
                            'firstLevel': 0,
                            'WTA_K': 2,
                            'scoreType': 0,
                            'patchSize': 31,
                            'fastThreshold': 20,
                        },
                    },
                    'kwargs_RANSAC': {  ## Parameters related to the RANSAC algorithm used for point/descriptor based registration methods.
                        'inl_thresh': 3.0,  ## Threshold for the inliers. Larger values mean more points are considered inliers.
                        'max_iter': 100,  ## Maximum number of iterations for the RANSAC algorithm.
                        'confidence': 0.99,  ## Confidence level for the RANSAC algorithm. Larger values mean more points are considered inliers.
                    },
                },
                'fit_nonrigid': {
                    'template': 0.5,  ## Which session to use as a registration template. If input is float (ie 0.0, 0.5, 1.0, etc.), then it is the fractional position of the session to use; if input is int (ie 1, 2, 3), then it is the index of the session to use (0-indexed)
                    'template_method': 'image',  ## Can be 'sequential' or 'image'. If 'sequential', then the template is the FOV_image of the previous session. If 'image', then the template is the FOV_image of the session specified by 'template'.
                    'method': 'DeepFlow',
                    'kwargs_method': {
                        'RoMa': {
                            'model_type': 'outdoor',
                        },
                        'DeepFlow': {},
                        'OpticalFlowFarneback': {
                            'pyr_scale': 0.7,
                            'levels': 5,
                            'winsize': 128,
                            'iterations': 15,
                            'poly_n': 5,
                            'poly_sigma': 1.5,            
                        },
                    },
                },
                'transform_ROIs': {
                    'normalize': True,  ## If True, normalize the spatial footprints to have a sum of 1.
                },
            },
            'blurring': {
                'kernel_halfWidth': 2.0,  ## Half-width of the cosine kernel used for blurring. Set value based on how much you think the ROIs move from session to session.
            },
            'ROInet': {
                'network': {
                    'download_method': 'check_local_first',  ## Check to see if a model has already been downloaded to the location (will skip if hash matches)
                    'download_url': 'https://osf.io/x3fd2/download',  ## URL of the model
                    'download_hash': '7a5fb8ad94b110037785a46b9463ea94',  ## Hash of the model file
                    'forward_pass_version': 'latent',  ## How the data is passed through the network
                },
                'dataloader': {
                    'jit_script_transforms': False,  ## (advanced) Whether or not to use torch.jit.script to speed things up
                    'batchSize_dataloader': 8,  ## (advanced) PyTorch dataloader batch_size
                    'pinMemory_dataloader': True,  ## (advanced) PyTorch dataloader pin_memory
                    'numWorkers_dataloader': -1,  ## (advanced) PyTorch dataloader num_workers. -1 is all cores.
                    'persistentWorkers_dataloader': True,  ## (advanced) PyTorch dataloader persistent_workers
                    'prefetchFactor_dataloader': 2,  ## (advanced) PyTorch dataloader prefetch_factor
                },
            },
            'SWT': {
                'kwargs_Scattering2D': {'J': 2, 'L': 12},  ## 'J' is the number of convolutional layers. 'L' is the number of wavelet angles.
                'batch_size': 100,  ## Batch size for each iteration (smaller is less memory but slower)
            },
            'similarity_graph': {
                'sparsification': {
                    'n_workers': -1,  ## Number of CPU cores to use. -1 for all.
                    'block_height': 128,  ## size of a block
                    'block_width': 128,  ## size of a block
                    'algorithm_nearestNeigbors_spatialFootprints': 'brute',  ## algorithm used to find the pairwise similarity for s_sf. ('brute' is slow but exact. See docs for others.)
                },
                'compute_similarity': {
                    'spatialFootprint_maskPower': 1.0,  ##  An exponent to raise the spatial footprints to to care more or less about bright pixels
                },
                'normalization': {
                    'k_max': 100,  ## Maximum number of nearest neighbors * n_sessions to consider for the normalizing distribution
                    'k_min': 10,  ## Minimum number of nearest neighbors * n_sessions to consider for the normalizing distribution
                    'algo_NN': 'kd_tree',  ## Nearest neighbors algorithm to use
                },
            },
            'clustering': {
                'mixing_method': 'automatic',  ## Can be 'automatic' or 'manual'. If 'automatic', then the parameters are found automatically. If 'manual', then the parameters are set manually.
                'parameters_automatic_mixing': {
                    'n_bins': None,  ## Number of bins to use for the histograms of the distributions
                    'smoothing_window_bins': None,  ## Number of bins to use to smooth the distributions
                    'kwargs_findParameters': {
                        'n_patience': 300,  ## Number of optimization epoch to wait for tol_frac to converge
                        'tol_frac': 0.001,  ## Fractional change below which optimization will conclude
                        'max_trials': 1200,  ## Max number of optimization epochs
                        'max_duration': 60*10,  ## Max amount of time (in seconds) to allow optimization to proceed for
                    },
                    'bounds_findParameters': {
                        'power_NN': [0.0, 2.],  ## Bounds for the exponent applied to s_NN
                        'power_SWT': [0.0, 2.],  ## Bounds for the exponent applied to s_SWT
                        'p_norm': [-5, -0.1],  ## Bounds for the p-norm p value (Minkowski) applied to mix the matrices
                        'sig_NN_kwargs_mu': [0., 1.0],  ## Bounds for the sigmoid center for s_NN
                        'sig_NN_kwargs_b': [0.1, 1.5],  ## Bounds for the sigmoid slope for s_NN
                        'sig_SWT_kwargs_mu': [0., 1.0],  ## Bounds for the sigmoid center for s_SWT
                        'sig_SWT_kwargs_b': [0.1, 1.5],  ## Bounds for the sigmoid slope for s_SWT
                    },
                    'n_jobs_findParameters': -1,  ## Number of CPU cores to use (-1 is all cores)
                },
                'parameters_manual_mixing': {
                    'power_SF': 1.0,   ## s_sf**power_SF   (Higher values means clustering is more sensitive to spatial overlap of ROIs)
                    'power_NN': 0.5,   ## s_NN**power_NN   (Higher values means clustering is more sensitive to visual similarity of ROIs)
                    'power_SWT': 0.5,  ## s_SWT**power_SWT (Higher values means clustering is more sensitive to visual similarity of ROIs)
                    'p_norm': -1.0,    ## norm([s_sf, s_NN, s_SWT], p=p_norm) (Higher values means clustering requires all similarity metrics to be high)
                #     'sig_SF_kwargs': {'mu':0.5, 'b':1.0},  ## Sigmoid parameters for s_sf (mu is the center, b is the slope)
                    'sig_SF_kwargs': None,
                    'sig_NN_kwargs': {'mu': 0.5, 'b': 1.0},  ## Sigmoid parameters for s_NN (mu is the center, b is the slope)
                #     'sig_NN_kwargs': None,
                    'sig_SWT_kwargs': {'mu': 0.5, 'b': 1.0},  ## Sigmoid parameters for s_SWT (mu is the center, b is the slope)
                #     'sig_SWT_kwargs': None,
                },
                'pruning': {
                    'd_cutoff': None,  ## Optionally manually specify a distance cutoff
                    'stringency': 1.0,  ## How to scale the d_cuttoff. This is a scalaing factor. Smaller numbers result in more pruning.
                    'convert_to_probability': False,  ## Whether or not to convert the similarity matrix and distance matrix to a probability matrix
                },
                'cluster_method': {
                    'method': 'automatic',  ## 'automatic', 'hdbscan', or 'sequential_hungarian'. 'automatic': selects which clustering algorithm to use (generally if n_sessions >=8 then hdbscan, else sequential_hungarian)
                    'n_sessions_switch': 6, ## Number of sessions to switch from sequential_hungarian to hdbscan
                },
                'hdbscan': {
                    'min_cluster_size': 2,  ## Minimum number of ROIs that can be considered a 'cluster'
                    'n_iter_violationCorrection': 6,  ## Number of times to redo clustering sweep after removing violations
                    'split_intraSession_clusters': True,  ## Whether or not to split clusters with ROIs from the same session
                    'cluster_selection_method': 'leaf',  ## (advanced) Method of cluster selection for HDBSCAN (see hdbscan documentation)
                    'd_clusterMerge': None,  ## Distance below which all ROIs are merged into a cluster
                    'alpha': 0.999,  ## (advanced) Scalar applied to distance matrix in HDBSCAN (see hdbscan documentation)
                    'discard_failed_pruning': True,  ## (advanced) Whether or not to set all ROIs that could be separated from clusters with ROIs from the same sessions to label=-1
                    'n_steps_clusterSplit': 100,  ## (advanced) How finely to step through distances to remove violations
                },
                'sequential_hungarian': {
                    'thresh_cost': 0.6, ## Threshold for the cost matrix. Lower numbers result in more clusters.
                },
            },
            'results_saving': {
                'dir_save': None,  ## Directory to save results to. If None, will not save.
                'prefix_name_save': str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")),  ## Prefix to append to the saved files
            },
        }

    ## Pipeline specific parameters
    ### prepare the different modules for each pipeline
    keys_pipeline = {
        'tracking': [
            'general',
            'data_loading',
            'alignment',
            'blurring',
            'ROInet',
            'SWT',
            'similarity_graph',
            'clustering',
            'results_saving',
        ],
        'classification_inference': [
            'general',
            'data_loading',
            'ROInet',
            'results_saving',
        ],
        'classification_training': [
            'general',
            'data_loading',
            'ROInet',
            'results_saving',
        ],
    }

    ### prepare pipeline specific parameters
    if pipeline == 'tracking':
        out = copy.deepcopy({key: defaults[key] for key in keys_pipeline[pipeline]})
        out['ROInet']['network'] = {
            'download_method': 'check_local_first',  ## Check to see if a model has already been downloaded to the location (will skip if hash matches)
            'download_url': 'https://osf.io/x3fd2/download',  ## URL of the model
            'download_hash': '7a5fb8ad94b110037785a46b9463ea94',  ## Hash of the model file
            'forward_pass_version': 'latent',  ## How the data is passed through the network
        }
    elif pipeline == 'classification_inference':
        out = copy.deepcopy({key: defaults[key] for key in keys_pipeline[pipeline]})
        out['ROInet']['network'] = {
            'download_method': 'check_local_first',  ## Check to see if a model has already been downloaded to the location (will skip if hash matches)
            'download_url': 'https://osf.io/c8m3b/download',  ## URL of the model
            'download_hash': '357a8d9b630ec79f3e015d0056a4c2d5',  ## Hash of the model file
            'forward_pass_version': 'head',  ## How the data is passed through the network
        }
    elif pipeline == 'classification_training':
        out = copy.deepcopy({key: defaults[key] for key in keys_pipeline[pipeline]})
        out['ROInet']['network'] = {
            'download_method': 'check_local_first',  ## Check to see if a model has already been downloaded to the location (will skip if hash matches)
            'download_url': 'https://osf.io/c8m3b/download',  ## URL of the model
            'download_hash': '357a8d9b630ec79f3e015d0056a4c2d5',  ## Hash of the model file
            'forward_pass_version': 'head',  ## How the data is passed through the network
        }
    else:
        raise NotImplementedError(f'pipeline={pipeline}, which is not implemented or not recognized. Should be one of: {list(keys_pipeline.keys())}')

    return out


def system_info(verbose: bool = False,) -> Dict:
    """
    Checks and prints the versions of various important software packages.
    RH 2022

    Args:
        verbose (bool): 
            Whether to print the software versions. 
            (Default is ``False``)

    Returns:
        (Dict): 
            versions (Dict):
                Dictionary containing the versions of various software packages.
    """
    ## Operating system and version
    import platform
    def try_fns(fn):
        try:
            return fn()
        except:
            return None        
    fns = {key: val for key, val in platform.__dict__.items() if (callable(val) and key[0] != '_')}
    operating_system = {key: try_fns(val) for key, val in fns.items() if (callable(val) and key[0] != '_')}
    print(f'== Operating System ==: {operating_system["uname"]}') if verbose else None

    ## CPU info
    try:
        import cpuinfo
        import multiprocessing as mp
        # cpu_info = cpuinfo.get_cpu_info()
        cpu_n_cores = mp.cpu_count()
        cpu_brand = cpuinfo.cpuinfo.CPUID().get_processor_brand(cpuinfo.cpuinfo.CPUID().get_max_extension_support())
        cpu_info = {'n_cores': cpu_n_cores, 'brand': cpu_brand}
        if 'flags' in cpu_info:
            cpu_info['flags'] = 'omitted'
    except Exception as e:
        warnings.warn(f'RH WARNING: unable to get cpu info. Got error: {e}')
        cpu_info = 'ROICaT Error: Failed to get'
    print(f'== CPU Info ==: {cpu_info}') if verbose else None

    ## RAM
    import psutil
    ram = psutil.virtual_memory()
    print(f'== RAM ==: {ram}') if verbose else None

    ## User
    import getpass
    user = getpass.getuser()

    ## GPU
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        gpu_info = {gpu.id: gpu.__dict__ for gpu in gpus}
    except Exception as e:
        warnings.warn(f'RH WARNING: unable to get gpu info. Got error: {e}')
        gpu_info = 'ROICaT Error: Failed to get'
    print(f'== GPU Info ==: {gpu_info}') if verbose else None
    
    ## Conda Environment
    import os
    if 'CONDA_DEFAULT_ENV' not in os.environ:
        conda_env = 'None'
    else:
        conda_env = os.environ['CONDA_DEFAULT_ENV']
    print(f'== Conda Environment ==: {conda_env}') if verbose else None

    ## Python
    import sys
    python_version = sys.version.split(' ')[0]
    print(f'== Python Version ==: {python_version}') if verbose else None

    ## GCC
    import subprocess
    try:
        gcc_version = subprocess.check_output(['gcc', '--version']).decode('utf-8').split('\n')[0].split(' ')[-1]
    except Exception as e:
        warnings.warn(f'RH WARNING: unable to get gcc version. Got error: {e}')
        gcc_version = 'Faled to get'
    print(f'== GCC Version ==: {gcc_version}') if verbose else None
    
    ## PyTorch
    import torch
    torch_version = str(torch.__version__)
    print(f'== PyTorch Version ==: {torch_version}') if verbose else None
    ## CUDA
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        torch_devices = [f'device {i}: Name={torch.cuda.get_device_name(i)}, Memory={torch.cuda.get_device_properties(i).total_memory / 1e9} GB' for i in range(torch.cuda.device_count())]
        print(f"== CUDA Version ==: {cuda_version}, CUDNN Version: {cudnn_version}, Number of Devices: {torch.cuda.device_count()}, Devices: {torch_devices}, ") if verbose else None
    else:
        cuda_version = None
        cudnn_version = None
        torch_devices = None
        print('== CUDA is not available ==') if verbose else None

    ## all packages in environment
    import importlib.metadata
    pkgs_dict = {dist.metadata['Name'].lower(): dist.version for dist in importlib.metadata.distributions()}

    ## roicat
    import time
    roicat_version = importlib.metadata.version("roicat")
    roicat_fileDate = time.ctime(os.path.getctime(importlib.metadata.distribution("roicat").locate_file('')))
    roicat_stuff = {'version': roicat_version, 'date_installed': roicat_fileDate}
    print(f'== ROICaT Version ==: {roicat_version}') if verbose else None
    print(f'== ROICaT date installed ==: {roicat_fileDate}') if verbose else None

    ## get datetime
    from datetime import datetime
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    versions = {
        'datetime': dt,
        'roicat': roicat_stuff,
        'operating_system': operating_system,
        'cpu_info': cpu_info,  ## This is the slow one.
        'user': user,
        'ram': ram,
        'gpu_info': gpu_info,
        'conda_env': conda_env,
        'python': python_version,
        'gcc': gcc_version,
        'torch': torch_version,
        'cuda': cuda_version,
        'cudnn': cudnn_version,
        'torch_devices': torch_devices,
        'pkgs': pkgs_dict,
    }

    def conv_str(obj):
        if isinstance(obj, (dict, collections.OrderedDict)):
            return {key: conv_str(val) for key, val in obj.items()}
        elif isinstance(obj, (list, tuple, set, frozenset)):
            return [conv_str(val) for val in obj]
        elif isinstance(obj, (int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
        
    versions = conv_str(versions)

    return versions


def set_random_seed(seed=None, deterministic=False):
    """
    Set random seed for reproducibility.
    RH 2023

    Args:
        seed (int, optional):
            Random seed.
            If None, a random seed (spanning int32 integer range) is generated.
        deterministic (bool, optional):
            Whether to make packages deterministic.

    Returns:
        (int):
            seed (int):
                Random seed.
    """
    ### random seed (note that optuna requires a random seed to be set within the pipeline)
    import numpy as np
    seed = int(np.random.randint(0, 2**31 - 1, dtype=np.uint32)) if seed is None else seed

    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import cv2
    cv2.setRNGSeed(seed)

    ## Make torch deterministic
    torch.use_deterministic_algorithms(deterministic)
    ## Make cudnn deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    
    return seed


class ROICaT_Module:
    """
    Super class for ROICaT modules.
    RH 2023

    Attributes:
        _system_info (object): 
            System information.

    """
    def __init__(self) -> None:
        """
        Initializes the ROICaT_Module class by gathering system information.
        """
        self._system_info = system_info()
        
        self.params = {}
        pass

    @property
    def serializable_dict(self) -> Dict[str, Any]:
        """
        Returns a serializable dictionary that can be saved to disk. This method
        goes through all items in self.__dict__ and checks if they are
        serializable. If they are, add them to a dictionary to be returned.

        Returns:
            (Dict[str, Any]): 
                serializable_dict (Dict[str, Any]): 
                    Dictionary containing serializable items.
        """
        from functools import partial
        ## Go through all items in self.__dict__ and check if they are serializable.
        ### If they are, add them to a dictionary to be returned.
        import pickle

        ## Define a list of libraries and classes that are allowed to be serialized.
        allowed_libraries = [
            'roicat',
            'builtins',
            'collections',
            'datetime',
            'itertools',
            'math',
            'numbers',
            'os',
            'pathlib',
            'string',
            'time',
            'numpy',
            'scipy',
            'sklearn',
        ]
        def is_library_allowed(obj):
            try:
                try:
                    module_name = obj.__module__.split('.')[0]
                except:
                    success = False
                try:
                    module_name = obj.__class__.__module__.split('.')[0]
                except:
                    success = False
            except:
                success = False
            else:
                ## Check if the module_name is in the allowed_libraries list.
                if module_name in allowed_libraries:
                    success = True
                else:
                    success = False
            return success
        
        def make_serializable_dict(obj, depth=0, max_depth=100, name=None):
            """
            Recursively go through all items in self.__dict__ and check if they are serializable.
            """
            # print(name)
            msd_partial = partial(make_serializable_dict, depth=depth+1, max_depth=max_depth)
            if depth > max_depth:
                raise Exception(f'RH ERROR: max_depth of {max_depth} reached with object: {obj}')
                
            serializable_dict = {}
            if hasattr(obj, '__dict__') and is_library_allowed(obj):
                for key, val in obj.__dict__.items():
                    try:
                        serializable_dict[key] = msd_partial(val, name=key)
                    except:
                        pass

            elif isinstance(obj, (list, tuple, set, frozenset)):
                serializable_dict = [msd_partial(v, name=f'{name}_{ii}') for ii,v in enumerate(obj)]
            elif isinstance(obj, dict):
                serializable_dict = {k: msd_partial(v, name=f'{name}_{k}') for k,v in obj.items()}
            else:
                try:
                    assert is_library_allowed(obj), f'RH ERROR: object {obj} is not serializable'
                    pickle.dumps(obj)
                except:
                    return {'__repr__': repr(obj)} if hasattr(obj, '__repr__') else {'__str__': str(obj)} if hasattr(obj, '__str__') else None

                serializable_dict = obj

            return serializable_dict
        
        serializable_dict = make_serializable_dict(self, depth=0, max_depth=100, name='self')
        return serializable_dict


    # def save(
    #     self, 
    #     path_save: Union[str, Path],
    #     save_as_serializable_dict: bool = False,
    #     allow_overwrite: bool = False,
    # ) -> None:
    #     """
    #     Saves Data_roicat object to pickle file.

    #     Args:
    #         path_save (Union[str, pathlib.Path]): 
    #             Path to save pickle file.
    #         save_as_serializable_dict (bool): 
    #             An archival-type format that is easy to load data from, but typically 
    #             cannot be used to re-instantiate the object. If ``True``, save the object 
    #             as a serializable dictionary. If ``False``, save the object as a Data_roicat 
    #             object. (Default is ``False``)
    #         allow_overwrite (bool): 
    #             If ``True``, allow overwriting of existing file. (Default is ``False``)

    #     """
    #     from pathlib import Path
    #     ## Check if file already exists
    #     if not allow_overwrite:
    #         assert not Path(path_save).exists(), f"RH ERROR: File already exists: {path_save}. Set allow_overwrite=True to overwrite."

    #     helpers.pickle_save(
    #         obj=self.serializable_dict if save_as_serializable_dict else self,
    #         filepath=path_save,
    #         mkdir=True,
    #         allow_overwrite=allow_overwrite,
    #     )
    #     print(f"Saved Data_roicat as a pickled object to {path_save}.") if self._verbose else None

    # def load(
    #     self,
    #     path_load: Union[str, Path],
    # ) -> None:
    #     """
    #     Loads attributes from a Data_roicat object from a pickle file.

    #     Args:
    #         path_load (Union[str, Path]): 
    #             Path to the pickle file.

    #     Note: 
    #         After calling this method, the attributes of this object are updated with those 
    #         loaded from the pickle file. If an object in the pickle file is a dictionary, 
    #         the object's attributes are set directly from the dictionary. Otherwise, if 
    #         the object in the pickle file has a 'import_from_dict' method, it is used 
    #         to load attributes. If it does not, the attributes are directly loaded from 
    #         the object's `__dict__` attribute.

    #     Example:
    #         .. highlight:: python
    #         .. code-block:: python

    #             obj = Data_roicat()
    #             obj.load('/path/to/pickle/file')
    #     """
    #     from pathlib import Path
    #     assert Path(path_load).exists(), f"RH ERROR: File does not exist: {path_load}."
    #     obj = helpers.pickle_load(path_load)
    #     assert isinstance(obj, (type(self), dict)), f"RH ERROR: Loaded object is not a Data_roicat object or dictionary. Loaded object is of type {type(obj)}."

    #     if isinstance(obj, dict):
    #         ## Set attributes from dict
    #         ### If the subclass has a load_from_dict method, use that.
    #         if hasattr(self, 'import_from_dict'):
    #             self.import_from_dict(obj)
    #         else:
    #             for key, val in obj.items():
    #                 setattr(self, key, val)
    #     else:
    #         ## Set attributes from object
    #         for key, val in obj.__dict__.items():
    #             setattr(self, key, val)

    #     print(f"Loaded Data_roicat object from {path_load}.") if self._verbose else None


    def _locals_to_params(
        self,
        locals_dict: Dict[str, Any],
        keys: List[str],
    ) -> None:
        """
        Returns a dictionary of the local variables with the specified keys.

        Args:
            locals_dict (Dict[str, Any]): 
                Dictionary of local variables.
            keys (List[str]): 
                List of keys to extract from the local variables.
        """
        def safe_getitem(d, key):
            try:
                return d[key]
            except KeyError:
                warnings.warn(f'RH WARNING: key={key} not found in locals_dict. Skipping.')

        return {key: safe_getitem(locals_dict, key) for key in keys}


class RichFile_ROICaT(rf.RichFile):
    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        check: Optional[bool] = True,
        safe_save: Optional[bool] = True,
    ):
        super().__init__(path=path, check=check, safe_save=safe_save)


        ## NUMPY ARRAY
        import numpy as np

        def save_npy_array(
            obj: np.ndarray,
            path: Union[str, Path],
            **kwargs,
        ) -> None:
            """
            Saves a NumPy array to the given path.
            """
            np.save(path, obj, **kwargs)

        def load_npy_array(
            path: Union[str, Path],
            **kwargs,
        ) -> np.ndarray:
            """
            Loads an array from the given path.
            """    
            return np.load(path, **kwargs)
        

        ## SCIPY SPARSE MATRIX
        import scipy.sparse

        def save_sparse_array(
            obj: scipy.sparse.spmatrix,
            path: Union[str, Path],
            **kwargs,
        ) -> None:
            """
            Saves a SciPy sparse matrix to the given path.
            """
            scipy.sparse.save_npz(path, obj, **kwargs)

        def load_sparse_array(
            path: Union[str, Path],
            **kwargs,
        ) -> scipy.sparse.csr_matrix:
            """
            Loads a sparse array from the given path.
            """        
            return scipy.sparse.load_npz(path, **kwargs)
        

        ## JSON DICT
        import collections
        import json

        def save_json_dict(
            obj: collections.UserDict,
            path: Union[str, Path],
            **kwargs,
        ) -> None:
            """
            Saves a dictionary to the given path.
            """
            with open(path, 'w') as f:
                json.dump(dict(obj), f, **kwargs)

        def load_json_dict(
            path: Union[str, Path],
            **kwargs,
        ) -> collections.UserDict:
            """
            Loads a dictionary from the given path.
            """
            with open(path, 'r') as f:
                return JSON_Dict(json.load(f, **kwargs))


        ## JSON LIST   
        def save_json_list(
            obj: collections.UserList,
            path: Union[str, Path],
            **kwargs,
        ) -> None:
            """
            Saves a list to the given path.
            """
            with open(path, 'w') as f:
                json.dump(list(obj), f, **kwargs)

        def load_json_list(
            path: Union[str, Path],
            **kwargs,
        ) -> collections.UserList:
            """
            Loads a list from the given path.
            """
            with open(path, 'r') as f:
                return JSON_List(json.load(f, **kwargs))
            

        ## OPTUNA STUDY
        import optuna
        import pickle

        ## load and save functions for optuna study
        def save_optuna_study(
            obj: optuna.study.Study,
            path: Union[str, Path],
            **kwargs,
        ) -> None:
            """
            Saves an Optuna study to the given path.
            """
            with open(path, 'wb') as f:
                pickle.dump(obj, f, **kwargs)

        def load_optuna_study(
            path: Union[str, Path],
            **kwargs,
        ) -> optuna.study.Study:
            """
            Loads an Optuna study from the given path.
            """
            with open(path, 'rb') as f:
                return pickle.load(f, **kwargs)
            
        
        ## TORCH TENSOR
        import torch

        def save_torch_tensor(
            obj: torch.Tensor,
            path: Union[str, Path],
            **kwargs,
        ) -> None:
            """
            Saves a PyTorch tensor to the given path as a NumPy array.
            """
            np.save(path, obj.detach().cpu().numpy(), **kwargs)

        def load_torch_tensor(
            path: Union[str, Path],
            **kwargs,
        ) -> torch.Tensor:
            """
            Loads a PyTorch tensor from the given path.
            """
            return torch.from_numpy(np.load(path, **kwargs))


        ## REPR
        def save_repr(
            obj: object,
            path: Union[str, Path],
            **kwargs,
        ) -> None:
            """
            Saves the repr of an object to the given path.
            """
            with open(path, 'w') as f:
                f.write(repr(obj))

        def load_repr(
            path: Union[str, Path],
            **kwargs,
        ) -> object:
            """
            Loads the repr of an object from the given path.
            """
            with open(path, 'r') as f:
                return f.read()

        import hdbscan

        
        ## PANDAS DATAFRAME
        import pandas as pd
        
        def save_pandas_dataframe(
            obj: pd.DataFrame,
            path: Union[str, Path],
            **kwargs,
        ) -> None:
            """
            Saves a Pandas DataFrame to the given path.
            """
            ## Save as a CSV file
            obj.to_csv(path, index=True, **kwargs)

        def load_pandas_dataframe(
            path: Union[str, Path],
            **kwargs,
        ) -> pd.DataFrame:
            """
            Loads a Pandas DataFrame from the given path.
            """
            ## Load as a CSV file
            return pd.read_csv(path, index_col=0, **kwargs)

        roicat_module_tds = [rf.functions.Container(
            type_name=type_name,
            object_class=object_class,
            suffix="roicat",
            library="roicat",
            versions_supported=[">=1.1", "<2"],
        ) for type_name, object_class in [
            # ("data_suite2p", data_importing.Data_suite2p),
            # ("data_caiman", data_importing.Data_caiman),
            # ("data_roiextractors", data_importing.Data_roiextractors),
            # ("data_roicat", data_importing.Data_roicat),
            # ("aligner", alignment.Aligner),
            # ("blurrer", blurring.ROI_Blurrer),
            # ("roinet", ROInet.ROInet_embedder),
            # ("swt", scatteringWaveletTransformer.SWT),
            # ("similarity_graph", similarity_graph.ROI_graph),
            # ("clusterer", clustering.Clusterer),

            ("toeplitz_conv", helpers.Toeplitz_convolution2d),
            ("convergence_checker_optuna", helpers.Convergence_checker_optuna),
            ("image_alignment_checker", helpers.ImageAlignmentChecker),
        ]]
        # roicat_module_tds = []
        

        type_dicts = [
            {
                "type_name":          "numpy_array",
                "function_load":      load_npy_array,
                "function_save":      save_npy_array,
                "object_class":       np.ndarray,
                "suffix":             "npy",
                "library":            "numpy",
                "versions_supported": [],
            },
            {
                "type_name":          "numpy_scalar",
                "function_load":      load_npy_array,
                "function_save":      save_npy_array,
                "object_class":       np.number,
                "suffix":             "npy",
                "library":            "numpy",
                "versions_supported": [],
            },
            {
                "type_name":          "scipy_sparse_array",
                "function_load":      load_sparse_array,
                "function_save":      save_sparse_array,
                "object_class":       scipy.sparse.spmatrix,
                "suffix":             "npz",
                "library":            "scipy",
                "versions_supported": [],
            },
            {
                "type_name":          "json_dict",
                "function_load":      load_json_dict,
                "function_save":      save_json_dict,
                "object_class":       JSON_Dict,
                "suffix":             "json",
                "library":            "python",
                "versions_supported": [],
            },
            {
                "type_name":          "json_list",
                "function_load":      load_json_list,
                "function_save":      save_json_list,
                "object_class":       JSON_List,
                "suffix":             "json",
                "library":            "python",
                "versions_supported": [],
            },
            {
                "type_name":          "optuna_study",
                "function_load":      load_optuna_study,
                "function_save":      save_optuna_study,
                "object_class":       optuna.study.Study,
                "suffix":             "optuna",
                "library":            "optuna",
                "versions_supported": [],
            },
            {
                "type_name":          "torch_tensor",
                "function_load":      load_torch_tensor,
                "function_save":      save_torch_tensor,
                "object_class":       torch.Tensor,
                "suffix":             "npy",
                "library":            "torch",
                "versions_supported": [],
            },
            {
                "type_name":          "model_swt",
                "function_load":      load_repr,
                "function_save":      save_repr,
                "object_class":       Model_SWT,
                "suffix":             "swt",
                "library":            "onnx2torch",
                "versions_supported": [],
            },
            {
                "type_name":          "torch_module",
                "function_load":      load_repr,
                "function_save":      save_repr,
                "object_class":       torch.nn.Module,
                "suffix":             "torch_module",
                "library":            "torch",
                "versions_supported": [],
            },
            {
                "type_name":          "torch_sequence",
                "function_load":      load_repr,
                "function_save":      save_repr,
                "object_class":       torch.nn.Sequential,
                "suffix":             "torch_sequence",
                "library":            "torch",
                "versions_supported": [],
            },
            {
                "type_name":          "torch_dataset",
                "function_load":      load_repr,
                "function_save":      save_repr,
                "object_class":       torch.utils.data.Dataset,
                "suffix":             "torch_dataset",
                "library":            "torch",
                "versions_supported": [],
            },
            {
                "type_name":          "torch_dataloader",
                "function_load":      load_repr,
                "function_save":      save_repr,
                "object_class":       torch.utils.data.DataLoader,
                "suffix":             "torch_dataloader",
                "library":            "torch",
                "versions_supported": [],
            },
            {
                "type_name":          "hdbscan",
                "function_load":      load_repr,
                "function_save":      save_repr,
                "object_class":       hdbscan.HDBSCAN,
                "suffix":             "hdbscan",
                "library":            "torch",
                "versions_supported": [],
            },
            {
                "type_name":          "pandas_dataframe",
                "function_load":      load_pandas_dataframe,
                "function_save":      save_pandas_dataframe,
                "object_class":       pd.DataFrame,
                "suffix":             "csv",
                "library":            "pandas",
                "versions_supported": [],
            },
        ] + [t.get_property_dict() for t in roicat_module_tds]

        [self.register_type_from_dict(d) for d in type_dicts]
        

######################################
######## CUSTOM DATA CLASSES #########
######################################

class JSON_Dict(dict):
    def __init__(self, *args, **kwargs):
        super(JSON_Dict, self).__init__(*args, **kwargs)
class JSON_List(list):
    def __init__(self, *args, **kwargs):
        super(JSON_List, self).__init__(*args, **kwargs)

## Wrapper for SWT
class Model_SWT(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super(Model_SWT, self).__init__()
        self.add_module('model', model)
    def forward(self, x):
        return self.model(x)


def make_session_bool(n_roi: np.ndarray,) -> np.ndarray:
    """
    Generates a boolean array representing ROIs (Region Of Interest) per session from an array of ROI counts.

    Args:
        n_roi (np.ndarray): 
            Array representing the number of ROIs per session. 
            *shape*: *(n_sessions,)*

    Returns:
        (np.ndarray): 
            session_bool (np.ndarray): 
                Boolean array of shape *(n_roi_total, n_session)* where each column represents a session 
                and each row corresponds to an ROI.
                
    Example:
        .. highlight:: python
        .. code-block:: python

            n_roi = np.array([3, 4, 2])
            session_bool = make_session_bool(n_roi)
    """
    n_roi_total = np.sum(n_roi)
    r = np.arange(n_roi_total, dtype=np.int64)
    n_roi_cumsum = np.concatenate([[0], np.cumsum(n_roi)])
    session_bool = np.vstack([(b_lower <= r) * (r < b_upper) for b_lower, b_upper in zip(n_roi_cumsum[:-1], n_roi_cumsum[1:])]).T
    return session_bool


def split_iby_session(
    x: Any,
    n_roi_per_session: Union[np.ndarray, List[int]],
):
    """
    Splits an array or iterable into a list of arrays or iterables based on the
    number of ROIs per session.

    Args:
        arr (Any): 
            Array to split.
        n_roi_per_session (Union[np.ndarray, List[int]]): 
            Number of ROIs per session.

    Returns:
        (List[Any]): 
            List of arrays split by session.
    """
    return [x[sum(n_roi_per_session[:ii]):sum(n_roi_per_session[:ii+1])] for ii in range(len(n_roi_per_session))]

##########################################################################################################################
############################################### UCID handling ############################################################
##########################################################################################################################

def check_dataStructure__list_ofListOrArray_ofDtype(
    lolod: Union[List[List[Union[int, float]]], List[np.ndarray]], 
    dtype: Type = np.int64, 
    fix: bool = True, 
    verbose: bool = True,
) -> Union[List[List[Union[int, float]]], List[np.ndarray]]:
    """
    Verifies and optionally corrects the data structure of 'lolod' (list of list
    of dtype).
    
    The structure should be a list of lists of dtypes or a list of numpy arrays
    of dtypes.

    Args:
        lolod (Union[List[List[Union[int, float]]], List[np.ndarray]]): 
            * The data structure to check. It should be a list of lists of
              dtypes or a list of numpy arrays of dtypes.
        
        dtype (Type): 
            * The expected dtype of the elements in 'lolod'. (Default is
              ``np.int64``)

        fix (bool): 
            * If ``True``, attempts to correct the data structure if it is not
              as expected. The corrections are as follows: \n
                * If 'lolod' is an array, it will be cast to [lolod]
                * If 'lolod' is a numpy object, it will be cast to
                  [np.array(lolod, dtype=dtype)]
                * If 'lolod' is a list of lists of numbers (int or float), it
                  will be cast to [np.array(lod, dtype=dtype) for lod in lolod]
                * If 'lolod' is a list of arrays of wrong dtype, it will be cast
                  to [np.array(lod, dtype=dtype) for lod in lolod] \n
            * If ``False``, raises an error if the structure is not as expected.
              (Default is ``True``)

        verbose (bool): 
            * If ``True``, prints warnings when the structure is not as expected
              and is corrected. (Default is ``True``)

    Returns:
        (Union[List[List[Union[int, float]]], List[np.ndarray]]): 
            lolod (Union[List[List[Union[int, float]]], List[np.ndarray]]):
                The verified or corrected data structure.
    """
    ## switch case for if it is a list or np.ndarray
    if isinstance(lolod, list):
        ## switch case for if the elements are lists or np.ndarray or numbers (int or float) or dtypes
        if all([isinstance(lod, list) for lod in lolod]):
            ## switch case for if the elements are numbers (int or float) or dtype or other
            if all([all([isinstance(l, (int, float, np.integer, np.floating)) for l in lod]) for lod in lolod]):
                if fix:
                    print(f'ROICaT WARNING: lolod is a list of lists of numbers (int or float). Converting to np.ndarray.') if verbose else None
                    lolod = [np.array(lod, dtype=dtype) for lod in lolod]
                else:
                    raise ValueError(f'ROICaT ERROR: lolod is a list of lists of numbers (int or float).')
            elif all([all([isinstance(l, dtype) for l in lod]) for lod in lolod]):
                pass
            else:
                raise ValueError(f'ROICaT ERROR: lolod is a list of lists, but the elements are not all numbers (int or float) or dtype.')
        elif all([isinstance(lod, np.ndarray) for lod in lolod]):
            ## switch case for if the elements are numbers (any non-object numpy dtype) or dtype or other
            if all([all([np.issubdtype(lod.dtype, dtype) for lod in lolod])]):
                pass
            if all([all([not np.issubdtype(lod.dtype, np.object_) for lod in lolod])]):
                if fix:
                    print(f'ROICaT WARNING: lolod is a list of np.ndarray of numbers (int or float). Converting to np.ndarray.') if verbose else None
                    lolod = [np.array(lod, dtype=dtype) for lod in lolod]
                else:
                    raise ValueError(f'ROICaT ERROR: lolod is a list of np.ndarray of numbers (int or float).')
            else:
                raise ValueError(f'ROICaT ERROR: lolod is a list of np.ndarray, but the elements are not all numbers (int or float) or dtype.')
        elif all([isinstance(lod, (int, float)) for lod in lolod]):
            if fix:
                print(f'ROICaT WARNING: lolod is a list of numbers (int or float). Converting to np.ndarray.') if verbose else None
                lolod = [np.array(lod, dtype=dtype) for lod in lolod]
            else:
                raise ValueError(f'ROICaT ERROR: lolod is a list of numbers (int or float).')
        elif all([isinstance(lod, dtype) for lod in lolod]):
            if fix:
                print(f'ROICaT WARNING: lolod is a list of dtype. Converting to np.ndarray.') if verbose else None
                lolod = [np.array(lolod, dtype=dtype)]
            else:
                raise ValueError(f'ROICaT ERROR: lolod is a list of dtype.')
        else:
            raise ValueError(f'ROICaT ERROR: lolod is a list, but the elements are not all lists or np.ndarray or numbers (int or float).')
    elif isinstance(lolod, np.ndarray):
        ## switch case for if the elements are numbers (any non-object numpy dtype) or dtype or other
        if np.issubdtype(lolod.dtype, dtype):
            if fix:
                print(f'ROICaT WARNING: lolod is a np.ndarray of dtype. Converting to list of np.ndarray of dtype.') if verbose else None
                lolod = [lolod]
        elif not np.issubdtype(lolod.dtype, np.object_):
            if fix:
                print(f'ROICaT WARNING: lolod is a np.ndarray of numbers (int or float). Converting to list of np.ndarray of dtype.') if verbose else None
                lolod = [np.array(lolod, dtype=dtype)]
            else:
                raise ValueError(f'ROICaT ERROR: lolod is a np.ndarray of numbers (int or float).')
        else:
            raise ValueError(f'ROICaT ERROR: lolod is a np.ndarray, but the elements are not all numbers (int or float) or dtype.')
    else:
        raise ValueError(f'ROICaT ERROR: lolod is not a list or np.ndarray.')

    return lolod


def mask_UCIDs_with_iscell(
    ucids: List[Union[List[int], np.ndarray]], 
    iscell: List[Union[List[bool], np.ndarray]]
) -> List[Union[List[int], np.ndarray]]:
    """
    Masks the UCIDs with the **iscell** array. If ``iscell`` is False, then the
    UCID is set to -1.

    Args:
        ucids (List[Union[List[int], np.ndarray]]): 
            List of lists of UCIDs for each session.\n
            Shape outer list: *(n_sessions,)*\n
            Shape inner list: *(n_roi_in_session,)*
        iscell (List[Union[List[bool], np.ndarray]]): 
            List of lists of boolean indicators for each UCID.\n
            ``True`` means that ROI is a cell, ``False`` means that ROI is not a
            cell.\n
            Shape outer list: *(n_sessions,)*\n
            Shape inner list: *(n_roi_in_session,)*

    Returns:
        (List[Union[List[int], np.ndarray]]): 
            ucids_out (List[Union[List[int], np.ndarray]]): 
                Masked list of lists of UCIDs. Elements that are not cells are
                set to -1 in each session.
    """
    ucids_out = copy.deepcopy(ucids)
    ucids_out = check_dataStructure__list_ofListOrArray_ofDtype(
        lolod=ucids_out,
        dtype=np.int64,
        fix=True,
        verbose=False,
    )
    iscell = check_dataStructure__list_ofListOrArray_ofDtype(
        lolod=iscell,
        dtype=bool,
        fix=True,
        verbose=False,
    )

    n_sesh = len(ucids)
    for i_sesh in range(n_sesh):
        ucids_out[i_sesh][~iscell[i_sesh]] = -1
    
    return ucids_out


def mask_UCIDs_by_label(
    ucids: List[Union[List[int], np.ndarray]],
    labels: Union[List[int], np.ndarray],
) -> List[Union[List[int], np.ndarray]]:
    """
    Sets labels in the UCIDs to -1 if they are not present in the **labels**
    array.\n
    RH 2024

    Args:
        ucids (List[Union[List[int], np.ndarray]]): 
            List of lists of UCIDs for each session.\n
            Shape outer list: *(n_sessions,)*\n
            Shape inner list: *(n_roi_in_session,)*
        labels (Union[List[int], np.ndarray]): 
            Array of labels to keep. All other labels are set to -1.
            Shape: *(n_labels,)*

    Returns:
        (List[Union[List[int], np.ndarray]]): 
            ucids_out (List[Union[List[int], np.ndarray]]): 
                Masked list of lists of UCIDs. Elements that are not in the
                **labels** array are set to -1 in each session.

    Example:
        .. highlight:: python
        .. code-block:: python

        ucids = [[1, 2, 3], [2, -1, 4], [3, 0, 5]]
        labels = [2, 3]
        ucids_out = mask_UCIDs_by_label(ucids, labels)
        # ucids_out = [[-1, 2, 3], [2, -1, -1], [3, -1, -1]]
    """
    ucids_out = copy.deepcopy(ucids)
    ucids_out = check_dataStructure__list_ofListOrArray_ofDtype(
        lolod=ucids_out,
        dtype=np.int64,
        fix=True,
        verbose=False,
    )
    labels = np.array(labels, dtype=np.int64)

    iscell = [np.isin(u_sesh, labels) for u_sesh in ucids_out]
    ucids_out = mask_UCIDs_with_iscell(ucids_out, iscell)

    return ucids_out


def discard_UCIDs_with_fewer_matches(
    ucids: List[Union[List[int], np.ndarray]], 
    n_sesh_thresh: Union[int, str] = 'all',
    verbose: bool = True
) -> List[Union[List[int], np.ndarray]]:
    """
    Discards UCIDs that do not appear in at least **n_sesh_thresh** sessions. If
    ``n_sesh_thresh='all'``, then only UCIDs that appear in all sessions are
    kept.

    Args:
        ucids (List[Union[List[int], np.ndarray]]): 
            List of lists of UCIDs for each session.
        n_sesh_thresh (Union[int, str]): 
            Number of sessions that a UCID must appear in to be kept. If
            ``'all'``, then only UCIDs that appear in all sessions are kept.
            (Default is ``'all'``)
        verbose (bool): 
            If ``True``, print verbose output. (Default is ``True``)

    Returns:
        (List[Union[List[int], np.ndarray]]): 
            ucids_out (List[Union[List[int], np.ndarray]]): 
                List of lists of UCIDs with UCIDs that do not appear in at least
                **n_sesh_thresh** sessions set to -1.
    """
    ucids_out = copy.deepcopy(ucids)
    ucids_out = check_dataStructure__list_ofListOrArray_ofDtype(
        lolod=ucids_out,
        dtype=np.int64,
        fix=True,
        verbose=False,
    )
    
    n_sesh = len(ucids)
    n_sesh_thresh = n_sesh if n_sesh_thresh == 'all' else n_sesh_thresh
    assert isinstance(n_sesh_thresh, int)
    
    ucids_unique = np.unique(np.concatenate(ucids_out, axis=0))
    ucids_inAllSesh = [u for u in ucids_unique if np.array([np.isin(u, u_sesh) for u_sesh in ucids_out]).sum() >= n_sesh_thresh]
    if verbose:
        fraction = (np.unique(ucids_inAllSesh) >= 0).sum() / (ucids_unique >= 0).sum()
        print(f'INFO: {fraction*100:.2f}% of UCIDs that appear in at least {n_sesh_thresh} sessions.')
    ucids_out = [[val * np.isin(val, ucids_inAllSesh) - np.logical_not(np.isin(val, ucids_inAllSesh)) for val in u] for u in ucids_out]
    
    return ucids_out
    

def squeeze_UCID_labels(
    ucids: List[Union[List[int], np.ndarray]],
    return_array: bool = False,
) -> List[Union[List[int], np.ndarray]]:
    """
    Squeezes the UCID labels. Finds all the unique UCIDs across all sessions,
    then removes spaces in the UCID labels by mapping the unique UCIDs to new
    values. Output UCIDs are contiguous integers starting at 0, and maintains
    elements with UCID=-1.

    Args:
        ucids (List[Union[List[int], np.ndarray]]): 
            List of lists of UCIDs for each session.
        return_array (bool):
            If ``True``, then the output will be a numpy array.
            (Default is ``False``)

    Returns:
        (List[Union[List[int], np.ndarray]]): 
            ucids_out (List[Union[List[int], np.ndarray]]): 
                List of lists of UCIDs with UCIDs that do not appear in at least
                **n_sesh_thresh** sessions set to -1.
    """
    ucids_out = copy.deepcopy(ucids)
    ucids_out = check_dataStructure__list_ofListOrArray_ofDtype(
        lolod=ucids_out,
        dtype=np.int64,
        fix=True,
        verbose=False,
    )

    uniques_all = np.unique(np.concatenate(ucids_out, axis=0))
    uniques_all = np.sort(uniques_all[uniques_all >= 0])
    ## make a mapping of the unique values to new values
    # mapping = {old: new for old, new in zip(uniques_all, helpers.squeeze_integers(uniques_all))}
    mapping = {old: new for old, new in zip(uniques_all, np.arange(len(uniques_all)))}
    mapping.update({-1: -1})
    ## apply the mapping to the data
    n_sesh = len(ucids_out)
    for i_sesh in range(n_sesh):
        ucids_out[i_sesh] = [int(mapping[val]) for val in ucids_out[i_sesh]]

    if not return_array:
        return ucids_out
    else:
        return [np.array(u) for u in ucids_out]


def match_arrays_with_ucids(
    arrays: Union[np.ndarray, List[np.ndarray]], 
    ucids: Union[List[np.ndarray], List[List[int]]], 
    return_indices: bool = False,
    squeeze: bool = False,
    force_sparse: bool = False,
    prog_bar: bool = False,
) -> List[Union[np.ndarray, scipy.sparse.lil_matrix]]:
    """
    Matches the indices of the arrays using the UCIDs. Array indices with UCIDs
    corresponding to -1 are set to ``np.nan``. This is useful for aligning
    Fluorescence and Spiking data across sessions using UCIDs.

    Args:
        arrays (Union[np.ndarray, List[np.ndarray]]): 
            List of numpy arrays for each session. Matching is done along the
            first dimension.
        ucids (Union[List[np.ndarray], List[List[int]]]): 
            List of lists of UCIDs for each session.
        return_indices (bool):
            If ``True``, then the indices of the UCIDs will also be returned.
            The indices will be of dtype np.float32 because it may contain NaNs.
            (Default is ``False``)
        squeeze (bool): 
            If ``True``, then UCIDs are squeezed to be contiguous integers.
            (Default is ``False``)
        force_sparse (bool):
            If ``True``, then the output will be a list of sparse matrices.
            (Default is ``False``)
        prog_bar (bool):
            If ``True``, then a progress bar will be displayed. (Default is
            ``False``)

    Returns:
        (List[Union[np.ndarray, scipy.sparse.lil_matrix]]): 
            arrays_out (List[Union[np.ndarray, scipy.sparse.lil_matrix]]): 
                List of arrays for each session. Array indices with UCIDs
                corresponding to -1 are set to ``np.nan``. Each array will have
                shape: *(n_ucids if squeeze==True OR max_ucid if squeeze==False,
                *array.shape[1:])*. UCIDs will be used as the index of the first
                dimension.
    """
    import scipy.sparse

    arrays = [arrays] if not isinstance(arrays, list) else arrays

    ucids_tu = check_dataStructure__list_ofListOrArray_ofDtype(
        lolod=ucids,
        dtype=np.int64,
        fix=True,
        verbose=False,
    )
    ## Error if dtype is not NaN compatible
    if not np.issubdtype(arrays[0].dtype, np.floating):
        raise ValueError(f'ROICaT ERROR: This function requires inputs to be of a dtype that is compatible with NaNs, like np.floating types: np.float32, np.float64, etc.')
    ## Squeeze UCIDs
    ucids_tu = squeeze_UCID_labels(ucids_tu) if squeeze else ucids_tu
    # max_ucid = (np.unique(np.concatenate(ucids_tu, axis=0)) >= 0).max()
    max_ucid = (np.unique(np.concatenate(ucids_tu, axis=0))).max().astype(int) + 1

    dicts_ucids = [{u: i for i, u in enumerate(u_sesh)} for u_sesh in ucids_tu]
    
    ## make ndarrays filled with np.nan for each session
    if isinstance(arrays[0], np.ndarray) and not force_sparse:
        arrays_out = [np.full((max_ucid, *a.shape[1:]), np.nan, dtype=arrays[0].dtype) for a in arrays]
    elif scipy.sparse.issparse(arrays[0]) or force_sparse:
        arrays_out = [scipy.sparse.lil_matrix((max_ucid, *a.shape[1:]), dtype=a.dtype) for a in arrays]
    else:
        raise ValueError(f'ROICaT ERROR: arrays[0] is not a numpy array or scipy.sparse matrix.')
    ## fill in the arrays with the data
    n_sesh = len(arrays)
    for i_sesh in tqdm(range(n_sesh), disable=not prog_bar):
        for u, idx in dicts_ucids[i_sesh].items():
            if u >= 0:
                arrays_out[i_sesh][u] = arrays[i_sesh][idx]

    if not return_indices:
        return arrays_out
    else:
        return arrays_out, match_arrays_with_ucids(
            arrays=[np.arange(len(a), dtype=np.float32) for a in arrays],
            ucids=ucids,
            return_indices=False,
            squeeze=squeeze,
            force_sparse=False,
            prog_bar=False,
        )

def match_arrays_with_ucids_inverse(
    arrays: Union[np.ndarray, List[np.ndarray]], 
    ucids: Union[List[np.ndarray], List[List[int]]],
    unsqueeze: bool = True,
) -> List[Union[np.ndarray, scipy.sparse.lil_matrix]]:
    """
    Inverts the matching of the indices of the arrays using the UCIDs. Arrays
    should have indices that correspond to the UCID values. The return will be a
    list of arrays with indices that correspond to the original indices of the
    arrays / ucids. Essentially, this function undoes the matching done by
    match_arrays_with_ucids().

    Args:
        arrays (Union[np.ndarray, List[np.ndarray]]): 
            List of numpy arrays for each session.
        ucids (Union[List[np.ndarray], List[List[int]]]): 
            List of lists of UCIDs for each session.
        unsqueeze (bool):
            If ``True``, then this algorithm assumes that the arrays were
            squeezed to remove unused UCIDs. This corresponds to and should
            match the argument ``squeeze`` used in match_arrays_with_ucids().

    Returns:
        (List[Union[np.ndarray, scipy.sparse.lil_matrix]]): 
            arrays_out (List[Union[np.ndarray, scipy.sparse.lil_matrix]]): 
                List of arrays with indices that correspond to the original
                indices of the arrays / ucids. 
    """
    arrays = [arrays] if not isinstance(arrays, list) else arrays

    ## Make a mapping of the UCIDs to the original indices ('aranges_matched')
    ucids_clean = copy.deepcopy(ucids)
    ucids_clean = check_dataStructure__list_ofListOrArray_ofDtype(
        lolod=ucids_clean,
        dtype=np.float32,
        fix=True,
        verbose=False,
    )
    aranges = [np.arange(len(u), dtype=np.float32) for u in ucids_clean]
    aranges_matched = match_arrays_with_ucids(
        arrays=aranges,
        ucids=ucids_clean,
        squeeze=False,
    )

    ## Make sure that unsqueeze is consistent with the arrays
    flag_same_len = all([len(u) == len(a) for u, a in zip(aranges_matched, arrays)])
    if unsqueeze == False:
        assert flag_same_len == True
    else:
        assert flag_same_len == False
        
    # Unsqueeze arrays
    if unsqueeze:
        idx_unsq = [(np.cumsum(~np.isnan(a)) - 1).astype(np.float32) for a in aranges_matched]
        for ii, a in enumerate(aranges_matched):
            idx_unsq[ii][np.isnan(a)] = np.nan
        arrays_unsq = [helpers.index_with_nans(a, idx) for a, idx in zip(arrays, idx_unsq)]
    else:
        arrays_unsq = arrays

    ## Invert the matching
    def negOne_to_nan(x):
        tmp = np.array(x, dtype=np.float32)
        np.place(arr=tmp, mask=tmp == -1, vals=np.nan)
        return tmp
    ucids_clean_nan = [negOne_to_nan(u) for u in ucids_clean]
    arrays_inv = [helpers.index_with_nans(a, o) for a, o in zip(arrays_unsq, ucids_clean_nan)]
    
    return arrays_inv
    

def labels_to_labelsBySession(labels, n_roi_bySession):
    """
    Converts a list of labels to a list of lists of labels by session.
    RH 2024

    Args:
        labels (list or np.ndarray): 
            List of labels.
        n_roi_bySession (list or np.ndarray): 
            Number of ROIs by session.

    Returns:
        (list): 
            List of lists of labels by session.
    """
    assert isinstance(labels, (list, np.ndarray)), f'labels is not a list or np.ndarray. labels={labels}'
    assert isinstance(n_roi_bySession, (list, np.ndarray)), f'n_roi_bySession is not a list or np.ndarray. n_roi_bySession={n_roi_bySession}'
    labels = np.array(labels)
    n_roi_bySession = np.array(n_roi_bySession, dtype=np.int64)
    assert labels.ndim == 1, f'labels.ndim={labels.ndim}, but should be 1.'
    assert n_roi_bySession.ndim == 1, f'n_roi_bySession.ndim={n_roi_bySession.ndim}, but should be 1.'

    assert np.sum(n_roi_bySession) == len(labels), f'np.sum(n_roi_bySession)={np.sum(n_roi_bySession)} != len(labels)={len(labels)}'

    labels_bySession = split_iby_session(x=labels, n_roi_per_session=n_roi_bySession)

    return labels_bySession