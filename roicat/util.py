from pathlib import Path
import warnings
import copy
from typing import Dict, Any, Optional, Union, List, Tuple, Callable, Iterable, Iterator, Type

import importlib

import numpy as np
import scipy.sparse

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

def make_params_default_tracking(dir_networkFiles: Optional[str] = None,) -> Dict:
    """
    Generates a dictionary of default parameters for a standard tracking
    pipeline.

    Args:
        dir_networkFiles (Optional[str]):
            Directory where the ROInet network files are stored. If ``None``,
            uses the current directory. (Default is ``None``)

    Returns:
        (Dict):
            params (Dict):
                Dictionary of default parameters.
    """
    dir_networkFiles = str(Path.cwd() / 'network_files') if dir_networkFiles is None else dir_networkFiles

    params = {
        'paths': {
            # 'dir_allOuterFolders': r"/home/rich/data/folder_containing_folders_containing_suite2p_output_files",  ## directory where directories containing below 'pathSuffixTo...' are
            # 'pathSuffixToStat': 'stat.npy',  ## path suffix to where the stat.npy file is
            # 'pathSuffixToOps': 'ops.npy',  ## path suffix to where the ops.npy file is
            # 'dir_save': r'/home/rich/data/roicat_results/',  ## default: None. Directory to save output file to. If None then saves in dir_allOuterFolders.
            # 'filenamePrefix_save': None,  ##  default: None. Filename prefix to save results to. If None then just uses the dir_allOuterFolders.name.
        },
        'importing': {
            'data_verbose': True,  ## default: True. Whether to print out data importing information
            'out_height_width': [36, 36],  ## default: [36,36]. Height and width of small cropped output images of each ROI. Check how large your ROIs are in pixels.
            'max_footprint_width': 1025,  ## default: 1025. Maximum length of a spatial footprint. If you get an error during importing, try increasing this value.
            'type_meanImg': 'meanImgE',  ## default: 'meanImgE'. Type of mean image to use for normalization. This is just a field in the ops.npy file.
            'um_per_pixel': 2.0,  ## default: 1.0. Number of microns per pixel for the imaging dataset. Doesn't need to be exact. Used for resizing the ROIs. Check the images of the resized ROIs to tweak.
            'new_or_old_suite2p': 'new',  ## default: 'new'. If using suite2p, this specifices whether the stat.npy file is in the old MATLAB format or new Python format.
            'FOV_images': None,  ## default: None. Set to None if you want to use the images extracted from Suite2p
            'centroid_method': 'centerOfMass',  ## default: 'centerOfMass'. Method to use for calculating the centroid of the ROI. 'centerOfMass' or 'median' available.
            'FOV_height_width': None,  ## default: None. Set to None if you want to use the images extracted from Suite2p. Otherwise, set to [height, width] of the FOV.
            'verbose': True,  ## default: True. Whether to print out importing information
        },
        'alignment': {
            'do_phaseCorrReg': True,  ## default: True. If you are having issues with alignment due to big movements of the FOV. Try setting this to False.
            'session_template': 0.5,  ## default: 0.5. Which session to use as a registration template. If input is float (ie 0.0, 0.5, 1.0, etc.), then it is the fractional position of the session to use; if input is int (ie 1, 2, 3), then it is the index of the session to use (0-indexed)
            'use_CLAHE': True,  ## default: False. Whether or not to use 'Contrast Limited Adaptive Histogram Equalization'. Useful if params['importing']['type_meanImg'] is not a contrast enhanced image (like 'meanImgE' in Suite2p)
            'CLAHE_nGrid': 10,  ## default 10. Unsed if 'use_CLAHE' is False. Defines how many blocks to split the FOV into to normalize the local contrast. See openCV's clahe function. Larger means more CLAHE.
            'phaseCorr': {
                'freq_highPass': 0.01,  ## default: 0.01. Spatial frequency upper-bound cut-off to use for phase correlation. Units in fraction of the height of the FOV. Spatial frequencies correlations higher than this will be set to zero.
                'freq_lowPass': 0.3,  ## default: 0.3. Spatial frequency lower-bound cut-off to use for phase correlation. Units in fraction of the height of the FOV. Spatial frequencies correlations lower than this will be set to zero.
                'template_method': 'sequential',  # default: 'sequential'. Either 'sequential' or 'image'. 'sequential' is better if there is significant drift over sessions. If 'sequential', then pcr.register(template=idx) where idx is the index of the image you want the shifts to be relative to. If 'image', then idx is FOVs[idx].
            },
            'nonrigid':{
                'method': 'createOptFlow_DeepFlow',  ## default: 'createOptFlow_DeepFlow'. Method to use for creating optical flow. 'calcOpticalFlowFarneback' and 'createOptFlow_DeepFlow' available.
                'kwargs_method': None,  ## default: None. Keyword arguments to pass to the cv2 optical flow method.
                'template_method': 'sequential',  ## default: 'image'. Either 'sequential' or 'image'. 'sequential' is better if there is significant drift over sessions.
                'return_sparse': True,  ## default: True. Whether to return a sparse matrix (True) or a dense matrix (False).
                'normalize': True,  ## default: True. If True, normalize the spatial footprints to have a sum of 1.
            },
        },
        'blurring': {
            'kernel_halfWidth': 2.0,  ## default: 2.0. Half-width of the cosine kernel used for blurring. Set value based on how much you think the ROIs move from session to session.
            'plot_kernel': False,  ## default: False. Whether to plot the kernel used for blurring.
        },
        'ROInet': {
            'device': 'cpu',  ## default: 'cpu'. Device to use for SWT. Recommend using a GPU (device='cuda').
            'hash_dict_true': {
                'params': ('params.json', '68cf1bd47130f9b6d4f9913f86f0ccaa'),
                'model': ('model.py', '61c85529b7aa33e0dfadb31ee253a7e1'),
                'state_dict': ('ConvNext_tiny__1_0_best__simCLR.pth', '3287e001ff28d07ada2ae70aa7d0a4da'),
            },
            'dir_networkFiles': '/home/rich/Downloads/ROInet',  ## local directory where network files are stored
            'download_from_gDrive': 'check_local_first',  ## default: 'check_local_first'. Whether to download the network files from Google Drive or to use the local files.
            'gDriveID': '1D2Qa-YUNX176Q-wgboGflW0K6un7KYeN',  ## default: '1FCcPZUuOR7xG-hdO6Ei6mx8YnKysVsa8'. Google Drive ID of the network files.
            'forward_pass_version': 'latent', # default: 'latent'. Leave as 'latent' for most things. Can be 'latent' (full pass through network), 'head' (output of the head layers), or 'base' (pass through just base layers)
            'verbose': True,  ## default: True. Whether to print out ROInet information.
            'pref_plot': False,  ## default: False. Whether to plot the ROI and the normalized ROI.
            'batchSize_dataloader': 8,  ## default: 8. Number of images to use for each batch.
            'pinMemory_dataloader': True,  ## default: True. Whether to pin the memory of the dataloader.
            'persistentWorkers_dataloader': True,  ## default: True. Whether to use persistent workers for the dataloader.
            'prefetchFactor_dataloader': 2,  ## default: 2. Number of prefetch factors to use for the dataloader.
        },
        'SWT': {
            'kwargs_Scattering2D': {'J': 2, 'L': 2},  ## default: {'J': 2, 'L': 2}. Keyword arguments to pass to the kymatio Scattering2D function.
            'device': 'cpu',  ## default: 'cpu'. Device to use for SWT. Recommend using a GPU (device='cuda').
        }, 
        'similarity': {
            'spatialFootprint_maskPower': 1.0,  ## default: 1.0. This determines the power to take the ROI mask images to. Higher for more dependent on brightness, lower for more dependent on binary overlap.
            'n_workers': -1,  ## default: -1. Number of workers to use for similarity. Set to -1 to use all available workers.
            'block_height': 128,  ## default: 64. Maximum height of the FOV block bins to use for pairwise ROI similarity calculations. Use smaller values (16-64) if n_sessions is large (<12), else keep around (64-128)
            'block_width': 128,  ## default: 64. Maximum width of the FOV block bins to use for pairwise ROI similarity calculations. Use smaller values (16-64) if n_sessions is large (<12), else keep around (64-128)
            'algorithm_nearestNeigbors_spatialFootprints': 'brute',  ## default: 'brute'. Algorithm to use for nearest neighbors.
            'verbose': True,  ## default: True. Whether to print out similarity information.
            'normalization': {
                'k_max': 4000,  ## default: 4000. Maximum kNN distance to use for building a distribution of pairwise similarities for each ROI.
                'k_min': 150,  ## default: 150. Set around n_sessions*10. Minimum kNN distance to use for building a distribution of pairwise similarities for each ROI. 
                'algo_NN': 'kd_tree',  ## default: 'kd_tree'. Algorithm to use for the nearest neighbors search across positional distances of different ROIs center positions. 'kd_tree' seems to be fastest. See sklearn nearest neighbor documentation for details.
                'device': 'cpu',  ## default: 'cpu'. Device to use for SWT. Recommend using a GPU (device='cuda').
            },
        },
        ## Cluster
        'clustering': {
            'plot_pref': True,
            'auto_pruning':{
                'n_bins': 50,  ## default: 50. Number of bins to use for estimating the distributions for 'different' and 'same' pairwise similarities
                'find_parameters_automatically': True,  ## default: True. Use optuna automatic parameter searching to find the best values for 'kwargs_makeConjunctiveDistanceMatrix'
                'n_jobs': -1, ## default: 2. Number of jobs to use for the optuna parameter search. Large values or -1 can result in high memory usage.
                'kwargs_findParameters': {
                    'n_patience': 100,
                    'tol_frac': 0.05,
                    'max_trials': 350,
                    'max_duration': 60*10,
                    'verbose': False,
                },
                'bounds_findParameters': {
                    'power_SF': (0.3, 2),
                    'power_NN': (0.2, 2),
                    'power_SWT': (0.1, 1),
                    'p_norm': (-5, 5),
                    'sig_NN_kwargs_mu': (0, 0.5),
                    'sig_NN_kwargs_b': (0.05, 2),
                    'sig_SWT_kwargs_mu': (0, 0.5),
                    'sig_SWT_kwargs_b': (0.05, 2),
                },
            },
            'method': 'auto',  ## default: 'auto'. Can be 'hungarian', 'hdbscan', or 'auto'. If 'auto', then if n_sessions >=8 'hdbscan' will be used.
            'hdbscan':{  ## Used only if 'method' is 'hdbscan'
                'min_cluster_size': 2,  ## default: 2. Best practice is to keep at 2 because issues can occur otherwise. Just manually throw out clusters with fewer ROIs if needed.
                'alpha': 0.999,  ## default: 0.999. Use slightly smaller values (~0.8) if you want bigger less conservative clusters. See hdbscan documentation: https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
                'd_clusterMerge': None,  ## default: None. Distance (mixed conjunctive distance) at which all samples less than this far apart are joined in clusters. If None, then set to mean + 1.0 std of the distribution. See 'cluster_selection_epsilon' in hdbscan documentation: https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
                'cluster_selection_method': 'leaf',  ## default: 'leaf'. 'leaf' is better for smaller homogeneous clusters, 'eom' is better for larger clusters of various densities. See hdbscan documentation: https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
                'split_intraSession_clusters': True,  ## default: True. Splits up clusters with multiple ROIs from the same session into multiple clusters.
                'd_step': 0.01,  ## default: 0.03. Size of steps to take when splitting clusters with multiple ROIs from the same session. Smaller values give higher quality clusters.
                'discard_failed_pruning': True,  ## default: Failsafe. If the splitting doesn't work for whatever reason, then just set all violating clusters to label=-1.
                'n_iter_violationCorrection': 6,  ## default: 5. Number of times to iterate a correcting process to improve clusters. Warning: This can increase run time linearly for large datasets. Typically converges after around 5 iterations. If n_sessions is large (>30), consider increasing.
            },
            'hungarian': {
                'thresh_cost': 0.6, ## default: 0.95. Threshold distance at which all clusters larger than this are discarded. Note that typically no change will occur after values > d_cutoff.
            },
        },
    }
    return params


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
    import pkg_resources
    pkgs_dict = {i.key: i.version for i in pkg_resources.working_set}

    ## roicat
    import roicat
    import time
    
    roicat_version = roicat.__version__
    roicat_fileDate = time.ctime(os.path.getctime(pkg_resources.get_distribution("roicat").location))
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

    return versions


def download_data_test_zip(dir_data_test: str,) -> str:
    """
    Downloads the test data if it does not exist and checks its hash. If the
    data already exists, it only checks the hash.

    Args:
        dir_data_test (str):
            Directory to download the zip file into.

    Returns:
        (str): 
            path_save (str):
                Path to the downloaded data_test.zip file.
    """
    path_save = str(Path(dir_data_test) / 'data_test.zip')
    helpers.download_file(
        url=r'https://github.com/RichieHakim/ROICaT/raw/main/tests/data_test.zip', 
        path_save=path_save, 
        check_local_first=True, 
        check_hash=True, 
        hash_type='MD5', 
        hash_hex=r'764d9b3fc481e078d1ef59373695ecce',
        mkdir=True,
        allow_overwrite=True,
        write_mode='wb',
        verbose=True,
        chunk_size=1024,
    )
    return path_save


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

    def save(
        self, 
        path_save: Union[str, Path],
        save_as_serializable_dict: bool = False,
        compress: bool = False,
        allow_overwrite: bool = False,
    ) -> None:
        """
        Saves Data_roicat object to pickle file.

        Args:
            path_save (Union[str, pathlib.Path]): 
                Path to save pickle file.
            save_as_serializable_dict (bool): 
                An archival-type format that is easy to load data from, but typically 
                cannot be used to re-instantiate the object. If ``True``, save the object 
                as a serializable dictionary. If ``False``, save the object as a Data_roicat 
                object. (Default is ``False``)
            compress (bool): 
                If ``True``, compress the pickle file. (Default is ``False``)
            allow_overwrite (bool): 
                If ``True``, allow overwriting of existing file. (Default is ``False``)

        """
        from pathlib import Path
        ## Check if file already exists
        if not allow_overwrite:
            assert not Path(path_save).exists(), f"RH ERROR: File already exists: {path_save}. Set allow_overwrite=True to overwrite."

        helpers.pickle_save(
            obj=self.serializable_dict if save_as_serializable_dict else self,
            path_save=path_save,
            zipCompress=compress,
            mkdir=True,
            allow_overwrite=allow_overwrite,
        )
        print(f"Saved Data_roicat as a pickled object to {path_save}.") if self._verbose else None

    def load(
        self,
        path_load: Union[str, Path],
    ) -> None:
        """
        Loads attributes from a Data_roicat object from a pickle file.

        Args:
            path_load (Union[str, Path]): 
                Path to the pickle file.

        Note: 
            After calling this method, the attributes of this object are updated with those 
            loaded from the pickle file. If an object in the pickle file is a dictionary, 
            the object's attributes are set directly from the dictionary. Otherwise, if 
            the object in the pickle file has a 'import_from_dict' method, it is used 
            to load attributes. If it does not, the attributes are directly loaded from 
            the object's `__dict__` attribute.

        Example:
            .. highlight:: python
            .. code-block:: python

                obj = Data_roicat()
                obj.load('/path/to/pickle/file')
        """
        from pathlib import Path
        assert Path(path_load).exists(), f"RH ERROR: File does not exist: {path_load}."
        obj = helpers.pickle_load(path_load)
        assert isinstance(obj, (type(self), dict)), f"RH ERROR: Loaded object is not a Data_roicat object or dictionary. Loaded object is of type {type(obj)}."

        if isinstance(obj, dict):
            ## Set attributes from dict
            ### If the subclass has a load_from_dict method, use that.
            if hasattr(self, 'import_from_dict'):
                self.import_from_dict(obj)
            else:
                for key, val in obj.items():
                    setattr(self, key, val)
        else:
            ## Set attributes from object
            for key, val in obj.__dict__.items():
                setattr(self, key, val)

        print(f"Loaded Data_roicat object from {path_load}.") if self._verbose else None


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
            if all([all([isinstance(l, (int, float)) for l in lod]) for lod in lolod]):
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
            List of lists of UCIDs for each session.
        iscell (List[Union[List[bool], np.ndarray]]): 
            List of lists of boolean indicators for each UCID. 

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
    
    ucids_inAllSesh = [u for u in np.unique(ucids_out[0]) if np.array([np.isin(u, u_sesh) for u_sesh in ucids_out]).sum() >= n_sesh_thresh]
    if verbose:
        fraction = (np.unique(ucids_inAllSesh) >= 0).sum() / (np.unique(ucids_out[0]) >= 0).sum()
        print(f'INFO: {fraction*100:.2f}% of UCIDs in first session appear in at least {n_sesh_thresh} sessions.')
    ucids_out = [[val * np.isin(val, ucids_inAllSesh) - np.logical_not(np.isin(val, ucids_inAllSesh)) for val in u] for u in ucids_out]
    
    return ucids_out
    

def squeeze_UCID_labels(
    ucids: List[Union[List[int], np.ndarray]]
) -> List[Union[List[int], np.ndarray]]:
    """
    Squeezes the UCID labels. Finds all the unique UCIDs across all sessions,
    then removes spaces in the UCID labels by mapping the unique UCIDs to new
    values. Output UCIDs are contiguous integers starting at 0, and maintains
    elements with UCID=-1.

    Args:
        ucids (List[Union[List[int], np.ndarray]]): 
            List of lists of UCIDs for each session.

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
    mapping = {old: new for old, new in zip(uniques_all, helpers.squeeze_integers(uniques_all))}
    mapping.update({-1: -1})
    ## apply the mapping to the data
    n_sesh = len(ucids_out)
    for i_sesh in range(n_sesh):
        ucids_out[i_sesh] = [mapping[val] for val in ucids_out[i_sesh]]

    return ucids_out


def match_arrays_with_ucids(
    arrays: Union[np.ndarray, List[np.ndarray]], 
    ucids: Union[List[np.ndarray], List[List[int]]], 
    squeeze: bool = False,
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
        squeeze (bool): 
            If ``True``, then UCIDs are squeezed to be contiguous integers.
            (Default is ``False``)

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
    ucids_tu = squeeze_UCID_labels(ucids_tu) if squeeze else ucids_tu
    # max_ucid = (np.unique(np.concatenate(ucids_tu, axis=0)) >= 0).max()
    max_ucid = (np.unique(np.concatenate(ucids_tu, axis=0))).max().astype(int) + 1

    dicts_ucids = [{u: i for i, u in enumerate(u_sesh)} for u_sesh in ucids_tu]
    
    ## make ndarrays filled with np.nan for each session
    if isinstance(arrays[0], np.ndarray):
        arrays_out = [np.full((max_ucid, *a.shape[1:]), np.nan, dtype=arrays[0].dtype) for a in arrays]
    elif scipy.sparse.issparse(arrays[0]):
        arrays_out = [scipy.sparse.lil_matrix((max_ucid, *a.shape[1:])) for a in arrays]
    ## fill in the arrays with the data
    n_sesh = len(arrays)
    for i_sesh in range(n_sesh):
        for u, idx in dicts_ucids[i_sesh].items():
            if u >= 0:
                arrays_out[i_sesh][u] = arrays[i_sesh][idx]

    return arrays_out

def match_arrays_with_ucids_inverse(
    arrays: Union[np.ndarray, List[np.ndarray]], 
    ucids: Union[List[np.ndarray], List[List[int]]],
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

    Returns:
        (List[Union[np.ndarray, scipy.sparse.lil_matrix]]): 
            arrays_out (List[Union[np.ndarray, scipy.sparse.lil_matrix]]): 
                List of arrays with indices that correspond to the original
                indices of the arrays / ucids. 
    """
    import scipy.sparse

    arrays = [arrays] if not isinstance(arrays, list) else arrays

    ucids_tu = squeeze_UCID_labels(ucids)
    n_ucids = (np.unique(np.concatenate(ucids_tu, axis=0)) >= 0).sum()

    dicts_ucids = [{u: i for i, u in enumerate(u_sesh)} for u_sesh in ucids_tu]
    
    ## make ndarrays filled with np.nan for each session
    if isinstance(arrays[0], np.ndarray):
        arrays_out = [np.full((len(u_sesh), *a.shape[1:]), np.nan) for u_sesh, a in zip(ucids_tu, arrays)]
    elif scipy.sparse.issparse(arrays[0]):
        arrays_out = [scipy.sparse.lil_matrix((len(u_sesh), *a.shape[1:])) for u_sesh, a in zip(ucids_tu, arrays)]
    ## fill in the arrays with the data
    n_sesh = len(arrays)
    for i_sesh in range(n_sesh):
        for u, idx in dicts_ucids[i_sesh].items():
            if u >= 0:
                arrays_out[i_sesh][idx] = arrays[i_sesh][u]

    return arrays_out
    