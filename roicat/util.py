from pathlib import Path
import warnings
import copy

import importlib_metadata

import numpy as np

from . import helpers

def get_roicat_version():
    """
    Get the version of the roicat package.
    """
    return importlib_metadata.version('roicat')


def make_params_default_tracking(
    dir_networkFiles=None,
):
    """
    Make a dictionary of default parameters for a 
     standard tracking pipeline.

    Args:
        dir_networkFiles (str):
            Directory where the ROInet network files are saved.
            If None, use the current directory.
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


def get_system_versions(verbose=False):
    """
    Checks the versions of various important softwares.
    Prints those versions
    RH 2022

    Args:
        verbose (bool): 
            Whether to print the versions

    Returns:
        versions (dict):
            Dictionary of versions
    """
    ## Operating system and version
    import platform
    operating_system = str(platform.system()) + ': ' + str(platform.release()) + ', ' + str(platform.version()) + ', ' + str(platform.machine()) + ', node: ' + str(platform.node()) 
    print(f'Operating System: {operating_system}') if verbose else None

    ## Conda Environment
    import os
    if 'CONDA_DEFAULT_ENV' not in os.environ:
        conda_env = 'None'
    else:
        conda_env = os.environ['CONDA_DEFAULT_ENV']
    print(f'Conda Environment: {conda_env}') if verbose else None

    ## Python
    import sys
    python_version = sys.version.split(' ')[0]
    print(f'Python Version: {python_version}') if verbose else None

    ## GCC
    import subprocess
    try:
        gcc_version = subprocess.check_output(['gcc', '--version']).decode('utf-8').split('\n')[0].split(' ')[-1]
    except Exception as e:
        warnings.warn(f'RH WARNING: unable to get gcc version. Got error: {e}')
        gcc_version = 'Faled to get'
    print(f'GCC Version: {gcc_version}') if verbose else None
    
    ## PyTorch
    import torch
    torch_version = str(torch.__version__)
    print(f'PyTorch Version: {torch_version}') if verbose else None
    ## CUDA
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"\
CUDA Version: {cuda_version}, \
CUDNN Version: {torch.backends.cudnn.version()}, \
Number of Devices: {torch.cuda.device_count()}, \
Devices: {[f'device {i}: Name={torch.cuda.get_device_name(i)}, Memory={torch.cuda.get_device_properties(i).total_memory / 1e9} GB' for i in range(torch.cuda.device_count())]}, \
") if verbose else None
    else:
        cuda_version = None
        print('CUDA is not available') if verbose else None

    ## Numpy
    import numpy
    numpy_version = numpy.__version__
    print(f'Numpy Version: {numpy_version}') if verbose else None

    ## OpenCV
    import cv2
    opencv_version = cv2.__version__
    print(f'OpenCV Version: {opencv_version}') if verbose else None
    # print(cv2.getBuildInformation())

    ## roicat
    import roicat
    roicat_version = roicat.__version__
    print(f'roicat Version: {roicat_version}') if verbose else None

    versions = {
        'roicat_version': roicat_version,
        'operating_system': operating_system,
        'conda_env': conda_env,
        'python_version': python_version,
        'gcc_version': gcc_version,
        'torch_version': torch_version,
        'cuda_version': cuda_version,
        'numpy_version': numpy_version,
        'opencv_version': opencv_version,
    }

    return versions


def download_data_test_zip(dir_data_test):
    """
    Downloads the test data if it does not exist.
    If the data exists, check its hash.

    Args:
        dir_data_test (str):
            directory to download zip file into

    Returns:
        path_save (str):
            path to data_test.zip file
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
    """
    def __init__(self):
        pass

    @property
    def serializable_dict(self):
        """
        Return a serializable dict that can be saved to disk.
        """
        ## Go through all items in self.__dict__ and check if they are serializable.
        ### If they are, add them to a dictionary to be returned.
        import pickle

        ## Define a list of libraries and classes that are allowed to be serialized.
        allowed_libraries = [
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
        
        def make_serializable_dict(obj, depth=0, max_depth=100):
            """
            Recursively go through all items in self.__dict__ and check if they are serializable.
            """
            if depth > max_depth:
                raise Exception(f'RH ERROR: max_depth of {max_depth} reached with object: {obj}')
            serializable_dict = {}
            for key, val in obj.__dict__.items():
                print(key)
                try:
                    ## Check if the value is in the allowed_libraries list.
                    if is_library_allowed(val):
                        pass
                    else:
                        continue
                    ## Check if the value is serializable.
                    pickle.dumps(val)
                    ## If it is, check to see if it has it's own serializable_dict property.
                    try:
                        serializable_dict[key] = make_serializable_dict(val, depth=depth+1, max_depth=max_depth)
                    except:
                        serializable_dict[key] = val
                except:
                    pass
            return serializable_dict
        
        serializable_dict = make_serializable_dict(self, depth=0, max_depth=100)
        return serializable_dict


##########################################################################################################################
############################################### UCID handling ############################################################
##########################################################################################################################

def check_dataStructure__list_ofListOrArray_ofDtype(lolod, dtype=np.int64, fix=True, verbose=True):
    """
    Checks 'lolod' (list of list of dtype) data structure.
    Structure should be a list of lists of dtypes OR a list of np.arrays of dtypes.

    Args:
        lolod (list):
            List of lists of dtypes OR a list of np.arrays of dtypes.
        fix (bool):
            Whether to attempt to fix the data structure if it is incorrect.
            If fix=False, then raises an error if it is not correct.

    Returns:
        lolod (list):
            List of lists of dtypes OR a list of np.arrays of dtypes.
    """

    ## switch case for if it is a list or np.ndarray
    if isinstance(lolod, list):
        ## switch case for if the elements are lists or np.ndarray or numbers (int or float) or dtypes
        if all([isinstance(lod, list) for lod in lolod]):
            ## switch case for if the elements are numbers (int or float) or dtype or other
            if all([all([isinstance(l, (int, float)) for l in lod]) for lod in lolod]):
                if fix:
                    print(f'ROICaT WARNING: lolod is a list of lists of numbers (int or float). Converting to np.ndarray.')
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
                    print(f'ROICaT WARNING: lolod is a list of np.ndarray of numbers (int or float). Converting to np.ndarray.')
                    lolod = [np.array(lod, dtype=dtype) for lod in lolod]
                else:
                    raise ValueError(f'ROICaT ERROR: lolod is a list of np.ndarray of numbers (int or float).')
            else:
                raise ValueError(f'ROICaT ERROR: lolod is a list of np.ndarray, but the elements are not all numbers (int or float) or dtype.')
        elif all([isinstance(lod, (int, float)) for lod in lolod]):
            if fix:
                print(f'ROICaT WARNING: lolod is a list of numbers (int or float). Converting to np.ndarray.')
                lolod = [np.array(lod, dtype=dtype) for lod in lolod]
            else:
                raise ValueError(f'ROICaT ERROR: lolod is a list of numbers (int or float).')
        elif all([isinstance(lod, dtype) for lod in lolod]):
            if fix:
                print(f'ROICaT WARNING: lolod is a list of dtype. Converting to np.ndarray.')
                lolod = [np.array(lolod, dtype=dtype)]
            else:
                raise ValueError(f'ROICaT ERROR: lolod is a list of dtype.')
        else:
            raise ValueError(f'ROICaT ERROR: lolod is a list, but the elements are not all lists or np.ndarray or numbers (int or float).')
    elif isinstance(lolod, np.ndarray):
        ## switch case for if the elements are numbers (any non-object numpy dtype) or dtype or other
        if np.issubdtype(lolod.dtype, dtype):
            if fix:
                print(f'ROICaT WARNING: lolod is a np.ndarray of dtype. Converting to list of np.ndarray of dtype.')
                lolod = [lolod]
        elif not np.issubdtype(lolod.dtype, np.object_):
            if fix:
                print(f'ROICaT WARNING: lolod is a np.ndarray of numbers (int or float). Converting to list of np.ndarray of dtype.')
                lolod = [np.array(lolod, dtype=dtype)]
            else:
                raise ValueError(f'ROICaT ERROR: lolod is a np.ndarray of numbers (int or float).')
        else:
            raise ValueError(f'ROICaT ERROR: lolod is a np.ndarray, but the elements are not all numbers (int or float) or dtype.')
    else:
        raise ValueError(f'ROICaT ERROR: lolod is not a list or np.ndarray.')

    return lolod


def mask_UCIDs_with_iscell(ucids, iscell):
    """
    Masks the UCIDs with the iscell array.
    If iscell is False, then the UCID is set to -1.

    Args:
        ucids (list of [list or array] of int):
            List of lists of UCIDs for each session.
        iscell (list of [list or array] of bool):
            List of lists of iscell.

    Returns:
        ucids_out (list of [list or array] of int):
            Masked list of lists of UCIDs.
            Elements that are not cells are set to -1 in each session.
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
    ucids, 
    n_sesh_thresh='all',
    verbose=True,
):
    """
    Discards UCIDs that do not appear in at least n_sesh_thresh sessions.
    If n_sesh_thresh='all', then only UCIDs that appear in all sessions are kept.

    Args:
        ucids (list of [list or array] of int):
            List of lists of UCIDs for each session.
        n_sesh_thresh (int or 'all'):
            Number of sessions that a UCID must appear in to be kept.
            If 'all', then only UCIDs that appear in all sessions are kept.

    Returns:
        ucids_out (list of [list or array] of int):
            List of lists of UCIDs with UCIDs that do not appear in at least
             n_sesh_thresh sessions set to -1.
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
    

def squeeze_UCID_labels(ucids):
    """
    Squeezes the UCID labels.
    Finds all the unique UCIDs across all sessions, then removes spaces in the
     UCID labels by mapping the unique UCIDs to new values. Output UCIDs are
     contiguous integers starting at 0, and maintains elements with UCID=-1.

    Args:
        ucids (list of [list or array] of int):
            List of lists of UCIDs for each session.

    Returns:
        ucids_out (list of [list or array] of int):
            List of lists of UCIDs with UCIDs that do not appear in at least
             n_sesh_thresh sessions set to -1.
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


def match_arrays_with_ucids(arrays, ucids):
    """
    Matches the indices of the arrays using the UCIDs.
    Array indices with UCIDs corresponding to -1 are set to np.nan.
    Useful for aligning Fluorescence and Spiking data across sessions
     using UCIDs.
    
    Args:
        arrays (list of np.array):
            List of of numpy arrays for each session.
            Matching is done along the first dimension.
        ucids (list of [list or array] of int):
            List of lists of UCIDs for each session.

    Returns:
        arrays_out (list of np.array):
            List of of arrays for each session.
            Array indices with UCIDs corresponding to -1 are set to np.nan.
    """
    import scipy.sparse

    arrays = [arrays] if not isinstance(arrays, list) else arrays

    ucids_tu = squeeze_UCID_labels(ucids)
    n_ucids = (np.unique(np.concatenate(ucids_tu, axis=0)) >= 0).sum()

    dicts_ucids = [{u: i for i, u in enumerate(u_sesh)} for u_sesh in ucids_tu]
    
    ## make ndarrays filled with np.nan for each session
    if isinstance(arrays[0], np.ndarray):
        arrays_out = [np.full((n_ucids, *a.shape[1:]), np.nan) for a in arrays]
    elif scipy.sparse.issparse(arrays[0]):
        arrays_out = [scipy.sparse.lil_matrix((n_ucids, *a.shape[1:])) for a in arrays]
    ## fill in the arrays with the data
    n_sesh = len(arrays)
    for i_sesh in range(n_sesh):
        for u, idx in dicts_ucids[i_sesh].items():
            if u >= 0:
                arrays_out[i_sesh][u] = arrays[i_sesh][idx]

    return arrays_out