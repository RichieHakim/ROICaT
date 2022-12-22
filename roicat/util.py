from pathlib import Path

import importlib_metadata

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
    conda_env = os.environ['CONDA_DEFAULT_ENV']
    print(f'Conda Environment: {conda_env}') if verbose else None

    ## Python
    import sys
    python_version = sys.version.split(' ')[0]
    print(f'Python Version: {python_version}') if verbose else None

    ## GCC
    import subprocess
    gcc_version = subprocess.check_output(['gcc', '--version']).decode('utf-8').split('\n')[0].split(' ')[-1]
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

    ## face-rhythm
    import roicat
    faceRhythm_version = roicat.__version__
    print(f'roicat Version: {faceRhythm_version}') if verbose else None

    versions = {
        'roicat_version': faceRhythm_version,
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
