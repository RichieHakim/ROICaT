from pathlib import Path

import importlib_metadata

def get_roicat_version():
    """
    Get the version of the roicat package.
    """
    return importlib_metadata.version('roicat')

# def get_

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
            # 'dir_allOuterFolders': dir_data,  ## directory where directories containing below 'pathSuffixTo...' are
            # 'folderName_inner': plane_name,
            # 'pathSuffixToStat': 'stat.npy',  ## path suffix to where the stat.npy file is
            # 'pathSuffixToOps':  'ops.npy',  ## path suffix to where the ops.npy file is
        },
        'importing': {
            'data_verbose': True,  ## default: True. Whether to print out data importing information
            'out_height_width': [36, 36],  ## default: [36, 36]. Height and width of output images (note that this must agree with the input of the ROInet input)
            'max_footprint_width': 1025,  ## default: 1025. Maximum length of a spatial footprint. If you get an error during importing, try increasing this value.
            'type_meanImg': 'mimg',  ## default: 'meanImgE'. Type of mean image to use for normalization. Use 'mimg' for old matlab suite2p files. This is just a field in the ops.npy file.
            'images': None,  ## default: None. Set to None if you want to use the images extracted from Suite2p
            'import_workers': -1, ## default: -1. Number of workers to use for importing. Set to -1 to use all available workers. Values other than 1 result in using multiprocessing.
            'um_per_pixel': 1.0,  ## default: 1.0. Microns per pixel of imaging field of view. A rough estimate (to within ~40% of true value) is okay.
            'new_or_old_suite2p': 'old',  ## default: 'new'. Set to 'old' if you are using old MATLAB style suite2p files.
        },
        'alignment': {
            'method': 'createOptFlow_DeepFlow',  ## default: 'createOptFlow_DeepFlow'. Method to use for creating optical flow.
            'kwargs_method': None,  ## default: None. Keyword arguments to pass to the method.
            'return_sparse': True,  ## default: True. Whether to return a sparse matrix or a dense matrix.
            'normalize': True,  ## default: True. Whether to normalize the optical flow.
        },
        'blurring': {
            'kernel_halfWidth': 1.4,  ## default: 2.0. Half-width of the Gaussian kernel used for blurring. Use smaller values for smaller ROIs (dendrites) and larger values for larger ROIs (somata).
            'device': 'cpu',  ## default: 'cpu'. Device to use for blurring. Recommend using 'cpu' even if you have a GPU.
            'plot_kernel': False,  ## default: False. Whether to plot the kernel used for blurring.
            'batch_size': 2000,  ## default: 2000. Number of images to use for each batch.
        },
        'ROInet': {
            'device': 'cuda:0',  ## default: 'cuda:0'. Device to use for ROInet. Recommend using a GPU.
            'dir_networkFiles': dir_networkFiles,  ## local directory where network files are stored
            'download_from_gDrive': 'check_local_first',  ## default: 'check_local_first'. Whether to download the network files from Google Drive or to use the local files.
            'gDriveID': '1FCcPZUuOR7xG-hdO6Ei6mx8YnKysVsa8',  ## default: '1FCcPZUuOR7xG-hdO6Ei6mx8YnKysVsa8'. Google Drive ID of the network files.
            'verbose': True,  ## default: True. Whether to print out ROInet information.
            'pref_plot': False,  ## default: False. Whether to plot the ROI and the normalized ROI.
            'batchSize_dataloader': 8,  ## default: 8. Number of images to use for each batch.
            'pinMemory_dataloader': True,  ## default: True. Whether to pin the memory of the dataloader.
            'persistentWorkers_dataloader': True,  ## default: True. Whether to use persistent workers for the dataloader.
            'numWorkers_dataloader': -1,  ## default: -1. num_workers as a positive integer will turn on multi-process data loading. 0 will not use multiprocessing
            'prefetchFactor_dataloader': 2,  ## default: 2. Number of prefetch factors to use for the dataloader.
        },
        'SWT': {
            'kwargs_Scattering2D': {'J': 2, 'L': 8},  ## default: {'J': 2, 'L': 8}. Keyword arguments to pass to the Scattering2D function.
            'image_shape': (36, 36),  ## default: (36,36). Shape of the images.
            'device': 'cuda:0',  ## default: 'cuda:0'. Device to use for SWT. Recommend using a GPU.
            'batch_size': 100,  ## default: 100. Number of images to use for each batch.
        }, 
        'similarity': {
            'device': 'cpu',  ## default: 'cpu'. Device to use for similarity. Recommend using 'cpu' even if you have a GPU.
            'n_workers': -1,  ## default: -1. Number of workers to use for similarity. Set to -1 to use all available workers.
            'spatialFootprint_maskPower': 0.8,  ## default: 0.8. Power to use for the spatial footprint.
            'block_height': 50,  ## default: 50. Height of the block to use for similarity.
            'block_width': 50,  ## default: 50. Width of the block to use for similarity.
            'overlapping_width_Multiplier': 0.1,  ## default: 0.1. Multiplier to use for the overlapping width.
            'algorithm_nearestNeigbors_spatialFootprints': 'brute',  ## default: 'brute'. Algorithm to use for nearest neighbors.
            'n_neighbors_nearestNeighbors_spatialFootprints': 'full',  ## default: 'full'. Number of neighbors to use for nearest neighbors.
            'locality': 1,  ## default: 1. Locality to use for nearest neighbors. Exponent applied to the similarity matrix input.
            'verbose': True,  ## default: True. Whether to print out similarity information.
        },
        'similarity_compute': {
            'linkage_methods': ['single', 'complete', 'ward', 'average'],  ## default: ['single', 'complete', 'ward', 'average']. Linkage methods to use for computing linkage distances and ultimately clusters.
            'bounded_logspace_args': (0.05, 2, 50),  ## default: (0.05, 2, 50). Linkage distances to use to find clusters.
            'min_cluster_size': 2,  ## default: 2. Minimum size of a cluster.
            'max_cluster_size': None,  ## default: None. Maximum size of a cluster. If None, then set to n_sessions.
            'batch_size_hashing': 100,  ## default: 100. Number of images to use for each batch.
            'cluster_similarity_reduction_intra': 'mean',  ## default: 'mean'. Reduction method to use for intra-cluster similarity.
            'cluster_similarity_reduction_inter': 'max',  ## default: 'max'. Reduction method to use for inter-cluster similarity.
            'cluster_silhouette_reduction_intra': 'mean',  ## default: 'mean'. Reduction method to use for intra-cluster silhouette.
            'cluster_silhouette_reduction_inter': 'max',  ## default: 'max'. Reduction method to use for inter-cluster silhouette.
            'n_workers': 8,  ## default: 8. Number of workers to use for similarity. WARNING, using many workers requires large memory requirement. Set to -1 to use all available workers.
            'power_clusterSize': 1.3,  ## default: 2. Used in calculating custom cluster score. This is the exponent applied to the number of ROIs in a cluster.
            'power_clusterSilhouette': 1.5,  ## default: 1.5. Used in calculating custom cluster score. This is the exponent applied to the silhouette score of a cluster.
        },
        'clusterAssigner': {
            'device': 'cuda:0',  ## default: 'cuda:0'. Device to use for clusterAssigner. Recommend using a GPU.
            'optimizer_partial_lr': 1e-1,  ## default: 1e-1. Learning rate for the optimizer.
            'optimizer_partial_betas': (0.9, 0.900),  ## default: (0.9, 0.900). Betas for the optimizer.
            'scheduler_partial_base_lr': 1e-3,  ## default: 1e-3. Base learning rate for the scheduler.
            'scheduler_partial_max_lr': 3e0,  ## default: 3e0. Maximum learning rate for the scheduler.
            'scheduler_partial_step_size_up': 250,  ## default: 250. Step size for the scheduler.
            'scheduler_partial_cycle_momentum': False,  ## default: False. Whether to cycle the momentum of the optimizer.
            'scheduler_partial_verbose': False,  ## default: False. Whether to print out scheduler information.
            'dmCEL_temp': 1,  ## default: 1. Temperature to use for the dmCEL loss.
            'dmCEL_sigSlope': 2,  ## default: 2. Slope to use for the dmCEL loss.
            'dmCEL_sigCenter': 0.5,  ## default: 0.5. Center to use for the dmCEL loss.
            'dmCEL_penalty': 1e0,  ## default: 1e0. Penalty to use for the dmCEL loss.
            'sampleWeight_softplusKwargs': {'beta': 150, 'threshold': 50},  ## default: {'beta': 150, 'threshold': 50}. Keyword arguments to pass to the softplus function.
            'sampleWeight_penalty': 1e3,  ## default: 1e3. Penalty to use for when an ROI is assigned to multiple clusters.
            'fracWeighted_goalFrac': 1.0,  ## default: 1.0. Goal fraction ROIs assigned to a cluster.
            'fracWeighted_sigSlope': 2,  ## default: 2. Slope to use for the sigmoid activation for the fracWeighted loss.
            'fracWeighted_sigCenter': 0.5,  ## default: 0.5. Center to use for the fracWeighted loss sigmoid.
            'fracWeight_penalty': 1e3,  ## default: 1e2. Penalty to use for the fracWeighted loss.
            'maskL1_penalty': 4e-4,  ## default: 2e-4. Penalty to use for the L1 loss applied to the number of non-zero clusters.
            'tol_convergence': 1e-9,  ## default: 1e-9. Tolerance to use for convergence.
            'window_convergence': 50,  ## default: 50. Number of past iterations to use in calculating a smooth value for the loss derivative for convergence.
            'freqCheck_convergence': 50,  ## default: 50. Period between checking for convergence.
            'verbose': True,  ## default: True. Whether to print out information about the initialization.
        },
        'clusterAssigner_fit': {
            'min_iter': 1e3,  ## default: 1e3. Minimum number of iterations to run.
            'max_iter': 5e3,  ## default: 5e3. Maximum number of iterations to run.
            'verbose': True,  ## default: True. Whether to print out information about the optimization.
            'verbose_interval': 100,  ## default: 100. Number of iterations between printing out information about the optimization.
            'm_threshold': 0.8,  ## default: 0.8. Threshold for the activated mask vector to define as an included cluster when making predictions.
        },
        'visualization': {
            'FOV_threshold_confidence': 0.5,  ## default: 0.5. Threshold for the confidence scores when displaying ROIs.
        }
    }

    return params