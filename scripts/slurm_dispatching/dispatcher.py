
## Import general libraries
from pathlib import Path
import os
import sys
import copy

import numpy as np
import itertools
import glob

### Import personal libraries
# dir_github = '/media/rich/Home_Linux_partition/github_repos'
dir_github = '/n/data1/hms/neurobio/sabatini/rich/github_repos'

import sys
sys.path.append(dir_github)
# %load_ext autoreload
# %autoreload 2
from basic_neural_processing_modules import container_helpers, server
# from s2p_on_o2 import remote_run_s2p


args = sys.argv
path_selfScript = args[0]
dir_save = args[1]
path_script = args[2]
name_job = args[3]
name_slurm = args[4]
dir_data = args[5]
plane_name = args[6]
dir_ROInet_networkFiles = args[7]

print(path_selfScript, dir_save, path_script, name_job, dir_data)

## set paths
# dir_save = '/n/data1/hms/neurobio/sabatini/rich/analysis/suite2p_output/'
Path(dir_save).mkdir(parents=True, exist_ok=True)


# path_script = '/n/data1/hms/neurobio/sabatini/rich/github_repos/s2p_on_o2/remote_run_s2p.py'


### Define directories for data and output.
## length of both lists should be the same
# dirs_data_all = ['/n/data1/hms/neurobio/sabatini/rich/analysis/suite2p_output']
# dirs_save_all = [str(Path(dir_save) / 'test_s2p_on_o2')]



params_template = {
    'paths': {
        'dir_github': dir_github,  ## directory where ROICat is
        'dir_allOuterFolders': dir_data,  ## directory where directories containing below 'pathSuffixTo...' are
        'folderName_inner': plane_name,
        'pathSuffixToStat': 'stat.npy',  ## path suffix to where the stat.npy file is
        'pathSuffixToOps':  'ops.npy',  ## path suffix to where the ops.npy file is
    },
    'importing': {
        'data_verbose': True,  ## default: True. Whether to print out data importing information
        'out_height_width': [72, 72],  ## default: [72, 72]. Height and width of output images (note that this must agree with the input of the ROInet input)
        'max_footprint_width': 1025,  ## default: 1025. Maximum length of a spatial footprint. If you get an error during importing, try increasing this value.
        'type_meanImg': 'meanImgE',  ## default: 'meanImgE'. Type of mean image to use for normalization. This is just a field in the ops.npy file.
        'images': None,  ## default: None. Set to None if you want to use the images extracted from Suite2p
        'import_workers': -1, ## default: -1. Number of workers to use for importing. Set to -1 to use all available workers. Values other than 1 result in using multiprocessing.
        'um_per_pixel': 1.0,  ## default: 1.0. Microns per pixel of imaging field of view. A rough estimate (to within ~40% of true value) is okay.
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
        'dir_networkFiles': dir_ROInet_networkFiles,  ## local directory where network files are stored
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
        'image_shape': (72, 72),  ## default: (36,36). Shape of the images.
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

## make params dicts with grid swept values
params = copy.deepcopy(params_template)
params = [params]
# params = [container_helpers.deep_update_dict(params, ['db', 'save_path0'], str(Path(val).resolve() / (name_save+str(ii)))) for val in dir_save]
# params = [container_helpers.deep_update_dict(param, ['db', 'save_path0'], val) for param, val in zip(params, dirs_save_all)]
# params = container_helpers.flatten_list([[container_helpers.deep_update_dict(p, ['lr'], val) for val in [0.00001, 0.0001, 0.001]] for p in params])

# params_unchanging, params_changing = container_helpers.find_differences_across_dictionaries(params)


## notes that will be saved as a text file in the outer directory
notes = \
"""
First attempt
"""
with open(str(Path(dir_save) / 'notes.txt'), mode='a') as f:
    f.write(notes)



## copy script .py file to dir_save
import shutil
shutil.copy2(path_script, str(Path(dir_save) / Path(path_script).name));



# ## save parameters to file
# parameters_batch = {
#     'params': params,
#     # 'params_unchanging': params_unchanging,
#     # 'params_changing': params_changing
# }
# import json
# with open(str(Path(dir_save) / 'parameters_batch.json'), 'w') as f:
#     json.dump(parameters_batch, f)

# with open(str(Path(dir_save) / 'parameters_batch.json')) as f:
#     test = json.load(f)


## run batch_run function
paths_scripts = [path_script]
params_list = params
# sbatch_config_list = [sbatch_config]
max_n_jobs=1
name_save=name_job


## define print log paths
paths_log = [str(Path(dir_save) / f'{name_save}{jobNum}' / 'print_log_%j.log') for jobNum in range(len(params))]

## define slurm SBATCH parameters
sbatch_config_list = \
[f"""#!/usr/bin/bash
#SBATCH --job-name={name_slurm}
#SBATCH --output={path}
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --partition=gpu_requeue
#SBATCH -c 8
#SBATCH -n 1
#SBATCH --mem=64GB
#SBATCH --time=0-01:00:00
unset XDG_RUNTIME_DIR
cd /n/data1/hms/neurobio/sabatini/rich/
date
echo "loading modules"
module load gcc/9.2.0 cuda/11.2
echo "activating environment"
source activate ROICaT
echo "starting job"
python "$@"
""" for path in paths_log]

server.batch_run(
    paths_scripts=paths_scripts,
    params_list=params_list,
    sbatch_config_list=sbatch_config_list,
    max_n_jobs=max_n_jobs,
    dir_save=str(dir_save),
    name_save=name_save,
    verbose=True,
)