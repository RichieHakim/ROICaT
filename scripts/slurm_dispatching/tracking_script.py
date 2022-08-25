import time
t_start = time.time()

###############################
## BATCH RUN STUFF#############
###############################

### batch_run stuff
from pathlib import Path

import sys
path_script, path_params, dir_save = sys.argv
dir_save = Path(dir_save)
                
import json
with open(path_params, 'r') as f:
    params = json.load(f)

import shutil
shutil.copy2(path_script, str(Path(dir_save) / Path(path_script).name));

# params = {
#     'paths': {
#         'dir_github': dir_github,  ## directory where ROICat is
#         'dir_allOuterFolders': r"/media/rich/bigSSD/other lab data/Harnett_lab/ROI_Tracking/Vincent_Valerio/4th_email/AllStatFiles/rbp16",  ## directory where directories containing below 'pathSuffixTo...' are
#         'pathSuffixToStat': 'plane0/stat.npy',  ## path suffix to where the stat.npy file is
#         'pathSuffixToOps': 'plane0/ops.npy',  ## path suffix to where the ops.npy file is
#     },
#     'importing': {
#         'data_verbose': True,  ## default: True. Whether to print out data importing information
#         'out_height_width': [72, 72],  ## default: [72, 72]. Height and width of output images (note that this must agree with the input of the ROInet input)
#         'max_footprint_width': 1025,  ## default: 1025. Maximum length of a spatial footprint. If you get an error during importing, try increasing this value.
#         'type_meanImg': 'meanImgE',  ## default: 'meanImgE'. Type of mean image to use for normalization. This is just a field in the ops.npy file.
#         'images': None,  ## default: None. Set to None if you want to use the images extracted from Suite2p
#         'import_workers': -1, ## default: -1. Number of workers to use for importing. Set to -1 to use all available workers.
#         'um_per_pixel': 1.0,  ## default: 1.0. Microns per pixel of imaging field of view. A rough estimate (to within ~40% of true value) is okay.
#     },
#     'alignment': {
#         'method': 'createOptFlow_DeepFlow',  ## default: 'createOptFlow_DeepFlow'. Method to use for creating optical flow.
#         'kwargs_method': None,  ## default: None. Keyword arguments to pass to the method.
#         'return_sparse': True,  ## default: True. Whether to return a sparse matrix or a dense matrix.
#         'normalize': True,  ## default: True. Whether to normalize the optical flow.
#     },
#     'blurring': {
#         'kernel_halfWidth': 1.4,  ## default: 2.0. Half-width of the Gaussian kernel used for blurring. Use smaller values for smaller ROIs (dendrites) and larger values for larger ROIs (somata).
#         'device': 'cpu',  ## default: 'cpu'. Device to use for blurring. Recommend using 'cpu' even if you have a GPU.
#         'plot_kernel': False,  ## default: False. Whether to plot the kernel used for blurring.
#         'batch_size': 2000,  ## default: 2000. Number of images to use for each batch.
#     },
#     'ROInet': {
#         'device': 'cuda:0',  ## default: 'cuda:0'. Device to use for ROInet. Recommend using a GPU.
#         'dir_networkFiles': '/home/rich/Downloads/ROInet',  ## local directory where network files are stored
#         'download_from_gDrive': 'check_local_first',  ## default: 'check_local_first'. Whether to download the network files from Google Drive or to use the local files.
#         'gDriveID': '1FCcPZUuOR7xG-hdO6Ei6mx8YnKysVsa8',  ## default: '1FCcPZUuOR7xG-hdO6Ei6mx8YnKysVsa8'. Google Drive ID of the network files.
#         'verbose': True,  ## default: True. Whether to print out ROInet information.
#         'ptile_norm': 90,  ## default: 90. Percentile to use for normalizing the ROI.
#         'pref_plot': False,  ## default: False. Whether to plot the ROI and the normalized ROI.
#         'batchSize_dataloader': 8,  ## default: 8. Number of images to use for each batch.
#         'pinMemory_dataloader': True,  ## default: True. Whether to pin the memory of the dataloader.
#         'persistentWorkers_dataloader': True,  ## default: True. Whether to use persistent workers for the dataloader.
#         'prefetchFactor_dataloader': 2,  ## default: 2. Number of prefetch factors to use for the dataloader.
#     },
#     'SWT': {
#         'kwargs_Scattering2D': {'J': 2, 'L': 8},  ## default: {'J': 2, 'L': 8}. Keyword arguments to pass to the Scattering2D function.
#         'image_shape': (36, 36),  ## default: (36,36). Shape of the images.
#         'device': 'cuda:0',  ## default: 'cuda:0'. Device to use for SWT. Recommend using a GPU.
#     }, 
#     'similarity': {
#         'device': 'cpu',  ## default: 'cpu'. Device to use for similarity. Recommend using 'cpu' even if you have a GPU.
#         'n_workers': -1,  ## default: -1. Number of workers to use for similarity. Set to -1 to use all available workers.
#         'spatialFootprint_maskPower': 0.8,  ## default: 0.8. Power to use for the spatial footprint.
#         'block_height': 50,  ## default: 50. Height of the block to use for similarity.
#         'block_width': 50,  ## default: 50. Width of the block to use for similarity.
#         'overlapping_width_Multiplier': 0.1,  ## default: 0.1. Multiplier to use for the overlapping width.
#         'algorithm_nearestNeigbors_spatialFootprints': 'brute',  ## default: 'brute'. Algorithm to use for nearest neighbors.
#         'n_neighbors_nearestNeighbors_spatialFootprints': 'full',  ## default: 'full'. Number of neighbors to use for nearest neighbors.
#         'locality': 1,  ## default: 1. Locality to use for nearest neighbors. Exponent applied to the similarity matrix input.
#         'verbose': True,  ## default: True. Whether to print out similarity information.
#     },
#     'similarity_compute': {
#         'linkage_methods': ['single', 'complete', 'ward', 'average'],  ## default: ['single', 'complete', 'ward', 'average']. Linkage methods to use for computing linkage distances and ultimately clusters.
#         'bounded_logspace_args': (0.05, 2, 50),  ## default: (0.05, 2, 50). Linkage distances to use to find clusters.
#         'min_cluster_size': 2,  ## default: 2. Minimum size of a cluster.
#         'max_cluster_size': None,  ## default: None. Maximum size of a cluster. If None, then set to n_sessions.
#         'batch_size_hashing': 100,  ## default: 100. Number of images to use for each batch.
#         'cluster_similarity_reduction_intra': 'mean',  ## default: 'mean'. Reduction method to use for intra-cluster similarity.
#         'cluster_similarity_reduction_inter': 'max',  ## default: 'max'. Reduction method to use for inter-cluster similarity.
#         'cluster_silhouette_reduction_intra': 'mean',  ## default: 'mean'. Reduction method to use for intra-cluster silhouette.
#         'cluster_silhouette_reduction_inter': 'max',  ## default: 'max'. Reduction method to use for inter-cluster silhouette.
#         'n_workers': 8,  ## default: 8. Number of workers to use for similarity. WARNING, using many workers requires large memory requirement. Set to -1 to use all available workers.
#         'power_clusterSize': 1.3,  ## default: 2. Used in calculating custom cluster score. This is the exponent applied to the number of ROIs in a cluster.
#         'power_clusterSilhouette': 1.5,  ## default: 1.5. Used in calculating custom cluster score. This is the exponent applied to the silhouette score of a cluster.
#     },
#     'clusterAssigner': {
#         'device': 'cuda:0',  ## default: 'cuda:0'. Device to use for clusterAssigner. Recommend using a GPU.
#         'optimizer_partial_lr': 1e-1,  ## default: 1e-1. Learning rate for the optimizer.
#         'optimizer_partial_betas': (0.9, 0.900),  ## default: (0.9, 0.900). Betas for the optimizer.
#         'scheduler_partial_base_lr': 1e-3,  ## default: 1e-3. Base learning rate for the scheduler.
#         'scheduler_partial_max_lr': 3e0,  ## default: 3e0. Maximum learning rate for the scheduler.
#         'scheduler_partial_step_size_up': 250,  ## default: 250. Step size for the scheduler.
#         'scheduler_partial_cycle_momentum': False,  ## default: False. Whether to cycle the momentum of the optimizer.
#         'scheduler_partial_verbose': False,  ## default: False. Whether to print out scheduler information.
#         'dmCEL_temp': 1,  ## default: 1. Temperature to use for the dmCEL loss.
#         'dmCEL_sigSlope': 2,  ## default: 2. Slope to use for the dmCEL loss.
#         'dmCEL_sigCenter': 0.5,  ## default: 0.5. Center to use for the dmCEL loss.
#         'dmCEL_penalty': 1e0,  ## default: 1e0. Penalty to use for the dmCEL loss.
#         'sampleWeight_softplusKwargs': {'beta': 150, 'threshold': 50},  ## default: {'beta': 150, 'threshold': 50}. Keyword arguments to pass to the softplus function.
#         'sampleWeight_penalty': 1e3,  ## default: 1e3. Penalty to use for when an ROI is assigned to multiple clusters.
#         'fracWeighted_goalFrac': 1.0,  ## default: 1.0. Goal fraction ROIs assigned to a cluster.
#         'fracWeighted_sigSlope': 2,  ## default: 2. Slope to use for the sigmoid activation for the fracWeighted loss.
#         'fracWeighted_sigCenter': 0.5,  ## default: 0.5. Center to use for the fracWeighted loss sigmoid.
#         'fracWeight_penalty': 1e3,  ## default: 1e2. Penalty to use for the fracWeighted loss.
#         'maskL1_penalty': 4e-4,  ## default: 2e-4. Penalty to use for the L1 loss applied to the number of non-zero clusters.
#         'tol_convergence': 1e-9,  ## default: 1e-9. Tolerance to use for convergence.
#         'window_convergence': 50,  ## default: 50. Number of past iterations to use in calculating a smooth value for the loss derivative for convergence.
#         'freqCheck_convergence': 50,  ## default: 50. Period between checking for convergence.
#         'verbose': True,  ## default: True. Whether to print out information about the initialization.
#     },
#     'clusterAssigner_fit': {
#         'min_iter': 1e3,  ## default: 1e3. Minimum number of iterations to run.
#         'max_iter': 5e3,  ## default: 5e3. Maximum number of iterations to run.
#         'verbose': True,  ## default: True. Whether to print out information about the optimization.
#         'verbose_interval': 100,  ## default: 100. Number of iterations between printing out information about the optimization.
#         'm_threshold': 0.8,  ## default: 0.8. Threshold for the activated mask vector to define as an included cluster when making predictions.
#     },
#     'visualization': {
#         'FOV_threshold_confidence': 0.5,  ## default: 0.5. Threshold for the confidence scores when displaying ROIs.
#     }
# }

###############################
## IMPORTS#####################
###############################

print(f'## Starting: Importing libraries')

import time

import sys

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import natsort

import torch

import gc
import time
import functools
import multiprocessing as mp

# check environment
import os
print(f'Conda Environment: ' + os.environ['CONDA_DEFAULT_ENV'])

import sys
sys.path.append(params['paths']['dir_github'])
# from ROICaT.tracking import data_importing, visualization, alignment, blurring, helpers, ROInet, scatteringWaveletTransformer, similarity_graph, cluster_assignment
from ROICaT.tracking import data_importing, visualization, alignment, helpers, ROInet, scatteringWaveletTransformer, similarity_graph, cluster_assignment  ## sans blurring



## check Python version
from platform import python_version
print(f'Python version: {python_version()}')

## check package versions
print(f'PyTorch version: {torch.__version__}')

## check devices
if torch.cuda.is_available():
    print(f'There are {torch.cuda.device_count()} CUDA devices available: {[torch.cuda.get_device_properties(ii) for ii in range(torch.cuda.device_count())]}')

print(f'## Completed: Importing libraries')




###############################
## IMPORT DATA ################
###############################

print(f'## Starting: Finding data paths')
# Import paths
def print_list(l):
    for item in l:
        print(item)

dir_allOuterFolders = Path(params['paths']['dir_allOuterFolders']).resolve()

folders_allSessions = natsort.natsorted(helpers.get_dir_contents(dir_allOuterFolders)[0])

dir_allS2pFolders = [dir_allOuterFolders / folder for folder in folders_allSessions]

pathSuffixToStat = params['paths']['pathSuffixToStat']
pathSuffixToOps = params['paths']['pathSuffixToOps']

paths_allStat = np.array([path / pathSuffixToStat for path in dir_allS2pFolders])[:2]
paths_allOps  = np.array([path / pathSuffixToOps for path in dir_allS2pFolders])[:2]

print(f'folder names of all sessions: \n{folders_allSessions}')
print(f'paths to all stat files: \n{paths_allStat}')

print(f'## Completed: Finding data paths')


print(f'## Starting: Importing data')
#Import data
data = data_importing.Data_suite2p(
    paths_statFiles=paths_allStat,
    paths_opsFiles=paths_allOps,
    um_per_pixel=params['importing']['um_per_pixel'],
    verbose=params['importing']['data_verbose'],
);

data.import_statFiles();

data.import_ROI_centeredImages(
    out_height_width=params['importing']['out_height_width'],
    max_footprint_width=params['importing']['max_footprint_width'],
);

data.import_FOV_images(
    type_meanImg=params['importing']['type_meanImg'],
    images=params['importing']['images'],
);

data.import_ROI_spatialFootprints(workers=params['importing']['import_workers']);

# visualization.display_toggle_image_stack(data.FOV_images)
print(f'## Completed: Importing data')


print(f'## Starting: Aligning FOVs')
# Alignment
aligner = alignment.Alinger(
    method=params['alignment']['method'],
    kwargs_method=params['alignment']['kwargs_method'],
)

aligner.register_ROIs(
    templateFOV=data.FOV_images[0],
    FOVs=data.FOV_images,
    ROIs=data.spatialFootprints,
    return_sparse=params['alignment']['return_sparse'],
    normalize=params['alignment']['normalize'],
);

# visualization.display_toggle_image_stack(aligner.FOVs_aligned)
# visualization.display_toggle_image_stack(aligner.get_ROIsAligned_maxIntensityProjection())
print(f'## Completed: Aligning FOVs')


# print(f'## Starting: Blurring FOVs')
# # Blur ROIs (optional)
# blurrer = blurring.ROI_Blurrer(
#     frame_shape=(data.FOV_height, data.FOV_width),
#     kernel_halfWidth=params['blurring']['kernel_halfWidth'],
#     device=params['blurring']['device'],
#     plot_kernel=params['blurring']['plot_kernel'],
# )

# blurrer.blur_ROIs(
#     spatialFootprints=aligner.ROIs_aligned,
#     batch_size=params['blurring']['batch_size'],
# );

# # visualization.display_toggle_image_stack(blurrer.get_ROIsBlurred_maxIntensityProjection())
# print(f'## Completed: Blurring FOVs')


print(f'## Starting: Passing ROIs through ROInet')
# Neural network embedding distances
hash_dict_true = {
    'params': ('params.json', '877e17df8fa511a03bc99cd507a54403'),
    'model': ('model.py', '6ef5c29793ae16a64e43e8cab33d9ff4'),
    'state_dict': ('ConvNext_tiny__1_0_unfrozen__simCLR.pth', 'a5fae4c9ea95f2c78b4690222b2928a5'),
}

roinet = ROInet.ROInet_embedder(
    device=params['ROInet']['device'],
    dir_networkFiles=params['ROInet']['dir_networkFiles'],
    download_from_gDrive=params['ROInet']['download_from_gDrive'],
    gDriveID=params['ROInet']['gDriveID'],
    hash_dict_networkFiles=hash_dict_true,
    verbose=params['ROInet']['verbose'],
)

roinet.generate_dataloader(
    ROI_images=data.ROI_images,
    um_per_pixel=data.um_per_pixel,
    pref_plot=params['ROInet']['pref_plot'],
    batchSize_dataloader=params['ROInet']['batchSize_dataloader'],
    pinMemory_dataloader=params['ROInet']['pinMemory_dataloader'],
    numWorkers_dataloader=mp.cpu_count(),
    persistentWorkers_dataloader=params['ROInet']['persistentWorkers_dataloader'],
    prefetchFactor_dataloader=params['ROInet']['prefetchFactor_dataloader'],    
)

roinet.generate_latents();

gc.collect()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()
print(f'## Completed: Passing ROIs through ROInet')


print(f'## Starting: Passing ROIs through scattering wavelet transform')
# Scattering wavelet embedding distances
swt = scatteringWaveletTransformer.SWT(
    kwargs_Scattering2D=params['SWT']['kwargs_Scattering2D'], 
    image_shape=params['SWT']['image_shape'], 
    device=params['SWT']['device'],
)

swt.transform(ROI_images=np.concatenate(data.ROI_images, axis=0));
print(f'## Completed: Passing ROIs through scattering wavelet transform')

print(f'## Starting: Computing similarities')
# Compute similarities
sim = similarity_graph.ROI_graph(
    device=params['similarity']['device'],
    n_workers=params['similarity']['n_workers'],
    spatialFootprint_maskPower=params['similarity']['spatialFootprint_maskPower'],
    frame_height=data.FOV_height,
    frame_width=data.FOV_width,
    block_height=params['similarity']['block_height'],
    block_width=params['similarity']['block_width'],
    overlapping_width_Multiplier=params['similarity']['overlapping_width_Multiplier'],
    algorithm_nearestNeigbors_spatialFootprints=params['similarity']['algorithm_nearestNeigbors_spatialFootprints'],
    n_neighbors_nearestNeighbors_spatialFootprints=params['similarity']['n_neighbors_nearestNeighbors_spatialFootprints'],
    locality=params['similarity']['locality'],
    verbose=params['similarity']['verbose'],
)

sim.visualize_blocks()

sim.compute_similarity_blockwise(
    # spatialFootprints=blurrer.ROIs_blurred,
    spatialFootprints=aligner.ROIs_aligned,
    features_NN=roinet.latents,
    features_SWT=swt.latents,
    ROI_session_bool=data.sessionID_concat,
    linkage_methods=params['similarity_compute']['linkage_methods'],
    linkage_distances=helpers.bounded_logspace(*params['similarity_compute']['bounded_logspace_args']),
    min_cluster_size=params['similarity_compute']['min_cluster_size'],
    max_cluster_size=params['similarity_compute']['max_cluster_size'],
    batch_size_hashing=params['similarity_compute']['batch_size_hashing'],
);

sim.compute_cluster_similarity_graph(
        cluster_similarity_reduction_intra=params['similarity_compute']['cluster_similarity_reduction_intra'],
        cluster_similarity_reduction_inter=params['similarity_compute']['cluster_similarity_reduction_inter'],
        cluster_silhouette_reduction_intra=params['similarity_compute']['cluster_silhouette_reduction_intra'],
        cluster_silhouette_reduction_inter=params['similarity_compute']['cluster_silhouette_reduction_inter'],
        n_workers=params['similarity_compute']['n_workers'],
);

sim.compute_cluster_scores(
    power_clusterSize=params['similarity_compute']['power_clusterSize'], 
    power_clusterSilhouette=params['similarity_compute']['power_clusterSilhouette'],
);

fig, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].plot(sim.scores.cpu())
# plt.ylim([0,1.1])
axs[1].plot(sim.scores.cpu())
axs[1].set_yscale('log')

plt.figure()
plt.hist(sim.scores.cpu(), 500)
plt.yscale('log')
plt.xscale('log')

plt.figure()
plt.scatter((np.array(sim.cluster_bool.sum(1)).squeeze()), sim.scores, alpha=0.01)

gc.collect()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()
print(f'## Completed: Computing similarities')

print(f'## Starting: Optimizing ROI clusters')
ca= cluster_assignment.Cluster_Assigner(
    c=sim.c_sim,
    h=sim.cluster_bool.T,
    w=sim.scores,
    device=params['clusterAssigner']['device'],
    m_init=(torch.ones(sim.c_sim.shape[0])*-5 + torch.rand(sim.c_sim.shape[0])*1).type(torch.float32),
    optimizer_partial=functools.partial(torch.optim.Adam, lr=params['clusterAssigner']['optimizer_partial_lr'], betas=params['clusterAssigner']['optimizer_partial_betas']),
    scheduler_partial=functools.partial(torch.optim.lr_scheduler.CyclicLR, base_lr=params['clusterAssigner']['scheduler_partial_base_lr'], max_lr=params['clusterAssigner']['scheduler_partial_max_lr'], step_size_up=params['clusterAssigner']['scheduler_partial_step_size_up'], cycle_momentum=params['clusterAssigner']['scheduler_partial_cycle_momentum'], verbose=params['clusterAssigner']['scheduler_partial_verbose']),
    dmCEL_temp=params['clusterAssigner']['dmCEL_temp'],
    dmCEL_sigSlope=params['clusterAssigner']['dmCEL_sigSlope'],
    dmCEL_sigCenter=params['clusterAssigner']['dmCEL_sigCenter'],
    dmCEL_penalty=params['clusterAssigner']['dmCEL_penalty'],
    sampleWeight_softplusKwargs=params['clusterAssigner']['sampleWeight_softplusKwargs'],
    sampleWeight_penalty=params['clusterAssigner']['sampleWeight_penalty'],
    fracWeighted_goalFrac=params['clusterAssigner']['fracWeighted_goalFrac'],
    fracWeighted_sigSlope=params['clusterAssigner']['fracWeighted_sigSlope'],
    fracWeighted_sigCenter=params['clusterAssigner']['fracWeighted_sigCenter'],
    fracWeight_penalty=params['clusterAssigner']['fracWeight_penalty'],
    maskL1_penalty=params['clusterAssigner']['maskL1_penalty'],
    tol_convergence=params['clusterAssigner']['tol_convergence'],
    window_convergence=params['clusterAssigner']['window_convergence'],
    freqCheck_convergence=params['clusterAssigner']['freqCheck_convergence'],
    verbose=params['clusterAssigner']['verbose'],
)

ca.fit(
    min_iter=params['clusterAssigner_fit']['min_iter'],
    max_iter=params['clusterAssigner_fit']['max_iter'],
    verbose=params['clusterAssigner_fit']['verbose'], 
    verbose_interval=params['clusterAssigner_fit']['verbose_interval'],
)

ca.plot_loss()

# del clusterAssigner

gc.collect()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()

# clusterAssigner.plot_c_masked_matrix()

preds, confidence, scores_samples, m_bool = ca.predict(m_threshold=params['clusterAssigner_fit']['m_threshold'])
# preds, confidence, scores_samples, m_bool = clusterAssigner.predict(m_threshold=0.99)

ca.plot_clusterWeights()

ca.plot_sampleWeights()

ca.plot_clusterScores(bins=200)
# plt.xscale('log')
# plt.yscale('log')

fig, axs = ca.plot_labelCounts()
axs[0].set_ylim([0,20]);

print(f'Number of clusters: {m_bool.sum()}')

print(f'## Completed: Optimizing ROI clusters')


print(f'## Starting: Saving results')
preds_by_session = [preds[idx].numpy() for idx in data.sessionID_concat.T]

ROIs = {
    "ROIs_aligned": aligner.ROIs_aligned,
    "ROIs_raw": data.spatialFootprints,
    "frame_height": data.FOV_height,
    "frame_width": data.FOV_width,
    "idx_roi_session": [np.where(idx)[0] for idx in data.sessionID_concat.T]
}

name_save = os.path.split(dir_allOuterFolders)[-1]
helpers.simple_save(
    {
        "UCIDs": list(ca.preds.numpy().astype(np.int64)),
        "UCIDs_bySession": preds_by_session,
        "ROIs": ROIs,
    },
    filename=Path(dir_save) / (name_save + '.plane0.rClust' '.pkl'),
#     filename='/media/rich/bigSSD/analysis_data/mouse 2_6/multiday_alignment/UCIDs.pkl'
)
print(f'## Completed: Saving results')

gc.collect()
torch.cuda.empty_cache()
gc.collect()
torch.cuda.empty_cache()

print(f'## RUN COMPLETE. Total duration: {time.time() - t_start}')