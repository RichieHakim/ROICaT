import sys
import time

import numpy as np
import torch
from bnpm import file_helpers, optimization
import sklearn.utils.class_weight
from torch import nn, optim
from tqdm import tqdm
import sklearn.linear_model
import multiprocessing as mp

import roicat.classification.classifier_util as cu
import scipy.sparse
import roicat
import bnpm.h5_handling

argv = sys.argv
# argv = ['/Users/josh/analysis/github_repos/ROICaT_paper/figure_classification_benchmarking/generate_preproc_simclr.py',
#         '/Users/josh/analysis/outputs/ROICaT_paper/simclr_syt_preproc/sample_dispatcher/jobNum_0/params.json',
#         '/Users/josh/analysis/outputs/ROICaT_paper/simclr_syt_preproc/sample_dispatcher/jobNum_0',]
# argv = [
#     '/Users/josh/analysis/github_repos/ROICaT_paper/figure_classification_benchmarking/generate_preproc_simclr.py',
#     '/Users/josh/analysis/github_repos/ROICaT_paper/figure_classification_benchmarking/DELETE/preproc_simclr/sample_run/jobNum_0/params.json',
#     '/Users/josh/analysis/github_repos/ROICaT_paper/figure_classification_benchmarking/DELETE/preproc_simclr/sample_run/jobNum_2/',
# ]


import sys
from pathlib import Path

path_script, path_params, directory_save = argv
directory_save = Path(directory_save)

import shutil
shutil.copy2(path_script, str(Path(directory_save) / Path(path_script).name));

tic = time.time()
tictoc = {}
tictoc['start'] = time.time() - tic

# Create parameters object to store all parameters
params = file_helpers.json_load(str(Path(path_params).resolve()))
params['device'] = torch.device('cpu') if not torch.cuda.is_available() else params['device']
assert params['method'] == 'preproc_simclr', 'This script is only for preprocessing data through the simclr model'
assert 'hyperparameters_training_simclr' in params, 'The simclr params.json file must include hyperparameters_training_simclr'
assert 'filename_labels' in params['paths'], 'JZ: The simclr params.json file must include paths.filename_labels'

directory_model = str(Path(params['paths']['directory_model']).resolve()) if 'directory_model' in params['paths'] else None
filepath_data_labels = str((Path(params['paths']['directory_data']) / params['paths']['filename_labels']).resolve())

if params['datatype'] == "stat_s2p":
    assert 'filename_stat' in params['paths'] and 'filename_ops' in params['paths'], 'JZ: The suite2p params.json file must include paths.filename_stat and paths.filename_ops for stat_s2p datatype'
    filepath_data_stat = str((Path(params['paths']['directory_data']) / params['paths']['filename_stat']).resolve())
    filepath_data_ops = str((Path(params['paths']['directory_data']) / params['paths']['filename_ops']).resolve())

    # Create data importing object to import suite2p data
    data = roicat.data_importing.Data_suite2p(
        paths_statFiles=[filepath_data_stat],
        paths_opsFiles=[filepath_data_ops],
        class_labels=[filepath_data_labels],
        um_per_pixel=params['hyperparameters_data']['um_per_pixel'],
        new_or_old_suite2p=params['hyperparameters_data']['new_or_old_suite2p'],
        out_height_width=params['hyperparameters_data']['out_height_width'],
        type_meanImg=params['hyperparameters_data']['type_meanImg'],
        FOV_images=params['hyperparameters_data']['FOV_images'],
        verbose=params['hyperparameters_data']['verbose'],
    )
elif params['datatype'] == "raw_images":
    assert 'filename_rawImages' in params['paths'], 'JZ: The suite2p params.json file must include paths.filename_rawImages for raw_images datatype'
    filepath_data_rawImages = str((Path(params['paths']['directory_data']) / params['paths']['filename_rawImages']).resolve())

    sf = scipy.sparse.load_npz(filepath_data_rawImages)
    labels = np.load(filepath_data_labels)

    data = roicat.data_importing.Data_roicat(verbose=True)
    data.set_ROI_images(ROI_images=[sf.A.reshape(sf.shape[0], 36, 36)], um_per_pixel=params['hyperparameters_data']['um_per_pixel'])
    data.set_class_labels(class_labels=[labels.astype(int)])
else:
    raise ValueError(f"Invalid datatype for simclr: {params['datatype']}")

tictoc['imported_data'] = time.time() - tic

ROI_images_rescaled = [roicat.ROInet.ROInet_embedder.resize_ROIs(rois, params['hyperparameters_data']['um_per_pixel']) for rois in data.ROI_images]

# Initialize concatendated data
ROI_images_init = np.concatenate(data.ROI_images, axis=0).astype(np.float32)
ROI_images_init_rescaled = np.concatenate(ROI_images_rescaled, axis=0).astype(np.float32)
labels_init = np.concatenate(data.class_labels, axis=0).astype(int).copy()

# Perform data cleaning
idx_violations = (np.isnan(ROI_images_init_rescaled.sum(axis=(1,2)))*1 + (np.sum(ROI_images_init_rescaled, axis=(1,2))==0)*1 + np.isnan(labels_init)) != 0
print('Number of idx_violations: ', idx_violations.sum(), ' out of ', len(idx_violations), ' total ROIs.')
print('Located at: ', np.where(idx_violations)[0])
print('Discarding these ROIs...')

ROI_images_filt = ROI_images_init_rescaled[~idx_violations]
labels_filt = labels_init[~idx_violations]

tictoc['cleaned_data'] = time.time() - tic

print(f'Shape of ROI_images_filt: {ROI_images_filt.shape}, shape of labels_remapped: {labels_filt.shape}')

# Create datasets and dataloaders to pass data through the transformations
transforms_final_train = cu.get_transforms(params['hyperparameters_augmentations_train'], scripted=True)
dataset_train = roicat.ROInet.dataset_simCLR(
        X=torch.as_tensor(ROI_images_filt, device='cpu', dtype=torch.float32),
        y=torch.as_tensor(labels_filt, device='cpu', dtype=torch.float32),
        n_transforms=1,
        class_weights=np.array([1]),
        transform=transforms_final_train,
        DEVICE='cpu',
        dtype_X=torch.float32,
)
dataloader_train = torch.utils.data.DataLoader( 
        dataset_train,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        num_workers=0,#mp.cpu_count(),
        persistent_workers=False,
        prefetch_factor=2,
)

transforms_final_val =cu.get_transforms(params['hyperparameters_augmentations_val'], scripted=True)
dataset_val = roicat.ROInet.dataset_simCLR(
        X=torch.as_tensor(ROI_images_filt, device='cpu', dtype=torch.float32),
        y=torch.as_tensor(labels_filt, device='cpu', dtype=torch.float32),
        n_transforms=1,
        class_weights=np.array([1]),
        transform=transforms_final_val, # *Use WarpPoints
        DEVICE='cpu',
        dtype_X=torch.float32,
    )
dataloader_val = torch.utils.data.DataLoader( 
        dataset_val,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        num_workers=0,#mp.cpu_count(),
        persistent_workers=False,
        prefetch_factor=2,
)

roinet = roicat.ROInet.ROInet_embedder(
    device=params['device'],
    dir_networkFiles=params['paths']['directory_simclrModel'],
    download_method='check_local_first',
    forward_pass_version='head',
    download_url=params['hyperparameters_training_simclr']['simclrModel_download_url'],
    download_hash=params['hyperparameters_training_simclr']['simclrModel_download_hash'],
    verbose=True,
)



# Dump once per epoch where an epoch = 
# Consistently dump to the same H5 file
# Able to append data to it well
print(f'Extracting transformed images from dataloaders, passing through roinet model, and saving to {directory_save}...')



features_val, labels_val, _idx_val, _sample_val = cu.extract_with_dataloader(
    dataloader_val,
    model=roinet.net,
    num_copies=1,
    device=params['device'],
)

print(f'Unaugmented run completed.')

bnpm.h5_handling.simple_save(
    dict_to_save={
        'latents_unaugmented': features_val,
        'labels': labels_val,
        'latents_augmented': {}
    },
    path=str((Path(directory_save) / 'dumped_simclr_passthroughs.h5').resolve()),
    write_mode='w-',
)

print(f'Saved to h5 file.')
