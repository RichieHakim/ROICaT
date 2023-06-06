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

import classifier_util as cu
import simclr_util as su
import scipy.sparse
import roicat
import glob
import bnpm.h5_handling
import random

try:
    from bnpm import resource_tracking
except Exception as e:
    print(f'JZ: Error importing resource_tracking')
    print(e)

argv = sys.argv

import sys
from pathlib import Path

path_script, path_params, directory_save = argv
directory_save = Path(directory_save)

import shutil
try:
    shutil.copy2(path_script, str(Path(directory_save) / Path(path_script).name));
except Exception as e:
    print(f'JZ: Error copying script to {directory_save}')
    print(e)

try:
    Path(str((directory_save).resolve())).mkdir(exist_ok=True, parents=True)
    shutil.copy2(path_params, str(Path(directory_save) / Path(path_params).name));
except Exception as e:
    print(f'JZ: Error copying params to {directory_save}')
    print(e)


# try:
#     checker_cpu = resource_tracking.CPU_Device_Checker()
#     checker_cpu.track_utilization(interval=5, path_save=str((Path(directory_save) / 'cpu_utilization.csv').resolve()))
# except Exception as e:
#     print(f'JZ: Error initializing CPU checker')
#     print(e)
# try:
#     checker_gpu = resource_tracking.NVIDIA_Device_Checker()
#     checker_gpu.track_utilization(interval=5, path_save=str((Path(directory_save) / 'gpu_utilization.csv').resolve()))
# except Exception as e:
#     print(f'JZ: Error initializing GPU checker')
#     print(e)



tic = time.time()
tictoc = {}
tictoc['start'] = time.time() - tic

# Create parameters object to store all parameters
# params = cu.Parameters(path_params,)
params = file_helpers.json_load(str(Path(path_params).resolve()))
params['device'] = torch.device('cpu') if not torch.cuda.is_available() else params['device']

assert params['method'] == 'simclr', 'This script is only for the simclr model'
assert 'hyperparameters_training_simclr' in params, 'The simclr params.json file must include hyperparameters_training_simclr'
assert 'filename_labels' in params['paths'], 'JZ: The simclr params.json file must include paths.filename_labels'


# filepath_data_labels_pred = str((Path(params['paths']['directory_data']) / params['paths']['filename_labels_pred']).resolve()) if 'filename_labels_pred' in params['paths'] else None
directory_model = str(Path(params['paths']['directory_model']).resolve()) if 'directory_model' in params['paths'] else None
filepath_data_labels = str((Path(params['paths']['directory_data']) / params['paths']['filename_labels']).resolve())


filepath_features_search = str((Path(params['paths']['directory_simclrPreproc']) / '*' / "dumped_simclr_passthroughs.h5").resolve())
list_globFilepath_features = glob.glob(filepath_features_search)

print('Loading H5 files...')
# Extract key useful constants from one of the h5 files
lst_h5s_loaded = []
for filepath_features in list_globFilepath_features:
    try:
        h5_file = bnpm.h5_handling.simple_load(filepath_features, return_dict=False, verbose=True)
        lst_h5s_loaded.append(h5_file)
    except Exception as e:
        print(f'JZ: Error loading {filepath_features}')
        print(e)

print('Extracting constants from H5 files...')
lst_labels = [h5['labels'][:] for h5 in lst_h5s_loaded if 'labels' in list(h5.keys())]
for labels in lst_labels:
    assert np.all(labels == lst_labels[0]), 'JZ: All h5 files must have the same labels'
lst_numLabels = [h5['labels'].shape[0] if 'labels' in list(h5.keys()) else np.nan for h5 in lst_h5s_loaded]
lst_numROIsAugmented = [h5['latents_augmented']['batch_1'].shape[0] if 'latents_augmented' in list(h5.keys()) and 'batch_1' in list(h5['latents_augmented'].keys()) else np.nan for h5 in lst_h5s_loaded]
lst_latentDimSizeAugmented = [h5['latents_augmented']['batch_1'].shape[1] if 'latents_augmented' in list(h5.keys()) and 'batch_1' in list(h5['latents_augmented'].keys()) else np.nan for h5 in lst_h5s_loaded]
lst_numROIsUnaugmented = [h5['latents_unaugmented'].shape[0] if 'latents_unaugmented' in list(h5.keys()) else np.nan for h5 in lst_h5s_loaded]
lst_latentDimSizeUnaugmented = [h5['latents_unaugmented'].shape[1] if 'latents_unaugmented' in list(h5.keys()) else np.nan for h5 in lst_h5s_loaded]
lst_numTotalAugmentations = [len(list(h5['latents_augmented'].keys())) if 'latents_augmented' in list(h5.keys()) and 'batch_1' in list(h5['latents_augmented'].keys()) else np.nan for h5 in lst_h5s_loaded]
lst_numTotalAugmentations_cumsum = np.cumsum([_ if not np.isnan(_) else 0 for _ in lst_numTotalAugmentations])

numLabels = np.unique([numLabels for numLabels in lst_numLabels if numLabels is not np.nan])
numROIsAugmented = np.unique([numROIsAugmented for numROIsAugmented in lst_numROIsAugmented if numROIsAugmented is not np.nan])
numROIsUnaugmented = np.unique([numROIsUnaugmented for numROIsUnaugmented in lst_numROIsUnaugmented if numROIsUnaugmented is not np.nan])
latentDimSizeAugmented = np.unique([latentDimSizeAugmented for latentDimSizeAugmented in lst_latentDimSizeAugmented if latentDimSizeAugmented is not np.nan])
latentDimSizeUnaugmented = np.unique([latentDimSizeUnaugmented for latentDimSizeUnaugmented in lst_latentDimSizeUnaugmented if latentDimSizeUnaugmented is not np.nan])

print('Assertion checks from H5 files...')
# Check that all h5 files have the same number of labels
assert len(numLabels) == 1, f'JZ: All h5 files must have the same number of labels. Instead found: {numLabels}'
assert len(numROIsAugmented) == 1, f'JZ: All h5 files must have the same number of ROIs. Instead found: {numROIsAugmented}'
assert len(numROIsUnaugmented) == 1, f'JZ: All h5 files must have the same number of ROIs. Instead found: {numROIsUnaugmented}'
assert len(latentDimSizeAugmented) == 1, f'JZ: All h5 files must have the same latent dimension size. Instead found: {latentDimSizeAugmented}'
assert len(latentDimSizeUnaugmented) == 1, f'JZ: All h5 files must have the same latent dimension size. Instead found: {latentDimSizeUnaugmented}'

# Check that the number of ROIs in augmented and unaugmented files match
assert numROIsAugmented == numROIsUnaugmented, 'JZ: The number of ROIs in augmented and unaugmented files must match'
assert latentDimSizeAugmented == latentDimSizeUnaugmented, 'JZ: The latent dimension size in augmented and unaugmented files must match'

# Check that the numbers of labels matches the number of ROIs
assert numLabels == numROIsAugmented, 'JZ: The number of labels must match the number of ROIs'

# Extract the single values associated with the np unique
numLabels = numLabels[0]
numROIsAugmented = numROIsAugmented[0]
numROIsUnaugmented = numROIsUnaugmented[0]
latentDimSizeAugmented = latentDimSizeAugmented[0]
latentDimSizeUnaugmented = latentDimSizeUnaugmented[0]

print('Splitting data...')
# Create data splitting object for stratified sampling into train and test sets (as well as downsampling)
data_split = cu.Datasplit(
    features=np.arange(numROIsAugmented),
    labels=labels,
    n_train=params['hyperparameters_split']['n_train'],
    test_size=params['hyperparameters_split']['test_size'],
)


print('Calculating augmentation counts...')
num_augmentations = int(np.ceil(200000 / data_split.features_train_subset.shape[0]))
if num_augmentations > lst_numTotalAugmentations_cumsum[-1]:
    print(f'JZ: The number of augmentations to request ({num_augmentations}) is not greater than the number of augmentations in the data ({lst_numTotalAugmentations_cumsum[-1]}). Using the maximum available amount.')
    num_augmentations = lst_numTotalAugmentations_cumsum[-1]

print('Creating X and y matrices for training data...')
X_train = np.ones((len(data_split.features_train_subset), num_augmentations, latentDimSizeAugmented))*np.nan
y_train = np.ones((len(data_split.features_train_subset), 1))*np.nan

for inx_ROI_dest, inx_ROI_src in enumerate(data_split.features_train_subset):
    inx_augmentationsUnselected = list(np.arange(lst_numTotalAugmentations_cumsum[-1]))
    random.shuffle(inx_augmentationsUnselected)
    for inx_augmentation_dest in range(num_augmentations):
        # Randomly sample an augmentation from the data, identify which n h5 file that augmentation number falls into, and then load that augmentation number from the appropriate entry in lst_h5s_loaded
        # inx_augmentationNumOverall = np.random.randint(low=0, high=lst_numTotalAugmentations_cumsum[-1])
        inx_augmentationNumOverall = inx_augmentationsUnselected.pop()
        inx_augmentationFileNum_src = np.where(lst_numTotalAugmentations_cumsum > inx_augmentationNumOverall)[0][0]
        inx_augmentationNumInFile_src = inx_augmentationNumOverall - lst_numTotalAugmentations_cumsum[inx_augmentationFileNum_src - 1] if inx_augmentationFileNum_src > 0 else inx_augmentationNumOverall
        latents_augmented_singleBatch = lst_h5s_loaded[inx_augmentationFileNum_src]['latents_augmented'][f'batch_{inx_augmentationFileNum_src + 1}'][inx_ROI_src]
        latents_augmented_singleBatch = latents_augmented_singleBatch.reshape(latents_augmented_singleBatch.shape[0])
        X_train[inx_ROI_dest, inx_augmentation_dest, :] = latents_augmented_singleBatch

    inx_labelsFileNum_src = np.where(np.array(lst_numLabels) > 0)[0][0]
    labels_singleBatch = lst_h5s_loaded[inx_labelsFileNum_src]['labels'][inx_ROI_src]
    y_train[inx_ROI_dest, 0] = labels_singleBatch

y_train = np.tile(y_train, (1, num_augmentations))
y_train = y_train.astype(int)

print('Creating X and y matrices for validation data...')
X_val = np.ones((len(data_split.features_val), 1, latentDimSizeAugmented))*np.nan
y_val = np.ones((len(data_split.features_val), 1))*np.nan

for inx_ROI_dest, inx_ROI_src in enumerate(data_split.features_val):
    inx_augmentationFileNum_src = np.where(np.array(lst_numROIsUnaugmented) > 0)[0][0]
    latents_unaugmented_singleBatch = lst_h5s_loaded[inx_augmentationFileNum_src]['latents_unaugmented'][inx_ROI_src]
    latents_unaugmented_singleBatch = latents_unaugmented_singleBatch.reshape(latents_unaugmented_singleBatch.shape[0])
    X_val[inx_ROI_dest, 0, :] = latents_unaugmented_singleBatch

    inx_labelsFileNum_src = np.where(np.array(lst_numLabels) > 0)[0][0]
    labels_singleBatch = lst_h5s_loaded[inx_labelsFileNum_src]['labels'][inx_ROI_src]
    y_val[inx_ROI_dest, 0] = labels_singleBatch

X_val = np.tile(X_val, (1, num_augmentations, 1))
y_val = np.tile(y_val, (1, num_augmentations))

y_val = y_val.astype(int)


# list_latents_augmented_allJobs = []
# # Loop through all possible h5 sources of features and load them
# for filepath_features in list_globFilepath_features:
#     h5_features = bnpm.h5_handling.simple_load(filepath_features, return_dict=False, verbose=True)
#     if 'latents_unaugmented' in h5_features:
#         latents_unaugmented = h5_features['latents_unaugmented']    
#     if 'labels' in h5_features:
#         labels = h5_features['labels']
#     latents_augmented_singleJob = np.stack([np.array(latents_augmented_singleBatch) for _key_singleBatch, latents_augmented_singleBatch in h5_features['latents_augmented'].items()], axis=1)
#     list_latents_augmented_allJobs.append(latents_augmented_singleJob)

print('Closing H5 files...')
[h5_single.close() for h5_single in lst_h5s_loaded]

print('Remapping labels...')
y_train = cu.remap_labels(y_train, params['label_remapping'])
y_val = cu.remap_labels(y_val, params['label_remapping'])
# latents_augmented = np.concatenate(list_latents_augmented_allJobs, axis=1)

tictoc['loaded_data'] = time.time() - tic

print('Calculating class weights...')
# Compute class-associated variables
num_classes = len(np.unique(labels))
class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)

labels_train = y_train.reshape(-1) # np.stack([data_split.labels_train_subset]*latents_augmented.shape[1], axis=1).reshape(-1)
features_train = X_train.reshape(-1, X_train.shape[-1]) # latents_augmented[data_split.features_train_subset].reshape(-1, latents_augmented.shape[2])

labels_val = y_val.reshape(-1) # data_split.labels_val
features_val = X_val.reshape(-1, X_val.shape[-1]) # latents_unaugmented[data_split.features_val]

n_train_actual = data_split.n_train_actual

# features_train = features_train.cpu().detach().numpy().astype(int)
# labels_train = labels_train.cpu().detach().numpy().astype(int)

# print('Passing images through net...')
# features_train = roinet.net(ROI_images_aug_train)
# features_val =  roinet.net(ROI_images_aug_val)

tictoc['splitted_data'] = time.time() - tic

print(f'Fitting model to data of dimensions: X: {X_train.shape}, y: {y_train.shape}...')
# Create lenet model, associated optimizer, loss function, and training tracker
model = sklearn.linear_model.LogisticRegression(
   solver=params['hyperparameters_training_simclr']['solver'],
   fit_intercept=params['hyperparameters_training_simclr']['fit_intercept'],
   max_iter=params['hyperparameters_training_simclr']['max_iter'],
   C=params['hyperparameters_training_simclr']['C'],
   class_weight={iClassWeight:classWeight for iClassWeight, classWeight in enumerate(class_weights)},
#    class_weight=class_weights,
)
model.fit(features_train, labels_train)

print(f'Calculating tracker outputs and saving to {directory_save}...')
training_tracker = cu.TrainingTracker(
    directory_save=directory_save,
    class_weights=class_weights, # Class Weights
    tictoc=tictoc, # Time Tracker
    n_train_actual=n_train_actual,
    model=({'coef':model.coef_, 'intercept':model.intercept_})
)

y_train_preds = model.predict(features_train).astype(int)
y_train_true = labels_train
y_val_preds = model.predict(features_val).astype(int)
y_val_true = labels_val

# Save training loop results from current epoch for training set
training_tracker.add_accuracy(0, 'accuracy_training', y_train_true, y_train_preds) # Generating training loss
training_tracker.add_confusion_matrix(0, 'confusionMatrix_training', y_train_true, y_train_preds) # Generating confusion matrix

# Save training loop results from current epoch for validation set
training_tracker.add_accuracy(0, 'accuracy_val', y_val_true, y_val_preds) # Generating validation accuracy
training_tracker.add_confusion_matrix(0, 'confusionMatrix_val', y_val_true, y_val_preds) # Generating validation confusion matrix

tictoc[f'completed_training_in_{0}'] = time.time() - tic

training_tracker.save_results()
training_tracker.print_results()
