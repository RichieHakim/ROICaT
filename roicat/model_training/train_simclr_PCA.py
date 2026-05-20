# Imports
import argparse
import sys
import os
import shutil
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import json
import pandas as pd
import argparse
import scipy.sparse
from pathlib import Path

from roicat import data_importing
from roicat import ROInet
from roicat.model_training import augmentation
from roicat.model_training import model
from roicat.model_training import simclr_training_helpers as sth

path_script = sys.argv[0]

# Parse arguments for data directory, parameters, and save directory
parser = argparse.ArgumentParser(
    prog='ROICaT SimCLR Training',
    description='This script runs the basic fit pipeline for a self-supervised ROI model using a json file containing the parameters.',
)
parser.add_argument(
    '--directory_data',
    '-d',
    required=False,
    metavar='',
    type=str,
    default=None,
    help='Path to raw ROI data to be used to train the model.',
)
parser.add_argument(
    '--path_params',
    '-p',
    required=False,
    metavar='',
    type=str,
    default=None,
    help='Path to json file containing parameters.',
)
parser.add_argument(
    '--directory_save',
    '-s',
    required=False,
    metavar='',
    type=str,
    default=None,
    help="Directory into which final model and evaluations should be saved.",
)
parser.add_argument(
    '--filepath_source_model',
    '-m',
    required=False,
    metavar='',
    type=str,
    default=None,
    help="Filepath with the .pth checkpoint (e.g. checkpoint_latest.pth) to be used for PCA training.",
)
parser.add_argument(
    '--test_option',
    '-t',
    required=False,
    metavar='',
    type=bool,
    default=False,
    help="Whether to run the script in test mode (True) or not (False). Reduces number of training examples to 3000.",
)
args = parser.parse_args()
directory_data = args.directory_data
filepath_params = args.path_params
directory_save = args.directory_save
filepath_source_model = args.filepath_source_model
test_option = args.test_option


# Load paths from directory_data and parameters from JSON
list_filepaths_data = [Path(os.path.join(directory_data, filename)) for filename in os.listdir(directory_data)]
with open(filepath_params) as f:
    dict_params = json.load(f)
## If directory_save is specified from the arg parser, overwrite the save directory in the params file
if directory_save is not None:
    dict_params['model']['filepath_model_wPCA'] = str(Path(directory_save) / (dict_params['model']['torchvision_model'] + '_' + 'trainingBest_wPCA.pth'))
    dict_params['model']['filepath_model_wPCA_params'] = str(Path(directory_save) / (dict_params['model']['torchvision_model'] + '_' + 'trainingBest_wPCA_params.json'))
## If filepath_source_model is specified from the arg parser, overwrite the source model in the params file
if filepath_source_model is not None:
    dict_params['model']['filepath_model_noPCA'] = filepath_source_model

assert dict_params['trainer']['forward_version'] == 'forward_head', "This script is only for training SimCLR with ."


# Load data from dir_data into Data object
ROI_sparse_all = [scipy.sparse.load_npz(filepath_ROI_images) for filepath_ROI_images in list_filepaths_data if filepath_ROI_images.suffix == '.npz']
ROI_images = [sf_sparse.toarray().reshape(sf_sparse.shape[0], 36,36).astype(np.float32) for sf_sparse in ROI_sparse_all]
data = data_importing.Data_roicat();

if test_option:
    ROI_images = [ROI_images[0][:300]]
    n_epochs = 2
else:
    n_epochs = dict_params['trainer']['n_epochs']

data.set_ROI_images(
    ROI_images=ROI_images,
    um_per_pixel=dict_params['data']['um_per_pixel'],
)


# Create dataset / dataloader without augmentations for PCA training
ROI_images_rs = ROInet.Resizer_ROI_images(
    nan_to_num=dict_params['data']['nan_to_num'],
    nan_to_num_val=dict_params['data']['nan_to_num_val'],
    verbose=dict_params['data']['verbose'],
).resize_ROIs(
    ROI_images=np.concatenate(data.ROI_images, axis=0),
    um_per_pixel=dict_params['data']['um_per_pixel'],
)

batchSize_dataloader = 64 if test_option else dict_params['dataloader']['batchSize_dataloader']

dataloader_generator = ROInet.Dataloader_ROInet(
    ROI_images_rs,
    batchSize_dataloader=batchSize_dataloader,
    pinMemory_dataloader=dict_params['dataloader']['pinMemory_dataloader'],
    numWorkers_dataloader=dict_params['dataloader']['numWorkers_dataloader'],
    persistentWorkers_dataloader=dict_params['dataloader']['persistentWorkers_dataloader'],
    prefetchFactor_dataloader=dict_params['dataloader']['prefetchFactor_dataloader'],
    transforms=None,
    n_transforms=2,
    img_size_out=tuple(dict_params['dataloader']['img_size_out']),
    jit_script_transforms=dict_params['dataloader']['jit_script_transforms'],
    shuffle_dataloader=dict_params['dataloader']['shuffle_dataloader'],
    drop_last_dataloader=dict_params['dataloader']['drop_last_dataloader'],
    verbose=dict_params['dataloader']['verbose'],
)
dataloader = dataloader_generator.dataloader


# Load Model
## Build architecture from params, then load weights from .pth checkpoint (mirrors ROInet.py:478-482).
model_container = model.Simclr_Model.from_dict_params(
    dict_params_model=dict_params['model'],
    base_model=torchvision.models.__dict__[dict_params['model']['torchvision_model']](weights='DEFAULT'),
    image_out_size=list(dataloader_generator.dataset[0][0][0].shape),
    forward_version=dict_params['trainer']['forward_version'],
    filepath_model_save=dict_params['model']['filepath_model_wPCA'],
)
filepath_source_pth = dict_params['model']['filepath_model_noPCA']
ckpt = torch.load(filepath_source_pth, map_location='cpu', weights_only=True)
## Simclr_Trainer checkpoints store the model state under 'state_dict'; tolerate bare state_dicts and the legacy 'model_state' key as well
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    model_state = ckpt['state_dict']
elif isinstance(ckpt, dict) and 'model_state' in ckpt:
    model_state = ckpt['model_state']
else:
    model_state = ckpt
model_container.model.load_state_dict(model_state)
model_container.model.eval()


# Specify dataloader and model conatainer for PCA training
trainer = sth.Simclr_PCA_Trainer(
    dataloader=dataloader,
    model_container=model_container
)

# Fit PCA layer and save as .pth + params.json
trainer.train()

## Build Simclr_Model_with_PCA from trainer's fitted attributes.
## pca_mean_fitted is the raw head mean BEFORE zeroing; pca_sklearn_fitted has mean_ zeroed.
net_wPCA = model.Simclr_Model_with_PCA.from_simclr_and_sklearn_pca(
    model_container=model_container,
    pca_sklearn=trainer.pca_sklearn_fitted,
    pca_mean=trainer.pca_mean_fitted,
    trainer_scale=trainer.scale,
)
net_wPCA.eval()

## Atomic write: tmp then replace
filepath_wPCA = dict_params['model']['filepath_model_wPCA']
filepath_wPCA_tmp = filepath_wPCA + '_tmp.pth'
torch.save(net_wPCA.state_dict(), filepath_wPCA_tmp)
os.replace(filepath_wPCA_tmp, filepath_wPCA)
print(f'Saved wPCA state dict to {filepath_wPCA}')

## Write sidecar params.json (flat; consumed by ROInet_embedder via make_model(**params))
## Keys must match make_model(**params) kwarg names in the bundled model.py.
## fwd_version is NOT included here: ROInet.py:470 passes it as an explicit kwarg,
## so including it in params.json would cause TypeError (duplicate keyword argument).
dict_params_wPCA = {
    'torchvision_model':       dict_params['model']['torchvision_model'],
    'head_pool_method':        dict_params['model']['head_pool_method'],
    'head_pool_method_kwargs': dict_params['model']['head_pool_method_kwargs'],
    'pre_head_fc_sizes':       dict_params['model']['pre_head_fc_sizes'],
    'post_head_fc_sizes':      dict_params['model']['post_head_fc_sizes'],
    'head_nonlinearity':       dict_params['model']['head_nonlinearity'],
    'head_nonlinearity_kwargs':dict_params['model']['head_nonlinearity_kwargs'],
    'block_to_unfreeze':       dict_params['model']['block_to_unfreeze'],
    'n_block_toInclude':       dict_params['model']['n_block_toInclude'],
    'pca_size':                dict_params['model']['pre_head_fc_sizes'][-1],
}
filepath_params_wPCA = dict_params['model']['filepath_model_wPCA_params']
filepath_params_wPCA_tmp = filepath_params_wPCA + '_tmp.json'
with open(filepath_params_wPCA_tmp, 'w') as f:
    json.dump(dict_params_wPCA, f, indent=2)
os.replace(filepath_params_wPCA_tmp, filepath_params_wPCA)
print(f'Saved wPCA params to {filepath_params_wPCA}')

## Copy model.py into the bundle dir so ROInet_embedder can load it via sys.path.append + bare `import model`.
## model.py is self-contained (no roicat imports); it exposes Simclr_Model_with_PCA + the make_model factory ROInet_embedder calls.
filepath_bundle_model = Path(filepath_wPCA).parent / 'model.py'
shutil.copy(str(Path(__file__).parent / 'model.py'), str(filepath_bundle_model))
print(f'Copied model.py to {filepath_bundle_model}')
