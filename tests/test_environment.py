import sys
from pathlib import Path
import re

import warnings


######################################################################################################################################
#################################################### TEST ROICAT MODULES #############################################################
######################################################################################################################################

def test_import_modules():
    """
    Test importing the inner modules of the roicat 
     package.
    This test expects a particular folder structure.
    """
    import roicat
    from roicat import classification, tracking, model_training, data_importing, helpers, ROInet
    # TODO: ## from roicat.classification import stuff
    from roicat.tracking import alignment, blurring, clustering, scatteringWaveletTransformer, similarity_graph, visualization
    from roicat.model_training import augmentation
    

######################################################################################################################################
################################################### TEST PACKAGE VERSIONS ############################################################
######################################################################################################################################

def test_torch(device='cuda', verbose=2):
    """
    Test to see if torch can do operations on GPU if CUDA is available.
    RH 2022

    Args:
        device (str):
            The device to use. Default is 'cuda'.
        verbose (int):
            If 0, do not print anything.
            If 1, print warnings.
            If 2, print all below and info.
    """
    import torch
    version = torch.__version__
    ## Check if CUDA is available
    if torch.cuda.is_available():
        print(f'FR: CUDA is available. Environment using PyTorch version: {version}') if verbose > 1 else None
        arr = torch.rand(1000, 10).to(device)
        arr2 = torch.rand(10, 1000).to(device)
        arr3 = (arr @ arr2).mean().cpu().numpy()
        print(f'FR: Torch can do basic operations on GPU. Environment using PyTorch version: {version}. Result of operations: {arr3}') if verbose > 1 else None

    else:
        warnings.warn(f'FR Warning: CUDA is not available. Environment using PyTorch version: {version}')
        
    ## Test CPU computations
    arr = torch.rand(1000, 10)
    arr2 = torch.rand(10, 1000)
    arr3 = (arr @ arr2).mean().numpy()
    print(f'FR: Torch can do basic operations on CPU. Environment using PyTorch version: {version}. Result of operations: {arr3}') if verbose > 1 else None
        

######################################################################################################################################
######################################################## TEST PYTHON #################################################################
######################################################################################################################################

# def test_python_version(
#     look_for_verion_in_environmentYaml=True, 
#     filename_environmentYaml='environment_GPU.yml',
#     fallback_version=3.9,
#     verbose=2,
# ):
#     """
#     Test python version.
#     Either use the version specified in environment.yaml file or
#      falls back to the version specified in fallback_version.
#     RH 2022

#     Args:
#         look_for_verion_in_environmentYaml (bool):
#             If True, look for the version in environment.yaml file.
#             If False, use the version specified in fallback_version.
#         filename_environmentYaml (str):
#             The name of the environment.yaml file.
#         fallback_version (float):
#             The version to use if look_for_verion_in_environmentYaml is False.
#             Set to None to raise an error if the version cannot be found.
#         verbose (int):
#             If 0, do not print anything.
#             If 1, print warnings.
#             If 2, print all below and info.

#     Returns:
#         success (bool): 
#             True if the version is correct, False otherwise.
#     """
#     ## find path to repo and environment.yml files
#     path_repo = str(Path(__file__).parent.parent)
#     print(path_repo) if verbose > 1 else None

#     ## check if environment.yml exists. If not, warn user and 
#     ##  use fallback version
#     if look_for_verion_in_environmentYaml:
#         path_envYaml = str(Path(path_repo) / filename_environmentYaml)
#         if not Path(path_envYaml).exists():
#             warnings.warn(f'FR Warning: {path_envYaml} does not exist. Using fallback version {fallback_version}')
#             if fallback_version is None:
#                 raise FileNotFoundError(f'FR Error: {path_envYaml} does not exist. Cannot find python version.')
#             version_test = fallback_version
#         else:
#             ## Read environment.yml file. Find string that contains python version.
#             ## Extract version number and convert to float
#             with open(path_envYaml, 'r') as f:
#                 lines = f.readlines()
#                 for line in lines:
#                     if re.search(' python=', line):
#                         version_test = float(line.split('=')[1])
#                         print(f'FR: Found environment specification for PYTHON VERSION {version_test} from line: {line} in {path_envYaml}') if verbose > 1 else None
#                         break
#     else:
#         assert fallback_version is not None, 'FR Error: fallback_version cannot be None if look_for_verion_in_environmentYaml is False.'
#         version_test = fallback_version

#     ## Test if python version and subversion are correct
#     version_system = float(f'{sys.version_info.major}.{sys.version_info.minor}')
#     print(f'FR: PYTHON VERSION system: {version_system}') if verbose > 1 else None
#     if version_test != version_system:
#         raise EnvironmentError(f'FR Error: PYTHON VERSION {version_system} does not match specification: {version_test}. Please check your environment.')
#     print(f'FR: PYTHON VERSION on system: {version_system} matches specification: {version_test}') if verbose > 1 else None

