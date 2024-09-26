from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import copy
import pickle
import re
import zipfile
import gc
from functools import partial
import typing
import tkinter as tk
import PIL
from PIL import ImageTk
import csv
import warnings
from typing import List, Dict, Tuple, Union, Optional, Any, Callable, Iterable, Sequence, Type, Any, MutableMapping

import numpy as np
import torch
import torchvision
import scipy.sparse
import sparse
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import yaml

"""
All of these are from basic_neural_processing_modules
"""

######################################################################################################################################
####################################################### TORCH HELPERS ################################################################
######################################################################################################################################


def set_device(
    use_GPU: bool = True, 
    device_num: int = 0, 
    device_types: List[str] = ['cuda', 'mps', 'xpu', 'cpu'],
    verbose: bool = True
) -> str:
    """
    Sets the device for PyTorch. If a GPU is available and **use_GPU** is
    ``True``, it will be set as the device. Otherwise, the CPU will be set as
    the device. 
    RH 2022

    Args:
        use_GPU (bool): 
            Determines if the GPU should be utilized: \n
            * ``True``: the function will attempt to use the GPU if a GPU is
              not available.
            * ``False``: the function will use the CPU. \n
            (Default is ``True``)
        device_num (int): 
            Specifies the index of the GPU to use. (Default is ``0``)
        device_types (List[str]):
            The types and order of devices to attempt to use. The first device
            type that is available will be used. Options are ``'cuda'``,
            ``'mps'``, ``'xpu'``, and ``'cpu'``.
        verbose (bool): 
            Determines whether to print the device information. \n
            * ``True``: the function will print out the device information.
            \n
            (Default is ``True``)

    Returns:
        (str): 
            device (str): 
                A string specifying the device, either *"cpu"* or
                *"cuda:<device_num>"*.
    """
    devices = list_available_devices()

    if not use_GPU:
        device = 'cpu'
    else:
        device = None
        for device_type in device_types:
            if len(devices[device_type]) > 0:
                device = devices[device_type][device_num]
                break

    if verbose:
        print(f'Using device: {device}')

    return device
    

def list_available_devices() -> dict:
    """
    Lists all available PyTorch devices on the system.
    RH 2024

    Returns:
        (dict): 
            A dictionary with device types as keys and lists of available devices as values.
    """
    devices = {}

    # Check for CPU devices
    if torch.cpu.is_available():
        devices['cpu'] = ['cpu']
    else:
        devices['cpu'] = []

    # Check for CUDA devices
    if torch.cuda.is_available():
        devices['cuda'] = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    else:
        devices['cuda'] = []

    # Check for MPS devices
    if torch.backends.mps.is_available():
        devices['mps'] = ['mps:0']
    else:
        devices['mps'] = []

    # Check for XPU devices
    if hasattr(torch, 'xpu'):
        if torch.xpu.is_available():
            devices['xpu'] = [f'xpu:{i}' for i in range(torch.xpu.device_count())]
        else:
            devices['xpu'] = []
    else:
        devices['xpu'] = []

    return devices


######################################################################################################################################
###################################################### CONTAINER HELPERS #############################################################
######################################################################################################################################


def merge_dicts(
    dicts: List[dict]
) -> dict:
    """
    Merges a list of dictionaries into a single dictionary.
    RH 2022
    
    Args:
        dicts (List[dict]): 
            List of dictionaries to merge.

    Returns:
        (dict): 
            result_dict (dict): 
                A single dictionary that contains all keys and values from the
                dictionaries in the input list.
    """
    out = {}
    [out.update(d) for d in dicts]
    return out 


def deep_update_dict(
    dictionary: dict, 
    key: List[str], 
    val: Any, 
    in_place: bool = False
) -> Union[dict, None]:
    """
    Updates a nested dictionary with a new value.
    RH 2023

    Args:
        dictionary (dict): 
            The original dictionary to update.
        key (List[str]): 
            List of keys representing the hierarchical path to the nested value
            to update. Each element should be a string that represents a level
            in the hierarchy. For example, to change a value in the dictionary
            `params` at key 'dataloader_kwargs' and subkey 'prefetch_factor', you would 
            pass `['dataloader_kwargs', 'prefetch_factor']`.
        val (Any): 
            The new value to set in the dictionary.
        in_place (bool): 
            * ``True``: the original dictionary will be updated in-place and no
              value will be returned. 
            * ``False``, a new dictionary will be created and returned. (Default
              is ``False``)

    Returns:
        (Union[dict, None]): 
            updated_dict (dict): 
                The updated dictionary. Only returned if ``in_place`` is ``False``.
                
    Example:
        .. highlight:: python
        .. code-block:: python

            original_dict = {"level1": {"level2": "old value"}}
            updated_dict = deep_update_dict(original_dict, ["level1", "level2"], "new value", in_place=False)
            # Now updated_dict is {"level1": {"level2": "new value"}}
    """
    def helper_deep_update_dict(d, key, val):
        if type(key) is str:
            key = [key]

        assert key[0] in d, f"RH ERROR, key: '{key[0]}' is not found"

        if type(key) is list:
            if len(key) > 1:
                helper_deep_update_dict(d[key[0]], key[1:], val)
            elif len(key) == 1:
                key = key[0]
                d.update({key:val})

    if in_place:
        helper_deep_update_dict(dictionary, key, val)
    else:
        d = copy.deepcopy(dictionary)
        helper_deep_update_dict(d, key, val)
        return d
        

def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='.') -> MutableMapping:
    """
    Flattens a dictionary of dictionaries into a single dictionary. NOTE: Turns
    all keys into strings. Stolen from https://stackoverflow.com/a/6027615.
    RH 2022

    Args:
        d (Dict):
            Dictionary to flatten
        parent_key (str):
            Key to prepend to flattened keys IGNORE: USED INTERNALLY FOR
            RECURSION
        sep (str):
            Separator to use between keys IGNORE: USED INTERNALLY FOR RECURSION

    Returns:
        (Dict):
            flattened dictionary (dict):
                Flat dictionary with the keys to deeper dictionaries joined by
                the separator.
    """

    items = []
    for k, v in d.items():
        new_key = str(parent_key) + str(sep) + str(k) if parent_key else str(k)
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

## parameter dictionary helpers ##

def fill_in_dict(
    d: Dict, 
    defaults: Dict,
    verbose: bool = True,
    hierarchy: List[str] = ['dict'], 
):
    """
    In-place. Fills in dictionary ``d`` with values from ``defaults`` if they
    are missing. Works hierachically.
    RH 2023

    Args:
        d (Dict):
            Dictionary to fill in.
            In-place.
        defaults (Dict):
            Dictionary of defaults.
        verbose (bool):
            Whether to print messages.
        hierarchy (List[str]):
            Used internally for recursion.
            Hierarchy of keys to d.
    """
    from copy import deepcopy
    for key in defaults:
        if key not in d:
            print(f"Key '{key}' not found in params dictionary: {' > '.join([f'{str(h)}' for h in hierarchy])}. Using default value: {defaults[key]}") if verbose else None
            d.update({key: deepcopy(defaults[key])})
        elif isinstance(defaults[key], dict):
            assert isinstance(d[key], dict), f"Key '{key}' is a dict in defaults, but not in params. {' > '.join([f'{str(h)}' for h in hierarchy])}."
            fill_in_dict(d[key], defaults[key], hierarchy=hierarchy+[key])
            

def check_keys_subset(d, default_dict, hierarchy=['defaults']):
    """
    Checks that the keys in d are all in default_dict. Raises an error if not.
    RH 2023

    Args:
        d (Dict):
            Dictionary to check.
        default_dict (Dict):
            Dictionary containing the keys to check against.
        hierarchy (List[str]):
            Used internally for recursion.
            Hierarchy of keys to d.
    """
    default_keys = list(default_dict.keys())
    for key in d.keys():
        assert key in default_keys, f"Parameter '{key}' not found in defaults dictionary: {' > '.join([f'{str(h)}' for h in hierarchy])}."
        if isinstance(default_dict[key], dict) and isinstance(d[key], dict):
            check_keys_subset(d[key], default_dict[key], hierarchy=hierarchy+[key])


def prepare_params(params, defaults, verbose=True):
    """
    Does the following:
        * Checks that all keys in ``params`` are in ``defaults``.
        * Fills in any missing keys in ``params`` with values from ``defaults``.
        * Returns a deepcopy of the filled-in ``params``.

    Args:
        params (Dict):
            Dictionary of parameters.
        defaults (Dict):
            Dictionary of defaults.
        verbose (bool):
            Whether to print messages.
    """
    from copy import deepcopy
    ## Check inputs
    assert isinstance(params, dict), f"p must be a dict. Got {type(params)} instead."
    ## Make sure all the keys in p are valid
    check_keys_subset(params, defaults)
    ## Fill in any missing keys with defaults
    params_out = deepcopy(params)
    fill_in_dict(params_out, defaults, verbose=verbose)

    return params_out


######################################################################################################################################
####################################################### MATH FUNCTIONS ###############################################################
######################################################################################################################################


def generalised_logistic_function(
    x: Union[np.ndarray, torch.Tensor], 
    a: float = 0, 
    k: float = 1, 
    b: float = 1, 
    v: float = 1, 
    q: float = 1, 
    c: float = 1,
    mu: float = 0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculates the **generalized logistic function**.

    Refer to `Generalised logistic function
    <https://en.wikipedia.org/wiki/Generalised_logistic_function>`_ for detailed
    information on the parameters. 
    RH 2021

    Args:
        x (Union[np.ndarray, torch.Tensor]): 
            The input to the logistic function.
        a (float): 
            The lower asymptote. (Default is *0*)
        k (float): 
            The upper asymptote when ``c=1``. (Default is *1*)
        b (float): 
            The growth rate. (Default is *1*)
        v (float): 
            Should be greater than *0*, it affects near which asymptote maximum growth
            occurs. (Default is *1*)
        q (float): 
            Related to the value Y(0). Center positions. (Default is *1*)
        c (float): 
            Typically takes a value of *1*. (Default is *1*)
        mu (float): 
            The center position of the function. (Default is *0*)

    Returns:
        (Union[np.ndarray, torch.Tensor]): 
            out (Union[np.ndarray, torch.Tensor]):
                The value of the logistic function for the input ``x``.
    """
    if type(x) is np.ndarray:
        exp = np.exp
    elif type(x) is torch.Tensor:
        exp = torch.exp
    return a + (k-a) / (c + q*exp(-b*(x-mu)))**(1/v)


def bounded_logspace(
    start: float, 
    stop: float, 
    num: int,
) -> np.ndarray:
    """
    Creates a **logarithmically spaced array**, similar to ``np.logspace``, but
    with a defined start and stop. 
    RH 2022

    Args:
        start (float): 
            The first value in the output array.
        stop (float): 
            The last value in the output array.
        num (int): 
            The number of values in the output array.
            
    Returns:
        (np.ndarray): 
            out (np.ndarray): 
                An array of logarithmically spaced values between ``start`` and
                ``stop``.
    """
    exp = 2  ## doesn't matter what this is, just needs to be > 1

    return exp ** np.linspace(np.log(start)/np.log(exp), np.log(stop)/np.log(exp), num, endpoint=True)


def make_odd(n, mode='up'):
    """
    Make a number odd.
    RH 2023

    Args:
        n (int):
            Number to make odd
        mode (str):
            'up' or 'down'
            Whether to round up or down to the nearest odd number

    Returns:
        output (int):
            Odd number
    """
    if n % 2 == 0:
        if mode == 'up':
            return n + 1
        elif mode == 'down':
            return n - 1
        else:
            raise ValueError("mode must be 'up' or 'down'")
    else:
        return n
def make_even(n, mode='up'):
    """
    Make a number even.
    RH 2023

    Args:
        n (int):
            Number to make even
        mode (str):
            'up' or 'down'
            Whether to round up or down to the nearest even number

    Returns:
        output (int):
            Even number
    """
    if n % 2 != 0:
        if mode == 'up':
            return n + 1
        elif mode == 'down':
            return n - 1
        else:
            raise ValueError("mode must be 'up' or 'down'")
    else:
        return n


######################################################################################################################################
####################################################### CLASSIFICATION ###############################################################
######################################################################################################################################


def confusion_matrix(
    y_hat: np.ndarray, 
    y_true: np.ndarray, 
    counts: bool = False,
) -> np.ndarray:
    """
    Computes the confusion matrix from ``y_hat`` and ``y_true``. ``y_hat``
    should be either predictions or probabilities.
    RH 2022
    
    Args:
        y_hat (np.ndarray): 
            Numpy array of predictions or probabilities. Either \n
            * 1D array of predictions *(n_samples,)*. Values should be integers.
            * 2D array of probabilities *(n_samples, n_classes)*. Values should
              be floats. \n
            (Default is 1D array of predictions)
        y_true (np.ndarray):
            Numpy array of ground truth labels. Either \n
            * 1D array of labels *(n_samples,)*. Values should be integers.
            * 2D array of one-hot labels *(n_samples, n_classes)*. Values should
              be integers. \n
            (Default is 1D array of labels)
        counts (bool):
            If ``False``, the output confusion matrix is normalized. If
            ``True``, the output contains counts. (Default is ``False``)
            
    Returns:
        (np.ndarray): 
            cmat (np.ndarray): 
                The computed confusion matrix.
    """
    n_classes = max(np.max(y_true)+1, np.max(y_hat)+1)
    if y_hat.ndim == 1:
        y_hat = idx_to_oneHot(y_hat, n_classes).astype('int')
    cmat = y_hat.T @ idx_to_oneHot(y_true, n_classes)
    if not counts:
        cmat = cmat / np.sum(cmat, axis=0)[None,:]
    return cmat


def squeeze_integers(
    intVec: Union[list, np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Makes integers in an array consecutive numbers starting from the smallest
    value. For example, [7,2,7,4,-1,0] -> [3,2,3,1,-1,0]. This is useful for
    removing unused class IDs. 
    RH 2023
    
    Args:
        intVec (Union[list, np.ndarray, torch.Tensor]):
            1-D array of integers.
    
    Returns:
        (Union[np.ndarray, torch.Tensor]): 
            squeezed_integers (Union[np.ndarray, torch.Tensor]): 
                1-D array of integers with consecutive numbers starting from the
                smallest value.
    """
    if isinstance(intVec, list):
        intVec = np.array(intVec, dtype=np.int64)
    if isinstance(intVec, np.ndarray):
        unique, arange = np.unique, np.arange
    elif isinstance(intVec, torch.Tensor):
        unique, arange = torch.unique, torch.arange
        
    u, inv = unique(intVec, return_inverse=True)  ## get unique values and their indices
    u_min = u.min()  ## get the smallest value
    u_s = arange(u_min, u_min + u.shape[0], dtype=u.dtype)  ## make consecutive numbers starting from the smallest value
    return u_s[inv]  ## return the indexed consecutive unique values


######################################################################################################################################
######################################################### OPTIMIZATION ###############################################################
######################################################################################################################################

class Convergence_checker_optuna:
    """
    Checks if the optuna optimization has converged.
    RH 2023

    Args:
        n_patience (int): 
            Number of trials to look back to check for convergence. 
            Also the minimum number of trials that must be completed 
            before starting to check for convergence. 
            (Default is *10*)
        tol_frac (float): 
            Fractional tolerance for convergence. 
            The best output value must change by less than this 
            fractional amount to be considered converged. 
            (Default is *0.05*)
        max_trials (int): 
            Maximum number of trials to run before stopping. 
            (Default is *350*)
        max_duration (float): 
            Maximum number of seconds to run before stopping. 
            (Default is *600*)
        value_stop (Optional[float]):
            Value at which to stop the optimization. If the best value is equal
            to or less than this value, the optimization will stop.
            (Default is *None*)
        verbose (bool): 
            If ``True``, print messages. 
            (Default is ``True``)

    Attributes:
        bests (List[float]):
            List to hold the best values obtained in the trials.
        best (float):
            Best value obtained among the trials. Initialized with infinity.

    Example:
        .. highlight:: python
        .. code-block:: python

            # Create a ConvergenceChecker instance
            convergence_checker = ConvergenceChecker(
                n_patience=15, 
                tol_frac=0.01, 
                max_trials=500, 
                max_duration=60*20, 
                verbose=True
            )
            
            # Assume we have a study and trial objects from optuna
            # Use the check method in the callback
            study.optimize(objective, n_trials=100, callbacks=[convergence_checker.check])    
    """
    def __init__(
        self, 
        n_patience: int = 10, 
        tol_frac: float = 0.05, 
        max_trials: int = 350, 
        max_duration: float = 60*10, 
        value_stop: Optional[float] = None,
        verbose: bool = True,
    ):
        """
        Initializes the ConvergenceChecker with the given parameters.
        """
        self.bests = []
        self.best = np.inf
        self.n_patience = n_patience
        self.tol_frac = tol_frac
        self.max_trials = max_trials
        self.max_duration = max_duration
        self.value_stop = value_stop
        self.num_trial = 0
        self.verbose = verbose
        
    def check(
        self, 
        study: object, 
        trial: object,
    ):
        """
        Checks if the optuna optimization has converged. This function should be
        used as the callback function for the optuna study.

        Args:
            study (optuna.study.Study): 
                Optuna study object.
            trial (optuna.trial.FrozenTrial): 
                Optuna trial object.
        """
        dur_first, dur_last = study.trials[0].datetime_complete, trial.datetime_complete
        if (dur_first is not None) and (dur_last is not None):
            duration = (dur_last - dur_first).total_seconds()
        else:
            duration = 0
        
        if trial.value < self.best:
            self.best = trial.value
        self.bests.append(self.best)
            
        bests_recent = np.unique(self.bests[-self.n_patience:])
        if self.best == 0:
            print(f'Stopping. Best value is 0.') if self.verbose else None
            study.stop()
        elif self.num_trial > self.n_patience and ((np.abs(bests_recent.max() - bests_recent.min()) / np.abs(self.best)) < self.tol_frac):
            print(f'Stopping. Convergence reached. Best value ({self.best*10000}) over last ({self.n_patience}) trials fractionally changed less than ({self.tol_frac})') if self.verbose else None
            study.stop()
        elif self.num_trial >= self.max_trials:
            print(f'Stopping. Trial number limit reached. num_trial={self.num_trial}, max_trials={self.max_trials}.') if self.verbose else None
            study.stop()
        elif duration > self.max_duration:
            print(f'Stopping. Duration limit reached. study.duration={duration}, max_duration={self.max_duration}.') if self.verbose else None
            study.stop()

        if self.value_stop is not None:
            if self.best <= self.value_stop:
                print(f'Stopping. Best value ({self.best}) is less than or equal to value_stop ({self.value_stop}).') if self.verbose else None
                study.stop()
            
        if self.verbose:
            print(f'Trial num: {self.num_trial}. Duration: {duration:.3f}s. Best value: {self.best:3e}. Current value:{trial.value:3e}') if self.verbose else None
        self.num_trial += 1


######################################################################################################################################
######################################################## FEATURIZATION ###############################################################
######################################################################################################################################


def idx_to_oneHot(
    arr: Union[np.ndarray, torch.Tensor], 
    n_classes: Optional[int] = None, 
    dtype: Optional[Type] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Converts an array of class labels to a matrix of one-hot vectors. 
    RH 2021

    Args:
        arr (Union[np.ndarray, torch.Tensor]): 
            A 1-D array of class labels. Values should be integers >= 0. Values
            will be used as indices in the output array.
        n_classes (Optional[int]):
            The number of classes. If ``None``, it will be derived from ``arr``.
            (Default is ``None``)
        dtype (Optional[Type]):
            The data type of the output array. If ``None``, it defaults to bool
            for numpy array and torch.bool for Torch tensor. (Default is
            ``None``)

    Returns:
        (Union[np.ndarray, torch.Tensor]): 
            oneHot (Union[np.ndarray, torch.Tensor]):
                A 2-D array of one-hot vectors.
    """
    if type(arr) is np.ndarray:
        max = np.max
        zeros = np.zeros
        arange = np.arange
        dtype = np.bool_ if dtype is None else dtype
    elif type(arr) is torch.Tensor:
        max = torch.max
        zeros = torch.zeros
        arange = torch.arange
        dtype = torch.bool if dtype is None else dtype
    assert arr.ndim == 1

    if n_classes is None:
        n_classes = max(arr)+1
    oneHot = zeros((len(arr), n_classes), dtype=dtype)
    oneHot[arange(len(arr)), arr] = True
    return oneHot


def cosine_kernel_2D(
    center: Tuple[int, int] = (5, 5), 
    image_size: Tuple[int, int] = (11, 11), 
    width: int = 5,
) -> np.ndarray:
    """
    Generates a 2D cosine kernel. RH 2021

    Args:
        center (Tuple[int, int]):  
            The mean position (X, Y) where high value is expected. It is 0-indexed. 
            Make the second value 0 to make it 1D. (Default is *(5, 5)*)
        image_size (Tuple[int, int]): 
            The total image size (width, height). Make the second value 0 to
            make it 1D. (Default is *(11, 11)*)
        width (int): 
            The full width of one cycle of the cosine. (Default is *5*)

    Returns:
        (np.ndarray): 
            k_cos (np.ndarray):
                2D or 1D array of the cosine kernel.
    """
    x, y = np.meshgrid(range(image_size[1]), range(image_size[0]))  # note dim 1:X and dim 2:Y
    dist = np.sqrt((y - int(center[1])) ** 2 + (x - int(center[0])) ** 2)
    dist_scaled = (dist/(width/2))*np.pi
    dist_scaled[np.abs(dist_scaled > np.pi)] = np.pi
    k_cos = (np.cos(dist_scaled) + 1)/2
    return k_cos


######################################################################################################################################
########################################################## INDEXING ##################################################################
######################################################################################################################################

def idx2bool(
    idx: np.ndarray, 
    length: Optional[int] = None,
) -> np.ndarray:
    """
    Converts a vector of indices to a boolean vector.
    RH 2021

    Args:
        idx (np.ndarray): 
            1-D array of indices.
        length (Optional[int]): 
            Length of boolean vector. If ``None``, the length will be set to the
            maximum index in ``idx`` + 1. (Default is ``None``)
    
    Returns:
        (np.ndarray):
            bool_vec (np.ndarray):
                1-D boolean array.
    """
    if length is None:
        length = np.uint64(np.max(idx) + 1)
    out = np.zeros(length, dtype=np.bool_)
    out[idx] = True
    return out


def make_batches(
    iterable: Iterable, 
    batch_size: Optional[int] = None, 
    num_batches: Optional[int] = None, 
    min_batch_size: int = 0, 
    return_idx: bool = False, 
    length: Optional[int] = None
) -> Iterable:
    """
    Creates batches from an iterable.
    RH 2021

    Args:
        iterable (Iterable): 
            The iterable to be batched.
        batch_size (Optional[int]): 
            The size of each batch. If ``None``, then the batch size is based on
            ``num_batches``. (Default is ``None``)
        num_batches (Optional[int]): 
            The number of batches to create. (Default is ``None``)
        min_batch_size (int): 
            The minimum size of each batch. (Default is ``0``)
        return_idx (bool): 
            If ``True``, return the indices of the batches. Output will be
            [start, end] idx. (Default is ``False``)
        length (Optional[int]): 
            The length of the iterable. If ``None``, then the length is
            len(iterable). This is useful if you want to make batches of
            something that doesn't have a __len__ method. (Default is ``None``)
    
    Returns:
        (Iterable):
            output (Iterable):
                Batches of the iterable.
    """
    if length is None:
        l = len(iterable)
    else:
        l = length
    
    if batch_size is None:
        batch_size = np.int64(np.ceil(l / num_batches))
    
    for start in range(0, l, batch_size):
        end = min(start + batch_size, l)
        if (end-start) < min_batch_size:
            break
        else:
            if return_idx:
                yield iterable[start:end], [start, end]
            else:
                yield iterable[start:end]


def sparse_to_dense_fill(
    arr_s: sparse.COO, 
    fill_val: float = 0.
) -> np.ndarray:
    """
    Converts a **sparse** array to a **dense** array and fills in sparse entries with a specified fill value.
    RH 2023

    Args:
        arr_s (sparse.COO): 
            Sparse array to be converted to dense.
        fill_val (float): 
            Value to fill the sparse entries. (Default is ``0.0``)

    Returns:
        (np.ndarray): 
            dense_arr (np.ndarray):
                Dense version of the input sparse array.
    """
    import sparse
    s = sparse.COO(arr_s)
    s.fill_value = fill_val
    return s.todense()


def sparse_mask(
    x: scipy.sparse.csr_matrix, 
    mask_sparse: scipy.sparse.csr_matrix, 
    do_safety_steps: bool = True
) -> scipy.sparse.csr_matrix:
    """
    Masks a **sparse matrix** with the non-zero elements of another sparse
    matrix.
    RH 2022

    Args:
        x (scipy.sparse.csr_matrix):
            Sparse matrix to mask.
        mask_sparse (scipy.sparse.csr_matrix):
            Sparse matrix to mask with.
        do_safety_steps (bool):
            Whether to do safety steps to ensure that things are working as
            expected. (Default is ``True``)

    Returns:
        (scipy.sparse.csr_matrix):
            output (scipy.sparse.csr_matrix):
                Masked sparse matrix.
    """
    if do_safety_steps:
        m = mask_sparse.copy()
        m.eliminate_zeros()
    else:
        m = mask_sparse
    return (m!=0).multiply(x)


class scipy_sparse_csr_with_length(scipy.sparse.csr_matrix):
    """
    A scipy sparse matrix with a **length** attribute.
    RH 2023

    Attributes:
        length (int):
            The length of the matrix (shape[0])

    Args:
        *args (object):
            Arbitrary arguments passed to scipy.sparse.csr_matrix.
        **kwargs (object):
            Arbitrary keyword arguments passed to scipy.sparse.csr_matrix.
    """
    def __init__(
        self, 
        *args: object, 
        **kwargs: object
    ):
        """
        Initializes the scipy_sparse_csr_with_length with the given arguments and keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.length = self.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        return self.__class__(super().__getitem__(key))


class lazy_repeat_obj():
    """
    Makes a lazy iterator that repeats an object.
    RH 2021

    Args:
        obj (Any):
            Object to repeat.
        pseudo_length (Optional[int]):
            Length of the iterator. (Default is ``None``).
    """
    def __init__(
        self, 
        obj: Any, 
        pseudo_length: Optional[int] = None,
    ):
        """
        Initializes the lazy iterator.
        """
        self.obj = obj
        self.pseudo_length = pseudo_length

    def __getitem__(self, i: int) -> Any:
        """
        Get item at index `i`. Always returns the repeated object, unless index
        is out of bounds.
        
        Args:
            i (int):
                Index of item to return. Ignored if pseudo_length is None.
        
        Returns:
            Any: The repeated object.

        Raises:
            IndexError: If `i` is out of bounds.
        """
        if self.pseudo_length is None:
            return self.obj
        elif i < self.pseudo_length:
            return self.obj
        else:
            raise IndexError('Index out of bounds')

    def __len__(self) -> int:
        """
        Get the length of the iterator.

        Returns:
            int or None: 
                The length of the iterator.
        """
        return self.pseudo_length

    def __repr__(self):
        return repr(self.item)


def find_nonredundant_idx(
    s: scipy.sparse.coo_matrix,
) -> np.ndarray:
    """
    Finds the indices of the nonredundant entries in a sparse matrix. Useful
    when manually populating a sparse matrix and you want to know which entries
    have already been populated.
    RH 2022

    Args:
        s (scipy.sparse.coo_matrix):
            Sparse matrix. Should be in COO format.

    Returns:
        (np.ndarray): 
            idx_unique (np.ndarray):
                Indices of the nonredundant entries.
    """
    if s.getformat() != 'coo':
        s = s.coo()
    idx_rowCol = np.vstack((s.row, s.col)).T
    u, idx_u = np.unique(idx_rowCol, axis=0, return_index=True)
    return idx_u

def remove_redundant_elements(
    s: scipy.sparse.coo_matrix, 
    inPlace: bool = False,
) -> scipy.sparse.coo_matrix:
    """
    Removes redundant entries from a sparse matrix. Useful when manually
    populating a sparse matrix and you want to remove redundant entries.
    RH 2022

    Args:
        s (scipy.sparse.coo_matrix):
            Sparse matrix. Should be in COO format.
        inPlace (bool):
            * If ``True``, the input matrix is modified in place.
            * If ``False``, a new matrix is returned. \n
            (Default is ``False``)

    Returns:
        (scipy.sparse.coo_matrix):
            s (scipy.sparse.coo_matrix):
                Sparse matrix with redundant entries removed.
    """
    idx_nonRed = find_nonredundant_idx(s)
    if inPlace:
        s_out = s
    else:
        s_out = scipy.sparse.coo_matrix(s.shape)
    s_out.row = s.row[idx_nonRed]
    s_out.col = s.col[idx_nonRed]
    s_out.data = s.data[idx_nonRed]
    return s_out

def merge_sparse_arrays(
        s_list: List[scipy.sparse.csr_matrix], 
        idx_list: List[np.ndarray], 
        shape_full: Tuple[int, int], 
        remove_redundant: bool = True, 
        elim_zeros: bool = True
) -> scipy.sparse.csr_matrix:
    """
    Merges a list of square sparse arrays into a single square sparse array.
    Redundant entries are not selected; only entries chosen by np.unique are kept.

    Args:
        s_list (List[scipy.sparse.csr_matrix]):
            List of sparse arrays to merge. Each array can be any shape.
        idx_list (List[np.ndarray]):
            List of integer arrays. Each array should be the same length as its
            corresponding array in s_list and contain integers in the range [0,
            shape_full[0]). These integers represent the row/column indices in
            the full array.
        shape_full (Tuple[int, int]):
            Shape of the full array.
        remove_redundant (bool):
            * ``True``: Removes redundant entries from the output array. 
            * ``False``: Keeps redundant entries.
        elim_zeros (bool):
            * ``True``: Eliminate zeros in the sparse matrix. 
            * ``False``: Keeps zeros.

    Returns:
        scipy.sparse.csr_matrix:
            s_full (scipy.sparse.csr_matrix):
                Full sparse matrix merged from the input list.
    """
    row, col, data = np.array([]), np.array([]), np.array([])
    for s, idx in zip(s_list, idx_list):
        s_i = s.tocsr() if s.getformat() != 'csr' else s
        s_i.eliminate_zeros() if elim_zeros else s_i
        idx_grid = np.meshgrid(idx, idx)
        row = np.concatenate([row, (s_i != 0).multiply(idx_grid[0]).data])
        col = np.concatenate([col, (s_i != 0).multiply(idx_grid[1]).data])
        data = np.concatenate([data, s_i.data])
    s_full = scipy.sparse.coo_matrix((data, (row, col)), shape=shape_full)
    if remove_redundant:
        remove_redundant_elements(s_full, inPlace=True)
    return s_full


def scipy_sparse_to_torch_coo(
    sp_array: scipy.sparse.coo_matrix, 
    dtype: Optional[type] = None
) -> torch.sparse_coo_tensor:
    """
    Converts a Scipy sparse array to a PyTorch sparse COO tensor.

    Args:
        sp_array (scipy.sparse.coo_matrix):
            Scipy sparse array to be converted to a PyTorch sparse COO tensor.
        dtype (Optional[type]):
            Data type to which the values of the input sparse array are to be
            converted before creating the PyTorch sparse tensor. If ``None``, 
            the data type of the input array's values is retained. 
            (Default is ``None``).

    Returns:
        coo_tensor (torch.sparse_coo_tensor):
            PyTorch sparse COO tensor converted from the input Scipy sparse array.
    """
    import torch

    coo = scipy.sparse.coo_matrix(sp_array)
    
    values = coo.data
    # print(values.dtype)
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    # v = torch.FloatTensor(values)
    v = torch.as_tensor(values, dtype=dtype) if dtype is not None else values
    shape = coo.shape

    return torch.sparse_coo_tensor(i, v, torch.Size(shape))


def pydata_sparse_to_torch_coo(
    sp_array: object,
) -> object:
    """
    Converts a PyData Sparse array to a PyTorch sparse COO tensor.

    This function extracts the coordinates and data from the sparse PyData array
    and uses them to create a new sparse COO tensor in PyTorch.

    Args:
        sp_array (object): 
            The PyData Sparse array to convert. It should be a COO sparse matrix 
            representation. 

    Returns:
        (object): 
            coo_tensor (object): 
                The converted PyTorch sparse COO tensor.
                
    Example:
        .. highlight:: python
        .. code-block:: python

            sp_array = sparse.COO(np.random.rand(1000, 1000))
            coo_tensor = pydata_sparse_to_torch_coo(sp_array)
    """
    coo = sparse.COO(sp_array)
    
    values = coo.data
#     indices = np.vstack((coo.row, coo.col))
    indices = coo.coords

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))


def index_with_nans(values, indices):
    """
    Indexes an array with a list of indices, allowing for NaNs in the indices.
    RH 2022
    
    Args:
        values (np.ndarray):
            Array to be indexed.
        indices (Union[List[int], np.ndarray]):
            1D list or array of indices to use for indexing. Can contain NaNs.
            Datatype should be floating point. NaNs will be removed and values
            will be cast to int.

    Returns:
        np.ndarray:
            Indexed array. Positions where `indices` was NaN will be filled with
            NaNs.
    """
    indices = np.array(indices, dtype=float) if not isinstance(indices, np.ndarray) else indices
    values = np.concatenate((np.full(shape=values.shape[1:], fill_value=np.nan, dtype=values.dtype)[None,...], values), axis=0)
    idx = indices.copy() + 1
    idx[np.isnan(idx)] = 0
    
    return values[idx.astype(np.int64)]


######################################################################################################################################
######################################################## FILE HELPERS ################################################################
######################################################################################################################################

def get_nums_from_string(string_with_nums):
    """
    Return the numbers from a string as an int
    RH 2021-2022

    Args:
        string_with_nums (str):
            String with numbers in it
    
    Returns:
        nums (int):
            The numbers from the string    
            If there are no numbers, return None.        
    """
    idx_nums = [ii in str(np.arange(10)) for ii in string_with_nums]
    
    nums = []
    for jj, val in enumerate(idx_nums):
        if val:
            nums.append(string_with_nums[jj])
    if not nums:
        return None
    nums = int(''.join(nums))
    return nums


def find_paths(
    dir_outer: Union[str, List[str]],
    reMatch: str = 'filename', 
    reMatch_in_path: Optional[str] = None,
    find_files: bool = True, 
    find_folders: bool = False, 
    depth: int = 0, 
    natsorted: bool = True, 
    alg_ns: Optional[str] = None,
    verbose: bool = False,
) -> List[str]:
    """
    Searches for files and/or folders recursively in a directory using a regex
    match. 
    RH 2022

    Args:
        dir_outer (Union[str, List[str]]):
            Path(s) to the directory(ies) to search. If a list of directories,
            then all directories will be searched.
        reMatch (str): 
            Regular expression to match. Each file or folder name encountered
            will be compared using ``re.search(reMatch, filename)``. If the
            output is not ``None``, the file will be included in the output.
        reMatch_in_path (Optional[str]):
            Additional regular expression to match anywhere in the path. Useful
            for finding files/folders in specific subdirectories. If ``None``, then
            no additional matching is done. \n
            (Default is ``None``)
        find_files (bool): 
            Whether to find files. (Default is ``True``)
        find_folders (bool): 
            Whether to find folders. (Default is ``False``)
        depth (int): 
            Maximum folder depth to search. (Default is *0*). \n
            * depth=0 means only search the outer directory. 
            * depth=2 means search the outer directory and two levels of
              subdirectories below it
        natsorted (bool): 
            Whether to sort the output using natural sorting with the natsort
            package. (Default is ``True``)
        alg_ns (str): 
            Algorithm to use for natural sorting. See ``natsort.ns`` or
            https://natsort.readthedocs.io/en/4.0.4/ns_class.html/ for options.
            Default is PATH. Other commons are INT, FLOAT, VERSION. (Default is
            ``None``)
        verbose (bool):
            Whether to print the paths found. (Default is ``False``)

    Returns:
        (List[str]): 
            paths (List[str]): 
                Paths to matched files and/or folders in the directory.
    """
    import natsort
    if alg_ns is None:
        alg_ns = natsort.ns.PATH

    def fn_match(path, reMatch, reMatch_in_path):
        # returns true if reMatch is basename and reMatch_in_path in full dirname
        if reMatch is not None:
            if re.search(reMatch, os.path.basename(path)) is None:
                return False
        if reMatch_in_path is not None:
            if re.search(reMatch_in_path, os.path.dirname(path)) is None:
                return False
        return True

    def get_paths_recursive_inner(dir_inner, depth_end, depth=0):
        paths = []
        for path in os.listdir(dir_inner):
            path = os.path.join(dir_inner, path)
            if os.path.isdir(path):
                if find_folders:
                    if fn_match(path, reMatch, reMatch_in_path):
                        print(f'Found folder: {path}') if verbose else None
                        paths.append(path)
                if depth < depth_end:
                    paths += get_paths_recursive_inner(path, depth_end, depth=depth+1)
            else:
                if find_files:
                    if fn_match(path, reMatch, reMatch_in_path):
                        print(f'Found file: {path}') if verbose else None
                        paths.append(path)
        return paths

    def fn_check_pathLike(obj):
        if isinstance(obj, (
            str,
            Path,
            os.PathLike,
            np.str_,
            bytes,
            memoryview,
            np.bytes_,
            re.Pattern,
            re.Match,
        )):
            return True
        else:
            return False            

    dir_outer = [dir_outer] if fn_check_pathLike(dir_outer) else dir_outer

    paths = list(set(sum([get_paths_recursive_inner(str(d), depth, depth=0) for d in dir_outer], start=[])))
    if natsorted:
        paths = natsort.natsorted(paths, alg=alg_ns)
    return paths


def prepare_path(
    path: str, 
    mkdir: bool = False, 
    exist_ok: bool = True,
) -> str:
    """
    Checks if a directory or file path is valid for different purposes: 
    saving, loading, etc.
    RH 2023

    * If exists:
        * If exist_ok=True: all good
        * If exist_ok=False: raises error
    * If doesn't exist:
        * If file:
            * If parent directory exists:
                * All good
            * If parent directory doesn't exist:
                * If mkdir=True: creates parent directory
                * If mkdir=False: raises error
        * If directory:
            * If mkdir=True: creates directory
            * If mkdir=False: raises error
            
    RH 2023

    Args:
        path (str): 
            Path to be checked.
        mkdir (bool): 
            If ``True``, creates parent directory if it does not exist. 
            (Default is ``False``)
        exist_ok (bool): 
            If ``True``, allows overwriting of existing file. 
            (Default is ``True``)

    Returns:
        (str): 
            path (str):
                Resolved path.
    """
    ## check if path is valid
    try:
        path_obj = Path(path).resolve()
    except FileNotFoundError as e:
        print(f'Invalid path: {path}')
        raise e
    
    ## check if path object exists
    flag_exists = path_obj.exists()

    ## determine if path is a directory or file
    if flag_exists:
        flag_dirFileNeither = 'dir' if path_obj.is_dir() else 'file' if path_obj.is_file() else 'neither'  ## 'neither' should never happen since path.is_file() or path.is_dir() should be True if path.exists()
        assert flag_dirFileNeither != 'neither', f'Path: {path} is neither a file nor a directory.'
        assert exist_ok, f'{path} already exists and exist_ok=False.'
    else:
        flag_dirFileNeither = 'dir' if path_obj.suffix == '' else 'file'  ## rely on suffix to determine if path is a file or directory

    ## if path exists and is a file or directory
    # all good. If exist_ok=False, then this should have already been caught above.
    
    ## if path doesn't exist and is a file
    ### if parent directory exists        
    # all good
    ### if parent directory doesn't exist
    #### mkdir if mkdir=True and raise error if mkdir=False
    if not flag_exists and flag_dirFileNeither == 'file':
        if Path(path).parent.exists():
            pass ## all good
        elif mkdir:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        else:
            assert False, f'File: {path} does not exist, Parent directory: {Path(path).parent} does not exist, and mkdir=False.'
        
    ## if path doesn't exist and is a directory
    ### mkdir if mkdir=True and raise error if mkdir=False
    if not flag_exists and flag_dirFileNeither == 'dir':
        if mkdir:
            Path(path).mkdir(parents=True, exist_ok=True)
        else:
            assert False, f'{path} does not exist and mkdir=False.'

    ## if path is neither a file nor a directory
    ### raise error
    if flag_dirFileNeither == 'neither':
        assert False, f'{path} is neither a file nor a directory. This should never happen. Check this function for bugs.'

    return str(path_obj)

def prepare_filepath_for_saving(
    filepath: str, 
    mkdir: bool = False, 
    allow_overwrite: bool = True
) -> str:
    """
    Prepares a file path for saving a file. Ensures the file path is valid and has the necessary permissions. 

    Args:
        filepath (str): 
            The file path to be prepared for saving.
        mkdir (bool): 
            If set to ``True``, creates parent directory if it does not exist. (Default is ``False``)
        allow_overwrite (bool): 
            If set to ``True``, allows overwriting of existing file. (Default is ``True``)

    Returns:
        (str): 
            path (str): 
                The prepared file path for saving.
    """
    return prepare_path(filepath, mkdir=mkdir, exist_ok=allow_overwrite)
def prepare_filepath_for_loading(
    filepath: str, 
    must_exist: bool = True
) -> str:
    """
    Prepares a file path for loading a file. Ensures the file path is valid and has the necessary permissions. 

    Args:
        filepath (str): 
            The file path to be prepared for loading.
        must_exist (bool): 
            If set to ``True``, the file at the specified path must exist. (Default is ``True``)

    Returns:
        (str): 
            path (str): 
                The prepared file path for loading.
    """
    path = prepare_path(filepath, mkdir=False, exist_ok=must_exist)
    if must_exist:
        assert Path(path).is_file(), f'{path} is not a file.'
    return path
def prepare_directory_for_saving(
    directory: str, 
    mkdir: bool = False, 
    exist_ok: bool = True
) -> str:
    """
    Prepares a directory path for saving a file. This function is rarely used.

    Args:
        directory (str): 
            The directory path to be prepared for saving.
        mkdir (bool): 
            If set to ``True``, creates parent directory if it does not exist. (Default is ``False``)
        exist_ok (bool): 
            If set to ``True``, allows overwriting of existing directory. (Default is ``True``)

    Returns:
        (str): 
            path (str): 
                The prepared directory path for saving.
    """
    return prepare_path(directory, mkdir=mkdir, exist_ok=exist_ok)
def prepare_directory_for_loading(
    directory: str, 
    must_exist: bool = True
) -> str:
    """
    Prepares a directory path for loading a file. This function is rarely used.

    Args:
        directory (str): 
            The directory path to be prepared for loading.
        must_exist (bool): 
            If set to ``True``, the directory at the specified path must exist. (Default is ``True``)

    Returns:
        (str): 
            path (str): 
                The prepared directory path for loading.
    """
    path = prepare_path(directory, mkdir=False, exist_ok=must_exist)
    if must_exist:
        assert Path(path).is_dir(), f'{path} is not a directory.'
    return path


def pickle_save(
    obj: Any, 
    filepath: str, 
    mode: str = 'wb', 
    zipCompress: bool = False, 
    mkdir: bool = False, 
    allow_overwrite: bool = True,
    **kwargs_zipfile: Dict[str, Any],
) -> None:
    """
    Saves an object to a pickle file using `pickle.dump`.
    Allows for zipping of the file.

    RH 2022

    Args:
        obj (Any): 
            The object to save.
        filepath (str): 
            The path to save the object to.
        mode (str): 
            The mode to open the file in. Options are: \n
            * ``'wb'``: Write binary.
            * ``'ab'``: Append binary.
            * ``'xb'``: Exclusive write binary. Raises FileExistsError if the
              file already exists. \n
            (Default is ``'wb'``)
        zipCompress (bool): 
            If ``True``, compresses pickle file using zipfileCompressionMethod,
            which is similar to ``savez_compressed`` in numpy (with
            ``zipfile.ZIP_DEFLATED``). Useful for saving redundant and/or sparse
            arrays objects. (Default is ``False``)
        mkdir (bool): 
            If ``True``, creates parent directory if it does not exist. (Default
            is ``False``)
        allow_overwrite (bool): 
            If ``True``, allows overwriting of existing file. (Default is
            ``True``)
        kwargs_zipfile (Dict[str, Any]): 
            Keyword arguments that will be passed into `zipfile.ZipFile`.
            compression=``zipfile.ZIP_DEFLATED`` by default.
            See https://docs.python.org/3/library/zipfile.html#zipfile-objects.
            Other options for 'compression' are (input can be either int or object): \n
                * ``0``:  zipfile.ZIP_STORED (no compression)
                * ``8``:  zipfile.ZIP_DEFLATED (usual zip compression)
                * ``12``: zipfile.ZIP_BZIP2 (bzip2 compression) (usually not as
                  good as ZIP_DEFLATED)
                * ``14``: zipfile.ZIP_LZMA (lzma compression) (usually better
                  than ZIP_DEFLATED but slower)
    """
    path = prepare_filepath_for_saving(filepath, mkdir=mkdir, allow_overwrite=allow_overwrite)

    if len(kwargs_zipfile)==0:
        kwargs_zipfile = {
            'compression': zipfile.ZIP_DEFLATED,
        }

    if zipCompress:
        with zipfile.ZipFile(path, 'w', **kwargs_zipfile) as f:
            f.writestr('data', pickle.dumps(obj))
    else:
        with open(path, mode) as f:
            pickle.dump(obj, f)

def pickle_load(
    filepath: str, 
    zipCompressed: bool = False,
    mode: str = 'rb',
) -> Any:
    """
    Loads an object from a pickle file.
    RH 2022

    Args:
        filepath (str): 
            Path to the pickle file.
        zipCompressed (bool): 
            If ``True``, the file is assumed to be a .zip file. The function
            will first unzip the file, then load the object from the unzipped
            file. 
            (Default is ``False``)
        mode (str): 
            The mode to open the file in. (Default is ``'rb'``)

    Returns:
        (Any): 
            obj (Any): 
                The object loaded from the pickle file.
    """
    path = prepare_filepath_for_loading(filepath, must_exist=True)
    if zipCompressed:
        with zipfile.ZipFile(path, 'r') as f:
            return pickle.loads(f.read('data'))
    else:
        with open(path, mode) as f:
            return pickle.load(f)

def json_save(
    obj: Any, 
    filepath: str, 
    indent: int = 4, 
    mode: str = 'w', 
    mkdir: bool = False, 
    allow_overwrite: bool = True,
) -> None:
    """
    Saves an object to a json file using `json.dump`.
    RH 2022

    Args:
        obj (Any): 
            The object to save.
        filepath (str): 
            The path to save the object to.
        indent (int): 
            Number of spaces for indentation in the output json file. (Default
            is *4*)
        mode (str): 
            The mode to open the file in. Options are: \n
            * ``'wb'``: Write binary.
            * ``'ab'``: Append binary.
            * ``'xb'``: Exclusive write binary. Raises FileExistsError if the
              file already exists. \n
            (Default is ``'w'``)
        mkdir (bool): 
            If ``True``, creates parent directory if it does not exist. (Default
            is ``False``)
        allow_overwrite (bool): 
            If ``True``, allows overwriting of existing file. (Default is
            ``True``)
    """
    import json
    path = prepare_filepath_for_saving(filepath, mkdir=mkdir, allow_overwrite=allow_overwrite)
    with open(path, mode) as f:
        json.dump(obj, f, indent=indent)

def json_load(
    filepath: str, 
    mode: str = 'r',
) -> Any:
    """
    Loads an object from a json file.
    RH 2022

    Args:
        filepath (str): 
            Path to the json file.
        mode (str): 
            The mode to open the file in. (Default is ``'r'``)

    Returns:
        (Any): 
            obj (Any): 
                The object loaded from the json file.
    """
    import json
    path = prepare_filepath_for_loading(filepath, must_exist=True)
    with open(path, mode) as f:
        return json.load(f)


def yaml_save(
    obj: object, 
    filepath: str, 
    indent: int = 4, 
    mode: str = 'w', 
    mkdir: bool = False, 
    allow_overwrite: bool = True,
) -> None:
    """
    Saves an object to a YAML file using the ``yaml.dump`` method.
    RH 2022

    Args:
        obj (object): 
            The object to be saved.
        filepath (str): 
            Path to save the object to.
        indent (int): 
            The number of spaces for indentation in the saved YAML file.
            (Default is *4*)
        mode (str): 
            Mode to open the file in. \n
            * ``'w'``: write (default)
            * ``'wb'``: write binary
            * ``'ab'``: append binary
            * ``'xb'``: exclusive write binary. Raises ``FileExistsError`` if
              file already exists. \n
            (Default is ``'w'``)
        mkdir (bool): 
            If ``True``, creates the parent directory if it does not exist.
            (Default is ``False``)
        allow_overwrite (bool): 
            If ``True``, allows overwriting of existing files. (Default is
            ``True``)
    """
    path = prepare_filepath_for_saving(filepath, mkdir=mkdir, allow_overwrite=allow_overwrite)
    with open(path, mode) as f:
        yaml.dump(obj, f, indent=indent, sort_keys=False)

def yaml_load(
    filepath: str, 
    mode: str = 'r', 
    loader: object = yaml.FullLoader,
) -> object:
    """
    Loads a YAML file.
    RH 2022

    Args:
        filepath (str): 
            Path to the YAML file to load.
        mode (str): 
            Mode to open the file in. (Default is ``'r'``)
        loader (object): 
            The YAML loader to use. \n
            * ``yaml.FullLoader``: Loads the full YAML language. Avoids
              arbitrary code execution. (Default for PyYAML 5.1+)
            * ``yaml.SafeLoader``: Loads a subset of the YAML language, safely.
              This is recommended for loading untrusted input.
            * ``yaml.UnsafeLoader``: The original Loader code that could be
              easily exploitable by untrusted data input.
            * ``yaml.BaseLoader``: Only loads the most basic YAML. All scalars
              are loaded as strings. \n
            (Default is ``yaml.FullLoader``)

    Returns:
        (object): 
            loaded_obj (object):
                The object loaded from the YAML file.
    """
    path = prepare_filepath_for_loading(filepath, must_exist=True)
    with open(path, mode) as f:
        return yaml.load(f, Loader=loader)    

def matlab_load(
    filepath: str, 
    simplify_cells: bool = True, 
    kwargs_scipy: Dict = {}, 
    kwargs_mat73: Dict = {}, 
    verbose: bool = False
) -> Dict:
    """
    Loads a matlab file. If the .mat file is not version 7.3, it uses
    ``scipy.io.loadmat``. If the .mat file is version 7.3, it uses
    ``mat73.loadmat``. RH 2023

    Args:
        filepath (str):
            Path to the matlab file.
        simplify_cells (bool): 
            If set to ``True`` and file is not version 7.3, it simplifies cells
            to numpy arrays. (Default is ``True``)
        kwargs_scipy (Dict): 
            Keyword arguments to pass to ``scipy.io.loadmat``. (Default is
            ``{}``)
        kwargs_mat73 (Dict): 
            Keyword arguments to pass to ``mat73.loadmat``. (Default is ``{}``)
        verbose (bool): 
            If set to ``True``, it prints information about the file. (Default
            is ``False``)

    Returns:
        (Dict): 
            out (Dict):
                The loaded matlab file content in a dictionary format.
    """
    path = prepare_filepath_for_loading(filepath, must_exist=True)
    assert path.endswith('.mat'), 'File must be .mat file.'

    try:
        import scipy.io
        out = scipy.io.loadmat(path, simplify_cells=simplify_cells, **kwargs_scipy)
    except NotImplementedError as e:
        print(f'File {path} is version 7.3. Loading with mat73.') if verbose else None
        import mat73
        out = mat73.loadmat(path, **kwargs_mat73)
        print(f'Loaded {path} with mat73.') if verbose else None
    return out

def matlab_save(
    obj: Dict, 
    filepath: str, 
    mkdir: bool = False, 
    allow_overwrite: bool = True,
    clean_string: bool = True,
    list_to_objArray: bool = True,
    none_to_nan: bool = True,
    kwargs_scipy_savemat: Dict = {
        'appendmat': True,
        'format': '5',
        'long_field_names': False,
        'do_compression': False,
        'oned_as': 'row',
    }
):
    """
    Saves data to a matlab file. It uses ``scipy.io.savemat`` and provides
    additional functionality such as cleaning strings, converting lists to
    object arrays, and converting None to np.nan.
    RH 2023

    Args:
        obj (Dict): 
            The data to save. This must be in dictionary format.
        filepath (str): 
            The path to save the file to.
        mkdir (bool): 
            If set to ``True``, creates parent directory if it does not exist.
            (Default is ``False``)
        allow_overwrite (bool): 
            If set to ``True``, allows overwriting of existing file. (Default is
            ``True``)
        clean_string (bool): 
            If set to ``True``, converts strings to bytes. (Default is ``True``)
        list_to_objArray (bool): 
            If set to ``True``, converts lists to object arrays. (Default is
            ``True``)
        none_to_nan (bool): 
            If set to ``True``, converts None to np.nan. (Default is ``True``)
        kwargs_scipy_savemat (Dict): 
            Keyword arguments to pass to ``scipy.io.savemat``. \n
            * ``'appendmat'``: Whether to append .mat to the end of the given
              filename, if it isn't already there.
            * ``'format'``: The format of the .mat file. '4' for Matlab 4 .mat
              files, '5' for Matlab 5 and above.
            * ``'long_field_names'``: Whether to allow field names of up to 63
              characters instead of the standard 31.
            * ``'do_compression'``: Whether to compress matrices on write.
            * ``'oned_as'``: Whether to save 1-D numpy arrays as row or column
              vectors in the .mat file. 'row' or 'column'. \n
            (Default is ``{'appendmat': True, 'format': '5', 'long_field_names':
            False, 'do_compression': False, 'oned_as': 'row'}``)

    """
    import numpy as np

    prepare_filepath_for_saving(filepath, mkdir=mkdir, allow_overwrite=allow_overwrite)

    def walk(d, fn):
        return {key: fn(val) if isinstance(val, dict)==False else walk(val, fn) for key, val in d.items()}
    
    fn_clean_string = (lambda x: x.encode('utf-8') if isinstance(x, str) and clean_string else x) if clean_string else (lambda x: x)
    fn_list_to_objArray = (lambda x: np.array(x, dtype=object) if isinstance(x, list) and list_to_objArray else x) if list_to_objArray else (lambda x: x)
    fn_none_to_nan = (lambda x: np.nan if x is None and none_to_nan else x) if none_to_nan else (lambda x: x)

    data_cleaned = walk(walk(walk(obj, fn_clean_string), fn_list_to_objArray), fn_none_to_nan)

    import scipy.io
    scipy.io.savemat(filepath, data_cleaned, **kwargs_scipy_savemat)


def download_file(
    url: Optional[str],
    path_save: str,
    check_local_first: bool = True,
    check_hash: bool = False,
    hash_type: str = 'MD5',
    hash_hex: Optional[str] = None,
    mkdir: bool = False,
    allow_overwrite: bool = True,
    write_mode: str = 'wb',
    verbose: bool = True,
    chunk_size: int = 1024,
) -> None:
    """
    Downloads a file from a URL to a local path using requests. Checks if file
    already exists locally and verifies the hash of the downloaded file against
    a provided hash if required.
    RH 2023

    Args:
        url (Optional[str]): 
            URL of the file to download. If ``None``, then no download is
            attempted. (Default is ``None``)
        path_save (str): 
            Path to save the file to.
        check_local_first (bool): 
            Whether to check if the file already exists locally. If ``True`` and
            the file exists locally, the download is skipped. If ``True`` and
            ``check_hash`` is also ``True``, the hash of the local file is
            checked. If the hash matches, the download is skipped. If the hash
            does not match, the file is downloaded. (Default is ``True``)
        check_hash (bool): 
            Whether to check the hash of the local or downloaded file against
            ``hash_hex``. (Default is ``False``)
        hash_type (str): 
            Type of hash to use. Options are: ``'MD5'``, ``'SHA1'``,
            ``'SHA256'``, ``'SHA512'``. (Default is ``'MD5'``)
        hash_hex (Optional[str]): 
            Hash to compare to, in hexadecimal format (e.g., 'a1b2c3d4e5f6...').
            Can be generated using ``hash_file()`` or ``hashlib.hexdigest()``.
            If ``check_hash`` is ``True``, ``hash_hex`` must be provided.
            (Default is ``None``)
        mkdir (bool): 
            If ``True``, creates the parent directory of ``path_save`` if it
            does not exist. (Default is ``False``)
        write_mode (str): 
            Write mode for saving the file. Options include: ``'wb'`` (write
            binary), ``'ab'`` (append binary), ``'xb'`` (write binary, fail if
            file exists). (Default is ``'wb'``)
        verbose (bool): 
            If ``True``, prints status messages. (Default is ``True``)
        chunk_size (int): 
            Size of chunks in which to download the file. (Default is *1024*)
    """
    import os
    import requests

    # Check if file already exists locally
    if check_local_first:
        if os.path.isfile(path_save):
            print(f'File already exists locally: {path_save}') if verbose else None
            # Check hash of local file
            if check_hash:
                hash_local = hash_file(path_save, type_hash=hash_type)
                if hash_local == hash_hex:
                    print('Hash of local file matches provided hash_hex.') if verbose else None
                    return True
                else:
                    print('Hash of local file does not match provided hash_hex.') if verbose else None
                    print(f'Hash of local file: {hash_local}') if verbose else None
                    print(f'Hash provided in hash_hex: {hash_hex}') if verbose else None
                    print('Downloading file...') if verbose else None
            else:
                return True
        else:
            print(f'File does not exist locally: {path_save}. Will attempt download from {url}') if verbose else None

    # Download file
    if url is None:
        print('No URL provided. No download attempted.') if verbose else None
        return None
    try:
        response = requests.get(url, stream=True)
    except requests.exceptions.RequestException as e:
        print(f'Error downloading file: {e}') if verbose else None
        return False
    # Check response
    if response.status_code != 200:
        print(f'Error downloading file. Response status code: {response.status_code}') if verbose else None
        return False
    # Create parent directory if it does not exist
    prepare_filepath_for_saving(path_save, mkdir=mkdir, allow_overwrite=allow_overwrite)
    # Download file with progress bar
    total_size = int(response.headers.get('content-length', 0))
    wrote = 0
    with open(path_save, write_mode) as f:
        with tqdm(total=total_size, disable=(verbose==False), unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for data in response.iter_content(chunk_size):
                wrote = wrote + len(data)
                f.write(data)
                pbar.update(len(data))
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")
        return False
    # Check hash
    hash_local = hash_file(path_save, type_hash=hash_type)
    if check_hash:
        if hash_local == hash_hex:
            print('Hash of downloaded file matches hash_hex.') if verbose else None
            return True
        else:
            print('Hash of downloaded file does not match hash_hex.') if verbose else None
            print(f'Hash of downloaded file: {hash_local}') if verbose else None
            print(f'Hash provided in hash_hex: {hash_hex}') if verbose else None
            return False
    else:
        print(f'Hash of downloaded file: {hash_local}') if verbose else None
        return True


def hash_file(
    path: str, 
    type_hash: str = 'MD5', 
    buffer_size: int = 65536,
) -> str:
    """
    Computes the hash of a file using the specified hash type and buffer size.
    RH 2022

    Args:
        path (str):
            Path to the file to be hashed.
        type_hash (str):
            Type of hash to use. (Default is ``'MD5'``). Either \n
            * ``'MD5'``: MD5 hash algorithm.
            * ``'SHA1'``: SHA1 hash algorithm.
            * ``'SHA256'``: SHA256 hash algorithm.
            * ``'SHA512'``: SHA512 hash algorithm.
        buffer_size (int):
            Buffer size (in bytes) for reading the file. 
            65536 corresponds to 64KB. (Default is *65536*)

    Returns:
        (str): 
            hash_val (str):
                The computed hash of the file.
    """
    import hashlib

    if type_hash == 'MD5':
        hasher = hashlib.md5()
    elif type_hash == 'SHA1':
        hasher = hashlib.sha1()
    elif type_hash == 'SHA256':
        hasher = hashlib.sha256()
    elif type_hash == 'SHA512':
        hasher = hashlib.sha512()
    else:
        raise ValueError(f'{type_hash} is not a valid hash type.')

    with open(path, 'rb') as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            hasher.update(data)

    hash_val = hasher.hexdigest()
        
    return hash_val
    

def get_dir_contents(
    directory: str,
) -> Tuple[List[str], List[str]]:
    """
    Retrieves the names of the folders and files in a directory (does not
    include subdirectories).
    RH 2021

    Args:
        directory (str):
            The path to the directory.

    Returns:
        (tuple): tuple containing:
            folders (List[str]):
                A list of folder names.
            files (List[str]):
                A list of file names.
    """
    walk = os.walk(directory, followlinks=False)
    folders = []
    files = []
    for ii,level in enumerate(walk):
        folders, files = level[1:]
        if ii==0:
            break
    return folders, files


def compare_file_hashes(
    hash_dict_true: Dict[str, Tuple[str, str]],
    dir_files_test: Optional[str] = None,
    paths_files_test: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[bool, Dict[str, bool], Dict[str, str]]:
    """
    Compares hashes of files in a directory or list of paths to provided hashes.
    RH 2022

    Args:
        hash_dict_true (Dict[str, Tuple[str, str]]):
            Dictionary of hashes to compare. Each entry should be in the format:
            *{'key': ('filename', 'hash')}*.
        dir_files_test (str): 
            Path to directory containing the files to compare hashes. 
            Unused if paths_files_test is not ``None``. (Optional)
        paths_files_test (List[str]): 
            List of paths to files to compare hashes. 
            dir_files_test is used if ``None``. (Optional)
        verbose (bool): 
            If ``True``, failed comparisons are printed out. (Default is ``True``)

    Returns:
        (tuple): tuple containing:
            total_result (bool):
                ``True`` if all hashes match, ``False`` otherwise.
            individual_results (Dict[str, bool]):
                Dictionary indicating whether each hash matched.
            paths_matching (Dict[str, str]):
                Dictionary of paths that matched. Each entry is in the format:
                *{'key': 'path'}*.
    """
    if paths_files_test is None:
        if dir_files_test is None:
            raise ValueError('Must provide either dir_files_test or path_files_test.')
        
        ## make a dict of {filename: path} for each file in dir_files_test
        files_test = {filename: (Path(dir_files_test).resolve() / filename).as_posix() for filename in get_dir_contents(dir_files_test)[1]} 
    else:
        files_test = {Path(path).name: path for path in paths_files_test}

    paths_matching = {}
    results_matching = {}
    for key, (filename, hash_true) in hash_dict_true.items():
        match = True
        if filename not in files_test:
            print(f'{filename} not found in test directory: {dir_files_test}.') if verbose else None
            match = False
        elif hash_true != hash_file(files_test[filename]):
            print(f'{filename} hash mismatch with {key, filename}.') if verbose else None
            match = False
        if match:
            paths_matching[key] = files_test[filename]
        results_matching[key] = match

    return all(results_matching.values()), results_matching, paths_matching


def extract_zip(
    path_zip: str,
    path_extract: Optional[str] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Extracts a zip file.
    RH 2022

    Args:
        path_zip (str): 
            Path to the zip file.
        path_extract (Optional[str]): 
            Path (directory) to extract the zip file to.
            If ``None``, extracts to the same directory as the zip file.
            (Default is ``None``)
        verbose (bool): 
            Whether to print progress. (Default is ``True``)

    Returns:
        (List[str]): 
            paths_extracted (List[str]):
                List of paths to the extracted files.
    """
    import zipfile

    if path_extract is None:
        path_extract = str(Path(path_zip).resolve().parent)
    path_extract = str(Path(path_extract).resolve().absolute())

    print(f'Extracting {path_zip} to {path_extract}.') if verbose else None

    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(path_extract)
        paths_extracted = [str(Path(path_extract) / p) for p in zip_ref.namelist()]

    print('Completed zip extraction.') if verbose else None

    return paths_extracted


######################################################################################################################################
###################################################### PLOTTING HELPERS ##############################################################
######################################################################################################################################


def plot_image_grid(
    images: Union[List[np.ndarray], np.ndarray], 
    labels: Optional[List[str]] = None, 
    grid_shape: Tuple[int, int] = (10,10), 
    show_axis: str = 'off', 
    cmap: Optional[str] = None, 
    kwargs_subplots: Dict = {}, 
    kwargs_imshow: Dict = {},
) -> Tuple[plt.Figure, Union[np.ndarray, plt.Axes]]:
    """
    Plots a grid of images.
    RH 2021

    Args:
        images (Union[List[np.ndarray], np.ndarray]): 
            A list of images or a 3D array of images, where the first dimension is the number of images.
        labels (Optional[List[str]]): 
            A list of labels to be displayed in the grid. (Default is ``None``)
        grid_shape (Tuple[int, int]): 
            Shape of the grid. (Default is *(10,10)*)
        show_axis (str): 
            Whether to show axes or not. (Default is 'off')
        cmap (Optional[str]): 
            Colormap to use. (Default is ``None``)
        kwargs_subplots (Dict): 
            Keyword arguments for subplots. (Default is {})
        kwargs_imshow (Dict): 
            Keyword arguments for imshow. (Default is {})

    Returns:
        (Tuple[plt.Figure, Union[np.ndarray, plt.Axes]]): tuple containing:
            fig (plt.Figure):
                Figure object.
            axs (Union[np.ndarray, plt.Axes]):
                Axes object.
    """
    if cmap is None:
        cmap = 'viridis'

    fig, axs = plt.subplots(nrows=grid_shape[0], ncols=grid_shape[1], **kwargs_subplots)
    axs_flat = axs.flatten(order='F') if isinstance(axs, np.ndarray) else [axs]
    for ii, ax in enumerate(axs_flat[:len(images)]):
        ax.imshow(images[ii], cmap=cmap, **kwargs_imshow);
        if labels is not None:
            ax.set_title(labels[ii]);
        ax.axis(show_axis);
    return fig, axs


def rand_cmap(
    nlabels: int, 
    first_color_black: bool = False, 
    last_color_black: bool = False,
    verbose: bool = True,
    under: List[float] = [0,0,0],
    over: List[float] = [0.5,0.5,0.5],
    bad: List[float] = [0.9,0.9,0.9],
) -> object:
    """
    Creates a random colormap to be used with matplotlib. Useful for segmentation tasks.

    Args:
        nlabels (int):
            Number of labels (size of colormap).
        first_color_black (bool):
            Option to use the first color as black. (Default is ``False``)
        last_color_black (bool):
            Option to use the last color as black. (Default is ``False``)
        verbose (bool):
            Prints the number of labels and shows the colormap if ``True``. 
            (Default is ``True``)
        under (List[float]):
            RGB values to use for the 'under' threshold in the colormap. 
            (Default is ``[0, 0, 0]``)
        over (List[float]):
            RGB values to use for the 'over' threshold in the colormap. 
            (Default is ``[0.5, 0.5, 0.5]``)
        bad (List[float]):
            RGB values to use for 'bad' values in the colormap. 
            (Default is ``[0.9, 0.9, 0.9]``)

    Returns:
        (LinearSegmentedColormap):
            colormap (LinearSegmentedColormap):
                Colormap for matplotlib.
    """
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    import numpy as np

    assert nlabels > 0, 'Number of labels must be greater than 0'

    if verbose:
        print('Number of labels: ' + str(nlabels))

    randRGBcolors = np.random.rand(nlabels, 3)
    randRGBcolors = randRGBcolors / np.max(randRGBcolors, axis=1, keepdims=True)

    if first_color_black:
        randRGBcolors[0] = [0, 0, 0]

    if last_color_black:
        randRGBcolors[-1] = [0, 0, 0]

    random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        fig, ax = plt.subplots(1, 1, figsize=(6, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    random_colormap.set_bad(bad)
    random_colormap.set_over(over)
    random_colormap.set_under(under)

    return random_colormap


def simple_cmap(
    colors: List[List[float]] = [
        [1,0,0],
        [1,0.6,0],
        [0.9,0.9,0],
        [0.6,1,0],
        [0,1,0],
        [0,1,0.6],
        [0,0.8,0.8],
        [0,0.6,1],
        [0,0,1],
        [0.6,0,1],
        [0.8,0,0.8],
        [1,0,0.6],
    ],
    under: List[float] = [0,0,0],
    over: List[float] = [0.5,0.5,0.5],
    bad: List[float] = [0.9,0.9,0.9],
    name: str = 'none',
) -> object:
    """
    Creates a colormap from a sequence of RGB values. 
    Borrowed with permission from Alex 
    (https://gist.github.com/ahwillia/3e022cdd1fe82627cbf1f2e9e2ad80a7ex)

    Args:
        colors (List[List[float]]): 
            List of RGB values. Each sub-list contains three float numbers 
            representing an RGB color. (Default is list of RGB colors ranging from red to purple)
        under (List[float]): 
            RGB values for the colormap under range. 
            (Default is ``[0,0,0]`` (black))
        over (List[float]): 
            RGB values for the colormap over range. 
            (Default is ``[0.5,0.5,0.5]`` (grey))
        bad (List[float]): 
            RGB values for the colormap bad range. 
            (Default is ``[0.9,0.9,0.9]`` (light grey))
        name (str): 
            Name of the colormap. (Default is 'none')

    Returns:
        (LinearSegmentedColormap): 
            cmap (LinearSegmentedColormap): 
                The generated colormap.

    Example:
        .. highlight:: python
        .. code-block:: python

            cmap = simple_cmap([(1,1,1), (1,0,0)]) # white to red colormap
            cmap = simple_cmap(['w', 'r'])         # white to red colormap
            cmap = simple_cmap(['r', 'b', 'r'])    # red to blue to red
    """
    from matplotlib.colors import LinearSegmentedColormap, colorConverter

    # check inputs
    n_colors = len(colors)
    if n_colors <= 1:
        raise ValueError('Must specify at least two colors')

    # convert colors to rgb
    colors = [colorConverter.to_rgb(c) for c in colors]

    # set up colormap
    r, g, b = colors[0]
    cdict = {'red': [(0.0, r, r)], 'green': [(0.0, g, g)], 'blue': [(0.0, b, b)]}
    for i, (r, g, b) in enumerate(colors[1:]):
        idx = (i+1) / (n_colors-1)
        cdict['red'].append((idx, r, r))
        cdict['green'].append((idx, g, g))
        cdict['blue'].append((idx, b, b))

    cmap = LinearSegmentedColormap(name, {k: tuple(v) for k, v in cdict.items()})
                                   
    cmap.set_bad(bad)
    cmap.set_over(over)
    cmap.set_under(under)

    return cmap


class ImageLabeler:
    """
    A simple graphical interface for labeling image classes. Use this class with
    a context manager to ensure the window is closed properly. The class
    provides a tkinter window which displays images from a provided numpy array
    one by one and lets you classify each image by pressing a key. The title of
    the window is the image index. The classification label and image index are
    stored as the ``self.labels_`` attribute and saved to a CSV file in
    self.path_csv. 
    RH 2023

    Args:
        image_array (np.ndarray): 
            A numpy array of images. Either 3D: *(n_images, height, width)* or
            4D: *(n_images, height, width, n_channels)*. Images should be scaled
            between 0 and 255 and will be converted to uint8.
        start_index (int): 
            The index of the first image to display. (Default is *0*)
        path_csv (str): 
            Path to the CSV file for saving results. If ``None``, results will
            not be saved.
        save_csv (bool):
            Whether to save the results to a CSV. (Default is ``True``)
        resize_factor (float): 
            A scaling factor for the fractional change in image size. (Default
            is *1.0*)
        normalize_images (bool):
            Whether to normalize the images between min and max values. (Default
            is ``True``)
        verbose (bool):
            Whether to print status updates. (Default is ``True``)
        key_end (str): 
            Key to press to end the session. (Default is ``'Escape'``)
        key_prev (str):
            Key to press to go back to the previous image. (Default is
            ``'Left'``)
        key_next (str):
            Key to press to go to the next image. (Default is ``'Right'``)

    Example:
        .. highlight:: python
        .. code-block:: python

            with ImageLabeler(images, start_index=0, resize_factor=4.0,
            key_end='Escape') as labeler:
                labeler.run()
            path_csv, labels = labeler.path_csv, labeler.labels_

    Attributes:
        image_array (np.ndarray):
            A numpy array of images. Either 3D: *(n_images, height, width)* or
            4D: *(n_images, height, width, n_channels)*. Images should be scaled
            between 0 and 255 and will be converted to uint8.
        start_index (int): 
            The index of the first image to display. (Default is *0*)
        path_csv (str): 
            Path to the CSV file for saving results. If ``None``, results will
            not be saved.
        save_csv (bool):
            Whether to save the results to a CSV. (Default is ``True``)
        resize_factor (float): 
            A scaling factor for the fractional change in image size. (Default
            is *1.0*)
        normalize_images (bool):
            Whether to normalize the images between min and max values. (Default
            is ``True``)
        verbose (bool):
            Whether to print status updates. (Default is ``True``)
        key_end (str): 
            Key to press to end the session. (Default is ``'Escape'``)
        key_prev (str):
            Key to press to go back to the previous image. (Default is
            ``'Left'``)
        key_next (str):
            Key to press to go to the next image. (Default is ``'Right'``)
        labels_ (list):
            A list of tuples containing the image index and classification label
            for each image. The list is saved to a CSV file in self.path_csv.
    """
    def __init__(
        self, 
        image_array: np.ndarray, 
        start_index: int=0,
        path_csv: Optional[str] = None, 
        save_csv: bool = True,
        resize_factor: float = 10.0, 
        normalize_images: bool = True,
        verbose: bool = True,
        key_end: str = 'Escape', 
        key_prev: str = 'Left',
        key_next: str = 'Right',
    ) -> None:
        """
        Initializes the ImageLabeler with the given image array, csv path, and UI 
        elements. Binds keys for classifying images and ending the session.
        """
        import tempfile
        import datetime
        ## Set attributes
        self.images = image_array
        self._resize_factor = resize_factor
        self._index = start_index - 1  ## -1 because we increment before displaying
        self.path_csv = path_csv if path_csv is not None else str(Path(tempfile.gettempdir()) / ('roicat_labels_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.csv'))
        self._save_csv = save_csv
        self.labels_ = {}
        self._img_tk = None
        self._key_end = key_end if key_end is not None else None
        self._key_prev = key_prev if key_prev is not None else None
        self._key_next = key_next if key_next is not None else None
        self._normalize_images = normalize_images
        self._verbose = verbose

        self.__call__ = self.run
        
    def run(self):
        """
        Runs the image labeler. Opens a tkinter window and displays the first
        image.
        """
        try:
            self._root = tk.Tk()
            self._img_label = tk.Label(self._root)
            self._img_label.pack()

            ## Bind keys
            self._root.bind("<Key>", self.classify)
            self._root.bind('<Key-' + self._key_end + '>', self.end_session) if self._key_end is not None else None
            self._root.bind('<Key-' + self._key_prev + '>', self.prev_img) if self._key_prev is not None else None
            self._root.bind('<Key-' + self._key_next + '>', self.next_img) if self._key_next is not None else None

            self._root.protocol("WM_DELETE_WINDOW", self._on_closing)

            ## Start the session
            self.next_img()
            self._root.mainloop()
        except Exception as e:
            warnings.warn('Error initializing image labeler: ' + str(e))

    def _on_closing(self):
        from tkinter import messagebox
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.end_session(None)

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_session(None)

    def next_img(self, event=None):
        """Displays the next image in the array, and resizes the image."""
        ## Display the image
        ### End the session if there are no more images
        self._index += 1
        if self._index < len(self.images):
            im = self.images[self._index]
            im = (im / np.max(im)) * 255 if self._normalize_images else im
            pil_img = PIL.Image.fromarray(np.uint8(im))  ## Convert to uint8 and PIL image
            ## Resize image
            width, height = pil_img.size
            new_width = int(width * self._resize_factor)
            new_height = int(height * self._resize_factor)
            pil_img = pil_img.resize((new_width, new_height), resample=PIL.Image.LANCZOS)
            ## Display image
            self._img_tk = ImageTk.PhotoImage(pil_img)
            self._img_label.image = self._img_tk  # keep a reference to the PhotoImage object
            self._img_label.config(image=self._img_label.image)
        else:
            self.end_session(None)
        
        self._root.title(str(self._index))  # update the window title to the current image index

    def prev_img(self, event=None):
        """
        Displays the previous image in the array.
        """
        self._index -= 2
        self.next_img()

    def classify(self, event):
        """
        Adds the current image index and pressed key as a label.
        Then saves the results and moves to the next image.

        Args:
            event (tkinter.Event):
                A tkinter event object.
        """
        label = event.char
        if label != '':
            print(f'Image {self._index}: {label}') if self._verbose else None
            self.labels_.update({self._index: str(label)})  ## Store the label
            self.save_classification() if self._save_csv else None ## Save the results
            self.next_img()  ## Move to the next image

    def end_session(self, event):
        """
        Ends the classification session by destroying the tkinter window.
        """
        self._img_tk = None
        self._root.destroy() if self._root is not None else None
        self._root = None
        
        import gc
        gc.collect()
        gc.collect()

    def save_classification(self):
        """
        Saves the classification results to a CSV file.
        This function does not append, it overwrites the entire file.
        The file contains two columns: 'image_index' and 'label'.
        """
        ## make directory if it doesn't exist
        Path(self.path_csv).parent.mkdir(parents=True, exist_ok=True)
        ## Save the results
        with open(self.path_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(('image_index', 'label'))
            writer.writerows(self.labels_.items())

    def get_labels(self, kind: str = 'dict') -> Union[dict, List[Tuple[int, str]], dict]:
        """
        Returns the labels. The format of the output is determined by the ``kind`` parameter. 
        If the labels dictionary is empty, returns ``None``. RH 2023

        Args:
            kind (str): 
                The type of object to return. (Default is ``'dict'``) \n
                * ``'dict'``: {idx: label, idx: label, ...}
                * ``'list'``: [label, label, ...] where the index is the image
                  index and unlabeled images are represented as ``'None'``.
                * ``'dataframe'``: {'index': [idx, idx, ...], 'label': [label, label, ...]}
                  This can be converted to a pandas dataframe with:
                  pd.DataFrame(self.get_labels('dataframe'))

        Returns:
            (Union[dict, List[Tuple[int, str]], dict]): 
                Depending on the ``kind`` parameter, it returns either: \n
                * dict: 
                    A dictionary where keys are the image indices and values are
                    the labels.
                * List[Tuple[int, str]]: 
                    A list of tuples, where each tuple contains an image index
                    and a label.
                * dict: 
                    A dictionary with keys 'index' and 'label' where values are
                    lists of indices and labels respectively.
        """
        ## if the dict is empty, return None
        if len(self.labels_) == 0:
            return None
        
        if kind == 'dict':
            return self.labels_
        elif kind == 'list':
            out = ['None',] * len(self.images)
            for idx, label in self.labels_.items():
                out[idx] = label
            return out
        elif kind == 'dataframe':
            import pandas as pd
            return pd.DataFrame(index=list(self.labels_.keys()), data={'label': list(self.labels_.values())})


def export_svg_hv_bokeh(
    obj: object, 
    path_save: str
) -> None:
    """
    Saves a scatterplot from holoviews as an SVG file.
    RH 2023

    Args:
        obj (object): 
            Holoviews plot object.
        path_save (str):
            Path to save the SVG file.
    """
    import holoviews as hv
    import bokeh
    plot_state = hv.renderer('bokeh').get_plot(obj).state
    plot_state.output_backend = 'svg'
    bokeh.io.export_svgs(plot_state, filename=path_save)


######################################################################################################################################
######################################################## H5 HANDLING #################################################################
######################################################################################################################################

## below is actually 'simple_load' from h5_handling
def h5_load(
    filepath: Union[str, Path],
    return_dict: bool = True,
    verbose: bool = False
) -> Union[dict, object]:
    """
    Returns a dictionary or an H5PY object from a given HDF file.
    RH 2023

    Args:
        filepath (Union[str, Path]): 
            Full pathname of the file to read.
        return_dict (bool):
            Whether or not to return a dict object. (Default is ``True``). \n
            * ``True``: a dict object is returned. 
            * ``False``: an H5PY object is returned.
        verbose (bool): 
            Whether to print detailed information during the execution. (Default
            is ``False``)

    Returns:
        (Union[dict, object]): 
            result (Union[dict, object]):
                Either a dictionary containing the groups as keys and the
                datasets as values from the HDF file or an H5PY object,
                depending on the ``return_dict`` parameter.
    """
    import h5py
    if return_dict:
        with h5py.File(filepath, 'r') as h5_file:
            if verbose:
                print(f'==== Loading h5 file with hierarchy: ====')
                show_item_tree(h5_file)
            result = {}
            def visitor_func(name, node):
                # Split name by '/' and reduce to nested dict
                keys = name.split('/')
                sub_dict = result
                for key in keys[:-1]:
                    sub_dict = sub_dict.setdefault(key, {})

                if isinstance(node, h5py.Dataset):
                    sub_dict[keys[-1]] = node[...]
                elif isinstance(node, h5py.Group):
                    sub_dict.setdefault(keys[-1], {})

            h5_file.visititems(visitor_func)            
            return result
    else:
        return h5py.File(filepath, 'r')
    
def show_item_tree(
    hObj: Optional[Union[object, dict]] = None, 
    path: Optional[Union[str, Path]] = None, 
    depth: Optional[int] = None, 
    show_metadata: bool = True, 
    print_metadata: bool = False, 
    indent_level: int = 0
) -> None:
    '''
    Recursively displays all the items and groups in an HDF5 object or Python dictionary.
    RH 2021

    Args:
        hObj (Optional[Union[object, dict]]):
            Hierarchical object, which can be an HDF5 object or a Python
            dictionary. (Default is ``None``)
        path (Optional[Union[str, Path]]): 
            If not ``None``, then the path to the HDF5 object is used instead of
            ``hObj``. (Default is ``None``)
        depth (Optional[int]):
            How many levels deep to show the tree. (Default is ``None`` which
            shows all levels)
        show_metadata (bool): 
            Whether or not to list metadata with items. (Default is ``True``)
        print_metadata (bool): 
            Whether or not to show values of metadata items. (Default is
            ``False``)
        indent_level (int):
            Used internally to the function. User should leave this as the
            default. (Default is *0*)

    Example:
        .. highlight:: python
        .. code-block:: python

            import h5py
            with h5py.File('test.h5', 'r') as f:
                show_item_tree(f)
    '''
    import h5py
    if depth is None:
        depth = int(10000000000000000000)
    else:
        depth = int(depth)

    if depth < 0:
        return

    if path is not None:
        with h5py.File(path , 'r') as f:
            show_item_tree(hObj=f, path=None, depth=depth-1, show_metadata=show_metadata, print_metadata=print_metadata, indent_level=indent_level)
    else:
        indent = f'  '*indent_level
        if hasattr(hObj, 'attrs') and show_metadata:
            for ii,val in enumerate(list(hObj.attrs.keys()) ):
                if print_metadata:
                    print(f'{indent}METADATA: {val}: {hObj.attrs[val]}')
                else:
                    print(f'{indent}METADATA: {val}: shape={hObj.attrs[val].shape} , dtype={hObj.attrs[val].dtype}')
        
        for ii,val in enumerate(list(iter(hObj))):
            if isinstance(hObj[val], h5py.Group):
                print(f'{indent}{ii+1}. {val}:----------------')
                show_item_tree(hObj[val], depth=depth-1, show_metadata=show_metadata, print_metadata=print_metadata , indent_level=indent_level+1)
            elif isinstance(hObj[val], dict):
                print(f'{indent}{ii+1}. {val}:----------------')
                show_item_tree(hObj[val], depth=depth-1, show_metadata=show_metadata, print_metadata=print_metadata , indent_level=indent_level+1)
            else:
                if hasattr(hObj[val], 'shape') and hasattr(hObj[val], 'dtype'):
                    print(f'{indent}{ii+1}. {val}:    '.ljust(20) + f'shape={hObj[val].shape} ,'.ljust(20) + f'dtype={hObj[val].dtype}')
                else:
                    print(f'{indent}{ii+1}. {val}:    '.ljust(20) + f'type={type(hObj[val])}')
        

######################################################################################################################################
####################################################### DECOMPOSITION ################################################################
######################################################################################################################################

def torch_pca(
    X_in: Union[torch.Tensor, np.ndarray], 
    device: str = 'cpu', 
    mean_sub: bool = True, 
    zscore: bool = False, 
    rank: Optional[int] = None, 
    return_cpu: bool = True, 
    return_numpy: bool = False,
) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """
    Conducts Principal Components Analysis using the Pytorch library. This
    function can run on either CPU or GPU devices. 
    RH 2021

    Args:
        X_in (Union[torch.Tensor, np.ndarray]):
            The data to be decomposed. This should be a 2-D array, with columns
            representing features and rows representing samples. PCA is
            performed column-wise.
        device (str):
            The device to use for computation, e.g., 'cuda' or 'cpu'. (Default
            is ``'cpu'``)
        mean_sub (bool):
            If ``True``, subtract the mean ('center') from the columns. (Default
            is ``True``)
        zscore (bool):
            If ``True``, z-score the columns. This is equivalent to conducting
            PCA on the correlation-matrix. (Default is ``False``)
        rank (int):
            Maximum estimated rank of the decomposition. If ``None``, then the
            rank is assumed to be X.shape[1]. (Default is ``None``)
        return_cpu (bool):  
            (Default is ``True``) \n
            * ``True``, all outputs are forced to be on the 'cpu' device.
            * ``False``, and device is not 'cpu', then the returns will be on the
              provided device.
        return_numpy (bool):
            If ``True``, all outputs are forced to be of type numpy.ndarray.
            (Default is ``False``)

    Returns:
        (tuple): tuple containing:
            components (torch.Tensor or np.ndarray):
                The components of the decomposition, represented as a 2-D array.
                Each column is a component vector and each row is a feature
                weight.
            scores (torch.Tensor or np.ndarray):
                The scores of the decomposition, represented as a 2-D array.
                Each column is a score vector and each row is a sample weight.
            singVals (torch.Tensor or np.ndarray):
                The singular values of the decomposition, represented as a 1-D
                array. Each element is a singular value.
            EVR (torch.Tensor or np.ndarray):
                The explained variance ratio of each component, represented as a
                1-D array. Each element is the explained variance ratio of the
                corresponding component.
                
    Example:
        .. highlight:: python
        .. code-block:: python

            components, scores, singVals, EVR = torch_pca(X_in)
    """
    if isinstance(X_in, torch.Tensor) == False:
        X = torch.from_numpy(X_in).to(device)
    elif X_in.device != device:
            X = X_in.to(device)
    else:
        X = copy.copy(X_in)
            
    if mean_sub and not zscore:
        X = X - torch.mean(X, dim=0)
    if zscore:
        X = X - torch.mean(X, dim=0)
        stds = torch.std(X, dim=0)
        X = X / stds[None,:]        
        
    if rank is None:
        rank = min(list(X.shape))
    
    (U,S,V) = torch.pca_lowrank(X, q=rank, center=False, niter=2)
    components = V
    scores = torch.matmul(X, V[:, :rank])

    singVals = (S**2)/(len(S)-1)
    EVR = (singVals) / torch.sum(singVals)
    
    if return_cpu:
        components = components.cpu()
        scores = scores.cpu()
        singVals = singVals.cpu()
        EVR = EVR.cpu()
    if return_numpy:
        components = components.cpu().numpy()
        scores = scores.cpu().numpy()
        singVals = singVals.cpu().numpy()
        EVR = EVR.cpu().numpy()
        
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    return components, scores, singVals, EVR


######################################################################################################################################
############################################################ VIDEO ###################################################################
######################################################################################################################################


def grayscale_to_rgb(
    array: Union[np.ndarray, torch.Tensor, List]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Converts a grayscale image (2D array) or movie (3D array) to RGB (3D or 4D
    array).

    RH 2023

    Args:
        array (Union[np.ndarray, torch.Tensor, list]):
            The 2D or 3D array of grayscale images.

    Returns:
        (Union[np.ndarray, torch.Tensor]):
            array (Union[np.ndarray, torch.Tensor]):
                The converted 3D or 4D array of RGB images.
    """
    if isinstance(array, list):
        if isinstance(array[0], np.ndarray):
            array = np.stack(array, axis=0)
        elif isinstance(array[0], torch.Tensor):
            array = torch.stack(array, axis=0)
        else:
            raise Exception(f'Failed to convert list of type {type(array[0])} to array')
    if isinstance(array, np.ndarray):
        return np.stack([array, array, array], axis=-1)
    elif isinstance(array, torch.Tensor):
        return torch.stack([array, array, array], dim=-1)
    


def save_gif(
    array: Union[np.ndarray, List], 
    path: str, 
    frameRate: float = 5.0, 
    loop: int = 0, 
    # backend='PIL', 
    kwargs_backend: Dict = {},
):
    """
    Saves an array of images as a gif.
    RH 2023

    Args:
        array (Union[np.ndarray, list]):
            The 3D (grayscale) or 4D (color) array of images. \n
            * If dtype is ``float`` type, then scale is from 0 to 1.
            * If dtype is ``int``, then scale is from 0 to 255.
        path (str):
            The path where the gif is saved.
        frameRate (float):
            The frame rate of the gif. (Default is ``5.0``)
        loop (int):
            The number of times to loop the gif. (Default is ``0``) \n
            * 0 means loop forever
            * 1 means play once
            * 2 means play twice (loop once)
            * etc.
        # backend (str):
        #     Which backend to use.
        #     Options: 'imageio' or 'PIL'
        kwargs_backend (Dict):
            The keyword arguments for the backend.
    """
    array = np.stack(array, axis=0) if isinstance(array, list) else array
    array = grayscale_to_rgb(array) if array.ndim == 3 else array
    if np.issubdtype(array.dtype, np.floating):
        array = (array*255).astype('uint8')
    
    kwargs_backend.update({'loop': loop} if loop != 1 else {})

    # if backend == 'imageio':
    #     import imageio
    #     imageio.mimsave(
    #         path, 
    #         array, 
    #         format='GIF',
    #         duration=1000/frameRate, 
    #         **kwargs_backend,
    #     )
    # elif backend == 'PIL':
    from PIL import Image
    frames = [Image.fromarray(array[i_frame]) for i_frame in range(array.shape[0])]
    frames[0].save(
        path, 
        format='GIF', 
        append_images=frames[1:], 
        save_all=True, 
        duration=1000/frameRate, 
        **kwargs_backend,
    )
    # else:
    #     raise Exception(f'Unsupported backend {backend}')


######################################################################################################################################
###################################################### IMAGE_PROCESSING ##############################################################
######################################################################################################################################


def mask_image_border(
    im: np.ndarray, 
    border_outer: Optional[Union[int, Tuple[int, int, int, int]]] = None, 
    border_inner: Optional[int] = None, 
    mask_value: float = 0,
) -> np.ndarray:
    """
    Masks an image within specified outer and inner borders.
    RH 2022

    Args:
        im (np.ndarray):
            Input image of shape: *(height, width)*.
        border_outer (Union[int, tuple[int, int, int, int], None]):
            Number of pixels along the border to mask. If ``None``, the border
            is not masked. If an int is provided, all borders are equally
            masked. If a tuple of ints is provided, borders are masked in the
            order: *(top, bottom, left, right)*. (Default is ``None``)
        border_inner (int, Optional):
            Number of pixels in the center to mask. Will be a square with side
            length equal to this value. (Default is ``None``)
        mask_value (float):
            Value to replace the masked pixels with. (Default is *0*)

    Returns:
        (np.ndarray):
            im_out (np.ndarray):
                Masked output image.
    """

    ## Find the center of the image
    height, width = im.shape
    center_y = cy = int(np.floor(height/2))
    center_x = cx = int(np.floor(width/2))

    ## Mask the center
    if border_inner is not None:
        ## make edge_lengths
        center_edge_length = cel = int(np.ceil(border_inner/2)) if border_inner is not None else 0
        im[cy-cel:cy+cel, cx-cel:cx+cel] = mask_value
    ## Mask the border
    if border_outer is not None:
        ## make edge_lengths
        if isinstance(border_outer, int):
            border_outer = (border_outer, border_outer, border_outer, border_outer)
        
        im[:border_outer[0], :] = mask_value
        im[-border_outer[1]:, :] = mask_value
        im[:, :border_outer[2]] = mask_value
        im[:, -border_outer[3]:] = mask_value

    return im


def make_Fourier_mask(
    frame_shape_y_x: Tuple[int, int] = (512,512),
    bandpass_spatialFs_bounds: List[float] = [1/128, 1/3],
    order_butter: int = 5,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    dtype_fft: object = torch.complex64,
    plot_pref: bool = False,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Generates a Fourier domain mask for phase correlation, primarily used in
    BWAIN.

    Args:
        frame_shape_y_x (Tuple[int, int]):
            Shape of the images that will be processed through this function.
            (Default is *(512, 512)*)
        bandpass_spatialFs_bounds (List[float]): 
            Specifies the lowcut and highcut in spatial frequency for the
            butterworth filter. (Default is *[1/128, 1/3]*)
        order_butter (int):
            Order of the butterworth filter. (Default is *5*)
        mask (Union[np.ndarray, torch.Tensor, None]):
            If not ``None``, this mask is used instead of creating a new one.
            (Default is ``None``)
        dtype_fft (object):
            Data type for the Fourier transform, default is ``torch.complex64``.
        plot_pref (bool):
            If ``True``, the absolute value of the mask is plotted. (Default is
            ``False``)
        verbose (bool):
            If ``True``, enables the print statements for debugging. (Default is
            ``False``)

    Returns:
        (torch.Tensor):
            mask_fft (torch.Tensor):
                The generated mask in the Fourier domain.
    """
    get_nd_butterworth_filter

    bandpass_spatialFs_bounds = list(bandpass_spatialFs_bounds)
    bandpass_spatialFs_bounds[0] = max(bandpass_spatialFs_bounds[0], 1e-9)
    
    if (isinstance(mask, (np.ndarray, torch.Tensor))) or ((mask != 'None') and (mask is not None)):
        mask = torch.as_tensor(mask, dtype=dtype_fft)
        mask = mask / mask.sum()
        mask_fftshift = torch.fft.fftshift(mask)
        print(f'User provided mask of shape: {mask.shape} was normalized to sum=1, fftshift-ed, and converted to a torch.Tensor')
    else:
        wfilt_h = get_nd_butterworth_filter(
            shape=frame_shape_y_x, 
            factor=bandpass_spatialFs_bounds[0], 
            order=order_butter, 
            high_pass=True, 
            real=False,
        )
        wfilt_l = get_nd_butterworth_filter(
            shape=frame_shape_y_x, 
            factor=bandpass_spatialFs_bounds[1], 
            order=order_butter, 
            high_pass=False, 
            real=False,
        )

        kernel = torch.as_tensor(
            wfilt_h * wfilt_l,
            dtype=dtype_fft,
        )

        mask = kernel / kernel.sum()
        # self.mask_fftshift = torch.fft.fftshift(self.mask)
        mask_fftshift = mask
        mask_fftshift = mask_fftshift.contiguous()

        if plot_pref and plot_pref!='False':
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(
                torch.abs(kernel.cpu()).numpy(), 
                # clim=[0,1],
            )
        if verbose:
            print(f'Created Fourier domain mask. self.mask_fftshift.shape: {mask_fftshift.shape}. Images input to find_translation_shifts will now be masked in the FFT domain.')

    return mask_fftshift

## Fixed the indexing because the skimage nerds drink too much coffee. RH.
def get_nd_butterworth_filter(
    shape: Tuple[int, ...], 
    factor: float, 
    order: float, 
    high_pass: bool, 
    real: bool,
    dtype: np.dtype = np.float64, 
    squared_butterworth: bool = True
) -> np.ndarray:
    """
    Creates an N-dimensional Butterworth mask for an FFT.

    Args:
        shape (Tuple[int, ...]): 
            Shape of the n-dimensional FFT and mask.
        factor (float): 
            Fraction of mask dimensions where the cutoff should be.
        order (float): 
            Controls the slope in the cutoff region.
        high_pass (bool): 
            Whether the filter is high pass (low frequencies attenuated) or low
            pass (high frequencies are attenuated).
        real (bool): 
            Whether the FFT is of a real (``True``) or complex (``False``)
            image.
        dtype (np.dtype): 
            The desired output data type of the Butterworth filter. (Default is
            ``np.float64``)
        squared_butterworth (bool): 
            If ``True``, the square of the Butterworth filter is used. (Default
            is ``True``)

    Returns:
        (np.ndarray): 
            wfilt (np.ndarray): 
                The FFT mask.
    """
    import functools
    ranges = []
    for i, d in enumerate(shape):
        # start and stop ensures center of mask aligns with center of FFT
        # axis = np.arange(-(d - 1) // 2, (d - 1) // 2 + 1) / (d * factor)
        axis = np.arange(-(d - 1) / 2, (d - 1) / 2 + 0.5) / (d * factor)  ## FIXED, RH 2023
        ranges.append(scipy.fft.ifftshift(axis ** 2))
    # for real image FFT, halve the last axis
    if real:
        limit = d // 2 + 1
        ranges[-1] = ranges[-1][:limit]
    # q2 = squared Euclidean distance grid
    q2 = functools.reduce(
            np.add, np.meshgrid(*ranges, indexing="ij", sparse=True)
            )
    q2 = q2.astype(dtype)
    q2 = np.power(q2, order)
    wfilt = 1 / (1 + q2)
    if high_pass:
        wfilt *= q2
    if not squared_butterworth:
        np.sqrt(wfilt, out=wfilt)
    return wfilt


def find_geometric_transformation(
    im_template: np.ndarray, 
    im_moving: np.ndarray,
    warp_mode: str = 'euclidean',
    n_iter: int = 5000,
    termination_eps: float = 1e-10,
    mask: Optional[np.ndarray] = None,
    gaussFiltSize: int = 1
) -> np.ndarray:
    """
    Find the transformation between two images.
    Wrapper function for cv2.findTransformECC
    RH 2022

    Args:
        im_template (np.ndarray):
            Template image. The dtype must be either ``np.uint8`` or ``np.float32``.
        im_moving (np.ndarray):
            Moving image. The dtype must be either ``np.uint8`` or ``np.float32``.
        warp_mode (str):
            Warp mode. \n
            * 'translation': Sets a translational motion model; warpMatrix is 2x3 with the first 2x2 part being the unity matrix and the rest two parameters being estimated.
            * 'euclidean':   Sets a Euclidean (rigid) transformation as motion model; three parameters are estimated; warpMatrix is 2x3.
            * 'affine':      Sets an affine motion model; six parameters are estimated; warpMatrix is 2x3. (Default)
            * 'homography':  Sets a homography as a motion model; eight parameters are estimated;`warpMatrix` is 3x3.
        n_iter (int):
            Number of iterations. (Default is *5000*)
        termination_eps (float):
            Termination epsilon. This is the threshold of the increment in the correlation coefficient between two iterations. (Default is *1e-10*)
        mask (np.ndarray):
            Binary mask. Regions where mask is zero are ignored during the registration. If ``None``, no mask is used. (Default is ``None``)
        gaussFiltSize (int):
            Gaussian filter size. If *0*, no gaussian filter is used. (Default is *1*)

    Returns:
        (np.ndarray): 
            warp_matrix (np.ndarray):
                Warp matrix. See cv2.findTransformECC for more info. Can be
                applied using cv2.warpAffine or cv2.warpPerspective.
    """
    LUT_modes = {
        'translation': cv2.MOTION_TRANSLATION,
        'euclidean': cv2.MOTION_EUCLIDEAN,
        'affine': cv2.MOTION_AFFINE,
        'homography': cv2.MOTION_HOMOGRAPHY,
    }
    assert warp_mode in LUT_modes.keys(), f"warp_mode must be one of {LUT_modes.keys()}. Got {warp_mode}"
    warp_mode = LUT_modes[warp_mode]
    if warp_mode in [cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE]:
        shape_eye = (2, 3)
    elif warp_mode == cv2.MOTION_HOMOGRAPHY:
        shape_eye = (3, 3)
    else:
        raise ValueError(f"warp_mode {warp_mode} not recognized (should not happen)")
    warp_matrix = np.eye(*shape_eye, dtype=np.float32)

    ## assert that the inputs are numpy arrays of dtype np.uint8
    assert isinstance(im_template, np.ndarray) and (im_template.dtype == np.uint8 or im_template.dtype == np.float32), f"im_template must be a numpy array of dtype np.uint8 or np.float32. Got {type(im_template)} of dtype {im_template.dtype}"
    assert isinstance(im_moving, np.ndarray) and (im_moving.dtype == np.uint8 or im_moving.dtype == np.float32), f"im_moving must be a numpy array of dtype np.uint8 or np.float32. Got {type(im_moving)} of dtype {im_moving.dtype}"
    ## cast mask to bool then to uint8
    if mask is not None:
        assert isinstance(mask, np.ndarray), f"mask must be a numpy array. Got {type(mask)}"
        if np.issubdtype(mask.dtype, np.bool_) or np.issubdtype(mask.dtype, np.uint8):
            pass
        else:
            mask = (mask != 0).astype(np.uint8)
    
    ## make gaussFiltSize odd
    gaussFiltSize = int(np.ceil(gaussFiltSize))
    gaussFiltSize = gaussFiltSize + (gaussFiltSize % 2 == 0)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        n_iter,
        termination_eps,
    )
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(
        templateImage=im_template, 
        inputImage=im_moving, 
        warpMatrix=warp_matrix,
        motionType=warp_mode, 
        criteria=criteria, 
        inputMask=mask, 
        gaussFiltSize=gaussFiltSize
    )
    return warp_matrix


def apply_warp_transform(
    im_in: np.ndarray,
    warp_matrix: np.ndarray,
    interpolation_method: int = cv2.INTER_LINEAR, 
    borderMode: int = cv2.BORDER_CONSTANT, 
    borderValue: int = 0
) -> np.ndarray:
    """
    Apply a warp transform to an image. 
    Wrapper function for ``cv2.warpAffine`` and ``cv2.warpPerspective``. 
    RH 2022

    Args:
        im_in (np.ndarray): 
            Input image with any dimensions.
        warp_matrix (np.ndarray): 
            Warp matrix. Shape should be *(2, 3)* for affine transformations,
            and *(3, 3)* for homography. See ``cv2.findTransformECC`` for more
            info.
        interpolation_method (int): 
            Interpolation method. See ``cv2.warpAffine`` for more info. (Default
            is ``cv2.INTER_LINEAR``)
        borderMode (int): 
            Border mode. Determines how to handle pixels from outside the image
            boundaries. See ``cv2.warpAffine`` for more info. (Default is
            ``cv2.BORDER_CONSTANT``)
        borderValue (int): 
            Value to use for border pixels if borderMode is set to
            ``cv2.BORDER_CONSTANT``. (Default is *0*)

    Returns:
        (np.ndarray): 
            im_out (np.ndarray): 
                Transformed output image with the same dimensions as the input
                image.
    """
    if warp_matrix.shape == (2, 3):
        im_out = cv2.warpAffine(
            src=im_in,
            M=warp_matrix,
            dsize=(im_in.shape[1], im_in.shape[0]),
            dst=copy.copy(im_in),
            flags=interpolation_method + cv2.WARP_INVERSE_MAP,
            borderMode=borderMode,
            borderValue=borderValue
        )
        
    elif warp_matrix.shape == (3, 3):
        im_out = cv2.warpPerspective(
            src=im_in,
            M=warp_matrix,
            dsize=(im_in.shape[1], im_in.shape[0]), 
            dst=copy.copy(im_in), 
            flags=interpolation_method + cv2.WARP_INVERSE_MAP, 
            borderMode=borderMode, 
            borderValue=borderValue
        )

    else:
        raise ValueError(f"warp_matrix.shape {warp_matrix.shape} not recognized. Must be (2, 3) or (3, 3)")
    
    return im_out


def warp_matrix_to_remappingIdx(
    warp_matrix: Union[np.ndarray, torch.Tensor], 
    x: int, 
    y: int
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert a warp matrix (2x3 or 3x3) into remapping indices (2D). 
    RH 2023
    
    Args:
        warp_matrix (Union[np.ndarray, torch.Tensor]): 
            Warp matrix of shape *(2, 3)* for affine transformations, and *(3,
            3)* for homography.
        x (int): 
            Width of the desired remapping indices.
        y (int): 
            Height of the desired remapping indices.
        
    Returns:
        (Union[np.ndarray, torch.Tensor]): 
            remapIdx (Union[np.ndarray, torch.Tensor]): 
                Remapping indices of shape *(x, y, 2)* representing the x and y
                displacements in pixels.
    """
    assert warp_matrix.shape in [(2, 3), (3, 3)], f"warp_matrix.shape {warp_matrix.shape} not recognized. Must be (2, 3) or (3, 3)"
    assert isinstance(x, int) and isinstance(y, int), f"x and y must be integers"
    assert x > 0 and y > 0, f"x and y must be positive"

    if isinstance(warp_matrix, torch.Tensor):
        stack, meshgrid, arange, hstack, ones, float32, array = torch.stack, torch.meshgrid, torch.arange, torch.hstack, torch.ones, torch.float32, torch.as_tensor
        stack_partial = lambda x: stack(x, dim=0)
    elif isinstance(warp_matrix, np.ndarray):
        stack, meshgrid, arange, hstack, ones, float32, array = np.stack, np.meshgrid, np.arange, np.hstack, np.ones, np.float32, np.array
        stack_partial = lambda x: stack(x, axis=0)
    else:
        raise ValueError(f"warp_matrix must be a torch.Tensor or np.ndarray")

    # create the grid
    mesh = stack_partial(meshgrid(arange(x, dtype=float32), arange(y, dtype=float32)))
    mesh_coords = hstack((mesh.reshape(2,-1).T, ones((x*y, 1), dtype=float32)))
    
    # warp the grid
    mesh_coords_warped = (mesh_coords @ warp_matrix.T)
    mesh_coords_warped = mesh_coords_warped[:, :2] / mesh_coords_warped[:, 2:3] if warp_matrix.shape == (3, 3) else mesh_coords_warped  ## if homography, divide by z
    
    # reshape the warped grid
    remapIdx = mesh_coords_warped.T.reshape(2, y, x)

    # permute the axes to (x, y, 2)
    remapIdx = remapIdx.permute(1, 2, 0) if isinstance(warp_matrix, torch.Tensor) else remapIdx.transpose(1, 2, 0)

    return remapIdx


def remap_images(
    images: Union[np.ndarray, torch.Tensor],
    remappingIdx: Union[np.ndarray, torch.Tensor],
    backend: str = "torch",
    interpolation_method: str = 'linear',
    border_mode: str = 'constant',
    border_value: float = 0,
    device: str = 'cpu',
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies remapping indices to a set of images. Remapping indices, similar to
    flow fields, describe the index of the pixel to sample from rather than the
    displacement of each pixel. RH 2023

    Args:
        images (Union[np.ndarray, torch.Tensor]): 
            The images to be warped. Shapes can be *(N, C, H, W)*, *(C, H, W)*,
            or *(H, W)*.
        remappingIdx (Union[np.ndarray, torch.Tensor]): 
            The remapping indices, describing the index of the pixel to sample
            from. Shape is *(H, W, 2)*.
        backend (str): 
            The backend to use. Can be either ``'torch'`` or ``'cv2'``. (Default
            is ``'torch'``)
        interpolation_method (str): 
            The interpolation method to use. Options are ``'linear'``,
            ``'nearest'``, ``'cubic'``, and ``'lanczos'``. Refer to `cv2.remap`
            or `torch.nn.functional.grid_sample` for more details. (Default is
            ``'linear'``)
        border_mode (str): 
            The border mode to use. Options include ``'constant'``,
            ``'reflect'``, ``'replicate'``, and ``'wrap'``. Refer to `cv2.remap`
            for more details. (Default is ``'constant'``)
        border_value (float): 
            The border value to use. Refer to `cv2.remap` for more details.
            (Default is ``0``)
        device (str):
            The device to use for computations. Commonly either ``'cpu'`` or
            ``'gpu'``. (Default is ``'cpu'``)

    Returns:
        (Union[np.ndarray, torch.Tensor]):
            warped_images (Union[np.ndarray, torch.Tensor]):
                The warped images. The shape will be the same as the input
                images, which can be *(N, C, H, W)*, *(C, H, W)*, or *(H, W)*.
    """
    # Check inputs
    assert isinstance(images, (np.ndarray, torch.Tensor)), f"images must be a np.ndarray or torch.Tensor"
    assert isinstance(remappingIdx, (np.ndarray, torch.Tensor)), f"remappingIdx must be a np.ndarray or torch.Tensor"
    if images.ndim == 2:
        images = images[None, None, :, :]
    elif images.ndim == 3:
        images = images[None, :, :, :]
    elif images.ndim != 4:
        raise ValueError(f"images must be a 2D, 3D, or 4D array. Got shape {images.shape}")
    assert remappingIdx.ndim == 3, f"remappingIdx must be a 3D array of shape (H, W, 2). Got shape {remappingIdx.shape}"
    assert images.shape[-2] == remappingIdx.shape[0], f"images H ({images.shape[-2]}) must match remappingIdx H ({remappingIdx.shape[0]})"
    assert images.shape[-1] == remappingIdx.shape[1], f"images W ({images.shape[-1]}) must match remappingIdx W ({remappingIdx.shape[1]})"

    # Check backend
    if backend not in ["torch", "cv2"]:
        raise ValueError("Invalid backend. Supported backends are 'torch' and 'cv2'.")
    if backend == 'torch':
        if isinstance(images, np.ndarray):
            images = torch.as_tensor(images, device=device, dtype=torch.float32)
        elif isinstance(images, torch.Tensor):
            images = images.to(device=device).type(torch.float32)
        if isinstance(remappingIdx, np.ndarray):
            remappingIdx = torch.as_tensor(remappingIdx, device=device, dtype=torch.float32)
        elif isinstance(remappingIdx, torch.Tensor):
            remappingIdx = remappingIdx.to(device=device).type(torch.float32)
        interpolation = {
            'linear': 'bilinear',
            'nearest': 'nearest',
            'cubic': 'bicubic',
            'lanczos': 'lanczos',
        }[interpolation_method]
        border = {
            'constant': 'zeros',
            'reflect': 'reflection',
            'replicate': 'replication',
            'wrap': 'circular',
        }[border_mode]
        ## Convert remappingIdx to normalized grid
        normgrid = cv2RemappingIdx_to_pytorchFlowField(remappingIdx)

        # Apply remappingIdx
        warped_images = torch.nn.functional.grid_sample(
            images, 
            normgrid[None,...],
            mode=interpolation, 
            padding_mode=border, 
            align_corners=True,  ## align_corners=True is the default in cv2.remap. See documentation for details.
        )

    elif backend == 'cv2':
        assert isinstance(images, np.ndarray), f"images must be a np.ndarray when using backend='cv2'"
        assert isinstance(remappingIdx, np.ndarray), f"remappingIdx must be a np.ndarray when using backend='cv2'"
        ## convert to float32 if not uint8
        images = images.astype(np.float32) if images.dtype != np.uint8 else images
        remappingIdx = remappingIdx.astype(np.float32) if remappingIdx.dtype != np.uint8 else remappingIdx

        interpolation = {
            'linear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4,
        }[interpolation_method]
        borderMode = {
            'constant': cv2.BORDER_CONSTANT,
            'reflect': cv2.BORDER_REFLECT,
            'replicate': cv2.BORDER_REPLICATE,
            'wrap': cv2.BORDER_WRAP,
        }[border_mode]

        # Apply remappingIdx
        def remap(ims):
            out = np.stack([cv2.remap(
                im,
                remappingIdx[..., 0], 
                remappingIdx[..., 1], 
                interpolation=interpolation, 
                borderMode=borderMode, 
                borderValue=border_value,
            ) for im in ims], axis=0)
            return out
        warped_images = np.stack([remap(im) for im in images], axis=0)

    return warped_images.squeeze()


def remap_sparse_images(
    ims_sparse: Union[scipy.sparse.spmatrix, List[scipy.sparse.spmatrix]],
    remappingIdx: np.ndarray,
    method: str = 'linear',
    fill_value: float = 0,
    dtype: Union[str, np.dtype] = None,
    safe: bool = True,
    n_workers: int = -1,
    verbose: bool = True,
) -> List[scipy.sparse.csr_matrix]:
    """
    Remaps a list of sparse images using the given remap field.
    RH 2023

    Args:
        ims_sparse (Union[scipy.sparse.spmatrix, List[scipy.sparse.spmatrix]]): 
            A single sparse image or a list of sparse images.
        remappingIdx (np.ndarray): 
            An array of shape *(H, W, 2)* representing the remap field. It
            should be the same size as the images in ims_sparse.
        method (str): 
            Interpolation method to use. See ``scipy.interpolate.griddata``.
            Options are:
            \n
            * ``'linear'``
            * ``'nearest'``
            * ``'cubic'`` \n
            (Default is ``'linear'``)
        fill_value (float): 
            Value used to fill points outside the convex hull. (Default is
            ``0.0``)
        dtype (Union[str, np.dtype]): 
            The data type of the resulting sparse images. Default is ``None``,
            which will use the data type of the input sparse images.
        safe (bool): 
            If ``True``, checks if the image is 0D or 1D and applies a tiny
            Gaussian blur to increase the image width. (Default is ``True``)
        n_workers (int): 
            Number of parallel workers to use. Default is *-1*, which uses all
            available CPU cores.
        verbose (bool):
            Whether or not to use a tqdm progress bar. (Default is ``True``)

    Returns:
        (List[scipy.sparse.csr_matrix]): 
            ims_sparse_out (List[scipy.sparse.csr_matrix]): 
                A list of remapped sparse images.

    Raises:
        AssertionError: If the image and remappingIdx have different spatial
        dimensions.
    """
    # Ensure ims_sparse is a list of sparse matrices
    ims_sparse = [ims_sparse] if not isinstance(ims_sparse, list) else ims_sparse

    # Assert that all images are sparse matrices
    assert all(scipy.sparse.issparse(im) for im in ims_sparse), "All images must be sparse matrices."
    
    # Assert and retrieve dimensions
    dims_ims = ims_sparse[0].shape
    dims_remap = remappingIdx.shape
    assert dims_ims == dims_remap[:-1], "Image and remappingIdx should have same spatial dimensions."
    
    dtype = ims_sparse[0].dtype if dtype is None else dtype
    
    if safe:
        conv2d = Toeplitz_convolution2d(
            x_shape=(dims_ims[0], dims_ims[1]),
            k=np.array([[0   , 1e-8, 0   ],
                        [1e-8, 1,    1e-8],
                        [0   , 1e-8, 0   ]], dtype=dtype),
            dtype=dtype,
        )

    def warp_sparse_image(
        im_sparse: scipy.sparse.csr_matrix,
        remappingIdx: np.ndarray,
        method: str = method,
        fill_value: float = fill_value,
        safe: bool = safe
    ) -> scipy.sparse.csr_matrix:
        
        # Convert sparse image to COO format
        im_coo = scipy.sparse.coo_matrix(im_sparse)

        # Get coordinates and values from COO format
        rows, cols = im_coo.row, im_coo.col
        data = im_coo.data

        if safe:
            # can't use scipy.interpolate.griddata with 1d values
            is_horz = np.unique(rows).size == 1
            is_vert = np.unique(cols).size == 1

            # check for diagonal pixels 
            # slope = rise / run --- don't need to check if run==0 
            rdiff = np.diff(rows)
            cdiff = np.diff(cols)
            is_diag = np.unique(cdiff / rdiff).size == 1 if not np.any(rdiff==0) else False
            
            # best practice to just convolve instead of interpolating if too few pixels
            is_smol = rows.size < 3 

            if is_horz or is_vert or is_smol or is_diag:
                # warp convolved sparse image directly without interpolation
                return warp_sparse_image(im_sparse=conv2d(im_sparse, batching=False), remappingIdx=remappingIdx)

        # Get values at the grid points
        try:
            grid_values = scipy.interpolate.griddata(
                points=(rows, cols), 
                values=data, 
                xi=remappingIdx[:,:,::-1], 
                method=method, 
                fill_value=fill_value,
            )
        except Exception as e:
            raise Exception(f"Error interpolating sparse image. Something is either weird about one of the input images or the remappingIdx. Error: {e}")
        
        # Create a new sparse image from the nonzero pixels
        warped_sparse_image = scipy.sparse.csr_matrix(grid_values, dtype=dtype)
        warped_sparse_image.eliminate_zeros()
        return warped_sparse_image
    
    wsi_partial = partial(warp_sparse_image, remappingIdx=remappingIdx)
    ims_sparse_out = map_parallel(func=wsi_partial, args=[ims_sparse,], method='multithreading', n_workers=n_workers, prog_bar=verbose)
    return ims_sparse_out


def invert_remappingIdx(
    remappingIdx: np.ndarray, 
    method: str = 'linear', 
    fill_value: Optional[float] = np.nan
) -> np.ndarray:
    """
    Inverts a remapping index field.

    Requires the assumption that the remapping index field is invertible or bijective/one-to-one and non-occluding.
    Defined 'remap_AB' as a remapping index field that warps image A onto image B, then 'remap_BA' is the remapping index field that warps image B onto image A. This function computes 'remap_BA' given 'remap_AB'.

    RH 2023

    Args:
        remappingIdx (np.ndarray): 
            An array of shape *(H, W, 2)* representing the remap field.
        method (str):
            Interpolation method to use. See ``scipy.interpolate.griddata``. Options are:
            \n
            * ``'linear'``
            * ``'nearest'``
            * ``'cubic'`` \n
            (Default is ``'linear'``)
        fill_value (Optional[float]):
            Value used to fill points outside the convex hull. 
            (Default is ``np.nan``)

    Returns:
        (np.ndarray): 
                An array of shape *(H, W, 2)* representing the inverse remap field.
    """
    H, W, _ = remappingIdx.shape
    
    # Create the meshgrid of the original image
    grid = np.mgrid[:H, :W][::-1].transpose(1,2,0).reshape(-1, 2)
    
    # Flatten the original meshgrid and remappingIdx
    remapIdx_flat = remappingIdx.reshape(-1, 2)
    
    # Interpolate the inverse mapping using griddata
    map_BA = scipy.interpolate.griddata(
        points=remapIdx_flat, 
        values=grid, 
        xi=grid, 
        method=method,
        fill_value=fill_value,
    ).reshape(H,W,2)
    
    return map_BA

def invert_warp_matrix(
    warp_matrix: np.ndarray
) -> np.ndarray:
    """
    Inverts a provided warp matrix for the transformation A->B to compute the
    warp matrix for B->A.
    RH 2023

    Args:
        warp_matrix (np.ndarray): 
            A 2x3 or 3x3 array representing the warp matrix. Shape: *(2, 3)* or
            *(3, 3)*.

    Returns:
        (np.ndarray): 
            inverted_warp_matrix (np.ndarray):
                The inverted warp matrix. Shape: same as input.
    """
    if warp_matrix.shape == (2, 3):
        # Convert 2x3 affine warp matrix to 3x3 by appending [0, 0, 1] as the last row
        warp_matrix_3x3 = np.vstack((warp_matrix, np.array([0, 0, 1])))
    elif warp_matrix.shape == (3, 3):
        warp_matrix_3x3 = warp_matrix
    else:
        raise ValueError("Input warp_matrix must be of shape (2, 3) or (3, 3)")

    # Compute the inverse of the 3x3 warp matrix
    inverted_warp_matrix_3x3 = np.linalg.inv(warp_matrix_3x3)

    if warp_matrix.shape == (2, 3):
        # Convert the inverted 3x3 warp matrix back to 2x3 by removing the last row
        inverted_warp_matrix = inverted_warp_matrix_3x3[:2, :]
    else:
        inverted_warp_matrix = inverted_warp_matrix_3x3

    return inverted_warp_matrix


def compose_remappingIdx(
    remap_AB: np.ndarray,
    remap_BC: np.ndarray,
    method: str = 'linear',
    fill_value: Optional[float] = np.nan,
    bounds_error: bool = False,
) -> np.ndarray:
    """
    Composes two remapping index fields using scipy.interpolate.interpn.
    
    This function computes 'remap_AC' from 'remap_AB' and 'remap_BC', where
    'remap_AB' is a remapping index field that warps image A onto image B, and
    'remap_BC' is a remapping index field that warps image B onto image C.
    
    RH 2023

    Args:
        remap_AB (np.ndarray): 
            An array of shape *(H, W, 2)* representing the remap field from
            image A to image B.
        remap_BC (np.ndarray): 
            An array of shape *(H, W, 2)* representing the remap field from
            image B to image C.
        method (str): 
            Interpolation method to use. Either \n
            * ``'linear'``: Use linear interpolation (default).
            * ``'nearest'``: Use nearest interpolation.
            * ``'cubic'``: Use cubic interpolation.
        fill_value (Optional[float]): 
            The value used for points outside the interpolation domain. (Default
            is ``np.nan``)
        bounds_error (bool):
            If ``True``, a ValueError is raised when interpolated values are
            requested outside of the domain of the input data. (Default is
            ``False``)
    
    Returns:
        (np.ndarray): 
            remap_AC (np.ndarray): 
                An array of shape *(H, W, 2)* representing the remap field from
                image A to image C.
    """
    # Get the shape of the remap fields
    H, W, _ = remap_AB.shape
    
    # Combine the x and y components of remap_AB into a complex number
    # This is done to simplify the interpolation process
    AB_complex = remap_AB[:,:,0] + remap_AB[:,:,1]*1j

    # Perform the interpolation using interpn
    AC = scipy.interpolate.interpn(
        (np.arange(H), np.arange(W)), 
        AB_complex, 
        remap_BC.reshape(-1, 2)[:, ::-1], 
        method=method, 
        bounds_error=bounds_error, 
        fill_value=fill_value
    ).reshape(H, W)

    # Split the real and imaginary parts of the interpolated result to get the x and y components
    remap_AC = np.stack((AC.real, AC.imag), axis=-1)

    return remap_AC


def compose_transform_matrices(
    matrix_AB: np.ndarray, 
    matrix_BC: np.ndarray,
) -> np.ndarray:
    """
    Composes two transformation matrices to create a transformation from one
    image to another. RH 2023
    
    This function is used to combine two transformation matrices, 'matrix_AB'
    and 'matrix_BC'. 'matrix_AB' represents a transformation that warps an image
    A onto an image B. 'matrix_BC' represents a transformation that warps image
    B onto image C. The result is 'matrix_AC', a transformation matrix that
    would warp image A directly onto image C.
    
    Args:
        matrix_AB (np.ndarray): 
            A transformation matrix from image A to image B. The array can have
            the shape *(2, 3)* or *(3, 3)*.
        matrix_BC (np.ndarray): 
            A transformation matrix from image B to image C. The array can have
            the shape *(2, 3)* or *(3, 3)*.

    Returns:
        (np.ndarray): 
            matrix_AC (np.ndarray):
                A composed transformation matrix from image A to image C. The
                array has the shape *(2, 3)* or *(3, 3)*.

    Raises:
        AssertionError: 
            If the input matrices do not have the shape *(2, 3)* or *(3, 3)*.

    Example:
        .. highlight:: python
        .. code-block:: python

            # Define the transformation matrices
            matrix_AB = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            matrix_BC = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            # Compose the transformation matrices
            matrix_AC = compose_transform_matrices(matrix_AB, matrix_BC)
    """
    assert matrix_AB.shape in [(2, 3), (3, 3)], "Matrix AB must be of shape (2, 3) or (3, 3)."
    assert matrix_BC.shape in [(2, 3), (3, 3)], "Matrix BC must be of shape (2, 3) or (3, 3)."

    # If the input matrices are (2, 3), extend them to (3, 3) by adding a row [0, 0, 1]
    if matrix_AB.shape == (2, 3):
        matrix_AB = np.vstack((matrix_AB, [0, 0, 1]))
    if matrix_BC.shape == (2, 3):
        matrix_BC = np.vstack((matrix_BC, [0, 0, 1]))

    # Compute the product of the extended matrices
    matrix_AC = matrix_AB @ matrix_BC

    # If the resulting matrix is (3, 3) and has the last row [0, 0, 1], convert it back to a (2, 3) matrix
    if (matrix_AC.shape == (3, 3)) and np.allclose(matrix_AC[2], [0, 0, 1]):
        matrix_AC = matrix_AC[:2, :]

    return matrix_AC


def _make_idx_grid(
    im: Union[np.ndarray, object],
) -> Union[np.ndarray, object]:
    """
    Helper function to make a grid of indices for an image. Used in
    ``flowField_to_remappingIdx`` and ``remappingIdx_to_flowField``.

    Args:
        im (Union[np.ndarray, object]): 
            An image represented as a numpy ndarray or torch Tensor.

    Returns:
        (Union[np.ndarray, object]):
            idx_grid (Union[np.ndarray, object]):
                Index grid for the given image.
    """
    if isinstance(im, torch.Tensor):
        stack, meshgrid, arange = partial(torch.stack, dim=-1), partial(torch.meshgrid, indexing='xy'), partial(torch.arange, device=im.device, dtype=im.dtype)
    elif isinstance(im, np.ndarray):
        stack, meshgrid, arange = partial(np.stack, axis=-1), partial(np.meshgrid, indexing='xy'), partial(np.arange, dtype=im.dtype)
    return stack(meshgrid(arange(im.shape[1]), arange(im.shape[0]))) # (H, W, 2). Last dimension is (x, y).
def flowField_to_remappingIdx(
    ff: Union[np.ndarray, object],
) -> Union[np.ndarray, object]:
    """
    Convert a flow field to a remapping index. **WARNING**: Technically, it is
    not possible to convert a flow field to a remapping index, since the
    remapping index describes an interpolation mapping, while the flow field
    describes a displacement.
    RH 2023

    Args:
        ff (Union[np.ndarray, object]): 
            Flow field represented as a numpy ndarray or torch Tensor. 
            It describes the displacement of each pixel. 
            Shape *(H, W, 2)*. Last dimension is *(x, y)*.

    Returns:
        (Union[np.ndarray, object]): 
            ri (Union[np.ndarray, object]):
                Remapping index. It describes the index of the pixel in 
                the original image that should be mapped to the new pixel. 
                Shape *(H, W, 2)*.
    """
    ri = ff + _make_idx_grid(ff)
    return ri
def remappingIdx_to_flowField(
    ri: Union[np.ndarray, object],
) -> Union[np.ndarray, object]:
    """
    Convert a remapping index to a flow field. **WARNING**: Technically, it is
    not possible to convert a remapping index to a flow field, since the
    remapping index describes an interpolation mapping, while the flow field
    describes a displacement.
    RH 2023

    Args:
        ri (Union[np.ndarray, object]): 
            Remapping index represented as a numpy ndarray or torch Tensor. 
            It describes the index of the pixel in the original image that 
            should be mapped to the new pixel. Shape *(H, W, 2)*. Last 
            dimension is *(x, y)*.

    Returns:
        (Union[np.ndarray, object]): 
            ff (Union[np.ndarray, object]):
                Flow field. It describes the displacement of each pixel. 
                Shape *(H, W, 2)*.
    """
    ff = ri - _make_idx_grid(ri)
    return ff
def cv2RemappingIdx_to_pytorchFlowField(
    ri: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Converts remapping indices from the OpenCV format to the PyTorch format. In
    the OpenCV format, the displacement is in pixels relative to the top left
    pixel of the image. In the PyTorch format, the displacement is in pixels
    relative to the center of the image. RH 2023

    Args:
        ri (Union[np.ndarray, torch.Tensor]): 
            Remapping indices. Each pixel describes the index of the pixel in
            the original image that should be mapped to the new pixel. Shape:
            *(H, W, 2)*. The last dimension is (x, y).
        
    Returns:
        (Union[np.ndarray, torch.Tensor]): 
            normgrid (Union[np.ndarray, torch.Tensor]): 
                "Flow field", in the PyTorch format. Technically not a flow
                field, since it doesn't describe displacement. Rather, it is a
                remapping index relative to the center of the image. Shape: *(H,
                W, 2)*. The last dimension is (x, y).
    """
    assert isinstance(ri, torch.Tensor), f"ri must be a torch.Tensor. Got {type(ri)}"
    im_shape = torch.flipud(torch.as_tensor(ri.shape[:2], dtype=torch.float32, device=ri.device))  ## (W, H)
    normgrid = ((ri / (im_shape[None, None, :] - 1)) - 0.5) * 2  ## PyTorch's grid_sample expects grid values in [-1, 1] because it's a relative offset from the center pixel. CV2's remap expects grid values in [0, 1] because it's an absolute offset from the top-left pixel.
    ## note also that pytorch's grid_sample expects align_corners=True to correspond to cv2's default behavior.
    return normgrid


def add_text_to_images(
    images: np.array, 
    text: List[List[str]], 
    position: Tuple[int, int] = (10,10), 
    font_size: int = 1, 
    color: Tuple[int, int, int] = (255,255,255), 
    line_width: int = 1, 
    font: Optional[str] = None, 
    frameRate: int = 30
) -> np.array:
    """
    Adds text to images using ``cv2.putText()``.
    RH 2022

    Args:
        images (np.array):
            Frames of video or images. Shape: *(n_frames, height, width, n_channels)*.
        text (list of lists):
            Text to add to images.
            The outer list: one element per frame.
            The inner list: each element is a line of text.
        position (tuple):
            (x, y) position of the text (top left corner). (Default is *(10,10)*)
        font_size (int):
            Font size of the text. (Default is *1*)
        color (tuple):
            (r, g, b) color of the text. (Default is *(255,255,255)*)
        line_width (int):
            Line width of the text. (Default is *1*)
        font (str):
            Font to use. If ``None``, then will use ``cv2.FONT_HERSHEY_SIMPLEX``.
            See ``cv2.FONT...`` for more options. (Default is ``None``)
        frameRate (int):
            Frame rate of the video. (Default is *30*)

    Returns:
        (np.array): 
            images_with_text (np.array): 
                Frames of video or images with text added.
    """
    import cv2
    import copy
    
    if font is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
    
    images_cp = copy.deepcopy(images)
    for i_f, frame in enumerate(images_cp):
        for i_t, t in enumerate(text[i_f]):
            fn_putText = lambda frame_gray: cv2.putText(frame_gray, t, [position[0] , position[1] + i_t*font_size*30], font, font_size, color, line_width)
            if frame.ndim == 3:
                [fn_putText(frame[:,:,ii]) for ii in range(frame.shape[2])]
            else:
                fn_putText(frame)
    return images_cp


def resize_images(
    images: Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]], 
    new_shape: Tuple[int, int] = (100,100),
    interpolation: str = 'BILINEAR',
    antialias: bool = False,
    device: str = 'cpu',
    return_numpy: bool = True,
) -> np.ndarray:
    """
    Resizes images using the ``torchvision.transforms.Resize`` method.
    RH 2023

    Args:
        images (Union[np.ndarray, List[np.ndarray]], torch.Tensor, List[torch.Tensor]): 
            Images or frames of a video. Can be 2D, 3D, or 4D. 
            * For a 2D array: shape is *(height, width)*
            * For a 3D array: shape is *(n_frames, height, width)*
            * For a 4D array: shape is *(n_frames, n_channels, height, width)*
        new_shape (Tuple[int, int]): 
            The desired height and width of resized images as a tuple. 
            (Default is *(100, 100)*)
        interpolation (str): 
            The interpolation method to use. See ``torchvision.transforms.Resize`` 
            for options.
            * ``'NEAREST'``: Nearest neighbor interpolation
            * ``'NEAREST_EXACT'``: Nearest neighbor interpolation
            * ``'BILINEAR'``: Bilinear interpolation
            * ``'BICUBIC'``: Bicubic interpolation
        antialias (bool): 
            If ``True``, antialiasing will be used. (Default is ``False``)
        device (str): 
            The device to use for ``torchvision.transforms.Resize``. 
            (Default is ``'cpu'``)
        return_numpy (bool):
            If ``True``, then will return a numpy array. Otherwise, will return
            a torch tensor on the defined device. (Default is ``True``)
            
    Returns:
        (np.ndarray): 
            images_resized (np.ndarray): 
                Frames of video or images with overlay added.
    """
    ## Convert images to torch tensor
    if isinstance(images, list):
        if isinstance(images[0], np.ndarray):
            images = torch.stack([torch.as_tensor(im, device=device) for im in images], dim=0)
    elif isinstance(images, np.ndarray):
        images = torch.as_tensor(images, device=device)
    elif isinstance(images, torch.Tensor):
        images = images.to(device=device)
    else:
        raise ValueError(f"images must be a np.ndarray or torch.Tensor or a list of np.ndarray or torch.Tensor. Got {type(images)}")        
    
    ## Convert images to 4D
    def pad_to_4D(ims):
        if ims.ndim == 2:
            ims = ims[None, None, :, :]
        elif ims.ndim == 3:
            ims = ims[None, :, :, :]
        elif ims.ndim != 4:
            raise ValueError(f"images must be a 2D, 3D, or 4D array. Got shape {ims.shape}")
        return ims
    ndim_orig = images.ndim
    images = pad_to_4D(images)
    
    ## Get interpolation method
    try:
        interpolation = getattr(torchvision.transforms.InterpolationMode, interpolation.upper())
    except Exception as e:
        raise Exception(f"Invalid interpolation method. See torchvision.transforms.InterpolationMode for options. Error: {e}")

    resizer = torchvision.transforms.Resize(
        size=new_shape,
        interpolation=interpolation,
        antialias=antialias,
    ).to(device=device)
    images_resized = resizer(images)
       
    ## Convert images back to original shape
    def unpad_to_orig(ims, ndim_orig):
        if ndim_orig == 2:
            ims = ims[0,0,:,:]
        elif ndim_orig == 3:
            ims = ims[0,:,:,:]
        elif ndim_orig != 4:
            raise ValueError(f"images must be a 2D, 3D, or 4D array. Got shape {ims.shape}")
        return ims
    images_resized = unpad_to_orig(images_resized, ndim_orig)
        
    ## Convert images to numpy
    if return_numpy:
        images_resized = images_resized.detach().cpu().numpy()
    
    return images_resized


######################################################################################################################################
######################################################## TIME SERIES #################################################################
######################################################################################################################################

class Convolver_1d():
    """
    Class for 1D convolution.
    Uses torch.nn.functional.conv1d.
    Stores the convolution and edge correction kernels for repeated use.
    RH 2023
    
    Attributes:
        pad_mode (str): 
            Mode for padding. See ``torch.nn.functional.conv1d`` for details.
        dtype (object): 
            Data type for the convolution. Default is ``torch.float32``.
        kernel (object): 
            Convolution kernel as a tensor.
        trace_correction (object): 
            Kernel for edge correction.
            
    Args:
        kernel (Union[np.ndarray, object]):
            1D array to convolve with. The array can be a numpy array or a
            tensor.
        length_x (Optional[int]):
            Length of the array to be convolved. 
            Must not be ``None`` if pad_mode is not 'valid'. (Default is
            ``None``)
        dtype (object): 
            Data type to use for the convolution. 
            (Default is ``torch.float32``)
        pad_mode (str): 
            Mode for padding. 
            See ``torch.nn.functional.conv1d`` for details. 
            (Default is 'same')
        correct_edge_effects (bool): 
            Whether or not to correct for edge effects. 
            (Default is ``True``)
        device (str): 
            Device to use for computation. 
            (Default is 'cpu')
    """
    def __init__(
        self,
        kernel: Union[np.ndarray, object],
        length_x: Optional[int] = None,
        dtype: object = torch.float32,
        pad_mode: str = 'same',
        correct_edge_effects: bool = True,
        device: str = 'cpu',
    ):
        """
        Initializes the Convolver_1d with the given kernel, length of array to
        be convolved, dtype, padding mode, edge effect correction setting, and
        device.
        """
        self.pad_mode = pad_mode
        self.dtype = dtype

        ## convert kernel to torch tensor
        self.kernel = torch.as_tensor(kernel, dtype=dtype, device=device)[None,None,:]

        ## compute edge correction kernel
        if pad_mode != 'valid':
            assert length_x is not None, "Must provide length_x if pad_mode is not 'valid'"
            assert length_x >= kernel.shape[0], "length_x must be >= kernel.shape[0]"
            
            self.trace_correction = torch.conv1d(
                input=torch.ones((1,1,length_x), dtype=dtype, device=device),
                weight=self.kernel,
                padding=pad_mode,
            )[0,0,:] if correct_edge_effects else None
        else:
            self.trace_correction = None

        self.__call__ = self.convolve
            
    def convolve(self, arr: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Convolve array with kernel.
        
        Args:
            arr (Union[np.ndarray, torch.Tensor]):
                Array to convolve. 
                Convolution performed along the last axis.
                Must be 1D, 2D, or 3D.

        Returns:
            (Union[np.ndarray, torch.Tensor]): 
                out (Union[np.ndarray, torch.Tensor]):
                    The output tensor after performing convolution and
                    correcting for edge effects.

        Example:
            .. highlight:: python
            .. code-block:: python

                convolver = Convolver_1d(kernel=my_kernel)
                result = convolver.convolve(my_array)
        """
        ## make array 3D by adding singleton dimensions if necessary
        ndim = arr.ndim
        if ndim == 1:
            arr = arr[None,None,:]
        elif ndim == 2:
            arr = arr[None,:,:]
        assert arr.ndim == 3, "Array must be 1D or 2D or 3D"

        ## convolve along last axis
        out = torch.conv1d(
            input=torch.as_tensor(arr, dtype=self.dtype, device=self.kernel.device),
            weight=self.kernel,
            padding=self.pad_mode,
        )

        ## correct for edge effects
        if self.trace_correction is not None:
            out = out / self.trace_correction[None,None,:]
            
        ## remove singleton dimensions if necessary
        if ndim == 1:
            out = out[0,0,:]
        elif ndim == 2:
            out = out[0,:,:]
        return out
    
    def __repr__(self) -> str:
        return f"Convolver_1d(kernel shape={self.kernel.shape}, pad_mode={self.pad_mode})"
        

######################################################################################################################################
####################################################### FEATURIZATION ################################################################
######################################################################################################################################


class Toeplitz_convolution2d():
    """
    Convolve a 2D array with a 2D kernel using the Toeplitz matrix
    multiplication method. This class is ideal when 'x' is very sparse
    (density<0.01), 'x' is small (shape <(1000,1000)), 'k' is small (shape
    <(100,100)), and the batch size is large (e.g. 1000+). Generally, it is
    faster than scipy.signal.convolve2d when convolving multiple arrays with the
    same kernel. It maintains a low memory footprint by storing the toeplitz
    matrix as a sparse matrix.
    RH 2022

    Attributes:
        x_shape (Tuple[int, int]):
            The shape of the 2D array to be convolved.
        k (np.ndarray):
            2D kernel to convolve with.
        mode (str):
            Either ``'full'``, ``'same'``, or ``'valid'``. See
            scipy.signal.convolve2d for details.
        dtype (Optional[np.dtype]):
            The data type to use for the Toeplitz matrix.
            If ``None``, then the data type of the kernel is used.

    Args:
        x_shape (Tuple[int, int]):
            The shape of the 2D array to be convolved.
        k (np.ndarray):
            2D kernel to convolve with.
        mode (str):
            Convolution method to use, either ``'full'``, ``'same'``, or
            ``'valid'``.
            See scipy.signal.convolve2d for details. (Default is 'same')
        dtype (Optional[np.dtype]):
            The data type to use for the Toeplitz matrix. Ideally, this matches
            the data type of the input array. If ``None``, then the data type of
            the kernel is used. (Default is ``None``)

    Example:
        .. highlight:: python
        .. code-block:: python

            # create Toeplitz_convolution2d object
            toeplitz_convolution2d = Toeplitz_convolution2d(
                x_shape=(100,30),
                k=np.random.rand(10,10),
                mode='same',
            )
            toeplitz_convolution2d(
                x=scipy.sparse.csr_matrix(np.random.rand(5,3000)),
                batch_size=True,
            )
    """
    def __init__(
        self,
        x_shape: Tuple[int, int],
        k: np.ndarray,
        mode: str = 'same',
        dtype: Optional[np.dtype] = None,
    ):
        """
        Initializes the Toeplitz_convolution2d object and stores the Toeplitz
        matrix.
        """
        self.k = k = np.flipud(k.copy())
        self.mode = mode
        self.x_shape = x_shape
        dtype = k.dtype if dtype is None else dtype

        if mode == 'valid':
            assert x_shape[0] >= k.shape[0] and x_shape[1] >= k.shape[1], "x must be larger than k in both dimensions for mode='valid'"

        self.so = so = size_output_array = ( (k.shape[0] + x_shape[0] -1), (k.shape[1] + x_shape[1] -1))  ## 'size out' is the size of the output array

        ## make the toeplitz matrices
        t = toeplitz_matrices = [scipy.sparse.diags(
            diagonals=np.ones((k.shape[1], x_shape[1]), dtype=dtype) * k_i[::-1][:,None], 
            offsets=np.arange(-k.shape[1]+1, 1), 
            shape=(so[1], x_shape[1]),
            dtype=dtype,
        ) for k_i in k[::-1]]  ## make the toeplitz matrices for the rows of the kernel
        tc = toeplitz_concatenated = scipy.sparse.vstack(t + [scipy.sparse.dia_matrix((t[0].shape), dtype=dtype)]*(x_shape[0]-1))  ## add empty matrices to the bottom of the block due to padding, then concatenate

        ## make the double block toeplitz matrix
        self.dt = double_toeplitz = scipy.sparse.hstack([self._roll_sparse(
            x=tc, 
            shift=(ii>0)*ii*(so[1])  ## shift the blocks by the size of the output array
        ) for ii in range(x_shape[0])]).tocsr()
    
    def __call__(
        self,
        x: Union[np.ndarray, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix],
        batching: bool = True,
        mode: Optional[str] = None,
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """
        Convolve the input array with the kernel.

        Args:
            x (Union[np.ndarray, scipy.sparse.csc_matrix,
            scipy.sparse.csr_matrix]): 
                Input array(s) (i.e. image(s)) to convolve with the kernel. \n
                * If ``batching==False``: Single 2D array to convolve with the
                  kernel. Shape: *(self.x_shape[0], self.x_shape[1])*
                * If ``batching==True``: Multiple 2D arrays that have been
                  flattened into row vectors (with order='C'). \n
                Shape: *(n_arrays, self.x_shape[0]*self.x_shape[1])*

            batching (bool): 
                * ``False``: x is a single 2D array.
                * ``True``: x is a 2D array where each row is a flattened 2D
                  array. \n
                (Default is ``True``)

            mode (Optional[str]): 
                Defines the mode of the convolution. Options are 'full', 'same'
                or 'valid'. See `scipy.signal.convolve2d` for details. Overrides
                the mode set in __init__. (Default is ``None``)

        Returns:
            (Union[np.ndarray, scipy.sparse.csr_matrix]):
                out (Union[np.ndarray, scipy.sparse.csr_matrix]): 
                    * ``batching==True``: Multiple convolved 2D arrays that have
                      been flattened into row vectors (with order='C'). Shape:
                      *(n_arrays, height*width)*
                    * ``batching==False``: Single convolved 2D array of shape
                      *(height, width)*
        """
        if mode is None:
            mode = self.mode  ## use the mode that was set in the init if not specified
        issparse = scipy.sparse.issparse(x)
        
        if batching:
            x_v = x.T  ## transpose into column vectors
        else:
            x_v = x.reshape(-1, 1)  ## reshape 2D array into a column vector
        
        if issparse:
            x_v = x_v.tocsc()
        
        out_v = self.dt @ x_v  ## if sparse, then 'out_v' will be a csc matrix
            
        ## crop the output to the correct size
        if mode == 'full':
            p_t = 0
            p_b = self.so[0]+1
            p_l = 0
            p_r = self.so[1]+1
        if mode == 'same':
            p_t = (self.k.shape[0]-1)//2
            p_b = -(self.k.shape[0]-1)//2
            p_l = (self.k.shape[1]-1)//2
            p_r = -(self.k.shape[1]-1)//2

            p_b = self.x_shape[0]+1 if p_b==0 else p_b
            p_r = self.x_shape[1]+1 if p_r==0 else p_r
        if mode == 'valid':
            p_t = (self.k.shape[0]-1)
            p_b = -(self.k.shape[0]-1)
            p_l = (self.k.shape[1]-1)
            p_r = -(self.k.shape[1]-1)

            p_b = self.x_shape[0]+1 if p_b==0 else p_b
            p_r = self.x_shape[1]+1 if p_r==0 else p_r
        
        if batching:
            idx_crop = np.zeros((self.so), dtype=np.bool_)
            idx_crop[p_t:p_b, p_l:p_r] = True
            idx_crop = idx_crop.reshape(-1)
            out = out_v[idx_crop,:].T
        else:
            if issparse:
                out = out_v.reshape((self.so)).tocsc()[p_t:p_b, p_l:p_r]
            else:
                out = out_v.reshape((self.so))[p_t:p_b, p_l:p_r]  ## reshape back into 2D array and crop
        return out
    
    def _roll_sparse(
        self,
        x: scipy.sparse.csr_matrix,
        shift: int,
    ):
        """
        Roll columns of a sparse matrix.
        """
        out = x.copy()
        out.row += shift
        return out
    

######################################################################################################################################
##################################################### PARALLEL HELPERS ###############################################################
######################################################################################################################################


class ParallelExecutionError(Exception):
    """
    Exception class for errors that occur during parallel execution.
    Intended to be used with the ``map_parallel`` function.
    RH 2023

    Attributes:
        index (int):
            Index of the job that failed.
        original_exception (Exception):
            The original exception that was raised.
    """
    def __init__(self, index, original_exception):
        self.index = index
        self.original_exception = original_exception

    def __str__(self):
        return f"Job {self.index} raised an exception: {self.original_exception}"

def map_parallel(
    func: Callable, 
    args: List[Any], 
    method: str = 'multithreading', 
    n_workers: int = -1, 
    prog_bar: bool = True
) -> List[Any]:
    """
    Maps a function to a list of arguments in parallel.
    RH 2022

    Args:
        func (Callable): 
            The function to be mapped.
        args (List[Any]): 
            List of arguments to which the function should be mapped.
            Length of list should be equal to the number of arguments.
            Each element should then be an iterable for each job that is run.
        method (str): 
            Method to use for parallelization. Either \n
            * ``'multithreading'``: Use multithreading from concurrent.futures.
            * ``'multiprocessing'``: Use multiprocessing from concurrent.futures.
            * ``'mpire'``: Use mpire.
            * ``'serial'``: Use list comprehension. \n
            (Default is ``'multithreading'``)
        workers (int): 
            Number of workers to use. If set to -1, all available workers are used. (Default is ``-1``)
        prog_bar (bool): 
            Whether to display a progress bar using tqdm. (Default is ``True``)

    Returns:
        (List[Any]): 
            output (List[Any]): 
                List of results from mapping the function to the arguments.
                
    Example:
        .. highlight::python
        .. code-block::python

            result = map_parallel(max, [[1,2,3,4],[5,6,7,8]], method='multiprocessing', n_workers=3)
    """
    if n_workers == -1:
        n_workers = mp.cpu_count()

    ## Assert that args is a list
    assert isinstance(args, list), "args must be a list"
    ## Assert that each element of args is an iterable
    assert all([hasattr(arg, '__iter__') for arg in args]), "All elements of args must be iterable"

    ## Assert that each element has a length
    assert all([hasattr(arg, '__len__') for arg in args]), "All elements of args must have a length"
    ## Get number of arguments. If args is a generator, make None.
    n_args = len(args[0]) if hasattr(args, '__len__') else None
    ## Assert that all args are the same length
    assert all([len(arg) == n_args for arg in args]), "All args must be the same length"

    ## Make indices
    indices = np.arange(n_args)

    ## Prepare args_map (input to map function)
    args_map = [[func] * n_args, *args, indices]
        
    if method == 'multithreading':
        executor = ThreadPoolExecutor
    elif method == 'multiprocessing':
        executor = ProcessPoolExecutor
    elif method == 'mpire':
        import mpire
        executor = mpire.WorkerPool
    # elif method == 'joblib':
    #     import joblib
    #     return joblib.Parallel(n_jobs=workers)(joblib.delayed(func)(arg) for arg in tqdm(args, total=n_args, disable=prog_bar!=True))
    elif method == 'serial':
        return    list(tqdm(map(_func_wrapper_helper, *args_map), total=n_args, disable=prog_bar!=True))
    else:
        raise ValueError(f"method {method} not recognized")

    with executor(n_workers) as ex:
        return list(tqdm(ex.map(_func_wrapper_helper, *args_map), total=n_args, disable=prog_bar!=True))
def _func_wrapper_helper(*func_args_index):
    """
    Wrapper function to catch exceptions.
    
    Args:
    *func_args_index (tuple):
        Tuple of arguments to be passed to the function.
        Should take the form of (func, arg1, arg2, ..., argN, index)
        The last element is the index of the job.
    """
    func = func_args_index[0]
    args = func_args_index[1:-1]
    index = func_args_index[-1]

    try:
        return func(*args)
    except Exception as e:
        raise ParallelExecutionError(index, e)


######################################################################################################################################
######################################################### CLUSTERING #################################################################
######################################################################################################################################


def compute_cluster_similarity_matrices(
    s: Union[scipy.sparse.csr_matrix, np.ndarray, sparse.COO], 
    l: np.ndarray, 
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the similarity matrices for each cluster in ``l``. This algorithm
    works best on large and sparse matrices.
    RH 2023

    Args:
        s (Union[scipy.sparse.csr_matrix, np.ndarray, sparse.COO]): 
            Similarity matrix. Entries should be non-negative floats.
        l (np.ndarray): 
            Labels for each row of ``s``. Labels should ideally be integers.
        verbose (bool): 
            Whether to print warnings. (Default is ``True``)

    Returns:
        (tuple): tuple containing:
            labels_unique (np.ndarray):
                Unique labels in ``l``.
            cs_mean (np.ndarray):
                Similarity matrix for each cluster. Each element is the mean
                similarity between all the pairs of samples in each cluster.
                **Note** that the diagonal here only considers non-self
                similarity, which excludes the diagonals of ``s``.
            cs_max (np.ndarray):
                Similarity matrix for each cluster. Each element is the maximum
                similarity between all the pairs of samples in each cluster.
                **Note** that the diagonal here only considers non-self
                similarity, which excludes the diagonals of ``s``.
            cs_min (np.ndarray):
                Similarity matrix for each cluster. Each element is the minimum
                similarity between all the pairs of samples in each cluster.
                Will be 0 if there are any sparse elements between the two
                clusters.
    """
    import sparse
    import scipy.sparse

    l_arr = np.array(l)
    ss = scipy.sparse.csr_matrix(s.astype(np.float32))

    ## assert that all labels have at least two samples
    l_u ,l_c = np.unique(l_arr, return_counts=True)
    # assert np.all(l_c >= 2), "All labels must have at least two samples."
    ## assert that s is a square matrix
    assert ss.shape[0] == ss.shape[1], "Similarity matrix must be square."
    ## assert that s is non-negative
    assert (ss < 0).sum() == 0, "Similarity matrix must be non-negative."
    ## assert that l is a 1-D array
    assert len(l.shape) == 1, "Labels must be a 1-D array."
    ## assert that l is the same length as s
    assert len(l) == ss.shape[0], "Labels must be the same length as the similarity matrix."
    if verbose:
        ## Warn if s is not symmetric
        if not (ss - ss.T).sum() == 0:
            print("Warning: Similarity matrix is not symmetric.") if verbose else None
        ## Warn if s is not sparse
        if not isinstance(ss, (np.ndarray, sparse.COO, scipy.sparse.csr_matrix)):
            print("Warning: Similarity matrix is not a recognized sparse type or np.ndarray. Will attempt to convert to sparse.COO") if verbose else None
        ## Warn if diagonal is not all ones. It will be converted
        if not np.allclose(np.array(ss[range(ss.shape[0]), range(ss.shape[0])]), 1):
            print("Warning: Similarity matrix diagonal is not all ones. Will set diagonal to all ones.") if verbose else None
        ## Warn if there are any values greater than 1
        if (ss > 1).sum() > 0:
            print("Warning: Similarity matrix has values greater than 1.") if verbose else None
        ## Warn if there are NaNs. Set to 0.
        if (np.isnan(ss.data)).sum() > 0:
            print("Warning: Similarity matrix has NaNs. Will set to 0.") if verbose else None
            ss.data[np.isnan(ss.data)] = 0

    ## Make a boolean matrix for labels
    l_bool = sparse.COO(np.stack([l_arr == u for u in l_u], axis=0))
    samp_per_clust = l_bool.sum(1).todense()
    n_clusters = len(samp_per_clust)
    n_samples = ss.shape[0]
    
    ## Force diagonal to be 1s
    ss = ss.tolil()
    ss[range(n_samples), range(n_samples)] = 1
    ss = sparse.COO(ss)

    ## Compute the similarity matrix for each pair of clusters
    s_big_conj = ss[None,None,:,:] * l_bool[None,:,:,None] * l_bool[:,None,None,:]  ## shape: (n_clusters, n_clusters, n_samples, n_samples)
    s_big_diag = sparse.eye(n_samples) * l_bool[None,:,:,None] * l_bool[:,None,None,:]

    ## Compute the mean similarity matrix for each cluster
    samp_per_clust_crossGrid = samp_per_clust[:,None] * samp_per_clust[None,:]  ## shape: (n_clusters, n_clusters). This is the product of the number of samples in each cluster. Will be used to divide by the sum of similarities.
    norm_mat = samp_per_clust_crossGrid.copy()  ## above variable will be used again and this one will be mutated.
    fixed_diag = samp_per_clust * (samp_per_clust - 1)  ## shape: (n_clusters,). For the diagonal, we need to subtract 1 from the number of samples in each cluster because samples have only 1 similarity with themselves along the diagonal.
    norm_mat[range(n_clusters), range(n_clusters)] = fixed_diag  ## Correcting the diagonal
    s_big_sum_raw = s_big_conj.sum(axis=(2,3)).todense()
    s_big_sum_raw[range(n_clusters), range(n_clusters)] = s_big_sum_raw[range(n_clusters), range(n_clusters)] - samp_per_clust  ## subtract off the number of samples in each cluster from the diagonal
    cs_mean = s_big_sum_raw / norm_mat  ## shape: (n_clusters, n_clusters). Compute mean by finding the sum of the similarities and dividing by the norm_mat.

    ## Compute the min similarity matrix for each cluster
    ### This is done in two steps:
    #### 1. Compute the minimum similarity between each pair of clusters by inverting the similarity matrix and finding the maximum similarity between each pair of clusters.
    #### 2. Since the first step doesn't invert any values that happen to be 0 (since they are sparse), we need to find out if there are any 0 values there are in each cluster pair, and if there then the minimum similarity between the two clusters is 0.
    val_max = s_big_conj.max() + 1
    cs_min = s_big_conj.copy()
    cs_min.data = val_max - cs_min.data  ## Invert the values
    cs_min = cs_min.max(axis=(2,3))  ## Find the max similarity
    cs_min.data = val_max - cs_min.data  ## Invert the values back
    cs_min.fill_value = 0.0  ## Set the fill value to 0.0 since it gets messed up by these subtraction operations
    
    n_missing_values = (samp_per_clust_crossGrid - (s_big_conj > 0).sum(axis=(2,3)).todense())  ## shape: (n_clusters, n_clusters). Compute the number of missing values by subtracting the number of non-zero values from the number of samples in each cluster.
    # n_missing_values[range(len(samp_per_clust)), range(len(samp_per_clust))] = (samp_per_clust**2 - samp_per_clust) - ((s_big_conj[range(len(samp_per_clust)), range(len(samp_per_clust))] > 0).sum(axis=(1,2))).todense()  ## Correct the diagonal by subtracting the number of non-zero values from the number of samples in each cluster. This is because the diagonal is the number of samples in each cluster squared minus the number of samples in each cluster.
    bool_nonMissing_values = (n_missing_values == 0)  ## shape: (n_clusters, n_clusters). Make a boolean matrix for where there are no missing values.
    cs_min = cs_min.todense() * bool_nonMissing_values  ## Set the minimum similarity to 0 where there are missing values.

    ## Compute the max similarity matrix for each cluster
    cs_max = (s_big_conj - s_big_diag).max(axis=(2,3))

    return l_u, cs_mean, cs_max.todense(), cs_min


######################################################################################################################################
########################################################## TESTING ###################################################################
######################################################################################################################################


class Equivalence_checker():
    """
    Class for checking if all items are equivalent or allclose (almost equal) in
    two complex data structures. Can check nested lists, dicts, and other data
    structures. Can also optionally assert (raise errors) if all items are not
    equivalent. 
    RH 2023

    Attributes:
        _kwargs_allclose (Optional[dict]): 
            Keyword arguments for the `numpy.allclose` function.
        _assert_mode (bool):
            Whether to raise an assertion error if items are not close.

    Args:
        kwargs_allclose (Optional[dict]): 
            Keyword arguments for the `numpy.allclose` function. (Default is
            ``{'rtol': 1e-7, 'equal_nan': True}``)
        assert_mode (bool): 
            Whether to raise an assertion error if items are not close.
        verbose (bool):
            How much information to print out:
                * ``False`` / ``0``: No information printed out.
                * ``True`` / ``1``: Mismatched items only.
                * ``2``: All items printed out.
    """
    def __init__(
        self,
        kwargs_allclose: Optional[dict] = {'rtol': 1e-7, 'equal_nan': True},
        assert_mode=False,
        verbose=False,
    ) -> None:
        """
        Initializes the Allclose_checker.
        """
        self._kwargs_allclose = kwargs_allclose
        self._assert_mode = assert_mode
        self._verbose = verbose
        
    def _checker(
        self, 
        test: Any,
        true: Any, 
        path: Optional[List[str]] = None,
    ) -> bool:
        """
        Compares the test and true values using numpy's allclose function.

        Args:
            test (Union[dict, list, tuple, set, np.ndarray, int, float, complex,
            str, bool, None]): 
                Test value to compare.
            true (Union[dict, list, tuple, set, np.ndarray, int, float, complex,
            str, bool, None]): 
                True value to compare.
            path (Optional[List[str]]): 
                The path of the data structure that is currently being compared.
                (Default is ``None``)

        Returns:
            (bool): 
                result (bool): 
                    Returns True if all elements in test and true are close.
                    Otherwise, returns False.
        """
        try:
            ## If the dtype is a kind of string (or byte string) or object, then allclose will raise an error. In this case, just check if the values are equal.
            if np.issubdtype(test.dtype, np.str_) or np.issubdtype(test.dtype, np.bytes_) or test.dtype == np.object_:
                out = bool(np.all(test == true))
            else:
                out = np.allclose(test, true, **self._kwargs_allclose)
        except Exception as e:
            out = None  ## This is not False because sometimes allclose will raise an error if the arrays have a weird dtype among other reasons.
            warnings.warn(f"WARNING. Equivalence check failed. Path: {path}. Error: {e}") if self._verbose else None
            
        if out == False:
            if self._assert_mode:
                raise AssertionError(f"Equivalence check failed. Path: {path}.")
            if self._verbose:
                ## Come up with a way to describe the difference between the two values. Something like the following:
                ### IF the arrays are numeric, then calculate the relative difference
                dtypes_numeric = (np.number, np.bool_, np.integer, np.floating, np.complexfloating)
                if any([np.issubdtype(test.dtype, dtype) and np.issubdtype(true.dtype, dtype) for dtype in dtypes_numeric]):
                    diff = np.abs(test - true)
                    at = np.abs(true)
                    r_diff = diff / at if np.all(at != 0) else np.inf
                    r_diff_mean, r_diff_max, any_nan = np.nanmean(r_diff), np.nanmax(r_diff), np.any(np.isnan(r_diff))
                    ## fraction of mismatches
                    n_elements = np.prod(test.shape)
                    n_mismatches = np.sum(diff > 0)
                    frac_mismatches = n_mismatches / n_elements
                    ## Use scientific notation and round to 3 decimal places
                    reason = f"Equivalence: Relative difference: mean={r_diff_mean:.3e}, max={r_diff_max:.3e}, any_nan={any_nan}, n_elements={n_elements}, n_mismatches={n_mismatches}, frac_mismatches={frac_mismatches:.3e}"
                else:
                    reason = f"Values are not numpy numeric types. types: {test.dtype}, {true.dtype}"
        else:
            reason = "equivlance"

        return out, reason

    def __call__(
        self,
        test: Union[dict, list, tuple, set, np.ndarray, int, float, complex, str, bool, None], 
        true: Union[dict, list, tuple, set, np.ndarray, int, float, complex, str, bool, None], 
        path: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Compares the test and true values and returns the comparison result.
        Handles various data types including dictionaries, iterables,
        np.ndarray, scalars, strings, numbers, bool, and None.

        Args:
            test (Union[dict, list, tuple, set, np.ndarray, int, float, complex,
            str, bool, None]): 
                Test value to compare.
            true (Union[dict, list, tuple, set, np.ndarray, int, float, complex,
            str, bool, None]): 
                True value to compare.
            path (Optional[List[str]]): 
                The path of the data structure that is currently being compared.
                (Default is ``None``)

        Returns:
            Dict[Tuple[bool, str]]: 
                result Dict[Tuple[bool, str]]: 
                    The comparison result as a dictionary or a tuple depending
                    on the data types of test and true.
        """
        if path is None:
            path = ['']

        if len(path) > 0:
            if path[-1].startswith('_'):
                return (None, 'excluded from testing')

        ## NP.NDARRAY
        if isinstance(true, np.ndarray):
            result = self._checker(test, true, path)
        ## NP.SCALAR
        elif np.isscalar(true):
            if isinstance(test, (int, float, complex, np.number)):
                result = self._checker(np.array(test), np.array(true), path)
            else:
                result = (test == true, 'equivalence')
        ## NUMBER
        elif isinstance(true, (int, float, complex)):
            result = self._checker(test, true, path)
        ## DICT
        elif isinstance(true, dict):
            result = {}
            for key in true:
                if key not in test:
                    result[str(key)] = (False, 'key not found')
                else:
                    result[str(key)] = self.__call__(test[key], true[key], path=path + [str(key)])
        ## ITERATABLE
        elif isinstance(true, (list, tuple, set)):
            if len(true) != len(test):
                result = (False, 'length_mismatch')
            else:
                if all([isinstance(i, (int, float, complex, np.number)) for i in true]):
                    result = self._checker(np.array(test), np.array(true), path)
                else:
                    result = {}
                    for idx, (i, j) in enumerate(zip(test, true)):
                        result[str(idx)] = self.__call__(i, j, path=path + [str(idx)])
        ## STRING
        elif isinstance(true, str):
            result = (test == true, 'equivalence')
        ## BOOL
        elif isinstance(true, bool):
            result = (test == true, 'equivalence')
        ## NONE
        elif true is None:
            result = (test is None, 'equivalence')

        ## OBJECT with __dict__
        elif hasattr(true, '__dict__'):
            result = {}
            for key in true.__dict__:
                if key.startswith('_'):
                    continue
                if not hasattr(test, key):
                    result[str(key)] = (False, 'key not found')
                else:
                    result[str(key)] = self.__call__(getattr(test, key), getattr(true, key), path=path + [str(key)])
        ## N/A
        else:
            result = (None, 'not tested')

        if isinstance(result, tuple):
            if self._assert_mode:
                assert (result[0] != False), f"Equivalence check failed. Path: {path}. Reason: {result[1]}"

            if self._verbose > 0:
                ## Print False results
                if result[0] == False:
                    print(f"Equivalence check failed. Path: {path}. Reason: {result[1]}")
            if self._verbose > 1:
                ## Print True results
                if result[0] == True:
                    print(f"Equivalence check passed. Path: {path}. Reason: {result[1]}")
                elif result[0] is None:
                    print(f"Equivalence check not tested. Path: {path}. Reason: {result[1]}")

        return result


######################################################################################################################################
###################################################### OTHER FUNCTIONS ###############################################################
######################################################################################################################################

def get_balanced_class_weights(
    labels: np.ndarray
) -> np.ndarray:
    """
    Balances the weights for classifier training.
    
    RH, JZ 2022
    
    Args:
        labels (np.ndarray): 
            Array that includes a list of labels to balance the weights for
            classifier training. *shape: (n,)*
    
    Returns:
        (np.ndarray): 
            weights (np.ndarray): 
                Weights by samples. *shape: (n,)*
    """
    labels = labels.astype(np.int64)
    vals, counts = np.unique(labels, return_counts=True)
    weights = len(labels) / counts
    return weights


def get_balanced_sample_weights(
    labels: np.ndarray, 
    class_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Balances the weights for classifier training.
    
    RH/JZ 2022
    
    Args:
        labels (np.ndarray): 
            Array that includes a list of labels to balance the weights for
            classifier training. *shape: (n,)*
        class_weights (np.ndarray, Optional): 
            Optional parameter which includes an array of pre-fit class weights.
            If ``None``, weights will be calculated using the provided sample
            labels. (Default is ``None``)
    
    Returns:
        (np.ndarray): 
            sample_weights (np.ndarray): 
                Sample weights by labels. *shape: (n,)*
    """
    if type(class_weights) is not np.ndarray and type(class_weights) is not np.array:
        print('Warning: Class weights not pre-fit. Using provided sample labels.')
        weights = get_balanced_class_weights(labels)
    else:
        weights = class_weights
    sample_weights = weights[labels]
    return sample_weights


def safe_set_attr(
    obj: Any, 
    attr: str, 
    value: Any, 
    overwrite: bool = False,
) -> None:
    """
    Safely sets an attribute on an object. If the attribute is not present, it
    will be created. If the attribute is present, it will only be overwritten if
    ``overwrite`` is set to ``True``.
    RH 2024

    Args:
        obj (Any): 
            Object to set the attribute on.
        attr (str): 
            Attribute name.
        value (Any): 
            Value to set the attribute to.
        overwrite (bool): 
            Whether to overwrite the attribute if it already exists.
            (Default is ``False``)
    """
    if not hasattr(obj, attr):
        setattr(obj, attr, value)
    elif overwrite:
        setattr(obj, attr, value)