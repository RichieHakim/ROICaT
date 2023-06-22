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
from typing import List, Dict, Tuple, Union, Optional, Any, Callable, Iterable, Sequence, Type

import numpy as np
import torch
import scipy.sparse
import sparse
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

"""
All of these are from basic_neural_processing_modules
"""

def set_device(use_GPU=True, device_num=0, verbose=True):
    """
    Set torch.cuda device to use.
    Assumes that only one GPU is available or
     that you wish to use cuda:0 only.
    RH 2021

    Args:
        use_GPU (int):
            If 1, use GPU.
            If 0, use CPU.
    """
    if use_GPU:
        print(f'devices available: {[torch.cuda.get_device_properties(ii) for ii in range(torch.cuda.device_count())]}') if verbose else None
        device = f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("no GPU available. Using CPU.") if verbose else None
        else:
            print(f"Using device: '{device}': {torch.cuda.get_device_properties(device_num)}") if verbose else None
    else:
        device = "cpu"
        print(f"device: '{device}'") if verbose else None

    return device


def get_dir_contents(directory):
    '''
    Get the contents of a directory (does not
     include subdirectories).
    RH 2021

    Args:
        directory (str):
            path to directory
    
    Returns:
        folders (List):
            list of folder names
        files (List):
            list of file names
    '''
    walk = os.walk(directory, followlinks=False)
    folders = []
    files = []
    for ii,level in enumerate(walk):
        folders, files = level[1:]
        if ii==0:
            break
    return folders, files


def bounded_logspace(start, stop, num,):
    """
    Like np.logspace, but with a defined start and
     stop.
    RH 2022
    
    Args:
        start (float):
            First value in output array
        stop (float):
            Last value in output array
        num (int):
            Number of values in output array
            
    Returns:
        output (np.ndarray):
            Array of values
    """

    exp = 2  ## doesn't matter what this is, just needs to be > 1

    return exp ** np.linspace(np.log(start)/np.log(exp), np.log(stop)/np.log(exp), num, endpoint=True)


def make_batches(iterable, batch_size=None, num_batches=None, min_batch_size=0, return_idx=False, length=None):
    """
    Make batches of data or any other iterable.
    RH 2021

    Args:
        iterable (iterable):
            iterable to be batched
        batch_size (int):
            size of each batch
            if None, then batch_size based on num_batches
        num_batches (int):
            number of batches to make
        min_batch_size (int):
            minimum size of each batch
        return_idx (bool):
            whether to return the indices of the batches.
            output will be [start, end] idx
        length (int):
            length of the iterable.
            if None, then length is len(iterable)
            This is useful if you want to make batches of 
             something that doesn't have a __len__ method.
    
    Returns:
        output (iterable):
            batches of iterable
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


class lazy_repeat_item():
    """
    Makes a lazy iterator that repeats an item.
     RH 2021
    """
    def __init__(self, item, pseudo_length=None):
        """
        Args:
            item (any object):
                item to repeat
            pseudo_length (int):
                length of the iterator.
        """
        self.item = item
        self.pseudo_length = pseudo_length

    def __getitem__(self, i):
        """
        Args:
            i (int):
                index of item to return.
                Ignored if pseudo_length is None.
        """
        if self.pseudo_length is None:
            return self.item
        elif i < self.pseudo_length:
            return self.item
        else:
            raise IndexError('Index out of bounds')


    def __len__(self):
        return self.pseudo_length

    def __repr__(self):
        return repr(self.item)


def cosine_kernel_2D(center=(5,5), image_size=(11,11), width=5):
    """
    Generate a 2D cosine kernel
    RH 2021
    
    Args:
        center (tuple):  
            The mean position (X, Y) - where high value expected. 0-indexed. Make second value 0 to make 1D
        image_size (tuple): 
            The total image size (width, height). Make second value 0 to make 1D
        width (scalar): 
            The full width of one cycle of the cosine
    
    Return:
        k_cos (np.ndarray): 
            2D or 1D array of the cosine kernel
    """
    x, y = np.meshgrid(range(image_size[1]), range(image_size[0]))  # note dim 1:X and dim 2:Y
    dist = np.sqrt((y - int(center[1])) ** 2 + (x - int(center[0])) ** 2)
    dist_scaled = (dist/(width/2))*np.pi
    dist_scaled[np.abs(dist_scaled > np.pi)] = np.pi
    k_cos = (np.cos(dist_scaled) + 1)/2
    return k_cos


def idx2bool(idx, length=None):
    '''
    Converts a vector of indices to a boolean vector.
    RH 2021

    Args:
        idx (np.ndarray):
            1-D array of indices.
        length (int):
            Length of boolean vector.
            If None then length will be set to
             the maximum index in idx + 1.
    
    Returns:
        bool_vec (np.ndarray):
            1-D boolean array.
    '''
    if length is None:
        length = np.uint64(np.max(idx) + 1)
    out = np.zeros(length, dtype=np.bool_)
    out[idx] = True
    return out


def merge_dicts(dicts):
    out = {}
    [out.update(d) for d in dicts]
    return out    


def nanmax(arr, dim=None, keepdim=False):
    """
    Compute the max of an array ignoring any NaNs.
    RH 2021
    """
    if dim is None:
        kwargs = {}
    else:
        kwargs = {
            'dim': dim,
            'keepdim': keepdim,
        }
    
    nan_mask = torch.isnan(arr)
    arr_no_nan = arr.masked_fill(nan_mask, float('-inf'))
    return torch.max(arr_no_nan, **kwargs)

def nanmin(arr, dim=None, keepdim=False):
    """
    Compute the min of an array ignoring any NaNs.
    RH 2021
    """
    if dim is None:
        kwargs = {}
    else:
        kwargs = {
            'dim': dim,
            'keepdim': keepdim,
        }
    
    nan_mask = torch.isnan(arr)
    arr_no_nan = arr.masked_fill(nan_mask, float('inf'))
    return torch.min(arr_no_nan, **kwargs)


def scipy_sparse_to_torch_coo(sp_array, dtype=None):
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


def pydata_sparse_to_torch_coo(sp_array):
    coo = sparse.COO(sp_array)
    
    values = coo.data
#     indices = np.vstack((coo.row, coo.col))
    indices = coo.coords

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))

def squeeze_integers(intVec):
    """
    Make integers in an array consecutive numbers
     starting from the smallest value. 
    ie. [7,2,7,4,-1,0] -> [3,2,3,1,-1,0].
    Useful for removing unused class IDs.
    This is v3.
    RH 2023
    
    Args:
        intVec (np.ndarray):
            1-D array of integers.
    
    Returns:
        intVec_squeezed (np.ndarray):
            1-D array of integers with consecutive numbers
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


def idx_to_oneHot(arr, n_classes=None, dtype=None):
    """
    Convert an array of class indices to matrix of
     one-hot vectors.
    RH 2021

    Args:
        arr (np.ndarray):
            1-D array of class indices.
            Values should be integers >= 0.
            Values will be used as indices in the
             output array.
        n_classes (int):
            Number of classes.
    
    Returns:
        oneHot (np.ndarray):
            2-D array of one-hot vectors.
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

def prepare_filepath_for_saving(path, mkdir=False, allow_overwrite=True):
    """
    Checks if a file path is valid.
    RH 2022

    Args:
        path (str):
            Path to check.
        mkdir (bool):
            If True, creates parent directory if it does not exist.
        allow_overwrite (bool):
            If True, allows overwriting of existing file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True) if mkdir else None
    assert allow_overwrite or not Path(path).exists(), f'{path} already exists.'
    assert Path(path).parent.exists(), f'{Path(path).parent} does not exist.'
    assert Path(path).parent.is_dir(), f'{Path(path).parent} is not a directory.'

def pickle_save(
    obj, 
    path_save, 
    mode='wb', 
    zipCompress=False, 
    mkdir=False, 
    allow_overwrite=True,
    **kwargs_zipfile,
):
    """
    Saves an object to a pickle file.
    Allows for zipping of file.
    Uses pickle.dump.
    RH 2022

    Args:
        obj (object):
            Object to save.
        path_save (str):
            Path to save object to.
        mode (str):
            Mode to open file in.
            Can be:
                'wb' (write binary)
                'ab' (append binary)
                'xb' (exclusive write binary. Raises FileExistsError if file already exists.)
        zipCompress (bool):
            If True, compresses pickle file using zipfileCompressionMethod.
            This is similar to savez_compressed in numpy (with zipfile.ZIP_DEFLATED),
             and is useful for saving redundant and/or sparse arrays objects.
        mkdir (bool):
            If True, creates parent directory if it does not exist.
        allow_overwrite (bool):
            If True, allows overwriting of existing file.        
        kwargs_zipfile (dict):
            Keyword arguments that will be passed into zipfile.ZipFile.
            compression=zipfile.ZIP_DEFLATED by default.
            See https://docs.python.org/3/library/zipfile.html#zipfile-objects.
            Other options for 'compression' are (input can be either int or object):
                0:  zipfile.ZIP_STORED (no compression)
                8:  zipfile.ZIP_DEFLATED (usual zip compression)
                12: zipfile.ZIP_BZIP2 (bzip2 compression) (usually not as good as ZIP_DEFLATED)
                14: zipfile.ZIP_LZMA (lzma compression) (usually better than ZIP_DEFLATED but slower)
    """
    prepare_filepath_for_saving(path_save, mkdir=mkdir, allow_overwrite=allow_overwrite)

    if len(kwargs_zipfile)==0:
        kwargs_zipfile = {
            'compression': zipfile.ZIP_DEFLATED,
        }

    if zipCompress:
        with zipfile.ZipFile(path_save, 'w', **kwargs_zipfile) as f:
            f.writestr('data', pickle.dumps(obj))
    else:
        with open(path_save, mode) as f:
            pickle.dump(obj, f)

def pickle_load(
    filename, 
    zipCompressed=False,
    mode='rb'
):
    """
    Loads a pickle file.
    Allows for loading of zipped pickle files.
    RH 2022

    Args:
        filename (str):
            Path to pickle file.
        zipCompressed (bool):
            If True, then file is assumed to be a .zip file.
            This function will first unzip the file, then
             load the object from the unzipped file.
        mode (str):
            Mode to open file in.

    Returns:
        obj (object):
            Object loaded from pickle file.
    """
    if zipCompressed:
        with zipfile.ZipFile(filename, 'r') as f:
            return pickle.loads(f.read('data'))
    else:
        with open(filename, mode) as f:
            return pickle.load(f)


def matlab_save(
    obj,
    filepath,
    mkdir=False, 
    allow_overwrite=True,
    clean_string=True,
    list_to_objArray=True,
    none_to_nan=True,
    kwargs_scipy_savemat={
        'appendmat': True,
        'format': '5',
        'long_field_names': False,
        'do_compression': False,
        'oned_as': 'row',
    }
):
    """
    Saves data to a matlab file.
    Uses scipy.io.savemat.
    Provides additional functionality by cleaning strings,
     converting lists to object arrays, and converting None to
     np.nan.
    RH 2023

    Args:
        obj (dict):
            Data to save.
        filepath (str):
            Path to save file to.
        clean_string (bool):
            If True, converts strings to bytes.
        list_to_objArray (bool):
            If True, converts lists to object arrays.
        none_to_nan (bool):
            If True, converts None to np.nan.
        kwargs_scipy_savemat (dict):
            Keyword arguments to pass to scipy.io.savemat.
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


def deep_update_dict(dictionary, key, val, in_place=False):
    """
    Updates a dictionary with a new value.
    RH 2022

    Args:
        dictionary (Dict):
            dictionary to update
        key (list of str):
            Key to update
            List elements should be strings.
            Each element should be a hierarchical
             level of the dictionary.
            DEMO:
                deep_update_dict(params, ['dataloader_kwargs', 'prefetch_factor'], val)
        val (any):
            Value to update with
        in_place (bool):
            whether to update in place

    Returns:
        output (Dict):
            updated dictionary
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
        

def sparse_mask(x, mask_sparse, do_safety_steps=True):
    """
    Masks a sparse matrix with the non-zero elements of another
     sparse matrix.
    RH 2022

    Args:
        x (scipy.sparse.csr_matrix):
            sparse matrix to mask
        mask_sparse (scipy.sparse.csr_matrix):
            sparse matrix to mask with
        do_safety_steps (bool):
            whether to do safety steps to ensure that things
             are working as expected.

    Returns:
        output (scipy.sparse.csr_matrix):
            masked sparse matrix
    """
    if do_safety_steps:
        m = mask_sparse.copy()
        m.eliminate_zeros()
    else:
        m = mask_sparse
    return (m!=0).multiply(x)


def generalised_logistic_function(
    x, 
    a=0, 
    k=1, 
    b=1, 
    v=1, 
    q=1, 
    c=1,
    mu=0,
    ):
    '''
    Generalized logistic function
    See: https://en.wikipedia.org/wiki/Generalised_logistic_function
     for parameters and details
    RH 2021

    Args:
        a: the lower asymptote
        k: the upper asymptote when C=1
        b: the growth rate
        v: > 0, affects near which asymptote maximum growth occurs
        q: is related to the value Y (0). Center positions
        c: typically takes a value of 1
        mu: the center position of the function

    Returns:
        output:
            Logistic function
     '''
    if type(x) is np.ndarray:
        exp = np.exp
    elif type(x) is torch.Tensor:
        exp = torch.exp
    return a + (k-a) / (c + q*exp(-b*(x-mu)))**(1/v)


class scipy_sparse_csr_with_length(scipy.sparse.csr_matrix):
    """
    A scipy sparse matrix with a length attribute.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length = self.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        return self.__class__(super().__getitem__(key))


def find_nonredundant_idx(s):
    """
    Finds the indices of the nonredundant entries in a sparse matrix.
    Useful when you are manually populating a spare matrix and want to
     know which entries you have already populated.
    RH 2022

    Args:
        s (scipy.sparse.coo_matrix):
            Sparse matrix. Should be in coo format.

    Returns:
        idx_unique (np.ndarray):
            Indices of the nonredundant entries
    """
    if s.getformat() != 'coo':
        s = s.coo()
    idx_rowCol = np.vstack((s.row, s.col)).T
    u, idx_u = np.unique(idx_rowCol, axis=0, return_index=True)
    return idx_u
def remove_redundant_elements(s, inPlace=False):
    """
    Removes redundant entries from a sparse matrix.
    Useful when you are manually populating a spare matrix and want to
     remove redundant entries.
    RH 2022

    Args:
        s (scipy.sparse.coo_matrix):
            Sparse matrix. Should be in coo format.
        inPlace (bool):
            If True, the input matrix is modified in place.
            If False, a new matrix is returned.

    Returns:
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

def merge_sparse_arrays(s_list, idx_list, shape_full, remove_redundant=True, elim_zeros=True):
    """
    Merges a list of square sparse arrays into a single square sparse array.
    Note that no selection is performed for removing redundant entries;
     just whatever is selected by np.unique is kept.

    Args:
        s_list (list of scipy.sparse.csr_matrix):
            List of sparse arrays to merge.
            Each array can be of any shape.
        idx_list (list of np.ndarray int):
            List of arrays of integers. Each array should be of the same
             length as the corresponding array in s_list and should contain
             integers in the range [0, shape_full[0]). These integers
             represent the row/col indices in the full array.
        shape_full (tuple of int):
            Shape of the full array.
        remove_redundant (bool):
            If True, redundant entries are removed from the output array.
            If False, redundant entries are kept.

    Returns:
        s_full (scipy.sparse.csr_matrix):
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


#########################
# visualization helpers #
#########################

def rand_cmap(
    nlabels, 
    first_color_black=False, 
    last_color_black=False,
    verbose=True,
    under=[0,0,0],
    over=[0.5,0.5,0.5],
    bad=[0.9,0.9,0.9],
    ):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    import colorsys
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
    colors=[
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
    under=[0,0,0],
    over=[0.5,0.5,0.5],
    bad=[0.9,0.9,0.9],
    name='none'):
    """Create a colormap from a sequence of rgb values.
    Stolen with love from Alex (https://gist.github.com/ahwillia/3e022cdd1fe82627cbf1f2e9e2ad80a7ex)
    
    Args:
        colors (list):
            List of RGB values
        name (str):
            Name of the colormap

    Returns:
        cmap:
            Colormap

    Demo:
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

def confusion_matrix(y_hat, y_true, counts=False):
    """
    Compute the confusion matrix from y_hat and y_true.
    y_hat should be either predictions ().
    RH 2021 / JZ 2022

    Args:
        y_hat (np.ndarray): 
            numpy array of predictions or probabilities. 
            Either PREDICTIONS: 2-D array of booleans
             ('one hots') or 1-D array of predicted 
             class indices.
            Or PROBABILITIES: 2-D array floats ('one hot
             like')
        y_true (np.ndarray):
            Either 1-D array of true class indices OR a
             precomputed onehot matrix.
    """
    n_classes = max(np.max(y_true)+1, np.max(y_hat)+1)
    if y_hat.ndim == 1:
        y_hat = idx_to_oneHot(y_hat, n_classes).astype('int')
    cmat = y_hat.T @ idx_to_oneHot(y_true, n_classes)
    if not counts:
        cmat = cmat / np.sum(cmat, axis=0)[None,:]
    return cmat


def get_keep_nonnan_entries(original_features):
    """
    Get the image indices (axis 0) where all values at those indices are non-nan
    JZ 2022

    Args:
        original_features (np.ndarray): 
            image values (images x height x width)
    """
    has_nan = [np.unique(np.where(np.isnan(of))[0]) for of in original_features]
    return np.array([_ for _ in range(original_features.shape[0]) if _ not in has_nan])


def find_paths(
    dir_outer, 
    reMatch='filename', 
    find_files=True, 
    find_folders=False, 
    depth=0, 
    natsorted=True, 
    alg_ns=None, 
):
    """
    Search for files and/or folders recursively in a directory.
    RH 2022

    Args:
        dir_outer (str):
            Path to directory to search
        reMatch (str):
            Regular expression to match
            Each path name encountered will be compared using
             re.search(reMatch, filename). If the output is not None,
             the file will be included in the output.
        find_files (bool):
            Whether to find files
        find_folders (bool):
            Whether to find folders
        depth (int):
            Maximum folder depth to search.
            depth=0 means only search the outer directory.
            depth=2 means search the outer directory and two levels
             of subdirectories below it.
        natsorted (bool):
            Whether to sort the output using natural sorting
             with the natsort package.
        alg_ns (str):
            Algorithm to use for natural sorting.
            See natsort.ns or
             https://natsort.readthedocs.io/en/4.0.4/ns_class.html
             for options.
            Default is PATH.
            Other commons are INT, FLOAT, VERSION.

    Returns:
        paths (List of str):
            Paths to matched files and/or folders in the directory
    """
    import natsort
    if alg_ns is None:
        alg_ns = natsort.ns.PATH

    def get_paths_recursive_inner(dir_inner, depth_end, depth=0):
        paths = []
        for path in os.listdir(dir_inner):
            path = os.path.join(dir_inner, path)
            if os.path.isdir(path):
                if find_folders:
                    if re.search(reMatch, path) is not None:
                        paths.append(path)
                if depth < depth_end:
                    paths += get_paths_recursive_inner(path, depth_end, depth=depth+1)
            else:
                if find_files:
                    if re.search(reMatch, path) is not None:
                        paths.append(path)
        return paths

    paths = get_paths_recursive_inner(dir_outer, depth, depth=0)
    if natsorted:
        paths = natsort.natsorted(paths, alg=alg_ns)
    return paths

def find_paths_requireAll(
    dir_outer,
    filenames,
    find_files=True,
    find_folders=False,
    depth=0,
    natsorted=True,
    alg_ns=None,
    ):
    """
    Search for files and/or folders recursively in a directory.
    JZ 2023

    Args:
        dir_outer (str):
            Path to directory to search
        filenames (List of str):
            Filenames to match
        find_files (bool):
            Whether to find files
        find_folders (bool):
            Whether to find folders
        depth (int):
            Maximum folder depth to search.
            depth=0 means only search the outer directory.
            depth=2 means search the outer directory and two levels
                of subdirectories below it.
        natsorted (bool):
            Whether to sort the output using natural sorting
                with the natsort package.
        alg_ns (str):
            Algorithm to use for natural sorting.
            See natsort.ns or
                https://natsort.readthedocs.io/en/4.0.4/ns_class.html
                for options.
            Default is PATH.
            Other commons are INT, FLOAT, VERSION.
    """
    paths = find_paths(dir_outer, filenames[0], find_files=True, find_folders=False, depth=0, natsorted=True, alg_ns=None)
    list_reMatches = [[] for _ in range(len(filenames))]
    for str_filepath in paths:
        path_filepath = Path(str_filepath)
        bool_continue = False
        for filename in filenames:
            if not (path_filepath.parent / filename).resolve().exists():
                bool_continue = True
                break
        if bool_continue:
            continue
        else:
            for i, filename in enumerate(filenames):
                list_reMatches[i].append(str((path_filepath.parent / filename).resolve()))
    return list_reMatches

######################################################################################################################################
########################################################## INDEXING ##################################################################
######################################################################################################################################

def sparse_to_dense_fill(arr_s, fill_val=0.):
    """
    Converts a sparse array to a dense array and fills
     in sparse entries with a fill value.
    """
    import sparse
    s = sparse.COO(arr_s)
    s.fill_value = fill_val
    return s.todense()


######################################################################################################################################
######################################################## FILE HELPERS ################################################################
######################################################################################################################################

def download_file(
    url, 
    path_save, 
    check_local_first=True, 
    check_hash=False, 
    hash_type='MD5', 
    hash_hex=None,
    mkdir=False,
    allow_overwrite=True,
    write_mode='wb',
    verbose=True,
    chunk_size=1024,
):
    """
    Download a file from a URL to a local path using requests.
    Allows for checking if file already exists locally and
    checking the hash of the downloaded file against a provided hash.
    RH 2022

    Args:
        url (str):
            URL of file to download.
            If url is None, then no download is attempted.
        path_save (str):
            Path to save file to.
        check_local_first (bool):
            If True, checks if file already exists locally.
            If True and file exists locally, plans to skip download.
            If True and check_hash is True, checks hash of local file.
             If hash matches, skips download. If hash does not match, 
             downloads file.
        check_hash (bool):
            If True, checks hash of local or downloaded file against
             hash_hex.
        hash_type (str):
            Type of hash to use. Can be:
                'MD5', 'SHA1', 'SHA256', 'SHA512'
        hash_hex (str):
            Hash to compare to. In hex format (e.g. 'a1b2c3d4e5f6...').
            Can be generated using hash_file() or hashlib and .hexdigest().
            If check_hash is True, hash_hex must be provided.
        mkdir (bool):
            If True, creates parent directory of path_save if it does not exist.
        write_mode (str):
            Write mode for saving file. Should be one of:
                'wb' (write binary)
                'ab' (append binary)
                'xb' (write binary, fail if file exists)
        verbose (bool):
            If True, prints status messages.
        chunk_size (int):
            Size of chunks to download file in.
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


def hash_file(path, type_hash='MD5', buffer_size=65536):
    """
    Gets hash of a file.
    Based on: https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
    RH 2022

    Args:
        path (str):
            Path to file to be hashed.
        type_hash (str):
            Type of hash to use. Can be:
                'MD5'
                'SHA1'
                'SHA256'
                'SHA512'
        buffer_size (int):
            Buffer size for reading file.
            65536 corresponds to 64KB.

    Returns:
        hash_val (str):
            Hash of file.
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
    
def compare_file_hashes(
    hash_dict_true,
    dir_files_test=None,
    paths_files_test=None,
    verbose=True,
):
    """
    Compares hashes of files in a directory or list of paths
     to user provided hashes.
    RH 2022

    Args:
        hash_dict_true (dict):
            Dictionary of hashes to compare to.
            Each entry should be:
                {'key': ('filename', 'hash')}
        dir_files_test (str):
            Path to directory to compare hashes of files in.
            Unused if paths_files_test is not None.
        paths_files_test (list of str):
            List of paths to files to compare hashes of.
            Optional. dir_files_test is used if None.
        verbose (bool):
            Whether or not to print out failed comparisons.

    Returns:
        total_result (bool):
            Whether or not all hashes were matched.
        individual_results (list of bool):
            Whether or not each hash was matched.
        paths_matching (dict):
            Dictionary of paths that matched.
            Each entry is:
                {'key': 'path'}
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
    path_zip,
    path_extract=None,
    verbose=True,
):
    """
    Extracts a zip file.
    RH 2022

    Args:
        path_zip (str):
            Path to zip file.
        path_extract (str):
            Path (directory) to extract zip file to.
            If None, extracts to the same directory as the zip file.
        verbose (int):
            Whether to print progress.

    Returns:
        paths_extracted (list):
            List of paths to extracted files.
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
######################################################## H5 HANDLING #################################################################
######################################################################################################################################

## below is actually 'simple_load' from h5_handling
def h5_load(filepath, return_dict=True, verbose=False):
    """
    Returns a dictionary object containing the groups
    as keys and the datasets as values from
    given hdf file.
    RH 2023

    Args:
        filepath (string or Path): 
            Full path name of file to read.
        return_dict (bool):
            Whether or not to return a dict object (True)
            or an h5py object (False)
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
    
def show_item_tree(hObj=None , path=None, depth=None, show_metadata=True, print_metadata=False, indent_level=0):
    '''
    Recursive function that displays all the items 
     and groups in an h5 object or python dict
    RH 2021

    Args:
        hObj:
            'hierarchical Object'. hdf5 object OR python dictionary
        path (Path or string):
            If not None, then path to h5 object is used instead of hObj
        depth (int):
            how many levels deep to show the tree
        show_metadata (bool): 
            whether or not to list metadata with items
        print_metadata (bool): 
            whether or not to show values of metadata items
        indent_level: 
            used internally to function. User should leave blank

    ##############
    
    example usage:
        with h5py.File(path , 'r') as f:
            h5_handling.show_item_tree(f)
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
    X_in, 
    device='cpu', 
    mean_sub=True, 
    zscore=False, 
    rank=None, 
    return_cpu=True, 
    return_numpy=False
):
    """
    Principal Components Analysis for PyTorch.
    If using GPU, then call torch.cuda.empty_cache() after.
    RH 2021

    Args:
        X_in (torch.Tensor or np.ndarray):
            Data to be decomposed.
            2-D array. Columns are features, rows are samples.
            PCA will be performed column-wise.
        device (str):
            Device to use. ie 'cuda' or 'cpu'. Use a function 
             torch_helpers.set_device() to get.
        mean_sub (bool):
            Whether or not to mean subtract ('center') the 
             columns.
        zscore (bool):
            Whether or not to z-score the columns. This is 
             equivalent to doing PCA on the correlation-matrix.
        rank (int):
            Maximum estimated rank of decomposition. If None,
             then rank is X.shape[1]
        return_cpu (bool):  
            Whether or not to force returns/outputs to be on 
             the 'cpu' device. If False, and device!='cpu',
             then returns will be on device.
        return_numpy (bool):
            Whether or not to force returns/outputs to be
             numpy.ndarray type.

    Returns:
        components (torch.Tensor or np.ndarray):
            The components of the decomposition. 
            2-D array.
            Each column is a component vector. Each row is a 
             feature weight.
        scores (torch.Tensor or np.ndarray):
            The scores of the decomposition.
            2-D array.
            Each column is a score vector. Each row is a 
             sample weight.
        singVals (torch.Tensor or np.ndarray):
            The singular values of the decomposition.
            1-D array.
            Each element is a singular value.
        EVR (torch.Tensor or np.ndarray):
            The explained variance ratio of each component.
            1-D array.
            Each element is the explained variance ratio of
             the corresponding component.
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
###################################################### IMAGE_PROCESSING ##############################################################
######################################################################################################################################


def mask_image_border(
    im, 
    border_outer=None, 
    border_inner=None, 
    mask_value=0
):
    """
    Mask an image with a border.
    RH 2022

    Args:
        im (np.ndarray):
            Input image.
        border_outer (int or tuple(int)):
            Outer border width.
            Number of pixels along the border to mask.
            If None, don't mask the border.
            If tuple of ints, then (top, bottom, left, right).
        border_inner (int):
            Inner border width.
            Number of pixels in the center to mask. Will be a square.
            Value is the edge length of the center square.
        mask_value (float):
            Value to mask with.
    
    Returns:
        im_out (np.ndarray):
            Output image.
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
    frame_shape_y_x=(512,512),
    bandpass_spatialFs_bounds=[1/128, 1/3],
    order_butter=5,
    mask=None,
    dtype_fft=torch.complex64,
    plot_pref=False,
    verbose=False,
):
    """
    Make a Fourier domain mask for the phase correlation.
    Used in BWAIN.

    Args:
        frame_shape_y_x (Tuple[int]):
            Shape of the images that will be passed through
                this class.
        bandpass_spatialFs_bounds (tuple): 
            (lowcut, highcut) in spatial frequency
            A butterworth filter is used to make the mask.
        order_butter (int):
            Order of the butterworth filter.
        mask (np.ndarray):
            If not None, use this mask instead of making one.
        plot_pref (bool):
            If True, plot the absolute value of the mask.

    Returns:
        mask_fft (torch.Tensor):
            Mask in the Fourier domain.
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
def get_nd_butterworth_filter(shape, factor, order, high_pass, real,
                               dtype=np.float64, squared_butterworth=True):
    """Create a N-dimensional Butterworth mask for an FFT
    Parameters
    ----------
    shape : tuple of int
        Shape of the n-dimensional FFT and mask.
    factor : float
        Fraction of mask dimensions where the cutoff should be.
    order : float
        Controls the slope in the cutoff region.
    high_pass : bool
        Whether the filter is high pass (low frequencies attenuated) or
        low pass (high frequencies are attenuated).
    real : bool
        Whether the FFT is of a real (True) or complex (False) image
    squared_butterworth : bool, optional
        When True, the square of the Butterworth filter is used.
    Returns
    -------
    wfilt : ndarray
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
    im_template, 
    im_moving,
    warp_mode='euclidean',
    n_iter=5000,
    termination_eps=1e-10,
    mask=None,
    gaussFiltSize=1
):
    """
    Find the transformation between two images.
    Wrapper function for cv2.findTransformECC
    RH 2022

    Args:
        im_template (np.ndarray):
            Template image
            dtype must be: np.uint8 or np.float32
        im_moving (np.ndarray):
            Moving image
            dtype must be: np.uint8 or np.float32
        warp_mode (str):
            warp mode.
            See cv2.findTransformECC for more info.
            'translation': sets a translational motion model; warpMatrix is 2x3 with the first 2x2 part being the unity matrix and the rest two parameters being estimated.
            'euclidean':   sets a Euclidean (rigid) transformation as motion model; three parameters are estimated; warpMatrix is 2x3.
            'affine':      sets an affine motion model (DEFAULT); six parameters are estimated; warpMatrix is 2x3.
            'homography':  sets a homography as a motion model; eight parameters are estimated;`warpMatrix` is 3x3.
        n_iter (int):
            Number of iterations
        termination_eps (float):
            Termination epsilon.
            Threshold of the increment in the correlation
             coefficient between two iterations
        mask (np.ndarray):
            Binary mask. If None, no mask is used.
            Regions where mask is zero are ignored 
             during the registration.
        gaussFiltSize (int):
            gaussian filter size. If 0, no gaussian 
             filter is used.
    
    Returns:
        warp_matrix (np.ndarray):
            Warp matrix. See cv2.findTransformECC for more info.
            Can be applied using cv2.warpAffine or 
             cv2.warpPerspective.
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
    im_in,
    warp_matrix,
    interpolation_method=cv2.INTER_LINEAR, 
    borderMode=cv2.BORDER_CONSTANT, 
    borderValue=0
):
    """
    Apply a warp transform to an image.
    Wrapper function for cv2.warpAffine and cv2.warpPerspective
    RH 2022

    Args:
        im_in (np.ndarray):
            Input image
        warp_matrix (np.ndarray):
            Warp matrix. See cv2.findTransformECC for more info.
        interpolation_method (int):
            Interpolation method.
            See cv2.warpAffine for more info.
        borderMode (int):
            Border mode.
            Whether to use a constant border value or not.
            See cv2.warpAffine for more info.
        borderValue (int):
            Border value.

    Returns:
        im_out (np.ndarray):
            Output image
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
    warp_matrix, 
    x, 
    y, 
):
    """
    Convert an warp matrix (2x3 or 3x3) into remapping indices (2D).
    RH 2023
    
    Args:
        warp_matrix (np.ndarray or torch.Tensor): 
            Warp matrix of shape (2, 3) [affine] or (3, 3) [homography].
        x (int): 
            Width of the desired remapping indices.
        y (int): 
            Height of the desired remapping indices.
        
    Returns:
        remapIdx (np.ndarray or torch.Tensor): 
            Remapping indices of shape (x, y, 2) representing the x and y displacements
             in pixels.
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
    images,
    remappingIdx,
    backend="torch",
    interpolation_method='linear',
    border_mode='constant',
    border_value=0,
    device='cpu',
):
    """
    Apply remapping indices to a set of images.
    Remapping indices are like flow fields, but instead of describing
     the displacement of each pixel, they describe the index of the pixel
     to sample from.
    RH 2023

    Args:
        images (np.ndarray or torch.Tensor):
            Images to be warped.
            Shape (N, C, H, W) or (C, H, W) or (H, W).
        remappingIdx (np.ndarray or torch.Tensor):
            Remapping indices. Describes the index of the pixel to 
             sample from.
            Shape (H, W, 2).
        backend (str):
            Backend to use. Either "torch" or "cv2".
        interpolation_method (str):
            Interpolation method to use.
            Can be: 'linear', 'nearest', 'cubic', 'lanczos'
            See cv2.remap or torch.nn.functional.grid_sample for details.
        borderMode (str):
            Border mode to use.
            Can be: 'constant', 'reflect', 'replicate', 'wrap'
            See cv2.remap for details.
        borderValue (float):
            Border value to use.
            See cv2.remap for details.

    Returns:
        warped_images (np.ndarray or torch.Tensor):
            Warped images.
            Shape (N, C, H, W) or (C, H, W).
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
    ims_sparse: typing.Union[scipy.sparse.spmatrix, typing.List[scipy.sparse.spmatrix]],
    remappingIdx: np.ndarray,
    method: str = 'linear',
    fill_value: float = 0,
    dtype: typing.Union[str, np.dtype] = None,
    safe: bool = True,
    n_workers: int = -1,
    verbose=True,
) -> typing.List[scipy.sparse.csr_matrix]:
    """
    Remaps a list of sparse images using the given remap field.

    Args:
        ims_sparse (scipy.sparse.spmatrix or List[scipy.sparse.spmatrix]): 
            A single sparse image or a list of sparse images.
        remappingIdx (np.ndarray): 
            An array of shape (H, W, 2) representing the remap field. It
             should be the same size as the images in ims_sparse.
        method (str): 
            Interpolation method to use. 
            See scipy.interpolate.griddata.
            Options are 'linear', 'nearest', 'cubic'.
        fill_value (float, optional): 
            Value used to fill points outside the convex hull. 
        dtype (np.dtype): 
            The data type of the resulting sparse images. 
            Default is None, which will use the data type of the input
             sparse images.
        safe (bool): 
            If True, checks if the image is 0D or 1D and applies a tiny
             Gaussian blur to increase the image width.
        n_workers (int): 
            Number of parallel workers to use. 
            Default is -1, which uses all available CPU cores.
        verbose (bool):
            Whether or not to use a tqdm progress bar.

    Returns:
        ims_sparse_out (List[scipy.sparse.csr_matrix]): 
            A list of remapped sparse images.

    Raises:
        AssertionError: If the image and remappingIdx have different spatial dimensions.
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

        # Account for 1d images by convolving image with tiny gaussian kernel to increase image width
        if safe:
            ## append if there are < 3 nonzero pixels
            if (np.unique(rows).size == 1) or (np.unique(cols).size == 1) or (rows.size < 3):
                return warp_sparse_image(im_sparse=conv2d(im_sparse, batching=False), remappingIdx=remappingIdx)

        # Get values at the grid points
        grid_values = scipy.interpolate.griddata(
            points=(rows, cols), 
            values=data, 
            xi=remappingIdx[:,:,::-1], 
            method=method, 
            fill_value=fill_value,
        )

        # Create a new sparse image from the nonzero pixels
        warped_sparse_image = scipy.sparse.csr_matrix(grid_values, dtype=dtype)
        warped_sparse_image.eliminate_zeros()

        return warped_sparse_image
    
    wsi_partial = partial(warp_sparse_image, remappingIdx=remappingIdx)
    ims_sparse_out = map_parallel(func=wsi_partial, args=(ims_sparse,), method='multithreading', workers=n_workers, prog_bar=verbose)
    return ims_sparse_out


def invert_remappingIdx(
    remappingIdx: np.ndarray, 
    method: str = 'linear', 
    fill_value: typing.Optional[float] = np.nan
) -> np.ndarray:
    """
    Inverts a remapping index field.
    Requires assumption that the remapping index field is:
    - invertible or bijective / one-to-one.
    - non oc
    Example:
        Define 'remap_AB' as a remapping index field that warps
         image A onto image B. Then, 'remap_BA' is the remapping
         index field that warps image B onto image A. This function
         computes 'remap_BA' given 'remap_AB'.
        
    RH 2023

    Args:
        remappingIdx (np.ndarray): 
            An array of shape (H, W, 2) representing the remap field.
        method (str):
            Interpolation method to use.
            See scipy.interpolate.griddata.
            Options are 'linear', 'nearest', 'cubic'.
        fill_value (float, optional):
            Value used to fill points outside the convex hull.

    Returns:
        remappingIdx_inv (np.ndarray): 
            An array of shape (H, W, 2) representing the inverse remap field.
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

def invert_warp_matrix(warp_matrix: np.ndarray) -> np.ndarray:
    """
    Invert a given warp matrix (2x3 or 3x3) for A->B to compute the warp matrix for B->A.
    RH 2023

    Args:
        warp_matrix (numpy.ndarray): 
            A 2x3 or 3x3 numpy array representing the warp matrix.

    Returns:
        numpy.ndarray: 
            The inverted warp matrix.
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
    fill_value: typing.Optional[float] = np.nan,
    bounds_error: bool = False,
) -> np.ndarray:
    """
    Composes two remapping index fields using scipy.interpolate.interpn.
    Example:
        Define 'remap_AB' as a remapping index field that warps
        image A onto image B. Define 'remap_BC' as a remapping
        index field that warps image B onto image C. This function
        computes 'remap_AC' given 'remap_AB' and 'remap_BC'.
    RH 2023

    Args:
        remap_AB (np.ndarray): 
            An array of shape (H, W, 2) representing the remap field.
        remap_BC (np.ndarray): 
            An array of shape (H, W, 2) representing the remap field.
        method (str, optional): 
            The interpolation method to use, default is 'linear'.
        fill_value (float, optional): 
            The value to use for points outside the interpolation domain,
             default is np.nan.
        bounds_error (bool, optional):
            If True, when interpolated values are requested outside of
             the domain of the input data, a ValueError is raised.
             
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
    matrix_BC: np.ndarray
) -> np.ndarray:
    """
    Composes two transformation matrices.
    Example:
        Define 'matrix_AB' as a transformation matrix that warps
        image A onto image B. Define 'matrix_BC' as a transformation
        matrix that warps image B onto image C. This function
        computes 'matrix_AC' given 'matrix_AB' and 'matrix_BC'.
    RH 2023

    Args:
        matrix_AB (np.ndarray): 
            An array of shape (2, 3) or (3, 3) representing the transformation matrix.
        matrix_BC (np.ndarray): 
            An array of shape (2, 3) or (3, 3) representing the transformation matrix.

    Returns:
        matrix_AC (np.ndarray): 
            An array of shape (2, 3) or (3, 3) representing the composed transformation matrix.

    Raises:
        AssertionError: If the input matrices are not of shape (2, 3) or (3, 3).
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


def make_idx_grid(im):
    """
    Helper function to make a grid of indices for an image.
    Used in flowField_to_remappingIdx and remappingIdx_to_flowField.
    """
    if isinstance(im, torch.Tensor):
        stack, meshgrid, arange = partial(torch.stack, dim=-1), partial(torch.meshgrid, indexing='xy'), partial(torch.arange, device=im.device, dtype=im.dtype)
    elif isinstance(im, np.ndarray):
        stack, meshgrid, arange = partial(np.stack, axis=-1), partial(np.meshgrid, indexing='xy'), partial(np.arange, dtype=im.dtype)
    return stack(meshgrid(arange(im.shape[1]), arange(im.shape[0]))) # (H, W, 2). Last dimension is (x, y).
def flowField_to_remappingIdx(ff):
    """
    Convert a flow field to a remapping index.
    WARNING: Technically, it is not possible to convert a flow field
     to a remapping index, since the remapping index describes an
     interpolation mapping, while the flow field describes a displacement.
    RH 2023

    Args:
        ff (np.ndarray or torch.Tensor): 
            Flow field.
            Describes the displacement of each pixel.
            Shape (H, W, 2). Last dimension is (x, y).

    Returns:
        ri (np.ndarray or torch.Tensor):
            Remapping index.
            Describes the index of the pixel in the original
             image that should be mapped to the new pixel.
            Shape (H, W, 2)
    """
    ri = ff + make_idx_grid(ff)
    return ri
def remappingIdx_to_flowField(ri):
    """
    Convert a remapping index to a flow field.
    WARNING: Technically, it is not possible to convert a flow field
     to a remapping index, since the remapping index describes an
     interpolation mapping, while the flow field describes a displacement.
    RH 2023

    Args:
        ri (np.ndarray or torch.Tensor):
            Remapping index.
            Describes the index of the pixel in the original
             image that should be mapped to the new pixel.
            Shape (H, W, 2). Last dimension is (x, y).

    Returns:
        ff (np.ndarray or torch.Tensor):
            Flow field.
            Describes the displacement of each pixel.
            Shape (H, W, 2)
    """
    ff = ri - make_idx_grid(ri)
    return ff
def cv2RemappingIdx_to_pytorchFlowField(ri):
    """
    Convert remapping indices from the OpenCV format to the PyTorch format.
    cv2 format: Displacement is in pixels relative to the top left pixel
     of the image.
    PyTorch format: Displacement is in pixels relative to the center of
     the image.
    RH 2023

    Args:
        ri (np.ndarray or torch.Tensor): 
            Remapping indices.
            Each pixel describes the index of the pixel in the original
             image that should be mapped to the new pixel.
            Shape (H, W, 2). Last dimension is (x, y).

    Returns:
        normgrid (np.ndarray or torch.Tensor):
            "Flow field", in the PyTorch format.
            Technically not a flow field, since it doesn't describe
             displacement. Rather, it is a remapping index relative to
             the center of the image.
            Shape (H, W, 2). Last dimension is (x, y).
    """
    assert isinstance(ri, torch.Tensor), f"ri must be a torch.Tensor. Got {type(ri)}"
    im_shape = torch.flipud(torch.as_tensor(ri.shape[:2], dtype=torch.float32, device=ri.device))  ## (W, H)
    normgrid = ((ri / (im_shape[None, None, :] - 1)) - 0.5) * 2  ## PyTorch's grid_sample expects grid values in [-1, 1] because it's a relative offset from the center pixel. CV2's remap expects grid values in [0, 1] because it's an absolute offset from the top-left pixel.
    ## note also that pytorch's grid_sample expects align_corners=True to correspond to cv2's default behavior.
    return normgrid


######################################################################################################################################
######################################################## TIME SERIES #################################################################
######################################################################################################################################

class Convolver_1d():
    """
    Class for 1D convolution.
    Uses torch.nn.functional.conv1d.
    Stores the convolution and edge correction kernels
     for repeated use.
    RH 2023
    """
    def __init__(
        self,
        kernel,
        length_x: int=None,
        dtype=torch.float32,
        pad_mode: str='same',
        correct_edge_effects: bool=True,
        device='cpu',
    ):
        """
        Args:
            kernel (np.ndarray or torch.Tensor):
                1D array to convolve with.
            length_x (int):
                Length of the array to be convolved.
                Must not be None if pad_mode is not 'valid'.
            pad_mode (str):
                Mode for padding.
                See torch.nn.functional.conv1d for details.
            correct_edge_effects (bool):
                Whether or not to correct for edge effects.
            device (str):
                Device to use.
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
            
    def convolve(self, arr) -> torch.Tensor:
        """
        Convolve array with kernel.
        Args:
            arr (np.ndarray or torch.Tensor):
                Array to convolve.
                Convolution performed along the last axis.
                ndim must be 1 or 2 or 3.
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
    
    def __call__(self, arr):
        return self.convolve(arr)
    def __repr__(self) -> str:
        return f"Convolver_1d(kernel shape={self.kernel.shape}, pad_mode={self.pad_mode})"
        

######################################################################################################################################
####################################################### FEATURIZATION ################################################################
######################################################################################################################################


class Toeplitz_convolution2d:
    """
    Convolve a 2D array with a 2D kernel using the Toeplitz matrix 
     multiplication method.
    Allows for SPARSE 'x' inputs. 'k' should remain dense.
    Ideal when 'x' is very sparse (density<0.01), 'x' is small
     (shape <(1000,1000)), 'k' is small (shape <(100,100)), and
     the batch size is large (e.g. 1000+).
    Generally faster than scipy.signal.convolve2d when convolving mutliple
     arrays with the same kernel. Maintains low memory footprint by
     storing the toeplitz matrix as a sparse matrix.

    See: https://stackoverflow.com/a/51865516 and https://github.com/alisaaalehi/convolution_as_multiplication
     for a nice illustration.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.convolution_matrix.html 
     for 1D version.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html#scipy.linalg.matmul_toeplitz 
     for potential ways to make this implementation faster.

    Test with: tests.test_toeplitz_convolution2d()
    RH 2022
    """
    def __init__(
        self,
        x_shape,
        k,
        mode='same',
        dtype=None,
    ):
        """
        Initialize the convolution object.
        Makes the Toeplitz matrix and stores it.

        Args:
            x_shape (tuple):
                The shape of the 2D array to be convolved.
            k (np.ndarray):
                2D kernel to convolve with
            mode (str):
                'full', 'same' or 'valid'
                see scipy.signal.convolve2d for details
            dtype (np.dtype):
                The data type to use for the Toeplitz matrix.
                Ideally, this matches the data type of the input array.
                If None, then the data type of the kernel is used.
        """
        self.k = k = np.flipud(k.copy())
        self.mode = mode
        self.x_shape = x_shape
        self.dtype = k.dtype if dtype is None else dtype

        if mode == 'valid':
            assert x_shape[0] >= k.shape[0] and x_shape[1] >= k.shape[1], "x must be larger than k in both dimensions for mode='valid'"

        self.so = so = size_output_array = ( (k.shape[0] + x_shape[0] -1), (k.shape[1] + x_shape[1] -1))  ## 'size out' is the size of the output array

        ## make the toeplitz matrices
        t = toeplitz_matrices = [scipy.sparse.diags(
            diagonals=np.ones((k.shape[1], x_shape[1]), dtype=self.dtype) * k_i[::-1][:,None], 
            offsets=np.arange(-k.shape[1]+1, 1), 
            shape=(so[1], x_shape[1]),
            dtype=self.dtype,
        ) for k_i in k[::-1]]  ## make the toeplitz matrices for the rows of the kernel
        tc = toeplitz_concatenated = scipy.sparse.vstack(t + [scipy.sparse.dia_matrix((t[0].shape), dtype=self.dtype)]*(x_shape[0]-1))  ## add empty matrices to the bottom of the block due to padding, then concatenate

        ## make the double block toeplitz matrix
        self.dt = double_toeplitz = scipy.sparse.hstack([self._roll_sparse(
            x=tc, 
            shift=(ii>0)*ii*(so[1])  ## shift the blocks by the size of the output array
        ) for ii in range(x_shape[0])]).tocsr()
    
    def __call__(
        self,
        x,
        batching=True,
        mode=None,
    ):
        """
        Convolve the input array with the kernel.

        Args:
            x (np.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix):
                Input array(s) (i.e. image(s)) to convolve with the kernel
                If batching==False: Single 2D array to convolve with the kernel.
                    shape: (self.x_shape[0], self.x_shape[1])
                    type: np.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
                If batching==True: Multiple 2D arrays that have been flattened
                 into row vectors (with order='C').
                    shape: (n_arrays, self.x_shape[0]*self.x_shape[1])
                    type: np.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
            batching (bool):
                If False, x is a single 2D array.
                If True, x is a 2D array where each row is a flattened 2D array.
            mode (str):
                'full', 'same' or 'valid'
                see scipy.signal.convolve2d for details
                Overrides the mode set in __init__.

        Returns:
            out (np.ndarray or scipy.sparse.csr_matrix):
                If batching==True: Multiple convolved 2D arrays that have been flattened
                 into row vectors (with order='C').
                    shape: (n_arrays, height*width)
                    type: np.ndarray or scipy.sparse.csc_matrix
                If batching==False: Single convolved 2D array of shape (height, width)
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
        x,
        shift,
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


def map_parallel(func, args, method='multithreading', workers=-1, prog_bar=True):
    """
    Map a function to a list of arguments in parallel.
    RH 2022

    Args:
        func (function):
            Function to map.
        args (list):
            List of arguments to map the function to.
            len(args) should be equal to the number of arguments.
            If the function takes multiple arguments, args should be an
             iterable (e.g. list, tuple, generator) of length equal to
             the number of arguments. Each element can then be an iterable
             for each iteration of the function.
        method (str):
            Method to use for parallelization. Options are:
                'multithreading': Use multithreading from concurrent.futures.
                'multiprocessing': Use multiprocessing from concurrent.futures.
                'mpire': Use mpire
                # 'joblib': Use joblib.Parallel
                'serial': Use list comprehension
        workers (int):
            Number of workers to use. If -1, use all available.
        prog_bar (bool):
            Whether to show a progress bar with tqdm.

    Returns:
        output (list):
            List of results from mapping the function to the arguments.
    """
    if workers == -1:
        workers = mp.cpu_count()

    ## Get number of arguments. If args is a generator, make None.
    n_args = len(args[0]) if hasattr(args, '__len__') else None

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
        # return [func(*arg) for arg in tqdm(args, disable=prog_bar!=True)]
        return list(tqdm(map(func, *args), total=n_args, disable=prog_bar!=True))
    else:
        raise ValueError(f"method {method} not recognized")

    with executor(workers) as ex:
        return list(tqdm(ex.map(func, *args), total=n_args, disable=prog_bar!=True))


def simple_multithreading(func, args, workers):
    with ThreadPoolExecutor(workers) as ex:
        res = ex.map(func, *args)
    return list(res)
def simple_multiprocessing(func, args, workers):
    with ProcessPoolExecutor(workers) as ex:
        res = ex.map(func, *args)
    return list(res)


######################################################################################################################################
######################################################### CLUSTERING #################################################################
######################################################################################################################################


def compute_cluster_similarity_matrices(
    s, 
    l, 
    verbose=True
):
    """
    Compute the similarity matrices for each cluster in l.
    This algorithm works best on large and sparse matrices. 
    RH 2023

    Args:
        s (scipy.sparse.csr_matrix or np.ndarray or sparse.COO):
            Similarity matrix.
            Entries should be non-negative floats.
        l (np.ndarray):
            Labels for each row of s.
            Labels should be integers ideally.
        verbose (bool):
            Whether to print warnings.

    Returns:
        cs_mean (np.ndarray):
            Similarity matrix for each cluster.
            Each element is the mean similarity between all the pairs
             of samples in each cluster.
            Note that the diagonal here only considers non-self similarity,
             which excludes the diagonals of s.
        cs_max (np.ndarray):
            Similarity matrix for each cluster.
            Each element is the maximum similarity between all the pairs
             of samples in each cluster.
            Note that the diagonal here only considers non-self similarity,
             which excludes the diagonals of s.
        cs_min (np.ndarray):
            Similarity matrix for each cluster.
            Each element is the minimum similarity between all the pairs
             of samples in each cluster. Will be 0 if there are any sparse
             elements between the two clusters.
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

    return cs_mean, cs_max.todense(), cs_min


######################################################################################################################################
###################################################### OTHER FUNCTIONS ###############################################################
######################################################################################################################################

def get_balanced_class_weights(labels):
    """
    Balances sample ways for classification
    
    RH/JZ 2022
    
    labels: np.array
        Includes list of labels to balance the weights for classifier training
    returns weights by samples
    """
    labels = labels.astype(np.int64)
    vals, counts = np.unique(labels, return_counts=True)
    weights = len(labels) / counts
    return weights

def get_balanced_sample_weights(labels, class_weights=None):
    """
    Balances sample ways for classification
    
    RH/JZ 2022
    
    labels: np.array
        Includes list of labels to balance the weights for classifier training
    returns weights by samples
    """
#     print(type(class_weights), class_weights)
    
    if type(class_weights) is not np.ndarray and type(class_weights) is not np.array:
        print('Warning: Class weights not pre-fit. Using provided sample labels.')
        weights = get_balanced_class_weights(labels)
    else:
        weights = class_weights
    sample_weights = weights[labels]
    return sample_weights

class Figure_Saver:
    """
    Class for saving figures
    RH 2022/JZ 2023
    """
    def __init__(
        self,
        dir_save: str=None,
        format_save: list=['png'],
        kwargs_savefig: dict={
            'bbox_inches': 'tight',
            'pad_inches': 0.1,
            'transparent': True,
            'dpi': 300,
        },
        overwrite: bool=False,
        verbose: int=1,
    ):
        """
        Initializes Figure_Saver object

        Args:
            dir_save (str):
                Directory to save the figure. Used if path_config is None.
                Must be specified if path_config is None.
            format (list of str):
                Format(s) to save the figure. Default is 'png'.
                Others: ['png', 'svg', 'eps', 'pdf']
            overwrite (bool):
                If True, then overwrite the file if it exists.
            kwargs_savefig (dict):
                Keyword arguments to pass to fig.savefig().
            verbose (int):
                Verbosity level.
                0: No output.
                1: Warning.
                2: All info.
        """
        self.dir_save = str(Path(dir_save).resolve().absolute()) if dir_save is not None else None

        assert isinstance(format_save, list), "RH ERROR: format_save must be a list of strings"
        assert all([isinstance(f, str) for f in format_save]), "RH ERROR: format_save must be a list of strings"
        self.format_save = format_save

        assert isinstance(kwargs_savefig, dict), "RH ERROR: kwargs_savefig must be a dictionary"
        self.kwargs_savefig = kwargs_savefig

        self.overwrite = overwrite
        self.verbose = verbose

    def save(
        self,
        fig,
        path_save: str=None,
        dir_save: str=None,
        name_file: str=None,

        path_save_notes: str=None,
        notes: str=None,
        note_ext: str='txt',
    ):
        """
        Save the figures.

        Args:
            fig (matplotlib.figure.Figure):
                Figure to save.
            path_save (str):
                Path to save the figure.
                Should not contain suffix.
                If None, then the dir_save must be specified here or in
                 the initialization and name_file must be specified.
            dir_save (str):
                Directory to save the figure. If None, then the directory
                 specified in the initialization is used.
            name_file (str):
                Name of the file to save. If None, then the name of 
                the figure is used.
        """
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure), "RH ERROR: fig must be a matplotlib.figure.Figure"

        ## Get path_save
        if path_save is not None:
            assert len(Path(path_save).suffix) == 0, "RH ERROR: path_save must not contain suffix"
            path_save = [str(Path(path_save).resolve()) + '.' + f for f in self.format_save]
        else:
            assert (dir_save is not None) or (self.dir_save is not None), "RH ERROR: dir_save must be specified if path_save is None"
            assert name_file is not None, "RH ERROR: name_file must be specified if path_save is None"

            ## Get dir_save
            dir_save = self.dir_save if dir_save is None else str(Path(dir_save).resolve())

            ## Get figure title
            if name_file is None:
                titles = [a.get_title() for a in fig.get_axes() if a.get_title() != '']
                name_file = '.'.join(titles)
            path_save = [str(Path(dir_save) / (name_file + '.' + f)) for f in self.format_save]
        
        if notes is not None:
            if path_save_notes is not None:
                assert len(Path(path_save_notes).suffix) == 0, "JZ ERROR: path_save_notes must not contain suffix"
                path_save_notes = str(Path(path_save_notes).resolve()) + '.' + note_ext
            elif dir_save is not None:
                assert (dir_save is not None) or (self.dir_save is not None), "RH ERROR: dir_save must be specified if path_save is None"
                assert name_file is not None, "JZ ERROR: name_file must be specified if path_save is None"
                path_save_notes = str(Path(dir_save) / (name_file + '.' + note_ext))
        else:
            path_save_notes = None

        ## Save figure
        for path, form in zip(path_save, self.format_save):
            if Path(path).exists():
                if self.overwrite:
                    print(f'RH Warning: Overwriting file. File: {path} already exists.') if self.verbose > 0 else None
                else:
                    print(f'RH Warning: Not saving anything. File exists and overwrite==False. {path} already exists.') if self.verbose > 0 else None
                    return None
            print(f'FR: Saving figure {path} as format(s): {form}') if self.verbose > 1 else None
            fig.savefig(path, format=form, **self.kwargs_savefig)
        ## Save notes
        if Path(path_save).exists():
            if self.overwrite:
                print(f'RH Warning: Overwriting file. File: {path} already exists.') if self.verbose > 0 else None
            else:
                print(f'RH Warning: Not saving anything. File exists and overwrite==False. {path} already exists.') if self.verbose > 0 else None
                return None
        print(f'FR: Saving figure {path} as format(s): {form}') if self.verbose > 1 else None
        fig.savefig(path, format=form, **self.kwargs_savefig)

        if path_save_notes is not None and notes is not None:
            with open(path_save_notes, 'w') as f:
                f.write(notes)
    
    def save_batch(
        self,
        figs,
        dir_save: str=None,
        names_files: str=None,
    ):
        """
        Save all figures in a list.

        Args:
            figs (list of matplotlib.figure.Figure):
                Figures to save.
            dir_save (str):
                Directory to save the figure. If None, then the directory
                 specified in the initialization is used.
            name_file (str):
                Name of the file to save. If None, then the name of 
                the figure is used.
        """
        import matplotlib.pyplot as plt
        assert isinstance(figs, list), "RH ERROR: figs must be a list of matplotlib.figure.Figure"
        assert all([isinstance(fig, plt.Figure) for fig in figs]), "RH ERROR: figs must be a list of matplotlib.figure.Figure"

        ## Get dir_save
        dir_save = self.dir_save if dir_save is None else str(Path(dir_save).resolve())

        for fig, name_file in zip(figs, names_files):
            self.save(fig, name_file=name_file, dir_save=dir_save)

    def __call__(
        self,
        fig,
        path_save: str=None,
        name_file: str=None,
        dir_save: str=None,
    ):
        """
        Calls save() method.
        """
        self.save(fig, path_save=path_save, name_file=name_file, dir_save=dir_save)

    def __repr__(self):
        return f"Figure_Saver(dir_save={self.dir_save}, format={self.format_save}, overwrite={self.overwrite}, kwargs_savefig={self.kwargs_savefig}, verbose={self.verbose})"

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

            ## Start the session
            self.next_img()
            self._root.mainloop()
        except Exception as e:
            warnings.warn('Error initializing image labeler: ' + str(e))

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
                * ``'list'``: [(idx, label), (idx, label), ...]
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
            # out = dict(self.labels_)
            # ## Check for duplicate indices
            # if len(out) != len(self.labels_):
            #     warnings.warn('Duplicate indices found in labels. Only the last label for each index is returned.')
            # return out
            return self.labels_
        elif kind == 'list':
            # return self.labels_
            return self.labels_.items()
        elif kind == 'dataframe':
            # return {'index': np.array([x[0] for x in self.labels_], dtype=np.int64), 'label': np.array([x[1] for x in self.labels_])}
            return {'index': np.array(list(self.labels_.keys()), dtype=np.int64), 'label': np.array(list(self.labels_.values()), dtype=str)}


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