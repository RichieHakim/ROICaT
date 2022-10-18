from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import os
import hashlib
from pathlib import Path
import copy
import pickle

import numpy as np
import torch
import scipy.sparse
import sparse
# import torch_sparse as ts
import gc

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


def simple_multithreading(func, args, workers):
    with ThreadPoolExecutor(workers) as ex:
        res = ex.map(func, *args)
    return list(res)
def simple_multiprocessing(func, args, workers):
    with ProcessPoolExecutor(workers) as ex:
        res = ex.map(func, *args)
    return list(res)


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
    out = np.zeros(length, dtype=np.bool8)
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
     starting from 0. ie. [7,2,7,4,1] -> [3,2,3,1,0].
    Useful for removing unused class IDs from y_true
     and outputting something appropriate for softmax.
    This is v2. The old version is busted.
    RH 2021
    
    Args:
        intVec (np.ndarray):
            1-D array of integers.
    
    Returns:
        intVec_squeezed (np.ndarray):
            1-D array of integers with consecutive numbers
    """
    uniques = np.unique(intVec)
    # unique_positions = np.arange(len(uniques))
    unique_positions = np.arange(uniques.min(), uniques.max()+1)
    return unique_positions[np.array([np.where(intVec[ii]==uniques)[0] for ii in range(len(intVec))]).squeeze()]


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
        dtype = np.bool8 if dtype is None else dtype
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

def pickle_save(obj, path_save, mode='wb', mkdir=False, allow_overwrite=True):
    Path(path_save).parent.mkdir(parents=True, exist_ok=True) if mkdir else None
    assert allow_overwrite or not Path(path_save).exists(), f'{path_save} already exists.'
    assert Path(path_save).parent.exists(), f'{Path(path_save).parent} does not exist.'
    assert Path(path_save).parent.is_dir(), f'{Path(path_save).parent} is not a directory.'
    with open(path_save, mode) as f:
        pickle.dump(obj, f,)
def pickle_load(filename, mode='rb'):
    with open(filename, mode) as f:
        return pickle.load(f)


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
    RH 2021

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
#     print(y_hat)
    if y_hat.ndim == 1:
        y_hat = idx_to_oneHot(y_hat, n_classes).astype('int')
#     print(y_hat)
    cmat = y_hat.T @ idx_to_oneHot(y_true, n_classes)
#     print(cmat)
    if not counts:
        cmat = cmat / np.sum(cmat, axis=0)[None,:]
#     print(cmat)
    return cmat


def get_keep_nonnan_entries(original_features):
    """
    Gets the image indices where a value is nan and returns a list of image indicies without any nan values
    original_features: np.array of image values (images x height x width)
    """
    has_nan = np.unique(np.where(np.isnan(original_features))[0])
    return np.array([_ for _ in range(original_features.shape[0]) if _ not in has_nan])


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
        rank = X.shape[1]
    
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
