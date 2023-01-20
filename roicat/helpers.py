from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import copy
import pickle
import re
import zipfile
import gc

import numpy as np
import torch
import scipy.sparse
import sparse
import hdfdict
from tqdm import tqdm

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


def h5_lazy_load(path=None):
    """
    Returns a lazy dictionary object (specific
    to hdfdict package) containing the groups
    as keys and the datasets as values from
    given hdf file.
    RH 2021

    Args:
        path (string or Path): 
            Full path name of file to read.
    
    Returns:
        h5_dict (LazyHdfDict):
            LazyHdfDict object containing the groups
    """
    
    h5Obj = hdfdict.load(str(path), **{'mode': 'r'})
    
    return h5Obj


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
