"""
Unit tests for ROICaT

The functions in this module are intended to be 
 found and run by pytest.

To run the tests, use the command (in a terminal):
    pytest -v test_unit.py
            ^
          verbose
"""

from pathlib import Path

import warnings
import pytest

import numpy as np
import scipy.sparse

from roicat import helpers, util

######################################################################################################################################
##################################################### TEST core packages #############################################################
######################################################################################################################################

def test_core_packages():
    """
    Runs pytest on the core packages.
    """
    
    corePackages = [
        'numpy',
        'scipy.sparse',
        'matplotlib',
        'sklearn',
        'tqdm',
        'umap',
        'hdbscan',
        'kymatio',
        'cv2',
        'sparse',
        'optuna',
    ]

    torchPackages = [
        'torch',
        'torchvision',
    ]

    for pkg in corePackages + torchPackages:
        try:
            exec(f'import {pkg}')
        except ModuleNotFoundError:
            warnings.warn(f'RH Warning: {pkg} not found. Skipping tests.')

    ## test numpy
    np.random.seed(0)
    arr1 = np.random.rand(1000, 10)
    arr2 = np.random.rand(10, 1000)
    arr3 = (arr1 @ arr2).mean()
    assert np.allclose(arr3, 2.5, rtol=0.1), 'RH Error: numpy test failed.'

    ## test scipy.sparse
    arr1 = scipy.sparse.rand(1000, 10, density=0.1).tocoo().tocsr()
    arr2 = scipy.sparse.rand(10, 1000, density=0.1).tocoo().tocsr()
    arr3 = (arr1 @ arr2).mean()
    assert np.allclose(arr3, 0.025, rtol=0.2), 'RH Error: scipy.sparse test failed.'

######################################################################################################################################
################################################### TEST data_importing.py ###########################################################
######################################################################################################################################

def test_data_suite2p(dir_data_test, array_hasher):
    """
    Test data_importing.Data_suite2p.
    RH 2022

    Args:
        dir_data_test (str):
            pytest fixture.
            Path to the test data directory.
    """
    
    ##########
    ## TEST 1: Basic import of multiple stat.npy + ops.npy files
    ##########

    ## Get paths to test data
    paths_stat = helpers.find_paths(
        dir_outer=str(Path(dir_data_test) / 'data__stat_ops_small__valerio_rbp10_plane0'),
        reMatch='stat.npy',
        find_files=True,
        find_folders=False,
        depth=2,
        natsorted=True,
    )
    paths_ops = [str(Path(p).parent / 'ops.npy') for p in paths_stat]
    assert all([Path(p).exists() for p in paths_stat]), 'FR Error: one or more stat.npy files do not exist.'
    assert all([Path(p).exists() for p in paths_ops]), 'FR Error: one or more ops.npy files do not exist.'
    print(f'Found {len(paths_stat)} stat.npy files and {len(paths_ops)} ops.npy files.')
    print(f'paths_stat: {paths_stat}')

    ## Import class
    from roicat.data_importing import Data_suite2p

    ## Get default parameters
    params = util.make_params_default_tracking(dir_networkFiles=dir_data_test)

    ## Instantiate class with test data
    data = Data_suite2p(
        paths_statFiles=paths_stat,
        paths_opsFiles=paths_ops,
        um_per_pixel=params['importing']['um_per_pixel'],
        new_or_old_suite2p='new',
        out_height_width=params['importing']['out_height_width'],
        type_meanImg='meanImgE',
        # FOV_images=None,
        centroid_method=params['importing']['centroid_method'],
        # class_labels=None,
    #     FOV_height_width=(512, 1024),
        verbose=True,
    )

    ## Test that the class was instantiated correctly
    ### General attributes
    assert all([isinstance(p, str) for p in data.paths_stat]), 'FR Error: data.paths_stat.dtype != str'
    assert all([p == p2 for p, p2 in zip(data.paths_stat, paths_stat)]), 'FR Error: data.paths_stat != paths_stat'
    assert all([isinstance(p, str) for p in data.paths_ops]), 'FR Error: data.paths_ops.dtype != str'
    assert all([p == p2 for p, p2 in zip(data.paths_ops, paths_ops)]), 'FR Error: data.paths_ops != paths_ops'
    assert data.um_per_pixel == params['importing']['um_per_pixel'], 'FR Error: data.um_per_pixel != params[importing][um_per_pixel]'
    assert data.n_sessions == len(paths_stat), 'FR Error: data.n_sessions != len(paths_stat)'
    ## Types
    assert all([c.dtype == np.int64 for c in data.centroids]), 'FR Error: data.centroids.dtype != np.uint64'
    assert all([im.dtype == np.float32 for im in data.FOV_images]), 'FR Error: data.FOV_images.dtype != np.float32'
    assert isinstance(data.FOV_width, int), 'FR Error: data.FOV_width.dtype != int'
    assert isinstance(data.FOV_height, int), 'FR Error: data.FOV_height.dtype != int'
    assert isinstance(data.n_sessions, int), 'FR Error: data.n_sessions.dtype != int'
    assert isinstance(data.n_roi_total, int), 'FR Error: data.n_roi_total.dtype != int'
    assert isinstance(data.n_roi, list), 'FR Error: data.n_roi.dtype != list'
    assert all([isinstance(n, int) for n in data.n_roi]), 'FR Error: data.n_roi.dtype != list of ints'
    assert isinstance(data.shifts, (list, tuple)), 'FR Error: data.shifts.dtype != list or tuple'
    assert all([isinstance(s, np.ndarray) for s in data.shifts]), 'FR Error: data.shifts.dtype != list or tuple of np.ndarrays'
    assert all([s.dtype == np.uint64 for s in data.shifts]), 'FR Error: data.shifts.dtype != list or tuple of np.ndarrays of dtype np.uint64'
    assert isinstance(data.um_per_pixel, float), 'FR Error: data.um_per_pixel.dtype != float'
    assert isinstance(data.paths_stat, list), 'FR Error: data.paths_stat.dtype != list'
    assert all([isinstance(p, str) for p in data.paths_stat]), 'FR Error: data.paths_stat.dtype != list of strings'
    assert isinstance(data.paths_ops, list), 'FR Error: data.paths_ops.dtype != list'
    assert all([isinstance(p, str) for p in data.paths_ops]), 'FR Error: data.paths_ops.dtype != list of strings'
    assert isinstance(data.ROI_images, list), 'FR Error: data.ROI_images.dtype != list'
    assert len(data.ROI_images) == len(paths_stat), 'FR Error: len(data.ROI_images) != len(paths_stat)'
    assert all([isinstance(im, np.ndarray) for im in data.ROI_images]), 'FR Error: data.ROI_images.dtype != list of np.ndarrays'
    assert all([im.dtype == np.float32 for im in data.ROI_images]), 'FR Error: data.ROI_images.dtype != list of np.ndarrays of dtype np.float32'
    assert isinstance(data.spatialFootprints, list), 'FR Error: data.spatialFootprints.dtype != list'
    assert len(data.spatialFootprints) == len(paths_stat), 'FR Error: len(data.spatialFootprints) != len(paths_stat)'
    assert all([isinstance(sf, scipy.sparse.csr_matrix) for sf in data.spatialFootprints]), 'FR Error: data.spatialFootprints.dtype != list of scipy.sparse.csr_matrix'
            
    ### Attributes specific to this dataset
    assert data.n_roi_total == 300*len(paths_stat), 'FR Error: data.n_roi_total != 300*len(paths_stat). stat.npy files expected to contain 300 ROIs each.'
    assert data.n_roi == [300]*len(paths_stat), 'FR Error: data.n_roi != [300]*len(paths_stat). stat.npy files expected to contain 300 ROIs each.'
    assert all([c.shape == (300, 2) for c in data.centroids]), 'FR Error: data.centroids.shape != (300, 2)'
    assert array_hasher(data.centroids[0]) == 'f98974f9430846ed', 'FR Error: data.centroids[0] != expected values. See code for expected values.'
    assert array_hasher(data.centroids[13]) == 'b073952b11a3c507', 'FR Error: data.centroids[13] != expected values. See code for expected values.'
    assert data.FOV_height == 512, 'FR Error: data_FOV_height != expected value.'
    assert data.FOV_width == 705, 'FR Error: data_FOV_width != expected value.'
    assert array_hasher(data.FOV_images[0]) == '2e335e2116ee4cfc', 'FR Error: data.FOV_images[0] != expected values. See code for expected values.'
    assert array_hasher(data.FOV_images[13]) == '597cc830474f1ff5', 'FR Error: data.FOV_images[13] != expected values. See code for expected values.'
    assert data.ROI_images[0].shape == tuple([300] + list(params['importing']['out_height_width'])), 'FR Error: data.ROI_images.shape != (300, out_height_width[0], out_height_width[1])'
    assert array_hasher(data.ROI_images[0]) == '04d986f3681778f0', 'FR Error: data.ROI_images[0] != expected values. See code for expected values.'
    assert array_hasher(data.ROI_images[13]) == 'de5a2c2c8c34c43e', 'FR Error: data.ROI_images[13] != expected values. See code for expected values.'
    assert data.spatialFootprints[0].shape[0] == 300, 'FR Error: data.spatialFootprints.shape[0] != 300'
    assert data.spatialFootprints[0].shape[1] == 512*705, 'FR Error: data.spatialFootprints.shape[1] != 512*705'
    assert array_hasher(data.spatialFootprints[0].toarray()) == '6319b48421caeb23', 'FR Error: data.spatialFootprints[0] != expected values. See code for expected values.'
    assert array_hasher(data.spatialFootprints[13].toarray()) == 'd5495d254954d56c', 'FR Error: data.spatialFootprints[13] != expected values. See code for expected values.'
