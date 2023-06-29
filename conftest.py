from pathlib import Path
import tempfile

import numpy as np
from typing import Union, Optional, List, Tuple

import pytest

import roicat

"""
WARNING: DO NOT REQUIRE ANY DEPENDENCIES FROM ANY NON-STANDARD LIBRARY
 MODULES IN THIS FILE. It is intended to be run before any other
 modules are imported.
"""


@pytest.fixture(scope='session')
def dir_data_test():
    """
    Prepares the directory containing the test data.
    Steps:
        1. Determine the path to the data directory.
        2. Create the data directory if it does not exist.
        3. Download the test data if it does not exist.
            If the data exists, check its hash.
        4. Extract the test data.
        5. Return the path to the data directory.
    """
    # dir_data_test = str(Path('data_test/').resolve().absolute())
    dir_data_test = str((Path(tempfile.gettempdir()) / 'data_test').resolve().absolute())
    print(dir_data_test)
    path_data_test_zip = download_data_test_zip(dir_data_test)
    roicat.helpers.extract_zip(
        path_zip=path_data_test_zip, 
        path_extract=dir_data_test,
        verbose=True,
    )
    return dir_data_test

def download_data_test_zip(directory):
    """
    Downloads the test data if it does not exist.
    If the data exists, check its hash.
    """
    path_save = str(Path(directory) / 'data_test.zip')
    roicat.helpers.download_file(
        url=r'https://github.com/RichieHakim/ROICaT/raw/dev/tests/data_test.zip', 
        path_save=path_save, 
        check_local_first=True, 
        check_hash=True, 
        hash_type='MD5', 
        hash_hex=r'd7662fcbaa44b4d0ebcf86bbdc6daa66',
        mkdir=True,
        allow_overwrite=True,
        write_mode='wb',
        verbose=True,
        chunk_size=1024,
    )
    return path_save

@pytest.fixture(scope='session')
def array_hasher():
    """
    Returns a function that hashes an array.
    """
    from functools import partial
    import xxhash
    return partial(xxhash.xxh64_hexdigest, seed=0)

@pytest.fixture(scope='session')
def make_ROIs(
    n_sessions=10,
    max_rois_per_session=100,
    size_im=(36,36)
    ):
    import numpy as np
    import torch
    import torchvision

    roi_prototype = torch.zeros(size_im, dtype=torch.uint8)
    roi_prototype[*torch.meshgrid(torch.arange(size_im[0]//2-8, size_im[0]//2+8), torch.arange(size_im[1]//2-8, size_im[1]//2+8), indexing='xy')] = 255
    transforms = torch.nn.Sequential(*[
        torchvision.transforms.RandomPerspective(distortion_scale=0.9, p=1.0),
        torchvision.transforms.RandomAffine(0, scale=(2.0, 2.0))
    ])
    ROIs = [[transforms(torch.as_tensor(roi_prototype[None,:,:]))[0].numpy() for i_roi in range(max_rois_per_session)] for i_sesh in range(n_sessions)]
    ROIs = [np.stack([roi for roi in ROIs_sesh if roi.sum() > 0], axis=0) for ROIs_sesh in ROIs]

    return ROIs


@pytest.fixture(scope='session')
def check_items():
    """
    Returns a function that checks if items in the test object match the corresponding items in the true object.
    """
    def check_items_fn(
        test: Union[dict, list, tuple, set, np.ndarray, int, float, complex, str, bool, None], 
        true: Union[dict, list, tuple, set, np.ndarray, int, float, complex, str, bool, None], 
        path: Optional[List[str]] = None,
        kwargs_allclose: Optional[dict] = {'rtol': 1e-7, 'equal_nan': True},
    ) -> None:
        """
        Checks if items in the test object match the corresponding items in the true object.
        RH 2023

        Args:
            test (Union[dict, list, tuple, set, np.ndarray, int, float, complex, str, bool, None]):
                The object to check against the true object.
            true (Union[dict, list, tuple, set, np.ndarray, int, float, complex, str, bool, None]):
                The object that contains the true values to compare to the test object.
            path (Optional[List[str]]): 
                List of strings that keeps track of the nested path during recursion. 
                This is used for error reporting to show where a mismatch occurred. 
                (Default is ``None``)

        Raises:
            KeyError: If a key in a dictionary object from true is not found in the test object.
            ValueError: If there is a mismatch between the test and true objects. 
                        The error message contains the path where the mismatch occurred.                    
        """
        if path is None:
            path = []
        
        if len(path) > 0:
            if path[-1].startswith('_'):
                return None
        print(f"Checking item: {path}")
        ## DICT
        if isinstance(true, dict):
            for key in true:
                if key not in test:
                    raise KeyError(f"Key {key} not found in test at path {path}")
                check_items_fn(test[key], true[key], path=path + [str(key)], kwargs_allclose=kwargs_allclose)
        ## ITERATABLE
        elif isinstance(true, (list, tuple, set)):
            if len(true) != len(test):
                raise ValueError(f"Length mismatch at path {path}")
            for idx, (i, j) in enumerate(zip(test, true)):
                check_items_fn(i, j, path=path + [str(idx)], kwargs_allclose=kwargs_allclose)
        ## NP.NDARRAY
        elif isinstance(true, np.ndarray):
            try:
                np.testing.assert_allclose(test, true)
            except AssertionError as e:
                raise ValueError(f"Value mismatch at path {path}: {e}")
        ## NP.SCALAR
        elif np.isscalar(true):
            if isinstance(test, (int, float, complex, np.number)):
                try:
                    np.testing.assert_allclose(np.array(test), np.array(true))
                except AssertionError as e:
                    raise ValueError(f"Numeric value mismatch at path {path}: {e}")
            else:
                if not test == true:
                    raise ValueError(f"Value mismatch at path {path}")
        ## STRING
        elif isinstance(true, str):
            if not test == true:
                raise ValueError(f"String value mismatch at path {path}")
        ## NUMBER
        elif isinstance(true, (int, float, complex)):
            try:
                np.testing.assert_allclose(np.array(test), np.array(true))
            except AssertionError as e:
                raise ValueError(f"Numeric value mismatch at path {path}: {e}")
        ## BOOL
        elif isinstance(true, bool):
            if test != true:
                raise ValueError(f"Boolean value mismatch at path {path}")
        ## NONE
        elif true is None:
            if test is not None:
                raise ValueError(f"None value mismatch at path {path}")
                
    return check_items_fn
