from pathlib import Path

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
    dir_data_test = str(Path('data_test/').resolve().absolute())
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
        url=r'https://github.com/RichieHakim/ROICaT/raw/main/tests/data_test.zip', 
        path_save=path_save, 
        check_local_first=True, 
        check_hash=True, 
        hash_type='MD5', 
        hash_hex=r'764d9b3fc481e078d1ef59373695ecce',
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