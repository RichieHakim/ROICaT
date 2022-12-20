from pathlib import Path

import pytest

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
    dir_data_test = str(Path('data_test/').resolve())
    print(dir_data_test)
    path_data_test_zip = download_data_test_zip(dir_data_test)
    extract_zip(
        path_zip=path_data_test_zip, 
        path_extract=dir_data_test,
        verbose=True,
    )
    return dir_data_test

def download_data_test_zip(dir_data_test):
    """
    Downloads the test data if it does not exist.
    If the data exists, check its hash.
    """
    path_save = str(Path(dir_data_test) / 'data_test.zip')
    download_file(
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

######################################################################################################################################
########################################################### HELPERS ##################################################################
######################################################################################################################################
"""
Helper functions.
These are copy and pasted from the helpers.py module so
 that they can be used in the tests without importing.

The download_file function is modified to remove all
 packages that are not part of the python standard library
 (e.g. tqdm).
"""


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
    import os
    walk = os.walk(directory, followlinks=False)
    folders = []
    files = []
    for ii,level in enumerate(walk):
        folders, files = level[1:]
        if ii==0:
            break
    return folders, files

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
        for data in response.iter_content(chunk_size):
            wrote = wrote + len(data)
            f.write(data)
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")
        return False
    # Check hash
    if check_hash:
        hash_local = hash_file(path_save, type_hash=hash_type)
        if hash_local == hash_hex:
            print('Hash of downloaded file matches hash_hex.') if verbose else None
            return True
        else:
            print('Hash of downloaded file does not match hash_hex.') if verbose else None
            print(f'Hash of downloaded file: {hash_local}') if verbose else None
            print(f'Hash provided in hash_hex: {hash_hex}') if verbose else None
            return False
    else:
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
            Path to extract zip file to.
            If None, extracts to the same directory as the zip file.
        verbose (int):
            Whether to print progress.
    """
    import zipfile
    if path_extract is None:
        path_extract = str(Path(path_zip).parent)

    print(f'Extracting {path_zip} to {path_extract}.') if verbose else None

    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(path_extract)

    print('Completed zip extraction.') if verbose else None



# if __name__ == '__main__':
#     dir_data_test()