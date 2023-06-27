import warnings


def test_internal_package_tests():
    """
    Test packages.
    RH 2023
    """
    from pathlib import Path

    import pytest

    ## List of packages to test
    packages = [
        'hdbscan',
        'umap',
        'optuna',
        'cv2',
        'skl2onnx',
        'holoviews',
        'bokeh',
        'sparse',
    ]

    ## Iterate over each package
    for pkg_s in packages:
        ## Try to import the package
        try:
            exec(f'import {pkg_s}')
            print(f'RH: Successfully imported {pkg_s}')
        except ImportError:
            warnings.warn(f'RH: Could not import {pkg_s}.')
            continue

        ## Get a handle on the package
        pkg_h = eval(pkg_s)

        ## Get the path to the package
        path_pkg = str(Path(pkg_h.__file__).parent)

        ## Run the tests
        pytest.main([path_pkg, '-v'])

def test_importing_packages():
    """
    Runs pytest on the core packages.
    """
    corePackages = [
        'hdbscan',
        'holoviews',
        'jupyter',
        'kymatio',
        'matplotlib',
        'natsort',
        'numpy',
        'cv2',
        'optuna',
        'PIL',
        'pytest',
        'sklearn',
        'scipy',
        'seaborn',
        'sparse',
        'tqdm',
        'umap',
        'xxhash',
        'bokeh',
        'psutil',
        'cpuinfo',
        'GPUtil',
        'yaml',
        'mat73',
        'torch',
        'torchvision',
        'torchaudio',
        'skl2onnx',
        'onnx',
        'onnxruntime',
    ]

    for pkg in corePackages:
        try:
            exec(f'import {pkg}')
        except ModuleNotFoundError:
            warnings.warn(f'RH Warning: {pkg} not found. Skipping tests.')

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
        print(f'RH: CUDA is available. Environment using PyTorch version: {version}') if verbose > 1 else None
        arr = torch.rand(1000, 10).to(device)
        arr2 = torch.rand(10, 1000).to(device)
        arr3 = (arr @ arr2).mean().cpu().numpy()
        print(f'RH: Torch can do basic operations on GPU. Environment using PyTorch version: {version}. Result of operations: {arr3}') if verbose > 1 else None

    else:
        warnings.warn(f'RH Warning: CUDA is not available. Environment using PyTorch version: {version}')
        
    ## Test CPU computations
    arr = torch.rand(1000, 10)
    arr2 = torch.rand(10, 1000)
    arr3 = (arr @ arr2).mean().numpy()
    print(f'RH: Torch can do basic operations on CPU. Environment using PyTorch version: {version}. Result of operations: {arr3}') if verbose > 1 else None

def test_numpy():
    import numpy as np
    ## test numpy
    np.random.seed(0)
    arr1 = np.random.rand(1000, 10)
    arr2 = np.random.rand(10, 1000)
    arr3 = (arr1 @ arr2).mean()
    assert np.allclose(arr3, 2.5, rtol=0.1), 'RH Error: numpy test failed.'

def test_scipy_sparse():
    import numpy as np
    import scipy.sparse
    ## test scipy.sparse
    arr1 = scipy.sparse.rand(1000, 10, density=0.1).tocoo().tocsr()
    arr2 = scipy.sparse.rand(10, 1000, density=0.1).tocoo().tocsr()
    arr3 = (arr1 @ arr2).mean()
    assert np.allclose(arr3, 0.025, rtol=0.2), 'RH Error: scipy.sparse test failed.'
