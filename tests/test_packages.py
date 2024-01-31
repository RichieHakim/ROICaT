import warnings


PACKAGES = [
    'GPUtil',
    'PIL',
    'cpuinfo',
    'cv2',
    'jupyter',
    'mat73',
    'matplotlib',
    'natsort',
    'numpy',
    'onnx',
    'onnxruntime',
    'optuna',
    'psutil',
    'pytest',
    'scipy',
    'seaborn',
    'skl2onnx',
    'sklearn',
    'sparse',
    'torch',
    'torchaudio',
    'torchvision',
    'tqdm',
    'xxhash',
    'yaml',
    'bokeh',
    'holoviews',
    'jupyter_bokeh',
    'umap',
    'hdbscan',
    'kymatio',
]

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
            warnings.warn(f'RH: Could not import {pkg_s}. Skipping tests.')
            continue

        else:
            try:
                ## Get a handle on the package
                pkg_h = eval(pkg_s)

                ## Get the path to the package
                path_pkg = str(Path(pkg_h.__file__).parent)

                ## Run the tests
                pytest.main([path_pkg, '-v'])
            except Exception as e:
                warnings.warn(f'RH: Could not run tests for {pkg_s}. Error: {e}')
                continue


def test_importing_packages():
    """
    Runs pytest on the core packages.
    """

    for pkg in PACKAGES:
        try:
            exec(f'import {pkg}')
        except ModuleNotFoundError:
            warnings.warn(f'RH Warning: {pkg} not found.')

def test_torch(
    device='cpu', 
    verbose=2
):
    """
    Test to see if torch can do operations on device.
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
        
    ## Test CPU computations
    arr = torch.rand(1000, 10, device=device)
    arr2 = torch.rand(10, 1000, device=device)
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
