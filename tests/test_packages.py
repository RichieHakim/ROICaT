import warnings


PACKAGES = [
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
    import importlib
    from pathlib import Path

    import pytest

    ## List of packages to test
    packages = [
        'hdbscan',
        'umap',
        'optuna',
        'cv2',
        'skl2onnx',
        # 'holoviews',
        'bokeh',
        'sparse',
    ]

    ## Iterate over each package
    for pkg_s in packages:
        ## Try to import the package
        try:
            pkg_h = importlib.import_module(pkg_s)
            print(f'RH: Successfully imported {pkg_s}')
        except ImportError:
            warnings.warn(f'RH: Could not import {pkg_s}. Skipping tests.')
            continue

        try:
            ## Get the path to the package
            path_pkg = str(Path(pkg_h.__file__).parent)

            ## Run the tests
            pytest.main([path_pkg, '-v'])
        except Exception as e:
            warnings.warn(f'RH: Could not run tests for {pkg_s}. Error: {e}')
            continue


def test_importing_packages():
    """
    Test that all required packages can be imported.
    """
    import importlib
    failed = []
    for pkg in PACKAGES:
        try:
            importlib.import_module(pkg)
        except ModuleNotFoundError:
            failed.append(pkg)
    if failed:
        warnings.warn(f'Packages not found: {failed}')
    ## At minimum, core packages must be importable
    core = ['numpy', 'scipy', 'torch', 'torchvision', 'sklearn']
    missing_core = [p for p in core if p in failed]
    assert not missing_core, f'Core packages missing: {missing_core}'


def test_torch(device='cpu'):
    """
    Test that torch can do basic operations on the given device.
    """
    import torch
    torch.manual_seed(0)
    arr = torch.rand(1000, 10, device=device)
    arr2 = torch.rand(10, 1000, device=device)
    arr3 = (arr @ arr2).mean().item()
    ## With fixed seed, matmul of uniform [0,1] matrices should give mean ~2.5
    assert 1.5 < arr3 < 3.5, f'Torch basic operation gave unexpected result: {arr3}'


def test_numpy():
    """
    Test that numpy can do basic operations deterministically.
    """
    import numpy as np
    rng = np.random.default_rng(seed=0)
    arr1 = rng.random((1000, 10))
    arr2 = rng.random((10, 1000))
    arr3 = (arr1 @ arr2).mean()
    assert np.allclose(arr3, 2.5, rtol=0.1), f'numpy test failed: expected ~2.5, got {arr3}'


def test_scipy_sparse():
    """
    Test that scipy.sparse can do basic operations deterministically.
    """
    import numpy as np
    import scipy.sparse
    rng = np.random.default_rng(seed=0)
    arr1 = scipy.sparse.random_array((1000, 10), density=0.1, rng=rng, format='csr')
    arr2 = scipy.sparse.random_array((10, 1000), density=0.1, rng=rng, format='csr')
    arr3 = (arr1 @ arr2).mean()
    assert np.allclose(arr3, 0.025, rtol=0.3), f'scipy.sparse test failed: expected ~0.025, got {arr3}'
