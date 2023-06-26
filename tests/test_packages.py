def test_packages():
    """
    Test packages.
    RH 2023
    """
    from pathlib import Path
    import warnings

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