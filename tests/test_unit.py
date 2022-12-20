"""
Unit tests for ROICaT

The functions in this module are intended to be 
 found and run by pytest.

To run the tests, use the command (in a terminal):
    pytest -v test_unit.py
            ^
          verbose
"""

import warnings
import pytest

# def test_download_from_url():
#     """
#     Test download_from_url function.
#     RH 2021
#     """
#     from roicat.utils import download_from_url

#     ## Test 1: download from url
#     url = 'https://raw.githubusercontent.com/roicat/roicat/master/roicat/utils.py'
#     path_save = 'utils.py'
#     download_from_url(url, path_save)
#     assert Path(path_save).exists()
#     Path(path