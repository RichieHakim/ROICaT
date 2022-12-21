## setup.py file for roicat

from distutils.core import setup

from pathlib import Path
import os

## Manually install dependencies from requirements.txt
### This is a workaround for the fact that pip does not support
### installing from a requirements.txt file with custom URLs
### (e.g. torch --extra-index-url https://download.pytorch.org/whl/torch_stable.html)
### 1. Find path to requirements.txt
### 2. Call pip install -r requirements.txt
path_reqs = Path(__file__) / 'requirements_GPU.txt'
assert path_reqs.exists(), 'No requirements.txt file found!'
os.system(f'pip install -r {path_reqs}')


# pip_deps = []
# with open('requirements_GPU.txt', 'r') as f:
#     for line in f.readlines():
#         pip_deps.append(line.strip())

setup(
    name='roicat',
    version='0.1.0',
    author='Richard Hakim',
    keywords=['neuroscience', 'neuroimaging', 'machine learning', 'deep learning'],
    packages=[
        'roicat',
        'roicat.tracking',
        'roicat.classification'
    ],
    license='LICENSE',
    description='A library for classifying and tracking ROIs.',
    long_description=open('README.md').read(),
    # install_requires=pip_deps,
    url='https://github.com/RichieHakim/ROICaT',
)