## setup.py file for roicat

from distutils.core import setup

from pathlib import Path
import os

import sys
import os
environmentVariables_names = ['INSTALL_ROICAT_GPU']
environmentVariables = {key: os.environ.get(key, None) for key in environmentVariables_names}

print(f"Installing with environment variables: {environmentVariables}")
print(f"\n")
print("To change, on Windows use: set 'ENVIRONMENT_VARIABLE_NAME=ENVIRONMENT_VARIABLE_VALUE'. \n On Linux use: ENVIRONMENT_VARIABLE_NAME=ENVIRONMENT_VARIABLE_VALUE. \n")

use_gpu = ['True', 'TRUE', 'true', '1', 'yes', 'YES', 'Yes', 'y', 'Y'].count(environmentVariables['INSTALL_ROICAT_GPU']) > 0

print(f"sys.argv: {sys.argv}")

## Manually install dependencies from requirements.txt
### This is a workaround for the fact that pip does not support
### installing from a requirements.txt file with custom URLs
### (e.g. torch --extra-index-url https://download.pytorch.org/whl/torch_stable.html)
### 1. Find path to requirements.txt
### 2. Call pip install -r requirements.txt
# path_reqs = Path(__file__).parent / 'requirements_GPU.txt' if use_gpu else Path(__file__).parent / 'requirements_CPU_only.txt'
# print(f'use_gpu: {use_gpu}, installing requirements from: {path_reqs}')
# path_reqs = Path(__file__).parent / 'requirements.txt'

# assert path_reqs.exists(), 'No requirements.txt file found!'
# os.system(f'pip install -r {path_reqs}')


pip_deps = []
with open('requirements.txt', 'r') as f:
    for line in f.readlines():
        pip_deps.append(line.strip())

extras_torchCPU = [
    'torch==1.12.1',
    'torchvision==0.13.1',
    'torchaudio==0.12.1',
]

path_README = str(Path(__file__).parent / 'README.md')

setup(
    name='roicat',
    version='0.1.0',
    author='Richard Hakim',
    keywords=['neuroscience', 'neuroimaging', 'machine learning', 'deep learning'],
    license='LICENSE',
    description='A library for classifying and tracking ROIs.',
    long_description=open(path_README).read(),
    url='https://github.com/RichieHakim/ROICaT',

    packages=[
        'roicat',
        'roicat.tracking',
        'roicat.classification'
    ],
    
    install_requires=pip_deps,
    extras_require={
        'torchCPU': extras_torchCPU,
    },
)