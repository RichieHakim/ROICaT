## setup.py file for roicat

from distutils.core import setup

from pathlib import Path
import os

import sys
import os

## Dependencies: core requirements
deps_core = [
    "umap-learn==0.5.3",
    "hdbscan==0.8.29",
    "gdown==4.5.1",
    "ipywidgets==7.7.1",
    "kymatio==0.2.1",
    "matplotlib==3.5.2",
    "numpy==1.23.2",
    "opencv_contrib_python==4.6.0.66",
    "pandas==1.4.3",
    "Pillow==9.2.0",
    "scikit_learn==1.1.2",
    "scipy==1.8.1",
    "seaborn==0.11.2",
    "sparse==0.13.0",
    "tqdm==4.64.0",
    "natsort==8.2.0",
    "jupyter",
    "paramiko==2.12.0",
    "pyyaml==5.4.1",
    "hdfdict==0.3.1",
    "optuna==3.0.1",
    "joblib==1.1.0",
    "Cython==0.29.32",
    "einops==0.6.0",
    "xxhash==3.1.0",
    "pytest==7.2.0",
]
## Dependencies: latest versions of core requirements
### remove everything starting and after the first =,>,<,! sign
deps_core_latest = [req.split('=')[0].split('>')[0].split('<')[0].split('!')[0] for req in deps_core]


# ## Dependencies: torch CPU
# deps_torchCPU = [
#     "torch==1.12.1",
#     "torchvision==0.13.1",
#     "torchaudio==0.12.1",
# ]
# ## Dependencies: torch CPU, latest versions
# deps_torchCPU_latest = [req.split('=')[0].split('>')[0].split('<')[0].split('!')[0] for req in deps_torchCPU]


## Get README.md
with open("README.md", "r") as f:
    readme = f.read()

setup(
    name='roicat',
    version='0.1.0',
    author='Richard Hakim',
    keywords=['neuroscience', 'neuroimaging', 'machine learning', 'deep learning'],
    license='LICENSE',
    description='A library for classifying and tracking ROIs.',
    long_description=readme,
    url='https://github.com/RichieHakim/ROICaT',

    packages=[
        'roicat',
        'roicat.tracking',
        'roicat.classification'
    ],
    
    install_requires=[],
    extras_require={
        'core': deps_core,
        'core_latest': deps_core_latest,
        # 'torchCPU': deps_torchCPU,
        # 'torchCPU_latest': deps_torchCPU_latest,
    },
)