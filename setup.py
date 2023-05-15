## setup.py file for roicat

from distutils.core import setup

## Dependencies: core requirements
# deps_core = [
#     "einops==0.6.1",
#     "hdbscan==0.8.29",
#     "jupyter==1.0.0",
#     "kymatio==0.3.0",
#     "matplotlib==3.7.1",
#     "natsort==8.3.1",
#     "numpy==1.24.3",
#     "opencv_contrib_python==4.7.0.72",
#     "optuna==3.1.1",
#     "paramiko==3.1.0",
#     "Pillow==9.5.0",
#     "pytest==7.3.1",
#     "scikit_learn==1.2.2",
#     "scipy==1.10.1",
#     "seaborn==0.12.2",
#     "sparse==0.14.0",
#     "tqdm==4.65.0",
#     "umap-learn==0.5.3",
#     "xxhash==3.2.0",

#     "torch==2.0.1",
#     "torchvision==0.15.2",
#     "torchaudio==2.0.2",
# ]

def read_requirements():
    with open('requirements.txt', 'r') as req:
        content = req.read()  ## read the file
        requirements = content.split("\n") ## make a list of requirements split by (\n) which is the new line character

    ## Filter out any empty strings from the list
    requirements = [req for req in requirements if req]
    ## Filter out any lines starting with #
    requirements = [req for req in requirements if not req.startswith("#")]
    ## Remove any commas, quotation marks, and spaces from each requirement
    requirements = [req.replace(",", "").replace("\"", "").replace("\'", "").strip() for req in requirements]

    return requirements

deps_core = read_requirements()

## Dependencies: latest versions of core requirements
### remove everything starting and after the first =,>,<,! sign
deps_core_latest = [req.split('=')[0].split('>')[0].split('<')[0].split('!')[0] for req in deps_core]

print({
        'core': deps_core,
        'core_latest': deps_core_latest,
    })

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
        'roicat.classification',
        'roicat.model_training',
    ],
    
    install_requires=[],
    extras_require={
        'core': deps_core,
        'core_latest': deps_core_latest,
    },
)