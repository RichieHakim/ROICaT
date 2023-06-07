## setup.py file for roicat
from pathlib import Path

from distutils.core import setup
import copy

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

dir_parent = Path(__file__).parent

def read_requirements():
    with open(str(dir_parent / "requirements.txt"), "r") as req:
        content = req.read()  ## read the file
        requirements = content.split("\n") ## make a list of requirements split by (\n) which is the new line character

    ## Filter out any empty strings from the list
    requirements = [req for req in requirements if req]
    ## Filter out any lines starting with #
    requirements = [req for req in requirements if not req.startswith("#")]
    ## Remove any commas, quotation marks, and spaces from each requirement
    requirements = [req.replace(",", "").replace("\"", "").replace("\'", "").strip() for req in requirements]

    return requirements

deps_all = read_requirements()

## Dependencies: latest versions of requirements
### remove everything starting and after the first =,>,<,! sign
deps_names = [req.split('=')[0].split('>')[0].split('<')[0].split('!')[0] for req in deps_all]
deps_all_dict = dict(zip(deps_names, deps_all))

deps_all_latest = copy.deepcopy(deps_names)

## Make different versions of dependencies
### Also pull out the version number from the requirements (specified in deps_all_dict values).
deps_core = [deps_all_dict[dep] for dep in [
    'einops',
    'jupyter',
    'matplotlib',
    'natsort',
    'numpy',
    'paramiko',
    'Pillow',
    'pytest',
    'scikit_learn',
    'scipy',
    'seaborn',
    'sparse',
    'tqdm',
    'xxhash',
    'torch',
    'torchvision',
    'torchaudio',
]]

deps_classification = [deps_all_dict[dep] for dep in [
    'opencv_contrib_python',
    'umap-learn',
]] + deps_core

deps_tracking = [deps_all_dict[dep] for dep in [
    'opencv_contrib_python',
    'hdbscan',
    'kymatio',
    'optuna',
]] + deps_core

print({
    'deps_all': deps_all,
    'deps_all_latest': deps_all_latest,
    'deps_core': deps_core,
    'deps_classification': deps_classification,
    'deps_tracking': deps_tracking,
})

## Get README.md
with open(str(dir_parent / "README.md"), "r") as f:
    readme = f.read()

## Get version number
with open(str(dir_parent / "roicat" / "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().replace("\"", "").replace("\'", "")
            break

setup(
    name='roicat',
    version=version,
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
        'all': deps_all,
        'all_latest': deps_all_latest,
        'core': deps_core,
        'classification': deps_classification,
        'tracking': deps_tracking,
    },
)