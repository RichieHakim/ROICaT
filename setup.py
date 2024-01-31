## setup.py file for roicat
from pathlib import Path
import copy
import platform

from distutils.core import setup

## Get the parent directory of this file
dir_parent = Path(__file__).parent

## Get requirements from requirements.txt
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

deps_all_latest = dict(zip(deps_names, deps_names))


# Operating system specific dependencies
# OpenCV >= 4.9 is not supported on macOS < 12
system, version_macos = platform.system(), platform.mac_ver()[0]
print(f"System: {system}")
if (system == "Darwin"):
    # Safely convert version string components to integers
    version_parts = version_macos.split('.')
    version_major_macos = int(version_parts[0])

    # Check macOS version and adjust the OpenCV version accordingly
    if (version_major_macos < 12) and ('opencv_contrib_python' in deps_all_dict):
        version_opencv_macos_sub12 = "opencv_contrib_python<=4.8.1.78"
        print(f"Detected macOS version {version_major_macos}, which is < 12. Installing an older version of OpenCV: {version_opencv_macos_sub12}")
        deps_all_dict['opencv_contrib_python'] = version_opencv_macos_sub12
        deps_all_latest['opencv_contrib_python'] = version_opencv_macos_sub12
import re
## find the numbers in the string
version_opencv = '.'.join(re.findall(r'[0-9]+', deps_all_dict['opencv_contrib_python']))
if len(version_opencv) > 0:
    version_opencv = f"<={version_opencv}"


## Make different versions of dependencies
### Also pull out the version number from the requirements (specified in deps_all_dict values).
deps_core = [deps_all_dict[dep] for dep in [
    'jupyter',
    'matplotlib',
    'mat73',
    'natsort',
    'numpy',
    'optuna',
    'Pillow',
    'pytest',
    'PyYAML',
    'scikit_learn',
    'scipy',
    'seaborn',
    'sparse',
    'tqdm',
    'xxhash',
    'torch',
    'torchvision',
    'torchaudio',
    'psutil',
    'py_cpuinfo',
    'GPUtil',
    'skl2onnx',
    'onnx',
    'onnxruntime',
    'opencv_contrib_python',
]]

deps_classification = [deps_all_dict[dep] for dep in [
    'umap_learn',
    'bokeh',
    'holoviews[recommended]',
    'jupyter_bokeh',
]] + deps_core

deps_tracking = [deps_all_dict[dep] for dep in [
    'hdbscan',
    'kymatio',
]] + deps_core


## Get README.md
with open(str(dir_parent / "README.md"), "r") as f:
    readme = f.read()

## Get ROICaT version number
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
    long_description_content_type="text/markdown",
    url='https://github.com/RichieHakim/ROICaT',

    packages=[
        'roicat',
        'roicat.tracking',
        'roicat.classification',
        'roicat.model_training',
    ],
    
    install_requires=[],
    extras_require={
        'all': list(deps_all_dict.values()),
        'all_latest': list(deps_all_latest.values()),
        'core': deps_core,
        'classification': deps_classification,
        'tracking': deps_tracking,
    },
)