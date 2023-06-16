# ROICaT
<div>
    <img src="media/logo1.png" alt="ROICaT" width="200"  align="right"  style="margin-left: 20px"/>
</div>

[![build](https://github.com/RichieHakim/ROICaT/actions/workflows/.github/workflows/build.yml/badge.svg)](https://github.com/RichieHakim/ROICaT/actions/workflows/build.yml) 
[![PyPI version](https://badge.fury.io/py/roicat.svg)](https://badge.fury.io/py/roicat)
[![Downloads](https://pepy.tech/badge/roicat)](https://pepy.tech/project/roicat)

- **Documentation: [https://roicat.readthedocs.io/en/latest/]()**
- Discussion forum: [https://groups.google.com/g/roicat_support]()
- Technical support: [Github Issues](https://github.com/RichieHakim/ROICaT/issues)

## **R**egion **O**f **I**nterest **C**lassification **a**nd **T**racking
A simple-to-use Python package for automatically classifying images of cells and tracking them across imaging sessions/planes.
<div>
    <img src="media/tracking_FOV_clusters_rich.gif" alt="tracking_FOV_clusters_rich"  width="400"  align="right" style="margin-bottom: 0px"/>
</div>

With this package, you can:
- **Classify ROIs** into different categories (e.g. neurons, dendrites, glia, etc.).
- **Track ROIs** across imaging sessions/planes.

We have found that ROICaT is capable of classifying cells with accuracy comparable to human relabeling performance, and tracking cells with higher accuracy than any other methods we have tried. Paper coming soon.

<br>
<br>

# How to use ROICaT
<div>
    <img src="media/umap_with_labels.png" alt="ROICaT" width="300"  align="right"  style="margin-left: 20px"/>
</div>

**TRACKING:** 
- [Interactive notebook](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/jupyter/tracking/tracking_interactive_notebook.ipynb)
- [Google CoLab](https://githubtocolab.com/RichieHakim/ROICaT/blob/main/notebooks/colab/tracking/tracking_interactive_notebook.ipynb)
- (TODO) [demo script](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/jupyter/tracking/tracking_scripted_notebook.ipynb)
  
**CLASSIFICATION:**
- [Interactive notebook - Drawing](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/jupyter/classification/classify_by_drawingSelection.ipynb)
- [Google CoLab - Drawing](https://githubtocolab.com/RichieHakim/ROICaT/blob/main/notebooks/colab/classification/classify_by_drawingSelection_colab.ipynb)
- (TODO) [Interactive notebook - Labeling]()
- (TODO) [Interactive notebook - Train classifier]()
- (TODO) [Interactive notebook - Inference with classifier]()

**OTHER:** 
- [Custom data importing notebook](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/jupyter/other/demo_custom_data_importing.ipynb)
- Train a new ROInet model using the provided Jupyter Notebook [TODO: link].
- Use the API to integrate ROICaT functions into your own code: [Documentation](https://roicat.readthedocs.io/en/latest/).

# Installation
ROICaT works on Windows, MacOS, and Linux. If you have any issues during the installation process, please make a [github issue](https://github.com/RichieHakim/ROICaT/issues) with the error.

### 0. Requirements
- Segmented data. For example Suite2p output data (stat.npy and ops.npy files), CaImAn output data (results.h5 files), or any other type of data using this [custom data importing notebook](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/jupyter/other/demo_custom_data_importing.ipynb).
- [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- If using Windows: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- If using linux/unix: GCC >= 5.4.0, ideally == 9.2.0. Google how to do this on your operating system. Check with: `gcc --version`.
- **Optional:** [CUDA compatible NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) and [drivers](https://developer.nvidia.com/cuda-toolkit-archive). Using a GPU can increase ROICaT speeds ~5-50x, though without it, ROICaT will still run reasonably quick. GPU support is not available for Macs.

### 1. (Recommended) Create a new conda environment
The below commands should be run in the terminal (Mac/Linux) or Anaconda Prompt (Windows).
```
conda create -n roicat python=3.11
conda activate roicat
pip install --upgrade pip
```

### 2. Install ROICaT
```
pip install --user roicat[all]
```
Note: if you are using a zsh terminal, change command to: `pip3 install --user 'roicat[all]'`

### 3. Clone the repo to get the scripts and notebooks
```
git clone https://github.com/RichieHakim/ROICaT
```

# Troubleshooting Installation
### Troubleshooting (Windows)
If you receive the error: `ERROR: Could not build wheels for hdbscan, which is required to install pyproject.toml-based projects` on Windows, make sure that you have installed Microsoft C++ Build Tools. If not, download from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and run the commands:
```
cd path/to/vs_buildtools.exe
vs_buildtools.exe --norestart --passive --downloadThenInstall --includeRecommended --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Workload.MSBuildTools
```
Then, try proceeding with the installation by rerunning the pip install commands above.
([reference](https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst))

### Troubleshooting (GPU support)
GPU support is not required. Windows users will often need to manually install a CUDA version of pytorch (see below). Note that you can check your nvidia driver version using the shell command: `nvidia-smi` if you have drivers installed. 

Use the following command to check your PyTorch version and if it is GPU enabled:
```
python -c "import torch, torchvision; print(f'Using versions: torch=={torch.__version__}, torchvision=={torchvision.__version__}');  print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')"
```
**Outcome 1:** Output expected if GPU is enabled:
```
Using versions: torch==X.X.X+cuXXX, torchvision==X.X.X+cuXXX
torch.cuda.is_available() = True
```
This is the ideal outcome. You are using a <u>CUDA</u> version of PyTorch and your GPU is enabled.

**Outcome 2:** Output expected if <u>non-CUDA</u> version of PyTorch is installed:
```
Using versions: torch==X.X.X, torchvision==X.X.X
OR
Using versions: torch==X.X.X+cpu, torchvision==X.X.X+cpu
torch.cuda.is_available() = False
```
If a <u>non-CUDA</u> version of PyTorch is installed, please follow the instructions here: https://pytorch.org/get-started/locally/ to install a CUDA version. If you are using a GPU, make sure you have a [CUDA compatible NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) and [drivers](https://developer.nvidia.com/cuda-toolkit-archive) that match the same version as the PyTorch CUDA version you choose. All CUDA 11.x versions are intercompatible, so if you have CUDA 11.8 drivers, you can install `torch==2.0.1+cu117`.

**Outcome 3:** Output expected if GPU is not available:
```
Using versions: torch==X.X.X+cuXXX, torchvision==X.X.X+cuXXX
torch.cuda.is_available() = False
```
If a CUDA version of PyTorch is installed but GPU is not available, make sure you have a [CUDA compatible NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) and [drivers](https://developer.nvidia.com/cuda-toolkit-archive) that match the same version as the PyTorch CUDA version you choose. All CUDA 11.x versions are intercompatible, so if you have CUDA 11.8 drivers, you can install `torch==2.0.1+cu117`.

# General workflow:
- **Pass ROIs through ROInet:** Images of the ROIs are passed through a neural network and outputs a feature vector for each image describing what the ROI looks like.
-  **Classification:** The feature vectors can then be used to classify ROIs:
- A simple classifier can be trained using user supplied labeled data (e.g. an array of images of ROIs and a corresponding array of labels for each ROI).
- Alternatively, classification can be done by projecting the feature vectors into a lower-dimensional space using UMAP and then simply circling the region of space to classify the ROIs.
-  **Tracking**: The feature vectors can be combined with information about the position of the ROIs to track the ROIs across imaging sessions/planes.

# TODO:
- Unify model training into this repo
- Improve classification notebooks, port to colab, make scripts
- Integration tests
- make better reference API
