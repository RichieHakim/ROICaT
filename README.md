# ROICaT <img src="logo.png"  width="300"  title="ROICaT"  alt="ROICaT"  align="right"  vspace = "60">

[![build](https://github.com/RichieHakim/ROICaT/actions/workflows/.github/workflows/build.yml/badge.svg)](https://github.com/RichieHakim/ROICaT/actions/workflows/build.yml) 


**R**egion **O**f **I**nterest **C**lassification **a**nd **T**racking
A simple-to-use Python package for classifying images of cells and tracking them across imaging sessions/planes.
Currently designed to be used with Suite2p output data (stat.npy and ops.npy files) and CaImAn output data (results.h5 files), but any image data can be used (see [TODO: link] for details on using non-standard data).

With this package, you can:
- **Classify cells** into different categories (e.g. neurons, glia, etc.) using a simple GUI.
- **Track cells** across imaging sessions/planes using a jupyter notebook or script.

We have found that ROICaT is capable of classifying cells with accuracy comparable to human relabeling performance, and tracking cells with higher accuracy than any other methods we have tried. Paper coming soon.

## Table of contents
- [Announcements](#Announcements)<br>
- [Installation](#Installation)<br>
- [How to use ROICaT](#HowTo)<br>
- [Frequently Asked Questions](#FAQs)<br>
- [TODO](#TODO)<br>

## Announcements
- **TRACKING:** Try it out in the demo notebook [here](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/tracking/tracking_interactive_notebook.ipynb) or the demo script [here](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/tracking/tracking_scripted_notebook.ipynb).
- **CLASSIFICATION:** still in Alpha. Contact me if you want to help test it.
- To help with development or beta test releases, please contact: rhakim@g.harvard.edu

# Installation
ROICaT works on Windows, MacOS, and Linux. If you have any issues during the installation process, please make a [github issue](https://github.com/RichieHakim/ROICaT/issues) with the error.

### 0. Requirements
- [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)<br>
- GCC >= 5.4.0, ideally == 9.2.0. Google how to do this on your operating system. For unix/linux: check with `gcc --version`.<br>
- On some Linux servers (like Harvard's O2 server), you may need to load modules instead of installing. To load conda, gcc, try: `module load conda3/latest gcc/9.2.0` or similar.<br>
- **Optional:** [CUDA compatible NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) and [drivers](https://developer.nvidia.com/cuda-toolkit-archive). Using a GPU can increase ROICaT speeds ~5-50x, though without it, ROICaT will still run reasonably quick. GPU support is not available for Macs.<br>

### 1. (Recommended) Create a new conda environment
```
conda create -n ROICaT python=3.11
conda activate ROICaT
```

### 2. Clone the repo
```
git clone https://github.com/RichieHakim/ROICaT
cd path/to/ROICaT/directory
```

### 3. Install ROICaT
Optional: `pip install --upgrade pip`<br>
```
pip install --user -v -e .[core]
```
Note: if you are using a zsh terminal, change command to: `pip3 install --user -v -e '.[core]'`

#### Troubleshooting (Windows)
If you receive the error: `ERROR: Could not build wheels for hdbscan, which is required to install pyproject.toml-based projects` on Windows, make sure that you have installed Microsoft C++ Build Tools. If not, download from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and run the commands:
```
cd path/to/vs_buildtools.exe
vs_buildtools.exe --norestart --passive --downloadThenInstall --includeRecommended --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Workload.MSBuildTools
```
Then, try proceeding with the installation by rerunning the pip install commands above.
([Source](https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst))

#### Troubleshooting (GPU support)
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


### 4. Use ROICaT<br>
- Beginner: Run a Jupyter Notebook: [Notebooks](https://github.com/RichieHakim/ROICaT/tree/main/notebooks)<br>
- Advanced: Make a parameter file and run in command line: `python -m ROICaT`. See [TODO: link to how-to] for details.<br>

# <a name="HowTo"></a>How to use ROICaT
  ***Ways to use ROICaT:***
-  **Easy:** Try out ROICaT on Google Colab: [TODO: Link]
-  **Intermediate:** Run it on your own computer. See [Installation](#Installation) for how to install.
- Using provided Jupyter Notebook(s): [Notebooks](https://github.com/RichieHakim/ROICaT/tree/main/notebooks).
- Using command line: `python -m ROICaT`. See [TODO: link to how-to] for details.
-  **Advanced:** Train a new ROInet model using the provided Jupyter Notebook [TODO: link]. Or contribute to the code base! This is a big collaborative effort, so please feel free to send a pull request or open an issue.

***General workflow:***
- **Pass ROIs through ROInet:** Images of the ROIs are passed through a neural network and outputs a feature vector for each image describing what the ROI looks like.
-  **Classification:** The feature vectors can then be used to classify ROIs:
- A simple classifier can be trained using user supplied labeled data (e.g. an array of images of ROIs and a corresponding array of labels for each ROI).
- Alternatively, classification can be done by projecting the feature vectors into a lower-dimensional space using UMAP and then simply circling the region of space to classify the ROIs.
-  **Tracking**: The feature vectors can be combined with information about the position of the ROIs to track the ROIs across imaging sessions/planes.

# <a name="FAQs"></a>Frequently asked questions:

# TODO:
- Unify model training into this repo
- Improve classification notebooks
- Try Bokeh for interactive plots
- Integration tests
- Port demo notebooks to CoLab
- make reference API
- make nice README.md with gifs
