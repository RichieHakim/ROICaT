# ROICaT <img src="logo.png"  width="300"  title="ROICaT"  alt="ROICaT"  align="right"  vspace = "60">

<!-- badges -->
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
- **TRACKING:** Try it out in the demo notebook [here](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/tracking_interactive_notebook.ipynb) or the demo script [here](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/tracking_scripted_notebook.ipynb).
- **CLASSIFICATION:** still in Alpha. Contact me if you want to help test it.
- To help with development or beta test releases, please contact: rhakim@g.harvard.edu

# Installation
ROICaT works on Windows, MacOS, and Linux. If you have any issues during the installation process, please make a [github issue](https://github.com/RichieHakim/ROICaT/issues) with the error.

### 0. Requirements
- [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)<br>
- GCC >= 5.4.0, ideally == 9.2.0. Google how to do this on your operating system. For unix/linux: check with `gcc --version`.<br>
- On some Linux servers (like Harvard's O2 server), you may need to load modules instead of installing. To load conda, gcc, try: `module load conda3/latest gcc/9.2.0` or similar.<br>
- **Optional:** [CUDA compatible NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) and [drivers](https://developer.nvidia.com/cuda-toolkit-archive). Using a GPU can increase ROICaT speeds 2-20x.

### 1. (Recommended) Create a new conda environment
```
conda create -n ROICaT python=3.9
conda activate ROICaT
```

### 2. Clone the repo
```
git clone https://github.com/RichieHakim/ROICaT
cd path/to/ROICaT/directory
```

<!-- ### 2. (Optional) Update base conda environment
**`conda update -n base -c defaults conda python=3.9`**<br>
If you are on OSX or the above fails, try:\
**`conda update -n base conda python=3.9`**<br>

### 3. Install dependencies (choose either 3A or 3B)
>Make sure current directory is the ROICaT repo directory (`cd path/to/ROICaT/directory`)<br>
>Note: If you are on a server and wish to install the GPU version, it might be necessary to load CUDA modules first using something like `module load gcc/9.2.0 cuda/11.2`.<br>
>If you get errors about GCC version, make sure you have version >=5.4.0. Check with `gcc --version`. On some Linux servers (like Harvard's O2 server), you may need to run `module load gcc/9.2.0` or similar.<br>

#### 3A. Install dependencies with GPU support (recommended)<br>
**`conda env create --file environment_GPU.yml`**<br>

#### 3B. Install dependencies with only CPU support<br>
**`conda env create --file environment_CPU_only.yml`**<br>

> If you'd like to give a custom name to the environment: `conda env create -n my_env_name --file environment_chooseGPUorCPUfile_.yml`<br>
> If you'd like to install environment into a different directory: ` conda env create --file environment_chooseGPUorCPUfile.yml --prefix /path/to/virtual/environment`<br> -->

### 3. Install PyTorch: <br>
- PyTorch is required for ROICaT and is well tested on torch versions `1.12.0` to `1.13.1`
- **Follow steps to install:** Follow the instructions here: https://pytorch.org/get-started/locally/. Read closely if you wish to install a GPU/CUDA version of PyTorch. The key takeaways are that you need a [CUDA compatible NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) and [drivers](https://developer.nvidia.com/cuda-toolkit-archive) that match the same version as the PyTorch CUDA version you choose. All CUDA 11.x versions are intercompatible, so if you have CUDA 11.8 drivers, you can install `torch==1.13.1+cu116`.

### 4. Install ROICaT
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

### 5. Use ROICaT<br>
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

-------------
# <a name="FAQs"></a>Frequently asked questions:
- Getting the error `OSError: [Errno 12] Cannot allocate memory` during `data.import_ROI_spatialFootprints()`
- There's something weird about the data you're using. I haven't figured out why this happens sometimes (albeit rarely).
- Solution: set `data.import_ROI_spatialFootprints(workers=1)` and `roinet.generate_dataloader(..., numWorkers_dataloader=0, persistentWorkers_dataloader=False)`

-------------
# TODO:
- unify and refactor backend
- integration tests
- make demo notebooks
- port demo notebooks to CoLab
- make reference API
- make nice README.md
- version testing
