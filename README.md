# ROICaT <img src="logo.png"  width="300"  title="ROICaT"  alt="ROICaT"  align="right"  vspace = "60">

<!-- badges -->
[![build](https://github.com/RichieHakim/ROICaT/actions/workflows/.github/workflows/build.yml/badge.svg)](https://github.com/RichieHakim/ROICaT/actions/workflows/build.yml) 


**R**egion **O**f **I**nterest **C**lassification **a**nd **T**racking
A simple-to-use Python package for classifying images of cells and tracking them across imaging sessions/planes.
Currently designed to be used with Suite2p output data (stat.npy and ops.npy files) and CaImAn output data (results.h5 files), but any image data can be used (see [TODO: link] for details on using non-standard data).

## Table of contents
- [Announcements](#Announcements)<br>
- [Installation](#Installation)<br>
- [How to use ROICaT](#HowTo)<br>
- [Frequently Asked Questions](#FAQs)<br>
- [TODO](#TODO)<br>

## Announcements

### TRACKING NOW IN BETA! Try it out in the demo notebook [here](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/tracking_interactive_notebook.ipynb) or the demo script [here](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/tracking_scripted_notebook.ipynb).
### Classification still in Alpha. Contact me if you want to help test it.
### To help with development or beta test releases, please contact: rhakim@g.harvard.edu

# Installation
**We want ROICaT to be installable on all systems. If you have any issues during the installation process, please make a [github issue](https://github.com/RichieHakim/ROICaT/issues) with the error.**

### 0. Requirements
- [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)<br>
- GCC >= 5.4.0, ideally == 9.2.0. Google how to do this on your operating system. For unix/linux: check with `gcc --version`.<br>
- For GPU support, you just need a CUDA compatible NVIDIA GPU and the relevant [drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us). There is no need to download CUDA or CUDNN as PyTorch takes care of this during the installation. Using a GPU is not required, but can increase speeds 2-20x depending on the GPU and your data. See https://developer.nvidia.com/cuda-gpus for a list of compatible GPUs.
- On some Linux servers (like Harvard's O2 server), you may need to load modules instead of installing. To load conda, gcc, try: `module load conda3/latest gcc/9.2.0` or similar.<br>

### 1. (Recommended) Create a new conda environment
```
conda create -n ROICaT python=3.10
conda activate ROICaT
```

### 2. Clone the repo
```
git clone https://github.com/RichieHakim/ROICaT
cd path/to/ROICaT/directory
```

<!-- ### 2. (Optional) Update base conda environment
**`conda update -n base -c defaults conda python=3.10`**<br>
If you are on OSX or the above fails, try:\
**`conda update -n base conda python=3.10`**<br>

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


### 3. Install ROICaT (choose A. or B.)<br>
-  **A.** **CPU**-only version: MacOS, Windows, Linux<br>
```
pip install -v --user -e .[torch_cpu]
```

-  **B.** **GPU** version: Windows and Linux systems with GPU<br>
  -- First, install the CUDA version of `torch` and `torchvision`. See here for instructions: https://pytorch.org/get-started/locally/<br>
example: 
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
- Then, install ROICaT:
```
pip install -v --user -e .
```

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

-------------
# <a name="FAQs"></a>Frequently asked questions:
- Getting the error `OSError: [Errno 12] Cannot allocate memory` during `data.import_ROI_spatialFootprints()`
- There's something weird about the data you're using. I haven't figured out why this happens sometimes (albeit rarely).
- Solution: set `data.import_ROI_spatialFootprints(workers=1)` and `roinet.generate_dataloader(..., numWorkers_dataloader=0, persistentWorkers_dataloader=False)`

-------------
# TODO:
- fix constant used for um_per_pixel in ROInet_embedder
- unify and refactor backend
- integration tests
- make demo notebooks
- port demo notebooks to CoLab
- make reference API
- make nice README.md
- version testing
