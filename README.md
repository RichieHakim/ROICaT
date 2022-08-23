# ROICaT
**R**egion **O**f **I**nterest **C**lassification **a**nd **T**racking

A [hopefully] simple-to-use package for classifying images of cells and tracking them across imaging sessions/planes.
Currently designed to be used with Suite2p output data (stat.npy and ops.npy files).

How it works:
- Pass ROIs through ROInet: There is a neural network that takes in images of the extracted ROIs (from the output of Suite2p or CaImAn) and outputs a feature vector for each image describing what the ROI looks like.
- Classification: The feature vectors can then be used to classify ROIs:
    - A simple classifier can be trained using user supplied labeled data (e.g. an array of images of ROIs and a corresponding array of labels for each ROI).
    - Alternatively, classification can be done by projecting the feature vectors into a lower-dimensional space using UMAP and then simply circling the region of space to classify the ROIs.
- Tracking: The feature vectors can be combined with information about the position of the ROIs to track the ROIs across imaging sessions/planes.


Installation
------------

### Requirements
- GCC >= 5.4.0, ideally == 9.2.0. Check with `gcc --version`. On some Linux servers (like Harvard's O2 server), you may need to run `module load gcc/9.2.0` or similar.
- For GPU support, you need to install the relevant CUDA toolkit. Currently, ROICaT supports CUDA 11.x (ideally 11.3): https://developer.nvidia.com/cuda-toolkit. On some Linux servers (like Harvard's O2 server), you may need to run `module load cuda/11.x`. CUDA has some intercompatibility between 11.x versions, so loading/installing v11.2 or similar is likely to work fine.

### 1. Clone the repo
**`git clone https://github.com/RichieHakim/ROICaT`**<br>
**`cd path/to/ROICaT/directory`**<br>

### 2. Create a conda environment and activate it
**`conda update -n base -c defaults conda`**<br>
**`conda create -n ROICaT python=3.9`**<br>
**`conda activate ROICaT`**<br>

### 3. Upgrade pip
`pip install --upgrade pip`
>If using Windows, then use: `python -m pip install --upgrade pip`<br>

### 4. Install PyTorch<br>
For installation on a computer with a GPU + CUDA(11.x, ideally 11.3) + CuDNN, use the following command:<br>
**`pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`**<br>

For installation on a computer with only CPU, use the following command:<br>
`pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu`<br>
>Ideally, try to install on a computer with a CUDA compatible GPU. How to install CUDA + CuDNN:<br>
>- Install CUDA 11 (ideally 11.3) [https://developer.nvidia.com/cuda-downloads or https://developer.nvidia.com/cuda-11-3-1-download-archive]<br>
>- Install CUDNN [https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html]<br>
>
>OR see [https://pytorch.org/get-started/locally/] for other versions<br>

### 5. Install PyTorch Sparse<br>
**`pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+${CUDA}.html`**
Note: This is very slow. It needs to compile a large amount of C code. May take around 20 minutes.
>See here for help and details: [https://github.com/rusty1s/pytorch_sparse]<br>
>If you get errors about GCC version, make sure you have version >=5.4.0. Check with `gcc --version`. On some Linux servers (like Harvard's O2 server), you may need to run `module load gcc/9.2.0` or similar.<br>

### 6. Install various dependencies<br>
**`pip install -r requirements.txt`**



-------------
## TODO:
- add tracking to repo
- add classification to repo
- unify and refactor backend
- add CaImAn support
- integration tests
- make demo notebooks
- port demo notebooks to CoLab
- make reference API
- make nice README.md
- version testing

