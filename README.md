# ROICaT
**R**egion **O**f **I**nterest **C**lassification **a**nd **T**racking

A [hopefully] simple-to-use pipeline for classifying images of cells and tracking them across imaging planes/sessions.
Currently designed to use with Suite2p output data (stat.npy and ops.npy files).

TODO:
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



Installation
------------

### 1. Clone the repo
**`git clone https://github.com/RichieHakim/ROICaT`**<br>
**`cd path/to/ROICaT/directory`**<br>

### 2. Create a conda environment and activate it
**`conda update -n base -c defaults conda`**<br>
**`conda create -n ROICaT python=3.9`**<br>
**`conda activate ROICaT`**<br>

### 3. Upgrade pip
`pip install --upgrade pip`
>*If using Windows, then use: `python -m pip install --upgrade pip`*<br>

### 4. Install PyTorch<br>
**`pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`**<br>
>*If possible, install on a computer with a CUDA compatible GPU:*<br>
>- Install CUDA 11 (ideally 11.3) [https://developer.nvidia.com/cuda-downloads or https://developer.nvidia.com/cuda-11-3-1-download-archive]<br>
>- Install CUDNN [https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html]<br>
>*If you don't have a GPU+CUDA, use `pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu`*<br>
>*OR see [https://pytorch.org/get-started/locally/] for other versions*<br>

### 5. Install torch-sparse<br>
**`pip install torch-scatter PyTorch Sparse -f https://data.pyg.org/whl/torch-1.12.1+${CUDA}.html`**
>*See here for help and details: [https://github.com/rusty1s/pytorch_sparse]<br>
>*If you get errors about GCC version, make sure you have version >=5.4.0. Check with `gcc --version`. On some Linux servers (like Harvard's O2 server), you may need to run `module load gcc/9.2.0` or similar.*<br>

### 6. Install various dependencies<br>
**`pip install -r requirements.txt`**