# ROICaT
Region Of Interest Classification and Tracking

A simple to use pipeline designed for classifying images of cells and tracking them across imaging planes/sessions.
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

`git clone https://github.com/RichieHakim/ROICaT` \
`cd path/to/ROICaT/directory`

`conda update -n base -c defaults conda` \
`conda create -n ROICaT python=3.9` \
`conda activate ROICaT` 

Upgrade pip. \
If using Windows, then use: `python -m pip install --upgrade pip` \
`pip install --upgrade pip`

**Install PyTorch.**<br>
*If possible, install on a computer with a CUDA compatible GPU:*<br>
<ul>
    <li>Install CUDA 11 (ideally 11.3) [https://developer.nvidia.com/cuda-downloads or https://developer.nvidia.com/cuda-11-3-1-download-archive]</li>
    <li>Install CUDNN [https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html].</li>
    
If you don't have a GPU+CUDA, use `pip install torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu`<br>
OR see [https://pytorch.org/get-started/locally/] for other versions
`pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`

`pip install -r requirements.txt`

Install torch_sparse. \
If you get errors about GCC version, make sure you have version >=5.4.0. Check with `gcc --version`. On some Linux servers [like Harvard's O2 server], you may need to run `module load gcc/9.2.0` or similar.) \
`pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.1+${CUDA}.html`
```