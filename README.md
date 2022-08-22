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


Installation
------------

```
git clone url.to.repo
cd path/to/ROICaT/directory

conda update -n base -c defaults conda
conda create -n ROICaT python=3.9
conda activate ROICaT

pip install --upgrade pip (if using Windows, then use: python -m pip install --upgrade pip)
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch==1.12.0+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113  (If you don't have a GPU+CUDA, see https://pytorch.org/ for other versions)
pip install -r requirements.txt
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html
```