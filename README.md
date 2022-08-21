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
conda create -n ROICaT python=3.9
conda activate ROICaT
cd path/to/ROICaT
pip install -r requirements.txt
```