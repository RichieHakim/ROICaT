# Welcome to ROICaT

<div>
    <img src="docs/media/logo1.png" alt="ROICaT" width="200"  align="right"  style="margin-left: 20px"/>
</div>

[![build](https://github.com/RichieHakim/ROICaT/actions/workflows/.github/workflows/build.yml/badge.svg)](https://github.com/RichieHakim/ROICaT/actions/workflows/build.yml) 
[![PyPI version](https://badge.fury.io/py/roicat.svg)](https://badge.fury.io/py/roicat)
[![Downloads](https://pepy.tech/badge/roicat)](https://pepy.tech/project/roicat)
[![build](https://github.com/RichieHakim/ROICaT/actions/workflows/.github/workflows/check_huggingface_space.yml/badge.svg)](https://github.com/RichieHakim/ROICaT/actions/workflows/check_huggingface_space.yml) 


**🎉 CONTRIBUTIONS WELCOME! 🎉** \
See the [TODO](#todo) section

- **Documentation: [https://roicat.readthedocs.io/en/latest/](https://roicat.readthedocs.io/en/latest/)**
- Discussion forum: [https://groups.google.com/g/roicat_support](https://groups.google.com/g/roicat_support)
- Technical support: [Github Issues](https://github.com/RichieHakim/ROICaT/issues)

## **R**egion **O**f **I**nterest **C**lassification **a**nd **T**racking ᗢ
A simple-to-use Python package for automatically classifying images of cells and tracking them across imaging sessions/planes.
<div>
    <img src="docs/media/tracking_FOV_clusters_rich.gif" alt="tracking_FOV_clusters_rich"  width="400"  align="right" style="margin-left: 20px"/>
</div>

**Why use ROICaT?**
- **It's easy to use. You don't need to know how to code. You can use the
  interactive notebooks or online app to run the pipelines with just a few
  clicks.**
- It's accurate. ROICaT was desgined to be better than existing tools. It is
  capable of classifying and tracking neuron ROIs at accuracies approaching
  human performance out of the box.
- It's fast and computational requirements are low. You can run it on a laptop.
  It was designed to be used with >1M ROIs, and can utilize GPUs to speed things
  up.

With ROICaT, you can:
- **Classify ROIs** into different categories (e.g. neurons, dendrites, glia,
  etc.).
- **Track ROIs** across imaging sessions/planes (e.g. ROI #1 in session 1 is the
  same as ROI #7 in session 2).

**What data types can ROICaT process?** 
- ROICaT can accept any imaging data format including: Suite2p, CaImAn, CNMF,
  NWB, raw/custom ROI data and more. See below for details on how to use any
  data type with ROICaT.

<br>
<br>

# How to use ROICaT
<div>
    <img src="docs/media/umap_with_labels.png" alt="ROICaT" width="300"  align="right"  style="margin-left: 20px"/>
</div>

### TRACKING: 
- [Online App](https://huggingface.co/spaces/richiehakim/ROICaT_tracking): Good for first time users. Try it out without installing anything.
- [Interactive
  notebook](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/tracking/1_tracking_interactive_notebook.ipynb). Or run on google colab: <a target="_blank" href="https://githubtocolab.com/RichieHakim/ROICaT/blob/main/notebooks/tracking/1_tracking_interactive_notebook.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- [Command line interface script](https://github.com/RichieHakim/ROICaT/blob/main/scripts/run_tracking.sh): 
```shell
roicat --pipeline tracking --path_params /path/to/params.yaml --dir_data /folder/with/data/ --dir_save /folder/save/ --prefix_name_save expName --verbose
```
  
### CLASSIFICATION:
- [Interactive notebook -
  Drawing](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/classification/A1_classify_by_drawingSelection.ipynb). Or run on google colab: <a target="_blank" href="https://githubtocolab.com/RichieHakim/ROICaT/blob/main/notebooks/classification/A1_classify_by_drawingSelection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- [Interactive notebook -
  Labeling](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/classification/B1_labeling_interactive.ipynb)
- [Interactive notebook - Train
  classifier](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/classification/B2_classifier_train_interactive.ipynb)
- [Interactive notebook - Inference with
  classifier](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/classification/B3_classifier_inference_interactive.ipynb)

**OTHER:** 
- [Custom data importing
  notebook](https://github.com/RichieHakim/ROICaT/blob/main/notebooks/other/demo_data_importing.ipynb)
- Use the API to integrate ROICaT functions into your own code:
  [Documentation](https://roicat.readthedocs.io/en/latest/roicat.html).
- Run the full tracking pipeline using the CLI or
  `roicat.pipelines.pipeline_tracking` with default parameters generated from
  `roicat.util.get_default_paramaters()` saved as a yaml file.
<!-- - Train a new ROInet model using the provided Jupyter Notebook [TODO: link]. -->


# Installation
ROICaT works on Windows, MacOS, and Linux. If you have any issues during the
installation process, please make a [github
issue](https://github.com/RichieHakim/ROICaT/issues) with the error.

### 0. Requirements
- [Anaconda](https://www.anaconda.com/distribution/) or
  [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- If using Windows: [Microsoft C++ Build
  Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- The below commands should be run in the terminal (Mac/Linux) or Anaconda
  Prompt (Windows).

### 1. (Recommended) Create a new conda environment
```
conda create -n roicat python=3.12
conda activate roicat
```
You will need to activate the environment with `conda activate roicat` each time
you want to use ROICaT.

### 2. Install ROICaT
```
pip install roicat[all]
pip install git+https://github.com/RichieHakim/roiextractors
```
**Note on zsh:** if you are using a zsh terminal, change command to: `pip3
install --user 'roicat[all]'` <br>
**Note on installing GPU support on Windows:** see
[GPU Troubleshooting](https://roicat.readthedocs.io/en/latest/installation.html#gpu-support-issues)
documentation.
<br>
**Note on opencv:** The headless version of opencv is installed by default. If
the regular version is already installed, you will need to uninstall it first.

### 3. Clone the repo to get the notebooks
```
git clone https://github.com/RichieHakim/ROICaT
```
Then, navigate to the `ROICaT/notebooks/jupyter` directory to run the notebooks.


# Upgrading versions
There are 2 parts to upgrading ROICaT: the **Python package** and the
**repository files** which contain the notebooks and scripts.\
Activate your environment first, then...\
To upgrade the Python package, run:
```
pip install --upgrade roicat[all]
```
To upgrade the repository files, navigate your terminal to the `ROICaT` folder and run:
```
git pull
```


# General workflow:
- **Pass ROIs through ROInet:** Images of the ROIs are passed through a neural
  network which outputs a feature vector for each image describing what the ROI
  looks like.
-  **Classification:** The feature vectors can then be used to classify ROIs:
   - A simple regression-like classifier can be trained using user-supplied
     labeled data (e.g. an array of images of ROIs and a corresponding array of
     labels for each ROI).
   - Alternatively, classification can be done by projecting the feature vectors
     into a lower-dimensional space using UMAP and then simply circling the
     region of space to classify the ROIs.
-  **Tracking**: The feature vectors can be combined with information about the
   position of the ROIs to track the ROIs across imaging sessions/planes.


# Run the app locally
Although, we recommend transitioning to using the notebooks or CLI instead of the app, you can download and run the app locally with the following command:
```
sudo docker run -it -p 7860:7860 --platform=linux/amd64 --shm-size=10g registry.hf.space/richiehakim-roicat-tracking:latest streamlit run app.py
```


# TODO:
#### algorithmic improvements:
- [ ] Add in method to use more similarity metrics for tracking
- [ ] Coordinate descent on each similarity metric
- [ ] Add F and Fneu to data_roicat, dFoF and trace quality metric functions
- [ ] Add in notebook for demonstrating using temporal similarity metrics (SWT on dFoF)
- [ ] Make a standard classifier
- [ ] Try other clustering methods
- [x] Make image aligner based on image similarity + RANSAC of centroids or s_SF
- [ ] Better post-hoc curation metrics and visualizations
- [ ] Improve non-rigid image registration methods (border performance)
- [ ] Make non-rigid image registration optional
#### code improvements:
- [ ] **Finish ROIextractors integration**
- [ ] Update automatic regression module (make new repo for it)
- [ ] Switch to ONNX for ROInet
- [ ] Some more integration tests
- [ ] Figure out RNG / OS differences issues for tests
- [ ] Add more documentation / tutorials
- [x] Make a GUI
- [ ] Add settings to the GUI
- ~~[ ] Make a Docker container~~
- ~~Make colab demo notebook not require user data~~
- [x] Make a better CLI
- [ ] Switch to pyproject.toml
- [ ] Improve params.json / default params system
- [ ] Spruce up training code
#### other:
- [ ] Write the paper
- [ ] Make tweet about it
- [ ] Make a video or two on how to use it
- [ ] Maybe use lightthetorch for torch installation
- [ ] Better Readme
- [ ] More documentation
- [ ] Make a regression model for in-plane-ness
- [ ] Formalize bounty program