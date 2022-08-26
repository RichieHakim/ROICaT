# ROICaT
**R**egion **O**f **I**nterest **C**lassification **a**nd **T**racking

### !!! REPO UNDER CONSTRUCTION !!!
### to help with development or beta test releases, please contact:  r* haki$ m @*g.$ harvard .edu  without the spaces, *, and $

A [hopefully] simple-to-use Python package for classifying images of cells and tracking them across imaging sessions/planes.
Currently designed to be used with Suite2p output data (stat.npy and ops.npy files), but any image data can be used (see [TODO: link] for details on using non-standard data).

***Ways to use ROICaT:***
- **Easy:** Try out ROICaT on Google Colab: [TODO: Link]
- **Intermediate:** Run it on your own computer. See [Installation](#Installation) for how to install.
    - Using provided Jupyter Notebook(s): [Notebooks](https://github.com/RichieHakim/ROICaT/tree/main/notebooks).
    - Using command line: `python -m ROICaT`. See [TODO: link to how-to] for details.

- **Advanced:** Train a new ROInet model using the provided Jupyter Notebook [TODO: link]. Or contribute to the code base! This is a big collaborative effort, so please feel free to send a pull request or open an issue.


# Installation

### Requirements
- [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)<br>
- GCC >= 5.4.0, ideally == 9.2.0. Google how to do this on your operating system. For unix/linux: check with `gcc --version`.<br>
- For GPU support, you just need a CUDA compatible NVIDIA GPU and the relevant drivers. 
- On some Linux servers (like Harvard's O2 server), you may need to load modules instead of installing. To load conda, gcc, try: `module load conda3/latest gcc/9.2.0` or similar.<br>

### 1. Clone the repo
**`git clone https://github.com/RichieHakim/ROICaT`**<br>
**`cd path/to/ROICaT/directory`**<br>

### 2. Create a conda environment and activate it
**`conda update -n base -c defaults conda`**<br>
**`conda create -n ROICaT python=3.9`**<br>
**`conda activate ROICaT`**<br>

### 3. Upgrade pip
**`pip install --upgrade pip`**<br>
>If using Windows, then use: `python -m pip install --upgrade pip`<br>

### 4. Install dependencies (choose either 4A or 4B)
>Make sure current directory is the ROICaT repo directory (`cd path/to/ROICaT/directory`)<br>
>Note: If you are on a server, it might be necessary to load CUDA modules first using something like `module load gcc/9.2.0 cuda/11.2`.<br>
>Note: This step is slow. It needs to compile a large amount of C code the first time you run it. May take around 20 minutes.<br>
>If you get errors about GCC version, make sure you have version >=5.4.0. Check with `gcc --version`. On some Linux servers (like Harvard's O2 server), you may need to run `module load gcc/9.2.0` or similar.<br>

#### 4A. Install dependencies with GPU support (recommended)<br>
**`pip install -r requirements_GPU.txt`**<br>

#### 4B. Install dependencies with only CPU support<br>
**`pip install -r requirements_CPU_only.txt`**<br>

### 5. Use ROICaT<br>
- Run a Jupyter Notebook: [Notebooks](https://github.com/RichieHakim/ROICaT/tree/main/notebooks)<br>
- Make a parameter file and run in command line: `python -m ROICaT`. See [TODO: link to how-to] for details.<br>

-------------

***General workflow:***
- **Pass ROIs through ROInet:** Images of the ROIs are passed through a neural network and outputs a feature vector for each image describing what the ROI looks like.
- **Classification:** The feature vectors can then be used to classify ROIs:
    - A simple classifier can be trained using user supplied labeled data (e.g. an array of images of ROIs and a corresponding array of labels for each ROI).
    - Alternatively, classification can be done by projecting the feature vectors into a lower-dimensional space using UMAP and then simply circling the region of space to classify the ROIs.
- **Tracking**: The feature vectors can be combined with information about the position of the ROIs to track the ROIs across imaging sessions/planes.


-------------
## TODO:
- add tracking to repo
- add classification to repo
- unify and refactor backend
- make better installation process (setup.py or PyPi package)
- add CaImAn support
- integration tests
- make demo notebooks
- port demo notebooks to CoLab
- make reference API
- make nice README.md
- version testing


-------------
# Frequently asked questions:
- Getting the error `OSError: [Errno 12] Cannot allocate memory`
    - There's something weird about the data you're using. I haven't figured out why this happens sometimes (albeit rarely).
    - Solution: set `data.import_ROI_spatialFootprints(workers=1)` and `roinet.generate_dataloader(..., numWorkers_dataloader=0)`