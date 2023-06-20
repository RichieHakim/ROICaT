Installation
============

ROICaT has been tested on Windows 10 & 11, MacOS, and Linux.

Requirements
############

-  Segmented data. For example Suite2p output data (stat.npy and ops.npy files),
   CaImAn output data (results.h5 files), or any other type of data using this
   `custom data importing notebook
   <https://github.com/RichieHakim/ROICaT/blob/main/notebooks/jupyter/other/demo_custom_data_importing.ipynb>`__.
-  `Anaconda <https://www.anaconda.com/distribution/>`__ or `Miniconda
   <https://docs.conda.io/en/latest/miniconda.html>`__.
-  If using Windows: `Microsoft C++ Build Tools
   <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`__
-  If using linux/unix: GCC >= 5.4.0, ideally == 9.2.0. Google how to do this on
   your operating system. Check with: ``gcc --version``.
-  **Optional:** `CUDA compatible NVIDIA GPU
   <https://developer.nvidia.com/cuda-gpus>`__ and `drivers
   <https://developer.nvidia.com/cuda-toolkit-archive>`__. Using a GPU can
   increase ROICaT speeds ~5-50x, though without it, ROICaT will still run
   reasonably quick. GPU support is not available for Macs.
-  The below commands should be run in the terminal (Mac/Linux) or Anaconda
   Prompt (Windows).


Installation
------------


1. **Recommended: Create a new conda environment**

::

    conda create -n roicat python=3.11
    conda activate roicat
    pip install --upgrade pip

You will need to activate the environment with ``conda activate roicat`` each
time you want to use ROICaT.

2. **Install ROICaT**
   
::

    pip install --user roicat[all]
    pip install git+https://github.com/RichieHakim/roiextractors

Note: if you are using a zsh terminal, add quotes around the pip install
command, i.e. ``pip install "roicat[all]"``

1. **Clone the repo to get the scripts and notebooks**
   
::

    git clone http://github.com/RichieHakim/ROICaT
    cd path/to/ROICaT/dir




Troubleshooting Installation
============================

Windows installation
####################

If you receive the error:
``ERROR: Could not build wheels for hdbscan, which is required to install pyproject.toml-based projects``
on Windows, make sure that you have installed Microsoft C++ Build Tools.
If not, download from
`here <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`__
and run the commands:

::

   cd path/to/vs_buildtools.exe
   vs_buildtools.exe --norestart --passive --downloadThenInstall --includeRecommended --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Workload.MSBuildTools

Then, try proceeding with the installation by rerunning the pip install
commands above.
(`reference <https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst>`__)


GPU support
###########

GPU support is not required. Windows users will often need to manually
install a CUDA version of pytorch (see below). Note that you can check
your nvidia driver version using the shell command: ``nvidia-smi`` if
you have drivers installed.

Use the following command to check your PyTorch version and if it is GPU
enabled:

::

  python -c “import torch, torchvision; print(f’Using versions:
  torch=={torch.\__version\_\_},
  torchvision=={torchvision.\__version\_\_}‘);
  print(f’torch.cuda.is_available() = {torch.cuda.is_available()}’)”


**Outcome 1:** Output expected if GPU is enabled:

::

   Using versions: torch==X.X.X+cuXXX, torchvision==X.X.X+cuXXX
   torch.cuda.is_available() = True

This is the ideal outcome. You are using a CUDA version of PyTorch and
your GPU is enabled.

**Outcome 2:** Output expected if non-CUDA version of PyTorch is
installed:

::

   Using versions: torch==X.X.X, torchvision==X.X.X
   OR
   Using versions: torch==X.X.X+cpu, torchvision==X.X.X+cpu
   torch.cuda.is_available() = False

If a non-CUDA version of PyTorch is installed, please follow the
instructions here: https://pytorch.org/get-started/locally/ to install a
CUDA version. If you are using a GPU, make sure you have a `CUDA
compatible NVIDIA GPU <https://developer.nvidia.com/cuda-gpus>`__ and
`drivers <https://developer.nvidia.com/cuda-toolkit-archive>`__ that
match the same version as the PyTorch CUDA version you choose. All CUDA
11.x versions are intercompatible, so if you have CUDA 11.8 drivers, you
can install ``torch==2.0.1+cu117``.

**Outcome 3:** Output expected if GPU is not available:

::

   Using versions: torch==X.X.X+cuXXX, torchvision==X.X.X+cuXXX
   torch.cuda.is_available() = False

If a CUDA version of PyTorch is installed but GPU is not available, make
sure you have a `CUDA compatible NVIDIA
GPU <https://developer.nvidia.com/cuda-gpus>`__ and
`drivers <https://developer.nvidia.com/cuda-toolkit-archive>`__ that
match the same version as the PyTorch CUDA version you choose. All CUDA
11.x versions are intercompatible, so if you have CUDA 11.8 drivers, you
can install ``torch==2.0.1+cu117``.