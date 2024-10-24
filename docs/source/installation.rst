Installation
============

.. include:: ../../README.md/
   :start-after: # Installation
   :end-before: # TODO
   :parser: myst_parser.sphinx_


Troubleshooting Installation
============================

package installation issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have issues importing packages like `roicat` or any of its dependencies, try reinstalling `roicat` with the following commands within the environment:

.. code-block:: bash

   pip uninstall roicat
   pip install --upgrade --force --no-cache-dir roicat[all]


HDBSCAN installation issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are using **Windows** receive the error: `ERROR: Could not build wheels for hdbscan, which is
required to install pyproject.toml-based projects` on Windows, make sure that
you have installed Microsoft C++ Build Tools. If not, download from
[here](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and run the
commands:

.. code-block:: bash

   cd path/to/vs_buildtools.exe
   vs_buildtools.exe --norestart --passive --downloadThenInstall --includeRecommended --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Workload.MSBuildTools

Then, try proceeding with the installation by rerunning the pip install commands
above.
([reference](https://stackoverflow.com/questions/64261546/how-to-solve-error-microsoft-visual-c-14-0-or-greater-is-required-when-inst))

GPU support issues
~~~~~~~~~~~~~~~~~~

GPU support is not required. Windows users will often need to manually install a
CUDA version of pytorch (see below). Note that you can check your nvidia driver
version using the shell command: `nvidia-smi` if you have drivers installed. 

Use the following command to check your PyTorch version and if it is GPU
enabled:

.. code-block:: bash

   python -c "import torch, torchvision; print(f'Using versions: torch=={torch.__version__}, torchvision=={torchvision.__version__}');  print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')"
    
**Outcome 1:** Output expected if GPU is enabled:

.. code-block:: bash

   Using versions: torch==X.X.X+cuXXX, torchvision==X.X.X+cuXXX
   torch.cuda.is_available() = True
    
This is the ideal outcome. You are using a CUDA version of PyTorch and
your GPU is enabled.

**Outcome 2:** Output expected if non-CUDA version of PyTorch is
installed:

.. code-block:: bash

   Using versions: torch==X.X.X, torchvision==X.X.X
   OR
   Using versions: torch==X.X.X+cpu, torchvision==X.X.X+cpu
   torch.cuda.is_available() = False
    
If a non-CUDA version of PyTorch is installed, please follow the
instructions here: https://pytorch.org/get-started/locally/ to install a CUDA
version. If you are using a GPU, make sure you have a [CUDA compatible NVIDIA
GPU](https://developer.nvidia.com/cuda-gpus) and
[drivers](https://developer.nvidia.com/cuda-toolkit-archive) that match the same
version as the PyTorch CUDA version you choose. All CUDA 11.x versions are
intercompatible, so if you have CUDA 11.8 drivers, you can install
`torch==2.0.1+cu117`.

**Solution:**
If you are sure you have a compatible GPU and correct drivers, you can force
install the GPU version of pytorch, see the pytorch installation instructions.
Links for the [latest version](https://pytorch.org/get-started/locally/) or
[older versions](https://pytorch.org/get-started/previous-versions/). Example:

.. code-block:: bash

   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

**Outcome 3:** Output expected if CUDA version of PyTorch is installed but GPU
is not available:

.. code-block:: bash

   Using versions: torch==X.X.X+cuXXX, torchvision==X.X.X+cuXXX
   torch.cuda.is_available() = False
   
If a CUDA version of PyTorch is installed but GPU is not available, make sure
you have a [CUDA compatible NVIDIA GPU](https://developer.nvidia.com/cuda-gpus)
and [drivers](https://developer.nvidia.com/cuda-toolkit-archive) that match the
same version as the PyTorch CUDA version you choose. All CUDA 11.x versions are
intercompatible, so if you have CUDA 11.8 drivers, you can install
`torch==2.0.1+cu117`.