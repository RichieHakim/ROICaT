
Troubleshooting
===============

GPU support
-----------

GPU support is not required. Windows users will often need to manually install a CUDA version of pytorch (see below).

*Note that you can check your nvidia driver version using the shell command:*
.. code-block:: console

    nvidia-smi

Check your PyTorch version and if it is GPU enabled
------------------------------------------------------
.. code-block:: python

    python -c "import torch, torchvision; print(f'Using versions: torch=={torch.__version__}, torchvision=={torchvision.__version__}');  print(f'torch.cuda.is_available() = {torch.cuda.is_available()}')"

* **Outcome 1:** GPU sucessfully enabled

.. code-block:: console

    Using versions: torch==X.X.X+cuXXX, torchvision==X.X.X+cuXXX
    torch.cuda.is_available() = True

This is the expected output if you have a GPU and have installed a CUDA version of PyTorch.

* **Outcome 2:** Non-CUDA version of PyTorch is installed

.. code-block:: console

    Using versions: torch==X.X.X, torchvision==X.X.X
    or
    Using versions: torch==X.X.X+cpu, torchvision==X.X.X+cpu
    torch.cuda.is_available() = False

If a non-CUDA version of PyTorch is installed, please follow the instructions `here <https://pytorch.org/get-started/locally/>`_ to install a CUDA version of PyTorch.

If you are using a GPU, make sure you have a `CUDA compatible NVIDIA GPU <https://developer.nvidia.com/cuda-gpus>`_ and have installed the `CUDA drivers <https://developer.nvidia.com/cuda-downloads>`_.

These must match the same version as the PyTorch CUDA version you install.

*Note*: All CUDA 11.x versions are intercompatible, so you can install any of the CUDA 11.x versions of PyTorch.

Windows ERROR: Could not build wheels for hdbscan, which is required to install pyproject.toml-based projects
-------------------------------------------------------------------------------------------------------------

Make sure you have installed Microsoft C++ Build Tools. You can download them from `microsoft here <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_.

Then, run the following commands: 

.. code-block:: console

    cd path/to/vs_buildtools.exe
    vs_buildtools.exe --norestart --passive --downloadThenInstall --includeRecommended --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Workload.MSBuildTools

Then, try proceeding with the installation again.

