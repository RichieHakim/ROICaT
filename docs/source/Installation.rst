
Installation
============

ROICaT has been tested on Windows 10 & 11, MacOs, and Linux.

Requirements
############

* Anaconda or Miniconda
* For unix/linux users: GCC >= 5.4.0, ideally == 9.2.0
* On *some* linux servers (like Harvard's O2 server), you may need to load modules instead
  of installing. To load modules try: 

  .. code-block:: console

    module load conda3/latest gcc/9.2.0

* **Optional**: For GPU support, CUDA compatbile NVIDIA GPU and drivers. 
  Using a GPU can increase ROICaT speeds ~5-50x. 
  GPU support is not available for MacOS users.
* For windows users: Visual Studio >=2019 (Community edition is fine)

How to install
##############

1. **Recommended: Create a new conda environment**

   .. code-block:: python

    conda create -n ROICaT python=3.11
    conda activate ROICaT

2. **Clone the repository**
   
   .. code-block:: python

    git clone http://github.com/RichieHakim/ROICaT
    cd path/to/ROICaT/dir

3. **Install ROICaT**
   
    .. code-block:: python

     pip install -v -e .[core]
    
  Optional: upgrade pip by running `pip install --upgrade pip`

  Note: if you using a zsh terminal, use the following for installation:

    .. code-block:: python

     pip3 install -v -e '.[core]'

4. **Use ROICaT**
   Beginner: Run a provided jupyter notebook
   Advanced: Make a parameter file and run in within terminal 

    .. code-block:: python

     python -m ROICaT

   