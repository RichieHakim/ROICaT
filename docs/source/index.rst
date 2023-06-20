ROICaT (Region of Interest Classification and Tracking)
=======================================================

`Github Repository <https://github.com/richiehakim/ROICaT>`_

.. image:: ../media/logo1.png
   :align: right
   :width: 200
   :alt: ROICaT logo_1

|


ROICaT is a simple-to-use Python package for classifying images of cells and tracking them across imaging sessions/planes.

--------
Contents
--------

.. toctree::
   :maxdepth: 1

   installation
   roicat


Installation
------------
For detailed instructions, see :doc:`installation`

1. **Recommended: Create a new conda environment**

::

      conda create -n roicat python=3.11
      conda activate roicat
      pip install --upgrade pip

You will need to activate the environment with ``conda activate roicat`` each time you want to use ROICaT.

2. **Install ROICaT**
   
::

      pip install --user roicat[all]
      pip install git+https://github.com/RichieHakim/roiextractors

Note: if you are using a zsh terminal, add quotes around the pip install command, i.e. ``pip install "roicat[all]"``

1. **Clone the repo to get the scripts and notebooks**
   
::

      git clone http://github.com/RichieHakim/ROICaT
      cd path/to/ROICaT/dir


Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _github: https://github.com/RichieHakim/ROICaT