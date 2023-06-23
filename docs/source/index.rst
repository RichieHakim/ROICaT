ROICaT
======

Region of Interest Classification and Tracking
##############################################

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

   howto

   roicat


Installation
------------
For detailed instructions, see :doc:`installation`

1. **Recommended: Create a new conda environment**

.. literalinclude:: ../helpers/create_env.txt

You will need to activate the environment with ``conda activate roicat`` each
time you want to use ROICaT.

2. **Install ROICaT**
   
.. literalinclude:: ../helpers/pip_install.txt

Note: if you are using a zsh terminal, add quotes around the pip install
command, i.e. ``pip install "roicat[all]"``

3. **Clone the repo to get the scripts and notebooks**
   
.. literalinclude:: ../helpers/clone_repo.txt


Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _github: https://github.com/RichieHakim/ROICaT