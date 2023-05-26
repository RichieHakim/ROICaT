
How to Use ROICaT
=================

General Workflow
################

* **Pass ROIs through ROInet:** 

    Images of the ROIs are passed through a neural network and outputs
    a feature vector for reach image describing what the ROI looks like.

* **Classification:**

    The feature vectors can then be used to classify ROIs.

    * A simple classifier can be trained using user supplied labeled data (e.g. an array of images of ROIs and a corresponding array of labels for each ROI).

    * Alternatively, classification can be done by projecting the feature vectors into a lower-dimensional space using UMAP and then simply circling the region of space to classify the ROIs.

* **Tracking:**

    The feature vectors can be combined with information about the position of the ROIs to track the ROIs across imaging sessions/planes.

Ways to deploy ROICaT
#####################

* **Google Colab**
   Try running ROICaT in Google Colab. This is a good option if you want to try out ROICaT without installing anything on your computer.

* **Local Installation**
   Install ROICaT on your computer. This is a good option if you want to use ROICaT on your own computer.
   You can also run ROICaT through our provided `Jupyter Notebook(s) <https://github.com/RichieHakim/ROICaT/tree/72eaa2918d0fbb452bfbc5ba0b2703a32bb4bed4/notebooks>`_ 

* **Command Line Interface (CLI)** 
   Use ROICaT through the command line. This is a good option if you want to use ROICaT on a remote server.

   .. code-block:: python

      python -m ROICaT

* **Train a new ROInet model**
   Train a new ROInet model. This is a good option if you want to train a new ROInet model on your own data.

* **Contribute to the code base!**
  This is a big collaborative effort, so please feel free to send a pull request or open an issue on GitHub.

Examples
########

