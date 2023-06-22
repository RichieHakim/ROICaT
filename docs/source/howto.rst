How to use ROICaT
=================

Listed below, we have a suite of easy to run notebooks for running the ROICaT
pipelines.

* The **Google CoLab notebooks** can be run fully remotely without installing
  anything on your computer.
* The **Jupyter notebooks** can be run locally on your computer and require you to
  `install ROICaT
  <https://roicat.readthedocs.io/en/latest/installation.html>`__.

.. image:: ../media/tracking_FOV_clusters_rich.gif
   :align: right
   :width: 300
   :alt: ROICaT logo_1

**TRACKING:** 
   * `Interactive notebook <https://github.com/RichieHakim/ROICaT/blob/main/notebooks/jupyter/tracking/tracking_interactive_notebook.ipynb>`__
   * `Google CoLab <https://githubtocolab.com/RichieHakim/ROICaT/blob/main/notebooks/colab/tracking/tracking_interactive_notebook.ipynb>`__
   * (TODO) `demo script <https://github.com/RichieHakim/ROICaT/blob/main/notebooks/jupyter/tracking/tracking_scripted_notebook.ipynb>`__

|

.. image:: ../media/umap_with_labels.png
   :align: right
   :width: 200
   :alt: ROICaT logo_1

**CLASSIFICATION:** 
   * `Interactive notebook - Drawing <https://github.com/RichieHakim/ROICaT/blob/main/notebooks/jupyter/classification/classify_by_drawingSelection.ipynb>`__
   * `Google CoLab - Drawing <https://githubtocolab.com/RichieHakim/ROICaT/blob/main/notebooks/colab/classification/classify_by_drawingSelection_colab.ipynb>`__
   * `Interactive notebook - Labeling <https://github.com/RichieHakim/ROICaT/blob/main/notebooks/jupyter/classification/labeling_interactive.ipynb>`__
   * (TODO) `Interactive notebook - Train classifier <https://github.com/RichieHakim/ROICaT>`__ 
   * (TODO) `Interactive notebook - Inference with classifier <https://github.com/RichieHakim/ROICaT>`__

|

**OTHER:** 
   * `Custom data importing notebook <https://github.com/RichieHakim/ROICaT/blob/main/notebooks/jupyter/other/demo_data_importing.ipynb>`__
   * Train a new ROInet model using the provided Jupyter Notebook [TODO: link]. 
   * Use the API to integrate ROICaT functions into your own code: `Documentation <https://roicat.readthedocs.io/en/latest/roicat.html>`__.

How ROICaT pipelines work:
##########################

   * **Pass ROIs through ROInet:** Images of the ROIs are passed through a
     neural network which outputs a feature vector for each image describing what
     the ROI looks like. These feature vectors are used in the next steps.
   * **Classification:**
      * A simple **classifier** can be trained using user supplied labeled data
        (e.g. an array of images of ROIs and a corresponding array of labels for
        each ROI).
      * Alternatively, classification can be done by projecting the feature
        vectors into a lower-dimensional space using UMAP and then simply
        **drawing** a circle around the region of space to classify the ROIs.
   * **Tracking**: The feature vectors can be combined with information about
     the position of the ROIs to track the ROIs across imaging sessions/planes.


ROInet model data
#################

Below are links to the current ROInet models. These models are trained on data
from many labs from around the world. They take in images of ROIs and output a
feature vector describing each ROI. See documentation on the `ROInet_embedder
<https://roicat.readthedocs.io/en/latest/roicat.html#roicat.ROInet.ROInet_embedder>`__
for more information on how to use these models.

.. include:: ../helpers/ROInet_links.txt