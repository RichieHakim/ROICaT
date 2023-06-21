How to use ROICaT
=================

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
   * `Custom data importing notebook <https://github.com/RichieHakim/ROICaT/blob/main/notebooks/jupyter/other/demo_custom_data_importing.ipynb>`__
   * Train a new ROInet model using the provided Jupyter Notebook [TODO: link]. 
   * Use the API to integrate ROICaT functions into your own code: `Documentation <https://roicat.readthedocs.io/en/latest/>`__.

General workflow:
#################

   * **Pass ROIs through ROInet:** Images of the ROIs are passed through a
     neural network and outputs a feature vector for each image describing what
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

OSF.io links to ROInet versions:

* ROInet_tracking:
    * Info: This version does not include occlusions or large
      affine transformations.
    * Link: https://osf.io/x3fd2/download
    * Hash (MD5 hex): 7a5fb8ad94b110037785a46b9463ea94
* ROInet_classification:
    * Info: This version includes occlusions and large affine
      transformations.
    * Link: https://osf.io/c8m3b/download
    * Hash (MD5 hex): 357a8d9b630ec79f3e015d0056a4c2d5