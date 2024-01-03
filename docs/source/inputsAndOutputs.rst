Inputs and Outputs
==================

Inputs
######

   - **Suite2p output files:** ``stat.npy`` and ``ops.npy`` files only.
   - **Other data formats:** Support for formats like CaImAn, custom ROIs, etc., can be facilitated through a custom data importing notebook found `here <https://github.com/RichieHakim/ROICaT/blob/main/notebooks/jupyter/other/demo_data_importing.ipynb>`_.

.. Outputs
.. -------

.. The outputs of ROICaT are encapsulated in a ``results.pkl`` file, which is a Python dictionary containing the following fields:

.. Clusters
.. ~~~~~~~~

.. - **labels:** Unique Cluster IDs (aka **'UCIDs'**) for each ROI. These are integer labels indicating which cluster each ROI belongs to. ``-1`` indicates an ROI that was not clustered. Array of shape: ``(n_ROIs_total,)``.
.. - **labels_bySession:** UCIDs for each ROI, by session. List of length ``n_sessions``, where each element is an array of shape ``(n_ROIs_session,)``.
.. - **labels_bool:** Sparse boolean matrix describing which ROIs are in which clusters. Rows are ROI indices, columns are UCIDs + 1.
.. - **labels_bool_bySession:** Same as ``labels_bool``, but by session.
.. - **labels_dict:** Dictionary mapping UCIDs to ROI indices. Keys are UCIDs, values are lists of ROI indices.

.. ROIs
.. ~~~~

.. - **ROIs_aligned:** Images of all ROIs, aligned by session.
.. - **ROIs_raw:** Raw spatial footprints of the ROIs.
.. - **frame_height, frame_width:** Dimensions of the Field of View (FOV).
.. - **idx_roi_session:** Session-wise ROI indices.
.. - **n_sessions:** Number of sessions.

.. Input Data
.. ~~~~~~~~~~

.. - **paths_stat, paths_ops:** File paths of the input data.

.. Quality Metrics
.. ~~~~~~~~~~~~~~~

.. The ``quality_metrics`` are crucial for assessing the effectiveness of the clustering process. They are calculated using the ``cluster_quality_metrics`` function.

.. cs_min
.. ^^^^^^

.. - **Definition:** Intra-cluster minimum similarity.
.. - **Context:** Reflects the least similarity within a cluster, indicating the lower bound of homogeneity within a cluster.
.. - **Shape:** (n_unique_ROIs,).
.. - **Image:** .. image:: /path/to/cs_min_image.png
..    :alt: cs_min

.. cs_max
.. ^^^^^^

.. - **Definition:** Intra-cluster maximum similarity.
.. - **Context:** Indicates the highest similarity within a cluster, showing the upper bound of cohesion.
.. - **Shape:** (n_unique_ROIs,).
.. - **Image:** .. image:: /path/to/cs_max_image.png
..    :alt: cs_max

.. cs_mean
.. ^^^^^^^

.. - **Definition:** Mean intra-cluster similarity.
.. - **Context:** Represents the average similarity within a cluster, offering a measure of overall cluster integrity.
.. - **Shape:** (n_unique_ROIs,).
.. - **Image:** .. image:: /path/to/cs_mean_image.png
..    :alt: cs_mean

.. cs_sil
.. ^^^^^^

.. - **Definition:** Cluster silhouette score.
.. - **Context:** A measure of how similar an ROI is to its own cluster compared to other clusters, which can be indicative of the appropriateness of the cluster assignment.
.. - **Shape:** (n_unique_ROIs,).
.. - **Image:** .. image:: /path/to/cs_sil_image.png
..    :alt: cs_sil

.. sample_sil
.. ^^^^^^^^^^

.. - **Definition:** Sample silhouette score.
.. - **Context:** Evaluates how well each ROI is clustered with its label, providing a perspective on the overall clustering quality.
.. - **Shape:** (n_ROIs_total,).
.. - **Image:** .. image:: /path/to/sample_sil_image.png
..    :alt: sample_sil
