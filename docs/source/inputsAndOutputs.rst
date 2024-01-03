Inputs and Outputs
==================

Inputs
######

- **Suite2p output files:** ``stat.npy`` and ``ops.npy`` files only.
- **Other data formats:** Support for formats like CaImAn, custom ROIs, etc., can be facilitated through a custom data importing notebook found `here <https://github.com/RichieHakim/ROICaT/blob/main/notebooks/jupyter/other/demo_data_importing.ipynb>`_.

Outputs
#######

The outputs of ROICaT are encapsulated in a ``results.pkl`` file, which is a Python dictionary containing the following fields:

Clusters
~~~~~~~~

- **labels:** Unique Cluster IDs (aka **'UCIDs'**) for each ROI. These are integer labels indicating which cluster each ROI belongs to. ``-1`` indicates an ROI that was not clustered. Array of shape: ``(n_ROIs_total,)``.
- **labels_bySession:** UCIDs for each ROI, by session. List of length ``n_sessions``, where each element is an array of shape ``(n_ROIs_session,)``.
- **labels_bool:** Sparse boolean matrix describing which ROIs are in which clusters. Rows are ROI indices, columns are UCIDs + 1.
- **labels_bool_bySession:** Same as ``labels_bool``, but by session.
- **labels_dict:** Dictionary mapping UCIDs to ROI indices. Keys are UCIDs, values are lists of ROI indices.

ROIs
~~~~

- **ROIs_aligned:** Images of all ROIs, aligned by session.
- **ROIs_raw:** Raw spatial footprints of the ROIs.
- **frame_height, frame_width:** Dimensions of the Field of View (FOV).
- **idx_roi_session:** Session-wise ROI indices.
- **n_sessions:** Number of sessions.

Quality Metrics
~~~~~~~~~~~~~~~

- **cs_min:** Intra-cluster minimum similarity. Defined as the lowest pairwise similarity within a cluster. *shape:* (n_clusters,).

.. image:: ../media/cluster_quality_metric_images/cs_min.png
   :align: right
   :width: 100
   :alt: cs_min

|

- **cs_max:** Intra-cluster maximum similarity. Defined as the highest similarity within a cluster. *shape:* (n_clusters,).

.. image:: ../media/cluster_quality_metric_images/cs_max.png
   :align: right
   :width: 100
   :alt: cs_max

|

- **cs_mean:** Mean intra-cluster similarity. Defined as the average similarity within a cluster. *shape:* (n_clusters,).

.. image:: ../media/cluster_quality_metric_images/cs_mean.png
   :align: right
   :width: 100
   :alt: cs_mean

|

- **cs_sil:** Cluster silhouette score. A measure of how similar an ROI is to its own cluster compared to other clusters, which can be indicative of the appropriateness of the cluster assignment. Defined as ``(intra - inter) / np.maximum(intra, inter)`` where ``intra=cs_intra_mean`` and ``inter=cs_inter_maxOfMaxes``. *shape:* (n_clusters,).

.. image:: ../media/cluster_quality_metric_images/cs_sil.png
   :align: right
   :width: 100
   :alt: cs_sil

|

- **sample_sil:** Sample silhouette score. A measure of how well each ROI is clustered with its label, providing a perspective on the overall clustering quality. Defined using ``sklearn.metrics.silhouette_score``. *shape:* (n_ROIs_total,).

.. image:: ../media/cluster_quality_metric_images/sample_sil.png
   :align: right
   :width: 100
   :alt: sample_sil

|

