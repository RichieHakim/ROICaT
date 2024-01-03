Inputs and outputs
=================

Inputs
######

   * Suite2p output files: **stat.npy** and **ops.npy** only
   * Other data formats (e.g. CaImAn, custom ROIs, etc.): `Custom data importing notebook <https://github.com/RichieHakim/ROICaT/blob/main/notebooks/jupyter/other/demo_data_importing.ipynb>`__

Outputs
#######

   * **results.pkl** file: Python dictionary containing:
      * results = {
         "clusters": {
             "labels": labels_squeezed, **## Unique Cluster IDs (UCIDs) for each ROI**
             "labels_bySession": labels_bySession, **## UCIDs for each ROI, by session**
             "labels_bool": labels_bool, **## Sparse boolean matrix describing which ROIs are in which clusters. Rows are ROI indices, columns are UCIDs + 1.**
             "labels_bool_bySession": labels_bool_bySession, **## Same as labels_bool, but by session.**
             "labels_dict": labels_dict, **## Dictionary mapping UCIDs to ROI indices. Keys are UCIDs, values are lists of ROI indices.**
         },
         "ROIs": { **## Contains images of all ROIs, by session**
             "ROIs_aligned": aligner.ROIs_aligned,
             "ROIs_raw": data.spatialFootprints,
             "frame_height": data.FOV_height,
             "frame_width": data.FOV_width,
             "idx_roi_session": np.where(data.session_bool)[1],
             "n_sessions": data.n_sessions,
         },
         "input_data": { **## Contains input file paths.**
             "paths_stat": data.paths_stat,
             "paths_ops": data.paths_ops,
         },
         "quality_metrics": clusterer.quality_metrics if hasattr(clusterer, 'quality_metrics') else None,
     }