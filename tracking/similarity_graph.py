import scipy.sparse
import scipy.cluster
import numpy as np
import torch
import sklearn
import sklearn.neighbors
from tqdm import tqdm
import matplotlib.pyplot as plt
# import sparse

# import gc
import multiprocessing as mp
# import time
# import copy

from . import helpers


class ROI_graph:
    """
    Class for:
     1. Building similarity and distance graphs between ROIs
      based on their features.
     2. Generating potential clusters of ROIs using linkage 
      clustering.
     3. Building a similarity graph between clusters of ROIs.
     4. Computing silhouette scores for each potential cluster.

    To accelerate computation and reduce memory usage, some of the
     computations are performed on 'blocks' of the full field of
     view.
    
    RH 2022
    """
    def __init__(
        self,
        device='cpu',
        n_workers=1,
        spatialFootprint_maskPower=0.8,
        frame_height=512,
        frame_width=1024,
        block_height=100,
        block_width=100,
        overlapping_width_Multiplier=0.2,
        algorithm_nearestNeigbors_spatialFootprints='brute',
        n_neighbors_nearestNeighbors_spatialFootprints='full',
        locality=1,
        verbose=True,
        **kwargs_nearestNeigbors_spatialFootprints
    ):
        """
        Initialize the class.
        Makes blocks of the field of view so that subsequent 
         computations can be done blockwise.

        Args:
            device (str):
                The device to use for the computations.
                Recommended to use 'cpu' for since the matmul operations
                 are relatively small.
            n_workers (int):
                The number of workers to use for the computations.
                Set to -1 to use all available cpu cores.
                Used for spatial footprint manahattan distance computation,
                 computing hashes of cluster idx, and computing linkages.
            spatialFootprint_maskPower (float):
                The power to use for the spatial footprint mask. Lower
                 values will make masks more binary looking for distance
                 computation.
            algorithm_nearestNeigbors_spatialFootprints (str):
                The algorithm to use for the nearest neighbors computation.
                See sklearn.neighbors.NearestNeighbors for more information.
            n_neighbors_nearestNeighbors_spatialFootprints (int or str):
                The number of neighbors to use for the nearest neighbors.
                Set to 'full' to use all available neighbors.
            locality (float):
                Value to use as an exponent for the cluster similarity calculations
                self.s remains unchanged, but self.c is computed using self.s**locality
            **kwargs_nearestNeigbors_spatialFootprints (dict):
                The keyword arguments to use for the nearest neighbors.
                Optional.
                See sklearn.neighbors.NearestNeighbors for more information.
        """
        self._device = device
        self._sf_maskPower = spatialFootprint_maskPower
        self._algo_sf = algorithm_nearestNeigbors_spatialFootprints
        self._nn_sf = n_neighbors_nearestNeighbors_spatialFootprints
        self._kwargs_sf = kwargs_nearestNeigbors_spatialFootprints

        self._locality = locality

        self._verbose = verbose

        self._n_workers = mp.cpu_count() if n_workers == -1 else n_workers

        self._frame_height = frame_height
        self._frame_width = frame_width

        self.blocks, (self._centers_y, self._centers_x) = self._make_block_batches(
            frame_height=frame_height,
            frame_width=frame_width,
            block_height=block_height,
            block_width=block_width,
            overlapping_width_Multiplier=overlapping_width_Multiplier,
            clamp_blocks_to_frame=True,
        )

        self.idxPixels_block = []
        for block in self.blocks:
            idx_tmp = np.zeros((self._frame_height, self._frame_width), dtype=np.bool8)
            idx_tmp[block[0][0]:block[0][1], block[1][0]:block[1][1]] = True
            idx_tmp = np.where(idx_tmp.reshape(-1))[0]
            self.idxPixels_block.append(idx_tmp)



    def compute_similarity_blockwise(
        self,
        spatialFootprints,
        features_NN,
        features_SWT,
        ROI_session_bool,
        linkage_methods=['single', 'complete', 'ward', 'average'],
        linkage_distances=[0.1, 0.2, 0.4, 0.8],
        min_cluster_size=2,
        max_cluster_size=None,
        batch_size_hashing=100,
    ):
        """
        Computes:
            1. The similarity graph between ROIs. This generates
             self.s, which is the pairwise similarity matrix of 
             shape (n_ROIs, n_ROIs).
            2. Generates clusters. This generates self.cluster_bool,
             and self.cluster_idx. These describe the which ROIs 
             are in which clusters.

        Args:
            spatialFootprints (scipy.sparse.csr_matrix):
                The spatial footprints of the ROIs.
                Can be obtained from blurring.ROI_blurrer.ROIs_blurred
                 or data_importing.Data_suite2p.spatialFootprints.
            features_NN (torch.Tensor):
                The output latents from the roinet neural network.
                Can be obtained from ROInet.ROInet_embedder.latents
            features_SWT (torch.Tensor):
                The output latents from the scattering wavelet transform.
                Can be obtained from scatteringWaveletTransform.SWT.latents
            ROI_session_bool (torch.Tensor):
                The boolean array indicating which ROIs (across all sessions)
                 belong to each session. shape (n_ROIs total, n_sessions)
            linkage_methods (list of str):
                The linkage methods to use for the linkage clustering.
                See scipy.cluster.hierarchy.linkage for more information.
            linkage_distances (list of float):
                The distances to use for the linkage clustering.
                Using more will result in higher resolution results, but slower
                 computation.
                See scipy.cluster.hierarchy.linkage for more information.
            min_cluster_size (int):
                The minimum size of a cluster to consider.
            max_cluster_size (int):
                The maximum size of a cluster to consider.
                If None, all clusters with size <= n_sessions are considered.
            batch_size_hashing (int):
                The number of ROIs to hash at a time.
        """

        self._n_sessions = ROI_session_bool.shape[1]

        self._linkage_methods = linkage_methods
        self._linkage_distances = linkage_distances
        self._min_cluster_size = min_cluster_size
        self._max_cluster_size = max_cluster_size
        self._batch_size_hashing = batch_size_hashing

        self.sf_cat = scipy.sparse.vstack(spatialFootprints)
        n_roi = self.sf_cat.shape[0]


        s_sf_all, s_NN_all, s_SWT_all, s_sesh_all, idxROI_block_all = [], [], [], [], []

        self.s_SWT = scipy.sparse.csr_matrix((n_roi, n_roi))
        s_empty = scipy.sparse.lil_matrix((n_roi, n_roi))
        self.d = scipy.sparse.csr_matrix((n_roi, n_roi))
        cluster_idx_all = []

        print('Computing pairwise similarity between ROIs...') if self._verbose else None
        for ii, block in tqdm(enumerate(self.blocks), total=len(self.blocks), mininterval=10):
            idxROI_block = np.where(self.sf_cat[:, self.idxPixels_block[ii]].sum(1) > 0)[0]
            
            ## Compute pairwise similarity matrix
            s_sf, s_NN, s_SWT, s_sesh = self._helper_compute_ROI_similarity_graph(
                spatialFootprints=self.sf_cat[idxROI_block].power(self._sf_maskPower),
                features_NN=features_NN[idxROI_block],
                features_SWT=features_SWT[idxROI_block],
                ROI_session_bool=ROI_session_bool[idxROI_block],
            )
            if s_sf is None: # If there are no ROIs in this block, s_block will be None, so we should skip the rest of the loop
                continue

            s_sf_all.append(s_sf)
            s_NN_all.append(s_NN)
            s_SWT_all.append(s_SWT)
            s_sesh_all.append(s_sesh)
            idxROI_block_all.append(idxROI_block)


        self.s_sf = scipy.sparse.lil_matrix((n_roi, n_roi))
        self.s_NN = scipy.sparse.lil_matrix((n_roi, n_roi))
        self.s_SWT = scipy.sparse.lil_matrix((n_roi, n_roi))
        self.s_sesh = scipy.sparse.lil_matrix((n_roi, n_roi))
        for ssf, snn, sswt, ss, idxROI_block in zip(s_sf_all, s_NN_all, s_SWT_all, s_sesh_all, idxROI_block_all):
            idx = np.meshgrid(idxROI_block, idxROI_block)
            self.s_sf[idx[0], idx[1]] = ssf
            self.s_NN[idx[0], idx[1]] = snn
            self.s_SWT[idx[0], idx[1]] = sswt
            self.s_sesh[idx[0], idx[1]] = ss
        self.s_sf = self.s_sf.tocsr()
        self.s_NN = self.s_NN.tocsr()
        self.s_SWT = self.s_SWT.tocsr()
        self.s_sesh = self.s_sesh.tocsr()

        return self.s_sf, self.s_NN, self.s_SWT, self.s_sesh

    
    def compute_cluster_scores(
        self, 
        power_clusterSize=2, 
        power_clusterSilhouette=1.5
    ):
        """
        Compute scores for each cluster. Score is intended to be
         related to the quality of a cluster.
        This is a somewhat arbitrary scoring system, but is useful when
         performing assignment/selection of clusters.

        Args:
            power_clusterSize (float): 
                power to raise the cluster size to
            power_clusterSilhouette (float):
                gain factor for exponentiating the cluster silhouette
                 score
        """

        self.scores = torch.as_tensor((np.array(self.cluster_bool.sum(1)).squeeze()**power_clusterSize) * (10**(self.c_sil*power_clusterSilhouette)))
        return self.scores



    def _helper_compute_ROI_similarity_graph(
        self,
        spatialFootprints,
        features_NN,
        features_SWT,
        ROI_session_bool,
    ):
        """
        Computes the similarity matrix between ROIs based on the 
         conjunction of the similarity matrices for different modes
         (like the NN embedding, the SWT embedding, and the spatial footprint
         overlap).

        Args:
            spatialFootprints (list of scipy.sparse.csr_matrix):
                The spatial footprints of the ROIs.
                list of shape (n_ROIs for each session, FOV height * FOV width)
            features_NN (torch.Tensor):
                The output latent embeddings of the NN model.
                shape (n_ROIs total, n_features)
            features_SWT (torch.Tensor):
                The output latent embeddings of the SWT model.
                shape (n_ROIs total, n_features)
            ROI_session_bool (np.ndarray):
                The boolean matrix indicating which ROIs belong to which session.
                shape (n_ROIs total, n_sessions)
        """
        ## if there are no ROIs in the block
        if spatialFootprints.shape[0] == 0:
            return None, None, None, None

        sf = scipy.sparse.vstack(spatialFootprints)
        sf = sf.power(self._sf_maskPower)
        sf = sf.multiply( 0.5 / sf.sum(1))
        sf = scipy.sparse.csr_matrix(sf)

        d_sf = sklearn.neighbors.NearestNeighbors(
            algorithm=self._algo_sf,
            n_neighbors=sf.shape[0],
            metric='manhattan',
            p=1,
            # n_jobs=self._n_workers,
            n_jobs=-1,
            **self._kwargs_sf
        ).fit(sf).kneighbors_graph(
            sf,
            n_neighbors=sf.shape[0],
            mode='distance'
        )

        s_sf = d_sf.copy()
        s_sf.data = 1 - s_sf.data
        s_sf.data[s_sf.data < 1e-5] = 0  ## Likely due to numerical errors, some values are < 0 and very small. Rectify to fix.
        s_sf.eliminate_zeros()
        
        features_NN_normd = torch.nn.functional.normalize(features_NN, dim=1)
        s_NN = torch.matmul(features_NN_normd, features_NN_normd.T) ## cosine similarity. ranges [0,1]
        s_NN[s_NN>(1-1e-5)] = 1.0
        
        features_SWT_normd = torch.nn.functional.normalize(features_SWT, dim=1)
        s_SWT = torch.matmul(features_SWT_normd, features_SWT_normd.T) ## cosine similarity. Normalized to [0,1]
        s_SWT[s_SWT < 0] = 0
        s_SWT[range(s_SWT.shape[0]), range(s_SWT.shape[0])] = 0

        session_bool = torch.as_tensor(ROI_session_bool, device='cpu', dtype=torch.float32)
        s_sesh = torch.logical_not((session_bool @ session_bool.T).type(torch.bool))

        s_sf = s_sf.multiply(s_sesh.numpy())
        s_sf.eliminate_zeros()
        # s_NN = s_NN * s_sesh
        # s_SWT = s_SWT * s_sesh

        s_sf = s_sf.maximum(s_sf.T)
        s_NN = torch.maximum(s_NN, s_NN.T)  # force symmetry
        s_SWT = torch.maximum(s_SWT, s_SWT.T)  # force symmetry
        
        s_NN  = helpers.sparse_mask(s_NN,  s_sf, do_safety_steps=True)
        s_SWT = helpers.sparse_mask(s_SWT, s_sf, do_safety_steps=True)
        s_sesh = helpers.sparse_mask(s_sesh, s_sf, do_safety_steps=True)

        return s_sf, s_NN, s_SWT, s_sesh

    def make_normalized_similarities(
        self,
        centers_of_mass,
        features_NN=None,
        features_SWT=None,
        k_max=3000,
        k_min=200,
        algo_NN='kd_tree',
    ):
        """
        Normalizes the similarity matrices (s_NN, s_SWT, but not s_sf)
         by z-scoring using the mean and std from the distributions of
         pairwise similarities between ROIs assumed to be 'different'.
         'Different' here is defined as ROIs that are spatiall distant
         from each other.

        Args:
            centers_of_mass (np.ndarray or list of np.ndarray):
                The centers of mass of the ROIs.
                shape (n_ROIs total, 2)
                or list of shape (n_ROIs for each session, 2)
            k_max (int):
                The maximum number of nearest neighbors to consider
                 for each ROI. This value will result in an intermediate
                 similarity matrix of shape (n_ROIs total, k_max) 
                 between each ROI and its k_max nearest neighbors.
                Based on centroid distance.
            k_min (int):
                The minimum number of nearest neighbors to consider
                 for each ROI. This value should be less than k_max, and
                 be chosen such that it is likely that any potential
                 'same' ROIs are within k_min nearest neighbors.
                Based on centroid distance.
            algo_NN (str):
                The algorithm to use for the nearest neighbor search.
                See sklearn.neighbors.NearestNeighbors for options.
                Can be 'kd_tree' or 'ball_tree' or 'brute'.
                'kd_tree' seems to be the fastest.
            features_NN (torch.Tensor):
                The output latent embeddings of the NN model.
                shape (n_ROIs total, n_features)
            features_SWT (torch.Tensor):
                The output latent embeddings of the SWT model.
                shape (n_ROIs total, n_features)

        Set Attributes:
            s_NN_z (scipy.sparse.csr_matrix):
                The z-scored similarity matrix between ROIs based on the
                 statistics of the NN embedding.
                shape (n_ROIs total, n_ROIs total)
                Note: This matrix is NOT symmetric; and therefore should
                 be treated as a directed graph.
            s_SWT_z (scipy.sparse.csr_matrix):
                The z-scored similarity matrix between ROIs based on the
                 statistics of the SWT embedding.
                shape (n_ROIs total, n_ROIs total)
                Note: This matrix is NOT symmetric; and therefore should
                 be treated as a directed graph.
        """
        k_max = min(k_max, self.s_NN.shape[0])

        coms = np.vstack(centers_of_mass) if isinstance(centers_of_mass, list) else centers_of_mass
        
        ## first get the indices of 'different' ROIs for each ROI.
        ##  'different here means they are more than the k-th nearest
        ##  neighbor based on centroid distance.
        idx_diff, _ = get_idx_in_kRange(
            X=coms,
            k_max=k_max,
            k_min=k_min,
            algo_kNN=algo_NN,
        )

        ## calculate similarity scores for each ROI against the 
        ##  'different' ROIs
        if features_NN is not None:
            s_NN_diff = cosine_similarity_customIdx(features_NN, idx_diff)
            mus_NN_diff = s_NN_diff.mean(1).numpy()
            stds_NN_diff = s_NN_diff.std(1).numpy()
            
            self.s_NN_z = self.s_NN.copy().tocoo()
            self.s_NN_z.data = ((self.s_NN_z.data - mus_NN_diff[self.s_NN_z.row]) / stds_NN_diff[self.s_NN_z.row])
            self.s_NN_z = self.s_NN_z.tocsr()

        if features_SWT is not None:
            s_SWT_diff = cosine_similarity_customIdx(features_SWT, idx_diff)
            mus_SWT_diff = s_SWT_diff.mean(1).numpy()
            stds_SWT_diff = s_SWT_diff.std(1).numpy()

            self.s_SWT_z = self.s_SWT.copy().tocoo()
            self.s_SWT_z.data = ((self.s_SWT_z.data - mus_SWT_diff[self.s_SWT_z.row]) / stds_SWT_diff[self.s_SWT_z.row])
            self.s_SWT_z = self.s_SWT_z.tocsr()
            


###########################
####### block stuff #######
###########################

    def _make_block_batches(
        self,
        frame_height=512,
        frame_width=1024,
        block_height=100, 
        block_width=100,
        overlapping_width_Multiplier=0.2,
        clamp_blocks_to_frame=True,
    ):     
        """
        Makes blocks. These blocks partition up the field of
         view into smaller sections. Computations in the module
         are often pairwise comparisons, so it is useful to
         restrict comparisons to smaller blocks.

        Args:
            frame_height (int):
                The height of the field of view.
            frame_width (int):
                The width of the field of view.
            block_height (int):
                The height of each block.
                This value will be adjusted slightly to tile cleanly
                 if clamp_blocks_to_frame is True.
            block_width (int):
                The width of each block.
                This value will be adjusted slightly to tile cleanly
                 if clamp_blocks_to_frame is True.
            overlapping_width_Multiplier (float):
                The fractional amount of overlap between blocks.
            clamp_blocks_to_frame (bool):
                If True, then edges of the blocks will be clamped to the
                 edges of the field of view.
        """
        # block prep
        block_height_half = block_height//2
        block_width_half = block_width//2
        
        if block_height is None:
            block_height = block_height * 1.0
            print(f'block height not specified. Using {block_height}')
        if block_width is None:
            block_width = block_width * 1.0
            print(f'block width not specified. Using {block_width}')
            
        block_height_half = block_height//2
        block_width_half = block_width//2
        
        # find centers of blocks
        n_blocks_x = np.ceil(frame_width / (block_width - (block_width*overlapping_width_Multiplier))).astype(np.int64)
        
        centers_x = np.linspace(
            start=block_width_half,
            stop=frame_width - block_width_half,
            num=n_blocks_x,
            endpoint=True
        )

        n_blocks_y = np.ceil(frame_height / (block_height - (block_height*overlapping_width_Multiplier))).astype(np.int64)
        centers_y = np.linspace(
            start=block_height_half,
            stop=frame_height - block_height_half,
            num=n_blocks_y,
            endpoint=True
        )
        
        # make blocks
        blocks = []
        for i_x in range(n_blocks_x):
            for i_y in range(n_blocks_y):
                blocks.append([
                    list(np.int64([centers_y[i_y] - block_height_half , centers_y[i_y] + block_height_half])),
                    list(np.int64([centers_x[i_x] - block_width_half , centers_x[i_x] + block_width_half]))
                ])
                                
        # clamp block to limits of frame
        if clamp_blocks_to_frame:
            for ii, block in enumerate(blocks):
                br_h = np.array(block[0]) # block range height
                br_w = np.array(block[1]) # block range width
                valid_h = (br_h>0) * (br_h<frame_height)
                valid_w = (br_w>0) * (br_w<frame_width)
                blocks[ii] = [
                    list( (br_h * valid_h) + (np.array([0, frame_height])*np.logical_not(valid_h)) ),
                    list( (br_w * valid_w) + (np.array([0, frame_width])*np.logical_not(valid_w)) ),            
                ]
            
        return blocks, (centers_y, centers_x)


    def visualize_blocks(
        self,
    ):
        """
        Displays the blocks over a field of view.
        """
        im = np.zeros((self._frame_height, self._frame_width, 3))
        for ii, block in enumerate(self.blocks):
            im[block[0][0]:block[0][1], block[1][0]:block[1][1], :] = ((np.random.rand(1)+0.5)/2)
        plt.figure()
        plt.imshow(im, vmin=0, vmax=1)


def get_idx_in_kRange(
    X,
    k_max=3000,
    k_min=100,
    algo_kNN='brute'
):
    import sklearn
    import scipy.sparse
    import numpy as np

    if (k_max - k_min) < 1000:
        f"RH Warning: Difference betwee 'k_max' and 'k_min' is small. This different is the total number of comparisons. Fitting distributions may struggle. Consider increasing difference to 1000-9000 if posisble."
    assert k_max > k_min, f"'k_max' must be greater than 'k_min'"
    
    ## note that this is approximate (using kd_tree for algo bc it is the fastest of the available methods for large datasets)
    d = sklearn.neighbors.NearestNeighbors(
        algorithm=algo_kNN,
        n_neighbors=k_max,
        metric='euclidean',
        p=2,
        # n_jobs=self._n_workers,
        n_jobs=-1,
    #     **self._kwargs_sf
    ).fit(X).kneighbors_graph(
        X,
        n_neighbors=k_max,
        mode='distance'
    ).tocoo()

    idx_nz = np.stack((d.row, d.col), axis=0).reshape(2, d.shape[0], k_max)  ## get indices of the non-zero values in the distance graph
    idx_topk = np.argpartition(np.array(d.data).reshape(d.shape[0], k_max), kth=k_min, axis=1)  ## partition the non-zero values of the distance graph at the k_min-th value
#     idx_topk = np.argpartition(d.A, kth=k_min, axis=1)  ## partition the distance graph at the k_min-th value
    mesh_i, mesh_j = np.meshgrid(np.arange(k_max), np.arange(d.shape[0]))  ## prep a meshgrid of the correct output size. Will only use the row idx since the column idx will be replaced with idx_topk
    idx_diff = np.array(idx_nz.data).reshape(2, d.shape[0], k_max)[:, mesh_j[:, k_min:], idx_topk[:, k_min:]][1]  ## put it all together. Get kRange  between k_min and k_max
    return idx_diff, d

def cosine_similarity_customIdx(features, idx):
    f = torch.nn.functional.normalize(features, dim=1)
    out = torch.stack([f[ii] @ f[idx[ii]].T for ii in tqdm(range(f.shape[0]))], dim=0)
    return out
