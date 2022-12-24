import scipy.sparse
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

from .. import helpers


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
        n_workers=-1,
        frame_height=512,
        frame_width=1024,
        block_height=100,
        block_width=100,
        overlapping_width_Multiplier=0.0,
        algorithm_nearestNeigbors_spatialFootprints='brute',
        verbose=True,
        **kwargs_nearestNeigbors_spatialFootprints
    ):
        """
        Initialize the class.
        Makes blocks of the field of view so that subsequent 
         computations can be done blockwise.

        Args:
            n_workers (int):
                The number of workers to use for the computations.
                Set to -1 to use all available cpu cores.
                Used for spatial footprint manahattan distance computation,
                 computing hashes of cluster idx, and computing linkages.
            algorithm_nearestNeigbors_spatialFootprints (str):
                The algorithm to use for the nearest neighbors computation.
                See sklearn.neighbors.NearestNeighbors for more information.
            **kwargs_nearestNeigbors_spatialFootprints (dict):
                The keyword arguments to use for the nearest neighbors.
                Optional.
                See sklearn.neighbors.NearestNeighbors for more information.
        """
        self._algo_sf = algorithm_nearestNeigbors_spatialFootprints
        self._kwargs_sf = kwargs_nearestNeigbors_spatialFootprints


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
        spatialFootprint_maskPower=1.0,
    ):
        """
        Computes:
            1. The similarity graph between ROIs. This generates
             self.s, which is the pairwise similarity matrix of 
             shape (n_ROIs, n_ROIs).
        Args:
            spatialFootprints (scipy.sparse.csr_matrix):
                The spatial footprints of the ROIs.
                Can be obtained from blurring.ROI_blurrer.ROIs_blurred
                 or data_importing.Data_suite2p.spatialFootprints.
            spatialFootprint_maskPower (float):
                The power to use for the spatial footprint mask. Lower
                 values will make masks more binary looking for distance
                 computation.
            features_NN (torch.Tensor):
                The output latents from the roinet neural network.
                Can be obtained from ROInet.ROInet_embedder.latents
            features_SWT (torch.Tensor):
                The output latents from the scattering wavelet transform.
                Can be obtained from scatteringWaveletTransform.SWT.latents
            ROI_session_bool (torch.Tensor):
                The boolean array indicating which ROIs (across all sessions)
                 belong to each session. shape (n_ROIs total, n_sessions)
            spatialFootprint_maskPower (float):
                The power to raise the spatial footprint mask to. Use 1.0 for
                 no change to the masks, low values (e.g. 0.5) to make the masks
                 more binary looking, and high values (e.g. 2.0) to make the
                 pairwise similarities highly dependent on the relative intensities
                 of the pixels in each mask.            
        """

        self._n_sessions = ROI_session_bool.shape[1]
        self._sf_maskPower = spatialFootprint_maskPower

        self.sf_cat = scipy.sparse.vstack(spatialFootprints).tocsr()
        n_roi = self.sf_cat.shape[0]


        s_sf_all, s_NN_all, s_SWT_all, s_sesh_all, idxROI_block_all = [], [], [], [], []

        self.s_SWT = scipy.sparse.csr_matrix((n_roi, n_roi))
        self.d = scipy.sparse.csr_matrix((n_roi, n_roi))

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
        # return s_sf_all, s_NN_all, s_SWT_all, s_sesh_all, idxROI_block_all

        print('Joining blocks into full similarity matrices...') if self._verbose else None
        # self.s_sf = helpers.merge_sparse_arrays(s_sf_all, idxROI_block_all, (n_roi, n_roi), remove_redundant=True).tocsr()
        # self.s_NN = helpers.merge_sparse_arrays(s_NN_all, idxROI_block_all, (n_roi, n_roi), remove_redundant=True).tocsr()
        # self.s_SWT = helpers.merge_sparse_arrays(s_SWT_all, idxROI_block_all, (n_roi, n_roi), remove_redundant=True).tocsr()
        # self.s_sesh = helpers.merge_sparse_arrays(s_sesh_all, idxROI_block_all, (n_roi, n_roi), remove_redundant=True).tocsr()

        def merge_sparse_arrays(s_list, idx_list, shape, shift_val=None, n_batches=1):
            def csr_to_coo_idxd(s_csr, idx, shift_val, shape):
                s_coo = s_csr.tocoo()
                return scipy.sparse.coo_matrix(
                    (s_coo.data + shift_val, (idx[s_coo.row], idx[s_coo.col])),
                    shape=shape
                )
            if shift_val == None:
                shift_val = min([s.min() for s in s_list]) + 1
            
            s_flat = scipy.sparse.vstack([
                csr_to_coo_idxd(s, idx, shift_val, shape).reshape(1,-1).tocsr() for s, idx in tqdm(zip(s_list, idx_list))
            ]).tocsr()
            s_flat = scipy.sparse.vstack([s.max(1) for s in helpers.make_batches(helpers.scipy_sparse_csr_with_length(s_flat.T), num_batches=n_batches)]).T
            
            s_merged = s_flat.reshape(shape)
            s_merged.data = s_merged.data - shift_val
            return s_merged  ## the max operation is why it's so slow

        print('Joining s_sf...') if self._verbose else None
        self.s_sf = merge_sparse_arrays(s_sf_all, idxROI_block_all, (n_roi, n_roi), n_batches=10).tocsr()
        print('Joining s_NN...') if self._verbose else None
        self.s_NN = merge_sparse_arrays(s_NN_all, idxROI_block_all, (n_roi, n_roi), n_batches=10).tocsr()
        print('Joining s_SWT...') if self._verbose else None
        self.s_SWT = merge_sparse_arrays(s_SWT_all, idxROI_block_all, (n_roi, n_roi), n_batches=10).tocsr()
        print('Joining s_sesh...') if self._verbose else None
        self.s_sesh = merge_sparse_arrays(s_sesh_all, idxROI_block_all, (n_roi, n_roi), n_batches=10).tocsr()


        # self.s_sf = scipy.sparse.lil_matrix((n_roi, n_roi))
        # self.s_NN = scipy.sparse.lil_matrix((n_roi, n_roi))
        # self.s_SWT = scipy.sparse.lil_matrix((n_roi, n_roi))
        # self.s_sesh = scipy.sparse.lil_matrix((n_roi, n_roi))
        # for ssf, snn, sswt, ss, idxROI_block in zip(s_sf_all, s_NN_all, s_SWT_all, s_sesh_all, idxROI_block_all):
        #     idx = np.meshgrid(idxROI_block, idxROI_block)
        #     self.s_sf[idx[0], idx[1]] = ssf
        #     self.s_NN[idx[0], idx[1]] = snn
        #     self.s_SWT[idx[0], idx[1]] = sswt
        #     self.s_sesh[idx[0], idx[1]] = ss
        # self.s_sf = self.s_sf.tocsr()
        # self.s_NN = self.s_NN.tocsr()
        # self.s_SWT = self.s_SWT.tocsr()
        # self.s_sesh = self.s_sesh.tocsr()

        return self.s_sf, self.s_NN, self.s_SWT, self.s_sesh

    
    # def compute_cluster_scores(
    #     self, 
    #     power_clusterSize=2, 
    #     power_clusterSilhouette=1.5
    # ):
    #     """
    #     Compute scores for each cluster. Score is intended to be
    #      related to the quality of a cluster.
    #     This is a somewhat arbitrary scoring system, but is useful when
    #      performing assignment/selection of clusters.

    #     Args:
    #         power_clusterSize (float): 
    #             power to raise the cluster size to
    #         power_clusterSilhouette (float):
    #             gain factor for exponentiating the cluster silhouette
    #              score
    #     """

    #     self.scores = torch.as_tensor((np.array(self.cluster_bool.sum(1)).squeeze()**power_clusterSize) * (10**(self.c_sil*power_clusterSilhouette)))
    #     return self.scores



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
            n_jobs=self._n_workers,
            **self._kwargs_sf
        ).fit(sf).kneighbors_graph(
            sf,
            n_neighbors=sf.shape[0],
            mode='distance'
        )

        s_sf = d_sf.copy()
        s_sf.data = 1 - s_sf.data
        s_sf.data[s_sf.data < 1e-5] = 0  ## Likely due to numerical errors, some values are < 0 and very small. Rectify to fix.
        s_sf[range(s_sf.shape[0]), range(s_sf.shape[0])] = 0
        s_sf.eliminate_zeros()
        
        features_NN_normd = torch.nn.functional.normalize(features_NN, dim=1)
        s_NN = torch.matmul(features_NN_normd, features_NN_normd.T) ## cosine similarity. ranges [0,1]
        s_NN[s_NN>(1-1e-5)] = 1.0
        s_NN[range(s_NN.shape[0]), range(s_NN.shape[0])] = 0
        
        features_SWT_normd = torch.nn.functional.normalize(features_SWT, dim=1)
        s_SWT = torch.matmul(features_SWT_normd, features_SWT_normd.T) ## cosine similarity. Normalized to [0,1]
        s_SWT[s_SWT < 0] = 0
        s_SWT[range(s_SWT.shape[0]), range(s_SWT.shape[0])] = 0

        session_bool = torch.as_tensor(ROI_session_bool, device='cpu', dtype=torch.float32)
        s_sesh = torch.logical_not((session_bool @ session_bool.T).type(torch.bool))

        # s_sf = s_sf.multiply(s_sesh.numpy())
        # s_sf.eliminate_zeros()
        # # s_NN = s_NN * s_sesh
        # # s_SWT = s_SWT * s_sesh

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
        device='cpu',
        verbose=True,
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
            features_NN (torch.Tensor):
                The output latent embeddings of the NN model.
                shape (n_ROIs total, n_features)
            features_SWT (torch.Tensor):
                The output latent embeddings of the SWT model.
                shape (n_ROIs total, n_features)
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
            device (str):
                The device to use for the similarity computations.
                Output will still be on CPU.

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

        print('Finding k-range of center of mass distance neighbors for each ROI...')
        coms = np.vstack(centers_of_mass) if isinstance(centers_of_mass, list) else centers_of_mass

        ## first get the indices of 'different' ROIs for each ROI.
        ##  'different here means they are more than the k-th nearest
        ##  neighbor based on centroid distance.
        idx_diff, _ = get_idx_in_kRange(
            X=coms,
            k_max=k_max,
            k_min=k_min,
            algo_kNN=algo_NN,
            n_workers=self._n_workers,
        )

        ## calculate similarity scores for each ROI against the 
        ##  'different' ROIs
        print('Normalizing Neural Network similarity scores...') if verbose else None
        if features_NN is not None:
            s_NN_diff = cosine_similarity_customIdx(features_NN.to(device), idx_diff)
            mus_NN_diff = s_NN_diff.mean(1).to('cpu').numpy()
            stds_NN_diff = s_NN_diff.std(1).to('cpu').numpy()
            
            self.s_NN_z = self.s_NN.copy().tocoo()
            self.s_NN_z.data = ((self.s_NN_z.data - mus_NN_diff[self.s_NN_z.row]) / stds_NN_diff[self.s_NN_z.row])
            self.s_NN_z = self.s_NN_z.tocsr()
        
        print('Normalizing SWT similarity scores...') if verbose else None
        if features_SWT is not None:
            s_SWT_diff = cosine_similarity_customIdx(features_SWT.to(device), idx_diff)
            mus_SWT_diff = s_SWT_diff.mean(1).to('cpu').numpy()
            stds_SWT_diff = s_SWT_diff.std(1).to('cpu').numpy()

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
        overlapping_width_Multiplier=0.0,
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
    algo_kNN='brute',
    n_workers=-1,
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
        n_jobs=n_workers,
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




def compute_cluster_similarity_graph(
    labels,
    s,
    sf_cat,
    idxPixels_block,
    blocks,
    locality=1,
    cluster_similarity_reduction_intra='mean',
    cluster_similarity_reduction_inter='max',
    n_workers=None,
):
    """
    Computers the similarity graph between clusters.
    Similarly to 'compute_similarity_blockwise', this function 
        operates over blocks, but computes the similarity between
        clusters as opposed to ROIs.
    
    Args:
        cluster_similarity_reduction_intra (str):
            The method to use for reducing the intra-cluster similarity.
        cluster_similarity_reduction_inter (str):
            The method to use for reducing the inter-cluster similarity.
        cluster_silhouette_reduction_intra (str):
            The method to use for reducing the intra-cluster silhouette.
        cluster_silhouette_reduction_inter (str):
            The method to use for reducing the inter-cluster silhouette.
    Attributes set:
        self.n_clusters (int):
            The number of clusters.
        self.sf_clusters (scipy.sparse.csr_matrix):
            The spatial fooprints of the sum of the rois in each cluster
                (summed over ROIs).
        self.c_sim (scipy.sparse.csr_matrix):
            The similarity matrix between clusters.
        self.c_sil (np.ndarray):
            The silhouette score for each cluster
        self.s_local (scipy.sparse.csr_matrix):
            The 'local' version of self.s, where s_local = s**locality.
    """
    import sparse

    # self._cluster_similarity_reduction_intra_method = cluster_similarity_reduction_intra
    # self._cluster_similarity_reduction_inter_method = cluster_similarity_reduction_inter

    # self._cluster_silhouette_reduction_intra_method = cluster_silhouette_reduction_intra
    # self._cluster_silhouette_reduction_inter_method = cluster_silhouette_reduction_inter

    if n_workers is None:
        n_workers = mp.cpu_count()

    # self.n_clusters = len(self.cluster_idx)
    n_clusters = len(np.unique(labels))

    ## make a sparse matrix of the spatial footprints of the sum of each cluster
    # print('Starting: Making cluster spatial footprints') if self._verbose else None
    spatialFootprints_coo = sparse.COO(sf_cat)
    # self.clusterBool_coo = sparse.COO(self.cluster_bool)
    clusterBool = np.stack([labels==l for l in np.unique(labels)], axis=0)
    clusterBool_coo = sparse.COO(clusterBool)
    batch_size = int(max(1e8 // spatialFootprints_coo.shape[0], 1000))

    def helper_make_sfClusters(
            cb_s_batch
        ):
        """
        Helper function to make a cluster spatial footprint
         using all the roi spatial footprints. and a boolean array
         of which rois belong to which cluster.
        """
        return (spatialFootprints_coo[None,:,:] * cb_s_batch).sum(axis=1)

    def reduction_inter(x, sizes_clusters, method='max'):
        if method == 'max':
            return x.max(axis=(2,3))
        elif method == 'mean':
            return x.sum(axis=(2,3)) / (sizes_clusters[:,None] * sizes_clusters[None,:])

    def reduction_intra(x, sizes_clusters, method='min'):
        if method == 'min':
            x.fill_value = np.inf
            return x.min(axis=(1,2))
        elif method == 'mean':
            return x.sum(axis=(1,2)) / (sizes_clusters * (sizes_clusters-1))
    def helper_compute_cluster_similarity_batch(
        i_block, 
        ):
        """
        Helper function to compute the similarity between clusters within
         a single block.
        
        Updates attribute: self.c_sim
        """
        cBool = sparse.COO(clusterBool[idxClusters_block[i_block]])
        sizes_clusters = cBool.sum(1)

        cs_inter = (s_local[None,None,:,:] * cBool[:, None, :, None]) * cBool[None, :, None, :]  ## arranges similarities between every roi ACROSS every pair of clusters. shape (n_clusters, n_clusters, n_ROI, n_ROI)
        c_block = reduction_inter(
            x=cs_inter, 
            sizes_clusters=sizes_clusters, 
            method=cluster_similarity_reduction_inter,
        ).todense()  ## compute the reduction of the cs_inter array along the ROI dimensions

        cs_intra = (s_local[None,:,:] * cBool[:, :, None]) * cBool[:, None, :]  ## arranges similarities between every roi WITHIN each cluster. shape (n_clusters, n_ROI, n_ROI)
        c_block[range(c_block.shape[0]), range(c_block.shape[0])] = reduction_intra(
            cs_intra, 
            sizes_clusters=sizes_clusters,
            method=cluster_similarity_reduction_intra,
        ).todense()  ## compute the reduction of the cs_intra array along the ROI dimensions

        c_block = np.maximum(c_block, c_block.T)  # force symmetry

        idx = np.meshgrid(idxClusters_block[i_block], idxClusters_block[i_block])
        c_sim[idx[0], idx[1]] = c_block
        return c_block


    ## make images of each cluster (so that we can determine which ones are in which block)
    sf_clusters = sparse.concatenate(
        helpers.simple_multithreading(
            helper_make_sfClusters,
            [helpers.make_batches(clusterBool_coo[:,:,None], batch_size=batch_size)],
            workers=n_workers
        )
    ).tocsr()
    # print('Completed: Making cluster spatial footprints') if self._verbose else None


    # print('Starting: Computing cluster similarities') if self._verbose else None
    c_sim = scipy.sparse.lil_matrix((n_clusters, n_clusters)) # preallocate a sparse matrix for the cluster similarity matrix
    s_local = sparse.COO(s.power(locality))
    idxClusters_block = [np.where(sf_clusters[:, idx_pixels].sum(1) > 0)[0] for idx_pixels in idxPixels_block]  ## find indices of the clusters that have at least one non-zero pixel in the block

    ## compute the similarity between clusters. self.c_sim is updated from within the helper function
    c = helpers.simple_multithreading(
            helper_compute_cluster_similarity_batch,
            [np.arange(len(blocks))],
            workers=n_workers
        )
        # [self._helper_compute_cluster_similarity_batch(i_block) for i_block in np.arange(len(self.blocks))]
        # print('Completed: Computing cluster similarities') if self._verbose else None
    for i_block in np.arange(len(blocks)):
        idx = np.meshgrid(idxClusters_block[i_block], idxClusters_block[i_block])
        c_sim[idx[0], idx[1]] = c[i_block]

    return c_sim.tocsr()