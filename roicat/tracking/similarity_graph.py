import multiprocessing as mp
from typing import Tuple, Union, List, Optional, Dict, Any, Callable

import scipy.sparse
import numpy as np
import torch
import sklearn
import sklearn.neighbors
from tqdm import tqdm
import matplotlib.pyplot as plt
import sparse

from .. import helpers, util

class ROI_graph(util.ROICaT_Module):
    """
    Class for building similarity and distance graphs between Regions of
    Interest (ROIs) based on their features, generating potential clusters of
    ROIs using linkage clustering, building a similarity graph between clusters
    of ROIs, and computing silhouette scores for each potential cluster. The
    computations are performed on 'blocks' of the full field of view to
    accelerate computation and reduce memory usage. 
    RH 2022

    Args:
        n_workers (int):
            The number of workers to use for the computations. If -1, all
            available cpu cores will be used. (Default is ``-1``)
        frame_height (int):
            The height of the frame. (Default is ``512``)
        frame_width (int):
            The width of the frame. (Default is ``1024``)
        block_height (int):
            The height of the block. (Default is ``100``)
        block_width (int):
            The width of the block. (Default is ``100``)
        overlapping_width_Multiplier (float):
            The multiplier for the overlapping width. (Default is ``0.0``)
        algorithm_nearestNeigbors_spatialFootprints (str):
            The algorithm to use for the nearest neighbors computation. See
            sklearn.neighbors.NearestNeighbors for more information. (Default is
            ``'brute'``)
        verbose (bool):
            If set to ``True``, outputs will be verbose. (Default is ``True``)
        **kwargs_nearestNeigbors_spatialFootprints (dict):
            The keyword arguments to use for the nearest neighbors. See
            sklearn.neighbors.NearestNeighbors for more information. (Optional)
            
    Attributes:
        s_sf (scipy.sparse.csr_matrix): 
            Pairwise similarity matrix based on spatial footprints.
        s_NN (scipy.sparse.csr_matrix): 
            Pairwise similarity matrix based on Neural Network features.
        s_SWT (scipy.sparse.csr_matrix): 
            Pairwise similarity matrix based on Scattering Wavelet Transform.
        s_sesh (scipy.sparse.csr_matrix): 
            Pairwise similarity matrix based on which session the ROIs belong to.
    """
    def __init__(
        self,
        n_workers: int = -1,
        frame_height: int = 512,
        frame_width: int = 1024,
        block_height: int = 100,
        block_width: int = 100,
        overlapping_width_Multiplier: float = 0.0,
        algorithm_nearestNeigbors_spatialFootprints: str = 'brute',
        verbose: bool = True,
        kwargs_nearestNeigbors_spatialFootprints: dict = {},
    ):
        """
        Initializes the ROI_graph class with the given parameters.
        """
        ## Imports
        super().__init__()

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
            idx_tmp = np.zeros((self._frame_height, self._frame_width), dtype=np.bool_)
            idx_tmp[block[0][0]:block[0][1], block[1][0]:block[1][1]] = True
            idx_tmp = np.where(idx_tmp.reshape(-1))[0]
            self.idxPixels_block.append(idx_tmp)

    def compute_similarity_blockwise(
        self,
        spatialFootprints: scipy.sparse.csr_matrix,
        features_NN: torch.Tensor,
        features_SWT: torch.Tensor,
        ROI_session_bool: torch.Tensor,
        spatialFootprint_maskPower: float = 1.0,
    ) -> None:
        """
        Computes the similarity graph between ROIs and updates the instance
        attributes: ``s_sf``, ``s_NN``, ``s_SWT``, ``s_sesh``.

        Args:
            spatialFootprints (scipy.sparse.csr_matrix): 
                The spatial footprints of the ROIs. Can be obtained from
                ``blurring.ROI_blurrer.ROIs_blurred`` or
                ``data_importing.Data_suite2p.spatialFootprints``.
            features_NN (torch.Tensor): 
                The output latents from the roinet neural network. Can be
                obtained from ``ROInet.ROInet_embedder.latents``.
            features_SWT (torch.Tensor): 
                The output latents from the scattering wavelet transform. Can be
                obtained from ``scatteringWaveletTransform.SWT.latents``.
            ROI_session_bool (torch.Tensor): 
                The boolean array indicating which ROIs (across all sessions)
                belong to each session. shape: *(n_ROIs total, n_sessions)*.
            spatialFootprint_maskPower (float): 
                The power to raise the spatial footprint mask to. Use 1.0 for no
                change to the masks, low values (e.g., 0.5) to make the masks
                more binary looking, and high values (e.g., 2.0) to make the
                pairwise similarities highly dependent on the relative
                intensities of the pixels in each mask. (Default is ``1.0``)

        Returns:
            (tuple): tuple containing:
                s_sf (scipy.sparse.csr_matrix): 
                    Pairwise similarity matrix based on spatial footprints.
                s_NN (scipy.sparse.csr_matrix): 
                    Pairwise similarity matrix based on Neural Network features.
                s_SWT (scipy.sparse.csr_matrix): 
                    Pairwise similarity matrix based on Scattering Wavelet Transform.
                s_sesh (scipy.sparse.csr_matrix): 
                    Pairwise similarity matrix based on which session the ROIs belong to.
        """
        self._n_sessions = ROI_session_bool.shape[1]
        self._sf_maskPower = spatialFootprint_maskPower

        self.sf_cat = scipy.sparse.vstack(spatialFootprints).tocsr()
        n_roi = self.sf_cat.shape[0]


        s_sf_all, s_NN_all, s_SWT_all, s_sesh_all, idxROI_block_all = [], [], [], [], []

        self.s_SWT = scipy.sparse.csr_matrix((n_roi, n_roi))

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

            assert [len(s_NN.data)==len(s_sf.data), len(s_SWT.data)==len(s_sf.data), len(s_sesh.data)==len(s_sf.data)], 'The number of data elements in the similarity matrices should be the same.'
        # return s_sf_all, s_NN_all, s_SWT_all, s_sesh_all, idxROI_block_all

        print('Joining blocks into full similarity matrices...') if self._verbose else None
        # self.s_sf = helpers.merge_sparse_arrays(s_sf_all, idxROI_block_all, (n_roi, n_roi), remove_redundant=True).tocsr()
        # self.s_NN = helpers.merge_sparse_arrays(s_NN_all, idxROI_block_all, (n_roi, n_roi), remove_redundant=True).tocsr()
        # self.s_SWT = helpers.merge_sparse_arrays(s_SWT_all, idxROI_block_all, (n_roi, n_roi), remove_redundant=True).tocsr()
        # self.s_sesh = helpers.merge_sparse_arrays(s_sesh_all, idxROI_block_all, (n_roi, n_roi), remove_redundant=True).tocsr()

        def merge_sparse_arrays(s_list, idx_list, shape, shift_val=None):
            def csr_to_coo_idxd(s_csr, idx, shift_val, shape):
                s_csr.data[s_csr.data < 0] = 1e-10
                s_coo = s_csr.tocoo()
                return scipy.sparse.coo_matrix(
                    (s_coo.data + shift_val, (idx[s_coo.row], idx[s_coo.col])),
                    shape=shape
                )
            if shift_val == None:
                shift_val = min([s.min() for s in s_list]) + 1
            
            s_flat = scipy.sparse.vstack([
                csr_to_coo_idxd(s, idx, shift_val, shape).reshape(1,-1).tocsr() for s, idx in zip(s_list, idx_list)
            ]).tocsr()  ## for each block, expand the shape of the sparse similarity matrix to be (n_roi, n_roi) and then stack them all together
            
            s_flat = sparse.COO(s_flat).max(0)[None,:].to_scipy_sparse().tocsr()  ## It is MUCH faster to do this with sparse.COO than with scipy.sparse
            
            s_merged = s_flat.reshape(shape)
            s_merged.data = s_merged.data - shift_val
            return s_merged  ## the max operation is why it's so slow

        print('Joining s_sf...') if self._verbose else None
        self.s_sf = merge_sparse_arrays(s_sf_all, idxROI_block_all, (n_roi, n_roi)).tocsr()
        print('Joining s_NN...') if self._verbose else None
        self.s_NN = merge_sparse_arrays(s_NN_all, idxROI_block_all, (n_roi, n_roi)).tocsr()
        print('Joining s_SWT...') if self._verbose else None
        self.s_SWT = merge_sparse_arrays(s_SWT_all, idxROI_block_all, (n_roi, n_roi)).tocsr()
        print('Joining s_sesh...') if self._verbose else None
        self.s_sesh = merge_sparse_arrays(s_sesh_all, idxROI_block_all, (n_roi, n_roi)).tocsr()

        ## Old method for joining sparse arrays
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

    def _helper_compute_ROI_similarity_graph(
        self,
        spatialFootprints: scipy.sparse.csr_matrix,
        features_NN: torch.Tensor,
        features_SWT: torch.Tensor,
        ROI_session_bool: np.ndarray,
    ) -> Tuple[scipy.sparse.csr_matrix, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the similarity matrix between ROIs based on the conjunction of
        the similarity matrices for different modes (like the NN embedding, the
        SWT embedding, and the spatial footprint overlap).
        RH 2022

        Args:
            spatialFootprints (scipy.sparse.csr_matrix): 
                The spatial footprints of the ROIs. with shape *(n_ROIs for each
                session, FOV height * FOV width)*.
            features_NN (torch.Tensor): 
                The output latent embeddings of the NN model with shape *(n_ROIs
                total, n_features)*.
            features_SWT (torch.Tensor): 
                The output latent embeddings of the SWT model with shape
                *(n_ROIs total, n_features)*.
            ROI_session_bool (np.ndarray): 
                The boolean matrix indicating which ROIs belong to which
                session. The shape is *(n_ROIs total, n_sessions)*.

        Returns:
            (tuple): tuple containing:
                s_sf (scipy.sparse.csr_matrix):
                    Pairwise similarity matrix based on spatial footprints.
                s_NN (torch.Tensor): 
                    Pairwise similarity matrix based on Neural Network features.
                s_SWT (torch.Tensor): 
                    Pairwise similarity matrix based on Scattering Wavelet
                    Transform.
                s_sesh (torch.Tensor): 
                    Pairwise similarity matrix based on session information.
        """
        ## if there are no ROIs in the block
        if spatialFootprints.shape[0] == 0:
            return None, None, None, None

        sf = spatialFootprints.power(self._sf_maskPower)
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
        # s_NN[s_NN < 0] = 0
        s_NN[range(s_NN.shape[0]), range(s_NN.shape[0])] = 0
        
        features_SWT_normd = torch.nn.functional.normalize(features_SWT, dim=1)
        s_SWT = torch.matmul(features_SWT_normd, features_SWT_normd.T) ## cosine similarity. Normalized to [0,1]
        # s_SWT[s_SWT>(1-1e-5)] = 1.0
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
        centers_of_mass: Union[np.ndarray, List[np.ndarray]],
        features_NN: Optional[torch.Tensor] = None,
        features_SWT: Optional[torch.Tensor] = None,
        k_max: int = 3000,
        k_min: int = 200,
        algo_NN: str = 'kd_tree',
        device: str = 'cpu',
        verbose: bool = True,
    ) -> None:
        """
        Normalizes the similarity matrices **s_NN**, **s_SWT** (but not
        **s_sf**) by z-scoring using the mean and standard deviation from the
        distributions of pairwise similarities between ROIs that are spatially
        distant from each other. This is done to make the similarity scores
        more comparable across different regions of the field of view.
        RH 2022

        Args:
            centers_of_mass (Union[np.ndarray, List[np.ndarray]]): 
                The centers of mass of the ROIs. Can be an array with shape:
                *(n_ROIs total, 2)*, or a list of arrays with shape: *(n_ROIs
                for each session, 2)*.
            features_NN (torch.Tensor): 
                The output latent embeddings of the NN model. Shape: *(n_ROIs
                total, n_features)*. (Default is ``None``)
            features_SWT (torch.Tensor): 
                The output latent embeddings of the SWT model. Shape: *(n_ROIs
                total, n_features)*. (Default is ``None``)
            k_max (int): 
                The maximum number of nearest neighbors to consider for each
                ROI. This value will result in an intermediate similarity matrix
                of shape *(n_ROIs total, k_max)* between each ROI and its k_max
                nearest neighbors. This value is based on centroid distance.
                (Default is ``3000``)
            k_min (int): 
                The minimum number of nearest neighbors to consider for each
                ROI. This value should be less than k_max and be chosen such
                that it is likely that any potential 'same' ROIs are within
                k_min nearest neighbors. This value is based on centroid
                distance. (Default is ``200``)
            algo_NN (str): 
                The algorithm to use for the nearest neighbor search. See
                `sklearn.neighbors.NearestNeighbors
                <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html>`_
                for options. It can be 'kd_tree', 'ball_tree', or 'brute'.
                'kd_tree' seems to be the fastest. (Default is ``'kd_tree'``)
            device (str): 
                The device to use for the similarity computations. The output
                will still be on CPU. (Default is ``'cpu'``)
            verbose (bool): 
                If ``True``, print progress updates. (Default is ``True``)

        Attributes:
            s_NN_z (scipy.sparse.csr_matrix): 
                The z-scored similarity matrix between ROIs based on the
                statistics of the NN embedding. Shape: *(n_ROIs total, n_ROIs
                total)*. Note: This matrix is not symmetric and therefore should
                be treated as a directed graph.
            s_SWT_z (scipy.sparse.csr_matrix): 
                The z-scored similarity matrix between ROIs based on the
                statistics of the SWT embedding. Shape: *(n_ROIs total, n_ROIs
                total)*. Note: This matrix is not symmetric and therefore should
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
        ##  'different' ROIs. Note that symmetry is lost here.
        print('Normalizing Neural Network similarity scores...') if verbose else None
        if features_NN is not None:
            s_NN_diff = cosine_similarity_customIdx(features_NN.to(device), idx_diff)
            mus_NN_diff = s_NN_diff.mean(1).to('cpu').numpy()
            stds_NN_diff = s_NN_diff.std(1).to('cpu').numpy()
            
            self.s_NN_z = self.s_NN.copy().tocoo()
            self.s_NN_z.data = ((self.s_NN_z.data - mus_NN_diff[self.s_NN_z.row]) / stds_NN_diff[self.s_NN_z.row])
            self.s_NN_z = self.s_NN_z.tocsr()
            self.s_NN_z.data[np.isnan(self.s_NN_z.data)] = 0
        
        print('Normalizing SWT similarity scores...') if verbose else None
        if features_SWT is not None:
            s_SWT_diff = cosine_similarity_customIdx(features_SWT.to(device), idx_diff)
            mus_SWT_diff = s_SWT_diff.mean(1).to('cpu').numpy()
            stds_SWT_diff = s_SWT_diff.std(1).to('cpu').numpy()

            self.s_SWT_z = self.s_SWT.copy().tocoo()
            self.s_SWT_z.data = ((self.s_SWT_z.data - mus_SWT_diff[self.s_SWT_z.row]) / stds_SWT_diff[self.s_SWT_z.row])
            self.s_SWT_z = self.s_SWT_z.tocsr()
            self.s_SWT_z.data[np.isnan(self.s_SWT_z.data)] = 0
            

###########################
####### block stuff #######
###########################

    def _make_block_batches(
        self,
        frame_height: int = 512,
        frame_width: int = 1024,
        block_height: int = 100, 
        block_width: int = 100,
        overlapping_width_Multiplier: float = 0.0,
        clamp_blocks_to_frame: bool = True,
    ) -> Tuple[List[List[List[int]]], Tuple[np.ndarray, np.ndarray]]:     
        """
        Generates blocks that partition the field of view into smaller sections.
        As computations in this module are often pairwise comparisons, it can be
        useful to restrict comparisons to smaller blocks.
        RH 2022

        Args:
            frame_height (int):
                The height of the field of view. (Default is ``512``)
            frame_width (int):
                The width of the field of view. (Default is ``1024``)
            block_height (int):
                The height of each block. If ``clamp_blocks_to_frame`` is
                ``True``, this value will be adjusted slightly to tile cleanly.
                (Default is ``100``)
            block_width (int):
                The width of each block. If ``clamp_blocks_to_frame`` is
                ``True``, this value will be adjusted slightly to tile cleanly.
                (Default is ``100``)
            overlapping_width_Multiplier (float):
                The fractional amount of overlap between blocks. (Default is
                ``0.0``)
            clamp_blocks_to_frame (bool):
                If ``True``, then edges of the blocks will be clamped to the
                edges of the field of view. (Default is ``True``)

        Returns:
            (Tuple): tuple
            containing:
                blocks (List[List[List[int]]]):
                    Blocks partitioning the field of view.
                centers (Tuple[np.ndarray, np.ndarray]):
                    X and Y centers of the blocks.
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


    def visualize_blocks(self) -> None:
        """
        Visualizes the blocks over a field of view by displaying them. This is
        primarily used for checking the correct partitioning of the blocks. 
        """
        im = np.zeros((self._frame_height, self._frame_width, 3))
        for ii, block in enumerate(self.blocks):
            im[block[0][0]:block[0][1], block[1][0]:block[1][1], :] = ((np.random.rand(1)+0.5)/2)
        fig = plt.figure()
        plt.imshow(im, vmin=0, vmax=1)
        return fig


def get_idx_in_kRange(
    X: np.ndarray,
    k_max: int = 3000,
    k_min: int = 100,
    algo_kNN: str = 'brute',
    n_workers: int = -1,
) -> Tuple[np.ndarray, scipy.sparse.coo_matrix]:
    """
    Get indices in a given range for k-Nearest Neighbors graph.
    RH 2022

    Args:
        X (np.ndarray): 
            Input data array where each row is a data point and each column is a
            feature.
        k_max (int): 
            Maximum number of neighbors to find. (Default is ``3000``)
        k_min (int): 
            Minimum number of neighbors to consider. (Default is ``100``)
        algo_kNN (str): 
            Algorithm to use for nearest neighbors search. (Default is
            ``'brute'``)
        n_workers (int): 
            Number of worker processes to use. If ``-1``, use all available
            cores. (Default is ``-1``)

    Returns:
        (Tuple[np.ndarray, scipy.sparse.coo_matrix]): tuple containing:
            idx_diff (np.ndarray): 
                Indices of the non-zero values in the distance graph, with a
                range between ``k_min`` and ``k_max``.
            d (scipy.sparse.coo_matrix): 
                Sparse matrix representing the distance graph from the k-Nearest
                Neighbors algorithm.
    """
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

def cosine_similarity_customIdx(
    features: torch.Tensor,
    idx: np.ndarray
) -> torch.Tensor:
    """
    Calculate cosine similarity using custom indices.

    Args:
        features (torch.Tensor): 
            A tensor of feature vectors. Shape: *(n, d)*, where *n* is the
            number of data points and *d* is the dimensionality of the data.
        idx (np.ndarray): 
            Array of indices. Shape should match the first dimension of the
            features tensor.

    Returns:
        (torch.Tensor): 
            result (torch.Tensor):
                Cosine similarity tensor calculated using the provided indices.
                Shape: *(n, d)*, where *n* is the number of data points and *d*
                is the dimensionality of the data.
    """
    f = torch.nn.functional.normalize(features, dim=1)
    out = torch.stack([f[ii] @ f[idx[ii]].T for ii in tqdm(range(f.shape[0]))], dim=0)
    return out

