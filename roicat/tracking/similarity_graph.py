import multiprocessing as mp
from dataclasses import dataclass, field, asdict
from typing import Tuple, Union, List, Optional, Dict, Any, Callable

import scipy.sparse
import numpy as np
import torch
import sklearn
import sklearn.neighbors
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import sparse

from .. import helpers, util



@dataclass
class SimilarityMetric:
    """
    Configuration for one similarity metric in the pluggable system.

    Each metric represents a pairwise comparison between ROIs. It can be
    computed from raw features (via ``similarity_fn``) or provided as a
    precomputed sparse similarity matrix.

    Args:
        name (str):
            Unique identifier for this metric. Used as dict key throughout.
            Examples: ``'sf'``, ``'nn'``, ``'swt'``, ``'temporal'``.
        similarity_fn (Union[str, Callable, None]):
            How to compute pairwise similarity from raw features: \n
            * ``'cosine'``: L2-normalize, then matmul. For dense feature
              vectors.
            * ``'manhattan'``: sklearn NearestNeighbors with manhattan
              metric. For sparse spatial footprints.
            * ``Callable``: Custom function with signature
              ``(features_block, **kwargs) -> similarity_matrix``.
              Must return a matrix (dense or sparse) of shape
              *(n_roi_block, n_roi_block)*.
            * ``None``: Metric uses a precomputed similarity matrix
              (no feature computation needed).
        is_sparsity_source (bool):
            If ``True``, this metric's nonzero pattern contributes to
            the global sparsity mask. When only one metric has this set,
            all others are masked to its pattern. When multiple metrics
            are sparsity sources, the intersection of their nonzero
            patterns is used.
        normalize_zscore (bool):
            If ``True``, z-score normalize this metric's similarity
            values using distant-neighbor statistics (same approach as
            the current ``make_normalized_similarities``).
        optimize_power (bool):
            If ``True``, include ``power_<name>`` in the DE optimization
            bounds. If ``False``, similarity values are used as-is
            (power=1) in the conjunctive distance computation.
        optimize_sigmoid (bool):
            If ``True``, estimate sigmoid parameters ``(mu, b)`` from
            NB calibration for this metric. If ``False``, no sigmoid
            activation is applied.
        power_bounds (Tuple[float, float]):
            Bounds for the power parameter in DE optimization. Only used
            when ``optimize_power=True``.
        similarity_fn_kwargs (dict):
            Additional keyword arguments passed to ``similarity_fn``.
            For ``'manhattan'``: can include ``'algorithm'``,
            ``'n_jobs'``, etc.
        post_process (Optional[Dict[str, Any]]):
            Per-metric post-processing after similarity computation.
            Supported keys: \n
            * ``'clip_min'`` (Optional[float]): Clip values below this
              threshold to zero. ``None`` means no clipping.
            * ``'clip_near_one'`` (bool): If ``True``, clip values
              above ``(1 - 1e-5)`` to exactly 1.0.
    """
    name: str
    similarity_fn: Union[str, Callable, None] = 'cosine'
    is_sparsity_source: bool = False
    normalize_zscore: bool = False
    optimize_power: bool = True
    optimize_sigmoid: bool = True
    power_bounds: Tuple[float, float] = (0.0, 2.0)
    similarity_fn_kwargs: dict = field(default_factory=dict)
    post_process: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """JSON-safe dict for serialization. Custom callables in
        ``similarity_fn`` are stored as a descriptive string (e.g.
        ``"<callable: my_func>"``). The original callable must be
        re-provided when reconstructing via ``from_dict``."""
        d = asdict(self)
        if callable(d['similarity_fn']):
            name = getattr(d['similarity_fn'], '__qualname__', None) or getattr(d['similarity_fn'], '__name__', repr(d['similarity_fn']))
            d['similarity_fn'] = f"<callable: {name}>"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> 'SimilarityMetric':
        """Reconstruct from dict. ``similarity_fn`` must be str or None."""
        return cls(**d)


## Default metric configurations matching the current 3-metric system
DEFAULT_METRICS = [
    SimilarityMetric(
        name='sf',
        similarity_fn='manhattan',
        is_sparsity_source=True,
        normalize_zscore=False,
        optimize_power=False,
        optimize_sigmoid=False,
        similarity_fn_kwargs={},
    ),
    SimilarityMetric(
        name='nn',
        similarity_fn='cosine',
        is_sparsity_source=False,
        normalize_zscore=True,
        optimize_power=True,
        optimize_sigmoid=True,
        power_bounds=(0.0, 2.0),
        post_process={'clip_min': None, 'clip_near_one': True},
    ),
    SimilarityMetric(
        name='swt',
        similarity_fn='cosine',
        is_sparsity_source=False,
        normalize_zscore=True,
        optimize_power=True,
        optimize_sigmoid=True,
        power_bounds=(0.0, 2.0),
        post_process={'clip_min': 0.0, 'clip_near_one': False},
    ),
]


class ROI_graph(util.ROICaT_Module):
    """
    Class for building similarity and distance graphs between Regions of
    Interest (ROIs) based on their features, generating potential clusters of
    ROIs using linkage clustering, building a similarity graph between clusters
    of ROIs, and computing silhouette scores for each potential cluster. The
    computations are performed on 'blocks' of the full field of view to
    accelerate computation and reduce memory usage.

    The similarity system is **pluggable**: each pairwise metric is described
    by a ``SimilarityMetric`` dataclass. The default configuration
    (``DEFAULT_METRICS``) reproduces the legacy 3-metric system (spatial
    footprints + ROInet NN + SWT).
    RH 2022 / 2026

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
            sklearn.neighbors.NearestNeighbors for more information. (Default
            is ``'brute'``)
        verbose (bool):
            If set to ``True``, outputs will be verbose. (Default is ``True``)
        metric_configs (Optional[List[SimilarityMetric]]):
            Pluggable metric configurations. If ``None``, defaults to
            ``DEFAULT_METRICS`` at compute time. (Default is ``None``)
        kwargs_nearestNeigbors_spatialFootprints (dict):
            The keyword arguments to use for the nearest neighbors. See
            sklearn.neighbors.NearestNeighbors for more information.
            (Optional)

    Attributes:
        similarities (Dict[str, scipy.sparse.csr_matrix]):
            Dict mapping metric name to pairwise similarity matrix. Populated
            after ``compute_similarity_blockwise`` is called.
        s_sesh (scipy.sparse.csr_matrix):
            Pairwise session-membership mask (True where ROIs come from
            different sessions).
        similarities_z (Dict[str, scipy.sparse.csr_matrix]):
            Dict mapping metric name to z-scored similarity matrix. Populated
            after ``make_normalized_similarities`` is called.
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
        metric_configs: Optional[List[SimilarityMetric]] = None,
        kwargs_nearestNeigbors_spatialFootprints: dict = {},
    ):
        """
        Initializes the ROI_graph class with the given parameters.
        """
        ## Imports
        super().__init__()

        ## Store parameter (but not data) args as attributes
        self.params['__init__'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'n_workers',
                'frame_height',
                'frame_width',
                'block_height',
                'block_width',
                'overlapping_width_Multiplier',
                'algorithm_nearestNeigbors_spatialFootprints',
                'verbose',
                'kwargs_nearestNeigbors_spatialFootprints',
            ],
        )

        ## Store NN algorithm settings (used as fallback for manhattan metrics)
        self._algo_sf = algorithm_nearestNeigbors_spatialFootprints
        self._kwargs_sf = kwargs_nearestNeigbors_spatialFootprints

        ## Store metric configs directly as SimilarityMetric objects.
        ## RichFile_ROICaT has a registered type handler for SimilarityMetric
        ## that serializes them as JSON via to_dict()/from_dict().
        ## None means use DEFAULT_METRICS at compute time.
        self._metric_configs_stored = list(metric_configs) if metric_configs is not None else None

        self._verbose = verbose
        self._n_workers = mp.cpu_count() if n_workers == -1 else n_workers

        self._frame_height = frame_height
        self._frame_width = frame_width

        ## Build block grid for spatial partitioning
        self.blocks, (self._centers_y, self._centers_x) = self._make_block_batches(
            frame_height=frame_height,
            frame_width=frame_width,
            block_height=block_height,
            block_width=block_width,
            overlapping_width_Multiplier=overlapping_width_Multiplier,
            clamp_blocks_to_frame=True,
        )

        ## Precompute pixel indices for each block
        self.idxPixels_block = []
        for block in self.blocks:
            idx_tmp = np.zeros((self._frame_height, self._frame_width), dtype=bool)
            idx_tmp[block[0][0]:block[0][1], block[1][0]:block[1][1]] = True
            idx_tmp = np.where(idx_tmp.reshape(-1))[0]
            self.idxPixels_block.append(idx_tmp)

    def __repr__(self):
        has_sim = hasattr(self, 'similarities') and self.similarities is not None and len(self.similarities) > 0
        if has_sim:
            ## Use the first metric's matrix for summary stats
            first_key = next(iter(self.similarities))
            first_mat = self.similarities[first_key]
            n_roi = first_mat.shape[0]
            nnz = first_mat.nnz
            metric_names = list(self.similarities.keys())
        else:
            n_roi = 0
            nnz = 0
            metric_names = []
        return f"ROI_graph(n_roi={n_roi}, nnz={nnz}, metrics={metric_names})"


    @property
    def _metric_configs(self) -> Optional[List[SimilarityMetric]]:
        """Return stored SimilarityMetric instances, or None if not set."""
        stored = getattr(self, '_metric_configs_stored', None)
        if stored is None:
            return None
        ## After RichFile load, stored objects are already SimilarityMetric.
        ## After legacy pickle load with dicts, reconstruct.
        if len(stored) > 0 and isinstance(stored[0], dict):
            return [SimilarityMetric.from_dict(d) for d in stored]
        return stored

    @_metric_configs.setter
    def _metric_configs(self, value):
        """Store metric configs.

        Accepts:
            - ``None``: clears stored configs.
            - ``List[SimilarityMetric]``: stores directly.
            - ``List[dict]``: stores as-is (reconstructed on access).
        """
        self._metric_configs_stored = value


    ###########################################################################
    ## Blockwise similarity computation
    ###########################################################################

    def compute_similarity_blockwise(
        self,
        spatialFootprints: scipy.sparse.csr_matrix,
        ROI_session_bool: torch.Tensor,
        features: Dict[str, torch.Tensor],
        spatialFootprint_maskPower: float = 1.0,
        precomputed_similarities: Optional[Dict[str, scipy.sparse.csr_matrix]] = None,
        metric_configs: Optional[List[SimilarityMetric]] = None,
    ) -> Dict[str, scipy.sparse.csr_matrix]:
        """
        Computes the similarity graph between ROIs for all configured metrics.
        Results are stored in ``self.similarities`` (dict mapping metric name
        to sparse similarity matrix) and ``self.s_sesh`` (session mask).

        Computation is done block-by-block over the field of view.

        Args:
            spatialFootprints (scipy.sparse.csr_matrix):
                The spatial footprints of the ROIs. Can be obtained from
                ``blurring.ROI_blurrer.ROIs_blurred`` or
                ``data_importing.Data_suite2p.spatialFootprints``.
            ROI_session_bool (torch.Tensor):
                Boolean array indicating which ROIs (across all sessions)
                belong to each session. Shape: *(n_ROIs total, n_sessions)*.
            features (Dict[str, torch.Tensor]):
                Dict mapping metric name to feature tensor. For example:
                ``{'nn': roinet.latents, 'swt': swt.latents}``.
                The ``'sf'`` key is handled automatically from
                ``spatialFootprints`` and should NOT be in this dict.
            spatialFootprint_maskPower (float):
                Power to raise the spatial footprint mask to. Use 1.0 for no
                change, low values (e.g. 0.5) for more binary masks, high
                values (e.g. 2.0) for intensity-dependent similarities.
                Applied ONCE inside ``_compute_manhattan_similarity``.
                (Default is ``1.0``)
            precomputed_similarities (Optional[Dict[str, scipy.sparse.csr_matrix]]):
                Optional dict of precomputed similarity matrices (keyed by
                metric name). These bypass the similarity computation step
                for the corresponding metric. (Default is ``None``)
            metric_configs (Optional[List[SimilarityMetric]]):
                Override metric configs for this call. If ``None``, uses
                ``self._metric_configs`` or ``DEFAULT_METRICS``.
                (Default is ``None``)

        Returns:
            Dict[str, scipy.sparse.csr_matrix]:
                Dict mapping metric name to pairwise similarity matrix.
                Also stored as ``self.similarities``.
        """
        ## Resolve metric configs: argument > instance > default
        if metric_configs is not None:
            resolved_configs = metric_configs
        elif self._metric_configs is not None:
            resolved_configs = self._metric_configs
        else:
            resolved_configs = DEFAULT_METRICS

        ## Validate: at least one sparsity source
        sparsity_sources = [m for m in resolved_configs if m.is_sparsity_source]
        assert len(sparsity_sources) > 0, (
            "At least one SimilarityMetric must have is_sparsity_source=True. "
            f"Got metric names: {[m.name for m in resolved_configs]}"
        )

        ## Validate: metric names are unique
        metric_names = [m.name for m in resolved_configs]
        assert len(metric_names) == len(set(metric_names)), (
            f"Metric names must be unique. Got duplicates in: {metric_names}"
        )

        ## Store resolved configs and mask power
        self._metric_configs = resolved_configs  ## setter converts to dicts
        self._sf_maskPower = spatialFootprint_maskPower

        ## Store parameter (but not data) args as attributes
        self.params['compute_similarity_blockwise'] = self._locals_to_params(
            locals_dict=locals(),
            keys=['spatialFootprint_maskPower'],
        )

        self._n_sessions = ROI_session_bool.shape[1]
        if precomputed_similarities is None:
            precomputed_similarities = {}

        ## Concatenate spatial footprints into single sparse matrix
        self.sf_cat = scipy.sparse.vstack(spatialFootprints).tocsr()
        n_roi = self.sf_cat.shape[0]  ## scalar: total number of ROIs

        ## Accumulators: one list per metric name, plus session mask
        block_results_per_metric = {m.name: [] for m in resolved_configs}
        s_sesh_all = []
        idxROI_block_all = []

        ## Iterate over spatial blocks
        print('Computing pairwise similarity between ROIs...') if self._verbose else None
        for ii, block in tqdm(enumerate(self.blocks), total=len(self.blocks), mininterval=10):
            ## Find which ROIs overlap this block
            idxROI_block = np.where(
                self.sf_cat[:, self.idxPixels_block[ii]].sum(1) > 0
            )[0]  ## shape: (n_roi_in_block,)

            ## Slice features for this block
            block_features = {}
            for feat_name, feat_tensor in features.items():
                block_features[feat_name] = feat_tensor[idxROI_block]

            ## Slice precomputed similarities for this block
            block_precomputed = {}
            for pre_name, pre_mat in precomputed_similarities.items():
                block_precomputed[pre_name] = pre_mat[idxROI_block][:, idxROI_block]

            ## Compute similarity for all metrics in this block
            block_similarities, block_s_sesh = self._helper_compute_ROI_similarity_graph(
                spatialFootprints=self.sf_cat[idxROI_block],
                block_features=block_features,
                block_precomputed=block_precomputed,
                ROI_session_bool=ROI_session_bool[idxROI_block],
                metric_configs=resolved_configs,
            )

            ## Skip empty blocks
            if block_similarities is None:
                continue

            ## Accumulate block results
            for m_name, s_block in block_similarities.items():
                block_results_per_metric[m_name].append(s_block)
            s_sesh_all.append(block_s_sesh)
            idxROI_block_all.append(idxROI_block)

            ## Validate: all metrics produced same number of nonzero entries
            nnz_counts = {name: s.nnz for name, s in block_similarities.items()}
            first_nnz = next(iter(nnz_counts.values()))
            assert all(c == first_nnz for c in nnz_counts.values()), (
                f"Block {ii}: nonzero counts differ across metrics: {nnz_counts}"
            )

        ## Merge block-level sparse matrices into a single full-size matrix.
        ## Algorithm:
        ##   1. Clamp negatives to 1e-10 (sparse format drops true zeros)
        ##   2. Shift all values positive by a constant
        ##   3. Remap local block indices → global ROI indices
        ##   4. Flatten each block to (1, n_roi*n_roi) and vstack
        ##   5. Take element-wise MAX across blocks (handles overlapping blocks)
        ##   6. Undo the shift
        ## Uses sparse.COO for the max reduction — much faster than scipy.
        def merge_sparse_arrays(s_list, idx_list, shape, shift_val=None):
            def csr_to_coo_idxd(s_csr, idx, shift_val, shape):
                ## Clamp negatives (z-scored metrics can have negative values
                ## that would be lost in sparse format without this shift trick)
                s_csr.data[s_csr.data < 0] = 1e-10
                s_coo = s_csr.tocoo()
                ## Remap local block indices to global indices and add shift
                return scipy.sparse.coo_matrix(
                    (s_coo.data + shift_val, (idx[s_coo.row], idx[s_coo.col])),
                    shape=shape,  ## (n_roi, n_roi)
                )
            if shift_val is None:
                shift_val = min([s.min() for s in s_list]) + 1

            ## Flatten each block to a row vector, stack vertically
            s_flat = scipy.sparse.vstack([
                helpers.reshape_coo_manual(
                    csr_to_coo_idxd(s, idx, shift_val, shape),
                    new_shape=(1, -1),
                ).tocsr()
                for s, idx in zip(s_list, idx_list)
            ]).tocsr()  ## shape: (n_blocks, n_roi * n_roi)

            ## Element-wise MAX across blocks (row axis)
            s_flat = sparse.COO(s_flat).max(0)[None, :].to_scipy_sparse().tocsr()

            ## Reshape back and undo shift
            s_merged = s_flat.reshape(shape)  ## shape: (n_roi, n_roi)
            s_merged.data = s_merged.data - shift_val
            return s_merged

        ## Merge each metric's block results into full similarity matrices
        print('Joining blocks into full similarity matrices...') if self._verbose else None
        self.similarities = {}
        for m_config in resolved_configs:
            m_name = m_config.name
            print(f'Joining {m_name}...') if self._verbose else None
            self.similarities[m_name] = merge_sparse_arrays(
                block_results_per_metric[m_name],
                idxROI_block_all,
                (n_roi, n_roi),
            ).tocsr()

        ## Merge session mask
        print('Joining s_sesh...') if self._verbose else None
        self.s_sesh = merge_sparse_arrays(
            s_sesh_all,
            idxROI_block_all,
            (n_roi, n_roi),
        ).tocsr()

        return self.similarities


    ###########################################################################
    ## Per-block similarity computation
    ###########################################################################

    def _helper_compute_ROI_similarity_graph(
        self,
        spatialFootprints: scipy.sparse.csr_matrix,
        block_features: Dict[str, torch.Tensor],
        block_precomputed: Dict[str, scipy.sparse.csr_matrix],
        ROI_session_bool: np.ndarray,
        metric_configs: List[SimilarityMetric],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Any]]:
        """
        Computes similarity matrices for all configured metrics within a
        single spatial block.

        For sparsity-source metrics, the raw similarity is computed first and
        its nonzero pattern defines the sparsity mask. All non-sparsity
        metrics are then masked to that pattern.

        Args:
            spatialFootprints (scipy.sparse.csr_matrix):
                Spatial footprints for ROIs in this block. Shape:
                *(n_roi_block, n_pixels)*. maskPower has NOT been applied.
            block_features (Dict[str, torch.Tensor]):
                Features for ROIs in this block, keyed by metric name.
            block_precomputed (Dict[str, scipy.sparse.csr_matrix]):
                Precomputed similarity sub-matrices for this block, keyed
                by metric name.
            ROI_session_bool (np.ndarray):
                Session membership for ROIs in this block. Shape:
                *(n_roi_block, n_sessions)*.
            metric_configs (List[SimilarityMetric]):
                Metric configurations to compute.

        Returns:
            Tuple of (similarities_dict, s_sesh), or (None, None) if no ROIs.
                similarities_dict: Dict[str, scipy.sparse.csr_matrix]
                s_sesh: scipy.sparse.csr_matrix
        """
        ## If no ROIs in block, skip
        if spatialFootprints.shape[0] == 0:
            return None, None

        n_roi_block = spatialFootprints.shape[0]  ## scalar

        ## ---------------------------------------------------------------
        ## Step 1: Compute sparsity source similarities
        ## ---------------------------------------------------------------
        sparsity_configs = [m for m in metric_configs if m.is_sparsity_source]
        sparsity_results = {}  ## name -> scipy.sparse.csr_matrix

        for config in sparsity_configs:
            if config.name in block_precomputed:
                ## Use precomputed matrix directly
                s_raw = block_precomputed[config.name]
                if not scipy.sparse.issparse(s_raw):
                    s_raw = scipy.sparse.csr_matrix(s_raw)
            else:
                ## Compute from features or spatial footprints
                s_raw = self._compute_metric_similarity(
                    config=config,
                    features=spatialFootprints if config.similarity_fn == 'manhattan' else block_features.get(config.name),
                )

            ## Force symmetry on sparse result
            if scipy.sparse.issparse(s_raw):
                s_raw = s_raw.maximum(s_raw.T)
            sparsity_results[config.name] = s_raw

        ## Build combined sparsity mask: intersection of all sparsity sources
        sparsity_mask = None
        for name, s_sparse in sparsity_results.items():
            s_csr = s_sparse.tocsr() if not scipy.sparse.isspmatrix_csr(s_sparse) else s_sparse
            binary_mask = (s_csr != 0).astype(np.float64)
            if sparsity_mask is None:
                sparsity_mask = binary_mask
            else:
                ## Intersection: element-wise multiply of binary masks
                sparsity_mask = sparsity_mask.multiply(binary_mask)
        sparsity_mask = sparsity_mask.tocsr()  ## shape: (n_roi_block, n_roi_block)

        ## ---------------------------------------------------------------
        ## Step 2: Compute non-sparsity-source metrics and apply mask
        ## ---------------------------------------------------------------
        similarities = {}

        ## Store sparsity source results (already sparse)
        for config in sparsity_configs:
            s = sparsity_results[config.name]
            if not scipy.sparse.isspmatrix_csr(s):
                s = s.tocsr()
            ## Apply combined mask (relevant when multiple sparsity sources)
            s_masked = helpers.sparse_mask(s, sparsity_mask, do_safety_steps=True)
            similarities[config.name] = s_masked

        ## Compute non-sparsity metrics
        non_sparsity_configs = [m for m in metric_configs if not m.is_sparsity_source]

        for config in non_sparsity_configs:
            if config.name in block_precomputed:
                ## Use precomputed matrix
                s_raw = block_precomputed[config.name]
                if scipy.sparse.issparse(s_raw):
                    s_raw = torch.as_tensor(s_raw.toarray(), dtype=torch.float32)
                elif not isinstance(s_raw, torch.Tensor):
                    s_raw = torch.as_tensor(np.asarray(s_raw), dtype=torch.float32)
            else:
                ## Compute from features
                feat = block_features.get(config.name)
                assert feat is not None, (
                    f"Metric '{config.name}' requires features but none were "
                    f"provided in the 'features' dict. Available keys: "
                    f"{list(block_features.keys())}"
                )
                s_raw = self._compute_metric_similarity(
                    config=config,
                    features=feat,
                )

            ## Force symmetry on dense result before masking
            if isinstance(s_raw, torch.Tensor):
                s_raw = torch.maximum(s_raw, s_raw.T)
            elif scipy.sparse.issparse(s_raw):
                s_raw = s_raw.maximum(s_raw.T)

            ## Mask to sparsity pattern
            s_masked = helpers.sparse_mask(s_raw, sparsity_mask, do_safety_steps=True)
            similarities[config.name] = s_masked

        ## ---------------------------------------------------------------
        ## Step 3: Compute session mask and apply sparsity pattern
        ## ---------------------------------------------------------------
        session_bool = torch.as_tensor(
            ROI_session_bool, device='cpu', dtype=torch.float32,
        )  ## shape: (n_roi_block, n_sessions)
        s_sesh = torch.logical_not(
            (session_bool @ session_bool.T).to(dtype=torch.bool)
        )  ## shape: (n_roi_block, n_roi_block), True where ROIs from different sessions

        s_sesh = helpers.sparse_mask(s_sesh, sparsity_mask, do_safety_steps=True)

        return similarities, s_sesh


    ###########################################################################
    ## Individual metric computation dispatch
    ###########################################################################

    def _compute_metric_similarity(
        self,
        config: SimilarityMetric,
        features: Any,
    ) -> Any:
        """
        Dispatches to the appropriate similarity function based on
        ``config.similarity_fn``.

        Args:
            config (SimilarityMetric):
                Metric configuration.
            features (Any):
                For ``'manhattan'``: scipy.sparse.csr_matrix of spatial
                footprints, shape *(n_roi_block, n_pixels)*.
                For ``'cosine'``: torch.Tensor of feature vectors, shape
                *(n_roi_block, n_features)*.
                For callable: passed directly to the callable.

        Returns:
            For ``'manhattan'``: scipy.sparse.csr_matrix (already sparse).
            For ``'cosine'``: torch.Tensor (dense, will be sparse-masked
            later).
            For callable: whatever the callable returns.
        """
        if config.similarity_fn == 'manhattan':
            return self._compute_manhattan_similarity(
                spatialFootprints=features,
                config=config,
            )
        elif config.similarity_fn == 'cosine':
            return self._compute_cosine_similarity(
                features=features,
                config=config,
            )
        elif callable(config.similarity_fn):
            return config.similarity_fn(features, **config.similarity_fn_kwargs)
        else:
            raise ValueError(
                f"Unknown similarity_fn '{config.similarity_fn}' for metric "
                f"'{config.name}'. Expected 'manhattan', 'cosine', a callable, "
                f"or None (for precomputed)."
            )


    ###########################################################################
    ## Manhattan (spatial footprint) similarity
    ###########################################################################

    def _compute_manhattan_similarity(
        self,
        spatialFootprints: scipy.sparse.csr_matrix,
        config: SimilarityMetric,
    ) -> scipy.sparse.csr_matrix:
        """
        Computes pairwise manhattan-distance-based similarity between spatial
        footprints. maskPower is applied ONCE here.

        Steps:
            1. Apply maskPower to spatial footprints
            2. Normalize each footprint to sum to 0.5
            3. Compute all-pairs manhattan distance via sklearn NearestNeighbors
            4. Convert distance to similarity: s = 1 - d
            5. Zero out self-similarities and near-zero values

        Args:
            spatialFootprints (scipy.sparse.csr_matrix):
                Raw (un-powered) spatial footprints. Shape:
                *(n_roi_block, n_pixels)*.
            config (SimilarityMetric):
                Metric configuration. ``similarity_fn_kwargs`` may contain
                ``'algorithm'`` and other kwargs for sklearn NearestNeighbors.

        Returns:
            scipy.sparse.csr_matrix:
                Pairwise similarity matrix. Shape:
                *(n_roi_block, n_roi_block)*.
        """
        ## Apply maskPower ONCE (fixes the double-application bug)
        sf = spatialFootprints.power(self._sf_maskPower)  ## shape: (n_roi_block, n_pixels)

        ## Normalize each footprint so rows sum to 0.5
        sf = sf.multiply(0.5 / sf.sum(1))  ## shape: (n_roi_block, n_pixels)
        sf = scipy.sparse.csr_matrix(sf)

        ## Resolve algorithm kwargs: config overrides, fall back to instance defaults
        algo = config.similarity_fn_kwargs.get('algorithm', self._algo_sf)
        extra_kwargs = {
            k: v for k, v in config.similarity_fn_kwargs.items()
            if k != 'algorithm'
        }
        ## Merge with instance-level kwargs (config takes precedence)
        merged_kwargs = {**self._kwargs_sf, **extra_kwargs}

        ## Compute all-pairs manhattan distance
        n_roi_block = sf.shape[0]  ## scalar
        d_sf = sklearn.neighbors.NearestNeighbors(
            algorithm=algo,
            n_neighbors=n_roi_block,
            metric='manhattan',
            p=1,
            n_jobs=self._n_workers,
            **merged_kwargs,
        ).fit(sf).kneighbors_graph(
            sf,
            n_neighbors=n_roi_block,
            mode='distance',
        )  ## shape: (n_roi_block, n_roi_block)

        ## Convert distance to similarity: s = 1 - d
        s_sf = d_sf.copy()
        s_sf.data = 1 - s_sf.data

        ## Rectify near-zero values (numerical artifacts from float arithmetic)
        s_sf.data[s_sf.data < 1e-5] = 0

        ## Zero out self-similarities (diagonal)
        s_sf[range(n_roi_block), range(n_roi_block)] = 0
        s_sf.eliminate_zeros()

        return s_sf  ## shape: (n_roi_block, n_roi_block)


    ###########################################################################
    ## Cosine similarity
    ###########################################################################

    def _compute_cosine_similarity(
        self,
        features: torch.Tensor,
        config: SimilarityMetric,
    ) -> torch.Tensor:
        """
        Computes pairwise cosine similarity from dense feature vectors.
        Applies per-metric post-processing (clipping) as specified in
        ``config.post_process``.

        Args:
            features (torch.Tensor):
                Feature vectors. Shape: *(n_roi_block, n_features)*.
            config (SimilarityMetric):
                Metric configuration. ``post_process`` dict controls clipping.

        Returns:
            torch.Tensor:
                Dense pairwise cosine similarity. Shape:
                *(n_roi_block, n_roi_block)*.
        """
        ## L2-normalize features along feature dimension
        features_normd = torch.nn.functional.normalize(
            features, dim=1,
        )  ## shape: (n_roi_block, n_features)

        ## Compute cosine similarity via matmul
        s = torch.matmul(
            features_normd, features_normd.T,
        )  ## shape: (n_roi_block, n_roi_block)

        ## Apply post-processing from config
        post = config.post_process or {}

        ## Clip near-one values to exactly 1.0 (handles numerical precision)
        if post.get('clip_near_one', False):
            s[s > (1 - 1e-5)] = 1.0

        ## Clip values below minimum threshold to zero
        clip_min = post.get('clip_min', None)
        if clip_min is not None:
            s[s < clip_min] = 0

        ## Zero out self-similarities (diagonal)
        n = s.shape[0]  ## scalar
        s[range(n), range(n)] = 0

        return s  ## shape: (n_roi_block, n_roi_block)


    ###########################################################################
    ## Z-score normalization
    ###########################################################################

    def make_normalized_similarities(
        self,
        centers_of_mass: Union[np.ndarray, List[np.ndarray]],
        features: Dict[str, torch.Tensor],
        k_max: int = 3000,
        k_min: int = 200,
        algo_NN: str = 'kd_tree',
        device: str = 'cpu',
        verbose: bool = True,
    ) -> None:
        """
        Normalizes similarity matrices by z-scoring using the mean and
        standard deviation from the distributions of pairwise similarities
        between ROIs that are spatially distant from each other. This makes
        similarity scores more comparable across different field-of-view
        regions.

        Only metrics with ``normalize_zscore=True`` in their config are
        z-scored. Metrics with ``normalize_zscore=False`` are copied into
        ``self.similarities_z`` as-is.

        Args:
            centers_of_mass (Union[np.ndarray, List[np.ndarray]]):
                Centers of mass of the ROIs. Array shape: *(n_ROIs total, 2)*,
                or a list of arrays with shape: *(n_ROIs per session, 2)*.
            features (Dict[str, torch.Tensor]):
                Dict mapping metric name to feature tensor. Only needed for
                metrics with ``normalize_zscore=True`` and
                ``similarity_fn='cosine'``.
            k_max (int):
                Maximum number of nearest centroid-distance neighbors.
                (Default is ``3000``)
            k_min (int):
                Minimum number of nearest centroid-distance neighbors. ROIs
                between k_min and k_max are used as the "different" reference
                distribution. (Default is ``200``)
            algo_NN (str):
                Algorithm for nearest neighbor search on centroids.
                (Default is ``'kd_tree'``)
            device (str):
                Device for similarity computations. Output is always CPU.
                (Default is ``'cpu'``)
            verbose (bool):
                If ``True``, print progress updates. (Default is ``True``)
        """
        assert hasattr(self, 'similarities') and self.similarities is not None, (
            "Must call compute_similarity_blockwise before make_normalized_similarities."
        )
        assert self._metric_configs is not None, (
            "No metric configs found. Call compute_similarity_blockwise first."
        )

        ## Store parameter (but not data) args as attributes
        self.params['make_normalized_similarities'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'k_max',
                'k_min',
                'algo_NN',
                'device',
            ],
        )

        ## Determine matrix size from first available similarity
        first_sim = next(iter(self.similarities.values()))
        k_max = min(k_max, first_sim.shape[0])

        ## Stack centers of mass into single array
        print('Finding k-range of center of mass distance neighbors for each ROI...') if verbose else None
        coms = np.vstack(centers_of_mass) if isinstance(centers_of_mass, list) else centers_of_mass  ## shape: (n_roi, 2)

        ## Find indices of 'different' (distant) ROIs for each ROI
        idx_diff, _ = get_idx_in_kRange(
            X=coms,
            k_max=k_max,
            k_min=k_min,
            algo_kNN=algo_NN,
            n_workers=self._n_workers,
        )  ## idx_diff shape: (n_roi, k_max - k_min)

        ## Initialize output dict
        self.similarities_z = {}

        ## Iterate over all metric configs
        for config in self._metric_configs:
            m_name = config.name

            ## Skip metrics not present in self.similarities
            if m_name not in self.similarities:
                continue

            if not config.normalize_zscore:
                ## Non-z-scored metrics (e.g. sf) are copied unchanged into
                ## similarities_z so that similarities_final contains ALL
                ## metrics. The Clusterer expects the full set.
                self.similarities_z[m_name] = self.similarities[m_name].copy()
                continue

            ## Z-score normalization requires features for cosine similarity
            assert m_name in features, (
                f"Metric '{m_name}' has normalize_zscore=True but no features "
                f"were provided. Available feature keys: {list(features.keys())}"
            )

            print(f'Normalizing {m_name} similarity scores...') if verbose else None

            ## Compute cosine similarity against 'different' ROIs
            feat = features[m_name]  ## shape: (n_roi, n_features)
            s_diff = cosine_similarity_customIdx(
                feat.to(device), idx_diff,
            )  ## shape: (n_roi, k_max - k_min)

            ## Compute per-ROI mean and std of distant similarities
            mus_diff = s_diff.mean(1).to('cpu').numpy()   ## shape: (n_roi,)
            stds_diff = s_diff.std(1).to('cpu').numpy()   ## shape: (n_roi,)

            ## Z-score the sparse similarity matrix
            s_z = self.similarities[m_name].copy().tocoo()
            s_z.data = (
                (s_z.data - mus_diff[s_z.row]) / stds_diff[s_z.row]
            )
            s_z = s_z.tocsr()

            ## Replace NaN values (from zero std) with zero
            s_z.data[np.isnan(s_z.data)] = 0

            self.similarities_z[m_name] = s_z


    @property
    def similarities_final(self) -> Dict[str, scipy.sparse.csr_matrix]:
        """
        Returns the final similarity matrices for downstream use (e.g.,
        Clusterer). For metrics with ``normalize_zscore=True``, returns
        the z-scored version. For others, returns the raw version.

        Requires :meth:`make_normalized_similarities` to have been called.
        Falls back to ``self.similarities`` if z-scored versions are not
        available.

        Returns:
            Dict[str, scipy.sparse.csr_matrix]:
                Dict mapping metric name to final similarity matrix.
        """
        if hasattr(self, 'similarities_z') and self.similarities_z is not None:
            return self.similarities_z
        return self.similarities


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
#     idx_topk = np.argpartition(d.toarray(), kth=k_min, axis=1)  ## partition the distance graph at the k_min-th value
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

