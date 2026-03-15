import warnings
from typing import Union, Tuple, List, Dict, Optional, Any, Callable

import numpy as np
import scipy
import scipy.optimize
import scipy.sparse
import scipy.signal
import sklearn
import sklearn.isotonic
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
# import optuna

from .. import helpers, util

class Clusterer(util.ROICaT_Module):
    """
    Class for clustering algorithms. Performs:
        * Optimal mixing and pruning of similarity matrices:
            * self.find_optimal_parameters_for_pruning()
            * self.make_pruned_similarity_graphs()
        * Clustering:
            * self.fit(): Which uses a modified HDBSCAN
            * self.fit_sequentialHungarian: Which uses a method similar to
              CaImAn's clustering method.
        * Quality control:
            * self.compute_cluster_quality_metrics()

    Initialization ingests and stores similarity matrices. RH 2023

    Args:
        s_sf (Optional[scipy.sparse.csr_matrix]):
            The similarity matrix for spatial footprints. Shape: *(n_rois,
            n_rois)*. Expecting input to be manhattan distance of spatial
            footprints normalized between 0 and 1.
        s_NN_z (Optional[scipy.sparse.csr_matrix]):
            The z-scored similarity matrix for neural network output
            similarities. Shape: *(n_rois, n_rois)*. Expecting input to be the
            cosine similarity matrix, z-scored row-wise.
        s_SWT_z (Optional[scipy.sparse.csr_matrix]):
            The z-scored similarity matrix for scattering transform output
            similarities. Shape: *(n_rois, n_rois)*. Expecting input to be the
            cosine similarity matrix, z-scored row-wise.
        s_sesh (Optional[scipy.sparse.csr_matrix]):
            The similarity matrix for session similarity. Shape: *(n_rois,
            n_rois)*. Boolean, with 1s where the two ROIs are from different
            sessions.
        custom_similarities (Optional[Dict[str, scipy.sparse.csr_matrix]]):
            Optional dict mapping metric names to user-supplied sparse
            similarity matrices. All matrices must have the same sparsity
            pattern (same ``nnz``) as ``s_sf``. These are integrated into
            the mixing pipeline via sigmoid -> power -> p-norm, just like
            the built-in metrics. Example: ``{'temporal': s_temporal,
            'size': s_size}``. (Default is ``None``)
        n_bins (int):
            Number of bins to use for the pairwise similarity distribution. If
            using automatic parameter finding, then using a large number of bins
            makes finding the separation point more noisy, and only slightly
            more accurate. If ``None``, then a heuristic is used to estimate the
            value based on the number of ROIs. (Default is ``50``)
        smoothing_window_bins (int): 
            Number of bins to use when smoothing the distribution. Using a small
            number of bins makes finding the separation point more noisy, and
            only slightly more accurate. Aim for 5-10% of the number of bins. If
            ``None``, then a heuristic is used. (Default is ``5``)
        verbose (bool):
            Specifies whether to print out information about the clustering
            process. (Default is ``True``)

    Attributes:
        s_sf (scipy.sparse.csr_matrix):
            The similarity matrix for spatial footprints. It is symmetric and
            has a shape of *(n_rois, n_rois)*.
        s_NN_z (scipy.sparse.csr_matrix):
            The z-scored similarity matrix for neural network output
            similarities. It is non-symmetric and has a shape of *(n_rois,
            n_rois)*. 
        s_SWT_z (scipy.sparse.csr_matrix):
            The z-scored similarity matrix for scattering transform output
            similarities. It is non-symmetric and has a shape of *(n_rois,
            n_rois)*.
        s_sesh (scipy.sparse.csr_matrix):
            The similarity matrix for session similarity. It is symmetric and
            has a shape of *(n_rois, n_rois)*.
        s_sesh_inv (scipy.sparse.csr_matrix):
            The inverse of the session similarity matrix. It is symmetric and
            has a shape of *(n_rois, n_rois)*.
        n_bins Optional[int]:
            Number of bins to use for the pairwise similarity distribution.
        smoothing_window_bins Optional[int]:
            Number of bins to use when smoothing the distribution.
        verbose (bool):
            Specifies how much information to print out: \n
                * 0/False: Warnings only
                * 1/True: Basic info, progress bar
                * 2: All info
    """
    def __init__(
        self,
        s_sf: Optional[scipy.sparse.csr_matrix] = None,
        s_NN_z: Optional[scipy.sparse.csr_matrix] = None,
        s_SWT_z: Optional[scipy.sparse.csr_matrix] = None,
        s_sesh: Optional[scipy.sparse.csr_matrix] = None,
        custom_similarities: Optional[Dict[str, scipy.sparse.csr_matrix]] = None,
        n_bins: Optional[int] = None,
        smoothing_window_bins: Optional[int] = None,
        session_bool: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """
        Initializes the Clusterer with the given similarity matrices and verbosity setting.
        """
        ## Imports
        super().__init__()

        ## Store parameter (but not data) args as attributes
        self.params['__init__'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'n_bins',
                'smoothing_window_bins',
                'verbose',
            ],
        )

        self.s_sf = s_sf
        self.s_NN_z = s_NN_z
        self.s_SWT_z = s_SWT_z
        self.s_sesh = s_sesh

        ## assert that all feature similarity matrices have the same nnz
        assert self.s_sf.nnz == self.s_NN_z.nnz == self.s_SWT_z.nnz

        ## Store and validate custom similarity matrices
        self.custom_similarities = custom_similarities or {}
        for name, s_custom in self.custom_similarities.items():
            assert scipy.sparse.issparse(s_custom), (
                f"custom_similarities['{name}'] must be a sparse matrix"
            )
            assert s_custom.nnz == self.s_sf.nnz, (
                f"custom_similarities['{name}'] has {s_custom.nnz} nonzeros, "
                f"expected {self.s_sf.nnz} (same as s_sf)"
            )

        self.s_sesh_inv = (self.s_sf != 0).astype(bool)
        self.s_sesh_inv[self.s_sesh.astype(bool)] = False
        self.s_sesh_inv.eliminate_zeros()

        self.s_sesh = self.s_sesh.tolil()
        self.s_sesh[range(self.s_sesh.shape[0]), range(self.s_sesh.shape[1])] = 0
        self.s_sesh = self.s_sesh.tocsr()

        self._verbose = verbose

        self.n_bins = max(min(self.s_sf.nnz // 10000, 200), 20) if n_bins is None else n_bins
        self.smooth_window = helpers.make_odd(self.n_bins // 10, mode='up') if smoothing_window_bins is None else smoothing_window_bins

        self._session_bool = session_bool

    def find_optimal_parameters_for_pruning(
        self,
        bounds_findParameters: Dict[str, List[float]] = {
            'power_NN': [0.0, 2.],
            'power_SWT': [0.0, 2.],
            'p_norm': [-5, -0.1],
        },
        de_kwargs: Dict[str, Any] = {
            'maxiter': 100,
            'tol': 1e-6,
            'popsize': 15,
            'mutation': (0.5, 1.5),
            'recombination': 0.7,
            'polish': True,
        },
        n_bins: Optional[int] = None,
        smoothing_window_bins: Optional[int] = None,
        subsample_pairs: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Find optimal mixing parameters for pruning the similarity graph.

        Two-stage approach:

        1. **Naive Bayes calibration**: For each similarity feature (SF, NN,
           SWT), estimates ``P(same | s_k)`` from histogram subtraction. The
           resulting per-feature calibration curves are used to analytically
           estimate optimal sigmoid parameters ``(mu, b)`` via Fisher's linear
           discriminant.
        2. **Differential evolution**: With sigmoid parameters frozen from
           stage 1, optimizes the remaining 3 parameters (``power_NN``,
           ``power_SWT``, ``p_norm``) by minimizing the histogram overlap loss.

        This method replaces the original Optuna TPE search (see
        :meth:`_find_optimal_parameters_for_pruning_optuna` in the legacy
        section). The two-stage approach achieves better separation quality
        (lower histogram overlap) on typical datasets.
        RH 2023 / 2025

        Args:
            bounds_findParameters (Dict[str, List[float]]):
                Bounds for the 3 optimized parameters: ``power_NN``,
                ``power_SWT``, ``p_norm``.
            de_kwargs (Dict[str, Any]):
                Keyword arguments for
                ``scipy.optimize.differential_evolution``: \n
                * ``maxiter`` (int): Maximum number of DE generations.
                * ``tol`` (float): Convergence tolerance on the loss.
                * ``popsize`` (int): Population size multiplier
                  (actual population = ``popsize * n_params``).
                * ``mutation`` (Tuple[float, float]): Differential
                  weight range ``(min, max)`` for dithering.
                * ``recombination`` (float): Crossover probability
                  in ``[0, 1]``.
                * ``polish`` (bool): If ``True``, run L-BFGS-B from
                  the best DE solution. Often has no effect on
                  piecewise-constant histogram loss.
            n_bins (Optional[int]):
                Overwrites ``n_bins`` from ``__init__``.
            smoothing_window_bins (Optional[int]):
                Overwrites ``smoothing_window_bins`` from ``__init__``.
            subsample_pairs (Optional[int]):
                If not ``None``, subsample this many pairs for histogram
                loss evaluation. Maintains intra/inter ratio. If ``None``,
                auto-computed based on pair counts.
            seed (Optional[int]):
                Random seed for reproducibility.

        Returns:
            (Dict):
                kwargs_makeConjunctiveDistanceMatrix_best (Dict):
                    Optimal parameters for
                    :meth:`make_conjunctive_distance_matrix`.
        """
        ## Store parameter (but not data) args as attributes
        self.params['find_optimal_parameters_for_pruning'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'bounds_findParameters', 'de_kwargs', 'n_bins',
                'smoothing_window_bins', 'subsample_pairs', 'seed',
            ],
        )

        ## NB calibration → Fisher sigmoid estimation → 3-param DE.
        return self._find_optimal_parameters_DE(
            bounds_findParameters=bounds_findParameters,
            de_kwargs=de_kwargs,
            n_bins=n_bins,
            smoothing_window_bins=smoothing_window_bins,
            subsample_pairs=subsample_pairs,
            seed=seed,
            freeze_sigmoid=True,
        )

    ####################################################################
    ## Differential evolution parameter optimization
    ####################################################################

    def _precompute_intra_mask(self) -> np.ndarray:
        """
        Build a boolean mask of shape ``(nnz,)`` indicating which nonzero
        entries in ``self.s_sf`` correspond to intra-session (known-different)
        ROI pairs.

        Uses an index-mapping trick: assigns each nonzero entry a unique 1-based
        index, multiplies by the inverse session matrix to isolate intra-session
        entries, then reads back which indices survived.

        The result is stored as a numpy bool array (so that serialization via
        ``serializable_dict`` preserves it). Call sites that need a torch tensor
        should wrap with ``torch.as_tensor(self._intra_mask)``.
        RH 2025

        Returns:
            (np.ndarray):
                intra_mask (np.ndarray):
                    Boolean numpy array, shape ``(self.s_sf.nnz,)``.
        """
        ## Map each nonzero entry to a unique 1-based index
        idx_mat = self.s_sf.copy().astype(np.float64)
        idx_mat.data = np.arange(1, self.s_sf.nnz + 1, dtype=np.float64)

        ## Elementwise multiply with s_sesh_inv to keep only intra-session entries
        masked = idx_mat.multiply(self.s_sesh_inv.astype(np.float64))
        masked.eliminate_zeros()

        ## Convert back to 0-based indices and build boolean mask
        intra_indices = (masked.data - 1).astype(np.int64)
        mask = np.zeros(self.s_sf.nnz, dtype=bool)
        mask[intra_indices] = True

        ## Store as a numpy bool array; callers convert with torch.as_tensor()
        self._intra_mask = mask
        return self._intra_mask

    def _subsample_pairs(
        self,
        n_subsample: int,
        intra_mask: torch.Tensor,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Subsample pair indices while preserving the intra/inter-session
        ratio. Returns indices into the full nnz-length arrays.

        Intra-session indices come first in the returned tensor, so the
        intra mask for the subsample is simply ``True`` for the first
        ``n_sample_intra`` entries.
        RH 2025

        Args:
            n_subsample (int):
                Target number of pairs to sample.
            intra_mask (torch.Tensor):
                Boolean tensor, shape ``(nnz,)``. ``True`` for
                intra-session (known-different) pairs.
            seed (Optional[int]):
                Random seed for reproducibility.

        Returns:
            (torch.Tensor):
                sample_idx (torch.Tensor):
                    1D int64 tensor of sampled pair indices, length
                    ``<= n_subsample``. Intra-session pairs come first.
        """
        ## Accept numpy or torch; convert to torch for all internal operations.
        intra_mask = torch.as_tensor(intra_mask)

        nnz = intra_mask.shape[0]
        n_intra = int(intra_mask.sum().item())
        n_inter = nnz - n_intra
        frac = n_subsample / nnz
        n_sample_intra = max(int(n_intra * frac), 100)
        n_sample_inter = max(int(n_inter * frac), 100)

        intra_idx = torch.where(intra_mask)[0]
        inter_idx = torch.where(~intra_mask)[0]

        rng = torch.Generator()
        rng.manual_seed(seed if seed is not None else 42)

        perm_intra = torch.randperm(n_intra, generator=rng)[:n_sample_intra]
        perm_inter = torch.randperm(n_inter, generator=rng)[:n_sample_inter]

        ## Intra first, then inter — intra_mask for subsample is True for
        ## the first n_sample_intra entries
        return torch.cat([intra_idx[perm_intra], inter_idx[perm_inter]])

    def _find_optimal_parameters_DE(
        self,
        bounds_findParameters: Dict[str, List[float]] = {
            'power_NN': [0.0, 2.],
            'power_SWT': [0.0, 2.],
            'p_norm': [-5, -0.1],
            'sig_NN_kwargs_mu': [0., 1.0],
            'sig_NN_kwargs_b': [0.1, 1.5],
            'sig_SWT_kwargs_mu': [0., 1.0],
            'sig_SWT_kwargs_b': [0.1, 1.5],
        },
        de_kwargs: Dict[str, Any] = {
            'maxiter': 100,
            'tol': 1e-6,
            'popsize': 15,
            'mutation': (0.5, 1.5),
            'recombination': 0.7,
            'polish': True,
        },
        n_bins: Optional[int] = None,
        smoothing_window_bins: Optional[int] = None,
        subsample_pairs: Optional[int] = None,
        seed: Optional[int] = None,
        freeze_sigmoid: bool = True,
    ) -> Dict:
        """
        Find optimal mixing parameters using scipy differential evolution.

        When ``freeze_sigmoid=True`` (default), sigmoid parameters (mu, b)
        for NN and SWT are estimated from NB calibration curves via Fisher's
        linear discriminant and held fixed, reducing the search to 3
        parameters (``power_NN``, ``power_SWT``, ``p_norm``). When
        ``False``, all 7 parameters are optimized jointly.

        The inner loop operates entirely on precomputed torch tensors — no
        scipy sparse operations per evaluation. When subsampling is active,
        the subsample is redrawn each DE generation to reduce overfitting
        to a specific pair subset.
        RH 2025

        Args:
            bounds_findParameters (Dict[str, List[float]]):
                Bounds for each parameter. When ``freeze_sigmoid=True``,
                only ``power_NN``, ``power_SWT``, ``p_norm`` are used.
                When ``False``, all 7 keys are needed.
            de_kwargs (Dict[str, Any]):
                Keyword arguments for
                ``scipy.optimize.differential_evolution``: \n
                * ``maxiter`` (int): Maximum number of DE generations.
                * ``tol`` (float): Convergence tolerance on the loss.
                * ``popsize`` (int): Population size multiplier
                  (actual population = ``popsize * n_params``).
                * ``mutation`` (Tuple[float, float]): Differential
                  weight range ``(min, max)`` for dithering.
                * ``recombination`` (float): Crossover probability
                  in ``[0, 1]``.
                * ``polish`` (bool): If ``True``, run L-BFGS-B from
                  the best DE solution. Often has no effect on
                  piecewise-constant histogram loss.
            n_bins (Optional[int]):
                Overwrites ``n_bins`` from __init__.
            smoothing_window_bins (Optional[int]):
                Overwrites ``smoothing_window_bins`` from __init__.
            subsample_pairs (Optional[int]):
                If not ``None``, subsample this many pairs for histogram
                loss. Maintains intra/inter ratio. If ``None``,
                auto-computed: subsamples to 1.1M (100k intra + 1M inter)
                when there are enough pairs, otherwise uses all pairs.
            seed (Optional[int]):
                Random seed for reproducibility.
            freeze_sigmoid (bool):
                If ``True``, fix sigmoid params from NB calibration,
                reducing DE to 3 parameters. If ``False``, optimize
                all 7 parameters jointly.

        Returns:
            (Dict):
                kwargs_makeConjunctiveDistanceMatrix_best (Dict):
                    Optimal parameters for
                    :meth:`make_conjunctive_distance_matrix`.
        """
        ## Store parameter (but not data) args as attributes
        self.params['_find_optimal_parameters_DE'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'bounds_findParameters', 'de_kwargs', 'n_bins',
                'smoothing_window_bins', 'subsample_pairs', 'seed',
                'freeze_sigmoid',
            ],
        )

        self.n_bins = self.n_bins if n_bins is None else n_bins
        self.smooth_window = self.smooth_window if smoothing_window_bins is None else smoothing_window_bins
        self.bounds_findParameters = bounds_findParameters
        self._seed = seed

        ################################################################
        ## Auto-compute subsample size if not specified
        ################################################################
        if not hasattr(self, '_intra_mask') or self._intra_mask is None:
            self._precompute_intra_mask()
        if subsample_pairs is None:
            n_intra = int(self._intra_mask.sum())
            n_inter = self.s_sf.nnz - n_intra
            ## Subsample only if we have enough pairs to meet minimums
            min_intra = 100_000
            min_inter = 1_000_000
            if n_intra >= min_intra and n_inter >= min_inter:
                subsample_pairs = min_intra + min_inter
            ## Otherwise use all pairs

        ################################################################
        ## Determine parameter layout: 3-param (frozen) or 7-param (full)
        ################################################################
        _frozen_sig = None
        if freeze_sigmoid:
            if not hasattr(self, 'calibrations_naive_bayes') or self.calibrations_naive_bayes is None:
                self.make_naive_bayes_distance_matrix()
            sig_params = self._estimate_sigmoid_params()
            _frozen_sig = {
                'mu_NN': sig_params['NN']['mu'],
                'b_NN': sig_params['NN']['b'],
                'mu_SWT': sig_params['SWT']['mu'],
                'b_SWT': sig_params['SWT']['b'],
            }
            print(
                f'  Freezing sigmoid: NN(mu={_frozen_sig["mu_NN"]:.3f}, '
                f'b={_frozen_sig["b_NN"]:.1f}), '
                f'SWT(mu={_frozen_sig["mu_SWT"]:.3f}, '
                f'b={_frozen_sig["b_SWT"]:.1f})'
            ) if self._verbose else None

        if freeze_sigmoid:
            param_keys = ['power_NN', 'power_SWT', 'p_norm']
        else:
            param_keys = [
                'power_NN', 'power_SWT', 'p_norm',
                'sig_NN_kwargs_mu', 'sig_NN_kwargs_b',
                'sig_SWT_kwargs_mu', 'sig_SWT_kwargs_b',
            ]

        ## Add custom metric power parameters (sorted for determinism)
        _custom_names_sorted = sorted(self.custom_similarities.keys())
        for name in _custom_names_sorted:
            key = f'power_{name}'
            param_keys.append(key)
            ## Use user-supplied bounds if available, else default [0.0, 2.0]
            if key not in bounds_findParameters:
                bounds_findParameters[key] = [0.0, 2.0]

        scipy_bounds = [tuple(bounds_findParameters[k]) for k in param_keys]

        ################################################################
        ## Precompute tensors — eliminates all sparse operations from
        ## the inner loop.
        ################################################################
        sf_t_full = torch.as_tensor(
            np.ascontiguousarray(self.s_sf.data), dtype=torch.float32,
        ).clone()  ## shape (nnz,)
        nn_t_full = torch.as_tensor(
            np.ascontiguousarray(self.s_NN_z.data), dtype=torch.float32,
        ).clone()  ## shape (nnz,)
        swt_t_full = torch.as_tensor(
            np.ascontiguousarray(self.s_SWT_z.data), dtype=torch.float32,
        ).clone()  ## shape (nnz,)

        ## Precompute custom similarity tensors
        custom_t_full = {}
        for name in _custom_names_sorted:
            custom_t_full[name] = torch.as_tensor(
                np.ascontiguousarray(self.custom_similarities[name].data),
                dtype=torch.float32,
            ).clone()  ## shape (nnz,)

        ## Boolean mask for intra-session (known-different) pairs
        if not hasattr(self, '_intra_mask') or self._intra_mask is None:
            self._precompute_intra_mask()
        intra_mask_full = torch.as_tensor(self._intra_mask)  ## shape (nnz,), bool

        ################################################################
        ## Helper: build working tensors (with optional subsampling)
        ################################################################
        def _build_working_tensors(resample_seed: Optional[int]):
            """Return (sf_t, nn_t, swt_t, custom_t_dict, intra_mask) after optional subsampling."""
            nnz_full = sf_t_full.shape[0]
            if subsample_pairs is not None and subsample_pairs < nnz_full:
                sidx = self._subsample_pairs(
                    n_subsample=subsample_pairs,
                    intra_mask=intra_mask_full,
                    seed=resample_seed if resample_seed is not None else 77777,
                )
                ## Intra pairs come first in sidx. Compute actual intra count
                ## using the same logic as _subsample_pairs to stay in sync.
                n_intra_full = int(intra_mask_full.sum().item())
                frac = subsample_pairs / nnz_full
                n_si = min(max(int(n_intra_full * frac), 100), n_intra_full)
                im = torch.zeros(sidx.shape[0], dtype=torch.bool)
                im[:n_si] = True
                custom_sub = {name: custom_t_full[name][sidx] for name in _custom_names_sorted}
                return sf_t_full[sidx], nn_t_full[sidx], swt_t_full[sidx], custom_sub, im
            else:
                return sf_t_full, nn_t_full, swt_t_full, {name: custom_t_full[name] for name in _custom_names_sorted}, intra_mask_full

        ## Initial working set
        sf_t, nn_t, swt_t, custom_t, intra_mask = _build_working_tensors(
            resample_seed=(seed + 77777) if seed is not None else 77777,
        )

        print(
            f'  Working set: {sf_t.shape[0]} pairs '
            f'({int(intra_mask.sum().item())} intra, '
            f'{sf_t.shape[0] - int(intra_mask.sum().item())} inter)'
        ) if self._verbose and subsample_pairs is not None else None

        ################################################################
        ## Build shared histogram infrastructure from current tensors
        ################################################################
        n_bins_val = self.n_bins
        edges = torch.linspace(0, 1, n_bins_val + 1, dtype=torch.float32)
        smooth_window = helpers.make_odd(n_bins_val // 10, mode='up')
        smoother = helpers.Convolver_1d(
            kernel=torch.ones(smooth_window),
            length_x=n_bins_val,
            pad_mode='same',
            correct_edge_effects=True,
            device='cpu',
        )

        ## Mutable containers so resample callback can update them in-place
        _state = {
            'sf_t': sf_t,
            'nn_t': nn_t,
            'swt_t': swt_t,
            'custom_t': custom_t,
            'intra_mask': intra_mask,
            'generation': 0,
        }
        _state['intra_indices'] = torch.where(_state['intra_mask'])[0]
        _state['sf_clamped'] = torch.clamp(_state['sf_t'], min=1e-8)
        _state['n_all'] = _state['sf_t'].shape[0]
        _state['n_intra'] = int(_state['intra_mask'].sum().item())
        _state['scale_factor'] = _state['n_all'] / max(_state['n_intra'], 1)

        ################################################################
        ## Resample callback — redraws subsampled tensors each generation
        ################################################################
        _generation_counter = [0]

        def _resample_callback(xk, convergence=None):
            """Redraw subsample at the start of each generation."""
            gen = _generation_counter[0]
            _generation_counter[0] += 1
            new_seed = (seed + gen * 1000 + 99999) if seed is not None else (gen * 1000 + 99999)
            sf_new, nn_new, swt_new, custom_new, im_new = _build_working_tensors(resample_seed=new_seed)
            _state['sf_t'] = sf_new
            _state['nn_t'] = nn_new
            _state['swt_t'] = swt_new
            _state['custom_t'] = custom_new
            _state['intra_mask'] = im_new
            _state['intra_indices'] = torch.where(im_new)[0]
            _state['sf_clamped'] = torch.clamp(sf_new, min=1e-8)
            _state['n_all'] = sf_new.shape[0]
            _state['n_intra'] = int(im_new.sum().item())
            _state['scale_factor'] = _state['n_all'] / max(_state['n_intra'], 1)

        ################################################################
        ## Scalar objective — evaluates one parameter vector at a time
        ################################################################
        ## Precompute the index offset where custom power params start
        ## in the parameter vector: after the 3 (frozen) or 7 (full) base params.
        _n_base_params = 3 if freeze_sigmoid else 7

        ## Frozen sigmoid params for custom metrics: use NB-estimated
        ## params when available (freeze_sigmoid=True triggers NB calibration
        ## which now includes custom metrics), else default (mu=0.5, b=1.0).
        _frozen_custom_sig = {}
        for name in _custom_names_sorted:
            if _frozen_sig is not None and name in sig_params:
                _frozen_custom_sig[name] = sig_params[name]
            else:
                _frozen_custom_sig[name] = {'mu': 0.5, 'b': 1.0}

        def objective_scalar(x):
            sf_w = _state['sf_t']
            nn_w = _state['nn_t']
            swt_w = _state['swt_t']
            ii = _state['intra_indices']
            sc = _state['scale_factor']

            ## Unpack parameter vector
            power_NN, power_SWT, p_norm_val = float(x[0]), float(x[1]), float(x[2])
            if _frozen_sig is not None:
                mu_NN = _frozen_sig['mu_NN']
                b_NN = _frozen_sig['b_NN']
                mu_SWT = _frozen_sig['mu_SWT']
                b_SWT = _frozen_sig['b_SWT']
            else:
                mu_NN, b_NN = float(x[3]), float(x[4])
                mu_SWT, b_SWT = float(x[5]), float(x[6])
            p = p_norm_val if abs(p_norm_val) > 1e-9 else 1e-9

            ## Compute NN activation
            nn_act = torch.sigmoid(b_NN * (nn_w - mu_NN))
            nn_act.clamp_(min=1e-8).pow_(power_NN)

            ## SWT activation
            swt_act = torch.sigmoid(b_SWT * (swt_w - mu_SWT))
            swt_act.clamp_(min=1e-8).pow_(power_SWT)

            ## p-norm mixing: distance = 1 - (mean(s_k^p))^(1/p)
            sf_p = torch.pow(_state['sf_clamped'], p)
            sum_sp = sf_p + torch.pow(nn_act, p) + torch.pow(swt_act, p)
            n_metrics = 3

            ## Include custom metrics in p-norm
            for i_custom, name in enumerate(_custom_names_sorted):
                custom_w = _state['custom_t'][name]
                power_custom = float(x[_n_base_params + i_custom])
                sig_kw = _frozen_custom_sig[name]
                custom_act = torch.sigmoid(sig_kw['b'] * (custom_w - sig_kw['mu']))
                custom_act = custom_act.clamp(min=1e-8).pow(power_custom)
                sum_sp = sum_sp + torch.pow(custom_act, p)
                n_metrics += 1

            dist = 1.0 - (sum_sp / n_metrics).pow(1.0 / p)

            ## Histogram overlap loss via shared helper
            loss, _, _ = self._compute_histogram_overlap(
                distances=dist,
                intra_indices=ii,
                edges=edges,
                smoother=smoother,
                scale_factor=sc,
            )

            return loss

        ################################################################
        ## Configure and run differential evolution
        ################################################################
        print('Finding mixing parameters using differential evolution...') if self._verbose else None

        de_kwargs_use = dict(de_kwargs)

        nnz_full = sf_t_full.shape[0]

        ## Always resample each generation when subsampling
        if subsample_pairs is not None and subsample_pairs < nnz_full:
            existing_cb = de_kwargs_use.pop('callback', None)
            def _combined_callback(xk, convergence=None):
                _resample_callback(xk, convergence)
                if existing_cb is not None:
                    return existing_cb(xk, convergence)  ## propagate stop signal
            de_kwargs_use['callback'] = _combined_callback

        self._de_result = scipy.optimize.differential_evolution(
            func=objective_scalar,
            bounds=scipy_bounds,
            seed=seed,
            **de_kwargs_use,
        )

        ## Extract best parameters
        x_best = self._de_result.x
        if _frozen_sig is not None:
            self.best_params = {
                'power_NN': float(x_best[0]),
                'power_SWT': float(x_best[1]),
                'p_norm': float(x_best[2]),
                'sig_NN_kwargs': {
                    'mu': _frozen_sig['mu_NN'],
                    'b': _frozen_sig['b_NN'],
                },
                'sig_SWT_kwargs': {
                    'mu': _frozen_sig['mu_SWT'],
                    'b': _frozen_sig['b_SWT'],
                },
            }
        else:
            self.best_params = {
                'power_NN': float(x_best[0]),
                'power_SWT': float(x_best[1]),
                'p_norm': float(x_best[2]),
                'sig_NN_kwargs': {'mu': float(x_best[3]), 'b': float(x_best[4])},
                'sig_SWT_kwargs': {'mu': float(x_best[5]), 'b': float(x_best[6])},
            }

        ## Extract custom metric powers from the parameter vector
        custom_powers_best = {}
        custom_sig_kwargs_best = {}
        for i_custom, name in enumerate(_custom_names_sorted):
            custom_powers_best[name] = float(x_best[_n_base_params + i_custom])
            custom_sig_kwargs_best[name] = _frozen_custom_sig[name]
        if custom_powers_best:
            self.best_params['custom_powers'] = custom_powers_best
            self.best_params['custom_sig_kwargs'] = custom_sig_kwargs_best

        self.kwargs_makeConjunctiveDistanceMatrix_best = {
            'power_SF': None,
            'power_NN': None,
            'power_SWT': None,
            'p_norm': None,
            'sig_SF_kwargs': None,
            'sig_NN_kwargs': None,
            'sig_SWT_kwargs': None,
        }
        self.kwargs_makeConjunctiveDistanceMatrix_best.update(self.best_params)

        print(
            f'Completed DE parameter search. '
            f'Best value: {self._de_result.fun:.2f}, '
            f'evaluations: {self._de_result.nfev}, '
            f'params: {self.best_params}'
        ) if self._verbose else None

        return self.kwargs_makeConjunctiveDistanceMatrix_best

    ####################################################################
    ## Naive Bayes calibration and sigmoid estimation
    ####################################################################

    def _calibrate_feature_1d(
        self,
        s_data: torch.Tensor,
        intra_mask: torch.Tensor,
        n_bins: int,
        smoother: 'helpers.Convolver_1d',
        prob_clip: Tuple[float, float] = (1e-4, 1 - 1e-4),
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate P(same | s_k) for a single similarity feature using
        histogram subtraction.

        Given raw similarity values for all pairs and the intra-session
        (known-different) subset, estimates the "same" distribution as the
        residual after subtracting the scaled intra-session distribution
        from the overall distribution. Monotonicity is enforced (higher
        similarity → higher P(same)).
        RH 2025

        Args:
            s_data (torch.Tensor):
                1D tensor of raw similarity values, shape ``(nnz,)``.
            intra_mask (torch.Tensor):
                Boolean tensor, shape ``(nnz,)``. ``True`` for
                intra-session (known-different) pairs.
            n_bins (int):
                Number of histogram bins.
            smoother (helpers.Convolver_1d):
                1D convolver for smoothing histogram counts.
            prob_clip (Tuple[float, float]):
                Clamp P(same) to ``[lo, hi]`` to avoid logit divergence.

        Returns:
            (Dict[str, torch.Tensor]):
                calibration (Dict[str, torch.Tensor]):
                    Dictionary with keys ``'edges'``, ``'counts_all'``,
                    ``'counts_diff'``, ``'counts_diff_smooth'``,
                    ``'counts_same'``, ``'p_same_bins'``.
        """
        n_all = s_data.shape[0]
        n_intra = int(intra_mask.sum().item())
        scale = n_all / n_intra

        ## Bin edges spanning the data range with small margin
        lo = float(s_data.min()) - 1e-6
        hi = float(s_data.max()) + 1e-6
        edges = torch.linspace(lo, hi, n_bins + 1, dtype=torch.float32)

        ## Histogram all values and intra-session values
        counts_all, _ = torch.histogram(s_data, edges)
        counts_intra, _ = torch.histogram(s_data[intra_mask], edges)

        ## Scale intra counts to estimate the full "different" distribution
        counts_diff = counts_intra * scale

        ## "Same" distribution = residual, clamped non-negative.
        ## Do NOT smooth counts_same — the smoothing kernel bleeds nonzero
        ## mass into the left tail where the true same-count is zero, creating
        ## an artificial P(same) floor. Raw residual preserves zeros.
        counts_same = torch.clamp(counts_all - counts_diff, min=0)

        ## Smooth only the "different" distribution (estimating the smooth
        ## population envelope). counts_same stays raw/clamped.
        counts_diff_smooth = smoother.convolve(counts_diff)

        ## P(same | bin) = counts_same / (counts_same + counts_diff_smooth)
        p_same_bins = counts_same / (counts_same + counts_diff_smooth + 1e-10)

        ## Enforce monotonic increasing via isotonic regression, weighted by
        ## total evidence per bin. Isotonic regression finds the best
        ## monotonically-increasing fit — sparse tail bins (with few
        ## observations) get negligible influence, avoiding artificial
        ## P(same) floors.
        ir = sklearn.isotonic.IsotonicRegression(
            increasing=True,
            y_min=float(prob_clip[0]),
            y_max=float(prob_clip[1]),
        )
        evidence_weights = (counts_all + counts_diff_smooth).numpy()
        evidence_weights = np.maximum(evidence_weights, 1e-10)
        p_same_np = ir.fit_transform(
            X=np.arange(n_bins, dtype=np.float64),
            y=p_same_bins.numpy().astype(np.float64),
            sample_weight=evidence_weights.astype(np.float64),
        )
        p_same_bins = torch.as_tensor(p_same_np, dtype=torch.float32)

        ## Store all tensors as numpy so that serializable_dict can
        ## preserve them (torch tensors are not in the allowed library list).
        ## Call sites that need torch tensors convert with torch.as_tensor().
        return {
            'edges': edges.numpy(),
            'counts_all': counts_all.numpy(),
            'counts_diff': counts_diff.numpy(),
            'counts_diff_smooth': counts_diff_smooth.numpy(),
            'counts_same': counts_same.numpy(),
            'p_same_bins': p_same_bins.numpy(),
        }

    def make_naive_bayes_distance_matrix(
        self,
        n_bins: Optional[int] = None,
        smoothing_window_bins: Optional[int] = None,
        prob_clip: Tuple[float, float] = (1e-4, 1 - 1e-4),
    ) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, Dict[str, Any]]:
        """
        Compute pairwise distance matrix using independent per-feature
        calibration combined via naive Bayes.

        For each similarity feature k (SF, NN, SWT), estimates the posterior
        ``P(same | s_k)`` from a 1D histogram of similarity values, using
        the intra-session (known-different) distribution as reference. The
        per-feature posteriors are combined under conditional independence:

        .. math::

            \\text{logit}(P(\\text{same} | \\mathbf{s})) = \\sum_k \\text{logit}(P(\\text{same} | s_k)) - (K-1) \\cdot \\text{logit}(\\pi)

        where :math:`\\pi` is the estimated prior P(same) and K is the
        number of features.

        **No iterative optimization** — just histogram + lookup. Typically
        completes in under 1 second even on large datasets.
        RH 2025

        Args:
            n_bins (Optional[int]):
                Number of histogram bins per feature. If ``None``, uses
                ``self.n_bins``.
            smoothing_window_bins (Optional[int]):
                Smoothing window. If ``None``, uses ``self.smooth_window``.
            prob_clip (Tuple[float, float]):
                Clamp P(same|s_k) to ``[lo, hi]`` before logit.

        Returns:
            (Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, Dict]):
                dConj (scipy.sparse.csr_matrix):
                    Distance matrix ``d = 1 - P(same|all)``.
                sConj (scipy.sparse.csr_matrix):
                    Similarity matrix ``s = P(same|all)``.
                calibrations (Dict[str, Any]):
                    Diagnostic dict with per-feature calibrations,
                    prior, and combined P(same).
        """
        self.params['make_naive_bayes_distance_matrix'] = self._locals_to_params(
            locals_dict=locals(),
            keys=['n_bins', 'smoothing_window_bins', 'prob_clip'],
        )

        n_bins = self.n_bins if n_bins is None else n_bins
        smooth_window = self.smooth_window if smoothing_window_bins is None else smoothing_window_bins

        print('Computing naive Bayes distance matrix...') if self._verbose else None

        ## Precompute intra-session mask
        if not hasattr(self, '_intra_mask') or self._intra_mask is None:
            self._precompute_intra_mask()
        intra_mask = torch.as_tensor(self._intra_mask)

        ## Build smoother (shared across features)
        smoother = helpers.Convolver_1d(
            kernel=torch.ones(smooth_window),
            length_x=n_bins,
            pad_mode='same',
            correct_edge_effects=True,
            device='cpu',
        )

        ## Features to calibrate: (name, raw similarity data)
        features_raw = {
            'SF': torch.as_tensor(self.s_sf.data, dtype=torch.float32),
            'NN': torch.as_tensor(self.s_NN_z.data, dtype=torch.float32),
            'SWT': torch.as_tensor(self.s_SWT_z.data, dtype=torch.float32),
        }
        ## Include custom similarity matrices in NB calibration
        for name in sorted(self.custom_similarities.keys()):
            features_raw[name] = torch.as_tensor(
                self.custom_similarities[name].data, dtype=torch.float32,
            )

        calibrations = {'features': {}}
        nnz = self.s_sf.nnz
        logit_sum = torch.zeros(nnz, dtype=torch.float32)

        ## Calibrate each feature independently
        for name, s_data in features_raw.items():
            cal = self._calibrate_feature_1d(
                s_data=s_data,
                intra_mask=intra_mask,
                n_bins=n_bins,
                smoother=smoother,
                prob_clip=prob_clip,
            )

            ## Look up P(same) for each pair from its histogram bin.
            ## cal values are numpy arrays; convert to torch for the lookup
            ## then store result as numpy for serialization safety.
            edges_t = torch.as_tensor(cal['edges'])
            p_same_bins_t = torch.as_tensor(cal['p_same_bins'])
            bin_idx = torch.searchsorted(
                edges_t[1:-1].contiguous(), s_data,
            )
            bin_idx = torch.clamp(bin_idx, 0, n_bins - 1)
            p_same_per_pair = p_same_bins_t[bin_idx]  ## torch, shape (nnz,)

            ## Accumulate logit for naive Bayes combination
            logit_p = torch.log(p_same_per_pair / (1.0 - p_same_per_pair))
            logit_sum += logit_p

            ## Store as numpy so serializable_dict preserves it
            cal['p_same_per_pair'] = p_same_per_pair.numpy()
            calibrations['features'][name] = cal

            print(
                f'  {name}: P(same) range '
                f'[{p_same_per_pair.min():.4f}, {p_same_per_pair.max():.4f}], '
                f'mean={p_same_per_pair.mean():.4f}'
            ) if self._verbose else None

        ## Estimate prior P(same)
        K = len(features_raw)
        prior_estimates = []
        for cal in calibrations['features'].values():
            total_same = cal['counts_same'].sum().item()
            total = total_same + cal['counts_diff_smooth'].sum().item()
            if total > 0:
                prior_estimates.append(total_same / total)
        prior = float(np.mean(prior_estimates)) if prior_estimates else 0.5
        prior = float(np.clip(prior, prob_clip[0], prob_clip[1]))
        calibrations['prior'] = prior

        ## Naive Bayes log-odds combination
        logit_prior = float(np.log(prior / (1.0 - prior)))
        logit_combined = logit_sum - (K - 1) * logit_prior

        ## Convert back to probability
        p_same_combined = torch.sigmoid(logit_combined).numpy()
        calibrations['p_same_combined'] = p_same_combined

        ## Build sparse similarity and distance matrices
        sConj = self.s_sf.copy()
        sConj.data = p_same_combined.astype(np.float64)

        dConj = sConj.copy()
        dConj.data = 1.0 - dConj.data

        ## Store for downstream use
        self.dConj = dConj
        self.sConj = sConj
        self.calibrations_naive_bayes = calibrations

        print(
            f'  Combined P(same): mean={p_same_combined.mean():.4f}, '
            f'prior={prior:.4f}, '
            f'P(same)>0.5: {(p_same_combined > 0.5).sum()}/{nnz} '
            f'({(p_same_combined > 0.5).mean() * 100:.1f}%)'
        ) if self._verbose else None

        return dConj, sConj, calibrations

    def _estimate_sigmoid_params(self) -> Dict[str, Dict[str, float]]:
        """
        Estimate sigmoid parameters (mu, b) for NN and SWT from
        NB calibration curves using Fisher's linear discriminant.

        For each feature, finds the sigmoid ``sigma(b * (s - mu))`` that
        best separates "same" and "different" distributions in the
        calibration histogram. Uses a grid search over (mu, b) to maximize
        the Fisher discriminant ratio in sigmoid-transformed space.

        Requires :meth:`make_naive_bayes_distance_matrix` to have been
        called first.
        RH 2025

        Returns:
            (Dict[str, Dict[str, float]]):
                sigmoid_params (Dict[str, Dict[str, float]]):
                    Mapping from feature name to ``{'mu': float, 'b': float}``.
        """
        assert hasattr(self, 'calibrations_naive_bayes') and self.calibrations_naive_bayes is not None, (
            "make_naive_bayes_distance_matrix() must be called before "
            "_estimate_sigmoid_params()."
        )

        result = {}
        ## Estimate sigmoid params for built-in features and any custom
        ## similarities that were included in the NB calibration.
        feature_names = ['NN', 'SWT'] + sorted(self.custom_similarities.keys())
        for name in feature_names:
            if name not in self.calibrations_naive_bayes['features']:
                continue
            cal = self.calibrations_naive_bayes['features'][name]
            ## cal values are numpy arrays (stored that way for serialization)
            edges = np.asarray(cal['edges'])
            counts_same = np.asarray(cal['counts_same'])
            counts_diff = np.asarray(cal['counts_diff_smooth'])

            ## Bin centers in similarity space, shape (n_bins,)
            centers_np = (edges[:-1] + edges[1:]) / 2.0

            ## Normalized distribution weights, shape (n_bins,)
            w_same = counts_same / (counts_same.sum() + 1e-10)
            w_diff = counts_diff / (counts_diff.sum() + 1e-10)

            ## Vectorized grid search over (mu, b) to maximize Fisher
            ## discriminant in sigmoid-transformed space.
            ## Grid shapes: mu (M,), b (B,) → sig_vals (M, B, n_bins)
            mu_grid = np.linspace(
                float(centers_np.min()), float(centers_np.max()), 50,
            )
            b_grid = np.linspace(0.5, 10.0, 30)
            ## Broadcasting: (M,1,1) * ((1,1,n_bins) - (M,1,1))
            sig_vals = 1.0 / (1.0 + np.exp(
                -b_grid[None, :, None] * (centers_np[None, None, :] - mu_grid[:, None, None])
            ))  ## shape (M, B, n_bins)

            ## Weighted moments in sigmoid-transformed space
            mu_same_sig = np.sum(w_same[None, None, :] * sig_vals, axis=2)  ## (M, B)
            mu_diff_sig = np.sum(w_diff[None, None, :] * sig_vals, axis=2)  ## (M, B)
            var_same_sig = np.sum(w_same[None, None, :] * (sig_vals - mu_same_sig[:, :, None]) ** 2, axis=2)
            var_diff_sig = np.sum(w_diff[None, None, :] * (sig_vals - mu_diff_sig[:, :, None]) ** 2, axis=2)

            ## Fisher discriminant ratio, shape (M, B)
            denom = var_same_sig + var_diff_sig + 1e-12
            fisher_grid = (mu_same_sig - mu_diff_sig) ** 2 / denom

            ## Find best (mu, b)
            best_idx = np.unravel_index(fisher_grid.argmax(), fisher_grid.shape)
            best_mu = float(mu_grid[best_idx[0]])
            best_b = float(b_grid[best_idx[1]])
            best_fisher = float(fisher_grid[best_idx])

            result[name] = {'mu': best_mu, 'b': best_b}

            print(
                f'  Sigmoid estimate for {name}: '
                f'mu={best_mu:.4f}, b={best_b:.2f} '
                f'(Fisher={best_fisher:.4f})'
            ) if self._verbose else None

        return result

    def make_pruned_similarity_graphs(
        self,
        convert_to_probability: bool = False,
        stringency: float = 1.0,
        kwargs_makeConjunctiveDistanceMatrix: Optional[Dict] = None,
        d_cutoff: Optional[float] = None,
    ) -> None:
        """
        Constructs pruned similarity graphs.
        RH 2023

        Args:
            convert_to_probability (bool): 
                Whether to convert the distance and similarity graphs to
                probability, *p(different)* and *p(same)*, respectively.
                (Default is ``False``)
            stringency (float): 
                Modifies the threshold for pruning the distance matrix. A higher
                value results in less pruning, a lower value leads to more
                pruning. This value is multiplied by the inferred threshold to
                generate a new one. (Default is *1.0*)
            kwargs_makeConjunctiveDistanceMatrix (Optional[Dict]): 
                Keyword arguments for the
                ``self.make_conjunctive_distance_matrix`` function. If ``None``,
                the best parameters found using ``self.find_optimal_parameters``
                are used. (Default is ``None``)
            d_cutoff (Optional[float]): 
                The cutoff distance for pruning the distance matrix. If
                ``None``, then the optimal cutoff distance is inferred. (Default
                is ``None``)
        """
        ## Store parameter (but not data) args as attributes
        self.params['make_pruned_similarity_graphs'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'convert_to_probability',
                'stringency',
                'kwargs_makeConjunctiveDistanceMatrix',
            ],
        )

        ## If 'precomputed', use self.dConj/sConj set by a prior call
        ## (e.g. make_naive_bayes_distance_matrix). Otherwise, compute
        ## the conjunctive distance matrix from mixing parameters.
        if kwargs_makeConjunctiveDistanceMatrix == 'precomputed':
            assert hasattr(self, 'dConj') and self.dConj is not None, (
                "kwargs_makeConjunctiveDistanceMatrix='precomputed' requires "
                "self.dConj to be set (call make_naive_bayes_distance_matrix first)."
            )
        elif kwargs_makeConjunctiveDistanceMatrix is None:
            if hasattr(self, 'kwargs_makeConjunctiveDistanceMatrix_best'):
                kwargs_makeConjunctiveDistanceMatrix = self.kwargs_makeConjunctiveDistanceMatrix_best
            else:
                kwargs_makeConjunctiveDistanceMatrix = {
                    'power_SF': 0.5,
                    'power_NN': 1.0,
                    'power_SWT': 0.1,
                    'p_norm': -4.0,
                    'sig_SF_kwargs': {'mu':0.5, 'b':0.5},
                    'sig_NN_kwargs': {'mu':0.5, 'b':0.5},
                    'sig_SWT_kwargs': {'mu':0.5, 'b':0.5},
                }
                warnings.warn(f'No kwargs_makeConjunctiveDistanceMatrix provided. Using default parameters: {kwargs_makeConjunctiveDistanceMatrix}')

        if kwargs_makeConjunctiveDistanceMatrix != 'precomputed':
            self.dConj, self.sConj, sSF_data, sNN_data, sSWT_data, sConj_data = self.make_conjunctive_distance_matrix(
                s_sf=self.s_sf,
                s_NN=self.s_NN_z,
                s_SWT=self.s_SWT_z,
                s_sesh=None,
                **kwargs_makeConjunctiveDistanceMatrix
            )
        dens_same_crop, dens_same, dens_diff, dens_all, edges, d_crossover = self._separate_diffSame_distributions(self.dConj)

        if convert_to_probability:        
            ## convert into probabilities
            ### first smooth dens_diff. (dens_same is already smoothed)
            dens_diff_smooth = self._fn_smooth(dens_diff)
            ### second, compute the probability of each bin
            prob_same = (dens_same / (dens_same + dens_diff_smooth)).numpy()
            ### force to be monotonic decreasing
            prob_same = np.maximum.accumulate(prob_same[::-1])[::-1]
            ### third, append 0 to the end
            prob_same = np.append(prob_same, 0)
            ### third, convert self.dConj to probabilities using interpolation
            import scipy.interpolate
            fn_interp = scipy.interpolate.interp1d(edges, prob_same, kind='linear', fill_value='extrapolate')
            self.sConj.data = fn_interp(self.dConj.data)
            self.dConj.data = 1 - self.sConj.data
            d_crossover = 1 - fn_interp(d_crossover)

        self.distributions_mixing = {
            'kwargs_makeConjunctiveDistanceMatrix': kwargs_makeConjunctiveDistanceMatrix,
            'dens_same_crop': dens_same_crop,
            'dens_same': dens_same,
            'dens_diff': dens_diff,
            'dens_all': dens_all,
            'edges': edges,
            'd_crossover': d_crossover,
        }

        ssf, snn, sswt, ssesh = self.s_sf.copy(), self.s_NN_z.copy(), self.s_SWT_z.copy(), self.s_sesh.copy()

        min_d = np.nanmin(self.dConj.data)
        if d_cutoff is None:
            range_d = d_crossover - min_d
            self.d_cutoff = min_d + range_d * stringency
        print(f'Pruning similarity graphs with d_cutoff = {self.d_cutoff}...') if self._verbose else None

        self.graph_pruned = self.dConj.copy()
        self.graph_pruned.data = self.graph_pruned.data < self.d_cutoff
        self.graph_pruned.eliminate_zeros()
        
        def prune(s, graph_pruned):
            import scipy.sparse
            if s is None:
                return None
            s_pruned = scipy.sparse.csr_matrix(graph_pruned.shape, dtype=np.float32)
            s_pruned[graph_pruned] = s[graph_pruned]
            s_pruned = s_pruned.tocsr()
            return s_pruned

        self.s_sf_pruned, self.s_NN_pruned, self.s_SWT_pruned, self.s_sesh_pruned = tuple([prune(s, self.graph_pruned) for s in [ssf, snn, sswt, ssesh]])
        self.dConj_pruned, self.sConj_pruned = prune(self.dConj, self.graph_pruned), prune(self.sConj, self.graph_pruned)

    def fit(
        self,
        d_conj: scipy.sparse.csr_matrix,
        session_bool: np.ndarray,
        min_cluster_size: int = 2,
        n_iter_violationCorrection: int = 5,
        cluster_selection_method: str = 'leaf',
        d_clusterMerge: Optional[float] = None,
        alpha: float = 0.999,
        split_intraSession_clusters: bool = True,
        discard_failed_pruning: bool = True,
        n_steps_clusterSplit: int = 100,
    ) -> np.ndarray:
        """
        Fits clustering using a modified HDBSCAN clustering algorithm.
        The approach is to use HDBSCAN but avoid having clusters with multiple ROIs from 
        the same session. This is achieved by repeating three steps: \n
        1. Fit HDBSCAN to the data. 
        2. Identify clusters that have multiple ROIs from the same session
           and walk back down the dendrogram until those clusters are split up
           into non-violating clusters. 
        3. Disconnect graph edges between ROIs within each new cluster and all
           other ROIs outside the cluster that are from the same session. \n

        Args:
            d_conj (scipy.sparse.csr_matrix): 
                Conjunctive distance matrix.
            session_bool (np.ndarray): 
                Boolean array indicating which ROIs belong to which session.
                Shape: *(n_rois, n_sessions)*
            min_cluster_size (int): 
                Minimum cluster size to be considered a cluster. Can be 'all'.
                (Default is *2*)
            n_iter_violationCorrection (int): 
                Number of iterations to correct for clusters with multiple ROIs
                per session. This is done to overcome the issues with
                single-linkage clustering finding clusters with multiple ROIs
                per session. (Default is *5*)
            cluster_selection_method (str): 
                Cluster selection method. Either ``'leaf'`` or ``'eom'``. 'leaf'
                leans towards smaller clusters, 'eom' towards larger clusters.
                (Default is ``'leaf'``)
            d_clusterMerge (Optional[float]): 
                Distance threshold for merging clusters. All clusters with ROIs
                closer than this distance will be merged. If ``None``, the
                distance is calculated as the mean + 1*std of the conjunctive
                distances. (Default is ``None``)
            alpha (float): 
                Alpha value. Smaller values result in more clusters. (Default is
                *0.999*)
            split_intraSession_clusters (bool): 
                If ``True``, clusters containing ROIs from multiple sessions
                will be split. Only set to ``False`` if you want clusters
                containing multiple ROIs from the same session. (Default is
                ``True``)
            discard_failed_pruning (bool): 
                If ``True``, clusters failing to prune are set to -1. (Default
                is ``True``)
            n_steps_clusterSplit (int): 
                Number of steps for splitting clusters with multiple ROIs from
                the same session. Lower values are faster but less accurate.
                (Default is *100*)

        Returns:
            (np.ndarray): 
                labels (np.ndarray): 
                    Cluster labels for each ROI, shape: *(n_rois_total)*
        """
        ## Store parameter (but not data) args as attributes
        self.params['fit'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'min_cluster_size',
                'n_iter_violationCorrection',
                'cluster_selection_method',
                'd_clusterMerge',
                'alpha',
                'split_intraSession_clusters',
                'discard_failed_pruning',
                'n_steps_clusterSplit',
            ],
        )

        import hdbscan
        d = d_conj.copy().multiply(self.s_sesh)

        if d.nnz == 0:
            print('No edges in graph. Returning all -1 labels.') if self._verbose else None
            self.labels = np.ones(d.shape[0], dtype=int) * -1
            return self.labels
            
        n_sessions = session_bool.shape[1]
        if min_cluster_size == 'all':
            min_cluster_size = n_sessions
            print(f'Setting min_cluster_size to {min_cluster_size} (all ROIs in a session)') if self._verbose else None

        print('Fitting with HDBSCAN and splitting clusters with multiple ROIs per session') if self._verbose else None
        for ii in tqdm(range(n_iter_violationCorrection)):
            ## Prep parameters for splitting clusters
            d_clusterMerge = float(np.mean(d.data) + 1*np.std(d.data)) if d_clusterMerge is None else float(d_clusterMerge)
            n_steps_clusterSplit = int(n_steps_clusterSplit)

            max_dist=(d.max() - d.min()) * 1000

            self.hdbs = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                cluster_selection_epsilon=d_clusterMerge,
                max_cluster_size=n_sessions,
                metric='precomputed',
                alpha=alpha,
                algorithm='generic',
                cluster_selection_method=cluster_selection_method,
                max_dist=max_dist,
            )

            self.hdbs.fit(attach_fully_connected_node(
                d, 
                dist_fullyConnectedNode=max_dist,
                n_nodes=1,
            ))
            labels = self.hdbs.labels_[:-1]
            self.labels = labels

            print(f'Initial number of violating clusters: {len(np.unique(labels)[np.array([(session_bool[labels==u].sum(0)>1).sum().item() for u in np.unique(labels)]) > 0])}, d_clusterMerge={d_clusterMerge:.2f}') if self._verbose else None

            ## Split up labels with multiple ROIs per session
            ## The below code is a bit of a mess, but it works.
            ##  It works by iteratively reducing the cutoff distance
            ##  and splitting up violating clusters until there are 
            ##  no more violations.
            if split_intraSession_clusters:
                labels = labels.copy()

                sb_t = torch.as_tensor(session_bool, dtype=torch.float32)  ## (n_rois, n_sessions)
                n = len(d_conj.data)
                dcd = np.sort(d_conj.data)
                cuts_all = np.sort(np.unique([dcd[0]/2] + [dcd[int(n*ii)] for ii in np.linspace(0., 1., num=n_steps_clusterSplit, endpoint=False)[::-1]] + [dcd[-1]]))[::-1]
                for d_cut in cuts_all:
                    labels_t = torch.as_tensor(labels, dtype=torch.int64)
                    lab_u_t, lab_u_idx_t = torch.unique(labels_t, return_inverse=True) # (n_clusters,), (n_rois,)
                    lab_oneHot_t = helpers.idx_to_oneHot(lab_u_idx_t, dtype=torch.float32)
                    violations_labels = lab_u_t[((sb_t.T @ lab_oneHot_t) > 1.5).sum(0) > 0]
                    violations_labels = violations_labels[violations_labels > -1]

                    if len(violations_labels) == 0:
                        break
                    
                    for l in violations_labels:
                        idx = np.where(labels==l)[0]
                        if d[idx][:,idx].nnz == 0:
                            labels[idx] = -1

                    labels_new = self.hdbs.single_linkage_tree_.get_clusters(
                        cut_distance=d_cut,
                        min_cluster_size=min_cluster_size,
                    )[:-1]
                    
                    idx_toUpdate = np.isin(labels, violations_labels)
                    labels[idx_toUpdate] = labels_new[idx_toUpdate] + labels.max() + 5
                    labels[(labels_new == -1) * idx_toUpdate] = -1
                
                if discard_failed_pruning:
                    labels[idx_toUpdate] = -1

            l_u = np.unique(labels)
            l_u = l_u[l_u > -1]
            if ii < n_iter_violationCorrection - 1:
                ## Find sessions represented in each cluster and set distances to ROIs in those sessions to 1.
                d = d.tocsr()
                for ii, l in enumerate(np.unique(labels)):
                    if l == -1:
                        continue
                    idx = np.where(labels==l)[0]
                    
                    d_sub = d[idx][:,idx]
                    idx_grid = np.meshgrid(idx, idx)
                    ## set distances of ROIs from same session to 0
                    sesh_to_exclude = 1 - (session_bool @ (session_bool[idx].max(0)))  ## make a mask of sessions that are not represented in the cluster
                    d[idx,:] = d[idx,:].multiply(sesh_to_exclude[None,:])  ## set distances to ROIs from sessions represented in the cluster to 1
                    d[:,idx] = d[:,idx].multiply(sesh_to_exclude[:,None])  ## set distances to ROIs from sessions represented in the cluster to 1
                    d[idx_grid[0], idx_grid[1]] = d_sub  ## undo the above for ROIs in the cluster
                d = d.tocsr()
                d.eliminate_zeros()  ## remove zeros


        labels = helpers.squeeze_integers(labels)
        
        violations_labels = np.unique(labels)[np.array([(session_bool[labels==u].sum(0)>1).sum().item() for u in np.unique(labels)]) > 0]
        violations_labels = violations_labels[violations_labels > -1]
        self.violations_labels = violations_labels

        ## Set clusters with too few ROIs to -1
        u, c = np.unique(labels, return_counts=True)
        labels[np.isin(labels, u[c<2])] = -1
        labels = helpers.squeeze_integers(labels)

        self.labels = labels
        return self.labels

    def fit_sequentialHungarian(
        self,
        d_conj: scipy.sparse.csr_matrix,
        session_bool: np.ndarray,
        thresh_cost: float = 0.95,
    ) -> np.ndarray:
        """
        Applies CaImAn's method for clustering. 
        
        For further details, please refer to:
            * [CaImAn's paper](https://elifesciences.org/articles/38173#s4)
            * [CaImAn's repository](https://github.com/flatironinstitute/CaImAn)
            * [Relevant script in CaImAn's repository](https://github.com/flatironinstitute/CaImAn/blob/master/caiman/base/rois.py)

        Args:
            d_conj (scipy.sparse.csr_matrix): 
                Distance matrix. 
                Shape: *(n_rois, n_rois)*
            session_bool (np.ndarray): 
                Boolean array indicating which ROIs are in which sessions. 
                Shape: *(n_rois, n_sessions)*
            thresh_cost (float): 
                Threshold below which ROI pairs are considered potential matches. 
                (Default is *0.95*)

        Returns:
            (np.ndarray): 
                labels (np.ndarray): 
                    Cluster labels. Shape: *(n_rois,)*
        """
        ## Store parameter (but not data) args as attributes
        self.params['fit_sequentialHungarian'] = self._locals_to_params(
            locals_dict=locals(),
            keys=['thresh_cost',],)

        print(f"Clustering with CaImAn's sequential Hungarian algorithm method...") if self._verbose else None
        def find_matches(D_s):
            matches = []
            costs = []
            for ii, D in enumerate(D_s):
                # we make a copy not to set changes in the original
                DD = D.copy()
                if np.sum(np.where(np.isnan(DD))) > 0:
                    raise Exception('Distance Matrix contains invalid value NaN')

                # we do the hungarian
                indexes = scipy.optimize.linear_sum_assignment(DD)
                indexes2 = [(ind1, ind2) for ind1, ind2 in zip(indexes[0], indexes[1])]
                matches.append(indexes)
                total = []
                # we want to extract those informations from the hungarian algo
                for row, column in indexes2:
                    value = DD[row, column]
                    total.append(value)
                costs.append(total)
                # send back the results in the format we want
            return matches, costs

        n_roi = session_bool.sum(0)
        n_roi_cum = np.concatenate(([0], np.cumsum(n_roi)))
        
        matchings = []
        matchings.append(list(range(n_roi[0])))

        idx_union = np.arange(n_roi[0])

        for i_sesh in tqdm(range(1,len(n_roi))):
            
            idx_sess = np.arange(n_roi_cum[i_sesh], n_roi_cum[i_sesh+1])
            
            d_sub = d_conj[idx_sess][:, idx_union]
            D = np.ones((len(idx_sess), len(idx_union)))*np.logical_not((d_sub != 0).toarray())*1 + d_sub.toarray()
            D = [D]
            
            matches, costs = find_matches(D)
            matches = matches[0]
            costs = costs[0]

            # store indices
            idx_tp = np.where(np.array(costs) < thresh_cost)[0]
            if len(idx_tp) > 0:
                matched_ROIs1 = matches[0][idx_tp]     # ground truth
                matched_ROIs2 = matches[1][idx_tp]     # algorithm - comp
                non_matched1 = np.setdiff1d(list(range(D[0].shape[0])), matches[0][idx_tp])
                non_matched2 = np.setdiff1d(list(range(D[0].shape[1])), matches[1][idx_tp])
                TP = np.sum(np.array(costs) < thresh_cost) * 1.
            else:
                TP = 0.
                matched_ROIs1 = []
                matched_ROIs2 = []
                non_matched1 = list(range(D[0].shape[0]))
                non_matched2 = list(range(D[0].shape[1]))

            # compute precision and recall
            FN = D[0].shape[0] - TP
            FP = D[0].shape[1] - TP
            TN = 0

            performance = dict()
            performance['recall'] = TP / (TP + FN)
            performance['precision'] = TP / (TP + FP)
            performance['accuracy'] = (TP + TN) / (TP + FP + FN + TN)
            performance['f1_score'] = 2 * TP / (2 * TP + FP + FN)

            mat_sess, mat_un, nm_sess, nm_un, performance, A2_len = matched_ROIs1, matched_ROIs2, non_matched1, non_matched2, performance, len(idx_union)
            
            idx_union[mat_un] = idx_sess[mat_sess]
            idx_union = np.concatenate((idx_union, idx_sess[nm_sess]))
            
            new_match = np.zeros(n_roi[i_sesh], dtype=int)
            new_match[mat_sess] = mat_un
            new_match[nm_sess] = range(A2_len, len(idx_union))
            matchings.append(new_match.tolist())

        self.seqHung_performance = performance

        labels = np.concatenate(matchings)
        u, c = np.unique(labels, return_counts=True)
        labels[np.isin(labels, u[c == 1])] = -1
        labels = helpers.squeeze_integers(labels)
        self.labels = labels
        return self.labels
            
    def make_conjunctive_distance_matrix(
        self,
        s_sf: Optional[scipy.sparse.csr_matrix] = None,
        s_NN: Optional[scipy.sparse.csr_matrix] = None,
        s_SWT: Optional[scipy.sparse.csr_matrix] = None,
        s_sesh: Optional[scipy.sparse.csr_matrix] = None,
        power_SF: float = 1,
        power_NN: float = 1,
        power_SWT: float = 1,
        p_norm: float = 1,
        sig_SF_kwargs: Dict[str, float] = {'mu':0.5, 'b':0.5},
        sig_NN_kwargs: Dict[str, float] = {'mu':0.5, 'b':0.5},
        sig_SWT_kwargs: Dict[str, float] = {'mu':0.5, 'b':0.5},
        custom_powers: Optional[Dict[str, float]] = None,
        custom_sig_kwargs: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Makes a distance matrix from the three similarity matrices.
        RH 2023

        Args:
            s_sf (Optional[scipy.sparse.csr_matrix]):
                Similarity matrix for spatial footprints. (Default is ``None``)
            s_NN (Optional[scipy.sparse.csr_matrix]):
                Similarity matrix for neural network features. (Default is
                ``None``)
            s_SWT (Optional[scipy.sparse.csr_matrix]):
                Similarity matrix for scattering wavelet transform features.
                (Default is ``None``)
            s_sesh (Optional[scipy.sparse.csr_matrix]):
                The session similarity matrix. (Default is ``None``)
            power_SF (float):
                Power to which to raise the spatial footprint similarity.
                (Default is *1*)
            power_NN (float):
                Power to which to raise the neural network similarity. (Default
                is *1*)
            power_SWT (float):
                Power to which to raise the scattering wavelet transform
                similarity. (Default is *1*)
            p_norm (float):
                p-norm to use for the conjunction of the similarity matrices.
                (Default is *1*)
            sig_SF_kwargs (Dict[str, float]):
                Keyword arguments for the sigmoid function applied to the
                spatial footprint overlap similarity matrix. See
                helpers.generalised_logistic_function for details. (Default is
                {'mu':0.5, 'b':0.5})
            sig_NN_kwargs (Dict[str, float]):
                Keyword arguments for the sigmoid function applied to the neural
                network similarity matrix. See
                helpers.generalised_logistic_function for details. (Default is
                {'mu':0.5, 'b':0.5})
            sig_SWT_kwargs (Dict[str, float]):
                Keyword arguments for the sigmoid function applied to the
                scattering wavelet transform similarity matrix. See
                helpers.generalised_logistic_function for details. (Default is
                {'mu':0.5, 'b':0.5})
            custom_powers (Optional[Dict[str, float]]):
                Per-metric power exponents for custom similarity matrices.
                Keys must match ``self.custom_similarities``. Missing keys
                default to ``1.0``. (Default is ``None``)
            custom_sig_kwargs (Optional[Dict[str, Dict[str, float]]]):
                Per-metric sigmoid parameters for custom similarity matrices.
                Keys must match ``self.custom_similarities``. Missing keys
                default to ``{'mu': 0.5, 'b': 1.0}``. (Default is ``None``)

        Returns:
            (Tuple): Tuple containing:
                dConj (scipy.sparse.csr_matrix):
                    Conjunction of the three similarity matrices.
                sConj (scipy.sparse.csr_matrix):
                    The session similarity matrix.
                sSF_data (np.ndarray):
                    Activated spatial footprint similarity matrix.
                sNN_data (np.ndarray):
                    Activated neural network similarity matrix.
                sSWT_data (np.ndarray):
                    Activated scattering wavelet transform similarity matrix.
                sConj_data (np.ndarray):
                    Activated session similarity matrix.
        """
        assert (s_sf is not None) or (s_NN is not None) or (s_SWT is not None), \
            'At least one of s_sf, s_NN, or s_SWT must be provided.'

        ## Store parameter (but not data) args as attributes
        self.params['make_conjunctive_distance_matrix'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'power_SF',
                'power_NN',
                'power_SWT',
                'p_norm',
                'sig_SF_kwargs',
                'sig_NN_kwargs',
                'sig_SWT_kwargs',
                'custom_powers',
                'custom_sig_kwargs',
            ],
        )

        p_norm = 1e-9 if p_norm == 0 else p_norm

        sSF_data = self._activation_function(s_sf.data, sig_SF_kwargs, power_SF) if s_sf is not None else None
        sNN_data = self._activation_function(s_NN.data, sig_NN_kwargs, power_NN) if s_NN is not None else None

        sSWT_data = self._activation_function(s_SWT.data, sig_SWT_kwargs, power_SWT) if s_SWT is not None else None

        s_list = [s for s in [sSF_data, sNN_data, sSWT_data] if s is not None]

        ## Handle custom similarities: apply sigmoid → power activation and
        ## include in the p-norm combination alongside built-in metrics.
        custom_powers = custom_powers or {}
        custom_sig_kwargs = custom_sig_kwargs or {}
        self._custom_activated = {}
        for name in sorted(self.custom_similarities.keys()):
            s_custom = self.custom_similarities[name]
            power = custom_powers.get(name, 1.0)
            sig_kw = custom_sig_kwargs.get(name, {'mu': 0.5, 'b': 1.0})
            activated = self._activation_function(s_custom.data, sig_kw, power)
            self._custom_activated[name] = activated
            s_list.append(activated)

        sConj_data = self._pNorm(
            s_list=s_list,
            p=p_norm,
        )

        ## make sConj
        sConj = s_sf.copy() if s_sf is not None else s_NN.copy() if s_NN is not None else s_SWT.copy()
        sConj.data = sConj_data.numpy()
        sConj = sConj.multiply(s_sesh) if s_sesh is not None else sConj

        ## make dConj
        dConj = sConj.copy()
        dConj.data = 1 - dConj.data

        return dConj, sConj, sSF_data, sNN_data, sSWT_data, sConj_data

    def _activation_function(
        self, 
        s: Optional[torch.Tensor] = None, 
        sig_kwargs: Optional[Dict[str, float]] = {'mu':0.0, 'b':1.0}, 
        power: Optional[float] = 1
    ) -> Optional[torch.Tensor]:
        """
        Applies an activation function to a similarity matrix.

        Args:
            s (Optional[torch.Tensor]): 
                The input similarity matrix. If ``None``, the function returns
                ``None``. (Default is ``None``)
            sig_kwargs (Dict[str, float]): 
                Keyword arguments for the sigmoid function applied to the
                similarity matrix. See helpers.generalised_logistic_function for
                details. (Default is {'mu':0.0, 'b':1.0})
            power (Optional[float]): 
                Power to which to raise the similarity. If ``None``, the power
                operation is not applied. (Default is *1*)

        Returns:
            (Optional[torch.Tensor]): 
                Activated similarity matrix. Returns ``None`` if the input
                similarity matrix is ``None``.
        """
        if s is None:
            return None
        
        s = torch.as_tensor(s, dtype=torch.float32)
        ## make functions such that if the param is None, then no operation is applied
        fn_sigmoid = lambda x, params: helpers.generalised_logistic_function(x, **params) if params is not None else x
        fn_power = lambda x, p: x ** p if p is not None else x
        
        return fn_power(torch.clamp(fn_sigmoid(s, sig_kwargs), min=0), power)

    def _pNorm(
        self, 
        s_list: List[Optional[torch.Tensor]], 
        p: float
    ) -> torch.Tensor:
        """
        Calculate the p-norm of a list of similarity matrices.

        Args:
            s_list (List[Optional[torch.Tensor]]): 
                List of similarity matrices.
            p (float): 
                p-norm to use.

        Returns:
            (torch.Tensor): 
                p-norm of the list of similarity matrices.
        """
        s_list_noNones = [s for s in s_list if s is not None]
        return (torch.mean(torch.stack(s_list_noNones, axis=0)**p, dim=0))**(1/p)

    def plot_similarity_relationships(
        self, 
        plots_to_show: List[int] = [1,2,3], 
        max_samples: int = 1000000, 
        kwargs_scatter: Dict[str, Union[int, float]] = {'s':1, 'alpha':0.1},
        kwargs_makeConjunctiveDistanceMatrix: Dict[str, Union[float, Dict[str, float]]] = {
            'power_SF': 0.5,
            'power_NN': 1.0,
            'power_SWT': 0.1,
            'p_norm': -4.0,
            'sig_SF_kwargs': {'mu':0.5, 'b':0.5},
            'sig_NN_kwargs': {'mu':0.5, 'b':0.5},
            'sig_SWT_kwargs': {'mu':0.5, 'b':0.5},
        },
    ) -> Tuple[plt.figure, plt.axes]:
        """
        Plot the similarity relationships between the three similarity matrices.

        Args:
            plots_to_show (List[int]): 
                Which plots to show. \n
                * *1*: Spatial footprints vs. neural network features.
                * *2*: Spatial footprints vs. scattering wavelet transform features.
                * *3*: Neural network features vs. scattering wavelet. \n
            max_samples (int): 
                Maximum number of samples to plot. Use smaller numbers for faster plotting. 
            kwargs_scatter (Dict[str, Union[int, float]]): 
                Keyword arguments for the matplotlib.pyplot.scatter plot. 
            kwargs_makeConjunctiveDistanceMatrix (Dict[str, Union[float, Dict[str, float]]]): 
                Keyword arguments for the makeConjunctiveDistanceMatrix method. 

        Returns:
            (Tuple[matplotlib.pyplot.figure, matplotlib.pyplot.axes]): tuple containing:
                fig (matplotlib.pyplot.figure): 
                    Figure object.
                axs (matplotlib.pyplot.axes): 
                    Axes object.
        """
        dConj, sConj, sSF_data, sNN_data, sSWT_data, sConj_data = self.make_conjunctive_distance_matrix(
            s_sf=self.s_sf,
            s_NN=self.s_NN_z,
            s_SWT=self.s_SWT_z,
            s_sesh=None,
            **kwargs_makeConjunctiveDistanceMatrix
        )

        ## subsample similarities for plotting
        idx_rand = np.floor(np.random.rand(min(max_samples, len(dConj.data))) * len(dConj.data)).astype(int)
        ssf_sub = sSF_data[idx_rand]
        snn_sub = sNN_data[idx_rand]
        sswt_sub = sSWT_data[idx_rand] if sSWT_data is not None else None
        d_conj_sub = dConj.data[idx_rand]

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20,4))
        ## set figure title
        fig.suptitle('Similarity relationships', fontsize=16)
        
        ## plot similarity relationships
        if 1 in plots_to_show:
            axs[0].scatter(ssf_sub, snn_sub, c=d_conj_sub, **kwargs_scatter)
            axs[0].set_xlabel('sim Spatial Footprint')
            axs[0].set_ylabel('sim Neural Network')
        if sSWT_data is not None:
            if 2 in plots_to_show:
                axs[1].scatter(ssf_sub, sswt_sub, c=d_conj_sub, **kwargs_scatter)
                axs[1].set_xlabel('sim Spatial Footprint')
                axs[1].set_ylabel('sim Scattering Wavelet Transform')
            if 3 in plots_to_show:
                axs[2].scatter(snn_sub, sswt_sub, c=d_conj_sub, **kwargs_scatter)
                axs[2].set_xlabel('sim Neural Network')
                axs[2].set_ylabel('sim Scattering Wavelet Transform')
        
        return fig, axs

    def plot_distSame(self, kwargs_makeConjunctiveDistanceMatrix: Optional[dict] = None) -> None:
        """
        Plot the estimated distribution of the pairwise similarities between
        matched ROI pairs of ROIs.

        Args:
            kwargs_makeConjunctiveDistanceMatrix (Optional[dict]): 
                Keyword arguments for the makeConjunctiveDistanceMatrix method.
                If ``None``, the function uses the object's best parameters.
                (Default is ``None``)
        """
        kwargs = kwargs_makeConjunctiveDistanceMatrix if kwargs_makeConjunctiveDistanceMatrix is not None else self.best_params
        dConj, sConj, sSF_data, sNN_data, sSWT_data, sConj_data = self.make_conjunctive_distance_matrix(
            s_sf=self.s_sf,
            s_NN=self.s_NN_z,
            s_SWT=self.s_SWT_z,
            s_sesh=None,
            **kwargs
        )
        dens_same_crop, dens_same, dens_diff, dens_all, edges, d_crossover = self._separate_diffSame_distributions(dConj)
        if edges is None:
            print('No crossover found, not plotting')
            return None
        
        fig = plt.figure()
        plt.stairs(dens_same, edges, linewidth=5)
        plt.stairs(dens_same_crop, edges, linewidth=3)
        plt.stairs(dens_diff, edges)
        plt.stairs(dens_all, edges)
        plt.axvline(d_crossover, color='k', linestyle='--')
        plt.ylim([dens_same.max()*-0.5, dens_same.max()*1.5])
        plt.title('Pairwise similarities')
        plt.xlabel('distance or prob(different)')
        plt.ylabel('counts or density')
        plt.legend(['same', 'same (cropped)', 'diff', 'all', 'crossover'])
        return fig

    def _fn_smooth(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Smooth a 1D tensor with a boxcar convolution.

        Args:
            x (torch.Tensor):
                1D tensor to be smoothed.

        Returns:
            (torch.Tensor):
                Smoothed tensor, same shape as ``x``.
        """
        return helpers.Convolver_1d(
            kernel=torch.ones(self.smooth_window),
            length_x=self.n_bins,
            pad_mode='same',
            correct_edge_effects=True,
            device='cpu',
        ).convolve(x)

    ####################################################################
    ## Shared histogram overlap computation
    ####################################################################

    def _compute_histogram_overlap(
        self,
        distances: torch.Tensor,
        intra_indices: torch.Tensor,
        edges: torch.Tensor,
        smoother: 'helpers.Convolver_1d',
        scale_factor: float,
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Compute the hard histogram overlap loss between the estimated
        'same' and 'different' distance distributions.

        Shared core used by :meth:`_separate_diffSame_distributions`,
        :meth:`_find_optimal_parameters_DE` (scalar objective), and
        :meth:`find_optimal_nb_combination_DE` (NB objective).

        The 'different' distribution is estimated by scaling the intra-session
        (known-different) histogram. The 'same' distribution is the residual
        after subtracting the scaled intra counts from all counts, clamped
        non-negative and then smoothed. The loss is the dot product of the
        two estimated distributions (overlap area).

        If no valid crossover point exists (i.e. the two distributions never
        separate), ``loss`` is set to ``1e6`` as a penalty.
        RH 2025

        Args:
            distances (torch.Tensor):
                1D float tensor of distance values, shape ``(n,)``.
            intra_indices (torch.Tensor):
                1D int64 tensor of indices into ``distances`` corresponding
                to intra-session (known-different) pairs.
            edges (torch.Tensor):
                1D float tensor of bin edges, shape ``(n_bins + 1,)``.
            smoother (helpers.Convolver_1d):
                Pre-built 1D convolver for smoothing the 'same' distribution.
            scale_factor (float):
                ``n_all / n_intra`` — multiplier that scales the intra counts
                up to the full population size.

        Returns:
            (Tuple[float, torch.Tensor, torch.Tensor]):
                loss (float):
                    Overlap area (dot product of dens_same and dens_diff),
                    or ``1e6`` if no valid crossover exists.
                dens_same (torch.Tensor):
                    Smoothed estimated 'same' distribution, shape ``(n_bins,)``.
                dens_diff (torch.Tensor):
                    Scaled intra-session distribution (estimated 'different'),
                    shape ``(n_bins,)``.
        """
        ## Histogram all distances and intra-session distances
        counts_all, _ = torch.histogram(distances, edges)
        counts_intra, _ = torch.histogram(distances[intra_indices], edges)

        ## Scale intra counts to estimate the full 'different' distribution
        dens_diff = counts_intra * scale_factor

        ## 'Same' distribution = residual, clamped non-negative, then smoothed
        dens_same = smoother.convolve(torch.clamp(counts_all - dens_diff, min=0))

        ## Penalize if no valid crossover exists between the two distributions
        dens_deriv = dens_diff - dens_same
        dens_deriv[int(dens_diff.argmax().item()):] = 0
        if not (dens_deriv < 0).any():
            return 1e6, dens_same, dens_diff

        return float((dens_same * dens_diff).sum().item()), dens_same, dens_diff

    def _separate_diffSame_distributions(
        self,
        d_conj: scipy.sparse.csr_matrix,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Estimate the 'same' and 'different' distance distributions from a
        conjunctive distance matrix.

        The 'same' distribution is estimated as the residual after subtracting
        the scaled intra-session (known-different) distribution from the overall
        distribution. Delegates core histogram overlap computation to
        :meth:`_compute_histogram_overlap`.

        Args:
            d_conj (scipy.sparse.csr_matrix):
                Conjunctive distance matrix.

        Returns:
            (tuple): tuple containing:
                dens_same_crop (np.ndarray):
                    'Same' distribution with values below the crossover point
                    zeroed out.
                dens_same (np.ndarray):
                    Un-cropped smoothed 'same' distribution.
                dens_diff (np.ndarray):
                    Scaled intra-session 'different' distribution.
                dens_all (np.ndarray):
                    Raw histogram counts over all pairs.
                edges (np.ndarray):
                    Bin edges used for all histograms.
                d_crossover (float):
                    Distance at which the 'same' and 'different' distributions
                    cross over.
        """
        ## Bin edges covering the full [0, 1] distance range
        edges = torch.linspace(start=0, end=1, steps=self.n_bins + 1, dtype=torch.float32)

        ## Extract all-pairs distances and compute raw histogram
        dist_all = torch.as_tensor(d_conj.data, dtype=torch.float32)
        dens_all, _ = torch.histogram(dist_all, edges)

        ## Extract intra-session (known-different) distances via s_sesh_inv mask
        d_intra = d_conj.multiply(self.s_sesh_inv)
        d_intra.eliminate_zeros()
        if len(d_intra.data) == 0:
            return None, None, None, None, None, None
        dist_intra = torch.as_tensor(d_intra.data, dtype=torch.float32)

        ## Scale factor: ratio of all pairs to intra-session pairs
        n_all = int(dens_all.sum().item())
        n_intra = dist_intra.shape[0]
        scale_factor = float(n_all) / max(n_intra, 1)

        ## Compute scaled intra (different) distribution
        counts_intra, _ = torch.histogram(dist_intra, edges)
        dens_diff = counts_intra * scale_factor

        ## Smoothed residual (same) — uses cached smoother in _fn_smooth
        dens_same = self._fn_smooth(torch.clamp(dens_all - dens_diff, min=0))

        ## Locate crossover: last bin before dens_diff peak where diff > same
        dens_deriv = dens_diff - dens_same
        dens_deriv[int(dens_diff.argmax().item()):] = 0
        crossover_candidates = torch.where(dens_deriv < 0)[0]
        if crossover_candidates.shape[0] == 0:
            return None, None, None, None, None, None
        idx_crossover = int(crossover_candidates[-1].item()) + 1
        d_crossover = float(edges[idx_crossover].item())

        ## Zero out 'same' distribution below the crossover point
        dens_same_crop = dens_same.clone()
        dens_same_crop[idx_crossover:] = 0

        return dens_same_crop, dens_same, dens_diff, dens_all, edges, d_crossover

    def compute_quality_metrics(
        self,
        sim_mat: Optional[object] = None,
        dist_mat: Optional[object] = None,
        labels: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Computes quality metrics of the dataset.
        RH 2023

        Args:
            sim_mat (Optional[object]): 
                Similarity matrix of shape *(n_samples, n_samples)*. 
                If ``None`` then self.sConj must exist. (Default is ``None``)
            dist_mat (Optional[object]): 
                Distance matrix of shape *(n_samples, n_samples)*. 
                If ``None`` then self.dConj must exist. (Default is ``None``)
            labels (Optional[np.ndarray]): 
                Cluster labels of shape *(n_samples,)*. 
                If ``None``, then self.labels must exist. (Default is ``None``)

        Returns:
            (Dict): 
                quality_metrics (Dict): 
                    Quality metrics dictionary that includes:
                    'cluster_intra_means', 'cluster_intra_mins',
                    'cluster_intra_maxs', 'cluster_silhouette',
                    'sample_silhouette', and other metrics if available.
        """
        if sim_mat is None:
            assert hasattr(self, 'sConj'), "self.sConj does not exist. Run self.find_optimal_parameters_for_pruning() first or specify sim_mat."
            sim_mat = self.sConj
        if dist_mat is None:
            assert hasattr(self, 'dConj'), "self.dConj does not exist. Run self.find_optimal_parameters_for_pruning() first or specify dist_mat."
            dist_mat = self.dConj
        if labels is None:
            assert hasattr(self, 'labels'), "self.labels does not exist. Run self.find_optimal_parameters_for_pruning() first or specify labels."
            labels = self.labels
            
        assert scipy.sparse.issparse(sim_mat), "sim_mat must be a scipy.sparse.csr_matrix."
        assert scipy.sparse.issparse(dist_mat), "dist_mat must be a scipy.sparse.csr_matrix."

        labels_unique, cs_intra_means, cs_intra_mins, cs_intra_maxs, cs_sil = cluster_quality_metrics(
            sim=sim_mat,
            labels=labels,
        )

        import sklearn
        import sparse
        d_dense = sparse.COO(dist_mat.copy().tocsr()).astype(np.float16)
        d_dense.fill_value = (dist_mat.data.max() - dist_mat.data.min()).astype(np.float16) * 10
        d_dense = d_dense.todense()
        np.fill_diagonal(d_dense, 0)
        ## Number of labels must be at least 2
        if len(np.unique(labels)) < 2:
            warnings.warn(f"Silhouette samples calculation requires at least 2 labels. Returning None. Found {len(np.unique(labels))} labels.")
            rs_sil = None
        else:
            rs_sil = sklearn.metrics.silhouette_samples(X=d_dense, labels=labels, metric='precomputed')

        def to_list_of_floats(x):
            return [float(i) for i in x]

        self.quality_metrics = util.JSON_Dict({
            'cluster_labels_unique': to_list_of_floats(labels_unique),
            'cluster_intra_means': to_list_of_floats(cs_intra_means),
            'cluster_intra_mins': to_list_of_floats(cs_intra_mins),
            'cluster_intra_maxs': to_list_of_floats(cs_intra_maxs),
            'cluster_silhouette': to_list_of_floats(cs_sil),
            'sample_silhouette': to_list_of_floats(rs_sil),
            'hdbscan': {
                'sample_outlierScores': to_list_of_floats(self.hdbs.outlier_scores_[:-1]),  ## Remove last element which is the outlier score for the new fully connected node
                'sample_probabilities': to_list_of_floats(self.hdbs.probabilities_[:-1]),  ## Remove last element which is the outlier score for the new fully connected node
            }  if hasattr(self, 'hdbs') else None,
            'sequentialHungarian': {
                'performance_recall': float(self.seqHung_performance['recall']),
                'performance_precision': float(self.seqHung_performance['precision']),
                'performance_f1': float(self.seqHung_performance['f1_score']),
                'performance_accuracy': float(self.seqHung_performance['accuracy']),
            } if hasattr(self, 'seqHung_performance') else None,
        })
        return self.quality_metrics

    ####################################################################
    ####################################################################
    ##
    ## LEGACY / COMPARISON METHODS
    ##
    ## The methods below are retained for backward compatibility and
    ## benchmarking. They are NOT used by the default pipeline.
    ##
    ## The recommended approach is find_optimal_parameters_for_pruning()
    ## which calls _find_optimal_parameters_DE(freeze_sigmoid=True).
    ##
    ####################################################################
    ####################################################################

    def _find_optimal_parameters_for_pruning_optuna(
        self,
        kwargs_findParameters: Dict[str, Union[int, float, bool]] = {
            'n_patience': 100,
            'tol_frac': 0.05,
            'max_trials': 350,
            'max_duration': 60*10,
            'value_stop': 0.0,
        },
        bounds_findParameters: Dict[str, List[float]] = {
            'power_NN': [0.0, 2.],
            'power_SWT': [0.0, 2.],
            'p_norm': [-5, -0.1],
            'sig_NN_kwargs_mu': [0., 1.0],
            'sig_NN_kwargs_b': [0.1, 1.5],
            'sig_SWT_kwargs_mu': [0., 1.0],
            'sig_SWT_kwargs_b': [0.1, 1.5],
        },
        n_jobs_findParameters: int = -1,
        n_bins: Optional[int] = None,
        smoothing_window_bins: Optional[int] = None,
        seed=None,
    ) -> Dict:
        """
        **LEGACY** — Original Optuna TPE optimizer (7-param). Superseded by
        :meth:`find_optimal_parameters_for_pruning` which uses NB calibration
        + freeze-sigmoid DE and achieves ~3.6x better separation quality.

        Requires ``optuna`` to be installed.
        RH 2023

        Args:
            kwargs_findParameters (Dict[str, Union[int, float, bool]]):
                Keyword arguments for the Convergence_checker class __init__.
            bounds_findParameters (Dict[str, List[float]]):
                Bounds for the 7 parameters to be optimized.
            n_jobs_findParameters (int):
                Number of parallel Optuna jobs. ``-1`` = all cores.
            n_bins (Optional[int]):
                Overwrites ``n_bins`` from ``__init__``.
            smoothing_window_bins (Optional[int]):
                Overwrites ``smoothing_window_bins`` from ``__init__``.
            seed (Optional[int]):
                Random seed.

        Returns:
            (Dict):
                kwargs_makeConjunctiveDistanceMatrix_best (Dict):
                    Optimal parameters for
                    :meth:`make_conjunctive_distance_matrix`.
        """
        import optuna

        self.params['find_optimal_parameters_for_pruning'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'kwargs_findParameters', 'bounds_findParameters',
                'n_jobs_findParameters', 'n_bins', 'smoothing_window_bins', 'seed',
            ],
        )

        self.n_bins = self.n_bins if n_bins is None else n_bins
        self.smooth_window = self.smooth_window if smoothing_window_bins is None else smoothing_window_bins
        self.bounds_findParameters = bounds_findParameters
        self._seed = seed
        np.random.seed(self._seed)

        print('Finding mixing parameters using automated hyperparameter tuning...') if self._verbose else None
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.checker = helpers.Convergence_checker_optuna(verbose=self._verbose >= 2, **kwargs_findParameters)
        prog_bar = helpers.OptunaProgressBar(
            n_trials=kwargs_findParameters['max_trials'],
            mininterval=5.0,
        )
        self.study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(
            n_startup_trials=kwargs_findParameters['n_patience'] // 2,
            seed=self._seed,
        ))
        self.study.optimize(
            func=self._objectiveFn_distSameMagnitude,
            n_jobs=n_jobs_findParameters,
            callbacks=[self.checker.check, prog_bar],
            n_trials=kwargs_findParameters['max_trials'],
            show_progress_bar=False,
        )

        self.best_params = self.study.best_params.copy()
        [self.best_params.pop(p) for p in [
            'sig_NN_kwargs_mu', 'sig_NN_kwargs_b',
            'sig_SWT_kwargs_mu', 'sig_SWT_kwargs_b',
        ] if p in self.best_params.keys()]
        self.best_params['sig_NN_kwargs'] = {
            'mu': self.study.best_params['sig_NN_kwargs_mu'],
            'b': self.study.best_params['sig_NN_kwargs_b'],
        }
        self.best_params['sig_SWT_kwargs'] = {
            'mu': self.study.best_params['sig_SWT_kwargs_mu'],
            'b': self.study.best_params['sig_SWT_kwargs_b'],
        }

        self.kwargs_makeConjunctiveDistanceMatrix_best = {
            'power_SF': None, 'power_NN': None, 'power_SWT': None,
            'p_norm': None, 'sig_SF_kwargs': None,
            'sig_NN_kwargs': None, 'sig_SWT_kwargs': None,
        }
        self.kwargs_makeConjunctiveDistanceMatrix_best.update(self.best_params)
        print(f'Completed automatic parameter search. Best value found: {self.study.best_value} with parameters {self.best_params}') if self._verbose else None
        return self.kwargs_makeConjunctiveDistanceMatrix_best

    def _objectiveFn_distSameMagnitude(
        self,
        trial: object,
    ) -> float:
        """
        **LEGACY** — Optuna objective function for histogram overlap loss.
        Used by :meth:`_find_optimal_parameters_for_pruning_optuna`.
        RH 2023
        """
        power_SF = 1
        power_NN = trial.suggest_float('power_NN', *self.bounds_findParameters['power_NN'], log=False)
        power_SWT = trial.suggest_float('power_SWT', *self.bounds_findParameters['power_SWT'], log=False)
        p_norm = trial.suggest_float('p_norm', *self.bounds_findParameters['p_norm'], log=False)
        sig_SF_kwargs = None
        sig_NN_kwargs = {
            'mu': trial.suggest_float('sig_NN_kwargs_mu', *self.bounds_findParameters['sig_NN_kwargs_mu'], log=False),
            'b': trial.suggest_float('sig_NN_kwargs_b', *self.bounds_findParameters['sig_NN_kwargs_b'], log=False),
        }
        sig_SWT_kwargs = {
            'mu': trial.suggest_float('sig_SWT_kwargs_mu', *self.bounds_findParameters['sig_SWT_kwargs_mu'], log=False),
            'b': trial.suggest_float('sig_SWT_kwargs_b', *self.bounds_findParameters['sig_SWT_kwargs_b'], log=False),
        }
        dConj, sConj, sSF_data, sNN_data, sSWT_data, sConj_data = self.make_conjunctive_distance_matrix(
            s_sf=self.s_sf, s_NN=self.s_NN_z, s_SWT=self.s_SWT_z, s_sesh=None,
            power_SF=power_SF, power_NN=power_NN, power_SWT=power_SWT, p_norm=p_norm,
            sig_SF_kwargs=sig_SF_kwargs, sig_NN_kwargs=sig_NN_kwargs, sig_SWT_kwargs=sig_SWT_kwargs,
        )
        dens_same_crop, dens_same, dens_diff, dens_all, edges, d_crossover = self._separate_diffSame_distributions(dConj)
        if dens_same_crop is None:
            return 0
        return (dens_same * dens_diff).sum().item()

    def find_optimal_nb_combination_DE(
        self,
        bounds: Optional[Dict[str, List[float]]] = None,
        de_kwargs: Optional[Dict[str, Any]] = None,
        subsample_pairs: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, Dict]:
        """
        **LEGACY** — Optimize weighted combination of per-feature NB P(same)
        curves using DE. Superseded by
        :meth:`find_optimal_parameters_for_pruning` (freeze-sigmoid DE)
        which achieves ~28x better loss on the same data.
        RH 2025
        """
        assert hasattr(self, 'calibrations_naive_bayes') and self.calibrations_naive_bayes is not None, (
            "make_naive_bayes_distance_matrix() must be called before "
            "find_optimal_nb_combination_DE()."
        )
        self.params['find_optimal_nb_combination_DE'] = self._locals_to_params(
            locals_dict=locals(),
            keys=['bounds', 'de_kwargs', 'subsample_pairs', 'seed'],
        )
        bounds_use = bounds if bounds is not None else {
            'p_norm': [0.1, 5.0], 'w_NN': [0.0, 5.0], 'w_SWT': [0.0, 5.0],
        }
        de_kwargs_use = de_kwargs if de_kwargs is not None else {
            'maxiter': 100, 'tol': 1e-6, 'popsize': 15,
            'mutation': (0.5, 1.5), 'recombination': 0.7, 'polish': True,
        }
        scipy_bounds = [tuple(bounds_use['p_norm']), tuple(bounds_use['w_NN']), tuple(bounds_use['w_SWT'])]

        cal = self.calibrations_naive_bayes['features']
        p_sf_t = torch.as_tensor(cal['SF']['p_same_per_pair'], dtype=torch.float32)
        p_nn_t = torch.as_tensor(cal['NN']['p_same_per_pair'], dtype=torch.float32)
        p_swt_t = torch.as_tensor(cal['SWT']['p_same_per_pair'], dtype=torch.float32)

        if not hasattr(self, '_intra_mask') or self._intra_mask is None:
            self._precompute_intra_mask()
        intra_mask = torch.as_tensor(self._intra_mask)

        nnz = p_sf_t.shape[0]
        if subsample_pairs is not None and subsample_pairs < nnz:
            sample_idx = self._subsample_pairs(
                n_subsample=subsample_pairs, intra_mask=intra_mask,
                seed=(seed + 77777) if seed is not None else 77777,
            )
            frac = subsample_pairs / nnz
            n_sample_intra = max(int(int(intra_mask.sum().item()) * frac), 100)
            p_sf_t, p_nn_t, p_swt_t = p_sf_t[sample_idx], p_nn_t[sample_idx], p_swt_t[sample_idx]
            intra_mask_use = torch.zeros(sample_idx.shape[0], dtype=torch.bool)
            intra_mask_use[:n_sample_intra] = True
        else:
            intra_mask_use = intra_mask

        n_bins_val = self.n_bins
        edges = torch.linspace(0.0, 1.0, n_bins_val + 1, dtype=torch.float32)
        smooth_window = helpers.make_odd(n_bins_val // 10, mode='up')
        smoother = helpers.Convolver_1d(
            kernel=torch.ones(smooth_window), length_x=n_bins_val,
            pad_mode='same', correct_edge_effects=True, device='cpu',
        )
        n_all = p_sf_t.shape[0]
        n_intra = int(intra_mask_use.sum().item())
        scale_factor = n_all / max(n_intra, 1)
        intra_indices = torch.where(intra_mask_use)[0]
        p_sf_c = torch.clamp(p_sf_t, min=1e-6, max=1.0 - 1e-6)
        p_nn_c = torch.clamp(p_nn_t, min=1e-6, max=1.0 - 1e-6)
        p_swt_c = torch.clamp(p_swt_t, min=1e-6, max=1.0 - 1e-6)
        w_sf_fixed = 1.0

        def objective(x):
            p_val, w_nn_val, w_swt_val = float(x[0]), float(x[1]), float(x[2])
            p = p_val if abs(p_val) > 1e-9 else 1e-9
            w_total = w_sf_fixed + w_nn_val + w_swt_val + 1e-10
            p_combined = (
                (w_sf_fixed * torch.pow(p_sf_c, p) + w_nn_val * torch.pow(p_nn_c, p)
                 + w_swt_val * torch.pow(p_swt_c, p)) / w_total
            ).pow(1.0 / p)
            d_combined = 1.0 - p_combined
            loss, _, _ = self._compute_histogram_overlap(
                distances=d_combined, intra_indices=intra_indices,
                edges=edges, smoother=smoother, scale_factor=scale_factor,
            )
            return loss

        print('Optimizing NB combination weights with differential evolution...') if self._verbose else None
        nb_de_result = scipy.optimize.differential_evolution(
            func=objective, bounds=scipy_bounds, seed=seed, **de_kwargs_use,
        )

        p_val_best, w_nn_best, w_swt_best = float(nb_de_result.x[0]), float(nb_de_result.x[1]), float(nb_de_result.x[2])
        p_best = p_val_best if abs(p_val_best) > 1e-9 else 1e-9
        w_total_best = w_sf_fixed + w_nn_best + w_swt_best + 1e-10
        p_sf_full = torch.clamp(torch.as_tensor(cal['SF']['p_same_per_pair'], dtype=torch.float32), min=1e-6, max=1.0 - 1e-6)
        p_nn_full = torch.clamp(torch.as_tensor(cal['NN']['p_same_per_pair'], dtype=torch.float32), min=1e-6, max=1.0 - 1e-6)
        p_swt_full = torch.clamp(torch.as_tensor(cal['SWT']['p_same_per_pair'], dtype=torch.float32), min=1e-6, max=1.0 - 1e-6)
        p_same_combined_np = (
            (w_sf_fixed * torch.pow(p_sf_full, p_best) + w_nn_best * torch.pow(p_nn_full, p_best)
             + w_swt_best * torch.pow(p_swt_full, p_best)) / w_total_best
        ).pow(1.0 / p_best).numpy()

        sConj = self.s_sf.copy()
        sConj.data = p_same_combined_np.astype(np.float64)
        dConj = sConj.copy()
        dConj.data = 1.0 - dConj.data
        self.dConj = dConj
        self.sConj = sConj
        best_params = {'p_norm': p_val_best, 'w_NN': w_nn_best, 'w_SWT': w_swt_best}
        result_info = {
            'best_params': best_params, 'loss': float(nb_de_result.fun),
            'n_evals': int(nb_de_result.nfev), 'p_same_combined': p_same_combined_np,
        }
        print(f'Completed NB combination DE. Best loss: {nb_de_result.fun:.2f}, params: {best_params}') if self._verbose else None
        return dConj, sConj, result_info



def attach_fully_connected_node(
    d: object,
    dist_fullyConnectedNode: Optional[float] = None,
    n_nodes: int = 1,
) -> object:
    """
    Appends a single node to a sparse distance graph that is weakly connected to all nodes.
    
    Args:
        d (object): 
            Sparse graph with multiple components. 
            Refer to scipy.sparse.csgraph.connected_components for details.
        dist_fullyConnectedNode (Optional[float]): 
            Value used for the connection strength to all other nodes. 
            This value will be appended as elements in a new row and column 
            at the ends of the 'd' matrix. If ``None``, then the value will be 
            set to 1000 times the difference between the maximum and minimum 
            values in 'd'. (Default is ``None``)
        n_nodes (int): 
            Number of nodes to append to the graph. (Default is *1*)

    Returns:
        (object): 
            d2 (object): 
                Sparse graph with only one component.
    """
    if dist_fullyConnectedNode is None:
        dist_fullyConnectedNode = (d.max() - d.min()) * 1000
    
    d2 = d.copy()
    d2 = scipy.sparse.vstack((d2, np.ones((n_nodes,d2.shape[1]), dtype=d.dtype)*dist_fullyConnectedNode))
    d2 = scipy.sparse.hstack((d2, np.ones((d2.shape[0],n_nodes), dtype=d.dtype)*dist_fullyConnectedNode))
    return d2.tocsr()


def score_labels(
    labels_test: np.ndarray, 
    labels_true: np.ndarray, 
    ignore_negOne: bool = False, 
    thresh_perfect: float = 0.9999999999, 
) -> Dict[str, Union[float, Tuple[int, int]]]:
    """
    Computes the score of the clustering by finding the best match using the
    linear sum assignment problem. The score is bounded between 0 and 1. Note:
    The score is not symmetric if the number of true and test labels are not the
    same. I.e., switching ``labels_test`` and ``labels_true`` can lead to
    different scores. This is because we are scoring how well each true set is
    matched by an optimally assigned test set.
    
    RH 2022
    
    Args:
        labels_test (np.ndarray): 
            Labels of the test clusters/sets. (shape: *(n,)*)
        labels_true (np.ndarray):
            Labels of the true clusters/sets. (shape: *(n,)*)
        ignore_negOne (bool): 
            Whether to ignore ``-1`` values in the labels. If set to ``True``,
            ``-1`` values will be ignored in the computation. (Default is
            ``False``)
        thresh_perfect (float): 
            Threshold for perfect match. Mostly used for numerical stability.
            (Default is *0.9999999999*)
    
    Returns:
        (dict): dictionary containing:
            score_weighted_partial (float):
                Average correlation between the best matched sets of true and
                test labels, weighted by the number of elements in each true
                set.
            score_weighted_perfect (float):
                Fraction of perfect matches, weighted by the number of elements
                in each true set.
            score_unweighted_partial (float):
                Average correlation between the best matched sets of true and
                test labels.
            score_unweighted_perfect (float):
                Fraction of perfect matches.
            adj_rand_score (float):
                Adjusted Rand score of the labels.
            adj_mutual_info_score (float):
                Adjusted mutual info score of the labels. None if
                ``compute_mutual_info`` is ``False``.
            ignore_negOne (bool):
                Whether ``-1`` values were ignored in the labels.
            idx_hungarian (Tuple[int, int]):
                'Hungarian Indices'. Indices of the best matched sets.
    """
    assert len(labels_test) == len(labels_true), 'RH ERROR: labels_test and labels_true must be the same length.'
    labels_test = np.array(labels_test, dtype=int)
    labels_true = np.array(labels_true, dtype=int)

    ## convert labels to boolean
    uniques_test, uniques_true = np.unique(labels_test), np.unique(labels_true)
    bool_test = np.stack([labels_test==l for l in uniques_test], axis=0).astype(np.float32)
    bool_true = np.stack([labels_true==l for l in uniques_true], axis=0).astype(np.float32)

    ## Hungarian matching score
    if ignore_negOne:
        bool_test[uniques_test == -1, :] = 0.0
        bool_true[uniques_true == -1, :] = 0.0
    if bool_test.shape[0] < bool_true.shape[0]:
        bool_test = np.concatenate((bool_test, np.zeros((bool_true.shape[0] - bool_test.shape[0], bool_true.shape[1]))))
    ## compute confusion / correlation matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        cc = np.nan_to_num((bool_true @ bool_test.T) / (bool_true.sum(axis=1)[:, None]), nan=0.0, posinf=0.0, neginf=0.0)  ## normalize by the number of elements in each set
    ## find hungarian assignment matching indices
    hi = scipy.optimize.linear_sum_assignment(cost_matrix=cc, maximize=True)
    ## extract correlation scores of matches
    cc_matched = cc[hi[0], hi[1]]
    label_weights = bool_true.sum(axis=1)[hi[0]]  ## reweighting vector is the number of elements in each true set
    label_weights_norm = label_weights / label_weights.sum()  ## normalize the weights
    hungarian_match_score_weighted_partial = cc_matched @ label_weights_norm
    hungarian_match_score_unweighted_partial = cc_matched.mean()
    hungarian_match_score_weighted_perfect = (cc_matched > thresh_perfect).astype(float) @ label_weights_norm
    hungarian_match_score_unweighted_perfect = (cc_matched > thresh_perfect).mean()

    ## SKLEARN METRICS
    ### First change all -1 values to a unique value that is not present in the labels
    uniques = np.unique(np.concatenate((labels_true, labels_test)))
    ### Make a bunch of values greater than the maximum value in the labels
    n_minusOne_true = np.sum(labels_true == -1)
    n_minusOne_test = np.sum(labels_test == -1)
    labels_true_sk, labels_test_sk = labels_true.copy(), labels_test.copy()
    labels_true_sk[labels_true == -1] = uniques.max() + np.arange(1, n_minusOne_true + 1)
    labels_test_sk[labels_test == -1] = uniques.max() + np.arange(n_minusOne_true + 1, n_minusOne_true + n_minusOne_test + 1)
    
    ## compute adjusted rand score
    score_rand = sklearn.metrics.adjusted_rand_score(labels_true=labels_true_sk, labels_pred=labels_test_sk)

    ## compute fowlkes mallows score
    score_fowlkes_mallows = sklearn.metrics.fowlkes_mallows_score(labels_true=labels_true_sk, labels_pred=labels_test_sk)

    ## compute adjusted mutual info score
    score_mutual_info = sklearn.metrics.adjusted_mutual_info_score(labels_true=labels_true_sk, labels_pred=labels_test_sk)

    ## compute homogeneity, completeness, and v-measure
    homogeneity, completeness, v_measure = sklearn.metrics.homogeneity_completeness_v_measure(labels_true=labels_true_sk, labels_pred=labels_test_sk, beta=1.0)

    ## compute pair confusion matrix
    pair_confusion = sklearn.metrics.cluster.pair_confusion_matrix(labels_true=labels_true_sk, labels_pred=labels_test_sk).tolist()
    if pair_confusion is not None:
        p = np.array(pair_confusion)
        TP, TN, FP, FN = p[1, 1], p[0, 0], p[0, 1], p[1, 0]
        pc_accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        N_all = TN + FP
        P_all = TP + FN
        pc_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        pc_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        pc_f1 = (2 * pc_precision * pc_recall) / (pc_precision + pc_recall) if (pc_precision + pc_recall) > 0 else 0
        pc_accuracy_norm = (TP/P_all + TN/N_all) / (TP/P_all + TN/N_all + FP/N_all + FN/P_all)
    else:
        pc_accuracy = None
        pc_accuracy_norm = None

    out = {
        'hungarian_score_weighted_partial': float(hungarian_match_score_weighted_partial),
        'hungarian_score_weighted_perfect': float(hungarian_match_score_weighted_perfect),
        'hungarian_score_unweighted_partial': float(hungarian_match_score_unweighted_partial),
        'hungarian_score_unweighted_perfect': float(hungarian_match_score_unweighted_perfect),
        'adj_rand_score': float(score_rand),
        'fowlkes_mallows_score': float(score_fowlkes_mallows),
        'adj_mutual_info_score': float(score_mutual_info),
        'homogeneity_score': float(homogeneity),
        'completeness_score': float(completeness),
        'v_measure_score': float(v_measure),
        'pair_confusion_matrix': pair_confusion,
        'pair_confusion_accuracy_score': float(pc_accuracy),
        'pair_confusion_accuracy_norm_score': float(pc_accuracy_norm),
        'pair_confusion_precision_score': float(pc_precision) if pc_precision is not None else None,
        'pair_confusion_recall_score': float(pc_recall) if pc_recall is not None else None,
        'pair_confusion_f1_score': float(pc_f1) if pc_f1 is not None else None,
        'labels_test': labels_test.tolist(),
        'ignore_negOne': ignore_negOne,
        'idx_hungarian': hi,
    }
    return out


def cluster_quality_metrics(
    sim: Union[np.ndarray, scipy.sparse.csr_matrix], 
    labels: np.ndarray,
) -> Tuple:
    """
    Computes the cluster quality metrics for a clustering solution including
    intra-cluster mean, minimum, maximum similarity, and cluster silhouette
    score. 
    RH 2023

    Args:
        sim (Union[np.ndarray, scipy.sparse.csr_matrix]):
            Similarity matrix. (shape: *(n_roi, n_roi)*) It can be obtained
            using `_, sConj, _,_,_,_ =
            clusterer.make_conjunctive_similarity_matrix()`.
        labels (np.ndarray):
            Cluster labels. (shape: *(n_roi,)*)

    Returns:
        (tuple): tuple containing:
            cs_intra_means (np.ndarray):
                Intra-cluster mean similarity. (shape: *(n_clusters,)*)
            cs_intra_mins (np.ndarray):
                Intra-cluster minimum similarity. (shape: *(n_clusters,)*)
            cs_intra_maxs (np.ndarray):
                Intra-cluster maximum similarity. (shape: *(n_clusters,)*)
            cs_sil (np.ndarray):
                Cluster silhouette score. (shape: *(n_clusters,)*)
                Describes intra_mean - inter_max_of_maxes
    """
    import sparse
    
    labels_unique, cs_mean, cs_max, cs_min = helpers.compute_cluster_similarity_matrices(sim, labels, verbose=True)
    fn_sil_score = lambda intra, inter: (intra - inter) / np.maximum(intra, inter)

    eye_inv = 1 - sparse.eye(cs_max.shape[0])

    cs_intra_means = cs_mean.diagonal()
    cs_inter_maxOfMaxs = (eye_inv * cs_max).max(0)
    cs_sil = fn_sil_score(cs_intra_means, cs_inter_maxOfMaxs)
    cs_intra_mins = cs_min.diagonal()
    cs_intra_maxs = cs_max.diagonal()
    
    return labels_unique, cs_intra_means, cs_intra_mins, cs_intra_maxs, cs_sil

def make_label_variants(
    labels: np.ndarray, 
    n_roi_bySession: np.ndarray,
) -> Tuple:
    """
    Creates convenient variants of label arrays.
    RH 2023

    Args:
        labels (np.ndarray):
            Cluster integer labels. (shape: *(n_roi,)*)
        n_roi_bySession (np.ndarray):
            Number of ROIs in each session.

    Returns:
        (tuple): tuple containing:
            labels_squeezed (np.ndarray):
                Cluster labels squeezed into a continuous range starting from 0.
            labels_bySession (List[np.ndarray]):
                List of label arrays split by session.
            labels_bool (scipy.sparse.csr_matrix):
                Sparse boolean matrix representation of labels.
            labels_bool_bySession (List[scipy.sparse.csr_matrix]):
                List of sparse boolean matrix representations of labels split by
                session.
            labels_dict (Dict[int, np.ndarray]):
                Dictionary mapping unique labels to their locations in the
                labels array.
    """
    import scipy.sparse

    ## assert that labels is a 1D np.array or list of numbers
    if isinstance(labels, list):
        labels = np.array(labels, dtype=np.int64)
    elif isinstance(labels, np.ndarray):
        labels = labels.astype(np.int64)
    else:
        raise TypeError('RH ERROR: labels must be a 1D np.array or list of numbers.')
    assert labels.ndim == 1, 'RH ERROR: labels must be a 1D np.array or list of numbers.'

    ## assert that n_roi_bySession is a 1D np.array or list of numbers
    if isinstance(n_roi_bySession, list):
        n_roi_bySession = np.array(n_roi_bySession, dtype=np.int64)
    elif isinstance(n_roi_bySession, np.ndarray):
        n_roi_bySession = n_roi_bySession.astype(np.int64)
    else:
        raise TypeError('RH ERROR: n_roi_bySession must be a 1D np.array or list of numbers.')
    assert n_roi_bySession.ndim == 1, 'RH ERROR: n_roi_bySession must be a 1D np.array or list of numbers.'
    
    ## assert that the number of labels adds up to the number of ROIs
    n_roi_total = n_roi_bySession.sum()
    n_roi_cumsum = np.concatenate([[0], n_roi_bySession.cumsum()])
    assert labels.shape[0] == n_roi_bySession.sum(), 'RH ERROR: the number of labels must add up to the number of ROIs.'

    ## make session_bool
    session_bool = util.make_session_bool(n_roi_bySession)

    ## make variants
    labels_squeezed = helpers.squeeze_integers(labels)
    labels_bySession = [labels_squeezed[idx] for idx in session_bool.T]
    labels_bool = scipy.sparse.vstack([scipy.sparse.csr_matrix(labels_squeezed==u) for u in np.sort(np.unique(labels_squeezed))]).T.tocsr()
    labels_bool_bySession = [labels_bool[idx] for idx in session_bool.T]
    labels_dict = {u: np.where(labels_squeezed==u)[0] for u in np.unique(labels_squeezed)}

    ## testing
    assert np.allclose(np.concatenate(labels_bySession), labels_squeezed)
    assert np.allclose(labels_bool.nonzero()[1] - 1, labels_squeezed)
    assert np.all([np.allclose(np.where(labels_squeezed==u)[0], ldu) for u, ldu in labels_dict.items()])

    ## Convert everything to native python types for JSON compatibility
    labels_squeezed = util.JSON_List([int(u) for u in labels_squeezed])
    labels_bySession = util.JSON_List([[int(u) for u in l] for l in labels_bySession])
    labels_dict = util.JSON_Dict({str(k): [int(v_i) for v_i in v] for k, v in labels_dict.items()})  ## Make keys strings for JSON compatibility

    return labels_squeezed, labels_bySession, labels_bool, labels_bool_bySession, labels_dict


def plot_quality_metrics(quality_metrics: dict, labels: Union[np.ndarray, list], n_sessions: int) -> None:
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,7))

    axs[0,0].hist(quality_metrics['cluster_silhouette'], 50);
    axs[0,0].set_xlabel('cluster_silhouette');
    axs[0,0].set_ylabel('cluster counts');

    axs[0,1].hist(quality_metrics['cluster_intra_means'], 50);
    axs[0,1].set_xlabel('cluster_intra_means');
    axs[0,1].set_ylabel('cluster counts');

    axs[1,0].hist(quality_metrics['sample_silhouette'], 50);
    axs[1,0].set_xlabel('sample_silhouette score');
    axs[1,0].set_ylabel('roi sample counts');

    u, c = np.unique((v:=np.array(labels))[v!=-1], return_counts=True)
    n_sesh = np.bincount(c)

    axs[1,1].bar(np.arange(len(n_sesh)), n_sesh);
    axs[1,1].set_xlabel('n_sessions')
    axs[1,1].set_ylabel('cluster counts');
    
    # Make the title include the number of excluded (label==-1) ROIs
    fig.suptitle(f'Quality metrics n_excluded: {np.sum(labels==-1)}, n_included: {np.sum(labels!=-1)}, n_total: {len(labels)}, n_clusters: {len(np.unique(labels[labels!=-1]))}, n_sessions: {n_sessions}')
    return fig, axs
