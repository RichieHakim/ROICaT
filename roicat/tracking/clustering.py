from typing import Union, Tuple, List, Dict, Optional, Any, Callable, Literal

import warnings
import math

import numpy as np
import scipy
import scipy.optimize
import scipy.sparse
import scipy.signal
import sklearn
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
        n_bins: Optional[int] = None,
        smoothing_window_bins: Optional[int] = None,
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

        self.s_sesh_inv = (self.s_sf != 0).astype(np.bool_)
        self.s_sesh_inv[self.s_sesh.astype(np.bool_)] = False
        self.s_sesh_inv.eliminate_zeros()

        self.s_sesh = self.s_sesh.tolil()
        self.s_sesh[range(self.s_sesh.shape[0]), range(self.s_sesh.shape[1])] = 0
        self.s_sesh = self.s_sesh.tocsr()

        self._verbose = verbose

        self.n_bins = max(min(self.s_sf.nnz // 10000, 200), 20) if n_bins is None else n_bins
        self.smooth_window = helpers.make_odd(self.n_bins // 10, mode='up') if smoothing_window_bins is None else smoothing_window_bins
        # print(f'Pruning similarity graphs with {self.n_bins} bins and smoothing window {smoothing_window}...') if self._verbose else None
        
    def find_optimal_parameters_for_pruning(
        self,
        kwargs_findParameters: Dict[str, Union[int, float, bool]] = {
            'n_patience': 100,
            'tol_frac': 0.05,
            'max_trials': 350,
            'max_duration': 60*10,
            'value_stop': 0.0,
        },
        bounds_findParameters: Dict[str, List[float]] = {
            'power_NN': [0.0, 2.],  ## Bounds for the exponent applied to s_NN
            'power_SWT': [0.0, 2.],  ## Bounds for the exponent applied to s_SWT
            'p_norm': [-5, -0.1],  ## Bounds for the p-norm p value (Minkowski) applied to mix the matrices
            'sig_NN_kwargs_mu': [0., 1.0],  ## Bounds for the sigmoid center for s_NN
            'sig_NN_kwargs_b': [0.1, 1.5],  ## Bounds for the sigmoid slope for s_NN
            'sig_SWT_kwargs_mu': [0., 1.0],  ## Bounds for the sigmoid center for s_SWT
            'sig_SWT_kwargs_b': [0.1, 1.5],  ## Bounds for the sigmoid slope for s_SWT
        },
        n_jobs_findParameters: int = -1,
        n_bins: Optional[int] = None,
        smoothing_window_bins: Optional[int] = None,
        seed=None,
    ) -> Dict:
        """
        Find the optimal parameters for pruning the similarity graph.
        How this function works: \n
        1. Make a conjunctive distance matrix using a set of parameters for
           the self.make_conjunctive_distance_matrix function.
        2. Estimates the distribution of pairwise distances between ROIs assumed
           to be the same and those assumed to be different ROIs. This is done
           by comparing the difference in the distribution of pairwise distances
           between ROIs from the same session and those from different sessions.
           Ideally, the main difference will be the presence of 'same' ROIs in
           the inter-session distribution. 
        3. The optimal parameters are then updated using optuna in order to
           maximize the separation between the 'same' and 'different'
           distributions. \n
        RH 2023

        Args:
            kwargs_findParameters (Dict[str, Union[int, float, bool]]): 
                Keyword arguments for the Convergence_checker class __init__.
            bounds_findParameters (Dict[str, Tuple[float, float]]):
                Bounds for the parameters to be optimized.
            n_jobs_findParameters (int):
                Number of jobs to use when finding the optimal parameters. If
                -1, use all available cores.
            n_bins Optional[int]: 
                Overwrites ``n_bins`` specified in __init__. \n
                Number of bins to use when estimating the distributions. Using a
                large number of bins makes finding the separation point more
                noisy, and only slightly more accurate. (Default is ``None`` or
                ``50``)
            smoothing_window_bins (int): 
                Overwrites ``smoothing_window_bins`` specified in __init__. \n
                Number of bins to use when smoothing the distributions. Using a
                small number of bins makes finding the separation point more
                noisy, and only slightly more accurate. Aim for 5-10% of the
                number of bins. (Default is ``None`` or ``5``)
            seed (int):
                Seed for the random number generator in the optuna sampler.
                None: use a random seed.

        Returns:
            Dict:
                kwargs_makeConjunctiveDistanceMatrix_best (Dict):
                    The optimal parameters for the
                    self.make_conjunctive_distance_matrix function.
        """
        import optuna

        ## Store parameter (but not data) args as attributes
        self.params['find_optimal_parameters_for_pruning'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'kwargs_findParameters',
                'bounds_findParameters',
                'n_jobs_findParameters',
                'n_bins',
                'smoothing_window_bins',
                'seed',
            ],
        )

        self.n_bins = self.n_bins if n_bins is None else n_bins
        self.smoothing_window_bins = self.smooth_window if smoothing_window_bins is None else smoothing_window_bins

        self.bounds_findParameters = bounds_findParameters

        self._seed = seed
        np.random.seed(self._seed)

        print('Finding mixing parameters using automated hyperparameter tuning...') if self._verbose else None
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.checker = helpers.Convergence_checker_optuna(verbose=self._verbose>=2, **kwargs_findParameters)
        prog_bar = helpers.OptunaProgressBar(
            n_trials=kwargs_findParameters['max_trials'], 
            # timeout=kwargs_findParameters['max_duration'],
            # timeout=10,
            mininterval=5.0,
        )
        self.study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(
            n_startup_trials=kwargs_findParameters['n_patience']//2,
            seed=self._seed,
        ))
        self.study.optimize(
            func=self._objectiveFn_distSameMagnitude, 
            n_jobs=n_jobs_findParameters, 
            callbacks=[self.checker.check, prog_bar],
            n_trials=kwargs_findParameters['max_trials'],
            # show_progress_bar=self._verbose >= 1,
            show_progress_bar=False,
        )

        self.best_params = self.study.best_params.copy()
        [self.best_params.pop(p) for p in [
            # 'sig_SF_kwargs_mu',
            # 'sig_SF_kwargs_b',
            'sig_NN_kwargs_mu',
            'sig_NN_kwargs_b',
            'sig_SWT_kwargs_mu',
            'sig_SWT_kwargs_b'] if p in self.best_params.keys()]
        # # self.best_params['sig_SF_kwargs'] = {'mu': self.study.best_params['sig_SF_kwargs_mu'],
        # #                                 'b': self.study.best_params['sig_SF_kwargs_b'],}
        # self.best_params['sig_SF_kwargs'] = None
        self.best_params['sig_NN_kwargs'] = {'mu': self.study.best_params['sig_NN_kwargs_mu'],
                                        'b': self.study.best_params['sig_NN_kwargs_b'],}
        self.best_params['sig_SWT_kwargs'] = {'mu': self.study.best_params['sig_SWT_kwargs_mu'],
                                            'b': self.study.best_params['sig_SWT_kwargs_b'],}

        self.kwargs_makeConjunctiveDistanceMatrix_best={
            'power_SF': None,
            'power_NN': None,
            'power_SWT': None,
            'p_norm': None,
            'sig_SF_kwargs': None,
            'sig_NN_kwargs': None,
            'sig_SWT_kwargs': None,
        }
        self.kwargs_makeConjunctiveDistanceMatrix_best.update(self.best_params)
        print(f'Completed automatic parameter search. Best value found: {self.study.best_value} with parameters {self.best_params}') if self._verbose else None
        return self.kwargs_makeConjunctiveDistanceMatrix_best

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

        if kwargs_makeConjunctiveDistanceMatrix is None:
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
        # ssf.data[ssf.data == 0] = 1e-10
        # snn.data[snn.data == 0] = 1e-10
        # sswt.data[sswt.data == 0] = 1e-10

        min_d = np.nanmin(self.dConj.data)
        range_d = d_crossover - min_d
        self.d_cutoff = min_d + range_d * stringency if d_cutoff is None else d_cutoff
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
            #     min_samples=None,
                cluster_selection_epsilon=d_clusterMerge,
                max_cluster_size=n_sessions,
                metric='precomputed',
                alpha=alpha,
            #     p=None,
                algorithm='generic',
            #     leaf_size=100,
            # #     memory=Memory(location=None),
            #     approx_min_span_tree=False,
            #     gen_min_span_tree=False,
            #     core_dist_n_jobs=mp.cpu_count(),
                cluster_selection_method=cluster_selection_method,
            #     allow_single_cluster=False,
            #     prediction_data=False,
            #     match_reference_implementation=False,
                max_dist=max_dist
            )

            self.hdbs.fit(attach_fully_connected_node(
                d, 
                dist_fullyConnectedNode=max_dist,
                n_nodes=1,
            ))
            labels = self.hdbs.labels_[:-1]
            self.labels = labels

            print(f'Initial number of violating clusters: {len(np.unique(labels)[np.array([(session_bool[labels==u].sum(0)>1).sum().item() for u in np.unique(labels)]) > 0])}') if self._verbose else None

            ## Split up labels with multiple ROIs per session
            ## The below code is a bit of a mess, but it works.
            ##  It works by iteratively reducing the cutoff distance
            ##  and splitting up violating clusters until there are 
            ##  no more violations.
            if split_intraSession_clusters:
                labels = labels.copy()
                # d_cut = float(d.data.max())

                sb_t = torch.as_tensor(session_bool, dtype=torch.float32) # (n_rois, n_sessions)
                # print(f'num violating clusters: {np.unique(labels)[np.array([(session_bool[labels==u].sum(0)>1).sum().item() for u in np.unique(labels)]) > 0]}')
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
                    # print(f"Setting all clusters with redundant ROIs to label: -1. Number of clusters with redundant ROIs: {len(np.unique(labels[idx_toUpdate]))}") if self._verbose else None
                    # print(f"IDs: {np.unique(labels[idx_toUpdate])}") if self._verbose else None
                    labels[idx_toUpdate] = -1

            l_u = np.unique(labels)
            l_u = l_u[l_u > -1]
            if ii < n_iter_violationCorrection - 1:
                # print(f'Post-separation number of violating clusters: {n_violating_clusters}') if self._verbose else None
                ## find sessions represented in each cluster and set distances to ROIs in those sessions to 1.
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
            # todo todocument

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
            ],
        )
        
        p_norm = 1e-9 if p_norm == 0 else p_norm

        sSF_data = self._activation_function(s_sf.data, sig_SF_kwargs, power_SF) if s_sf is not None else None
        sNN_data = self._activation_function(s_NN.data, sig_NN_kwargs, power_NN) if s_NN is not None else None
        sSWT_data = self._activation_function(s_SWT.data, sig_SWT_kwargs, power_SWT) if s_SWT is not None else None

        s_list = [s for s in [sSF_data, sNN_data, sSWT_data] if s is not None]
        
        sConj_data = self._pNorm(
            s_list=s_list,
            p=p_norm,
        )
        # sConj_data = sConj_data * s_sesh.data if s_sesh is not None else sConj_data
        # sConj_data = sConj_data * np.logical_not(s_sesh.data) if s_sesh is not None else sConj_data

        ## make sConj
        sConj = s_sf.copy() if s_sf is not None else s_NN.copy() if s_NN is not None else s_SWT.copy()
        sConj.data = sConj_data.numpy() 
        sConj = sConj.multiply(s_sesh) if s_sesh is not None else sConj
        # sConj.eliminate_zeros()

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
        # return np.linalg.norm(np.stack(s_list_noNones, axis=0), ord=p, axis=0)

    # def plot_sigmoids(self):
    #     fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16,4))
    #     axs[0].plot(np.linspace(-0,2,1001), self.sig_sf(np.linspace(-5,5,1001)))
    #     axs[0].set_title('sigmoid SF')
    #     axs[1].plot(np.linspace(-1,1,1001), self.sig_NN(np.linspace(-5,5,1001)))
    #     axs[1].set_title('sigmoid NN')
    #     axs[2].plot(np.linspace(-1,1,1001), self.sig_SWT(np.linspace(-5,5,1001)))
    #     axs[2].set_title('sigmoid SWT')


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
        # centers = (edges[1:] + edges[:-1]) / 2
        if edges is d_crossover:
            print('No crossover found')
        
        fig = plt.figure()
        plt.stairs(dens_same, edges, linewidth=5, label='same')
        plt.stairs(dens_same_crop, edges, linewidth=3, label='same_cropped') if dens_same_crop is not None else None
        plt.stairs(dens_diff, edges, label='diff')
        plt.stairs(dens_all, edges, label='all')
        # plt.stairs(dens_diff - dens_same, edges)
        # plt.stairs(dens_all - dens_diff, edges)
        # plt.stairs((dens_diff * dens_same)*1000, edges)
        plt.axvline(d_crossover, color='k', linestyle='--') if d_crossover is not None else None
        plt.ylim([dens_same.max()*-0.5, dens_same.max()*1.5])
        plt.title('Pairwise similarities')
        plt.xlabel('distance or prob(different)')
        plt.ylabel('counts or density')
        # plt.legend(['same', 'same (cropped)', 'diff', 'all', 'diff - same', 'all - diff', '(diff * same) * 1000', 'crossover'])
        plt.legend()
        return fig

    def _fn_smooth(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Smooths a 1D tensor using a convolution operation.

        Args:
            x (torch.Tensor): 
                1D tensor to be smoothed.

        Returns:
            (torch.Tensor): 
                Smoothed tensor.
        """
        return helpers.Convolver_1d(
            kernel=torch.ones(self.smooth_window),
            length_x=self.n_bins,
            pad_mode='same',
            correct_edge_effects=True,
            device='cpu',
        ).convolve(x)

    def _separate_diffSame_distributions(
        self, 
        d_conj: scipy.sparse.csr_matrix
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Estimates the distribution of pairwise similarities for 'same' and
        'different' pairs of ROIs. Estimate the 'same' distribution as the
        difference between all pairwise distances (includes different and same)
        and intra-session distances (known different). 

        Args:
            d_conj (scipy.sparse.csr_matrix): 
                Conjunctive distance matrix.

        Returns:
            (tuple): tuple containing:
                dens_same_crop (np.ndarray):
                    Distribution density of pairwise similarities 
                    that are assumed to be from the same ROI. It is
                    'cropped' because values below crossover point
                    between same and different distributions are set
                    to zero.
                dens_same (np.ndarray):
                    Un-cropped version of dens_same_crop.
                dens_diff (np.ndarray):
                    Distribution density of pairwise similarities
                    that are assumed to be from different ROIs.
                dens_all (np.ndarray):
                    Distribution density of all pairwise similarities.
                edges (np.ndarray):
                    Edges of bins used to compute densities.
                d_crossover (float):
                    Distance at which the same and different distributions
                    crossover.
        """
        edges = torch.linspace(0,1, self.n_bins+1, dtype=torch.float32)
        
        d_all = d_conj.copy()
        counts, _ = torch.histogram(torch.as_tensor(d_all.data, dtype=torch.float32), edges)
        # dens_all = fn_smooth(counts / counts.sum())  ## distances of all pairs of ROIs
        # dens_all = counts / counts.sum()  ## distances of all pairs of ROIs
        dens_all = counts  ## distances of all pairs of ROIs
        # dens_all = counts / counts[-1]  ## distances of all pairs of ROIs

        d_intra = d_conj.multiply(self.s_sesh_inv)
        d_intra.eliminate_zeros()
        if len(d_intra.data) == 0:
            return None, None, None, None, None, None
        counts, _ = torch.histogram(torch.as_tensor(d_intra.data, dtype=torch.float32), edges)
        # dens_diff = fn_smooth(counts / counts.sum())  ## distances of known differents
        # dens_diff = counts / counts.sum()  ## distances of known differents
        dens_diff = counts * (len(d_all.data) / len(d_intra.data))
        # dens_diff = counts / counts[-1]  ## distances of known differents

        dens_same = dens_all - dens_diff  ## estimate the 'same' distribution as the different between all distances (includes different and same) and intra-session distances (known different)
        dens_same = torch.maximum(dens_same, torch.as_tensor([0], dtype=torch.float32))
        # dens_same = torch.as_tensor(scipy.signal.savgol_filter(dens_same.cpu().numpy(), self.smooth_window, 3))  ## smooth the 'same' distribution
        dens_same = self._fn_smooth(dens_same)

        dens_deriv = dens_diff - dens_same  ## difference in 'diff' and 'same' distributions
        dens_deriv[dens_diff.argmax():] = 0
        if torch.where(dens_deriv < 0)[0].shape[0] == 0:  ## if no crossover point, return None
            return None, dens_same, dens_diff, dens_all, edges, None
        idx_crossover = torch.where(dens_deriv < 0)[0][-1] + 1 
        d_crossover = edges[idx_crossover].item()

        dens_same_crop = dens_same.clone()
        dens_same_crop[idx_crossover:] = 0
        return dens_same_crop, dens_same, dens_diff, dens_all, edges, d_crossover
    def _objectiveFn_distSameMagnitude(
        self, 
        trial: object,
    ) -> float:
        """
        Computes the magnitude of the 'same' distribution for Optuna
        hyperparameter optimization.

        The 'same' distribution refers to the distribution of distances between
        pairs of ROIs that are estimated to be identical. As the parameters for
        building the conjunctive distance matrix are optimized, the 'same' and
        'different' distributions should separate from each other. The less the
        two overlap, the larger the effective magnitude of the 'same'
        distribution.

        Args:
            trial (optuna.trial.Trial): 
                The Optuna trial object.

        Returns:
            (float): 
                loss (float):
                    The magnitude of the 'same' distribution. This output must
                    be a scalar and is used to update the hyperparameters.
        """
        # power_SF = trial.suggest_float('power_SF', *self.bounds_findParameters['power_SF'], log=False)
        power_SF = 1
        power_NN = trial.suggest_float('power_NN', *self.bounds_findParameters['power_NN'], log=False)
        power_SWT = trial.suggest_float('power_SWT', *self.bounds_findParameters['power_SWT'], log=False)
        # power_SWT = 0
        p_norm = trial.suggest_float('p_norm', *self.bounds_findParameters['p_norm'], log=False)
        
        # sig_SF_kwargs = {
        #     'mu':trial.suggest_float('sig_SF_kwargs_mu', 0.1, 0.5, log=False),
        #     'b':trial.suggest_float('sig_SF_kwargs_b', 0.1, 2, log=False),
        # }
        sig_SF_kwargs = None
        sig_NN_kwargs = {
            'mu':trial.suggest_float('sig_NN_kwargs_mu', *self.bounds_findParameters['sig_NN_kwargs_mu'], log=False),
            'b':trial.suggest_float('sig_NN_kwargs_b', *self.bounds_findParameters['sig_NN_kwargs_b'], log=False),
        }
        # sig_NN_kwargs = None
        sig_SWT_kwargs = {
            'mu':trial.suggest_float('sig_SWT_kwargs_mu', *self.bounds_findParameters['sig_SWT_kwargs_mu'], log=False),
            'b':trial.suggest_float('sig_SWT_kwargs_b', *self.bounds_findParameters['sig_SWT_kwargs_b'], log=False),
        }
        # sig_SWT_kwargs = None

        dConj, sConj, sSF_data, sNN_data, sSWT_data, sConj_data = self.make_conjunctive_distance_matrix(
            s_sf=self.s_sf,
            s_NN=self.s_NN_z,
            s_SWT=self.s_SWT_z,
            s_sesh=None,
            power_SF=power_SF,
            power_NN=power_NN,
            power_SWT=power_SWT,
            p_norm=p_norm,
        #     sig_sf_kwargs={'mu':1.0, 'b':0.5},
            sig_SF_kwargs=sig_SF_kwargs,
            sig_NN_kwargs=sig_NN_kwargs,
            sig_SWT_kwargs=sig_SWT_kwargs,
        )
        
        dens_same_crop, dens_same, dens_diff, dens_all, edges, d_crossover = self._separate_diffSame_distributions(dConj)
        if dens_same_crop is None:
            return 0

        # loss = dens_same_crop.sum().item()
        loss = (dens_same * dens_diff).sum().item()
        
        return loss  # Output must be a scalar. Used to update the hyperparameters
    

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
        ## Warn that current version is memory intensive and might be improved when sklearn 1.3 is released
        # warnings.warn("Current version of silhouette samples calculation is memory intensive and will be improved when sklearn 1.3 is released.")
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

    # Hungarian matching score
    if ignore_negOne:        
        # labels_test = labels_test[labels_true > -1].copy()
        # labels_true = labels_true[labels_true > -1].copy()
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

    _, counts = np.unique(labels[labels!=-1], return_counts=True)

    axs[1,1].hist(counts, n_sessions*2 + 1, range=(0, n_sessions+1));
    axs[1,1].set_xlabel('n_sessions')
    axs[1,1].set_ylabel('cluster counts');
    
    # Make the title include the number of excluded (label==-1) ROIs
    fig.suptitle(f'Quality metrics n_excluded: {np.sum(labels==-1)}, n_included: {np.sum(labels!=-1)}, n_total: {len(labels)}, n_clusters: {len(np.unique(labels[labels!=-1]))}, n_sessions: {n_sessions}')
    return fig, axs
