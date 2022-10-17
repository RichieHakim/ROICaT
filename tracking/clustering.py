import time

import numpy as np
import scipy
import scipy.optimize
import scipy.sparse
import sklearn
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

import optuna
import hdbscan

from . import helpers

class Clusterer:
    """
    Base class for clustering algorithms.

    RH 2022
    """
    def __init__(
        self,
        s_sf=None,
        s_NN_z=None,
        s_SWT_z=None,
        s_sesh=None,
        verbose=True,
    ):
        """
        Initialise the clusterer object.

        Args:
            s_sf (scipy.sparse.csr_matrix):
                Similarity matrix for spatial footprints.
                Shape: (n_rois, n_rois). Symmetric.
                Expecting input to be manhattan distance of
                 spatial footprints normalized between 0 and 1.
            s_NN_z (scipy.sparse.csr_matrix):
                Z-scored similarity matrix for neural network
                 output similaries.
                Shape: (n_rois, n_rois). Non-symmetric.
                Expecting input to be the cosine similarity 
                 matrix, z-scored row-wise.
            s_SWT_z (scipy.sparse.csr_matrix):
                Z-scored similarity matrix for scattering 
                 transform output similarities.
                Shape: (n_rois, n_rois). Non-symmetric.
                Expecting input to be the cosine similarity
                 matrix, z-scored row-wise.
            s_sesh (scipy.sparse.csr_matrix, boolean):
                Similarity matrix for session similarity.
                Shape: (n_rois, n_rois). Symmetric.
                Expecting input to be boolean, with 1s where
                 the two ROIs are from DIFFERENT sessions.
            verbose (bool):
                Whether to print out information about the
                 clustering process.
        """

        self.s_sf = s_sf
        self.s_NN_z = s_NN_z
        self.s_SWT_z = s_SWT_z
        self.s_sesh = s_sesh

        self.s_sesh_inv = (self.s_sf != 0).astype(np.bool8)
        self.s_sesh_inv[self.s_sesh.astype(np.bool8)] = False
        self.s_sesh_inv.eliminate_zeros()

        self.s_sesh = self.s_sesh.tolil()
        self.s_sesh[range(self.s_sesh.shape[0]), range(self.s_sesh.shape[1])] = 0
        self.s_sesh = self.s_sesh.tocsr()

        self._verbose = verbose

    def find_optimal_parameters_for_pruning(
        self,
        n_bins=50,
        find_parameters_automatically=True,
        kwargs_findParameters={
            'n_patience': 100,
            'tol_frac': 0.05,
            'max_trials': 350,
            'max_duration': 60*10,
            'verbose': False,
        },
        bounds_findParameters={
            'power_SF': (0.3, 2),
            'power_NN': (0.2, 2),
            'power_SWT': (0.1, 1),
            'p_norm': (-5, 5),
            'sig_NN_kwargs_mu': (0, 0.5),
            'sig_NN_kwargs_b': (0.05, 2),
            'sig_SWT_kwargs_mu': (0, 0.5),
            'sig_SWT_kwargs_b': (0.05, 2),
        },
        n_jobs_findParameters=-1,
    ):
        """
        Find the optimal parameters for pruning the similarity graph.
        How this function works:
            1. Make a conjunctive distance matrix using a set of parameters
             for the self.make_conjunctive_distance_matrix function.
            2. Estimates the distribution of pairwise distances
             between ROIs assumed to be the same and those assumed to be
             different ROIs. This is done by comparing the difference in the 
             distribution of pairwise distances between ROIs from the same
             session and those from different sessions. Ideally, the
             main difference will be the presence of 'same' ROIs in the
             inter-session distribution. 
            3. The optimal parameters are then updated using optuna in order
             to maximize the separation between the 'same' and 'different'
             distributions.

        Args:
            n_bins (int):
                Number of bins to use when estimating the distributions. Using
                 a large number of bins makes finding the separation point more
                 noisy, and only slightly more accurate.
            find_parameters_automatically (bool):
                Whether to find the optimal parameters automatically using
                 optuna. If False, the parameters are set to the 
                 kwargs_makeConjunctiveDistanceMatrix input.
            kwargs_findParameters (dict):
                Keyword arguments for the Convergence_checker class __init__.
            bounds_findParameters (dict):
                Bounds for the parameters to be optimized.
            n_jobs_findParameters (int):
                Number of jobs to use when finding the optimal parameters.
                If -1, use all available cores.

        Returns:
            kwargs_makeConjunctiveDistanceMatrix_best (dict):
                The optimal (for pruning) keyword arguments using the
                 self.make_conjunctive_distance_matrix function.
        """

        self.bounds_findParameters = bounds_findParameters

        self.n_bins = self.s_sf.nnz // 10000 if n_bins is None else n_bins
        # smoothing_window = self.n_bins // 100
        # print(f'Pruning similarity graphs with {self.n_bins} bins and smoothing window {smoothing_window}...') if self._verbose else None

        if find_parameters_automatically:
            print('Finding mixing parameters using automated hyperparameter tuning...') if self._verbose else None
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.checker = Convergence_checker(**kwargs_findParameters)
            self.study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(n_startup_trials=10))
            self.study.optimize(self._objectiveFn_distSameMagnitude, n_jobs=n_jobs_findParameters, callbacks=[self.checker.check])

            self.best_params = self.study.best_params.copy()
            [self.best_params.pop(p) for p in [
                # 'sig_SF_kwargs_mu',
                # 'sig_SF_kwargs_b',
                'sig_NN_kwargs_mu',
                'sig_NN_kwargs_b',
                'sig_SWT_kwargs_mu',
                'sig_SWT_kwargs_b']]
            # self.best_params['sig_SF_kwargs'] = {'mu': self.study.best_params['sig_SF_kwargs_mu'],
            #                                 'b': self.study.best_params['sig_SF_kwargs_b'],}
            self.best_params['sig_SF_kwargs'] = None
            self.best_params['sig_NN_kwargs'] = {'mu': self.study.best_params['sig_NN_kwargs_mu'],
                                            'b': self.study.best_params['sig_NN_kwargs_b'],}
            self.best_params['sig_SWT_kwargs'] = {'mu': self.study.best_params['sig_SWT_kwargs_mu'],
                                             'b': self.study.best_params['sig_SWT_kwargs_b'],}
            self.kwargs_makeConjunctiveDistanceMatrix_best = self.best_params
            print(f'Best value found: {self.study.best_value} with parameters {self.best_params}') if self._verbose else None
        else:
            self.kwargs_makeConjunctiveDistanceMatrix_best = self.best_params
        return self.kwargs_makeConjunctiveDistanceMatrix_best

    def make_pruned_similarity_graphs(
        self,
        d_cutoff=None,
        kwargs_makeConjunctiveDistanceMatrix={
            'power_SF': 0.5,
            'power_NN': 1.0,
            'power_SWT': 0.1,
            'p_norm': -4.0,
            'sig_SF_kwargs': {'mu':0.5, 'b':0.5},
            'sig_NN_kwargs': {'mu':0.5, 'b':0.5},
            'sig_SWT_kwargs': {'mu':0.5, 'b':0.5},
        },
    ):
        dConj, sSF_data, sNN_data, sSWT_data, sConj_data = self.make_conjunctive_distance_matrix(
            s_sf=self.s_sf,
            s_NN=self.s_NN_z,
            s_SWT=self.s_SWT_z,
            **kwargs_makeConjunctiveDistanceMatrix
        )
        dens_same_crop, dens_same, dens_diff, dens_all, edges, d_crossover = self._separate_diffSame_distributions(dConj)

        ssf, snn, sswt = self.s_sf.copy(), self.s_NN_z.copy(), self.s_SWT_z.copy()
        ssf.data[ssf.data == 0] = 1e-10
        snn.data[snn.data == 0] = 1e-10
        sswt.data[sswt.data == 0] = 1e-10

        self.d_cutoff = d_crossover if d_cutoff is None else d_cutoff
        self.graph_pruned = dConj > self.d_cutoff
        self.graph_pruned.eliminate_zeros()
        
        def prune(s, graph_pruned):
            if s is None:
                return None
            s_pruned = s.copy()
            s_pruned[graph_pruned] = 0
            s_pruned.eliminate_zeros()
            return s_pruned

        self.s_sf_pruned, self.s_NN_pruned, self.s_SWT_pruned = tuple([prune(s, self.graph_pruned) for s in [ssf, snn, sswt]])

    def fit(self,
        session_bool,
        min_cluster_size=2,
        cluster_selection_method='leaf',
        d_clusterMerge=None,
        alpha=0.999,
        n_iter_violationCorrection=5,
        d_conj=None,
        kwargs_makeConjunctiveDistanceMatrix={
            'power_SF': 1.0,
            'power_NN': 1.0,
            'power_SWT': 0.1,
            'p_norm': -2,
            'sig_SF_kwargs': None,
            'sig_NN_kwargs':  {'mu':0, 'b':0.2},
            'sig_SWT_kwargs': {'mu':0, 'b':0.2},
        },
        split_intraSession_clusters=True,
        discard_failed_pruning=True,
        d_step=0.05,
    ):
        """
        Fit clustering using the ROICaT clustering algorithm.

        Args:
            session_bool (np.ndarray of bool):
                Boolean array indicating which ROIs belong to which session.
                shape: (n_rois, n_sessions)
            min_cluster_size (int):
                Minimum cluster size to be considered a cluster.
            d_clusterMerge (float):
                Distance threshold for merging clusters.
                All clusters with ROIs closer than this distance will be merged.
                If None, the distance is calculated as the mean + 1*std of the 
                 conjunctive distances.
            cluster_selection_method (str):
                Cluster selection method. Either 'leaf' or 'eom'.
                'leaf' tends towards smaller clusters, 'eom' towards larger clusters.
                See HDBSCAN documentation: 
                 https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
            alpha (float):
                Alpha value. Avoid messing with this if possible.
                Smaller values will result in more clusters.
                See HDBSCAN documentation: 
                 https://hdbscan.readthedocs.io/en/latest/parameter_selection.html
            n_iter_violationCorrection (int):
                Number of iterations to correct for clusters with multiple ROIs per session.
                After cluster splitting, this will go through each cluster and disconnect
                 the ROIs in that cluster from all other ROIs that share a session with any
                 ROI in that cluster. Then it will re-run the HDBSCAN clustering algorithm.
                This is done to overcome the issues with single-linkage clustering finding
                 clusters with multiple ROIs per session.
                Usually this converges after about 5 iterations, and n_sessions at most.
            d_conj (float):
                Conjunctive distance matrix.
                If None, the distance matrix is calculated using the
                 kwargs_makeConjunctiveDistanceMatrix.
            kwargs_makeConjunctiveDistanceMatrix (dict):
                Keyword arguments for the make_conjunctive_distance_matrix function.
                Only used if d_conj is None.
            split_intraSession_clusters (bool):
                If True, clusters that contain ROIs from multiple sessions will be split.
                Only set to False if you want clusters containing multiple
                 ROIs from the same session.
            discard_failed_pruning (bool):
                If True, clusters that fail to prune will be set to -1.
            d_step (float):
                Distance step size for splitting clusters with multiple ROIs from
                 the same session. Higher values are faster but less accurate.

        Returns:
            labels (np.ndarray of int):
                Cluster labels for each ROI.
                shape: (n_rois_total)
        """
        ## Make conjunctive distance matrix
        if d_conj is None:
            d_conj, _, _, _, _ = self.make_conjunctive_distance_matrix(
                s_sf=self.s_sf_pruned,
                s_NN=self.s_NN_pruned,
                s_SWT=self.s_SWT_pruned,
                **kwargs_makeConjunctiveDistanceMatrix
            )

        d = d_conj.copy()

        print('Fitting with HDBSCAN and splitting clusters with multiple ROIs per session') if self._verbose else None
        for ii in tqdm(range(n_iter_violationCorrection)):
            ## Prep parameters for splitting clusters
            d_clusterMerge = float(np.mean(d.data) + 1*np.std(d.data)) if d_clusterMerge is None else float(d_clusterMerge)
            d_step = float(d_step)

            n_sessions = session_bool.shape[1]
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
                dist_fullyConnectedNode=max_dist
            ))
            labels = self.hdbs.labels_[:-1]
            self.labels = labels

            ## Split up labels with multiple ROIs per session
            ## The below code is a bit of a mess, but it works.
            ##  It works by iteratively reducing the cutoff distance
            ##  and splitting up violating clusters until there are 
            ##  no more violations.
            if split_intraSession_clusters:
                labels = labels.copy()
                d_cut = float(d.data.max())

                sb_t = torch.as_tensor(session_bool, dtype=torch.float32) # (n_rois, n_sessions)
                # print(f'num violating clusters: {np.unique(labels)[np.array([(session_bool[labels==u].sum(0)>1).sum().item() for u in np.unique(labels)]) > 0]}')
                success = False
                while success == False:
                    labels_t = torch.as_tensor(labels, dtype=torch.int64)
                    lab_u_t, lab_u_idx_t = torch.unique(labels_t, return_inverse=True) # (n_clusters,), (n_rois,)
                    lab_oneHot_t = helpers.idx_to_oneHot(lab_u_idx_t, dtype=torch.float32)
                    violations_labels = lab_u_t[((sb_t.T @ lab_oneHot_t) > 1.5).sum(0) > 0]
                    violations_labels = violations_labels[violations_labels > -1]

                    if len(violations_labels) == 0:
                        success = True
                        break
                    
                    for l in violations_labels:
                        idx = np.where(labels==l)[0]
                        if d[idx][:,idx].nnz == 0:
                            labels[idx] = -1

                    labels_new = self.hdbs.single_linkage_tree_.get_clusters(
                        cut_distance=d_cut,
                        min_cluster_size=2,
                    )[:-1]
                    
                    idx_toUpdate = np.isin(labels, violations_labels)
                    labels[idx_toUpdate] = labels_new[idx_toUpdate] + labels.max() + 5
                    labels[(labels_new == -1) * idx_toUpdate] = -1
                    # labels = helpers.squeeze_integers(labels)
                    d_cut -= d_step
                    
                    if d_cut < 0.0:
                        print(f"RH WARNING: Redundant session cluster splitting did not complete fully. Distance walk failed at 'd_cut':{d_cut}.")
                        if discard_failed_pruning:
                            print(f"Setting all clusters with redundant ROIs to label: -1.")
                            labels[idx_toUpdate] = -1
                        break
                
            if ii < n_iter_violationCorrection - 1:
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
        session_bool,
        thresh_cost=0.95,
        d_conj=None,
        kwargs_makeConjunctiveDistanceMatrix={
            'power_SF': 1.0,
            'power_NN': 1.0,
            'power_SWT': 0.1,
            'p_norm': -2,
            'sig_SF_kwargs': None,
            'sig_NN_kwargs':  {'mu':0, 'b':0.2},
            'sig_SWT_kwargs': {'mu':0, 'b':0.2},
        },
    ):
        """
        Use CaImAn's method for clustering.
        See their paper and repo for details:
            https://elifesciences.org/articles/38173#s4
            https://github.com/flatironinstitute/CaImAn
            https://github.com/flatironinstitute/CaImAn/blob/master/caiman/base/rois.py
        """
        print(f"Clustering with CaImAn's iterative Hungarian algorithm method...") if self._verbose else None
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

        if d_conj is None:
            d_conj, _, _, _, _ = self.make_conjunctive_distance_matrix(
                s_sf=self.s_sf_pruned,
                s_NN=self.s_NN_pruned,
                s_SWT=self.s_SWT_pruned,
                **kwargs_makeConjunctiveDistanceMatrix
            )

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

        self.performance = performance

        labels = np.concatenate(matchings)
        u, c = np.unique(labels, return_counts=True)
        labels[np.isin(labels, u[c == 1])] = -1
        self.labels = labels
        return self.labels
            

    def make_conjunctive_distance_matrix(
        self,
        s_sf=None,
        s_NN=None,
        s_SWT=None,
        power_SF=1,
        power_NN=1,
        power_SWT=1,
        p_norm=1,
        sig_SF_kwargs={'mu':0.5, 'b':0.5},
        sig_NN_kwargs={'mu':0.5, 'b':0.5},
        sig_SWT_kwargs={'mu':0.5, 'b':0.5},
    ):
        """
        Make a distance matrix from the three similarity matrices.

        Args:
            s_sf (scipy.sparse.csr_matrix):
                Similarity matrix for spatial footprints.
            s_NN_z (scipy.sparse.csr_matrix):
                Z-scored similarity matrix for neural network features.
            s_SWT_z (scipy.sparse.csr_matrix):
                Z-scored similarity matrix for scattering wavelet 
                 transform features.
            power_SF (float):
                Power to which to raise the spatial footprint similarity.
            power_NN (float):
                Power to which to raise the neural network similarity.
            power_SWT (float):
                Power to which to raise the scattering wavelet transform 
                 similarity.
            p_norm (float):
                p-norm to use for the conjunction of the similarity
                 matrices.
            sig_SF_kwargs (dict):
                Keyword arguments for the sigmoid function applied to the
                 spatial footprint overlap similarity matrix.
                See helpers.generalised_logistic_function for details.
            sig_NN_kwargs (dict):
                Keyword arguments for the sigmoid function applied to the
                 neural network similarity matrix.
                See helpers.generalised_logistic_function for details.
            sig_SWT_kwargs (dict):
                Keyword arguments for the sigmoid function applied to the
                 scattering wavelet transform similarity matrix.
                See helpers.generalised_logistic_function for details.
            plot_sigmoid (bool):
                Whether to plot the sigmoid functions applied to the
                 neural network and scattering wavelet transform
                 similarity matrices.
        """
        assert (s_sf is not None) or (s_NN is not None) or (s_SWT is not None), \
            'At least one of s_sf, s_NN, or s_SWT must be provided.'
        
        p_norm = 1e-9 if p_norm == 0 else p_norm

        sSF_data = self._activation_function(s_sf.data, sig_SF_kwargs, power_SF)
        sNN_data = self._activation_function(s_NN.data, sig_NN_kwargs, power_NN)
        sSWT_data = self._activation_function(s_SWT.data, sig_SWT_kwargs, power_SWT)
        
        sConj_data = self._pNorm(
            s_list=[sSF_data, sNN_data, sSWT_data],
            p=p_norm,
        )

        dConj = s_sf.copy()
        dConj.data = sConj_data.numpy()
        # dConj.eliminate_zeros()
        dConj.data = 1 - dConj.data

        return dConj, sSF_data, sNN_data, sSWT_data, sConj_data

    def _activation_function(self, s, sig_kwargs={'mu':0.0, 'b':1.0}, power=1):
        if s is None:
            return None
        if (sig_kwargs is not None) and (power is not None):
            return helpers.generalised_logistic_function(torch.as_tensor(s, dtype=torch.float32), **sig_kwargs)**power
        elif (sig_kwargs is None) and (power is not None):
            return torch.maximum(torch.as_tensor(s, dtype=torch.float32), torch.as_tensor([0], dtype=torch.float32))**power
            # return torch.as_tensor(s, dtype=torch.float32)**power
        elif (sig_kwargs is not None) and (power is None):
            return helpers.generalised_logistic_function(torch.as_tensor(s, dtype=torch.float32), **sig_kwargs)

    def _pNorm(self, s_list, p):
        """
        Calculate the p-norm of a list of similarity matrices.
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
        plots_to_show=[1,2,3], 
        max_samples=1000000, 
        kwargs_scatter={'s':1, 'alpha':0.1},
        kwargs_makeConjunctiveDistanceMatrix={
            'power_SF': 0.5,
            'power_NN': 1.0,
            'power_SWT': 0.1,
            'p_norm': -4.0,
            'sig_SF_kwargs': {'mu':0.5, 'b':0.5},
            'sig_NN_kwargs': {'mu':0.5, 'b':0.5},
            'sig_SWT_kwargs': {'mu':0.5, 'b':0.5},
        },
    ):
        """
        Plot the similarity relationships between the three similarity
         matrices.

        Args:
            plots_to_show (list):
                Which plots to show.
                1: Spatial footprints vs. neural network features.
                2: Spatial footprints vs. scattering wavelet transform
                 features.
                3: Neural network features vs. scattering wavelet.
            max_samples (int):
                Maximum number of samples to plot.
                Use smaller numbers for faster plotting.
            kwargs_scatter (dict):
                Keyword arguments for the matplotlib.pyplot.scatter plot.
            kwargs_makeConjunctiveDistanceMatrix (dict):
                Keyword arguments for the makeConjunctiveDistanceMatrix
                 method.

        Returns:
            fig (matplotlib.pyplot.figure):
                Figure object.
            axs (matplotlib.pyplot.axes):
                Axes object.
        """
        dConj, sSF_data, sNN_data, sSWT_data, sConj_data = self.make_conjunctive_distance_matrix(
            s_sf=self.s_sf,
            s_NN=self.s_NN_z,
            s_SWT=self.s_SWT_z,
            **kwargs_makeConjunctiveDistanceMatrix
        )

        ## subsample similarities for plotting
        idx_rand = np.floor(np.random.rand(min(max_samples, len(sSF_data))) * len(sSF_data)).astype(int)
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

    def plot_distSame(self, kwargs_makeConjunctiveDistanceMatrix=None):
        """
        Plot the estimated distribution of the pairwise similarities
         between matched ROI pairs of ROIs.
        """
        kwargs = kwargs_makeConjunctiveDistanceMatrix if kwargs_makeConjunctiveDistanceMatrix is not None else self.best_params
        dConj, sSF_data, sNN_data, sSWT_data, sConj_data = self.make_conjunctive_distance_matrix(
            s_sf=self.s_sf,
            s_NN=self.s_NN_z,
            s_SWT=self.s_SWT_z,
            **kwargs
        )
        dens_same_crop, dens_same, dens_diff, dens_all, edges, d_crossover = self._separate_diffSame_distributions(dConj)
        centers = (edges[1:] + edges[:-1]) / 2
        
        plt.figure()
        plt.stairs(dens_same, edges)
        plt.stairs(dens_same_crop, edges)
        plt.stairs(dens_diff, edges)
        plt.stairs(dens_all, edges)
        plt.stairs(dens_diff - dens_same, edges)
        plt.stairs((dens_diff * dens_same)*1000, edges)
        plt.axvline(d_crossover, color='k', linestyle='--')
        plt.ylim([dens_same.max()*-0.5, dens_same.max()*1.5])
        plt.xlabel('distance')
        plt.ylabel('density')
        plt.legend(['same', 'same (cropped)', 'diff', 'all', 'diff - same', '(diff * same) * 1000', 'crossover'])

    def _separate_diffSame_distributions(self, d_conj):
        """
        Estimates the distribution of pairwise similarities for
         'same' and 'different' pairs of ROIs. 
        estimate the 'same' distribution as the different between
         all pairwise distances (includes different and same) and
         intra-session distances (known different). same = all - intra
        See _objectiveFn_distSameMagnitude docstring for more details.

        Args:
            d_conj (scipy.sparse.csr_matrix):
                conjunctive distance matrix.

        Returns:
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
        dens_all = counts / counts.sum()  ## distances of all pairs of ROIs

        d_intra = d_conj.multiply(self.s_sesh_inv)
        d_intra.eliminate_zeros()
        counts, _ = torch.histogram(torch.as_tensor(d_intra.data, dtype=torch.float32), edges)
        dens_diff = counts / counts.sum()  ## distances of known differents

        dens_same = dens_all - dens_diff  ## estimate the 'same' distribution as the different between all distances (includes different and same) and intra-session distances (known different)
        dens_same = torch.maximum(dens_same, torch.as_tensor([0], dtype=torch.float32))

        dens_deriv = dens_diff - dens_same  ## difference in 'diff' and 'same' distributions
        dens_deriv[dens_diff.argmax():] = 0
        if torch.where(dens_deriv < 0)[0].shape[0] == 0:
            return None, None, None, None, None, None
        idx_crossover = torch.where(dens_deriv < 0)[0][-1] + 1 
        d_crossover = edges[idx_crossover].item()

        dens_same_crop = dens_same.clone()
        dens_same_crop[idx_crossover:] = 0
        return dens_same_crop, dens_same, dens_diff, dens_all, edges, d_crossover
    def _objectiveFn_distSameMagnitude(self, trial):
        """
        Objective function for Optuna hyperparameter optimization.
        Measures and outputs the magnitude of the 'same' distribution.
        The 'same' distribution is the distribution of distances 
         between pairs of ROIs that are estimated to be the same.
        As the parameters for building the conjuctive distance matrix
         are optimized, the 'same' and 'different' distributions should
         separate from each other. The less the two overlap, the larger
         the effective magnitude of the 'same' distribution.
        
        Args:
            trial (optuna.trial.Trial): 
                Optuna trial object.

        Returns:
            loss (float):
                Magnitude of the 'same' distribution.
        """
        power_SF=trial.suggest_float('power_SF', *self.bounds_findParameters['power_SF'], log=False)
        power_NN=trial.suggest_float('power_NN', *self.bounds_findParameters['power_NN'], log=False)
        power_SWT=trial.suggest_float('power_SWT', *self.bounds_findParameters['power_SWT'], log=False)
        p_norm=trial.suggest_float('p_norm', *self.bounds_findParameters['p_norm'], log=False)
        
        # sig_SF_kwargs = {
        #     'mu':trial.suggest_float('sig_SF_kwargs_mu', 0.1, 0.5, log=False),
        #     'b':trial.suggest_float('sig_SF_kwargs_b', 0.1, 2, log=False),
        # }
        sig_SF_kwargs = None
        sig_NN_kwargs = {
            'mu':trial.suggest_float('sig_NN_kwargs_mu', *self.bounds_findParameters['sig_NN_kwargs_mu'], log=False),
            'b':trial.suggest_float('sig_NN_kwargs_b', *self.bounds_findParameters['sig_NN_kwargs_b'], log=False),
        }
        sig_SWT_kwargs = {
            'mu':trial.suggest_float('sig_SWT_kwargs_mu', *self.bounds_findParameters['sig_SWT_kwargs_mu'], log=False),
            'b':trial.suggest_float('sig_SWT_kwargs_b', *self.bounds_findParameters['sig_SWT_kwargs_b'], log=False),
        }

        dConj, sSF_data, sNN_data, sSWT_data, sConj_data = self.make_conjunctive_distance_matrix(
            s_sf=self.s_sf,
            s_NN=self.s_NN_z,
            s_SWT=self.s_SWT_z,
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

        loss = dens_same_crop.sum().item()
        
        return loss  # Output must be a scalar. Used to update the hyperparameters
        

def attach_fully_connected_node(d, dist_fullyConnectedNode=None):
    """
    This function takes in a sparse distance graph (csr_matrix) that has
     more than one component (multiple unconnected subgraphs) and appends
     a single node to the graph that is weakly connected to all nodes.
     
    Args:
        d (scipy.sparse.csr_matrix):
            Sparse graph with multiple components.
            See scipy.sparse.csgraph.connected_components
        dist_fullyConnectedNode (float):
            Value to use for the connection strengh to all other nodes.
            Value will be appended as elements in a new row and column at
             the ends of the 'd' matrix.
             
     Returns:
         d2 (scipy.sparse.csr_matrix):
             Sparse graph with only one component.
    """
    if dist_fullyConnectedNode is None:
        dist_fullyConnectedNode = (d.max() - d.min()) * 1000
    
    d2 = d.copy()
    d2 = scipy.sparse.vstack((d2, np.ones((1,d2.shape[1]))*dist_fullyConnectedNode))
    d2 = scipy.sparse.hstack((d2, np.ones((d2.shape[0],1))*dist_fullyConnectedNode))

    return d2.tocsr()

class Convergence_checker:
    """
    A class that is used to check if the optuna optimization has converged.
    """
    def __init__(
        self, 
        n_patience=10, 
        tol_frac=0.05, 
        max_trials=350, 
        max_duration=60*10, 
        verbose=True
    ):
        """
        Args:
            n_patience (int):
                Number of trials to look back to check for convergence.
                Also the minimum number of trials that must be completed
                 before starting to check for convergence.
            tol_frac (float):
                Fractional tolerance for convergence.
                The best output value must change by less than this 
                 fractional amount to be considered converged.
            max_trials (int):
                Maximum number of trials to run before stopping.
            max_duration (float):
                Maximum number of seconds to run before stopping.
            verbose (bool):
                If True, print messages.
        """
        self.bests = []
        self.best = -np.inf
        self.n_patience = n_patience
        self.tol_frac = tol_frac
        self.max_trials = max_trials
        self.max_duration = max_duration
        self.num_trial = 0
        self.verbose = verbose
        
    def check(self, study, trial):
        """
        Check if the optuna optimization has converged.
        This function should be used as the callback function for the
         optuna study.

        Args:
            study (optuna.study.Study):
                Optuna study object.
            trial (optuna.trial.FrozenTrial):
                Optuna trial object.
        """
        dur_first, dur_last = study.trials[0].datetime_complete, trial.datetime_complete
        if (dur_first is not None) and (dur_last is not None):
            duration = (dur_last - dur_first).total_seconds()
        else:
            duration = 0
        
        if trial.value > self.best:
            self.best = trial.value
        self.bests.append(self.best)
            
        bests_recent = np.unique(self.bests[-self.n_patience:])
        if self.num_trial > self.n_patience and (((bests_recent.max() - bests_recent.min())/bests_recent.max()) < self.tol_frac):
            print(f'Stopping. Convergence reached. Best value ({self.best}) over last ({self.n_patience}) trials fractionally changed less than ({self.tol_frac})') if self.verbose else None
            study.stop()
        if self.num_trial >= self.max_trials:
            print(f'Stopping. Trial number limit reached. num_trial={self.num_trial}, max_trials={self.max_trials}.') if self.verbose else None
            study.stop()
        if duration > self.max_duration:
            print(f'Stopping. Duration limit reached. study.duration={duration}, max_duration={self.max_duration}.') if self.verbose else None
            study.stop()
            
        if self.verbose:
            print(f'Trial num: {self.num_trial}. Duration: {duration:.3f}s. Best value: {self.best:3f}. Current value:{trial.value:3f}') if self.verbose else None
        self.num_trial += 1


def score_labels(labels_test, labels_true, ignore_negOne=False, thresh_perfect=0.9999999999, compute_mutual_info=False):
    """
    Compute the score of the clustering.
    The best match is found by solving the linear sum assignment problem.
    The score is bounded between 0 and 1.

    Note: The score is not symmetric if the number of true and test
     labels are not the same. I.e. switching labels_test and labels_true
     can lead to different scores. This is not a bug. This is because
     we are scoring how well each true set is matched by an optimally
     assigned test set.

    RH 2022

    Args:
        labels_test (np.array): 
            Labels of the test clusters/sets.
        labels_true (np.array):
            Labels of the true clusters/sets.
        thresh_perfect (float):
            threshold for perfect match.
            Mostly used for numerical stability.

    Returns:
        score_unweighted_partial (float):
            Average correlation between the 
             best matched sets of true and test labels.
        score_unweighted_perfect (float):
            Fraction of perfect matches.
        score_weighted_partial (float):
            Average correlation between the best matched sets of
             true and test labels. Weighted by the number of elements
             in each true set.
        score_weighted_perfect (float):
            Fraction of perfect matches. Weighted by the number of
             elements in each true set.
        hi (np.array):
            'Hungarian Indices'. Indices of the best matched sets.
    """
    assert len(labels_test) == len(labels_true), 'RH ERROR: labels_test and labels_true must be the same length.'
    if ignore_negOne:
        labels_test = np.array(labels_test.copy(), dtype=int)
        labels_true = np.array(labels_true.copy(), dtype=int)
        
        labels_test = labels_test[labels_true > -1].copy()
        labels_true = labels_true[labels_true > -1].copy()

    ## convert labels to boolean
    bool_test = np.stack([labels_test==l for l in np.unique(labels_test)], axis=0).astype(np.float32)
    bool_true = np.stack([labels_true==l for l in np.unique(labels_true)], axis=0).astype(np.float32)

    if bool_test.shape[0] < bool_true.shape[0]:
        bool_test = np.concatenate((bool_test, np.zeros((bool_true.shape[0] - bool_test.shape[0], bool_true.shape[1]))))

    ## compute cross-correlation matrix, and crop to 
    na = bool_true.shape[0]
    cc = np.corrcoef(x=bool_true, y=bool_test)[:na][:,na:]  ## corrcoef returns the concatenated cross-corr matrix (self corr mat along diagonal). The indexing at the end is to extract just the cross-corr mat
    cc[np.isnan(cc)] = 0

    ## find hungarian assignment matching indices
    hi = scipy.optimize.linear_sum_assignment(
        cost_matrix=cc,
        maximize=True,
    )

    ## extract correlation scores of matches
    cc_matched = cc[hi[0], hi[1]]

    ## compute score
    score_weighted_partial = np.sum(cc_matched * bool_true.sum(axis=1)[hi[0]]) / bool_true[hi[0]].sum()
    score_unweighted_partial = np.mean(cc_matched)
    ## compute perfect score
    score_weighted_perfect = np.sum(bool_true.sum(axis=1)[hi[0]] * (cc_matched > thresh_perfect)) / bool_true[hi[0]].sum()
    score_unweighted_perfect = np.mean(cc_matched > thresh_perfect)
    
    ## compute adjusted rand score
    score_rand = sklearn.metrics.adjusted_rand_score(labels_true, labels_test)
    
    ## compute adjusted mutual info score
    score_mutual_info = sklearn.metrics.adjusted_mutual_info_score(labels_true, labels_test) if compute_mutual_info else None

    out = {
        'score_weighted_partial': score_weighted_partial,
        'score_weighted_perfect': score_weighted_perfect,
        'score_unweighted_partial': score_unweighted_partial,
        'score_unweighted_perfect': score_unweighted_perfect,
        'adj_rand_score': score_rand,
        'adj_mutual_info_score': score_mutual_info,
        'ignore_negOne': ignore_negOne,
        'idx_hungarian': hi,
    }
    return out