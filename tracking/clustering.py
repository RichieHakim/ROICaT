import numpy as np
import scipy
import scipy.optimize
import scipy.sparse
import sklearn
import matplotlib.pyplot as plt
import torch

import optuna
import hdbscan

from functools import partial
import time

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
        self.s_sf = s_sf
        self.s_NN_z = s_NN_z
        self.s_SWT_z = s_SWT_z
        self.s_sesh = s_sesh

        self.s_sesh_inv = (self.s_sf != 0).astype(np.bool8)
        self.s_sesh_inv[self.s_sesh.astype(np.bool8)] = False
        self.s_sesh_inv.eliminate_zeros()

        self.s_sesh[range(self.s_sesh.shape[0]), range(self.s_sesh.shape[1])] = 0

        self._verbose = verbose

    def find_optimal_parameters_for_pruning(
        self,
        n_bins=100,
        find_parameters_automatically=True,
        kwargs_findParameters={
            'n_patience': 100,
            'tol_frac': 0.05,
            'max_trials': 350,
            'max_duration': 60*10,
            'verbose': False,
        },
        n_jobs_findParameters=-1,
        # fallback_d_cutoff=0.5,
        # plot_pref=True,
        # kwargs_makeConjunctiveDistanceMatrix={
        #     'power_sf': 0.5,
        #     'power_NN': 1.0,
        #     'power_SWT': 0.1,
        #     'p_norm': -4.0,
        #     'sig_NN_kwargs': {'mu': -1.5, 'b': 1.0},
        #     'sig_SWT_kwargs': {'mu': -2.0, 'b': 1.0}
        # },
    ):
        self.n_bins = self.s_sf.nnz // 10000 if n_bins is None else n_bins
        # smoothing_window = self.n_bins // 100
        # print(f'Pruning similarity graphs with {self.n_bins} bins and smoothing window {smoothing_window}...') if self._verbose else None

        if find_parameters_automatically:
            print('Finding parameters using automated hyperparameter tuning...') if self._verbose else None
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.checker = Convergence_checker(**kwargs_findParameters)
            self.study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(n_startup_trials=10))
            self.study.optimize(self._objectiveFn_distSameMagnitude, n_jobs=n_jobs_findParameters, callbacks=[self.checker.check])

            self.best_params = self.study.best_params.copy()
            [self.best_params.pop(p) for p in [
                'sig_SF_kwargs_mu',
                'sig_SF_kwargs_b',
                'sig_NN_kwargs_mu',
                'sig_NN_kwargs_b',
                'sig_SWT_kwargs_mu',
                'sig_SWT_kwargs_b']]
            self.best_params['sig_SF_kwargs'] = {'mu': self.study.best_params['sig_SF_kwargs_mu'],
                                            'b': self.study.best_params['sig_SF_kwargs_b'],}
            self.best_params['sig_NN_kwargs'] = {'mu': self.study.best_params['sig_NN_kwargs_mu'],
                                            'b': self.study.best_params['sig_NN_kwargs_b'],}
            self.best_params['sig_SWT_kwargs'] = {'mu': self.study.best_params['sig_SWT_kwargs_mu'],
                                             'b': self.study.best_params['sig_SWT_kwargs_b'],}
            self.kwargs_makeConjunctiveDistanceMatrix_best = self.best_params
            print(f'Best value found: {self.study.best_value} with parameters {self.best_params}') if self._verbose else None
        return self.kwargs_makeConjunctiveDistanceMatrix_best

        # print('Making conjunctive distance matrix...') if self._verbose else None
        # d_conj = self.make_conjunctive_distance_matrix(**kwargs_makeConjunctiveDistanceMatrix)

        # print('Finding intermode cutoff...') if self._verbose else None
        # edges = np.linspace(d_conj.min(), d_conj.max(), self.n_bins+1)
        # centers = (edges[1:] + edges[:-1])/2
        # counts, bins = np.histogram(d_conj.data, bins=edges)
        
        # counts_smooth = scipy.signal.savgol_filter(counts[1:], smoothing_window, 1)
        # fp = scipy.signal.find_peaks(-counts_smooth[:-1])
        # if len(fp[0]) == 0:
        #     print('No peaks found. Using user defined cutoff.') if self._verbose else None
        #     self.d_cut = fallback_d_cutoff
        # else:
        #     idx_localMin = fp[0][0]
        #     self.d_cut = centers[idx_localMin+1]
        # # self.d_cut = 0.5
        # print(f'Using intermode cutoff of {self.d_cut:.3f}') if self._verbose else None
        
        # self.idx_d_toKeep = d_conj.copy()
        # # self.idx_d_toKeep.data = self.idx_d_toKeep.data + 1e-9
        # # print(np.where(self.idx_d_toKeep.data < self.d_cut)[0])
        # # return self.idx_d_toKeep, self.d_cut
        # self.idx_d_toKeep.data[np.where(self.idx_d_toKeep.data < self.d_cut)[0].astype(int)] = 99999999999  ## set to large unique number
        # self.idx_d_toKeep = self.idx_d_toKeep == 99999999999
        # self.idx_d_toKeep.eliminate_zeros()

        # if plot_pref:
        #     plt.figure()
        #     plt.stairs(counts, edges, fill=True)
        #     plt.plot(centers[1:], counts_smooth, c='orange')
        #     plt.axvline(self.d_cut, c='r')
        #     plt.scatter(self.d_cut, counts_smooth[idx_localMin], s=100, c='r')
        #     plt.xlabel('distance')
        #     plt.ylabel('count')
        #     plt.ylim([0, counts_smooth[:len(counts_smooth)//2].max()*2])
        #     plt.ylim([0,500000])
        #     plt.title('conjunctive distance histogram')

        # return d_conj

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

        self.s_sf_pruned, self.s_NN_pruned, self.s_SWT_pruned = tuple([prune(s, self.graph_pruned) for s in [self.s_sf, self.s_NN_z, self.s_SWT_z]])

    def fit(self,
        session_bool,
        min_cluster_size=2,
        split_intraSession_clusters=True,
        discard_failed_pruning=True,
        d_conj=None,
        d_cut=None,
        d_step=0.05,
    ):
        """
        Fit the clustering algorithm to the data.
        """
        print('Clustering...') if self._verbose else None
        n_sessions = session_bool.shape[1]
        
        max_dist=(d_conj.max() - d_conj.min()) * 1000

        self.hdbs = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
        #     min_samples=None,
        #     cluster_selection_epsilon=0.0,
            max_cluster_size=n_sessions,
            metric='precomputed',
            alpha=0.999,
        #     p=None,
            algorithm='generic',
        #     leaf_size=100,
        # #     memory=Memory(location=None),
        #     approx_min_span_tree=False,
        #     gen_min_span_tree=False,
        #     core_dist_n_jobs=mp.cpu_count(),
        #     cluster_selection_method='eom',
        #     allow_single_cluster=False,
        #     prediction_data=False,
        #     match_reference_implementation=False,
            max_dist=max_dist
        )

        print('Fitting HDBSCAN...') if self._verbose else None
        self.hdbs.fit(attach_fully_connected_node(
            d_conj, 
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
            print('Splitting up clusters with multiple ROIs per session...') if self._verbose else None
            labels = labels.copy()
            d_cut = self.d_cut if d_cut is None else d_cut

            success = False
            while success == False:
                violations_labels = np.unique(labels)[np.array([(session_bool[labels==u].sum(0)>1).sum().item() for u in np.unique(labels)]) > 0]
                violations_labels = violations_labels[violations_labels > -1]
                # print(violations_labels)
                if len(violations_labels) == 0:
                    success = True
                    break
                
                for l in violations_labels:
                    idx = np.where(labels==l)[0]
                    if d_conj[idx][:,idx].nnz == 0:
                        labels[idx] = -1
                        # print('no neighbors')

                labels_new = self.hdbs.single_linkage_tree_.get_clusters(
                    cut_distance=d_cut,
                    min_cluster_size=2,
                )[:-1]
                
                idx_toUpdate = np.isin(labels, violations_labels)
                labels[idx_toUpdate] = labels_new[idx_toUpdate] + labels.max() + 3
                labels = helpers.squeeze_integers(labels)
                d_cut -= d_step
                
                if d_cut < 0.01:
                    print(f"RH WARNING: Redundant session cluster splitting did not complete fully. Distance walk failed at 'd_cut':{d_cut}.")
                    if discard_failed_pruning:
                        print(f"Setting all clusters with redundant ROIs to label: -1.")
                        labels[idx_toUpdate] = -1
                    break
        
        violations_labels = np.unique(labels)[np.array([(session_bool[labels==u].sum(0)>1).sum().item() for u in np.unique(labels)]) > 0]
        violations_labels = violations_labels[violations_labels > -1]
        self.violations_labels = violations_labels

        ## Set clusters with too few ROIs to -1
        u, c = np.unique(labels, return_counts=True)
        labels[np.isin(labels, u[c<2])] = -1
        labels = helpers.squeeze_integers(labels)

        self.labels = labels
        return self.labels, self.hdbs

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
        if len([s for s in [s_sf, s_NN, s_SWT] if s is not None]) == 0:
            raise Exception("RH ERROR: All similarity matrices are None.")

        p_norm = 1e-9 if p_norm == 0 else p_norm

        sSF_data = self._activation_function(s_sf, sig_SF_kwargs, power_SF)
        sNN_data = self._activation_function(s_NN, sig_NN_kwargs, power_NN)
        sSWT_data = self._activation_function(s_SWT, sig_SWT_kwargs, power_SWT)
        
        sConj_data = self._pNorm(
            s_list=[sSF_data, sNN_data, sSWT_data],
            p=p_norm,
        )

        dConj = s_sf.copy()
        dConj.data = 1 - sConj_data.numpy()

        return dConj, sSF_data, sNN_data, sSWT_data, sConj_data

    def _activation_function(self, s, sig_kwargs={'mu':0.0, 'b':1.0}, power=1):
        if s is None:
            return None
        if (sig_kwargs is not None) and (power is not None):
            return helpers.generalised_logistic_function(torch.as_tensor(s.data, dtype=torch.float32), **sig_kwargs)**power
        elif (sig_kwargs is None) and (power is not None):
            return torch.as_tensor(s.data, dtype=torch.float32)**power
        elif (sig_kwargs is not None) and (power is None):
            return helpers.generalised_logistic_function(torch.as_tensor(s.data, dtype=torch.float32), **sig_kwargs)

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
        kwargs = kwargs_makeConjunctiveDistanceMatrix if kwargs_makeConjunctiveDistanceMatrix is not None else self.best_params
        dConj, sSF_data, sNN_data, sSWT_data, sConj_data = self.make_conjunctive_distance_matrix(
            s_sf=self.s_sf,
            s_NN=self.s_NN_z,
            s_SWT=self.s_SWT_z,
            **kwargs
        )
        dens_same_crop, dens_same, dens_diff, dens_all, edges, d_crossover = self._separate_diffSame_distributions(dConj)
        centers = (edges[1:] + edges[:-1]) / 2

        # d_crossover = centers[np.where(dens_same_crop>0)[0][-1]]
        
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
        edges = torch.linspace(0,1, self.n_bins+1, dtype=torch.float32)

        d_all = d_conj.copy()
        counts, _ = torch.histogram(torch.as_tensor(d_all.data, dtype=torch.float32), edges)
        dens_all = counts / counts.sum()  ## distances of all pairs of ROIs

        d_intra = d_conj.multiply(self.s_sesh_inv)
        d_intra.eliminate_zeros()
        counts, _ = torch.histogram(torch.as_tensor(d_intra.data, dtype=torch.float32), edges)
        dens_diff = counts / counts.sum()  ## distances of known differents

        dens_same = dens_all - dens_diff
        dens_same = torch.maximum(dens_same, torch.as_tensor([0], dtype=torch.float32))

        dens_deriv = dens_diff - dens_same
        dens_deriv[dens_diff.argmax():] = 0
        idx_crossover = torch.where(dens_deriv < 0)[0][-1]
        d_crossover = edges[idx_crossover].item()

        dens_same_crop = dens_same.clone()
        dens_same_crop[idx_crossover:] = 0
        return dens_same_crop, dens_same, dens_diff, dens_all, edges, d_crossover

    def _objectiveFn_distSameMagnitude(self, trial):
        power_SF=trial.suggest_float('power_SF', 0.1, 3, log=False)
        power_NN=trial.suggest_float('power_NN', 0.1, 3, log=False)
        power_SWT=trial.suggest_float('power_SWT', 0.1, 3, log=False)
        p_norm=trial.suggest_float('p_norm', -10, 10, log=False)
        
        sig_SF_kwargs={
            'mu':trial.suggest_float('sig_SF_kwargs_mu', 0.1, 0.5, log=False),
            'b':trial.suggest_float('sig_SF_kwargs_b', 0.1, 2, log=False),
        }
        sig_NN_kwargs={
            'mu':trial.suggest_float('sig_NN_kwargs_mu', 0, 0.5, log=False),
            'b':trial.suggest_float('sig_NN_kwargs_b', 0.01, 2, log=False),
        }
        sig_SWT_kwargs={
            'mu':trial.suggest_float('sig_SWT_kwargs_mu', -0.5, 0.5, log=False),
            'b':trial.suggest_float('sig_SWT_kwargs_b', 0.01, 2, log=False),
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

        loss = dens_same_crop.sum().item()
        
        return loss  # An objective value linked with the Trial object.
        

def attach_fully_connected_node(d, dist_fullyConnectedNode=None):
    """
    This function takes in a sparse graph (csr_matrix) that has more than
     one component (multiple unconnected subgraphs) and appends another 
     node to the graph that is weakly connected to all other nodes.
     
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
    def __init__(
        self, 
        n_patience=10, 
        tol_frac=0.05, 
        max_trials=350, 
        max_duration=60*10, 
        verbose=True
    ):
        self.bests = []
        self.best = -np.inf
        self.n_patience = n_patience
        self.tol_frac = tol_frac
        self.max_trials = max_trials
        self.max_duration = max_duration
        self.num_trial = 0
        self.verbose = verbose
        
    def check(self, study, trial):
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


def score_labels(labels_test, labels_true, ignore_negOne=False, thresh_perfect=0.9999999999):
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
    if ignore_negOne:
        labels_test = labels_test[labels_true > -1].copy()
        labels_true = labels_true[labels_true > -1].copy()

    ## convert labels to boolean
    bool_test = np.stack([labels_test==l for l in np.unique(labels_test)], axis=0)
    bool_true = np.stack([labels_true==l for l in np.unique(labels_true)], axis=0)

    ## compute cross-correlation matrix, and crop to 
    na = bool_true.shape[0]
    cc = np.corrcoef(x=bool_true, y=bool_test)[:na][:,na:]  ## corrcoef returns the concatenated cross-corr matrix (self corr mat along diagonal). The indexing at the end is to extract just the cross-corr mat
    
    ## find hungarian assignment matching indices
    hi = scipy.optimize.linear_sum_assignment(
        cost_matrix=cc,
        maximize=True,
    )

    ## extract correlation scores of matches
    cc_matched = cc[hi[0], hi[1]]

    ## compute score
    score_weighted_partial = np.sum(cc_matched * bool_true.sum(axis=1)[hi[0]]) / bool_true.sum()
    score_unweighted_partial = np.mean(cc_matched)
    ## compute perfect score
    score_weighted_perfect = np.sum(bool_true.sum(axis=1)[hi[0]] * (cc_matched > thresh_perfect)) / bool_true.sum()
    score_unweighted_perfect = np.mean(cc_matched > thresh_perfect)
    
    ## compute adjusted rand score
    score_rand = sklearn.metrics.adjusted_rand_score(labels_true, labels_test)

    ## compute adjusted mutual info score
    score_mutual_info = sklearn.metrics.adjusted_mutual_info_score(labels_true, labels_test)

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