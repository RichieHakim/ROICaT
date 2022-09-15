import numpy as np
import scipy
import scipy.optimize
import scipy.sparse
import sklearn
import matplotlib.pyplot as plt

import hdbscan

from functools import partial

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
        verbose=True,
    ):
        self.s_sf = s_sf
        self.s_NN_z = s_NN_z
        self.s_SWT_z = s_SWT_z
        self._verbose = verbose

    def prune_similarity_graphs(
        self,
        fallback_d_cutoff=0.5,
        plot_pref=True,
        kwargs_makeConjunctiveDistanceMatrix={
            'power_sf': 0.5,
            'power_NN': 1.0,
            'power_SWT': 0.1,
            'p_norm': -4.0,
            'sig_NN_kwargs': {'mu': -1.5, 'b': 1.0},
            'sig_SWT_kwargs': {'mu': -2.0, 'b': 1.0}
        },
    ):
        n_bins = self.s_sf.nnz // 10000
        smoothing_window = n_bins // 100
        print(f'Pruning similarity graphs with {n_bins} bins and smoothing window {smoothing_window}...') if self._verbose else None

        d_conj = self.make_conjunctive_distance_matrix(**kwargs_makeConjunctiveDistanceMatrix)
    
        print('Finding intermode cutoff...') if self._verbose else None
        edges = np.linspace(d_conj.min(), d_conj.max(), n_bins+1)
        centers = (edges[1:] + edges[:-1])/2
        counts, bins = np.histogram(d_conj.data, bins=edges)
        
        counts_smooth = scipy.signal.savgol_filter(counts[1:], smoothing_window, 1)
        fp = scipy.signal.find_peaks(-counts_smooth[:-1])
        if len(fp[0]) == 0:
            print('No peaks found. Using user defined cutoff.') if self._verbose else None
            self.d_cut = fallback_d_cutoff
        else:
            idx_localMin = fp[0][0]
            self.d_cut = centers[idx_localMin+1]
        # self.d_cut = 0.5
        print(f'Using intermode cutoff of {self.d_cut:.3f}') if self._verbose else None
        
        self.idx_d_toKeep = d_conj.copy()
        # self.idx_d_toKeep.data = self.idx_d_toKeep.data + 1e-9
        # print(np.where(self.idx_d_toKeep.data < self.d_cut)[0])
        # return self.idx_d_toKeep, self.d_cut
        self.idx_d_toKeep.data[np.where(self.idx_d_toKeep.data < self.d_cut)[0].astype(int)] = 99999999999  ## set to large unique number
        self.idx_d_toKeep = self.idx_d_toKeep == 99999999999
        self.idx_d_toKeep.eliminate_zeros()

        if plot_pref:
            plt.figure()
            plt.stairs(counts, edges, fill=True)
            plt.plot(centers[1:], counts_smooth, c='orange')
            plt.axvline(self.d_cut, c='r')
            plt.scatter(self.d_cut, counts_smooth[idx_localMin], s=100, c='r')
            plt.xlabel('distance')
            plt.ylabel('count')
            plt.ylim([0, counts_smooth[:len(counts_smooth)//2].max()*2])
            plt.ylim([0,500000])
            plt.title('conjunctive distance histogram')

    def fit(self,
        session_bool,
        min_cluster_size=2,
        split_intraSession_clusters=True,
        discard_failed_pruning=True,
        d_conj=None,
        d_cut=None,
        d_step=0.05,
        kwargs_makeConjunctiveDistanceMatrix={
            'power_sf': 0.5,
            'power_NN': 1.0,
            'power_SWT': 0.1,
            'p_norm': -4.0,
            'sig_NN_kwargs': {'mu': -1.5, 'b': 1.0},
            'sig_SWT_kwargs': {'mu': -2.0, 'b': 1.0}
        },
    ):
        """
        Fit the clustering algorithm to the data.
        """
        print('Clustering...') if self._verbose else None
        n_sessions = session_bool.shape[1]

        if d_conj is None:
            d_conj = self.make_conjunctive_distance_matrix(**kwargs_makeConjunctiveDistanceMatrix)
            d_cut = self.d_cut if d_cut is None else d_cut
            d_conj[d_conj > d_cut] = 0
            d_conj.eliminate_zeros()
        
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
        return self.labels

    def make_conjunctive_distance_matrix(
        self,
        power_sf=1,
        power_NN=1,
        power_SWT=1,
        p_norm=1,
        sig_sf_kwargs={'mu':0.5, 'b':0.5},
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
            power_sf (float):
                Power to which to raise the spatial footprint similarity.
            power_NN (float):
                Power to which to raise the neural network similarity.
            power_SWT (float):
                Power to which to raise the scattering wavelet transform 
                 similarity.
            p_norm (float):
                p-norm to use for the conjunction of the similarity
                 matrices.
            sig_sf_kwargs (dict):
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
        print('Making conjunctive distance matrix...') if self._verbose else None

        p_norm = 1e-9 if p_norm == 0 else p_norm

        sSF_data = self._activation_function(self.s_sf, sig_sf_kwargs, power_sf)
        sNN_data = self._activation_function(self.s_NN_z, sig_NN_kwargs, power_NN)
        sSWT_data = self._activation_function(self.s_SWT_z, sig_SWT_kwargs, power_SWT)

        sConj_data = self._pNorm(
            s_list=[sSF_data, sNN_data, sSWT_data],
            p=p_norm,
        )

        dConj = self.s_sf.copy()
        dConj.data = 1 - sConj_data

        return dConj

    def _activation_function(self, s, sig_kwargs={'mu':0.0, 'b':1.0}, power=1):
        sig = partial(helpers.generalised_logistic_function, **sig_kwargs) if sig_kwargs is not None else lambda x: x
        if s is not None:
            return sig(s.data)**power
        else:
            return None

    def _pNorm(self, s_list, p):
        """
        Calculate the p-norm of a list of similarity matrices.
        """
        s_list_noNones = [s for s in s_list if s is not None]
        return (np.mean(np.stack(s_list_noNones, axis=0)**p, axis=0))**(1/p)

    def plot_sigmoids(self):
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16,4))
        axs[0].plot(np.linspace(-0,2,1001), self.sig_sf(np.linspace(-5,5,1001)))
        axs[0].set_title('sigmoid SF')
        axs[1].plot(np.linspace(-1,1,1001), self.sig_NN(np.linspace(-5,5,1001)))
        axs[1].set_title('sigmoid NN')
        axs[2].plot(np.linspace(-1,1,1001), self.sig_SWT(np.linspace(-5,5,1001)))
        axs[2].set_title('sigmoid SWT')


    def plot_similarity_relationships(self, plots_to_show=[1,2,3], max_samples=1000000, kwargs_scatter={'s':1, 'alpha':0.1}):
        ## subsample similarities for plotting
        idx_rand = np.floor(np.random.rand(min(max_samples, len(self.ssf))) * len(self.ssf)).astype(int)
        ssf_sub = self.ssf[idx_rand]
        snn_sub = self.snn[idx_rand]
        sswt_sub = self.sswt[idx_rand] if self.sswt is not None else None
        d_conj_sub = self.d_conj.data[idx_rand]

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20,4))
        ## set figure title
        fig.suptitle('Similarity relationships', fontsize=16)
        
        ## plot similarity relationships
        if 1 in plots_to_show:
            axs[0].scatter(ssf_sub, snn_sub, c=d_conj_sub, **kwargs_scatter)
            axs[0].set_xlabel('sim Spatial Footprint')
            axs[0].set_ylabel('sim Neural Network')
        if self.sswt is not None:
            if 2 in plots_to_show:
                axs[1].scatter(ssf_sub, sswt_sub, c=d_conj_sub, **kwargs_scatter)
                axs[1].set_xlabel('sim Spatial Footprint')
                axs[1].set_ylabel('sim Scattering Wavelet Transform')
            if 3 in plots_to_show:
                axs[2].scatter(snn_sub, sswt_sub, c=d_conj_sub, **kwargs_scatter)
                axs[2].set_xlabel('sim Neural Network')
                axs[2].set_ylabel('sim Scattering Wavelet Transform')
        
        return fig, axs



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