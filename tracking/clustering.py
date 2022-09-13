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
    def __init__(self):
        pass

    def make_conjunctive_distance_matrix(
        self,
        s_sf,
        s_NN_z,
        s_SWT_z,
        power_sf=1,
        power_NN=1,
        power_SWT=1,
        p_norm=1,
        sig_NN_kwargs={'mu':0.5, 'b':0.5},
        sig_SWT_kwargs={'mu':0.5, 'b':0.5},
        plot_sigmoid=True,
    ):
        p_norm = 1e-9 if p_norm == 0 else p_norm
        self.ssf = s_sf.data**power_sf

        sig_NN = partial(helpers.generalised_logistic_function, **sig_NN_kwargs)
        sig_SWT = partial(helpers.generalised_logistic_function, **sig_SWT_kwargs)
        if plot_sigmoid:
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
            axs[0].plot(np.linspace(-5,5,1001), sig_NN(np.linspace(-5,5,1001)))
            axs[0].set_title('sigmoid NN')
            axs[1].plot(np.linspace(-5,5,1001), sig_SWT(np.linspace(-5,5,1001)))
            axs[1].set_title('sigmoid SWT')

        self.snn = np.maximum(sig_NN(s_NN_z.data), 0)**power_NN
        self.sswt = np.maximum(sig_SWT(s_SWT_z.data), 0)**power_SWT

        s_data_norm = (np.mean(np.stack((self.snn, self.ssf, self.sswt), axis=0)**p_norm, axis=0))**(1/p_norm)

        self.d_conj = s_sf.copy()
        self.d_conj.data = 1 - s_data_norm

        self.d_conj = self.d_conj.tolil()
        # self.d_conj[range(self.d_conj.shape[0]), range(self.d_conj.shape[1])] = 1e-9
        self.d_conj = self.d_conj.tocsr()
        self.d_conj.eliminate_zeros()

    def plot_similarity_relationships(self,):
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20,4))
        ## set figure title
        fig.suptitle('Similarity relationships', fontsize=16)
        
        ## plot similarity relationships
        axs[0].scatter(self.ssf, self.snn, s=5, alpha=0.05, c=self.d_conj.data)
        axs[0].set_xlabel('sim Spatial Footprint')
        axs[0].set_ylabel('sim Neural Network')
        axs[1].scatter(self.ssf, self.sswt, s=5, alpha=0.05, c=self.d_conj.data)
        axs[1].set_xlabel('sim Spatial Footprint')
        axs[1].set_ylabel('sim Scattering Wavelet Transform')
        axs[2].scatter(self.snn, self.sswt, s=5, alpha=0.05, c=self.d_conj.data)
        axs[2].set_xlabel('sim Neural Network')
        axs[2].set_ylabel('sim Scattering Wavelet Transform')
        
        return fig, axs

    def find_intermode_cutoff(self,
        n_bins=100, 
        smoothing_window=15,
        plot_pref=True
    ):

        edges = np.linspace(self.d_conj.min(), self.d_conj.max(), n_bins+1)
        centers = (edges[1:] + edges[:-1])/2
        counts, bins = np.histogram(self.d_conj.data, bins=edges)

        counts_smooth = scipy.signal.savgol_filter(counts[1:], smoothing_window, 1)
        idx_localMin = scipy.signal.find_peaks(-counts_smooth[:-1])[0][0]
        self.d_localMin = centers[idx_localMin+1]
        
        self.d_conj_cutoff = self.d_conj.copy()
        # self.d_conj_cutoff.data = self.d_conj_cutoff.data + 1e-9
        self.d_conj_cutoff.data[self.d_conj_cutoff.data > self.d_localMin] = 0
        self.d_conj_cutoff.eliminate_zeros()

        if plot_pref:
            plt.figure()
            plt.stairs(counts, edges, fill=True)
            plt.plot(centers[1:], counts_smooth, c='orange')
            plt.axvline(self.d_localMin, c='r')
            plt.scatter(self.d_localMin, counts_smooth[idx_localMin], s=100, c='r')
            plt.xlabel('distance')
            plt.ylabel('count')
            plt.ylim([0, counts_smooth[:len(counts_smooth)//2].max()*2])
            plt.title('conjunctive distance histogram')
        
    def fit(self,
        session_bool,
        min_cluster_size=2,
        d_conj=None,
        d_localMin=None,
    ):
        """
        Fit the clustering algorithm to the data.
        """

        n_sessions = session_bool.shape[1]

        if d_conj is None:
            d_conj = self.d_conj_cutoff.tocsr()
        if d_localMin is None:
            d_localMin = self.d_localMin
        
        max_dist=(d_conj.max() - d_conj.min()) * 1000

        hdbs = hdbscan.HDBSCAN(
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

        hdbs.fit(attach_fully_connected_node(
            d_conj, 
            dist_fullyConnectedNode=max_dist
        ))
        labels_pre = hdbs.labels_[:-1]
        self.labels = labels_pre

        ## Split up labels with multiple ROIs per session
        ## The below code is a bit of a mess, but it works.
        ##  It works by iteratively reducing the cutoff distance
        ##  and splitting up violating clusters until there are 
        ##  no more violations.
        labels = labels_pre.copy()
        dChange = 0.01
        d_cut = self.d_localMin

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

            labels_new = hdbs.single_linkage_tree_.get_clusters(
                cut_distance=d_cut,
                min_cluster_size=2,
            )[:-1]
            
            idx_toUpdate = np.isin(labels, violations_labels)
            labels[idx_toUpdate] = labels_new[idx_toUpdate] + labels.max() + 3
            labels = helpers.squeeze_integers(labels)
            d_cut -= d_cut*dChange
            
            if d_cut < 0.01:
                # print('failed')
                # break
                raise ValueError('RH ERROR: the search did not complete')

        ## Set clusters with too few ROIs to -1
        u, c = np.unique(labels, return_counts=True)
        labels[np.isin(labels, u[c<2])] = -1
        labels = helpers.squeeze_integers(labels)

        self.labels = labels
        return self.labels


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