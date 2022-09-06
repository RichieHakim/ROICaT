import numpy as np
import scipy
import scipy.optimize


def score_labels(labels_test, labels_true, thresh_perfect=0.9999999999):
    """
    Compute the score of the clustering.
    The best match is found by solving the linear sum assignment problem.
    The score is bounded between 0 and 1.
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
        score_partial (float):
            Score of the clustering. Average correlation between the 
            best matched sets of true and test labels.
        score_perfect (float):
            Fraction of 'perfectly' matched label sets.
        hi (np.array):
            'Hungarian Indices'. Indices of the best matched sets.
    """
    ## convert labels to boolean
    bool_test = np.stack([labels_test==l for l in np.unique(labels_test)], axis=0)
    bool_true = np.stack([labels_true==l for l in np.unique(labels_true)], axis=0)

    ## compute cross-correlation matrix, and crop to 
    na = bool_true.shape[0]
#     nb = bool_test.shape[0]
    cc = np.corrcoef(x=bool_true, y=bool_test)[:na][:,na:]  ## corrcoef returns the concatenated cross-corr matrix (self corr mat along diagonal). The indexing at the end is to extract just the cross-corr mat
    
    ## find hungarian assignment matching indices
    hi = hungarian_idx = scipy.optimize.linear_sum_assignment(
        cost_matrix=cc,
        maximize=True,
    )

    ## extract correlation scores of matches
    cc_matched = cc[hi[0], hi[1]]
    
    score_partial = (cc_matched).mean()  ## average the correlation of best matches to true labels
    score_perfect = (cc_matched > thresh_perfect).mean()  ## get fraction of 'perfectly' matched labels
    
    return score_partial, score_perfect, hi