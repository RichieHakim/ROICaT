import numpy as np
import torch
from .. import helpers

class Preprocessor():
    
    def __init__(self, use_pca=True, use_zscore=True):
        """
        Instantiate a copy of Preprocessor with
        specifications to use or not use PCA/zscoring
        """
        self.comp_nn = None
        self.SVs = None
        self.EVR_nn = None
        self.use_pca = use_pca
        self.use_zscore = use_zscore
    
    # =========================
    
    def fit_transform_preprocess(self, x, rank=None):
        """
        Fit PCA (if use_pca is True) and apply PCA (of rank, rank) followed by z-scoring
        to a specified dataset (x).
        """
        scores = self.fit_transform_pca(x, rank=rank) if self.use_pca else x
        zscores = self.transform_zscore(scores) if self.use_zscore else scores
        return zscores
        
    def transform_preprocess(self, x, rank=None):
        """
        Apply a pre-fitted PCA (if use_pca is True) to a specified dataset (x) followed
        by z-scoring.
        """
        scores = self.transform_pca(x, rank=rank) if self.use_pca else x
        zscores = self.transform_zscore(scores) if self.use_zscore else scores
        return zscores
    
    def save_preprocess(self, filename):
        
        return
        
    def load_preprocess(self, filename):
        
        return
    
    
    
    
    # =========================
    
    def fit_transform_pca(self, x, rank=None):
        """
        Fit PCA to the specified data, x, save the relevant values outputted,
        and return the PCA truncated at the specified rank.
        """
        # PCA
        self.comp_nn, scores_nn, self.SVs, self.EVR_nn = helpers.torch_pca(x)
#         print(self.comp_nn)
#         print(scores_nn)
        if rank is None:
            rank = self.comp_nn.shape[1]
        return scores_nn[:, :rank]
    
    def transform_pca(self, x, rank=None):
        """
        Apply a prefitted PCA to the specified data, x, save the relevant values outputted,
        and return the PCA truncated at the specified rank.
        """
        assert type(self.comp_nn) is np.array or type(self.comp_nn) is torch.Tensor, "Error: PCA must be fit (via fit_transform_pca) before it can be used."
        if rank is None:
            rank = self.comp_nn.shape[1]
        scores_nn = torch.matmul(x, self.comp_nn[:, :rank])
#         print(self.comp_nn)
#         print(scores_nn)
        return scores_nn
    
    # =========================
    
    def transform_zscore(self, x):
        """
        Apply z-scoring to the dataset, x along axis 0.
        """
#         zsc = scipy.stats.zscore(x, axis=0)
        zsc = (x-x.mean(axis=0, keepdims=True))/x.std(axis=0, keepdims=True)
        zsc = zsc[:, ~torch.isnan(zsc[0,:])]
        zsc = torch.as_tensor(zsc, dtype=torch.float32)
        
#         zsc = torch.cat([_ / torch.std(_, dim=0).mean() for _ in [self.scores_nn]], dim=1)
        
        return zsc

    