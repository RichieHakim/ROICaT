import numpy as np
import torch
import joblib
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit
StratifiedShuffleSplit
from .. import helpers
from copy import deepcopy

class Pipe():
    
    def __init__(self, *pipeline, pipeline_fileloc=None):
        """
        Create a Pipe from a Pipeline object or a pickled Pipeline object
        """
        if pipeline and (type(pipeline) is list or type(pipeline) is tuple):
            self.pipeline = make_pipeline(*pipeline)
        elif pipeline:
            self.pipeline = pipeline
        else:
            self.pipeline = self.load(pipeline_fileloc)
    
    def save(self, pipeline_fileloc):
        """
        Save the Pipe as a pickled file
        """
        joblib.dump(self.pipeline, pipeline_fileloc)
        
    def load(self, pipeline_fileloc):
        """
        Load a Pipe from a pickled file
        """
        return joblib.load(pipeline_fileloc)


def fit_pipe(feat_train, labels_train, preproc_init, classify_init, preproc_refit=True):
    
    preproc = deepcopy(preproc_init)
    if preproc_refit:
        preproc.fit(feat_train, labels_train)

    classify = deepcopy(classify_init)
    classify.fit(preproc.transform(feat_train), labels_train)
    pipe = Pipe(preproc, classify)
    
    return pipe
    
def fit_n_train(features_train, labels_train, preproc_init, classify_init, preproc_refit=True, n_train=1e1):
    
    if n_train < features_train.shape[0]:
        train_size = n_train/features_train.shape[0]
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size)
        train_subset_inx, _ = list(sss.split(features_train, labels_train))[0]
    else:
        n_train = features_train.shape[0]
        train_subset_inx = list(range(n_train))
    
#     print(train_size, sss, len(train_subset_inx), train_subset_inx)
    
    features_train_subset, labels_train_subset = features_train[train_subset_inx], labels_train[train_subset_inx]
    pipe = fit_pipe(features_train_subset, labels_train_subset, preproc_init, classify_init, preproc_refit=preproc_refit)
    
    return pipe, n_train
    