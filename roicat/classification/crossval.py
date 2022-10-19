import numpy as np
import torch
import joblib
from tqdm.notebook import tqdm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from . import pipeline, evaluate
from .. import helpers
from copy import deepcopy

class CrossValidation():
    """
    Class for loading and splitting data for the purposes of Cross Validation
    JZ 2022
    """
    def __init__(self, splitter):
        """
        Construct the CrossValidation object
        Args:
            splitter (sklearn object inheriting from BaseShuffleSplit):
             Splitter object to use to split data for validation
        """
        self.splitter = splitter
        self.split_inx = None
    
    def split_to_inx(self, *args, **kwargs):
        """
        Generate list of split indices to be used in validation splits
        Args:
            args: Positional arguments to feed into self.splitter.split
            kwargs: Keword arguments to feed into self.splitter.split
        """
        self.split_inx = list(self.splitter.split(*args, **kwargs))
    
    def inx_to_data(self, x, y):
        """
        Generate lists of x/y, training/validation data for validation loop
        Args:
            x (np.array): Predictive features (examples along axis=0)
            y (np.array): Response labels (examples along axis=0)
        """
        self.split_x_tr = [x[_[0]] for _ in self.split_inx]
        self.split_x_val = [x[_[1]] for _ in self.split_inx]
        self.split_y_tr = [y[_[0]] for _ in self.split_inx]
        self.split_y_val = [y[_[1]] for _ in self.split_inx]
    
#     def get_inx_data(self, inx):
#         return self.split_x_tr[inx], self.split_x_val[inx], self.split_y_tr[inx], self.split_y_val[inx]
    
    def get_inx_data(self):
        """
        Return a list of zipped validation splits for x/y, training/validation data
        for validation iteration
        
        JZ 2022
        """
        return list(zip(self.split_x_tr, self.split_y_tr, self.split_x_val, self.split_y_val))
    

def split_loop_c(cv, preproc_init, classifier_kwargs, preproc_refit=True, c_lst=[1, 0.1, 0.01]):
    """
    Return a list of zipped validation splits for x/y, training/validation data
    for validation iteration
    
    JZ 2022
    
    Args:
        cv (CrossValidation object):
         Object for generating validation iterator
        preproc_init (sklearn Transform object):
         Initial preprocessor Pipeline pre-fitting
        classifier_kwargs (dict):
         Keword arguments to be used in constructing the LogisticRegression classifier
        preproc_refit (bool):
         Whether or not to refit the preprocessing pipeline on every iteration of the validation loop
        c_lst (list):
         Whether or not to refit the preprocessing pipeline on every iteration of the validation loop
    
    Returns:
        Dictionary of confustion matrices & accuracies from the validation loop
    """
    results_dct = {}
    for ic, c in enumerate(c_lst):
        cm_train, cm_val, acc_train, acc_val = [], [], [], []

        for tmp_trainp_X, tmp_trainp_y, tmp_val_X, tmp_val_y in tqdm(cv.get_inx_data()):
            
            classify_init = LogisticRegression(**classifier_kwargs, C=c)
            pipe = pipeline.fit_pipe(tmp_trainp_X, tmp_trainp_y, preproc_init, classify_init, preproc_refit=preproc_refit)
            
            evaluator = evaluate.Evaluation(pipe.pipeline)
            cm_train.append(evaluator.confusion_matrix(tmp_trainp_X, tmp_trainp_y))
            cm_val.append(evaluator.confusion_matrix(tmp_val_X, tmp_val_y))

            acc_train.append(evaluator.score_classifier_logreg(tmp_trainp_X, tmp_trainp_y))
            acc_val.append(evaluator.score_classifier_logreg(tmp_val_X, tmp_val_y))
        
        results_dct[f'cm_{c}'] = (np.round(np.mean(cm_train,axis=0),2), np.round(np.mean(cm_val,axis=0),2), np.round(np.mean(acc_train, axis=0),2), np.round(np.mean(acc_val, axis=0),2))
    
    return results_dct
    

def view_cv_dict(cv_dct):
    """
    Plot all confusion matrices stored in the cv_dct
    
    JZ 2022
    
    Args:
        cv_dct (dict):
         Dictionary of tuples associated with c-value classifier performance metrics
         consisting of averaged (train CM, val CM, train acc, val acc)
    """
    for k in cv_dct:
        c = k.replace('cm_','')
        fig, ax = plt.subplots(1, 2, figsize=(10,3))
        fig.suptitle(f'C: {c}')
        sns.heatmap(cv_dct[k][0], annot=True, annot_kws={"size": 16}, vmax=1., cmap=plt.get_cmap('gray'), ax=ax[0])
        sns.heatmap(cv_dct[k][1], annot=True, annot_kws={"size": 16}, vmax=1., cmap=plt.get_cmap('gray'), ax=ax[1])
        ax[0].set_title(f'Train — Acc: {cv_dct[k][2]}')
        ax[1].set_title(f'Val — Acc: {cv_dct[k][3]}')
