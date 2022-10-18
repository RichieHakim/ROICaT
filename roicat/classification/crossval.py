import numpy as np
import torch
import joblib
from tqdm.notebook import tqdm
from sklearn.linear_model import LogisticRegression
from . import pipeline, evaluate
from .. import helpers

class CrossValidation():
    
    def __init__(self, splitter):
        self.splitter = splitter
        self.split_inx = None
    
    def split_to_inx(self, *args, **kwargs):
        self.split_inx = list(self.splitter.split(*args, **kwargs))
    
    def inx_to_data(self, x, y):
        self.split_x_tr = [x[_[0]] for _ in self.split_inx]
        self.split_x_val = [x[_[1]] for _ in self.split_inx]
        self.split_y_tr = [y[_[0]] for _ in self.split_inx]
        self.split_y_val = [y[_[1]] for _ in self.split_inx]
    
#     def get_inx_data(self, inx):
#         return self.split_x_tr[inx], self.split_x_val[inx], self.split_y_tr[inx], self.split_y_val[inx]
    
    def get_inx_data(self):
        return list(zip(self.split_x_tr, self.split_y_tr, self.split_x_val, self.split_y_val))
    
    
    
def split_loop(cv, preproc, classifier_kwargs, preproc_refit=True, c_lst=[100, 10, 1, 0.1, 0.01, 0.001], n_trial_fit=[1e1,1e2,1e3,1e4,1e5]):
    results_dct = {}
    
    for ic, c in enumerate(c_lst):
        cm_train, cm_val, acc_train, acc_val = [], [], [], []

        for tmp_trainp_X, tmp_trainp_y, tmp_val_X, tmp_val_y in tqdm(cv.get_inx_data()):
            
            if preproc_refit:
                preproc.fit(tmp_trainp_X, tmp_trainp_y)
            
            classify = LogisticRegression(**classifier_kwargs, C=c)
            classify.fit(preproc.transform(tmp_trainp_X), tmp_trainp_y)
            pipe = pipeline.Pipe(preproc, classify)
            
            evaluator = evaluate.Evaluation(pipe.pipeline)
            cm_train.append(evaluator.confusion_matrix(tmp_trainp_X, tmp_trainp_y))
            cm_val.append(evaluator.confusion_matrix(tmp_val_X, tmp_val_y))

            acc_train.append(evaluator.score_classifier_logreg(tmp_trainp_X, tmp_trainp_y))
            acc_val.append(evaluator.score_classifier_logreg(tmp_val_X, tmp_val_y))

        results_dct[f'cm_{c}'] = (np.round(np.mean(cm_train,axis=0),2), np.round(np.mean(cm_val,axis=0),2), np.round(np.mean(acc_train, axis=0),2), np.round(np.mean(acc_val, axis=0),2))
    
    return results_dct