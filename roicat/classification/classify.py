import numpy as np
import sklearn
from .. import helpers

class Classifier():
    
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.classifier = None
        return
    
    # =========================
    
    def fit_classifier(self, x, y, rank=None, max_iter=10000, C=1):
        pp_x = self.preprocessor.fit_transform_preprocess(x, rank=rank)
        self.classifier = self.logreg_classifier(pp_x, y, max_iter=max_iter, C=C)
    
    def classify(self, x):
        """
        Fit a logistic regression classifier to the training data
        X_eval: Head from which to classify examples
        y_eval: True labels for examples for evaluation
        counts: Whether to return confusion matrix as counts (False) or percentages (True)
        github_loc: Location which cincludes basic_nerual_processing_modules
        """
        pp_x = self.preprocessor.transform_preprocess(x)
#         self.classifier = self.logreg_predict(pp_x)
        proba = self.classifier.predict_proba(pp_x)
        preds = np.argmax(proba, axis=1)
        return proba, preds
    
    def save_classifier(self):
        # TODO: Complete
        return
        
    def load_classifier(self):
        # TODO: Complete
        return
        
    # =========================
    
    def logreg_classifier(self, x, y, **kwargs):
        """
        Fit a logistic regression classifier to the training data
        X_eval: Head from which to classify examples
        y_eval: True labels for examples for evaluation
        counts: Whether to return confusion matrix as counts (False) or percentages (True)
        github_loc: Location which cincludes basic_nerual_processing_modules
        """
        logreg = sklearn.linear_model.LogisticRegression(
                solver='lbfgs',
                fit_intercept=True, 
                class_weight='balanced',
                **kwargs
        )
        logreg.fit(x, y)
        return logreg  
        
    