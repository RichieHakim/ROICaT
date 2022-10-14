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
        """
        Fit a preprocessor and associated logistic regression classifier to the training data
        x: latents from which to classify examples
        y: True labels for examples for evaluation
        rank: PCA rank to use in preprocessing step
        max_iter: maximum number of iterations for logistic regression
        C: regularization parameter for logistic regression
        """
        pp_x = self.preprocessor.fit_transform_preprocess(x, rank=rank)
        self.classifier = self.logreg_classifier(pp_x, y, max_iter=max_iter, C=C)
    
    def classify(self, x):
        """
        Classify dataset x, returning probabilities and class predictions
        x: latents from which to classify examples
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
        x: Latents from which to classify examples
        y: True labels for examples for training
        """
        logreg = sklearn.linear_model.LogisticRegression(
                solver='lbfgs',
                fit_intercept=True, 
                class_weight='balanced',
                **kwargs
        )
        logreg.fit(x, y)
        return logreg  
        
    