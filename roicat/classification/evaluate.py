import numpy as np
from .. import helpers
    
class Evaluation():
    def __init__(self, classifier):
        self.classifier = classifier
        return
    
    def confusion_matrix(self, x, y, counts=False):
        """
        Generate a confusion matrix for the dataset based on the classifier

        JZ 2022

        logreg: sklearn model with score method
        X_eval: Head from which to classify examples
        y_eval: True labels for examples for evaluation
        counts: Whether to return confusion matrix as counts (False) or percentages (True)
        """
        preds = self.classifier.predict(x).astype(np.int32)
        cm = helpers.confusion_matrix(preds, y.astype(np.int32), counts=counts)
        return cm
        
    def score_classifier_logreg(self, x, y):
        """
        Generate a classification score for dataset based on the classifier
        
        JZ 2022
        
        logreg: sklearn model with score method
        X_eval: Head from which to classify examples
        y_eval: True labels for examples for evaluation
        """
        acc = self.classifier.score(x, y.astype(np.int32), sample_weight=get_balanced_sample_weights(y.astype(np.int32)))
        return acc
    
    # =========================

def get_balanced_sample_weights(labels):
    """
    Balances sample ways for classification
    
    JZ 2022
    
    labels: np.array
        Includes list of labels to balance the weights for classifier training
    returns weights by samples
    """
    labels = np.int64(labels.copy())
    counts, vals = np.histogram(labels, bins=np.concatenate((np.unique(labels), [labels.max()+1])))
    vals = vals[:-1]

    n_labels = len(labels)
    weights = n_labels / counts
    
    sample_weights = np.array([weights[l] for l in labels])
    
    return sample_weights