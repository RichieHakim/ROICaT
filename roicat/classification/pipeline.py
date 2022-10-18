import numpy as np
import torch
import joblib
from sklearn.pipeline import Pipeline, make_pipeline
from .. import helpers

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
        