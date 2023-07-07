# Imports
import sys
import os
import numpy as np
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import time
import copy
import json
import random
import pandas as pd
import math
import argparse
import pickle
import roicat

class Simclr_Model():
    def __init__(self, hyperparameters=None, filepath_model=None):
        pass
    def save_onnx(self, filepath_model):
        pass
    def load_onnx(self, filepath_model):
        pass

    # Load pretrained weights, freeze all layers

    ### TODO: JZ: Download convnext from online source
    ### Freeze untrained layers

    # Chop model off at layer _, pool output, add linear layer unfrozen, flatten
    # Loop through parameters and freeze/unfreeze relevant layers
    # Model to device, prep_contrast, define forward

class Simclr_Trainer():
    def __init__(self, data, model, hyperparameters):
        pass

    # Save model, optimizer, scheduler, etc. to dir_save
    # Save training loss to dir_save
