import json
from bnpm import file_helpers
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
import copy
import numpy as np
import torch
from roicat.model_training import augmentation
from torch import nn
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm, trange
from roicat import ROInet, data_importing
import warnings
import h5py

activation_lookup = {
               'relu': nn.ReLU,
               'elu': nn.ELU,
               'gelu': nn.GELU,
               'selu': nn.SELU,
               'sigmoid': nn.Sigmoid,
    }

class Datasplit():
    """
    Split data into training and validation sets with additional helper methods
    JZ 2023
    """
    def __init__(self, features, labels, n_train=None, test_size=None):
        """
        Split data into training and validation sets
        Args:
            features (np.array):
                Features to be split (e.g. images or latents)
            labels (np.array):
                Labels to be split
            n_train (int):
                Number of stratified downsampled training samples that should be used for training (if more than features, all training samples are used)
            test_size (float):
                Fraction of data to be used for validation
        """
        self.features = features
        self.labels = labels
        self.n_train = n_train
        self.test_size = test_size
    
        self.idx_train, self.idx_val = self.stratified_sample(self.features, self.labels, n_splits=1, test_size=self.test_size)
        
        self.features_train = self.features[self.idx_train]
        self.labels_train = self.labels[self.idx_train]
        
        self.features_val = self.features[self.idx_val]
        self.labels_val = self.labels[self.idx_val]

        self.idx_train_subset, _ = self.downsample_train(self.features_train, self.labels_train, len(self.idx_train))
        self.features_train_subset = self.features_train[self.idx_train_subset]
        self.labels_train_subset = self.labels_train[self.idx_train_subset]

        self.n_train_actual = len(self.idx_train_subset)
    
    def downsample_train(self, features, labels, n_train_actual_prv):
        """
        Downsample training data for forced reduced number of training samples via stratified sampling
        Args:
            features (np.array):
                Features to downsample for prediction (e.g. images or latents)
            labels (np.array):
                Labels to downsample
            idx_train (np.array):
                Indices of training data
        """
        train_size = self.n_train/n_train_actual_prv if self.n_train < n_train_actual_prv else 1.0
        if train_size < 1.0:
            return self.stratified_sample(features, labels, n_splits=1, train_size=train_size)
        else:
            idx_train_downsampled = np.arange(len(features))
            np.random.shuffle(idx_train_downsampled)
            return (idx_train_downsampled, np.array([]))
    
    def stratified_sample(self, features, labels, n_splits=1, train_size=None, test_size=None):
        """
        Stratified sampling of data
        Args:
            features (np.array):
                Features for prediction (e.g. images or latents)
            labels (np.array):
                Labels
            n_splits (int):
                Number of splits
            train_size (float):
                Fraction of data to use for training
            test_size (float):
                Fraction of data to use for validation
        Returns:
            idx_train (np.array):
                Indices of training data
            idx_val (np.array):
                Indices of validation data
        """
        assert (train_size is not None or test_size is not None) and not (train_size is None and test_size is None), "JZ Error: Exactly one of train_size and test_size should be specified"
        
        if train_size is not None:
            sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size)
        else:
            sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
        return list(sss.split(features, labels))[0]


class TrainingTracker():
    """
    Helper class to track training progress and performance
    JZ 2023
    """
    def __init__(self, directory_save, convergence_checker=None, class_weights=None, tictoc={}, n_train_actual=None, model=None):
        """
        Track training progress
        Args:
            directory_save (str):
                Directory to save the training tracker results
            convergence_checker (ConvergenceChecker):
                Convergence checker object to use to determine if training should stop
            class_weights (np.array):
                Class weights used for accuracy and loss calculations
            tictoc (dict):
                Dictionary to store timing information
        """
        self.directory_save = directory_save
        self.filename_results = str((Path(directory_save) / 'results_training.csv').resolve())
        self.filename_timing = str((Path(directory_save) / 'results_timing.json').resolve())
        self.filename_params_realtime = str((Path(directory_save) / 'params_realtime.json').resolve())
        self.filename_model = str((Path(directory_save) / 'model.npy').resolve())

        self.params_realtime = dict(n_train_actual=n_train_actual)
        self.data = dict()
        self.convergence_checker = convergence_checker
        self.class_weights = class_weights
        self.tictoc = tictoc
        self.model = model

    def append(self, epoch, key, value):
        """
        Add value to dictionary associated with key's associated epoch value
        Args:
            epoch (int):
                Current training epoch
            key (str):
                Identifier for the training tracker value
            value (any):
                Value to add to training tracker associated with the given key & epoch
        """
        self.data[key] = self.data.get(key, {})
        self.data[key][epoch] = value
    
    def add_confusion_matrix(self, epoch, confusion_matrix_id, labels, predictions):
        """
        Calculate and add a confusion matrix to training tracker
        Args:
            epoch (int):
                Current training epoch
            confusion_matrix_id (str):
                Identifier for the confusion matrix to be calculated
            labels (np.array):
                Labels to use in the calculation of the confusion matrix
            predictions (np.array):
                Predictions to use in the calculation of the confusion matrix
        """
        self.append(epoch, confusion_matrix_id, confusion_matrix(labels, predictions, normalize='true').T)

    def add_loss(self, epoch, loss_id, loss_value):
        """
        Add loss to training tracker
        Args:
            epoch (int):
                Current training epoch
            loss_id (str):
                Identifier for the loss to be added
            loss_value (float):
                Loss value to add to training tracker associated with the given loss_id & epoch
        """
        self.append(epoch, loss_id, loss_value)

    def add_accuracy(self, epoch, accuracy_id, labels, predictions):
        """
        Calculate and add accuracy to training tracker
        Args:
            epoch (int):
                Current training epoch
            accuracy_id (str):
                Identifier for the accuracy to be calculated
            labels (np.array):
                Labels to use in the calculation of the accuracy
            predictions (np.array):
                Predictions to use in the calculation of the accuracy
        """
        self.append(epoch, accuracy_id, accuracy_score(labels,
                                                       predictions,
                                                       sample_weight=self.get_balanced_sample_weights(labels, class_weights=self.class_weights)))

    def check_convergence(self, key):
        """
        Check if training should stop
        Args:
            key (str):
                Identifier for the training tracker value to be used to check convergence
        """
        return self.convergence_checker([v for k, v in self.data[key].items()])

    def save_results(self):
        """
        Save training tracker results to file(s).
        Note: Results DataFrame is saved to a CSV file based on self.filename_results and timing information is saved to a JSON file based on self.filename_timing
        """
        print('Saving results: ', self.filename_results, self.filename_timing)
        df = pd.DataFrame(self.data)
        df.to_csv(self.filename_results)

        if self.tictoc:
            with open(self.filename_timing, 'w') as f:
                json.dump(self.tictoc, f)
        
        if self.params_realtime:
            with open(self.filename_params_realtime, 'w') as f:
                json.dump(self.params_realtime, f)
        
        if type(self.model) != type(None):
            np.save(self.filename_model, self.model, allow_pickle=True)

    def print_results(self):
        """
        Print training tracker results to console
        """
        print(f'{self.tictoc=}')
        print(f'{self.model=}')
        print(pd.DataFrame(self.data))

    def get_balanced_class_weights(self, labels):
        """
        Balances sample ways for classification
        
        RH/JZ 2022
        
        labels: np.array
            Includes list of labels to balance the weights for classifier training
        returns weights by samples
        """
        labels = labels.astype(np.int64)
        values, counts = np.unique(labels, return_counts=True)
        weights = len(labels) / counts
        return weights

    def get_balanced_sample_weights(self, labels, class_weights=None):
        """
        Balances sample ways for classification
        
        RH/JZ 2022
        
        labels: np.array
            Includes list of labels to balance the weights for classifier training
        class_weights: np.array
            If provided, uses these weights instead of calculating them from the labels
        returns weights by samples
        """        
        if type(class_weights) is not np.ndarray and type(class_weights) is not np.array:
            print('Warning: Class weights not pre-fit. Using provided sample labels.')
            weights = self.get_balanced_class_weights(labels)
        else:
            weights = class_weights
        sample_weights = weights[labels]
        return sample_weights


    def __getitem__(self, key):
        """
        Get value associated with key
        Args:
            key (str):
                Identifier for the training tracker value
        Returns: value (any): Value associated with the given key
        """
        return self.data[key]
    
    def __setitem__(self, key, value):
        """
        Set value associated with key
        Args:
            key (str):
                Identifier for the training tracker value
            value (any):
                Value to add to training tracker associated with the given key
        """
        self.data[key] = value


def remap_labels(labels, label_remapping={4:2}):
    """
    Remap labels
    JZ 2023
    Args:
        labels (np.array):
            Labels to remap
        label_remapping (dict):
            Dictionary of label remapping (key: old label, value: new label)
    Returns:
        labels (np.array):
            Labels with remapping applied
    """
    labels = copy.deepcopy(labels)
    for k in label_remapping:
        labels[labels==int(k)] = label_remapping[k]

    return labels

def extract_with_dataloader(dataloader, model=None, num_copies=1, device='cpu'):
    if model is None:
        model = lambda x: x
    data_extracted = [(model(x[0].to(device)), y.to(device), idx, sample_weight) for _ in trange(num_copies) for x, y, idx, sample_weight in dataloader]
    X, y, idx, sample_weight = (
        torch.cat([copy_iteration[0] for copy_iteration in data_extracted], dim=0).to('cpu'),
        torch.cat([copy_iteration[1] for copy_iteration in data_extracted], dim=0).to('cpu'),
        torch.cat([copy_iteration[2] for copy_iteration in data_extracted], dim=0).to('cpu'),
        torch.cat([copy_iteration[3] for copy_iteration in data_extracted], dim=0).to('cpu')
    )
    return X, y, idx, sample_weight
    
def get_transforms(augmentations, scripted=True):
    """
    Create transforms from parameters to use in datasets and dataloaders for the training and validation sets
    """
    transforms = torch.nn.Sequential(*[augmentation.__dict__[key](**val) for key,val in augmentations.items()])
    transforms_final = torch.jit.script(transforms) if scripted else transforms
    return transforms_final

def get_activation(activation):
    return activation_lookup[activation]

def merge_dict_to_hdf5(data_dict, group=None):
    """
    Merge a dictionary into an existing HDF5 file identified by an h5py.File object.

    Args:
        data_dict (dict): The dictionary containing the data to be merged.
        group (h5py.Group, optional): The HDF5 group to which the data should be merged. Defaults to None.
    """
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # If the value is a dictionary, check if the group already exists, and either skip it or merge the data
            if key in group:
                merge_dict_to_hdf5(value, group[key])
            else:
                subgroup = group.create_group(key)
                merge_dict_to_hdf5(value, subgroup)
        else:
            # If the value is not a dictionary, convert it to a numpy array and create a dataset
            if key in group:
                del group[key]
            group.create_dataset(key, data=value)

def merge_dict_to_hdf5_file(file_path, data_dict):
    """
    Merge a dictionary into an existing HDF5 file identified by a file path.

    Args:
        file_path (str): The file path of the HDF5 file.
        data_dict (dict): The dictionary containing the data to be merged.
    """
    with h5py.File(file_path, 'a') as file:
        merge_dict_to_hdf5(data_dict, file)