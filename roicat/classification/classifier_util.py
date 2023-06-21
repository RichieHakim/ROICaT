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
import roicat
from roicat import ROInet, data_importing
import warnings
import h5py
import umap

activation_lookup = {
               'relu': nn.ReLU,
               'elu': nn.ELU,
               'gelu': nn.GELU,
               'selu': nn.SELU,
               'sigmoid': nn.Sigmoid,
    }

class UMAP_Labeler(umap.UMAP):
    def __init__(self,
            n_neighbors=30,
            n_components=2,
            metric='euclidean',
            metric_kwds=None,
            output_metric='euclidean',
            output_metric_kwds=None,
            n_epochs=None,
            learning_rate=1.0,
            init='spectral',
            min_dist=0.1,
            spread=1.0,
            low_memory=True,
            n_jobs=-1,
            set_op_mix_ratio=1.0,
            local_connectivity=1.0,
            repulsion_strength=1.0,
            negative_sample_rate=5,
            transform_queue_size=4.0,
            a=None,
            b=None,
            random_state=None,
            angular_rp_forest=False,
            target_n_neighbors=-1,
            target_metric='categorical',
            target_metric_kwds=None,
            target_weight=0.5,
            transform_seed=42,
            transform_mode='embedding',
            force_approximation_algorithm=False,
            verbose=False,
            tqdm_kwds=None,
            unique=False,
            densmap=False,
            dens_lambda=2.0,
            dens_frac=0.3,
            dens_var_shift=0.1,
            output_dens=False,
            disconnection_distance=None,
            precomputed_knn=(None, None, None),
    ):
        self.umap_params = dict(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=metric,
            metric_kwds=metric_kwds,
            output_metric=output_metric,
            output_metric_kwds=output_metric_kwds,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            init=init,
            min_dist=min_dist,
            spread=spread,
            low_memory=low_memory,
            n_jobs=n_jobs,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity,
            repulsion_strength=repulsion_strength,
            negative_sample_rate=negative_sample_rate,
            transform_queue_size=transform_queue_size,
            a=a,
            b=b,
            random_state=random_state,
            angular_rp_forest=angular_rp_forest,
            target_n_neighbors=target_n_neighbors,
            target_metric=target_metric,
            target_metric_kwds=target_metric_kwds,
            target_weight=target_weight,
            transform_seed=transform_seed,
            transform_mode=transform_mode,
            force_approximation_algorithm=force_approximation_algorithm,
            verbose=verbose,
            tqdm_kwds=tqdm_kwds,
            unique=unique,
            densmap=densmap,
            dens_lambda=dens_lambda,
            dens_frac=dens_frac,
            dens_var_shift=dens_var_shift,
            output_dens=output_dens,
            disconnection_distance=disconnection_distance,
            precomputed_knn=precomputed_knn,
        )
        super().__init__(**self.umap_params)

    def plot_annotated(
            self,
            data,
            embeddings,
            figsize=(10,10),
            **kwargs
        ):
        assert embeddings.shape[1] == 2, "Embeddings must be 2-dimensional"
        import matplotlib.pyplot as plt
        import matplotlib.offsetbox

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(embeddings[:,0], embeddings[:,1], s=5, cmap='gist_rainbow')

        inx_lst = np.arange(np.concatenate(data.ROI_images).shape[0])
        inx_lst = np.random.choice(inx_lst, 100, replace=False)
        img_lst = np.concatenate(data.ROI_images)[inx_lst]
        x = embeddings[inx_lst, 0]
        y = embeddings[inx_lst, 1]

        for x0, y0, ROI_single in zip(x, y, img_lst):
            offsetImage = matplotlib.offsetbox.OffsetImage(ROI_single, cmap='gray', zoom=1) # initialize offset image to contain ROI visualization
            ab = matplotlib.offsetbox.AnnotationBbox(offsetImage, (x0, y0), frameon=False)
            ax.add_artist(ab)
        
        return fig, ax


def import_data(
        list_dict_data=[]

        # list_dict_data_suite2p=[],
        # list_dict_data_caiman=[],
        # list_dict_data_raw=[],
        # list_dict_data_raw_sparse=[],

        # um_per_pixel,
        # new_or_old_suite2p,
        # out_height_width,
        # type_meanImg,
        # FOV_images,
        # verbose,
):
    print(f"JZ: Number of sessions to be imported from suite2p = {len(list_dict_data_suite2p)}")
    print(f"JZ: Number of sessions to be imported from caiman = {len(list_dict_data_caiman)}")
    print(f"JZ: Number of sessions to be imported from raw = {len(list_dict_data_raw)}")
    print(f"JZ: Number of sessions to be imported from raw_sparse = {len(list_dict_data_raw_sparse)}")

    if len(list_dict_data) > 0:
        # Check that all suite2p imports have the same parameters (except for filepath_stat and filepath_ops)
        params_import = {}
        for dict_data in list_dict_data:
            if dict_data['datatype'] == 'suite2p': assert 'filepath_stat' in dict_data and 'filepath_ops' in dict_data, 'JZ: suite2p imports must include filepath_stat and filepath_ops'
            if dict_data['datatype'] == 'raw_ROIs': assert 'filename_rawImages' in dict_data, 'JZ: raw_ROIs imports must include filepath_stat and filepath_ops'
            if dict_data['datatype'] == 'raw_ROIs_sparse': assert 'filename_rawImages_sparse' in dict_data, 'JZ: raw_ROIs_sparse imports must include filepath_stat and filepath_ops'
            if dict_data['datatype'] == 'caiman': assert False
            
            for key, value in dict_data.items():
                if key == 'datatype':
                    continue
                elif 'filepath' in key:
                    params_import_suite2p[key] = params_import_suite2p.get(key, []) + [value]
                    continue
                elif key in params_import:
                    assert params_import[key] == dict_data[key], f'JZ: All suite2p imports must have the same {key}'
                else:
                    params_import[key] = dict_data[key]

        # Create data importing object to import suite2p data
        data = roicat.data_importing.Data_suite2p(**params_import_suite2p)

        # paths_statFiles=params_import_suite2p['filepath_stat'],
        # paths_opsFiles=params_import_suite2p['filepath_ops'],
        # class_labels=params_import_suite2p['filepath_labels'],
        # um_per_pixel=params['hyperparameters_data']['um_per_pixel'],
        # new_or_old_suite2p=params['hyperparameters_data']['new_or_old_suite2p'],
        # out_height_width=params['hyperparameters_data']['out_height_width'],
        # type_meanImg=params['hyperparameters_data']['type_meanImg'],
        # FOV_images=params['hyperparameters_data']['FOV_images'],
        # verbose=params['hyperparameters_data']['verbose'],
    

    if len(list_dict_data_caiman) > 0:
        # Check that all suite2p imports have the same parameters (except for filepath_stat and filepath_ops)
        params_import_suite2p = {}
        for dict_data_suite2p in list_dict_data_suite2p:
            assert 'filepath_stat' in dict_data_suite2p and 'filepath_ops' in dict_data_suite2p, 'JZ: suite2p imports must include filepath_stat and filepath_ops'
            for key, value in dict_data_suite2p.items():
                if 'filepath' in key:
                    params_import_suite2p[key] = params_import_suite2p.get(key, []) + [value]
                    continue
                elif key in params_import_suite2p:
                    assert params_import_suite2p[key] == dict_data_suite2p[key], f'JZ: All suite2p imports must have the same {key}'
                else:
                    params_import_suite2p[key] = dict_data_suite2p[key]

        # Create data importing object to import suite2p data
        data = roicat.data_importing.Data_suite2p(**params_import_suite2p)

        # paths_statFiles=params_import_suite2p['filepath_stat'],
        # paths_opsFiles=params_import_suite2p['filepath_ops'],
        # class_labels=params_import_suite2p['filepath_labels'],
        # um_per_pixel=params['hyperparameters_data']['um_per_pixel'],
        # new_or_old_suite2p=params['hyperparameters_data']['new_or_old_suite2p'],
        # out_height_width=params['hyperparameters_data']['out_height_width'],
        # type_meanImg=params['hyperparameters_data']['type_meanImg'],
        # FOV_images=params['hyperparameters_data']['FOV_images'],
        # verbose=params['hyperparameters_data']['verbose'],
    

    elif params['datatype'] == "caiman":
        
        # TODO: Add Caiman data importing
        # # assert 'filename_stat' in params['paths'] and 'filename_ops' in params['paths'], 'JZ: The caiman params.json file must include paths.filename_stat and paths.filename_ops for stat_s2p datatype'
        # filepath_data_stat = str((Path(params['paths']['directory_data']) / params['paths']['filename_stat']).resolve())
        # filepath_data_ops = str((Path(params['paths']['directory_data']) / params['paths']['filename_ops']).resolve())

        # # # Create data importing object to import suite2p data
        # # data = roicat.data_importing.Data_caiman(
        # #     paths_statFiles=[filepath_data_stat],
        # #     paths_opsFiles=[filepath_data_ops],
        # #     class_labels=[filepath_data_labels],
        #     # um_per_pixel=params['hyperparameters_data']['um_per_pixel'],
        #     # new_or_old_suite2p=params['hyperparameters_data']['new_or_old_suite2p'],
        #     # out_height_width=params['hyperparameters_data']['out_height_width'],
        #     # type_meanImg=params['hyperparameters_data']['type_meanImg'],
        #     # FOV_images=params['hyperparameters_data']['FOV_images'],
        #     # verbose=params['hyperparameters_data']['verbose'],
        # # )
        pass
    elif params['datatype'] == "raw_images":
        assert 'filename_rawImages' in params['paths'], 'JZ: The suite2p params.json file must include paths.filename_rawImages for raw_images datatype'
        filepath_data_rawImages = str((Path(params['paths']['directory_data']) / params['paths']['filename_rawImages']).resolve())

        sf = scipy.sparse.load_npz(filepath_data_rawImages)
        labels = np.load(filepath_data_labels)

        data = roicat.data_importing.Data_roicat(verbose=True)
        data.set_ROI_images(ROI_images=[sf.A.reshape(sf.shape[0], 36, 36)], um_per_pixel=params['hyperparameters_data']['um_per_pixel'])
        data.set_class_labels(class_labels=[labels.astype(int)])
    else:
        raise ValueError(f"Invalid datatype for simclr: {params['datatype']}")

    return

class Datasplit():
    """
    Split data into training and validation sets with additional helper methods
    JZ 2023
    """
    def __init__(self, features, labels, n_train=None, train_size=None, val_size=None, test_size=None):
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
        # Assert that exactly 2 or 3 of train_size, val_size, and test_size are not None and that the sum of those specified add up to 1 if 3 are specified or <= 1 if 2 are specified
        assert sum([train_size is not None, val_size is not None, test_size is not None]) == 2 or sum([train_size is not None, val_size is not None, test_size is not None]) == 3, 'JZ: Exactly 2 or 3 of train_size, val_size, and test_size must be specified'
        if sum([train_size is not None, val_size is not None, test_size is not None]) == 3:
            assert train_size + val_size + test_size == 1, 'JZ: train_size + val_size + test_size must equal 1'
        elif sum([train_size is not None, val_size is not None, test_size is not None]) == 2:
            assert sum([set_size for set_size in [train_size, val_size, test_size] if set_size is not None]) <= 1, 'JZ: Specified train_size + val_size + test_size must be <= 1'

        self.features = features
        self.labels = labels
        self.n_train = n_train

        self.train_size = 1 - val_size - test_size if train_size is None else train_size
        self.val_size = 1 - train_size - test_size if train_size is None else train_size
        self.test_size = 1 - train_size - val_size if train_size is None else train_size

        # Extract training data
        self.idx_train, self.idx_nonTrain = self.stratified_sample(self.features, self.labels, n_splits=1, test_size=(self.val_size + self.test_size))
        self.features_train = self.features[self.idx_train]
        self.labels_train = self.labels[self.idx_train]

        # Temporary Holder for non-train data to be split into validation and test sets
        self.features_nonTrain = self.features[self.idx_nonTrain]
        self.labels_nonTrain = self.labels[self.idx_nonTrain]

        # Extract validation and test data
        self.idx_val, self.idx_test = self.stratified_sample(self.features_nonTrain, self.labels_nonTrain, n_splits=1, test_size=self.test_size/(1 - self.train_size))
        self.features_val = self.features_nonTrain[self.idx_train]
        self.labels_val = self.labels_nonTrains[self.idx_train]
        self.features_test = self.features_nonTrain[self.idx_test]
        self.labels_test = self.labels_nonTrains[self.idx_test]

        # Downsample training data if specified
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
