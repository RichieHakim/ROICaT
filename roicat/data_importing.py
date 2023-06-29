import pathlib
from pathlib import Path
import copy
import warnings
from typing import List, Optional, Union, Tuple, Dict, Any, Callable, Iterable

import numpy as np
from tqdm import tqdm
import scipy.sparse
import sparse

from . import helpers, util


"""
Classes for importing data into the roicat package.

Conventions:
    - Data_roicat is the super class for all data objects.
    - Data_roicat can be used to make a custom data object.
    - Subclasses like Data_suite2p and Data_caiman should be used
        to import data from files and convert it to a
        Data_roicat ingestable format.
    - Subclass import methods should be functions that return
        properly formatted data ready for the superclass to ingest.
    - Avoid directly setting attributes. Try to always use a
        .set_attribute() method.
    - Subclass import methods should operate at the multi-session
        level. That is, they should take in lists of objects
        corresponding to multiple sessions.
    - Subclasses should be able to initialize classification and
        tracking independently. Minimize interdependent attributes.
    - Users should have flexibility in the following switch-cases:
        - FOV_images:
            - From file
            - From object
            - Only specify FOV_height and FOV_width
    - Only default to importing from file if the file is deriving
        from a standardized format (ie suite2p or caiman). Do not
        require standardization for custom data objects like class
        labels.
"""

############################################################################################################################
####################################### SUPER CLASS FOR ALL DATA OBJECTS ###################################################
############################################################################################################################

class Data_roicat(util.ROICaT_Module):
    """
    Superclass for all data objects. Can be used as a template for creating
    custom data objects. RH 2022

    Args:
        verbose (bool): 
            Determines whether to print status updates. (Default is ``True``)

    Attributes:
        type (object): 
            The type of the data object. Set by the subclass.
        n_sessions (int): 
            The number of imaging sessions.
        n_roi (int): 
            The number of ROIs in each session.
        n_roi_total (int): 
            The total number of ROIs across all sessions.
        FOV_height (int): 
            The height of the field of view in pixels.
        FOV_width (int): 
            The width of the field of view in pixels.
        FOV_images (List[np.ndarray]): 
            A list of numpy arrays, each with shape *(FOV_height, FOV_width)*.
            Each element represents an imaging session.
        ROI_images (List[np.ndarray]): 
            A list of numpy arrays, each with shape *(n_roi, height, width)*.
            Each element represents an imaging session and each element of the
            numpy array (first dimension) is an ROI.
        spatialFootprints (List[object]): 
            A list of scipy.sparse.csr_matrix objects, each with shape *(n_roi,
            FOV_height*FOV_width)*. Each element represents an imaging session.
        class_labels_raw (List[np.ndarray]): 
            A list of numpy arrays, each with shape *(n_roi,)*, where each
            element is an integer. Each element of the list is an imaging
            session and each element of the numpy array is a class label.
        class_labels_index (List[np.ndarray]): 
            A list of numpy arrays, each with shape *(n_roi,)*, where each
            element is an integer. Each element of the list is an imaging
            session and each element of the numpy array is the index of the
            class label obtained from passing the raw class label through
            np.unique(*, return_inverse=True).
        um_per_pixel (float): 
            The conversion factor from pixels to microns. This is used to scale
            the ROI_images to a common size.
        session_bool (np.ndarray): 
            A boolean matrix with shape *(n_roi_total, n_sessions)*. Each
            element is ``True`` if the ROI is present in the session.
    """
    def __init__(
        self, 
        verbose: bool = True,
    ) -> None:
        """
        Initializes the Data_roicat object with the specified verbosity.
        """
        ## Imports
        super().__init__()

        self._verbose = verbose
    
        self.type = type(self)  ## Overwrites the superclass attribute self.type

    #########################################################
    ################# CLASSIFICATION ########################
    #########################################################

    def set_ROI_images(
        self,
        ROI_images: List[np.ndarray],
        um_per_pixel: Optional[float] = None,
    ) -> None:
        """
        Imports ROI images into the class. Images are expected to be formatted
        as a list of numpy arrays. Each element is an imaging session. Each
        element is a numpy array of shape *(n_roi, FOV_height, FOV_width)*. This
        method will set the attributes: self.ROI_images, self.n_roi,
        self.n_roi_total, self.n_sessions. If any of these attributes are
        already set, it will verify the new values match the existing ones.

        Args:
            ROI_images (List[np.ndarray]): 
                List of numpy arrays each of shape *(n_roi, FOV_height,
                FOV_width)*.
            um_per_pixel (Optional[float]): 
                The number of microns per pixel. This is used to resize the
                images to a common size. (Default is ``None``)
        """
        ## Warn if no um_per_pixel is provided
        if um_per_pixel is None:
            ## Check if it is already set
            if hasattr(self, 'um_per_pixel'):
                um_per_pixel = self.um_per_pixel
            warnings.warn("RH WARNING: No um_per_pixel provided. We recommend making an educated guess. Assuming 1.0 um per pixel. This will affect the embedding results.")
            um_per_pixel = 1.0

        print(f"Starting: Importing ROI images") if self._verbose else None

        ## Check the validity of the inputs
        ### Check ROI_images
        if isinstance(ROI_images, np.ndarray):
            print("RH WARNING: ROI_images is a numpy array. Assuming n_sessions==1 and wrapping array in a list.")
            ROI_images = [ROI_images]
        assert isinstance(ROI_images, list), f"ROI_images should be a list. It is a {type(ROI_images)}"
        assert all([isinstance(roi, np.ndarray) for roi in ROI_images]), f"ROI_images should be a list of numpy arrays. First element of list is of type {type(ROI_images[0])}"
        assert all([roi.ndim==3 for roi in ROI_images]), f"ROI_images should be a list of numpy arrays of shape (n_roi, FOV_height, FOV_width). First element of list is of shape {ROI_images[0].shape}"
        ### Assert that all the FOV heights and widths are the same
        assert all([roi.shape[1]==ROI_images[0].shape[1] for roi in ROI_images]), f"All the FOV heights should be the same. First element of list is of shape {ROI_images[0].shape}"
        assert all([roi.shape[2]==ROI_images[0].shape[2] for roi in ROI_images]), f"All the FOV widths should be the same. First element of list is of shape {ROI_images[0].shape}"

        self._check_um_per_pixel(um_per_pixel)
        um_per_pixel = float(um_per_pixel)
            
        ## Define some variables
        n_sessions = len(ROI_images)
        n_roi = [roi.shape[0] for roi in ROI_images]
        n_roi_total = int(sum(n_roi))

        ## Check that attributes match if they already exist as an attribute
        if hasattr(self, 'n_sessions'):
            assert self.n_sessions == n_sessions, f"n_sessions is already set to {self.n_sessions} but new value is {n_sessions}"
        if hasattr(self, 'n_roi'):
            assert self.n_roi == n_roi, f"n_roi is already set to {self.n_roi} but new value is {n_roi}"
        if hasattr(self, 'n_roi_total'):
            assert self.n_roi_total == n_roi_total, f"n_roi_total is already set to {self.n_roi_total} but new value is {n_roi_total}"

        ## Set attributes
        self.n_sessions = n_sessions
        self.n_roi = n_roi
        self.n_roi_total = n_roi_total
        self.ROI_images = ROI_images
        self.um_per_pixel = um_per_pixel
        
        print(f"Completed: Imported {n_sessions} sessions. Each session has {n_roi} ROIs. Total number of ROIs is {n_roi_total}. The um_per_pixel is {um_per_pixel} um per pixel.") if self._verbose else None

    def set_class_labels(
        self,
        labels: Optional[Union[List[np.ndarray], np.ndarray]] = None,
        path_labels: Optional[Union[str, List[str]]] = None,
        n_classes: Optional[int] = None,
    ) -> None:
        """
        Imports class labels into the class. 

        * labels are expected to be formatted as a list of numpy arrays or
          strings. Each element in the list is a session, and each element in
          the numpy array is associated with the nth element of the
          self.ROI_images list. Each element is a numpy array of shape
          *(n_roi,)*.

        * Sets the attributes: self.class_labels_raw, self.class_labels_index,
          self.n_classes, self.n_class_labels, self.n_class_labels_total,
          self.unique_class_labels. If any of these attributes are already set,
          they will verify the new values match the existing ones.

        Args:
            labels (Optional[Union[List[np.ndarray], np.ndarray]]): \n
                * If ``None``: path_labels must be specified. 
                * If a ``list`` of ``np.ndarray``: each element should be a 1D
                  array of integers or strings of length *n_roi* specifying the
                  class label for each ROI. \n
                (Default is ``None``)
            path_labels (Optional[Union[str, List[str]]]): \n
                * If ``None``: labels must be specified.
                * If a ``list`` of ``str``: each element should be a path to a
                  either: \n
                    * A ``.npy`` file containing a numpy array of shape
                      *(n_roi,)* OR
                    * A ``.pkl`` or ``.npy`` file containing a dictionary with
                      an item that has key 'labels' and value of a numpy
                      array of shape *(n_roi,)*.  \n
                The numpy array should be of integers or strings specifying the
                class label

            n_classes (Optional[int]): 
                Number of classes. If not provided, it will be inferred from the
                class labels. (Default is ``None``)
        """
        print(f"Starting: Importing class labels") if self._verbose else None

        if path_labels is not None:
            assert labels is None, f"labels is not None but path_labels is not None. Please specify only one of them."
            ## Convert to a list if it is not already
            if isinstance(path_labels, list) == False:
                print(f'Input labels is not a list. Wrapping it in a list.') if self._verbose else None
                path_labels =[path_labels]
            ## Assert that all the elements are strings
            assert all([isinstance(l, str) for l in path_labels]), f"path_labels should be a list of strings. First element of list is of type {type(path_labels[0])}"
            ## Check the file extension
            extension = Path(path_labels[0]).suffix
            ## Load the labels
            if extension == '.npy':
                self.class_labels_raw = [np.load(p, allow_pickle=True)[()] for p in path_labels]
            elif extension == '.pkl':
                self.class_labels_raw = [helpers.pickle_load(p) for p in path_labels]
            else:
                raise ValueError(f"File extension {extension} is not supported. Please use either .npy or .pkl")
            ## Check that if the inputs are dictionaries, we extract the labels
            if isinstance(self.class_labels_raw[0], dict):
                assert all(['labels' in l for l in self.class_labels_raw]), f"Found a dictionary in the .npy file. The dictionary should have a key 'labels' with a value of a numpy array of shape (n_roi,)."
                self.class_labels_raw = [l['labels'] for l in self.class_labels_raw]
        else:
            assert labels is not None, f"Either labels or path_labels must be specified."
            assert isinstance(labels, str) == False, f"labels is a string. Did you mean to specify path_labels?"
            self.class_labels_raw = labels

        ## Convert to a list if it is not already
        if isinstance(self.class_labels_raw, list) == False:
            print(f'Input labels is not a list. Wrapping it in a list.') if self._verbose else None
            self.class_labels_raw = [self.class_labels_raw]
        ## Assert that all the elements are numpy arrays
        assert all([isinstance(l, np.ndarray) for l in self.class_labels_raw]), f"labels should be a list of numpy arrays. First element of list is of type {type(self.class_labels_raw[0])}"
        ## Assert that all the elements are 1D
        assert all([l.ndim==1 for l in self.class_labels_raw]), f"labels should be a list of 1D numpy arrays. First element of list is of shape {self.class_labels_raw[0].shape}"

        ## Define some variables
        n_sessions = len(self.class_labels_raw)
        labels_cat = np.concatenate(self.class_labels_raw, axis=0)
        labels_cat_squeezeInt = np.unique(labels_cat, return_inverse=True)[1].astype(np.int64)
        unique_class_labels = np.unique(labels_cat)
        if n_classes is not None:
            assert len(unique_class_labels) <= n_classes, f"RH ERROR: User provided n_classes={n_classes} but there are {len(unique_class_labels)} unique class labels in the provided class_labels." if self._verbose else None
        else:
            n_classes = len(unique_class_labels)
        n_class_labels = [lbls.shape[0] for lbls in self.class_labels_raw]
        n_class_labels_total = sum(n_class_labels)
        class_labels_squeezeInt = [labels_cat_squeezeInt[sum(n_class_labels[:ii]):sum(n_class_labels[:ii+1])] for ii in range(n_sessions)]

        ## Set attributes
        self.class_labels_index = class_labels_squeezeInt
        self.n_classes = n_classes
        self.n_class_labels = n_class_labels
        self.n_class_labels_total = n_class_labels_total
        self.unique_class_labels = unique_class_labels

        ## Check if label data shapes match ROI_image data shapes
        self._checkValidity_classLabels_vs_ROIImages()

        print(f"Completed: Imported labels for {n_sessions} sessions. Each session has {n_class_labels} class labels. Total number of class labels is {n_class_labels_total}.") if self._verbose else None

    def _check_um_per_pixel(self, um_per_pixel: float,) -> None:
        """
        Checks whether um_per_pixel is of appropriate type and is positive.

        Args:
            um_per_pixel (float):
                The number of microns per pixel.
        """
        assert isinstance(um_per_pixel, (int, float)), f"um_per_pixel should be a float. It is a {type(um_per_pixel)}"
        assert um_per_pixel > 0, f"um_per_pixel should be a positive number. It is {um_per_pixel}"

    def _checkValidity_classLabels_vs_ROIImages(
        self, 
        verbose: Optional[bool] = None,
    ) -> None:
        """
        Checks that the class labels and the ROI images have the same number of
        sessions and the same number of ROIs in each session.

        Args:
            verbose (Optional[bool]):
                If ``None``, the verbosity level set in the class is used.
                (Default is ``None``)
        """
        if verbose is None:
            verbose = self._verbose

        ## Check if class_labels and ROI_images exist
        if not (hasattr(self, 'class_labels_index') and hasattr(self, 'ROI_images')):
            print("Cannot check validity of class_labels_index and ROI_images because one or both do not exist as attributes.") if verbose else None
            return False
        ## Check num sessions
        n_sessions_classLabels = len(self.class_labels_index)
        n_sessions_ROIImages = len(self.ROI_images)
        assert n_sessions_classLabels == n_sessions_ROIImages, f"RH ERROR: Number of sessions (list elements) in class_labels_index ({n_sessions_classLabels}) does not match number of sessions (list elements) in ROI_images ({n_sessions_ROIImages})."
        ## Check num ROIs
        n_ROIs_classLabels = [lbls.shape[0] for lbls in self.class_labels_index]
        n_ROIs_ROIImages = [img.shape[0] for img in self.ROI_images]
        assert all([l == r for l, r in zip(n_ROIs_classLabels, n_ROIs_ROIImages)]), f"RH ERROR: Number of ROIs in each session in class_labels_index ({n_ROIs_classLabels}) does not match number of ROIs in each session in ROI_images ({n_ROIs_ROIImages})."
        print(f"Labels and ROI Images match in shapes: Class labels and ROI images have the same number of sessions and the same number of ROIs in each session.") if verbose else None
        return True

    #########################################################
    #################### TRACKING ###########################
    #########################################################

    def set_spatialFootprints(
        self,
        spatialFootprints: List[Union[np.ndarray, scipy.sparse.csr_matrix, Dict[str, Any]]],
        um_per_pixel: Optional[float] = None,
    ):
        """
        Sets the **spatialFootprints** attribute.

        Args:
            spatialFootprints (List[Union[np.ndarray, csr_matrix, Dict[str, Any]]]): 
                One of the following: \n
                * List of **numpy.ndarray** objects, one for each session. Each
                  array should have shape *(n_ROIs, FOV_height, FOV_width)*.
                * List of **scipy.sparse.csr_matrix** objects, one for each
                  session. Each matrix should have shape *(n_ROIs, FOV_height *
                  FOV_width)*. Reshaping should be done with 'C' indexing
                  (standard).
                * List of dictionaries, one for each session. This dictionary
                  should be a serialized **scipy.sparse.csr_matrix** object. It
                  should contains keys: 'data', 'indices', 'indptr', 'shape'.
                  See **scipy.sparse.csr_matrix** for more information.
            um_per_pixel (Optional[float]): 
                The number of microns per pixel. This is used to resize the
                images to a common size. (Default is ``None``)
        """
        ## Warn if no um_per_pixel is provided
        if um_per_pixel is None:
            ## Check if it is already set
            if hasattr(self, 'um_per_pixel'):
                um_per_pixel = self.um_per_pixel
            else:
                warnings.warn("RH WARNING: No um_per_pixel provided. We recommend making an educated guess. Assuming 1.0 um per pixel. This will affect the embedding results.")
                um_per_pixel = 1.0

        ## Check inputs
        if isinstance(spatialFootprints, list)==False:
            print(f'RH WARNING: Input spatialFootprints is not a list. Converting to list.')
            spatialFootprints = [spatialFootprints]

        ## If the input are dictionaries, assume that it is a serialized scipy.sparse.csr_matrix object and convert it
        if all([isinstance(s, dict) for s in spatialFootprints]):
            print("RH WARNING: spatialFootprints are dictionaries, assuming that they are serialized scipy.sparse.csr_matrix objects and converting them.") if self._verbose else None
            sf_all = [scipy.sparse.csr_matrix((sf['data'], sf['indices'], sf['indptr']), shape=sf['_shape']) for sf in spatialFootprints]
        ## If the input are numpy.ndarray objects, convert them to scipy.sparse.csr_matrix objects
        elif all([isinstance(s, np.ndarray) for s in spatialFootprints]):
            print("RH WARNING: spatialFootprints are numpy.ndarray objects. Assuming structure is a list of arrays (1 per session) of shape (n_roi, height, width), converting them to scipy.sparse.csr_matrix objects.") if self._verbose else None
            sf_all = [scipy.sparse.csr_matrix(sf.reshape(sf.shape[0], -1), copy=False) for sf in spatialFootprints]
            self.set_FOVHeightWidth(FOV_height=spatialFootprints[0].shape[1], FOV_width=spatialFootprints[0].shape[2])
        elif all([scipy.sparse.issparse(s) for s in spatialFootprints]):
            sf_all = [sf.tocsr() for sf in spatialFootprints]
        else:
            raise ValueError(f"spatialFootprints should be a list of numpy.ndarray objects, scipy.sparse.csr_matrix objects, or dictionaries of csr_matrix input arguments (see documentation). Found elements of type: {type(spatialFootprints[0])}")

        self._check_um_per_pixel(um_per_pixel)
        um_per_pixel = float(um_per_pixel)

        ## Get some variables
        n_sessions = len(sf_all)
        n_roi = [sf.shape[0] for sf in sf_all]
        n_roi_total = int(np.sum(n_roi))

        ## Check that attributes match if they already exist as an attribute
        if hasattr(self, 'n_sessions'):
            if self.n_sessions != n_sessions:
                warnings.warn(f"RH WARNING: n_sessions is already set to {self.n_sessions} but new value is {n_sessions}")
        if hasattr(self, 'n_roi'):
            if all([n == r for n, r in zip(self.n_roi, n_roi)])==False:
                warnings.warn(f"RH WARNING: n_roi is already set to {self.n_roi} but new value is {n_roi}")
        if hasattr(self, 'n_roi_total'):
            if self.n_roi_total != n_roi_total:
                warnings.warn(f"RH WARNING: n_roi_total is already set to {self.n_roi_total} but new value is {n_roi_total}")

        ## Set attributes
        self.spatialFootprints = sf_all
        self.um_per_pixel = um_per_pixel
        self.n_sessions = n_sessions
        self.n_roi = n_roi
        self.n_roi_total = n_roi_total
        print(f"Completed: Set spatialFootprints for {len(sf_all)} sessions successfully.") if self._verbose else None

    def set_FOV_images(
        self,
        FOV_images: List[np.ndarray],
    ):
        """
        Sets the **FOV_images** attribute.

        Args:
            FOV_images (List[np.ndarray]): 
                List of 2D **numpy.ndarray** objects, one for each session. Each
                array should have shape *(FOV_height, FOV_width)*.
        """
        if isinstance(FOV_images, np.ndarray):
            assert FOV_images.ndim == 3, f"RH ERROR: FOV_images must be a list of 2D numpy arrays."
            FOV_images = [fov for fov in FOV_images]
        ## Check inputs
        assert isinstance(FOV_images, list), f"RH ERROR: FOV_images must be a list."
        assert all([isinstance(img, np.ndarray) for img in FOV_images]), f"RH ERROR: All elements in FOV_images must be numpy arrays."
        assert all([img.ndim == 2 for img in FOV_images]), f"RH ERROR: All elements in FOV_images must be 2D numpy arrays."
        assert all([img.shape[0] == FOV_images[0].shape[0] for img in FOV_images]), f"RH ERROR: All elements in FOV_images must have the same height and width."
        assert all([img.shape[1] == FOV_images[0].shape[1] for img in FOV_images]), f"RH ERROR: All elements in FOV_images must have the same height and width."

        ## Set attributes
        self.FOV_images = [np.array(f, dtype=np.float32) for f in FOV_images]
        self.FOV_height = int(FOV_images[0].shape[0])
        self.FOV_width = int(FOV_images[0].shape[1])

        ## Get some variables
        n_sessions = len(FOV_images)

        ## Check that attributes match if they already exist as an attribute
        if hasattr(self, 'n_sessions'):
            if self.n_sessions != n_sessions:
                warnings.warn(f"RH WARNING: n_sessions is already set to {self.n_sessions} but new value is {n_sessions}")

        print(f"Completed: Set FOV_images for {len(FOV_images)} sessions successfully.") if self._verbose else None

    def set_FOVHeightWidth(
        self,
        FOV_height: int,
        FOV_width: int,
    ):
        """
        Sets the **FOV_height** and **FOV_width** attributes.

        Args:
            FOV_height (int): 
                The height of the field of view (FOV) in pixels.
            FOV_width (int): 
                The width of the field of view (FOV) in pixels.
        """
        ## Check inputs
        assert isinstance(FOV_height, int), f"RH ERROR: FOV_height must be an integer."
        assert isinstance(FOV_width, int), f"RH ERROR: FOV_width must be an integer."

        ## Set attributes
        self.FOV_height = FOV_height
        self.FOV_width = FOV_width

        print(f"Completed: Set FOV_height and FOV_width successfully.") if self._verbose else None

    def get_maxIntensityProjection_spatialFootprints(
        self, 
        sf: Optional[List[scipy.sparse.csr_matrix]] = None,
        normalize: bool = True,
    ):
        """
        Returns the maximum intensity projection of the spatial footprints.

        Args:
            sf (List[scipy.sparse.csr_matrix]): 
                List of spatial footprints, one for each session.
            normalize (bool):
                If True, normalizes the [min, max] range of each ROI to [0, 1]
                before computing the maximum intensity projection.

        Returns:
            List[np.ndarray]: 
                List of maximum intensity projections, one for each session.
        """
        if sf is None:
            assert hasattr(self, 'spatialFootprints'), f"RH ERROR: spatialFootprints must be set as an attribute if not provided as an argument."
            sf = copy.deepcopy(self.spatialFootprints)
        else:
            if isinstance(sf, list) == False:
                sf = [sf]
            assert all([isinstance(s, scipy.sparse.csr_matrix) for s in sf]), f"RH ERROR: All elements in sf must be scipy.sparse.csr_matrix objects."

        assert hasattr(self, 'FOV_height'), f"RH ERROR: FOV_height must be set as an attribute."
        assert hasattr(self, 'FOV_width'), f"RH ERROR: FOV_width must be set as an attribute."

        if normalize:
            sf = [s.multiply(s.max(axis=1).power(-1)) for s in sf]
        mip = [(s).max(axis=0).reshape(self.FOV_height, self.FOV_width).toarray() for s in sf]
        return mip

    def _checkValidity_spatialFootprints_and_FOVImages(
        self,
        verbose: Optional[bool] = None,
    ):
        """
        Checks that **spatialFootprints** and **FOV_images** are compatible.

        Args:
            verbose (Optional[bool]): 
                If ``True``, outputs progress and error messages. 
                (Default is ``None``)
        """
        if verbose is None:
            verbose = self._verbose
        if hasattr(self, 'spatialFootprints') and hasattr(self, 'FOV_images'):
            assert len(self.spatialFootprints) == len(self.FOV_images), f"RH ERROR: spatialFootprints and FOV_images must have the same length."
            assert all([sf.shape[1] == self.FOV_images[0].size for sf in self.spatialFootprints]), f"RH ERROR: spatialFootprints and FOV_images must have the same size."
            print("Completed: spatialFootprints and FOV_images are compatible.") if verbose else None
            return True
        else:
            print("Cannot check validity of spatialFootprints and FOV_images because one or both do not exist as attributes.") if verbose else None
            return False

    def check_completeness(
        self, 
        verbose: bool = True
    ) -> None:
        """
        Checks which pipelines the data object is capable of running given the
        attributes that have been set.

        Args:
            verbose (bool): 
                If ``True``, outputs progress and error messages. (Default is
                ``True``)
        """
        completeness = {}
        keys_classification_inference = ['ROI_images', 'um_per_pixel']
        keys_classification_training = ['ROI_images', 'um_per_pixel', 'class_labels_index']
        keys_tracking = ['ROI_images', 'um_per_pixel', 'spatialFootprints', 'FOV_images']
        
        ## Check classification inference:
        ### ROI_images, um_per_pixel
        if all([hasattr(self, key) for key in keys_classification_inference]):
            completeness['classification_inference'] = True
        else:
            print(f"RH WARNING: Classification-Inference incomplete because following attributes are missing: {[key for key in keys_classification_inference if not hasattr(self, key)]}") if verbose else None
            completeness['classification_inference'] = False
        ## Check classification training:
        ### ROI_images, um_per_pixel, class_labels_index
        if all([hasattr(self, key) for key in keys_classification_training]):
            completeness['classification_training'] = True
        else:
            print(f"RH WARNING: Classification-Training incomplete because the following attributes are missing: {[key for key in keys_classification_training if not hasattr(self, key)]}") if verbose else None
            completeness['classification_training'] = False
        ## Check tracking:
        ### um_per_pixel, spatialFootprints, FOV_images
        if all([hasattr(self, key) for key in keys_tracking]):
            completeness['tracking'] = True
        else:
            print(f"RH WARNING: Tracking incomplete because the following attributes are missing: {[key for key in keys_tracking if not hasattr(self, key)]}") if verbose else None
            completeness['tracking'] = False

        self._checkValidity_classLabels_vs_ROIImages(verbose=verbose)
        self._checkValidity_spatialFootprints_and_FOVImages(verbose=verbose)

        ## Print completeness
        print(f"Data_roicat object completeness: {completeness}") if verbose else None
        return completeness


    def _make_session_bool(self) -> np.ndarray:
        """
        Creates a boolean array where each row is a boolean vector indicating
        which session(s) the ROI was present in. Uses the ``self.n_roi``
        attribute to determine which rows belong to which session.

        Returns:
            np.ndarray:
                self.session_bool (np.ndarray):
                    A boolean array where each row is a boolean vector
                    indicating which session(s) the ROI was present in. Shape:
                    *(n_roi_total, n_sessions)*
        """
        ## Check that n_roi is set
        assert hasattr(self, 'n_roi'), f"RH ERROR: n_roi must be set before session_bool can be created."
        ## Check that n_roi is the correct length
        assert len(self.n_roi) == self.n_sessions, f"RH ERROR: n_roi must be the same length as n_sessions."
        ## Check that n_roi_total is correct
        assert sum(self.n_roi) == self.n_roi_total, f"RH ERROR: n_roi must sum to n_roi_total."
        ## Create session_bool
        self.session_bool = util.make_session_bool(self.n_roi)

        print(f"Completed: Created session_bool.") if self._verbose else None

        return self.session_bool


    def _make_spatialFootprintCentroids(
        self, 
        method: str = 'centerOfMass'
    ) -> np.ndarray:
        """
        Calculates the centroids of a sparse array of flattened spatial
        footprints. The centroid position is calculated as the center of mass of
        the ROI.
        JZ, RH 2022

        Args:
            method (str): 
                Method to use to calculate the centroid. Either \n
                * ``'centerOfMass'``: Calculates the centroid position as the
                  mean center of mass of the ROI.
                * ``'median'``: Calculates the centroid position as the median
                  center of mass of the ROI. \n
                (Default is ``'centerOfMass'``)

        Returns:
            (np.ndarray): 
                centroids (np.ndarray): 
                    Centroids of the ROIs with shape *(2, n_roi)*. Consists of
                    (y, x) coordinates.
        """
        ## Check that sf is a list of csr sparse arrays
        assert isinstance(self.spatialFootprints, list), f"RH ERROR: spatialFootprints must be a list of scipy.sparse.csr_matrix."
        assert all([isinstance(sf, scipy.sparse.csr_matrix) for sf in self.spatialFootprints]), f"RH ERROR: spatialFootprints must be a list of scipy.sparse.csr_matrix."
        ## Check that FOV_height and FOV_width are set
        assert hasattr(self, 'FOV_height') and hasattr(self, 'FOV_width'), f"RH ERROR: FOV_height and FOV_width must be set before centroids can be calculated."
        ## Check that sf is the correct shape
        assert all([sf.shape[1] == self.FOV_height*self.FOV_width for sf in self.spatialFootprints]), f"RH ERROR: spatialFootprints must have shape (n_roi, FOV_height*FOV_width)."
        ## Check that centroid_method is set
        assert method in ['centerOfMass', 'median'], f"RH ERROR: centroid_method must be one of ['centerOfMass', 'median']."

        ## Calculate centroids
        sf = self.spatialFootprints
        FOV_height, FOV_width = self.FOV_height, self.FOV_width
        ## Reshape sf to (n_roi, FOV_height, FOV_width)
        sf_rs = [sparse.COO(s).reshape((s.shape[0], FOV_height, FOV_width), order='C') for s in sf]
        ## Calculate the sum of the weights along each axis
        y_w, x_w = [s.sum(axis=2) for s in sf_rs], [s.sum(axis=1) for s in sf_rs]
        ## Calculate the centroids
        if method == 'centerOfMass':
            y_cent = [(((w*np.arange(w.shape[1]).reshape(1,-1))).sum(1)/(w.sum(1)+1e-12)).todense() for w in y_w]
            x_cent = [(((w*np.arange(w.shape[1]).reshape(1,-1))).sum(1)/(w.sum(1)+1e-12)).todense() for w in x_w]
        elif method == 'median':
            y_cent = [((((w!=0)*np.arange(w.shape[1]).reshape(1,-1, order='C'))).todense()).astype(np.float32) for w in y_w]
            y_cent = [np.ma.masked_array(w, mask=(w==0)).filled(np.nan) for w in y_cent]
            y_cent = [np.nanmedian(w, axis=1) for w in y_cent]
            x_cent = [((((w!=0)*np.arange(w.shape[1]).reshape(1,-1, order='C'))).todense()).astype(np.float32) for w in x_w]
            x_cent = [np.ma.masked_array(w, mask=(w==0)).filled(np.nan) for w in x_cent]
            x_cent = [np.nanmedian(w, axis=1) for w in x_cent]

        ## Round to nearest integer
        y_cent = [np.round(h) for h in y_cent]
        x_cent = [np.round(w) for w in x_cent]
        
        ## Concatenate and store
        self.centroids = [np.stack([y, x], axis=1).astype(np.int64) for y, x in zip(y_cent, x_cent)]
        print(f"Completed: Created centroids.") if self._verbose else None

    
    def _transform_spatialFootprints_to_ROIImages(
        self, 
        out_height_width: Tuple[int, int] = (36, 36)
    ) -> np.ndarray:
        """
        Transforms sparse spatial footprints to dense ROI images.

        Args:
            out_height_width (Tuple[int, int]): 
                Height and width of the output images. 
                (Default is *(36, 36)*)

        Returns:
            (np.ndarray):
                self.ROI_images (np.ndarray):
                    ROI images with shape *(n_roi, out_height_width[0], out_height_width[1])*.
        """
        ## Check inputs
        assert hasattr(self, 'spatialFootprints'), f"RH ERROR: spatialFootprints must be set before ROI images can be created."
        assert hasattr(self, 'FOV_height') and hasattr(self, 'FOV_width'), f"RH ERROR: FOV_height and FOV_width must be set before ROI images can be created."
        assert isinstance(out_height_width, (tuple, list)), f"RH ERROR: out_height_width must be a tuple or list containing two elements (y, x)."
        assert len(out_height_width) == 2, f"RH ERROR: out_height_width must be a tuple of length 2."
        assert all([isinstance(h, int) for h in out_height_width]), f"RH ERROR: out_height_width must be a tuple of integers."
        assert all([h > 0 for h in out_height_width]), f"RH ERROR: out_height_width must be a tuple of positive integers."

        if hasattr(self, 'centroids') == False:
            print(f"Centroids must be set before ROI images can be created. Creating centroids now.") if self._verbose else None
            self._make_spatialFootprintCentroids()

        ## Make helper function
        def sf_to_centeredROIs(sf, centroids):
            half_widths = np.ceil(np.array(out_height_width)/2).astype(int)
            sf_rs = sparse.COO(sf).reshape((sf.shape[0], self.FOV_height, self.FOV_width))

            coords_diff = np.diff(sf_rs.coords[0])
            assert np.all(coords_diff < 1.01) and np.all(coords_diff > -0.01), \
                "RH ERROR: sparse.COO object has strange .coords attribute. sf_rs.coords[0] should all be 0 or 1. An ROI is possibly all zeros."
            
            idx_split = (sf_rs>0).astype(np.bool_).sum((1,2)).todense().cumsum()[:-1]
            coords_split = [np.split(sf_rs.coords[ii], idx_split) for ii in [0,1,2]]
            coords_split[1] = [coords - centroids[0][ii] + half_widths[0] for ii,coords in enumerate(coords_split[1])]
            coords_split[2] = [coords - centroids[1][ii] + half_widths[1] for ii,coords in enumerate(coords_split[2])]
            sf_rs_centered = sf_rs.copy()
            sf_rs_centered.coords = np.array([np.concatenate(c) for c in coords_split])
            sf_rs_centered = sf_rs_centered[:, :out_height_width[0], :out_height_width[1]]
            return sf_rs_centered.todense().astype(np.float32)
            
        ## Transform
        print(f"Staring: Creating centered ROI images from spatial footprints...") if self._verbose else None
        self.ROI_images = [sf_to_centeredROIs(sf, centroids.T) for sf, centroids in zip(self.spatialFootprints, self.centroids)]
        print(f"Completed: Created ROI images.") if self._verbose else None

        return self.ROI_images
        

    def __repr__(self):
        ## Check which attributes are set
        attr_to_print = {key: val for key,val in self.__dict__.items() if key in [
            'um_per_pixel', 
            'n_sessions', 
            'n_classes', 
            'n_class_labels', 
            'n_class_labels_total', 
            'unique_class_labels',
            'n_roi',
            'n_roi_total',
            'FOV_height',
            'FOV_width',
        ]}
        return f"Data_roicat object: {attr_to_print}."

    def import_from_dict(
        self,
        dict_load: Dict[str, Any],
    ) -> None:
        """
        Imports attributes from a dictionary. This is useful if a dictionary
        that can be serialized was saved.

        Args:
            dict_load (Dict[str, Any]): 
                Dictionary containing attributes to load.

        Note: 
            This method does not return anything. It modifies the object state
            by importing attributes from the provided dictionary.
        """
        ## Go through each important attribute in Data_roicat and look for it in dict_load
        methods = {
            self.set_ROI_images: ['ROI_images', 'um_per_pixel'],
            self.set_spatialFootprints: ['spatialFootprints', 'um_per_pixel'],
            self.set_FOV_images: ['FOV_images'],
            self.set_class_labels: ['class_labels_raw'],
        }

        methodKeys_all = list(set(sum(list(methods.values()), [])))
        
        ## Set other attributes
        for key, val in dict_load.items():
            if key not in methodKeys_all:
                setattr(self, key, val)

        ## Set attributes using methods
        for method, methodKeys in methods.items():
            if all([key in dict_load for key in methodKeys]):
                method(**{key: dict_load[key] for key in methodKeys})
            else:
                print(f"RH WARNING: Could not load attribute using method {method.__name__}. Keys {methodKeys} not found in dict_load.") if self._verbose else None
        

############################################################################################################################
############################## CUSTOM CLASSES FOR SUITE2P AND CAIMAN OUTPUT FILES ##########################################
############################################################################################################################

#########################################################
#################### DATA S2P ###########################
#########################################################

class Data_suite2p(Data_roicat):
    """
    Class for handling suite2p output files and data. In particular stat.npy and
    ops.npy files. Imports FOV images and spatial footprints, and prepares ROI
    images. 
    RH 2022

    Args:
        paths_statFiles (list of str or pathlib.Path):
            List of paths to the stat.npy files. Elements should be one of: str,
            pathlib.Path, list of str or list of pathlib.Path.
        paths_opsFiles (list of str or pathlib.Path, optional):
            List of paths to the ops.npy files. Elements should be one of: str,
            pathlib.Path, list of str or list of pathlib.Path. Optional. Used to
            get FOV_images, FOV_height, FOV_width, and shifts (if old matlab ops
            file).
        um_per_pixel (float):
            Resolution in micrometers per pixel of the imaging field of view.
        new_or_old_suite2p (str):
            Type of suite2p output files. Matlab=old, Python=new. Should be:
            ``'new'`` or ``'old'``.
        out_height_width (tuple of int):
            Height and width of output ROI images. Should be: *(int, int)* *(y,
            x)*.
        type_meanImg (str):
            Type of mean image to use. Should be: ``'meanImgE'`` or
            ``'meanImg'``.
        FOV_images (np.ndarray, optional):
            FOV images. Array of shape *(n_sessions, FOV_height, FOV_width)*.
            Optional.
        centroid_method (str):
            Method for calculating the centroid of an ROI. Should be:
            ``'centerOfMass'`` or ``'median'``.
        class_labels ((list of np.ndarray) or (list of str to paths) or None):
            Optional. If ``None``, class labels are not set. If list of
            np.ndarray, each element should be a 1D integer array of length
            n_roi specifying the class label for each ROI. If list of str, each
            element should be a path to a .npy file containing an array of
            length n_roi specifying the class label for each ROI.
        FOV_height_width (tuple of int, optional):
            FOV height and width. If ``None``, **paths_opsFiles** must be
            provided to get FOV height and width.
        verbose (bool):
            If ``True``, prints results from each function.
    """
    def __init__(
        self,
        paths_statFiles: Union[str, pathlib.Path, List[Union[str, pathlib.Path]]],
        paths_opsFiles: Optional[Union[str, pathlib.Path, List[Union[str, pathlib.Path]]]] = None,
        um_per_pixel: float = 1.0,
        new_or_old_suite2p: str = 'new',
        out_height_width: Tuple[int, int] = (36, 36),
        type_meanImg: str = 'meanImgE',
        FOV_images: Optional[np.ndarray] = None,
        centroid_method: str = 'centerOfMass',
        class_labels: Optional[Union[List[np.ndarray], List[str], None]] = None,
        FOV_height_width: Optional[Tuple[int, int]] = None,
        verbose: bool = True,
    ):
        """
        Initialize the Data_suite2p object.
        """

        ## Inherit from Data_roicat
        super().__init__()

        self.paths_stat = fix_paths(paths_statFiles)
        self.paths_ops = fix_paths(paths_opsFiles) if paths_opsFiles is not None else None
        self.n_sessions = len(self.paths_stat)

        self._verbose = verbose
        
        ## shifts are applied to convert the 'old' matlab version of suite2p indexing (where there is an offset and its 1-indexed)
        self.shifts = self._make_shifts(paths_ops=self.paths_ops, new_or_old_suite2p=new_or_old_suite2p)

        ## Import FOV images
        ### Assert only one of self.paths_ops, FOV_images, or FOV_height_width is provided
        assert sum([self.paths_ops is not None, FOV_images is not None, FOV_height_width is not None]) == 1, "RH ERROR: One (and only one) of self.paths_ops, FOV_images, or FOV_height_width must be provided."

        ### Import FOV images if self.paths_ops or FOV_images is provided
        if self.paths_ops is not None:
            FOV_images = self.import_FOV_images(type_meanImg=type_meanImg)
        ### Set FOV height and width if FOV_height_width is provided
        elif FOV_height_width is not None:
            assert isinstance(FOV_height_width, tuple), "RH ERROR: FOV_height_width must be a tuple of length 2."
            assert len(FOV_height_width) == 2, "RH ERROR: FOV_height_width must be a tuple of length 2."
            assert all([isinstance(x, int) for x in FOV_height_width]), "RH ERROR: FOV_height_width must be a tuple of length 2 of integers."
            self.set_FOVHeightWidth(FOV_height=FOV_height_width[0], FOV_width=FOV_height_width[1])
        self.set_FOV_images(FOV_images=FOV_images) if FOV_images is not None else None

        ## Import spatial footprints
        spatialFootprints = self.import_spatialFootprints()
        self.set_spatialFootprints(spatialFootprints=spatialFootprints, um_per_pixel=um_per_pixel)

        ## Make session_bool
        self._make_session_bool()

        ## Make spatial footprint centroids
        self._make_spatialFootprintCentroids(method=centroid_method)
        
        ## Transform spatial footprints to ROI images
        self._transform_spatialFootprints_to_ROIImages(out_height_width=out_height_width)

        ## Make class labels
        self.set_class_labels(labels=class_labels) if class_labels is not None else None


    def import_FOV_images(
        self,
        type_meanImg: str = 'meanImgE',
    ) -> List[np.ndarray]:
        """
        Imports the FOV images from ops files or user defined image arrays.

        Args:
            type_meanImg (str):
                Type of the mean image. References the key in the ops.npy file.
                Options are: \n
                * ``'meanImgE'``: Enhanced mean image. 
                * ``'meanImg'``: Mean image.
        
        Returns:
            FOV_images (List[np.ndarray]):
                List of FOV images. Length of the list is the same as
                self.paths_files. Each element is a numpy.ndarray of shape
                *(n_files, height, width)*.
        """

        print(f"Starting: Importing FOV images from ops files") if self._verbose else None
        
        assert self.paths_ops is not None, "RH ERROR: paths_ops is None. Please set paths_ops before calling this function."
        assert len(self.paths_ops) > 0, "RH ERROR: paths_ops is empty. Please set paths_ops before calling this function."
        assert all([Path(path).exists() for path in self.paths_ops]), "RH ERROR: One or more paths in paths_ops do not exist."

        FOV_images = [np.load(path, allow_pickle=True)[()][type_meanImg] for path in self.paths_ops]

        assert all([FOV_images[0].shape[0] == FOV_images[i].shape[0] for i in range(1, len(FOV_images))]), f"RH ERROR: FOV images are not all the same height. Shapes: {[FOV_image.shape for FOV_image in FOV_images]}"
        assert all([FOV_images[0].shape[1] == FOV_images[i].shape[1] for i in range(1, len(FOV_images))]), f"RH ERROR: FOV images are not all the same width. Shapes: {[FOV_image.shape for FOV_image in FOV_images]}"

        FOV_images = np.stack(FOV_images, axis=0).astype(np.float32)

        self.set_FOVHeightWidth(FOV_height=FOV_images[0].shape[0], FOV_width=FOV_images[0].shape[1])
        
        print(f"Completed: Imported {len(FOV_images)} FOV images.") if self._verbose else None
        
        return FOV_images
    
    def import_spatialFootprints(
        self,
        frame_height_width: Optional[Union[List[int], Tuple[int, int]]] = None,
        dtype: np.dtype = np.float32,
    ) -> List[scipy.sparse.csr_matrix]:
        """
        Imports and converts the spatial footprints of the ROIs in the stat
        files into images in sparse arrays.

        Generates **self.session_bool** which is a bool np.ndarray of shape
        *(n_roi, n_sessions)* indicating which session each ROI belongs to.

        Args:
            frame_height_width (Optional[Union[List[int], Tuple[int, int]]]):
                The *height* and *width* of the frame in the form *[height,
                width]*. If ``None``, ``self.import_FOV_images`` must be called
                before this method, and the frame height and width will be taken
                from the first FOV image. (Default is ``None``)
            dtype (np.dtype):
                Data type of the sparse array. (Default is ``np.float32``)

        Returns:
            (List[scipy.sparse.csr_matrix]): 
                sf (List[scipy.sparse.csr_matrix]):
                    Spatial footprints. Length of the list is the same as
                    ``self.paths_files``. Each element is a
                    scipy.sparse.csr_matrix of shape *(n_roi, frame_height *
                    frame_width)*.
        """
        print("Importing spatial footprints from stat files.") if self._verbose else None

        ## Check and fix inputs
        if frame_height_width is None:
            frame_height_width = [self.FOV_height, self.FOV_width]

        assert self.paths_stat is not None, "RH ERROR: paths_stat is None. Please set paths_stat before calling this function."
        assert len(self.paths_stat) > 0, "RH ERROR: paths_stat is empty. Please set paths_stat before calling this function."
        assert all([Path(path).exists() for path in self.paths_stat]), "RH ERROR: One or more paths in paths_stat do not exist."

        assert hasattr(self, 'shifts'), "RH ERROR: shifts is not defined. Please call ._make_shifts before calling this function."

        statFiles = [np.load(path, allow_pickle=True) for path in self.paths_stat]

        n = self.n_sessions
        spatialFootprints = [
            self._transform_statFile_to_spatialFootprints(
                frame_height_width=frame_height_width,
                stat=statFiles[ii],
                shifts=self.shifts[ii],
                dtype=dtype,
                normalize_mask=True,
            ) for ii in tqdm(range(n))]

        if self._verbose:
            print(f"Imported {len(spatialFootprints)} sessions of spatial footprints into sparse arrays.")

        return spatialFootprints
    

    def import_neuropil_masks(
        self,
        frame_height_width: Optional[Union[List[int], Tuple[int, int]]] = None,
    ) -> List[scipy.sparse.csr_matrix]:
        """
        Imports and converts the neuropil masks of the ROIs in the stat files
        into images in sparse arrays.

        Args:
            frame_height_width (Optional[Union[List[int], Tuple[int, int]]]):
                The *height* and *width* of the frame in the form *[height,
                width]*. If ``None``, the height and width will be taken from
                the FOV images. (Default is ``None``)

        Returns:
            (List[scipy.sparse.csr_matrix]): 
                neuropilMasks (List[scipy.sparse.csr_matrix]):
                    List of neuropil masks. Length of the list is the same as
                    ``self.paths_stat``. Each element is a sparse array of shape
                    *(n_roi, frame_height, frame_width)*.
        """
        print("Importing neuropil masks from stat files.") if self._verbose else None

        ## Check and fix inputs
        if frame_height_width is None:
            frame_height_width = [self.FOV_height, self.FOV_width]

        assert self.paths_stat is not None, "RH ERROR: paths_stat is None. Please set paths_stat before calling this function."
        assert len(self.paths_stat) > 0, "RH ERROR: paths_stat is empty. Please set paths_stat before calling this function."
        assert all([Path(path).exists() for path in self.paths_stat]), "RH ERROR: One or more paths in paths_stat do not exist."

        assert hasattr(self, 'shifts'), "RH ERROR: shifts is not defined. Please call ._make_shifts before calling this function."

        statFiles = [np.load(path, allow_pickle=True) for path in self.paths_stat]

        n = self.n_sessions
        neuropilMasks = [
            self._transform_statFile_to_neuropilMasks(
                frame_height_width=frame_height_width,
                stat=statFiles[ii],
                shifts=self.shifts[ii],
            ) for ii in tqdm(range(n))]
        
        if self._verbose:
            print(f"Imported {len(neuropilMasks)} sessions of neuropil masks into sparse arrays.")  

        self.neuropilMasks = neuropilMasks
        return neuropilMasks
    

    def _make_shifts(
        self, 
        paths_ops: Optional[List[str]] = None, 
        new_or_old_suite2p: str = 'new',
    ) -> List[np.ndarray]:
        """
        Helper function to make the shifts for the old suite2p indexing.

        Args:
            paths_ops (list of str, optional):
                List of paths to the ops.npy files. Default is ``None``.
            new_or_old_suite2p (str):
                Type of suite2p output files. Should be: ``'new'`` or ``'old'``.
                Default is ``'new'``.

        Returns:
            (List[np.ndarray]):
                shifts (List[np.ndarray]):
                    List of shifts. Length of the list is the same as
                    ``self.paths_files``. Each element is a numpy array of
                    shape *(2,)*.
        """        
        if paths_ops is None:
            shifts = [np.array([0,0], dtype=np.uint64)]*self.n_sessions
            return shifts

        if new_or_old_suite2p == 'old':
            shifts = [np.array([op['yrange'].min()-1, op['xrange'].min()-1], dtype=np.uint64) for op in [np.load(path, allow_pickle=True)[()] for path in paths_ops]]
        elif new_or_old_suite2p == 'new':
            shifts = [np.array([0,0], dtype=np.uint64)]*len(paths_ops)
        else:
            raise ValueError(f"RH ERROR: new_or_old_suite2p should be 'new' or 'old'. Got {new_or_old_suite2p}")
        return shifts

    @staticmethod
    def _transform_statFile_to_spatialFootprints(
        frame_height_width: Tuple[int, int], 
        stat: np.ndarray, 
        shifts: Tuple[int, int] = (0, 0), 
        dtype: Optional[np.dtype] = None, 
        normalize_mask: bool = True,
    ) -> scipy.sparse.csr_matrix:
        """
        Populates a sparse array with the spatial footprints from ROIs in a stat
        file.

        Args:
            frame_height_width (Tuple[int, int]):
                Height and width of the frame.
            stat (np.ndarray):
                Stat file containing ROIs information.
            shifts (Tuple[int, int]):
                Shifts in x and y coordinates to apply to ROIs. Default is (0,
                0).
            dtype (Optional[np.dtype]):
                Data type of the array elements. If ``None``, it will be
                inferred from the data. Default is ``None``.
            normalize_mask (bool):
                If True, normalize the mask. Default is ``True``.

        Returns:
            (scipy.sparse.csr_matrix):
                spatialFootprints (scipy.sparse.csr_matrix):
                    Sparse array of shape *(n_roi, frame_height * frame_width)*
                    containing the spatial footprints of the ROIs.
        """
        isInt = np.issubdtype(dtype, np.integer)

        rois_to_stack = []
        
        for jj, roi in enumerate(stat):
            lam = np.array(roi['lam'], ndmin=1)
            dtype = dtype if dtype is not None else lam.dtype
            if isInt:
                lam = dtype(lam / lam.sum() * np.iinfo(dtype).max) if normalize_mask else dtype(lam)
            else:
                lam = lam / lam.sum() if normalize_mask else lam
            ypix = np.array(roi['ypix'], dtype=np.uint64, ndmin=1) + shifts[0]
            xpix = np.array(roi['xpix'], dtype=np.uint64, ndmin=1) + shifts[1]
        
            tmp_roi = scipy.sparse.csr_matrix((lam, (ypix, xpix)), shape=(frame_height_width[0], frame_height_width[1]), dtype=dtype)
            rois_to_stack.append(tmp_roi.reshape(1,-1))

        return scipy.sparse.vstack(rois_to_stack).tocsr()

    @staticmethod
    def _transform_statFile_to_neuropilMasks(
        frame_height_width: Tuple[int, int], 
        stat: np.ndarray, 
        shifts: Tuple[int, int] = (0, 0)
    ) -> scipy.sparse.csr_matrix:
        """
        Populates a sparse array with the neuropil masks from ROIs in a stat
        file.

        Args:
            frame_height_width (Tuple[int, int]):
                Height and width of the frame.
            stat (np.ndarray):
                Stat file containing ROIs information.
            shifts (Tuple[int, int]):
                Shifts in x and y coordinates to apply to ROIs. Default is (0,
                0).

        Returns:
            (scipy.sparse.csr_matrix):
                neuropilMasks (scipy.sparse.csr_matrix):
                    Sparse array of shape *(n_roi, frame_height * frame_width)*
                    containing the neuropil masks of the ROIs.
        """
        
        rois_to_stack = []
        
        for jj, roi in enumerate(stat):
            lam = np.ones(len(roi['neuropil_mask']), dtype=np.bool_)
            dtype = np.bool_
            ypix, xpix = np.unravel_index(roi['neuropil_mask'], shape=(frame_height_width[0], frame_height_width[1]), order='C')
            ypix = ypix + shifts[0]
            xpix = xpix + shifts[1]
        
            tmp_roi = scipy.sparse.csr_matrix((lam, (ypix, xpix)), shape=(frame_height_width[0], frame_height_width[1]), dtype=dtype)
            rois_to_stack.append(tmp_roi.reshape(1,-1))

        return scipy.sparse.vstack(rois_to_stack).tocsr()


#########################################################
################## DATA CAIMAN ##########################
#########################################################

class Data_caiman(Data_roicat):
    """
    Class for importing data from CaImAn output files, specifically hdf5 results
    files.

    Args:
        paths_resultsFiles (List[str]):
            List of paths to the results files.
        include_discarded (bool):
            If ``True``, include ROIs that were discarded by CaImAn. Default is
            ``True``.
        um_per_pixel (float):
            Microns per pixel. Default is 1.0.
        out_height_width (List[int]):
            Output height and width. Default is [36, 36].
        centroid_method (str):
            Method for calculating the centroid of an ROI. Should be:
            ``'centerOfMass'`` or ``'median'``.
        verbose (bool):
            If ``True``, print statements will be printed. Default is ``True``.
        class_labels (str, optional):
            Class labels. Default is ``None``.
    """
    def __init__(
        self,
        paths_resultsFiles: List[str],
        include_discarded: bool = True,
        um_per_pixel: float = 1.0,
        out_height_width: List[int] = [36,36],        
        centroid_method: str = 'median',
        verbose: bool = True,
        class_labels: Optional[str] = None,
    ) -> None:
        
        ## Inherit from Data_roicat
        super().__init__()

        self.paths_resultsFiles = fix_paths(paths_resultsFiles)
        self.n_sessions = len(self.paths_resultsFiles)
        # self._include_discarded = include_discarded
        self._verbose = verbose

        # 1. import_caiman_results
        # # self.spatialFootprints
        # ?? # self.overall_caiman_labels
        # ?? # self.cnn_caiman_preds
        # # self.n_roi
        # # self.n_roi_total
        
        spatialFootprints = [self.import_spatialFootprints(path, include_discarded=include_discarded) for path in self.paths_resultsFiles]
        self.set_spatialFootprints(spatialFootprints=spatialFootprints, um_per_pixel=um_per_pixel)

        overall_caimanLabels = [self.import_overall_caiman_labels(path, include_discarded=include_discarded) for path in self.paths_resultsFiles]
        self.set_caimanLabels(overall_caimanLabels=overall_caimanLabels)

        cnn_caimanPreds = [self.import_cnn_caiman_preds(path, include_discarded=include_discarded) for path in self.paths_resultsFiles]
        self.set_caimanPreds(cnn_caimanPreds=cnn_caimanPreds) if cnn_caimanPreds[0] is not None else None

        FOV_images = self.import_FOV_images(self.paths_resultsFiles)
        self.set_FOV_images(FOV_images=FOV_images)
        self._make_spatialFootprintCentroids(method=centroid_method)
        self._make_session_bool()
        self._transform_spatialFootprints_to_ROIImages(out_height_width=out_height_width)
        self.set_class_labels(labels=class_labels) if class_labels is not None else None

    def set_caimanLabels(self, overall_caimanLabels: List[List[bool]]) -> None:
        """
        Sets the CaImAn labels.

        Args:
            overall_caimanLabels (List[List[bool]]):
                List of lists of CaImAn labels.
                The outer list corresponds to sessions, and the inner list corresponds to ROIs.
        """
        assert len(overall_caimanLabels) == self.n_sessions
        print('kept labels', sum(overall_caimanLabels).sum(), len(sum(overall_caimanLabels))-sum(overall_caimanLabels).sum())
        assert all([len(overall_caimanLabels[i]) == self.n_roi[i] for i in range(self.n_sessions)])
        self.cnn_caimanLabels = overall_caimanLabels

    def set_caimanPreds(self, cnn_caimanPreds: List[List[bool]]) -> None:
        """
        Sets the CNN-CaImAn predictions.

        Args:
            cnn_caimanPreds (List[List[bool]]):
                List of lists of CNN-CaImAn predictions. The outer list
                corresponds to sessions, and the inner list corresponds to ROIs.
        """
        assert len(cnn_caimanPreds) == self.n_sessions, f"{len(cnn_caimanPreds)} != {self.n_sessions}"
        assert all([len(cnn_caimanPreds[i]) == self.n_roi[i] for i in range(self.n_sessions)]), f"{[len(cnn_caimanPreds[i]) for i in range(self.n_sessions)]} != {[self.n_roi[i] for i in range(self.n_sessions)]}"
        self.cnn_caimanPreds = cnn_caimanPreds

    def import_spatialFootprints(
        self, 
        path_resultsFile: Union[str, pathlib.Path], 
        include_discarded: bool = True
    ) -> scipy.sparse.csr_matrix:
        """
        Imports the spatial footprints from the results file. Note that CaImAn's
        ``data['estimates']['A']`` is similar to ``self.spatialFootprints``, but
        uses 'F' order. This function converts this into 'C' order to form
        ``self.spatialFootprints``.

        Args:
            path_resultsFile (Union[str, pathlib.Path]):
                Path to a single results file.
            
            include_discarded (bool):
                If ``True``, include ROIs that were discarded by CaImAn. Default
                is ``True``.

        Returns:
            (scipy.sparse.csr_matrix):
                Spatial footprints (scipy.sparse.csr_matrix):
                    Spatial footprints.
        """
        with helpers.h5_load(path_resultsFile, return_dict=False) as data:
            FOV_height, FOV_width = data['estimates']['dims'][()]
            
            ## initialize the estimates.A matrix, which is a 'Fortran' indexed version of sf. Note the flipped dimensions for shape.
            sf_included = scipy.sparse.csr_matrix((data['estimates']['A']['data'][()], data['estimates']['A']['indices'], data['estimates']['A']['indptr'][()]), shape=data['estimates']['A']['shape'][()][::-1])
            print('kept ROIs',sf_included.shape)
            if include_discarded:
                try:
                    discarded = data['estimates']['discarded_components'][()]
                    sf_discarded = scipy.sparse.csr_matrix((discarded['A']['data'], discarded['A']['indices'], discarded['A']['indptr']), shape=discarded['A']['shape'][::-1])
                    print('dropped ROIs',sf_discarded.shape)
                    sf_F = scipy.sparse.vstack([sf_included, sf_discarded])
                except:
                    sf_F = sf_included
            else:
                sf_F = sf_included

            ## reshape sf_F (which is in Fortran flattened format) into C flattened format
            sf = sparse.COO(sf_F).reshape((sf_F.shape[0], FOV_width, FOV_height)).transpose((0,2,1)).reshape((sf_F.shape[0], FOV_width*FOV_height)).tocsr()
            
            return sf

    def import_overall_caiman_labels(
        self, 
        path_resultsFile: Union[str, pathlib.Path], 
        include_discarded: bool = True
    ) -> np.ndarray:
        """
        Imports the overall CaImAn labels from the results file.
        
        Args:
            path_resultsFile (Union[str, pathlib.Path]):
                Path to a single results file.
            
            include_discarded (bool):
                If ``True``, include ROIs that were discarded by CaImAn. Default
                is ``True``.

        Returns:
            (np.ndarray):
                labels (np.ndarray):
                    Overall CaImAn labels.
        """

        with helpers.h5_load(path_resultsFile, return_dict=False) as data:
            labels_included = np.ones(data['estimates']['A']['indptr'][()].shape[0] - 1)
            if include_discarded:
                try:
                    discarded = data['estimates']['discarded_components'][()]
                    labels_discarded = np.zeros(discarded['A']['indptr'].shape[0] - 1)
                    labels = np.hstack([labels_included, labels_discarded])
                except:
                    print('no discarded components for labels')
                    labels = labels_included
            else:
                labels = labels_included

            return labels

    def import_cnn_caiman_preds(
        self, 
        path_resultsFile: Union[str, pathlib.Path], 
        include_discarded: bool = True,
    ) -> Union[np.ndarray, None]:
        """
        Imports the CNN-based CaImAn prediction probabilities from the given
        file.

        Args:
            path_resultsFile (Union[str, pathlib.Path]): 
                Path to a single results file. Can be either a string or a
                pathlib.Path object.
            include_discarded (bool): 
                If set to True, the function will include ROIs that were
                discarded by CaImAn. By default, this is set to True.

        Returns:
            (np.ndarray):
                preds (np.ndarray):
                    CNN-based CaImAn prediction probabilities.
        """

        with helpers.h5_load(path_resultsFile, return_dict=False) as data:
            preds_included = data['estimates']['cnn_preds'][()]
            if preds_included == b'NoneType':
                warnings.warn('No CNN preds found in results file')
                return None
            
            if include_discarded:
                try:
                    discarded = data['estimates']['discarded_components'][()]
                    preds_discarded = discarded['cnn_preds']
                    preds = np.hstack([preds_included, preds_discarded])
                except:
                    print('no discarded components for cnn_preds')
                    preds = preds_included
            else:
                preds = preds_included
            
            return preds

    def import_ROI_centeredImages(self, out_height_width: List[int] = [36,36]) -> np.ndarray:
        """
        Imports the ROI centered images from the CaImAn results files.

        Args:
            out_height_width (List[int]): 
                Height and width of the output images. Default is *[36,36]*.

        Returns:
            (np.ndarray):
                ROI centered images (np.ndarray):
                    ROI centered images. Shape is *(nROIs, out_height_width[0],
                    out_height_width[1])*.
        """
        def sf_to_centeredROIs(sf, centroids, out_height_width=36):
            out_height_width = np.array([36,36])
            half_widths = np.ceil(out_height_width/2).astype(int)
            sf_rs = sparse.COO(sf).reshape((sf.shape[0], self.FOV_height, self.FOV_width))
            
            coords_diff = np.diff(sf_rs.coords[0])
            assert np.all(coords_diff < 1.01) and np.all(coords_diff > -0.01), \
                "RH ERROR: sparse.COO object has strange .coords attribute. sf_rs.coords[0] should all be 0 or 1. An ROI is possibly all zeros."
            
            idx_split = (sf_rs>0).astype(np.bool_).sum((1,2)).todense().cumsum()[:-1]
            coords_split = [np.split(sf_rs.coords[ii], idx_split) for ii in [0,1,2]]
            coords_split[1] = [coords - centroids[0][ii] + half_widths[0] for ii,coords in enumerate(coords_split[1])]
            coords_split[2] = [coords - centroids[1][ii] + half_widths[1] for ii,coords in enumerate(coords_split[2])]
            sf_rs_centered = sf_rs.copy()
            sf_rs_centered.coords = np.array([np.concatenate(c) for c in coords_split])
            sf_rs_centered = sf_rs_centered[:, :out_height_width[0], :out_height_width[1]]
            return sf_rs_centered.todense()

        print(f"Computing ROI centered images from spatial footprints") if self._verbose else None
        ROI_images = [sf_to_centeredROIs(sf, centroids.T, out_height_width=out_height_width) for sf, centroids in zip(self.spatialFootprints, self.centroids)]

        return ROI_images

    def import_FOV_images(
        self,
        paths_resultsFiles: Optional[List] = None,
        images: Optional[List] = None,
    ) -> List[np.ndarray]:
        """
        Imports the FOV images from the CaImAn results files.

        Args:
            paths_resultsFiles (Optional[List]):
                List of paths to CaImAn results files. If not provided, will use
                the paths stored in the class instance.
            images (Optional[List]):
                List of FOV images. If None, the function will import the
                `estimates.b` image from the paths specified in
                `paths_resultsFiles`.

        Returns:
            List[np.ndarray]:
                FOV images (np.ndarray):
                    FOV images. Shape is *(nROIs, FOV_height, FOV_width)*.
        """
    
        def _import_FOV_image(path_resultsFile):
            with helpers.h5_load(path_resultsFile, return_dict=False) as data:
                FOV_height, FOV_width = data['estimates']['dims'][()]
                FOV_image = data['estimates']['b'][()][:,0].reshape(FOV_height, FOV_width, order='F')
                return FOV_image.astype(np.float32)

        if images is not None:
            if self._verbose:
                print("Using provided images for FOV_images.")
            FOV_images = images
        else:
            if paths_resultsFiles is None:
                paths_resultsFiles = self.paths_resultsFiles
            FOV_images = np.stack([_import_FOV_image(p) for p in paths_resultsFiles])
            FOV_images = FOV_images - FOV_images.min(axis=(1,2), keepdims=True)
            FOV_images = FOV_images / FOV_images.mean(axis=(1,2), keepdims=True)

        return FOV_images
    

############################################
############ DATA ROIEXTRACTORS ############
############################################

class Data_roiextractors(Data_roicat):
    """
    A class for importing all roiextractors supported data. This class will loop
    through each object and ingest data for roicat. 
    RH, JB 2023
    
    Args:
        segmentation_extractor_objects (list): 
            List of segmentation extractor objects. All objects must be of the
            same type.
        um_per_pixel (float, optional): 
            The resolution, specified as 'micrometers per pixel' of the imaging
            field of view. Defaults to 1.0.
        out_height_width (tuple of int, optional): 
            The height and width of output ROI images, specified as *(y, x)*.
            Defaults to *[36,36]*.
        FOV_image_name (str, optional): 
            If provided, this key will be used to extract the FOV image from the
            segmentation object's self.get_images_dict() method. If None, the
            function will attempt to pull out a mean image. Defaults to None.
        fallback_FOV_height_width (tuple of int, optional): 
            If the FOV images cannot be imported automatically, this will be
            used as the FOV height and width. Otherwise, FOV height and width
            are set from the first object in the list. Defaults to *[512,512]*.
        centroid_method (str, optional): 
            The method for calculating the centroid of the ROI. This should be
            either ``'centerOfMass'`` or ``'median'``. Defaults to
            ``'centerOfMass'``.
        class_labels (list, optional): 
            A list of class labels for each object. Defaults to ``None``.
        verbose (bool, optional): 
            If set to True, print statements will be displayed. Defaults to
            ``True``.
    """
    def __init__(
            self,
            segmentation_extractor_objects: List[Any],
            um_per_pixel: float = 1.0,
            out_height_width: Tuple[int, int] = (36,36),
            FOV_image_name: Optional[str] = None,
            fallback_FOV_height_width: Tuple[int, int] = (512,512),
            centroid_method: str = 'centerOfMass',
            class_labels: Optional[List[Any]] = None,
            verbose: bool = True,
    ):
        """
        Initializer for the `Data_roiextractors` class.
        """
        import roiextractors
        
        ## Inherit from Data_roicat
        super().__init__()

        self._verbose = verbose

        types_roiextractors = {
            'caiman': roiextractors.extractors.caiman.caimansegmentationextractor.CaimanSegmentationExtractor,
            'cnmf': roiextractors.extractors.schnitzerextractor.cnmfesegmentationextractor.CnmfeSegmentationExtractor,
            'extract': roiextractors.extractors.schnitzerextractor.extractsegmentationextractor.NewExtractSegmentationExtractor,
            'nwb': roiextractors.extractors.nwbextractors.nwbextractors.NwbSegmentationExtractor,
            'suite2p': roiextractors.extractors.suite2p.suite2psegmentationextractor.Suite2pSegmentationExtractor,
        }
        types_roiextractors_inv = {val: key for key,val in types_roiextractors.items()}

        ## if the input segmentation extractor objects are not a list, make it one
        self.segmentation_extractor_objects = [segmentation_extractor_objects] if isinstance(segmentation_extractor_objects, list) == False else segmentation_extractor_objects

        self.class_roiextractors = type(self.segmentation_extractor_objects[0])

        ## assert all segmentation extractor objects are the same type
        assert all([self.class_roiextractors == type(obj) for obj in self.segmentation_extractor_objects]), 'All segmentation extractor objects must be of the same type.'
        ## assert that the type of the segmentation extractor object is supported
        assert self.class_roiextractors in types_roiextractors_inv.keys(), f'Segmentation extractor object type {type(self.segmentation_extractor_objects[0])} not supported. Please use one of the following: {types_roiextractors_inv.keys()}'
        ## set the type of the segmentation extractor object
        self.type_roiextractors = types_roiextractors_inv[self.class_roiextractors]
        self.class_roiextractors

        ## set spatial footprints
        self.set_spatialFootprints(
            spatialFootprints=[self._make_spatialFootprints(obj) for obj in self.segmentation_extractor_objects],
            um_per_pixel=um_per_pixel
        )

        # Get the FOV images from the segmentation extractor object
        types_FOV_images = {
            'caiman': 'mean',
            'cnmf': 'correlation',
            'extract': 'summary_image',
            'nwb': 'mean',
            'suite2p': 'mean',
        }
        type_FOV_image = types_FOV_images[self.type_roiextractors] if FOV_image_name is None else FOV_image_name

        try:
            if type_FOV_image not in self.segmentation_extractor_objects[0].get_images_dict().keys():
                warnings.warn(f'FOV image type {type_FOV_image} not found in segmentation extractor object. Please set FOV images manually using self.set_FOV_images()')
            FOV_images = [obj.get_images_dict()[type_FOV_image] for obj in self.segmentation_extractor_objects]
            self.set_FOV_images(FOV_images=FOV_images)
        except Exception as e:
            warnings.warn(f'Failed to retrieve and/or set FOV images. Please set FOV images manually using self.set_FOV_images(). Error: {e}')
            self.set_FOVHeightWidth(FOV_height=fallback_FOV_height_width[0], FOV_width=fallback_FOV_height_width[1])

        ## Make session_bool
        self._make_session_bool()

        ## Make spatial footprint centroids
        self._make_spatialFootprintCentroids(method=centroid_method)

        ## Transform spatial footprints to ROI images
        self._transform_spatialFootprints_to_ROIImages(out_height_width=out_height_width)

        ## Make class labels
        self.set_class_labels(labels=class_labels) if class_labels is not None else None

    def _make_spatialFootprints(self, segObj: Any) -> scipy.sparse.csr_matrix:
        """
        Creates spatial footprints from the given roiextractors segmentation
        object.

        Args:
            segObj (Any): 
                An roiextractors segmentation object.

        Returns:
            (scipy.sparse.csr.csr_matrix):
                sf (scipy.sparse.csr.csr_matrix): 
                    A scipy CSR (Compressed Sparse Row) matrix that represents
                    the spatial footprints.
        """

        roi_pixel_masks = segObj.get_roi_pixel_masks()

        data = [r[:,2] for r in roi_pixel_masks]
        ij_all = [r[:,:2].astype(np.int64) for r in roi_pixel_masks]

        sf = scipy.sparse.vstack(
            [
                scipy.sparse.coo_matrix(
                    (d, (ij[:,0], ij[:,1])),
                    shape=tuple(segObj.get_image_size())
                ).reshape(1, -1) for d,ij in zip(data, ij_all)
            ]
        ).tocsr()
        return sf



####################################
######### HELPER FUNCTIONS #########
####################################

def fix_paths(paths: Union[List[Union[str, pathlib.Path]], str, pathlib.Path]) -> List[str]:
    """
    Ensures the input paths are a list of strings.

    Args:
        paths (Union[List[Union[str, pathlib.Path]], str, pathlib.Path]):
            The input can be either a list of strings or pathlib.Path objects,
            or a single string or pathlib.Path object.
            
    Returns:
        List[str]: 
            A list of strings representing the paths.

    Raises:
        TypeError: 
            If the input isn't a list of str or pathlib.Path objects, a single
            str, or a pathlib.Path object.
    """
    
    if isinstance(paths, (str, pathlib.Path)):
        paths_files = [Path(paths).resolve()]
    elif isinstance(paths[0], (str, pathlib.Path)):
        paths_files = [Path(path).resolve() for path in paths]
    else:
        raise TypeError("path_files must be a list of str or list of pathlib.Path or a str or pathlib.Path")
    return [str(p) for p in paths_files]


def make_smaller_data(
    data: Data_roicat,
    n_ROIs: Optional[int] = 300,
    n_sessions: Optional[int] = 10,
    bounds_x: Tuple[int, int] = (200,400),
    bounds_y: Tuple[int, int] = (200,400),
) -> Data_roicat:
    """
    Reduces the size of a Data_roicat object by limiting the number of regions
    of interest (ROIs) and sessions, and adjusting the bounds on the x and y
    axes. This function is useful for making test datasets.

    Args:
        data (Data_roicat): 
            The input data object of the ``Data_roicat`` type.
        n_ROIs (Optional[int]): 
            The number of regions of interest to include in the output data. If
            ``None``, all ROIs will be included. 
        n_sessions (Optional[int]): 
            The number of sessions to include in the output data. If ``None``,
            all sessions will be included.
        bounds_x (Tuple[int, int]): 
            The x-axis bounds for the output data. The bounds should be a tuple
            of two integers.
        bounds_y (Tuple[int, int]): 
            The y-axis bounds for the output data. The bounds should be a tuple
            of two integers.

    Returns:
        (Data_roicat): 
            data_out (Data_roicat): 
                The output data, which is a reduced version of the input data according to the specified parameters.
    """
    import sparse
    data_out = copy.deepcopy(data)

    n_sessions = min(n_sessions, len(data_out.spatialFootprints)) if n_sessions is not None else len(data_out.spatialFootprints)

    d_height = data.FOV_height
    d_width = data.FOV_width    
    d_n_ROIs = [sf.shape[0] for sf in data.spatialFootprints[:n_sessions]]

    data_out.set_FOV_images(FOV_images=[im[bounds_y[0]:bounds_y[1], bounds_x[0]:bounds_x[1]] \
        for im in data.FOV_images[:n_sessions]])
    n_ROIs_per_sesh = [min(n_ROIs, n) for n in d_n_ROIs] if n_ROIs is not None else d_n_ROIs

    frame = np.zeros((d_height, d_width), dtype=np.bool_)
    frame[bounds_y[0]:bounds_y[1], bounds_x[0]:bounds_x[1]] = True
    frame_flat = frame.reshape(-1)

    sf_tmp = [sf[:n] for sf, n in zip(data_out.spatialFootprints[:n_sessions], n_ROIs_per_sesh)]
    good_rois = [np.array((sf.multiply(frame_flat[None,:])).sum(1) > 0).squeeze() \
        for sf in sf_tmp]

    sf_tmp = [sf[g,:] for sf, g in zip(sf_tmp, good_rois)]
    data_out.set_spatialFootprints(
        spatialFootprints=[sparse.COO(s).reshape(
                shape=(s.shape[0], data.FOV_height, data.FOV_width)
            )[:, bounds_y[0]:bounds_y[1], :][:, :, bounds_x[0]:bounds_x[1]].reshape(shape=(s.shape[0], -1)).tocsr() \
            for s in sf_tmp
        ],
        um_per_pixel=data_out.um_per_pixel,
    )

    data_out._make_spatialFootprintCentroids()
    data_out._transform_spatialFootprints_to_ROIImages()
    data_out._make_session_bool()

    return data_out