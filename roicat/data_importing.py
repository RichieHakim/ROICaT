import pathlib
from pathlib import Path
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
import scipy.sparse
import sparse

from . import helpers


############################################################################################################################
####################################### SUPER CLASS FOR ALL DATA OBJECTS ###################################################
############################################################################################################################

class Data_roicat:
    """
    Super class for all data objects.
    Incase you want to make a custom data object,
     you can use this class as a template to fill
     in the required attributes.
    RH 2022
    """
    def __init__(
        self,
        verbose=True,
    ):       
        # self.FOV_images = None
        # self.FOV_height = None
        # self.FOV_width = None
        # self.spatialFootprints = None
        # self.ROI_images = None
        # self.n_roi = None
        # self.n_roi_total = None
        # self.n_sessions = None
        # self.centroids = None
        # self.um_per_pixel = None
        self._verbose = verbose
    
    #########################################################
    ################# CLASSIFICATION ########################
    #########################################################

    def set_ROI_images(
        self,
        ROI_images: list,
        um_per_pixel: float=None,
    ):
        """
        Imports ROI images into the class.
        Images are expected to be formated as a list of 
         numpy arrays. Each element is an imaging session.
         Each element is a numpy array of shape 
         (n_roi, FOV_height, FOV_width).
        This function will set the attributes:
            self.ROI_images, self.n_roi, self.n_roi_total,
             self.n_sessions.
        If any of these attributes are already set, they will
         check to make sure the new values are the same.

        Args:
            ROI_images (list of np.ndarray):
                List of numpy arrays of shape (n_roi, FOV_height, FOV_width).
            um_per_pixel (float):
                The number of microns per pixel. This is used to
                 resize the images to a common size.
        """

        ## Warn if no um_per_pixel is provided
        if um_per_pixel is None:
            ## Check if it is already set
            if hasattr(self, 'um_per_pixel'):
                um_per_pixel = self.um_per_pixel
            print("RH WARNING: No um_per_pixel provided. We recommend making an educated guess. Assuming 1.0 um per pixel. This will affect the embedding results.")
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
        n_roi_total = sum(n_roi)

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
        class_labels,
        n_classes=None,
    ):
        """
        Imports class labels into the class.
        Class labels are expected to be formated as a list of 
         numpy arrays. Each element is an imaging session and
         is associated with the n-th element of the self.ROI_images
         list.
        Each element is a numpy array of shape (n_roi,).
        Sets the attributes:
            self.class_labels, self.n_classes, self.n_class_labels,
            self.n_class_labels_total, self.unique_class_labels.
        If any of these attributes are already set, they will
         check to make sure the new values are the same.

        Args:
            class_labels (list of np.ndarray):
                List of numpy arrays of shape (n_roi,).
                Each element is an imaging session and is associated
                 with the n-th element of the self.ROI_images list.
                Label values will be 'squeezed' to remove non-contiguous
                 integer values: [2, 4, 6, 7] -> [0, 1, 2, 3].
            n_classes (int):
                Optional.
                Number of classes. If not provided, will be inferred
                 from the class labels.
        """

        print(f"Starting: Importing class labels") if self._verbose else None

        ## Check the validity of the input
        if isinstance(class_labels, np.ndarray):
            print("RH WARNING: class_labels is a numpy array. Assuming n_sessions==1 and wrapping array in a list.")
            class_labels = [class_labels]
        assert isinstance(class_labels, list), f"class_labels should be a list. It is a {type(class_labels)}"
        assert all([isinstance(lbls, np.ndarray) for lbls in class_labels]), f"class_labels should be a list of 1-Dnumpy arrays. First element of list is of type {type(class_labels[0])}"
        assert all([lbls.ndim==1 for lbls in class_labels]), f"class_labels should be a list of 1-D numpy arrays. First element of list is of shape {class_labels[0].shape}"
        assert all([lbls.dtype==np.int for lbls in class_labels]), f"class_labels should be a list of 1-D numpy arrays of dtype np.int. First element of list is of dtype {class_labels[0].dtype}"
        assert all([np.all(lbls>=0) for lbls in class_labels]), f"All class labels should be non-negative. Found negative values."

        ## Define some variables
        n_sessions = len(class_labels)
        class_labels_cat = np.concatenate(class_labels)
        class_labels_cat_squeezeInt = helpers.squeeze_integers(class_labels_cat)
        unique_class_labels = np.unique(class_labels_cat)
        if n_classes is not None:
            assert len(unique_class_labels) <= n_classes, f"RH ERROR: User provided n_classes={n_classes} but there are {len(unique_class_labels)} unique class labels in the provided class_labels." if self._verbose else None
        else:
            n_classes = len(unique_class_labels)
        n_class_labels = [lbls.shape[0] for lbls in class_labels]
        n_class_labels_total = sum(n_class_labels)
        class_labels_squeezeInt = [class_labels_cat_squeezeInt[sum(n_class_labels[:ii]):sum(n_class_labels[:ii+1])] for ii in range(n_sessions)]

        ## Check that attributes match if they already exist as an attribute
        if hasattr(self, 'n_sessions'):
            assert self.n_sessions == n_sessions, f"n_sessions is already set to {self.n_sessions} but new value is {n_sessions}"
        if hasattr(self, 'n_classes'):
            assert self.n_classes == n_classes, f"n_classes is already set to {self.n_classes} but new value is {n_classes}"
        if hasattr(self, 'n_class_labels'):
            assert self.n_class_labels == n_class_labels, f"n_class_labels is already set to {self.n_class_labels} but new value is {n_class_labels}"
        if hasattr(self, 'n_class_labels_total'):
            assert self.n_class_labels_total == n_class_labels_total, f"n_class_labels_total is already set to {self.n_class_labels_total} but new value is {n_class_labels_total}"
        if hasattr(self, 'unique_class_labels'):
            assert np.array_equal(self.unique_class_labels, unique_class_labels), f"unique_class_labels is already set to {self.unique_class_labels} but new value is {unique_class_labels}"

        ## Set attributes
        self.class_labels = class_labels_squeezeInt
        self.n_classes = n_classes
        self.n_class_labels = n_class_labels
        self.n_class_labels_total = n_class_labels_total
        self.unique_class_labels = unique_class_labels

        ## Check if label data shapes match ROI_image data shapes
        self._checkValidity_classLabels_vs_ROIImages()

        print(f"Completed: Imported labels for {n_sessions} sessions. Each session has {n_class_labels} class labels. Total number of class labels is {n_class_labels_total}.") if self._verbose else None

    def _check_um_per_pixel(self, um_per_pixel):
        ### Check um_per_pixel
        assert isinstance(um_per_pixel, (int, float)), f"um_per_pixel should be a float. It is a {type(um_per_pixel)}"
        assert um_per_pixel > 0, f"um_per_pixel should be a positive number. It is {um_per_pixel}"


    
    def _checkValidity_classLabels_vs_ROIImages(self, verbose=None):
        """
        Checks that the class labels and the ROI images have the same
         number of sessions and the same number of ROIs in each session.
        """
        if verbose is None:
            verbose = self._verbose

        ## Check if class_labels and ROI_images exist
        if not (hasattr(self, 'class_labels') and hasattr(self, 'ROI_images')):
            print("Cannot check validity of class_labels and ROI_images because one or both do not exist as attributes.") if verbose else None
            return False
        ## Check num sessions
        n_sessions_classLabels = len(self.class_labels)
        n_sessions_ROIImages = len(self.ROI_images)
        assert n_sessions_classLabels == n_sessions_ROIImages, f"RH ERROR: Number of sessions (list elements) in class_labels ({n_sessions_classLabels}) does not match number of sessions (list elements) in ROI_images ({n_sessions_ROIImages})."
        ## Check num ROIs
        n_ROIs_classLabels = [lbls.shape[0] for lbls in self.class_labels]
        n_ROIs_ROIImages = [img.shape[0] for img in self.ROI_images]
        assert all([l == r for l, r in zip(n_ROIs_classLabels, n_ROIs_ROIImages)]), f"RH ERROR: Number of ROIs in each session in class_labels ({n_ROIs_classLabels}) does not match number of ROIs in each session in ROI_images ({n_ROIs_ROIImages})."
        print(f"Labels and ROI Images match in shapes: Class labels and ROI images have the same number of sessions and the same number of ROIs in each session.") if verbose else None
        return True
    
    #########################################################
    #################### TRACKING ###########################
    #########################################################

    def set_spatialFootprints(
        self,
        spatialFootprints: list,
        um_per_pixel: float=None,
    ):
        """
        Sets the spatialFootprints attribute.

        Args:
            spatialFootprints (list):
                List of scipy.sparse.csr_matrix objects, one for
                 each session. Each matrix should have shape
                 (n_ROIs, FOV_height * FOV_width). Reshaping should
                 be done with 'C' indexing (standard).
            um_per_pixel (float):
                The number of microns per pixel. This is used to
                 resize the images to a common size.
        """
        ## Warn if no um_per_pixel is provided
        if um_per_pixel is None:
            ## Check if it is already set
            if hasattr(self, 'um_per_pixel'):
                um_per_pixel = self.um_per_pixel
            print("RH WARNING: No um_per_pixel provided. We recommend making an educated guess. Assuming 1.0 um per pixel. This will affect the embedding results.")
            um_per_pixel = 1.0

        ## Check inputs
        assert isinstance(spatialFootprints, list), f"RH ERROR: spatialFootprints must be a list."
        assert all([scipy.sparse.issparse(sf) for sf in spatialFootprints]), f"RH ERROR: All elements in spatialFootprints must be scipy.sparse.csr_matrix objects."
        if spatialFootprints[0].format != 'csr':
            spatialFootprints = [sf.tocsr() for sf in spatialFootprints]
        assert all([isinstance(sfp, scipy.sparse.csr_matrix) for sfp in spatialFootprints]), f"RH ERROR: All elements in spatialFootprints must be scipy.sparse.csr_matrix objects."

        self._check_um_per_pixel(um_per_pixel)
        um_per_pixel = float(um_per_pixel)

        ## Get some variables
        n_sessions = len(spatialFootprints)
        n_roi = [sf.shape[0] for sf in spatialFootprints]
        n_roi_total = np.sum(n_roi)

        ## Check that attributes match if they already exist as an attribute
        if hasattr(self, 'n_sessions'):
            assert self.n_sessions == n_sessions, f"n_sessions is already set to {self.n_sessions} but new value is {n_sessions}"
        if hasattr(self, 'n_roi'):
            assert self.n_roi == n_roi, f"n_roi is already set to {self.n_roi} but new value is {n_roi}"
        if hasattr(self, 'n_roi_total'):
            assert self.n_roi_total == n_roi_total, f"n_roi_total is already set to {self.n_roi_total} but new value is {n_roi_total}"

        ## Set attributes
        self.spatialFootprints = spatialFootprints
        self.um_per_pixel = um_per_pixel
        self.n_sessions = n_sessions
        self.n_roi = n_roi
        self.n_roi_total = n_roi_total
        print(f"Completed: Set spatialFootprints for {len(spatialFootprints)} sessions successfully.") if self._verbose else None


    def set_FOV_images(
        self,
        FOV_images: list,
    ):
        """
        Sets the FOV_images attribute.

        Args:
            FOV_images (list):
                List of 2D numpy arrays, one for each session.
                 Each array should have shape (FOV_height, FOV_width).
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
        self.FOV_images = FOV_images
        self.FOV_height = FOV_images[0].shape[0]
        self.FOV_width = FOV_images[0].shape[1]

        ## Get some variables
        n_sessions = len(FOV_images)

        ## Check that attributes match if they already exist as an attribute
        if hasattr(self, 'n_sessions'):
            assert self.n_sessions == n_sessions, f"n_sessions is already set to {self.n_sessions} but new value is {n_sessions}"

        print(f"Completed: Set FOV_images for {len(FOV_images)} sessions successfully.") if self._verbose else None


    def _checkValidity_spatialFootprints_and_FOVImages(self, verbose=None):
        """
        Checks that spatialFootprints and FOV_images are compatible.
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

    def check_completeness(self, verbose=True):
        """
        Checks which pipelines the data object is capable of running
         given the attributes that have been set.
        """
        completeness = {}
        ## Check classification inference:
        ### ROI_images, um_per_pixel
        if hasattr(self, 'ROI_images') and hasattr(self, 'um_per_pixel'):
            completeness['classification_inference'] = True
        else:
            print(f"RH WARNING: Classification-Inference incomplete because following attributes are missing: {[key for key in ['ROI_images', 'um_per_pixel'] if not hasattr(self, key)]}") if verbose else None
            completeness['classification_inference'] = False
        ## Check classification training:
        ### ROI_images, um_per_pixel, class_labels
        if hasattr(self, 'ROI_images') and hasattr(self, 'um_per_pixel') and hasattr(self, 'class_labels'):
            completeness['classification_training'] = True
        else:
            print(f"RH WARNING: Classification-Training incomplete because following attributes are missing: {[key for key in ['ROI_images', 'um_per_pixel', 'class_labels'] if not hasattr(self, key)]}") if verbose else None
            completeness['classification_training'] = False
        ## Check tracking:
        ### um_per_pixel, spatialFootprints, FOV_images
        if hasattr(self, 'ROI_images') and hasattr(self, 'um_per_pixel') and hasattr(self, 'spatialFootprints') and hasattr(self, 'FOV_images'):
            completeness['tracking'] = True
        else:
            print(f"RH WARNING: Tracking incomplete because following attributes are missing: {[key for key in ['ROI_images', 'um_per_pixel', 'spatialFootprints', 'FOV_images'] if not hasattr(self, key)]}") if verbose else None
            completeness['tracking'] = False

        self._checkValidity_classLabels_vs_ROIImages(verbose=verbose)
        self._checkValidity_spatialFootprints_and_FOVImages(verbose=verbose)

        ## Print completeness
        print(f"Data_roicat object completeness: {completeness}") if verbose else None
        return completeness


    def _make_sessionID_concat(self):
        """
        Creates a boolean array of shape (n_roi_total, n_sessions) 
         where each row is a boolean vector indicating which session(s) 
         the ROI was present in.
        Use the self.n_roi attribute to determine which rows belong to 
         which session.
        """
        ## Check that n_roi is set
        assert hasattr(self, 'n_roi'), f"RH ERROR: n_roi must be set before sessionID_concat can be created."
        ## Check that n_roi is the correct length
        assert len(self.n_roi) == self.n_sessions, f"RH ERROR: n_roi must be the same length as n_sessions."
        ## Check that n_roi_total is correct
        assert sum(self.n_roi) == self.n_roi_total, f"RH ERROR: n_roi must sum to n_roi_total."
        ## Create sessionID_concat
        self.sessionID_concat = np.vstack([np.array([helpers.idx2bool(i_sesh, length=self.n_sessions)]*n) for i_sesh, n in enumerate(self.n_roi)])
        print(f"Completed: Created sessionID_concat.") if self._verbose else None


    def _make_spatialFootprintCentroids(self, method='centerOfMass'):
        """
        Gets the centroids of a sparse array of flattented spatial footprints.
        Calculates the centroid position as the center of mass of the ROI.
        JZ, RH 2022

        Args:
            method (str):
                Method to use to calculate the centroid.
                Options:
                    'centerOfMass':
                        Calculates the centroid position as the mean center of
                         mass of the ROI.
                    'median':
                        Calculates the centroid position as the median center of
                         mass of the ROI.

        Input Attributes:
            self.spatialFootprints (scipy.sparse.csr_matrix):
                Spatial footprints.
                Shape: (n_roi, FOV_height*FOV_width) in C flattened format.
            self.FOV_height (int):
                Height of the FOV.
            self.FOV_width (int):
                Width of the FOV.

        Returns:
            self.centroids (np.ndarray):
                Centroids of the ROIs.
                Shape: (2, n_roi). (y, x) coordinates.
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
        y_cent = [np.round(h).astype(np.int64) for h in y_cent]
        x_cent = [np.round(w).astype(np.int64) for w in x_cent]
        
        ## Concatenate and store
        self.centroids = [np.stack([y, x], axis=1) for y, x in zip(y_cent, x_cent)]
        print(f"Completed: Created centroids.") if self._verbose else None

    
    def _transform_spatialFootprints_to_ROIImages(self, out_height_width=(36,36)):
        """
        Transform sparse spatial footprints to dense ROI images.

        Args:
            out_height_width (tuple):
                Height and width of the output images. Default is (36,36).
        """
        ## Check inputs
        assert hasattr(self, 'spatialFootprints'), f"RH ERROR: spatialFootprints must be set before ROI images can be created."
        assert hasattr(self, 'FOV_height') and hasattr(self, 'FOV_width'), f"RH ERROR: FOV_height and FOV_width must be set before ROI images can be created."
        assert hasattr(self, 'centroids'), f"RH ERROR: centroids must be set before ROI images can be created."
        assert isinstance(out_height_width, (tuple, list)), f"RH ERROR: out_height_width must be a tuple or list containing two elements (y, x)."
        assert len(out_height_width) == 2, f"RH ERROR: out_height_width must be a tuple of length 2."
        assert all([isinstance(h, int) for h in out_height_width]), f"RH ERROR: out_height_width must be a tuple of integers."
        assert all([h > 0 for h in out_height_width]), f"RH ERROR: out_height_width must be a tuple of positive integers."

        ## Make helper function
        def sf_to_centeredROIs(sf, centroids):
            half_widths = np.ceil(np.array(out_height_width)/2).astype(int)
            sf_rs = sparse.COO(sf).reshape((sf.shape[0], self.FOV_height, self.FOV_width))

            coords_diff = np.diff(sf_rs.coords[0])
            assert np.all(coords_diff < 1.01) and np.all(coords_diff > -0.01), \
                "RH ERROR: sparse.COO object has strange .coords attribute. sf_rs.coords[0] should all be 0 or 1. An ROI is possibly all zeros."
            
            idx_split = (sf_rs>0).astype(np.bool8).sum((1,2)).todense().cumsum()[:-1]
            coords_split = [np.split(sf_rs.coords[ii], idx_split) for ii in [0,1,2]]
            coords_split[1] = [coords - centroids[0][ii] + half_widths[0] for ii,coords in enumerate(coords_split[1])]
            coords_split[2] = [coords - centroids[1][ii] + half_widths[1] for ii,coords in enumerate(coords_split[2])]
            sf_rs_centered = sf_rs.copy()
            sf_rs_centered.coords = np.array([np.concatenate(c) for c in coords_split])
            sf_rs_centered = sf_rs_centered[:, :out_height_width[0], :out_height_width[1]]
            return sf_rs_centered.todense()
            
        ## Transform
        print(f"Staring: Creating centered ROI images from spatial footprints...") if self._verbose else None
        self.ROI_images = [sf_to_centeredROIs(sf, centroids.T) for sf, centroids in zip(self.spatialFootprints, self.centroids)]
        print(f"Completed: Created ROI images.") if self._verbose else None
        

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
    
    def save(
        self, 
        path_save,
        compress=False,
        allow_overwrite=False,
    ):
        """
        Save Data_roicat object to pickle file.
        
        Args:
            save_path (str or pathlib.Path):
                Path to save pickle file.
        """
        from pathlib import Path
        ## Check if file already exists
        if not allow_overwrite:
            assert not Path(path_save).exists(), f"RH ERROR: File already exists: {path_save}. Set allow_overwrite=True to overwrite."

        helpers.pickle_save(
            obj=self,
            path_save=path_save,
            zipCompress=compress,
            allow_overwrite=allow_overwrite,
        )
        print(f"Saved Data_roicat as a pickled object to {path_save}.") if self._verbose else None

    def load(self, path_load):
        """
        Load attributes from Data_roicat object from pickle file.
        
        Args:
            path_load (str or pathlib.Path):
                Path to pickle file.
        
        Returns:
            Data_roicat object.
        """
        # print(dir(cls))
        from pathlib import Path
        assert Path(path_load).exists(), f"RH ERROR: File does not exist: {path_load}."
        obj = helpers.pickle_load(path_load)
        assert isinstance(obj, type(self)), f"RH ERROR: Loaded object is not a Data_roicat object. Loaded object is of type {type(obj)}."

        ## Set attributes
        for key, val in obj.__dict__.items():
            setattr(self, key, val)
        

        print(f"Loaded Data_roicat object from {path_load}.") if obj._verbose else None



############################################################################################################################
############################## CUSTOM CLASSES FOR SUITE2P AND CAIMAN OUTPUT FILES ##########################################
############################################################################################################################

#########################################################
#################### DATA S2P ###########################
#########################################################

class Data_suite2p(Data_roicat):
    """
    Class for handling suite2p output files and data.
    In particular stat.npy and ops.npy files.
    RH, JZ 2022
    """
    def __init__(
        self,
        paths_statFiles,
        paths_opsFiles=None,
        um_per_pixel=1.0,
        new_or_old_suite2p='new',
        
        out_height_width=(36,36),
        type_meanImg='meanImgE',
        FOV_images=None,
        workers=-1,
        
        centroid_method = 'centroid',
        
        verbose=True,
    ):
        """
        Initializes the class for importing spatial footprints.
        Args:
            paths_stat (list of str or pathlib.Path):
                List of paths to the stat.npy files.
                Elements should be one of: str, pathlib.Path,
                 list of str or list of pathlib.Path
            paths_ops (list of str or pathlib.Path):
                Optional. Only used to get FOV images.
                List of paths to the ops.npy files.
                Elements should be one of: str, pathlib.Path,
                 list of str or list of pathlib.Path
            um_per_pixel (float):
                'micrometers per pixel' of the imaging field
                  of view.
            verbose (bool):
                If True, prints results from each function.
        """

        self.paths_stat = fix_paths(paths_statFiles)
        self.n_sessions = len(self.paths_stat)

        self.statFiles = None

        self.um_per_pixel = um_per_pixel
        self._new_or_old_suite2p = new_or_old_suite2p
        self.centroid_method = centroid_method
        self._verbose = verbose
        
        ## shifts are applied to convert the 'old' matlab version of suite2p indexing (where there is an offset and its 1-indexed)
        if paths_opsFiles is not None:
            self.paths_ops = fix_paths(paths_opsFiles)
            if self._new_or_old_suite2p == 'old':
                self.shifts = [np.array([op['yrange'].min()-1, op['xrange'].min()-1], dtype=np.uint64) for op in [np.load(path, allow_pickle=True)[()] for path in self.paths_ops]]
            else:
                self.shifts = [np.array([0,0], dtype=np.uint64)]*len(paths_statFiles)
            
        else:
            self.paths_ops = None
            self.shifts = [np.array([0,0], dtype=np.uint64)]*len(paths_statFiles)

        FOV_images = self.import_FOV_images(type_meanImg=type_meanImg)
        self.set_FOV_images(FOV_images=FOV_images)

        spatialFootprints = self.import_spatialFootprints()
        self.set_spatialFootprints(spatialFootprints=spatialFootprints, um_per_pixel=um_per_pixel)

        self._make_sessionID_concat()
        self._make_spatialFootprintCentroids()
        
        self._transform_spatialFootprints_to_ROIImages(out_height_width=out_height_width)
        

    def import_FOV_images(
        self,
        type_meanImg='meanImgE',
    ):
        """
        Imports the FOV images from ops files or user defined
         image arrays.

        Args:
            type_meanImg (str):
                Type of the mean image.
                References the key in the ops.npy file.
                Options are:
                    'meanImgE':
                        Enhanced mean image.
                    'meanImg':
                        Mean image.
        
        Returns:
            FOV_images (list):
                List of FOV images.
                Length of the list is the same self.paths_files.
                Each element is a numpy.ndarray of shape:
                 (n_files, height, width)
        """

        print(f"Starting: Importing FOV images from ops files") if self._verbose else None
        
        assert self.paths_ops is not None, "RH ERROR: paths_ops is None. Please set paths_ops before calling this function."
        assert len(self.paths_ops) > 0, "RH ERROR: paths_ops is empty. Please set paths_ops before calling this function."
        assert all([Path(path).exists() for path in self.paths_ops]), "RH ERROR: One or more paths in paths_ops do not exist."

        FOV_images = np.array([np.load(path, allow_pickle=True)[()][type_meanImg] for path in self.paths_ops]).astype(np.float32)
        
        print(f"Completed: Imported {len(FOV_images)} FOV images.") if self._verbose else None
        
        return FOV_images
    

    def import_spatialFootprints(
        self,
        frame_height_width=None,
        dtype=np.float32,
    ):
        """
        Imports and converts the spatial footprints of the ROIs
         in the stat files into images in sparse arrays.
        Output will be a list of arrays of shape 
         (n_roi, frame height, frame width).
        Also generates self.sessionID_concat which is a bool np.ndarray
         of shape(n_roi, n_sessions) indicating which session each ROI
         belongs to.
        
        Args:
            frame_height_width (list or tuple):
                [height, width] of the frame.
                If None, then import_FOV_images must be
                 called before this method, and the frame
                 height and width will be taken from the first FOV 
                 image.
            dtype (np.dtype):
                Data type of the sparse array.

        Returns:
            sf (list):
                Spatial Footprints.
                Length of the list is the same self.paths_files.
                Each element is a np.ndarray of shape:
                    (n_roi, frame_height_width[0], frame_height_width[1])
        """

        print("Importing spatial footprints from stat files.") if self._verbose else None

        ## Check and fix inputs
        if frame_height_width is None:
            frame_height_width = [self.FOV_height, self.FOV_width]

        isInt = np.issubdtype(dtype, np.integer)

        assert self.paths_stat is not None, "RH ERROR: paths_stat is None. Please set paths_stat before calling this function."
        assert len(self.paths_stat) > 0, "RH ERROR: paths_stat is empty. Please set paths_stat before calling this function."
        assert all([Path(path).exists() for path in self.paths_stat]), "RH ERROR: One or more paths in paths_stat do not exist."

        statFiles = [np.load(path, allow_pickle=True) for path in self.paths_stat]

        n = self.n_sessions
        spatialFootprints = [
            self._transform_statFile_to_spatialFootprints(
                frame_height_width=frame_height_width,
                stat=statFiles[ii],
                dtype=dtype,
                isInt=isInt,
                shifts=self.shifts[ii]
            ) for ii in tqdm(range(n))]

        if self._verbose:
            print(f"Imported {len(spatialFootprints)} sessions of spatial footprints into sparse arrays.")

        return spatialFootprints
    
    @staticmethod
    def _transform_statFile_to_spatialFootprints(frame_height_width, stat, dtype, isInt, shifts=(0,0)):
        """
        Populates a sparse array with the spatial footprints from ROIs
        in a stat file.
        """
        
        rois_to_stack = []
        
        for jj, roi in enumerate(stat):
            lam = np.array(roi['lam'], ndmin=1)
            if isInt:
                lam = dtype(lam / lam.sum() * np.iinfo(dtype).max)
            else:
                lam = lam / lam.sum()
            ypix = np.array(roi['ypix'], dtype=np.uint64, ndmin=1) + shifts[0]
            xpix = np.array(roi['xpix'], dtype=np.uint64, ndmin=1) + shifts[1]
        
            tmp_roi = scipy.sparse.csr_matrix((lam, (ypix, xpix)), shape=(frame_height_width[0], frame_height_width[1]), dtype=dtype)
            rois_to_stack.append(tmp_roi.reshape(1,-1))

        return scipy.sparse.vstack(rois_to_stack).tocsr()



#########################################################
################## DATA CAIMAN ##########################
#########################################################

class Data_caiman(Data_roicat):
    """
    Class for importing data from CaImAn output files.
    In particular, the hdf5 results files.
    RH, JZ 2022
    """
    def __init__(
        self,
        paths_resultsFiles,
        paths_labelFiles=None,
        include_discarded=True,
        um_per_pixel=1.0,
        
        out_height_width=[36,36],        
        centroid_method = 'centroid',
        
        verbose=True 
    ):
        """
        Args:
            paths_resultsFiles (list):
                List of paths to the results files.
            um_per_pixel (float):
                Microns per pixel.
            verbose (bool):
                If True, print statements will be printed.

        Attributes set:
            self.paths_resultsFiles (list):
                List of paths to the CaImAn results files.
            self.spatialFootprints (list):
                List of spatial footprints.
                Each element is a scipy.sparse.csr_matrix that contains
                 the flattened (order='C', C-memory order) spatial footprint masks for
                 each ROI in a given session. Each element is a session,
                 and each element has shape (n_roi, frame_height_width[0]*frame_height_width[1]).
            self.sessionID_concat (np.ndarray):
                a bool np.ndarray of shape(n_roi, n_sessions) indicating
                 which session each ROI belongs to.
            self.n_sessions (int):
                Number of sessions.
            self.n_roi (list):
                List of number of ROIs in each session.
            self.n_roi_total (int):
                Total number of ROIs across all sessions.
            self.FOV_height (int):
                Height of the FOV in pixels.
            self.FOV_width (int):
                Width of the FOV in pixels.
            self.um_per_pixel (float):
                Microns per pixel of the FOV.
            self.centroid_method (str):
                Either 'centroid' or 'median'. Centroid computes the weighted
                mean location of an ROI. Median takes the median of all x and y
                pixels of an ROI.
            self._verbose (bool):
                If True, print statements will be printed.
            self._include_discarded (bool):
                If True, include ROIs that were discarded by CaImAn.
        """

        self.paths_resultsFiles = fix_paths(paths_resultsFiles)
        self.n_sessions = len(self.paths_resultsFiles)
        self.um_per_pixel = um_per_pixel
        self.centroid_method = centroid_method
        self._include_discarded = include_discarded
        self._verbose = verbose

        for path in self.paths_resultsFiles:
            if not path.exists():
                raise FileNotFoundError(f"RH ERROR: {path} does not exist.")

        self.import_caimanResults(paths_resultsFiles, include_discarded=self._include_discarded)

        print(f"Computing centroids from spatial footprints") if self._verbose else None
        self.centroids = [self.get_centroids(s, self.FOV_height, self.FOV_width).T for s in self.spatialFootprints]

        self.sessionID_concat = np.vstack([np.array([helpers.idx2bool(i_sesh, length=len(self.spatialFootprints))]*sesh.shape[0]) for i_sesh, sesh in enumerate(self.spatialFootprints)])
        
        
        self.import_ROI_centeredImages(
            out_height_width=out_height_width
        )
        
        if paths_labelFiles is not None:
            self.paths_labels = fix_paths(paths_labelFiles)
            self.import_ROI_labels()
        else:
            self.paths_labels = None
            self.labelFiles = None
            
    
    def import_caimanResults(self, paths_resultsFiles, include_discarded=True):
        """
        Imports the results file from CaImAn.
        RH 2022

        Args:
            path_resultsFile (str or pathlib.Path):
                Path to the results file.

        Returns:
            data (dict):
                Dictionary of data from the results file.
        """

        def _import_spatialFootprints(path_resultsFile, include_discarded=True):
            """
            Imports the spatial footprints from the results file.
            Note that CaImAn's data['estimates']['A'] is similar to 
             self.spatialFootprints, but uses 'F' order. ROICaT converts
             this into 'C' order to make self.spatialFootprints.
            RH 2022

            Args:
                path_resultsFile (str or pathlib.Path):
                    Path to a single results file.
                include_discarded (bool):
                    If True, include ROIs that were discarded by CaImAn.

            Returns:
                spatialFootprints (scipy.sparse.csr_matrix):
                    Spatial footprints.
            """
            data = helpers.h5_lazy_load(path_resultsFile)
            FOV_height, FOV_width = data['estimates']['dims']
            
            ## initialize the estimates.A matrix, which is a 'Fortran' indexed version of sf. Note the flipped dimensions for shape.
            sf_included = scipy.sparse.csr_matrix((data['estimates']['A']['data'], data['estimates']['A']['indices'], data['estimates']['A']['indptr']), shape=data['estimates']['A']['shape'][::-1])
            if include_discarded:
                discarded = data['estimates']['discarded_components']
                sf_discarded = scipy.sparse.csr_matrix((discarded['A']['data'], discarded['A']['indices'], discarded['A']['indptr']), shape=discarded['A']['shape'][::-1])
                sf_F = scipy.sparse.vstack([sf_included, sf_discarded])
            else:
                sf_F = sf_included

            ## reshape sf_F (which is in Fortran flattened format) into C flattened format
            sf = sparse.COO(sf_F).reshape((sf_F.shape[0], FOV_width, FOV_height)).transpose((0,2,1)).reshape((sf_F.shape[0], FOV_width*FOV_height)).tocsr()
            
            return sf
        
        def _import_overall_caiman_labels(path_resultsFile, include_discarded=True):
            """
            
            """
            data = helpers.h5_lazy_load(path_resultsFile)
            labels_included = np.ones(data['estimates']['A']['indptr'].shape[0])
            if include_discarded:
                discarded = data['estimates']['discarded_components']
                labels_discarded = np.zeros(discarded['A']['indptr'].shape[0])
                labels = np.hstack([labels_included, labels_discarded])
            else:
                labels = labels_included
            
            return labels
        
        def _import_cnn_caiman_preds(path_resultsFile, include_discarded=True):
            """
            
            """
            data = helpers.h5_lazy_load(path_resultsFile)
            preds_included = data['estimates']['cnn_preds']
            if include_discarded:
                discarded = data['estimates']['discarded_components']
                preds_discarded = discarded['cnn_preds']
                preds = np.hstack([preds_included, preds_discarded])
            else:
                preds = preds_included
            
            return preds

        print(f"Importing spatial footprints from CaImAn results hdf5 files") if self._verbose else None
        self.spatialFootprints = [_import_spatialFootprints(path, include_discarded=include_discarded) for path in paths_resultsFiles]
        self.overall_caiman_labels = [_import_overall_caiman_labels(path, include_discarded=include_discarded) for path in paths_resultsFiles]
        self.cnn_caiman_preds = [_import_cnn_caiman_preds(path, include_discarded=include_discarded) for path in paths_resultsFiles]

        self.n_roi = [s.shape[0] for s in self.spatialFootprints]
        self.n_roi_total = sum(self.n_roi)

        print(f"Importing FOV images from CaImAn results hdf5 files") if self._verbose else None
        self.import_FOV_images(paths_resultsFiles)

        print(f"Completed: Imported spatial footprints from {len(self.spatialFootprints)} CaImAn results files into class as self.spatialFootprints. Total number of ROIs: {self.n_roi_total}. Number of ROI from each file: {self.n_roi}") if self._verbose else None

    def import_ROI_centeredImages(
        self,
        out_height_width=[36,36],
    ):
        """
        Imports the ROI centered images from the CaImAn results files.
        RH, JZ 2022

        Args:
            out_height_width (list):
                Height and width of the output images. Default is [36,36].

        Returns:
            sf_rs_centered (np.ndarray):
                Centered ROI masks.
                Shape: (n_roi, out_height_width[0], out_height_width[1]).
        """
        def sf_to_centeredROIs(sf, centroids, out_height_width=36):
            out_height_width = np.array([36,36])
            half_widths = np.ceil(out_height_width/2).astype(int)
            sf_rs = sparse.COO(sf).reshape((sf.shape[0], self.FOV_height, self.FOV_width))

            coords_diff = np.diff(sf_rs.coords[0])
            assert np.all(coords_diff < 1.01) and np.all(coords_diff > -0.01), \
                "RH ERROR: sparse.COO object has strange .coords attribute. sf_rs.coords[0] should all be 0 or 1. An ROI is possibly all zeros."
            
            idx_split = (sf_rs>0).astype(np.bool8).sum((1,2)).todense().cumsum()[:-1]
            coords_split = [np.split(sf_rs.coords[ii], idx_split) for ii in [0,1,2]]
            coords_split[1] = [coords - centroids[0][ii] + half_widths[0] for ii,coords in enumerate(coords_split[1])]
            coords_split[2] = [coords - centroids[1][ii] + half_widths[1] for ii,coords in enumerate(coords_split[2])]
            sf_rs_centered = sf_rs.copy()
            sf_rs_centered.coords = np.array([np.concatenate(c) for c in coords_split])
            sf_rs_centered = sf_rs_centered[:, :out_height_width[0], :out_height_width[1]]
            return sf_rs_centered.todense()

        print(f"Computing ROI centered images from spatial footprints") if self._verbose else None
        self.ROI_images = [sf_to_centeredROIs(sf, centroids.T, out_height_width=out_height_width) for sf, centroids in zip(self.spatialFootprints, self.centroids)]


    def import_FOV_images(
        self,
        paths_resultsFiles=None,
        images=None,
    ):
        """
        Imports the FOV images from the CaImAn results files.
        RH, JZ 2022

        Args:
            paths_resultsFiles (list):
                List of paths to CaImAn results files.
            images (list):
                List of FOV images. If None, will import the 
                 estimates.b image from paths_resultsFiles.

        Returns:
            FOV_images (list):
                List of FOV images (np.ndarray).
        """
        def _import_FOV_image(path_resultsFile):
            data = helpers.h5_lazy_load(path_resultsFile)
            FOV_height, FOV_width = data['estimates']['dims']
            FOV_image = data['estimates']['b'][:,0].reshape(FOV_height, FOV_width, order='F')
            return FOV_image.astype(np.float32)

        if images is not None:
            if self._verbose:
                print("Using provided images for FOV_images.")
            self.FOV_images = images
        else:
            if paths_resultsFiles is None:
                paths_resultsFiles = self.paths_resultsFiles
            self.FOV_images = np.stack([_import_FOV_image(p) for p in paths_resultsFiles])
            self.FOV_images = self.FOV_images - self.FOV_images.min(axis=(1,2), keepdims=True)
            self.FOV_images = self.FOV_images / self.FOV_images.mean(axis=(1,2), keepdims=True)

        self.FOV_height, self.FOV_width = self.FOV_images[0].shape

        return self.FOV_images
    
    def import_ROI_labels(self):
        """
        Imports the image labels from an npy file. Should
        have the same 0th dimension as the stats files.

        Returns:
            self.labelFiles (np.array):
                Concatenated set of image labels.
        """
        
        print(f"Starting: Importing labels footprints from npy files") if self._verbose else None
        
        raw_labels = [np.load(path) for path in self.paths_labels]
        self.n_label = [len(stat) for stat in raw_labels]
        self.n_label_total = sum(self.n_label)
        self.labelFiles = helpers.squeeze_integers(np.concatenate(raw_labels))
        if type(self.statFiles) is np.ndarray:
            assert self.statFiles.shape[0] == self.labelFiles.shape[0] , 'num images in stat files does not correspond to num labels'
                
        print(f"Completed: Imported {len(self.labelFiles)} labels into class as self.labelFiles. Total number of ROIs: {self.n_label_total}. Number of ROI from each file: {self.n_label}") if self._verbose else None
        
        return self.labelFiles
    
    def get_centroids(self, sf, FOV_height, FOV_width):
        """
        Gets the centroids of a sparse array of flattented spatial footprints.
        Calculates the centroid position as the center of mass of the ROI.
        JZ 2022

        Args:
            sf (scipy.sparse.csr_matrix):
                Spatial footprints.
                Shape: (n_roi, FOV_height*FOV_width) in C flattened format.
            FOV_height (int):
                Height of the FOV.
            FOV_width (int):
                Width of the FOV.

        Returns:
            centroids (np.ndarray):
                Centroids of the ROIs.
                Shape: (2, n_roi). (y, x) coordinates.
        """
        sf_rs = sparse.COO(sf).reshape((sf.shape[0], FOV_height, FOV_width))
        w_wt, h_wt = sf_rs.sum(axis=2), sf_rs.sum(axis=1)
        if self.centroid_method == 'centroid':
            h_mean = (((w_wt*np.arange(w_wt.shape[1]).reshape(1,-1))).sum(1)/w_wt.sum(1)).todense()
            w_mean = (((h_wt*np.arange(h_wt.shape[1]).reshape(1,-1))).sum(1)/h_wt.sum(1)).todense()
        elif self.centroid_method == 'median':
            h_mean = ((((w_wt!=0)*np.arange(w_wt.shape[1]).reshape(1,-1))).todense()).astype(float)
            h_mean[h_mean==0] = np.nan
            h_mean = np.nanmedian(h_mean, axis=1)
            w_mean = ((((h_wt!=0)*np.arange(h_wt.shape[1]).reshape(1,-1))).todense()).astype(float)
            w_mean[w_mean==0] = np.nan
            w_mean = np.nanmedian(w_mean, axis=1)
        else:
            raise ValueError('Only valid methods are "centroid" or "median"')
        return np.round(np.vstack([h_mean, w_mean])).astype(np.int64)




####################################
######### HELPER FUNCTIONS #########
####################################

def fix_paths(paths):
    """
    Make sure path_files is a list of pathlib.Path
    
    Args:
        paths (list of str or pathlib.Path or str or pathlib.Path):
            Potentially dirty input.
            
    Returns:
        paths (list of pathlib.Path):
            List of pathlib.Path
    """
    
    if (type(paths) is str) or (type(paths) is pathlib.PosixPath):
        paths_files = [Path(paths)]
    elif type(paths[0]) is str:
        paths_files = [Path(path) for path in paths]
    elif type(paths[0]) is pathlib.PosixPath or type(paths[0]) is pathlib.WindowsPath:
        paths_files = paths
    else:
        raise TypeError("path_files must be a list of str or list of pathlib.Path or a str or pathlib.Path")

    return paths_files




