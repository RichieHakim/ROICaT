import pathlib
from pathlib import Path
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
import scipy.sparse
import sparse

from . import helpers

class Data_suite2p:
    """
    Class for handling suite2p output files and data.
    In particular stat.npy and ops.npy files.
    RH 2022
    """
    def __init__(
        self,
        paths_statFiles,
        paths_opsFiles=None,
        um_per_pixel=1.0,
        new_or_old_suite2p='new',
        
        out_height_width=[36,36],
        max_footprint_width=1025,
        type_meanImg='meanImgE',
        images=None,
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
        if paths_opsFiles is not None:
            self.paths_ops = fix_paths(paths_opsFiles)
        else:
            self.paths_ops = None
        self.n_sessions = len(self.paths_stat)

        self.statFiles = None

        self.um_per_pixel = um_per_pixel
        self._new_or_old_suite2p = new_or_old_suite2p
        self._verbose = verbose
        
        ## shifts are applied to convert the 'old' matlab version of suite2p indexing (where there is an offset and its 1-indexed)
        if self.paths_ops is not None:
            self.shifts = [
                np.array([op['yrange'].min()-1, op['xrange'].min()-1], dtype=np.uint64) for op in [np.load(path, allow_pickle=True)[()] for path in self.paths_ops]
            ] if self._new_or_old_suite2p == 'old' else [np.array([0,0], dtype=np.uint64)]*len(paths_statFiles)
        else:
            self.shifts = [np.array([0,0], dtype=np.uint64)]*len(paths_statFiles)

        self.import_statFiles()
        
        
        self.import_ROI_centeredImages(
            out_height_width=out_height_width,
            max_footprint_width=max_footprint_width,
        )

        self.import_FOV_images(
            type_meanImg=type_meanImg,
            images=images
        )

        self.import_ROI_spatialFootprints(workers=workers);
        
        self.centroid_method = centroid_method
#         self.centroids = self._get_midCoords()
        
        print(type(self.spatialFootprints[0]))
        
        self.centroids = [self.get_centroids(sf, self.FOV_height, self.FOV_width) for sf in self.spatialFootprints]
        
        

    def import_statFiles(self):
        """
        Imports the stats.npy contents into the class.
        This method can be called before any other function.

        Returns:
            self.statFiles (list):
                List of imported files. Type depends on sf_type.
        """

        print(f"Starting: Importing spatial footprints from stat files") if self._verbose else None

        self.statFiles = [np.load(path, allow_pickle=True) for path in self.paths_stat]

        self.n_roi = [len(stat) for stat in self.statFiles]
        self.n_roi_total = sum(self.n_roi)

        print(f"Completed: Imported {len(self.statFiles)} stat files into class as self.statFiles. Total number of ROIs: {self.n_roi_total}. Number of ROI from each file: {self.n_roi}") if self._verbose else None

        return self.statFiles


    def import_FOV_images(
        self,
        type_meanImg='meanImgE',
        images=None
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
            images (list of np.ndarray):
                Optional. If provided, the FOV images are 
                 defined by these images.
                If not provided, the FOV images are defined by
                 the ops.npy files from self.paths_ops.
                len(images) must be equal to len(self.paths_stat)
                Images must be of the same shape.
        
        Returns:
            self.FOV_images (list):
                List of FOV images.
                Length of the list is the same self.paths_files.
                Each element is a numpy.ndarray of shape:
                 (n_files, height, width)
        """

        if images is not None:
            if self._verbose:
                print("Using provided images for FOV_images.")
            self.FOV_images = images

        else:
            if self.paths_ops is None:
                raise ValueError("'path_ops' must be defined in initialization if 'images' is not provided.")
            self.FOV_images = np.array([np.load(path, allow_pickle=True)[()][type_meanImg] for path in self.paths_ops]).astype(np.float32)
            self.FOV_images = self.FOV_images - self.FOV_images.min(axis=(1,2), keepdims=True)
            self.FOV_images = self.FOV_images / self.FOV_images.mean(axis=(1,2), keepdims=True)

            if self._verbose:
                print(f"Imported {len(self.FOV_images)} FOV images into class as self.FOV_images")

        self.FOV_height = self.FOV_images[0].shape[0]
        self.FOV_width = self.FOV_images[0].shape[1]

        return self.FOV_images


    def _get_midCoords(
        self,
    ):
        """
        Returns the middle coordinates of the ROIs.

        Returns:
            midPositions (list of np.ndarray):
                List of middle coordinates of the ROIs.
        """
        
        statFiles = self.import_statFiles() if self.statFiles is None else self.statFiles

        return [np.array([stat[jj]['med'] for jj in range(len(stat))]) for stat in statFiles]
        

    def import_ROI_centeredImages(
        self,
        out_height_width=[36,36], 
        max_footprint_width=1025, 
    ):
        """
        Converts the spatial footprints into images, stores them
         within the class and returns them.
        This method selects the appropriate method to use based on
         the type of the spatial footprints.
        If you want to dump the images from the class into a
         variable:
            var = None
            var, self.ROI_images = self.ROI_images, var

        Args:
            out_height_width (list):
                [height, width] of the output spatial footprints.
            max_footprint_width (int):
                Maximum width of the spatial footprints.
                Must be odd number.
                Make sure this number is larger than the largest
                 ROI you want to convert to an image.

        Returns:
            ROI_images (list):
                List of images.
                Length of the list is the same self.paths_files.
                Each element is a numpy.ndarray of shape:
                 (n_roi, self._out_height_width[0], self._out_height_width[1])
        """

        assert out_height_width[0]%2 == 0 and out_height_width[1]%2 == 0 , "'out_height_width' must be list of 2 EVEN integers"
        assert max_footprint_width%2 != 0 , "'max_footprint_width' must be odd"

        self._out_height_width = np.uint64(out_height_width)
        self._max_footprint_width = np.uint64(max_footprint_width)

        statFiles = self.import_statFiles() if self.statFiles is None else self.statFiles

        self.ROI_images = self._convert_stat_to_centeredImages(statFiles=statFiles)
        
        if self._verbose:
            print(f"Converted {len(self.ROI_images)} spatial footprint files into small centered images in self.ROI_images.")
        
        return self.ROI_images


    def import_ROI_spatialFootprints(
        self,
        frame_height_width=None,
        dtype=np.float32,
        workers=1,
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
            workers (int):
                Number of workers to use for multiprocessing.
                Set to -1. Note that this will use more memory.
            new_or_old_suite2p (str):
                'new': Python versions of Suite2p
                'old': Matlab versions of Suite2p

        Returns:
            sf (list):
                Spatial Footprints.
                Length of the list is the same self.paths_files.
                Each element is a np.ndarray of shape:
                    (n_roi, frame_height_width[0], frame_height_width[1])
        """

        print("Importing spatial footprints from stat files.") if self._verbose else None

        if frame_height_width is None:
            frame_height_width = [self.FOV_height, self.FOV_width]

        isInt = np.issubdtype(dtype, np.integer)

        statFiles = self.import_statFiles() if self.statFiles is None else self.statFiles

        n = self.n_sessions
        if workers == -1:
            workers = mp.cpu_count()
        if workers != 1:
            self.spatialFootprints = helpers.simple_multiprocessing(
                _helper_populate_sf, 
                (self.n_roi, [frame_height_width]*n, statFiles, [dtype]*n, [isInt]*n, self.shifts),
                workers=mp.cpu_count()
            )
        else:
            self.spatialFootprints = [
                _helper_populate_sf(
                    n_roi=self.n_roi[ii], 
                    frame_height_width=frame_height_width,
                    stat=statFiles[ii],
                    dtype=dtype,
                    isInt=isInt,
                    shifts=self.shifts[ii]
                ) for ii in tqdm(range(n), mininterval=60)]

        self.sessionID_concat = np.vstack([np.array([helpers.idx2bool(i_sesh, length=len(self.spatialFootprints))]*sesh.shape[0]) for i_sesh, sesh in enumerate(self.spatialFootprints)])

        if self._verbose:
            print(f"Imported {len(self.spatialFootprints)} sessions of spatial footprints into sparse arrays.")

        return self.spatialFootprints


    def _convert_stat_to_centeredImages(
        self,
        statFiles=None, 
    ):
        """
        Converts stat files to centered images.
        
        Args:
            statFiles (list):
                List of paths (str or pathlib.Path)
                 or stat files (numpy.ndarray).

        Returns:
            stat_all (list):
                List of stat files.
        """

        # sf_big: 'spatial footprints' prior to cropping. sf is after cropping
        sf_big_width = self._max_footprint_width # make odd number
        sf_big_mid = np.uint64(sf_big_width // 2)

        sf_all_list = []
        for ii, stat in tqdm(enumerate(statFiles), mininterval=60):
            if type(stat) is str or type(stat) is Path:
                stat = np.load(stat, allow_pickle=True)
            n_roi = stat.shape[0]

            sf_big = np.zeros((n_roi, sf_big_width, sf_big_width))
            for ii in range(n_roi):
                yIdx = np.array(stat[ii]['ypix'], dtype=np.uint64) - np.int64(stat[ii]['med'][0]) + sf_big_mid
                xIdx = np.array(stat[ii]['xpix'], dtype=np.uint64) - np.int64(stat[ii]['med'][1]) + sf_big_mid
                if np.any(yIdx < 0) or np.any(xIdx < 0) or np.any(yIdx >= sf_big_width) or np.any(xIdx >= sf_big_width):
                    raise IndexError(f"RH ERROR: Spatial footprint is out of bounds. Increase max_footprint_width.")
                sf_big[ii][np.uint64(yIdx), np.uint64(xIdx)] = stat[ii]['lam'] # (dim0: ROI#) (dim1: y pix) (dim2: x pix)

            sf = sf_big[:,  
                        sf_big_mid - np.uint64(self._out_height_width[0]//2) : sf_big_mid + np.uint64(self._out_height_width[0]//2),
                        sf_big_mid - np.uint64(self._out_height_width[1]//2) : sf_big_mid + np.uint64(self._out_height_width[1]//2)]

            sf_all_list.append(sf)

        return sf_all_list
    
    
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
#         sf_rs = sparse.COO(sf).reshape((sf.shape[0], FOV_height, FOV_width))
        w_wt, h_wt = sf_rs.sum(axis=2), sf_rs.sum(axis=1)
        if self.centroid_method == 'centroid':
            h_mean = (((w_wt*np.arange(w_wt.shape[1]).reshape(1,-1))).sum(1)/w_wt.sum(1)).todense()
            w_mean = (((h_wt*np.arange(h_wt.shape[1]).reshape(1,-1))).sum(1)/h_wt.sum(1)).todense()
        elif self.centroid_method == 'median':
            return None
        else:
            raise ValueError('Only valid methods are "centroid" or "median"')
        return np.round(np.vstack([h_mean, w_mean])).astype(np.int64)


class Data_caiman:
    """
    Class for importing data from CaImAn output files.
    In particular, the hdf5 results files.
    RH, JZ 2022
    """
    def __init__(
        self,
        paths_resultsFiles,
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
                 the flattened ('C' order) spatial footprint masks for
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
                sf_discarded = scipy.sparse.csr_matrix((data['estimates']['A']['data'], data['estimates']['A']['indices'], data['estimates']['A']['indptr']), shape=data['estimates']['A']['shape'][::-1])
                sf_F = scipy.sparse.vstack([sf_included, sf_discarded])
            else:
                sf_F = sf_included

            ## reshape sf_F (which is in Fortran flattened format) into C flattened format
            sf = sparse.COO(sf_F).reshape((sf_F.shape[0], FOV_width, FOV_height)).transpose((0,2,1)).reshape((sf_F.shape[0], FOV_width*FOV_height)).tocsr()
            
            return sf

        print(f"Importing spatial footprints from CaImAn results hdf5 files") if self._verbose else None
        self.spatialFootprints = [_import_spatialFootprints(path, include_discarded=include_discarded) for path in paths_resultsFiles]

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


class Data_custom:
    """
    Incase you want to make a custom data object,
     you can use this class as a template to fill
     in the required attributes.
    RH 2022
    """
    def __init__(
        self,
    ):       
        self.FOV_images = None
        self.FOV_height = None
        self.FOV_width = None
        self.spatialFootprints = None
        self.ROI_images = None
        self.n_roi = None
        self.n_roi_total = None
        self.n_sessions = None
        self.centroids = None
        self.um_per_pixel = None
        self._verbose = True 

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
    elif type(paths[0]) is pathlib.PosixPath:
        paths_files = paths
    else:
        raise TypeError("path_files must be a list of str or list of pathlib.Path or a str or pathlib.Path")

    return paths_files


def _helper_populate_sf(n_roi, frame_height_width, stat, dtype, isInt, shifts=(0,0)):
    """
    Helper function for populate_sf.
    Populates a sparse array with the spatial footprints from ROIs
     in a stat file. See import_ROI_spatialFootprints for more
     details.
    This needs to be a separate function because it is 
     used in a multiprocessing. Functions are only 
     picklable if they are defined at the top-level of a module.
    """
    sf = np.zeros((n_roi, frame_height_width[0], frame_height_width[1]), dtype)
    for jj, roi in enumerate(stat):
        lam = np.array(roi['lam'])
        if isInt:
            lam = dtype(lam / lam.sum() * np.iinfo(dtype).max)
        else:
            lam = lam / lam.sum()
        ypix = np.array(roi['ypix'], dtype=np.uint64) + shifts[0]
        xpix = np.array(roi['xpix'], dtype=np.uint64) + shifts[1]
        sf[jj, ypix, xpix] = lam

    return scipy.sparse.csr_matrix(sf.reshape(sf.shape[0], -1))
