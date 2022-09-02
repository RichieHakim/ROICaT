import pathlib
from pathlib import Path
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
import scipy.sparse

from . import helpers

class Data_suite2p:
    """
    Class for handling suite2p files and data.
    RH 2022
    """
    def __init__(
        self,
        paths_statFiles,
        paths_opsFiles=None,
        um_per_pixel=1.0,
        new_or_old_suite2p='new',
        verbose=True 
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
            self.FOV_images = np.array([np.load(path, allow_pickle=True)[()][type_meanImg] for path in self.paths_ops])

            if self._verbose:
                print(f"Imported {len(self.FOV_images)} FOV images into class as self.FOV_images")

        self.FOV_height = self.FOV_images[0].shape[0]
        self.FOV_width = self.FOV_images[0].shape[1]

        return self.FOV_images


    def get_midCoords(
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
    if type(paths[0]) is str:
        paths_files = [Path(path) for path in paths]
    if type(paths[0]) is pathlib.PosixPath:
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
