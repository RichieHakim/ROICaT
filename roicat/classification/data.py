import numpy as np
import pathlib
from pathlib import Path
from .. import helpers

class Data():
    def __init__(
        self,
        paths_statFiles,
        paths_labelFiles=None,
        um_per_pixel=1.0,
        new_or_old_suite2p='new',
        verbose=True 
                ):
        """
        Initializes the class for importing spatial footprints.
        Args:
            paths_statFiles (list of str or pathlib.Path):
                List of paths to the stat.npy files.
                Elements should be one of: str, pathlib.Path,
                 list of str or list of pathlib.Path
            paths_labelFiles (list of str or pathlib.Path):
                Optional. Only used to train a classifier.
                List of paths to the label .npy files.
                Elements should be one of: str, pathlib.Path,
                 list of str or list of pathlib.Path
            um_per_pixel (float):
                'micrometers per pixel' of the imaging field
                  of view.
            verbose (bool):
                If True, prints results from each function.
        """

        self.paths_stat = fix_paths(paths_statFiles)
        if paths_labelFiles is not None:
            self.paths_lbl = fix_paths(paths_labelFiles)
        else:
            self.paths_lbl = None

        self.n_sessions = len(self.paths_stat)
        self.statFiles = None
        self.labelFiles = None
        self.um_per_pixel = um_per_pixel
        self._new_or_old_suite2p = new_or_old_suite2p
        self._verbose = verbose
        
        return
    
    def import_statFiles(self):
        """
        Imports the stats.npy contents into the class.
        This method can be called before any other function.

        Returns:
            self.statFiles (np.array):
                Concatenated set of imported files.
        """

        print(f"Starting: Importing spatial footprints from stat files") if self._verbose else None

        statFiles_lst = []
        for path_stat in self.paths_stat:
            if Path(path_stat).suffix == '.npz':
                dat = np.load(path_stat)
                images_labeled = scipy.sparse.csr_matrix((dat['data'], dat['indices'], dat['indptr']), shape=dat['shape']).toarray()
                images_labeled = [images_labeled.reshape([-1, 36,36])]

            elif Path(path_stat).suffix == '.npy':
                images_labeled = \
                    import_multiple_stat_files(   
                        paths_statFiles=[path_stat],
                        out_height_width=[36,36],
                        max_footprint_width=241
                    )
            else:
                raise ValueError(f'path_stat: {path_stat} is not an npy or npz file.')
            
            statFiles_lst.extend(images_labeled)
        
        
        self.n_roi = [len(stat) for stat in statFiles_lst]
        self.n_roi_total = sum(self.n_roi)
        
        self.statFiles = np.concatenate(statFiles_lst,axis=0)
        
        if type(self.labelFiles) is np.ndarray:
            assert self.statFiles.shape[0] == self.labelFiles.shape[0] , 'num images in stat files does not correspond to num labels'

        print(f"Completed: Imported {len(self.statFiles)} stat files into class as self.statFiles. Total number of ROIs: {self.n_roi_total}. Number of ROI from each file: {self.n_roi}") if self._verbose else None
        
        
        return self.statFiles
        
    def import_labelFiles(self):
        """
        Imports the image labels from an npy file. Should
        have the same 0th dimension as the stats files.

        Returns:
            self.labelFiles (np.array):
                Concatenated set of image labels.
        """
        
        print(f"Starting: Importing labels footprints from npy files") if self._verbose else None
        
        raw_labels = [np.load(path) for path in self.paths_lbl]
        self.n_lbl = [len(stat) for stat in raw_labels]
        self.n_lbl_total = sum(self.n_roi)
        self.labelFiles = helpers.squeeze_integers(np.concatenate(raw_labels))
        if type(self.statFiles) is np.ndarray:
            assert self.statFiles.shape[0] == self.labelFiles.shape[0] , 'num images in stat files does not correspond to num labels'
                
        print(f"Completed: Imported {len(self.labelFiles)} labels into class as self.labelFiles. Total number of ROIs: {self.n_lbl_total}. Number of ROI from each file: {self.n_lbl}") if self._verbose else None
        
        return self.labelFiles


    def drop_nan_rois(self):
        """
        Identifies all entries along the 0th dimension of self.statFiles that
        have any NaN value in any of their dimensions and removes
        those entries from both self.statFiles and self.labelFiles
        """
        idx_nne = helpers.get_keep_nonnan_entries(self.statFiles)
        self.statFiles = self.statFiles[idx_nne]
        self.labelFiles = self.labelFiles[idx_nne]
        return self.statFiles, self.labelFiles



        
    def relabeling(self):
        """
        TBD if should be implemembted for relabeling all instances of one class to another
        Or to drop a given class entirely.
        """
#         https://github.com/seung-lab/fastremap

#         # Relabel values based on relabels definition
#         # Used for combining e.g. 4 into 3 via {4: 3}
#         new_labels = fg.relabel(labels, relabels)

#         # Identify the examples to keep by with labels not in the list "lbls_to_drop"
#         # E.g. If all label 6s are bad data to not be classified, pass a list of [6]
#         keep_tf = fg.get_keep_labels(new_labels, lbls_to_drop)


#         # Create a final list of indices that should be kept (for use in filtering both labels and ROI images)
#         idx_toKeep = fg.get_keep_entries(keep_tf)

#         # 
#         idx_nne = fg.get_keep_nonnan_entries(images_labeled_clean)
#         images_labeled_clean = images_labeled_clean[idx_nne]
#         latents_clean = latents_clean[idx_nne]
#         labels_clean = labels_clean[idx_nne]
            
#         # Keep only ROI values that are labelled by not dropped classes
#         images_labeled_clean = raw_images_dup[idx_toKeep]
#         latents_clean = latents_dup[idx_toKeep]
#         labels_clean = labels[idx_toKeep]

#         # Set lowest label value to 0 and sweeze all other label numbers to be sequential integers
#         labels_clean -= labels_clean.min()
#         labels_clean = helpers.squeeze_integers(labels_clean)

#         # 
#         idx_nne = fg.get_keep_nonnan_entries(images_labeled_clean)
#         images_labeled_clean = images_labeled_clean[idx_nne]
#         latents_clean = latents_clean[idx_nne]
#         labels_clean = labels_clean[idx_nne]
            
        return

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
        paths = [Path(paths)]
    if type(paths[0]) is str:
        paths = [Path(path) for path in paths]
    if type(paths[0]) is pathlib.PosixPath:
        paths = paths
    else:
        raise TypeError("path_files must be a list of str or list of pathlib.Path or a str or pathlib.Path")

    return paths
    
    

def import_multiple_stat_files(paths_statFiles=None, dir_statFiles=None, fileNames_statFiles=None, out_height_width=[36,36], max_footprint_width=241):
    """
    Imports multiple stat files.
    RH 2021 
    
    Args:
        paths_statFiles (list):
            List of paths to stat files.
            Elements can be either str or pathlib.Path.
        dir_statFiles (str):
            Directory of stat files.
            Optional: if paths_statFiles is provided, this
             argument is ignored.
        fileNames_statFiles (list):
            List of file names of stat files.
            Optional: if paths_statFiles is provided, this
             argument is ignored.
        out_height_width (list):
            [height, width] of the output spatial footprints.
        max_footprint_width (int):
            Maximum width of the spatial footprints.
        plot_pref (bool):
            If True, plots the spatial footprints.

    Returns:
        stat_all (list):
            List of stat files.
    """
    if paths_statFiles is None:
        paths_statFiles = [pathlib.Path(dir_statFiles) / fileName for fileName in fileNames_statFiles]

    sf_all_list = [statFile_to_spatialFootprints(path_statFile=path_statFile,
                                                 out_height_width=out_height_width,
                                                 max_footprint_width=max_footprint_width)
                  for path_statFile in paths_statFiles]
    return sf_all_list


def statFile_to_spatialFootprints(path_statFile=None, statFile=None, out_height_width=[36,36], max_footprint_width=241):
    """
    Converts a stat file to a list of spatial footprint images.
    RH 2021

    Args:
        path_statFile (pathlib.Path or str):
            Path to the stat file.
            Optional: if statFile is provided, this
             argument is ignored.
        statFile (dict):
            Suite2p stat file dictionary
            Optional: if path_statFile is provided, this
             argument is ignored.
        out_height_width (list):
            [height, width] of the output spatial footprints.
        max_footprint_width (int):
            Maximum width of the spatial footprints.
    
    Returns:
        sf_all (list):
            List of spatial footprints images
    """
    assert out_height_width[0]%2 == 0 and out_height_width[1]%2 == 0 , "RH: 'out_height_width' must be list of 2 EVEN integers"
    assert max_footprint_width%2 != 0 , "RH: 'max_footprint_width' must be odd"
    if statFile is None:
        stat = np.load(path_statFile, allow_pickle=True)
    else:
        stat = statFile
    n_roi = stat.shape[0]
    
    # sf_big: 'spatial footprints' prior to cropping. sf is after cropping
    sf_big_width = max_footprint_width # make odd number
    sf_big_mid = sf_big_width // 2

    sf_big = np.zeros((n_roi, sf_big_width, sf_big_width))
    for ii in range(n_roi):
        sf_big[ii , stat[ii]['ypix'] - np.int16(stat[ii]['med'][0]) + sf_big_mid, stat[ii]['xpix'] - np.int16(stat[ii]['med'][1]) + sf_big_mid] = stat[ii]['lam'] # (dim0: ROI#) (dim1: y pix) (dim2: x pix)

    sf = sf_big[:,  
                sf_big_mid - out_height_width[0]//2:sf_big_mid + out_height_width[0]//2,
                sf_big_mid - out_height_width[1]//2:sf_big_mid + out_height_width[1]//2]
    
    return sf

