import pathlib
import copy
import pickle
import glob

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import ctypes
import multiprocessing as mp
import torchvision
import PIL

###############################################################################
############################## IMPORT STAT FILES ##############################
###############################################################################

def statFile_to_spatialFootprints(path_statFile=None, statFile=None, out_height_width=[36,36], max_footprint_width=241, plot_pref=True):
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
        plot_pref (bool):
            If True, plots the spatial footprints.
    
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
    if plot_pref:
        plt.figure()
        plt.imshow(np.max(sf, axis=0)**0.2)
        plt.title('spatial footprints cropped MIP^0.2')
    
    return sf

def import_multiple_stat_files(paths_statFiles=None, dir_statFiles=None, fileNames_statFiles=None, out_height_width=[36,36], max_footprint_width=241, plot_pref=True):
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
                                                 max_footprint_width=max_footprint_width,
                                                 plot_pref=plot_pref)
                  for path_statFile in paths_statFiles]
    return sf_all_list

def convert_multiple_stat_files(statFiles_list=None, statFiles_dict=None, out_height_width=[36,36], max_footprint_width=241, print_pref=False, plot_pref=False):
    """
    Converts multiple stat files to spatial footprints.
    RH 2021

    Args:
        statFiles_list (list):
            List of stat files.
        out_height_width (list):
            [height, width] of the output spatial footprints.
        max_footprint_width (int):
            Maximum width of the spatial footprints.
        plot_pref (bool):
            If True, plots the spatial footprints.
    """
    if statFiles_dict is None:
        sf_all_list = [statFile_to_spatialFootprints(statFile=statFile,
                                                    out_height_width=out_height_width,
                                                    max_footprint_width=max_footprint_width,
                                                    plot_pref=plot_pref)
                    for statFile in statFiles_list]
    else:
        sf_all_list = []
        for key, stat in statFiles_dict.items():
            if print_pref:
                print(key)
            sf_all_list.append(statFile_to_spatialFootprints(statFile=stat,
                                                    out_height_width=out_height_width,
                                                    max_footprint_width=max_footprint_width,
                                                    plot_pref=plot_pref))
    return sf_all_list
    

def import_multiple_label_files(paths_labelFiles=None, dir_labelFiles=None, fileNames_labelFiles=None, plot_pref=True):
    """
    Imports multiple label files.
    RH 2021

    Args:
        paths_labelFiles (list):
            List of paths to label files.
            Elements can be either str or pathlib.Path.
        dir_labelFiles (str):
            Directory of label files.
            Optional: if paths_labelFiles is provided, this
             argument is ignored.
        fileNames_labelFiles (list):
            List of file names of label files.
            Optional: if paths_labelFiles is provided, this
             argument is ignored.
        plot_pref (bool):
            If True, plots the label files.
    """
    if paths_labelFiles is None:
        paths_labelFiles = [pathlib.Path(dir_labelFiles) / fileName for fileName in fileNames_labelFiles]

    labels_all_list = [np.load(path_labelFile, allow_pickle=True) for path_labelFile in paths_labelFiles]

    if plot_pref:
        for ii, labels in enumerate(labels_all_list):
            plt.figure()
            plt.hist(labels, 20)
            plt.title('labels ' + str(ii))
    return labels_all_list

###############################################################################
############################### Directory functions ###########################
###############################################################################

# def query_directory(base_dir=None,
#                     query=None):
#     '''
#     Find a file in a directory and its recursive subdirectories

#     JZ 2021

#     Args:
#         base_dir: str
#             The base directory in which to start the search
#         query : str
#             The file name to look for in the subdirectories of base_dir
#     Returns:
#         A list of strings of full path directories to files named like the given file in question
#     '''
#     sub_base_dir = base_dir + r'/' if base_dir[-2:] != '/' else base_dir

#     traversal_list = []
#     traversal_list.append(str(sub_base_dir))

#     seen_values = []

#     counter = 1

#     while True:
#         if len(traversal_list) == 0:
#             break
#         print(f'Currently Exploring Directory # {counter}')
#         directory = traversal_list.pop(0)
#         if query in directory:
#             seen_values.append(directory)
#         else:
#             traversal_list.extend(glob.glob(str(directory + r'/*')))

#         counter += 1

#     return seen_values

# result = query_directory()

# paths_all = {}
# for path in result:
#     paths_all[path] = np.load(path, allow_pickle=True)
    
# path_save = r'\\research.files.med.harvard.edu\Neurobio\MICROSCOPE\Rich\data\res2p\scanimage data\all_stat_files_20211022.pkl'
# with open(path_save, 'wb') as file:
#     pickle.dump(paths_all, file)


def get_trainable_parameters(model):
    
    params_trainable = []
    for param in list(model.parameters()):
        if param.requires_grad:
            params_trainable.append(param)
    return params_trainable

def resize_affine(img, scale, clamp_range=False):
    """
    Wrapper for torchvision.transforms.Resize.
    Useful for resizing images to match the size of the images
     used in the training of the neural network.
    RH 2022
    """
    img_rs = np.array(torchvision.transforms.functional.affine(
#         img=torch.as_tensor(img[None,...]),
        img=PIL.Image.fromarray(img),
        angle=0, translate=[0,0], shear=0,
        scale=scale,
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC
    ))
    
    if clamp_range:
        clamp_high = img.max()
        clamp_low = img.min()
    
        img_rs[img_rs>clamp_high] = clamp_high
        img_rs[img_rs<clamp_low] = clamp_low
    
    return img_rs