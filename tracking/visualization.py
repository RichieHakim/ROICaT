import matplotlib.pyplot as plt
from ipywidgets import interact, widgets

import numpy as np
import sparse
import scipy.sparse

import copy

from . import helpers

def display_toggle_image_stack(images, clim=None, **kwargs):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imshow_FOV = ax.imshow(
        images[0],
#         vmax=clim[1]
        **kwargs
    )

    def update(i_frame = 0):
        fig.canvas.draw_idle()
        imshow_FOV.set_data(images[i_frame])
        imshow_FOV.set_clim(clim)


    interact(update, i_frame=widgets.IntSlider(min=0, max=len(images)-1, step=1, value=0));


def display_toggle_2channel_image_stack(images, clim=None):

    fig, axs = plt.subplots(1,2 , figsize=(14,8))
    ax_1 = axs[0].imshow(images[0][...,0], clim=clim)
    ax_2 = axs[1].imshow(images[0][...,1], clim=clim)

    def update(i_frame = 0):
        fig.canvas.draw_idle()
        ax_1.set_data(images[i_frame][...,0])
        ax_2.set_data(images[i_frame][...,1])


    interact(update, i_frame=widgets.IntSlider(min=0, max=len(images)-1, step=1, value=0));


def compute_colored_FOV(
    spatialFootprints,
    FOV_height,
    FOV_width,
    boolSessionID,
    labels,
    confidence=None,
):
    """
    Computes a set of images of FOVs of spatial footprints, colored
     by the predicted class.

    Args:
        spatialFootprints (list of scipy.sparse.csr_matrix):
            List where each element is the sparse array of all the spatial
             footprints in that session.
        FOV_height (int):
            Height of the field of view
        FOV_width (int):
            Width of the field of view
        labels (np.ndarray):
            Label (will be a unique color) for each spatial footprint
        confidence (np.ndarray):
            Confidence for each spatial footprint
            If None, all confidence values are set to 1.
            Spatial fooprints with confidence < threshold_confidence
             have a color set to cmap(-1), which is typically black.
        threshold_confidence (float):
            Threshold for the confidence values.
    """
    if confidence is None:
        confidence = np.ones(len(labels))
    
    h, w = FOV_height, FOV_width

    rois = scipy.sparse.vstack(spatialFootprints)
    rois = rois.multiply(1.2/rois.max(1).A).power(1)

    u = np.unique(labels)

    n_c = len(u)

    colors = helpers.rand_cmap(nlabels=n_c, verbose=False)(np.linspace(0.,1.,n_c, endpoint=False))
    colors = colors / colors.max(1, keepdims=True)

    if np.isin(-1, labels):
        colors[0] = [0,0,0,0]

    labels_squeezed = helpers.squeeze_integers(labels)
    labels_squeezed -= labels_squeezed.min()

    rois_c = scipy.sparse.hstack([rois.multiply(colors[labels_squeezed, ii][:,None]) for ii in range(4)]).tocsr()
    rois_c.data = np.minimum(rois_c.data, 1)

    rois_c_bySessions = [rois_c[idx] for idx in boolSessionID.T]

    rois_c_bySessions_FOV = [r.max(0).toarray().reshape(4, h, w).transpose(1,2,0)[:,:,:3] for r in rois_c_bySessions]

    return rois_c_bySessions_FOV


def crop_cluster_ims(ims):
    """
    Crops the images to the smallest rectangle containing all non-zero pixels.
    RH 2022

    Args:
        ims (np.ndarray):
            Images to crop.

    Returns:
        np.ndarray:
            Cropped images.
    """
    ims_max = np.max(ims, axis=0)
    z_im = ims_max > 0
    z_where = np.where(z_im)
    z_top = z_where[0].max()
    z_bottom = z_where[0].min()
    z_left = z_where[1].min()
    z_right = z_where[1].max()
    
    ims_copy = copy.deepcopy(ims)
    im_out = ims_copy[:, max(z_bottom-1, 0):min(z_top+1, ims.shape[1]), max(z_left-1, 0):min(z_right+1, ims.shape[2])]
    im_out[:,(0,-1),:] = 1
    im_out[:,:,(0,-1)] = 1
    return im_out