import matplotlib.pyplot as plt
from ipywidgets import interact, widgets

import numpy as np
import sparse
import scipy.sparse

import copy

from . import helpers

def display_toggle_image_stack(images, clim=None):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imshow_FOV = ax.imshow(
        images[0],
#         vmax=clim[1]
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
    preds,
    confidence=None,
    threshold_confidence = 0.5
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
        preds (np.ndarray):
            Predicted class for each spatial footprint
        confidence (np.ndarray):
            Confidence for each spatial footprint
            If None, all confidence values are set to 1.
            Spatial fooprints with confidence < threshold_confidence
             have a color set to cmap(-1), which is typically black.
        threshold_confidence (float):
            Threshold for the confidence values.
    """
    if confidence is None:
        confidence = np.ones(len(preds))
    
    idx_roi_session = np.concatenate([np.ones(sfs.shape[0])*ii for ii,sfs in enumerate(spatialFootprints)])

    n_sessions = len(spatialFootprints)

    n_planes = n_sessions
    labels = helpers.squeeze_integers(np.array(preds).astype(np.int64))

    labels[(confidence < threshold_confidence)] = -1
    # labels = labels

    ucid_toUse = labels
    idx_roi_session_toUse = idx_roi_session

    colors = sparse.COO(helpers.rand_cmap(len(np.unique(ucid_toUse)), verbose=False)(np.int64(ucid_toUse))[:,:3])
    # colors *= (1-((scores_samples / scores_samples.max()).numpy())**7)[:,None]
    # colors *= (((1/scores_samples) / (1/scores_samples).max()).numpy()**1)[:,None]

    plane_oneHot = helpers.idx_to_oneHot(idx_roi_session_toUse.astype(np.int32))

    ROIs_csr = scipy.sparse.csr_matrix(scipy.sparse.vstack(spatialFootprints))
    ROIs_csr_scaled = ROIs_csr.multiply(ROIs_csr.max(1).power(-1))
    ROIs_sCOO = sparse.COO(ROIs_csr_scaled)

    def tile_sparse(arr, n_tiles):
        """
        tiles along new (last) dimension
        """
        out = sparse.stack([arr for _ in range(n_tiles)], axis=-1)
        return out

    ROIs_tiled = tile_sparse(tile_sparse(ROIs_sCOO, n_planes), 3)

    ROIs_colored = ROIs_tiled * colors[:,None,None,:] * plane_oneHot[:,None,:,None]

    FOV_ROIs_colored = ROIs_colored.sum(0).reshape((FOV_height, FOV_width, n_planes, 3)).transpose((2,0,1,3))

    FOV_all_noClip = copy.copy(FOV_ROIs_colored.todense())
    FOV_all_noClip[FOV_all_noClip>1] = 1

    return FOV_all_noClip