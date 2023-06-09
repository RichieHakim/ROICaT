import matplotlib.pyplot as plt
import seaborn as sns

import os
import numpy as np
import sparse
import scipy.sparse

import copy

from . import util, helpers


def display_toggle_image_stack(images, image_size=None, clim=None, interpolation='nearest'):
    """
    Display images in a slider using Jupyter Notebook.
    RH 2023

    Args:
        images (list of numpy arrays or PyTorch tensors):
            List of images as numpy arrays or PyTorch tensors
        image_size (tuple of ints, optional):
            Tuple of (width, height) for resizing images.
            If None (default), images are not resized.
        clim (tuple of floats, optional):
            Tuple of (min, max) values for scaling pixel intensities.
            If None (default), min and max values are computed from the images
             and used as bounds for scaling.
        interpolation (string, optional):
            String specifying the interpolation method for resizing.
            Options: 'nearest', 'box', 'bilinear', 'hamming', 'bicubic', 'lanczos'.
            Uses the Image.Resampling.* methods from PIL.
    """
    from IPython.display import display, HTML
    import numpy as np
    import base64
    from PIL import Image
    from io import BytesIO
    import torch
    import datetime
    import hashlib
    import sys
    
    def normalize_image(image, clim=None):
        """Normalize the input image using the min-max scaling method. Optionally, use the given clim values for scaling."""
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if clim is None:
            clim = (np.min(image), np.max(image))

        norm_image = (image - clim[0]) / (clim[1] - clim[0])
        norm_image = np.clip(norm_image, 0, 1)
        return (norm_image * 255).astype(np.uint8)
    def resize_image(image, new_size, interpolation):
        """Resize the given image to the specified new size using the specified interpolation method."""
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        pil_image = Image.fromarray(image.astype(np.uint8))
        resized_image = pil_image.resize(new_size, resample=interpolation)
        return np.array(resized_image)
    def numpy_to_base64(numpy_array):
        """Convert a numpy array to a base64 encoded string."""
        img = Image.fromarray(numpy_array.astype('uint8'))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("ascii")
    def process_image(image):
        """Normalize, resize, and convert image to base64."""
        # Normalize image
        norm_image = normalize_image(image, clim)

        # Resize image if requested
        if image_size is not None:
            norm_image = resize_image(norm_image, image_size, interpolation_method)

        # Convert image to base64
        return numpy_to_base64(norm_image)


    # Check if being called from a Jupyter notebook
    if 'ipykernel' not in sys.modules:
        raise RuntimeError("This function must be called from a Jupyter notebook.")

    # Create a dictionary to map interpolation string inputs to Image objects
    interpolation_methods = {
        'nearest': Image.Resampling.NEAREST,
        'box': Image.Resampling.BOX,
        'bilinear': Image.Resampling.BILINEAR,
        'hamming': Image.Resampling.HAMMING,
        'bicubic': Image.Resampling.BICUBIC,
        'lanczos': Image.Resampling.LANCZOS,
    }

    # Check if provided interpolation method is valid
    if interpolation not in interpolation_methods:
        raise ValueError("Invalid interpolation method. Choose from 'nearest', 'box', 'bilinear', 'hamming', 'bicubic', or 'lanczos'.")

    # Get the actual Image object for the specified interpolation method
    interpolation_method = interpolation_methods[interpolation]

    # Generate a unique identifier for the slider
    slider_id = hashlib.sha256(str(datetime.datetime.now()).encode()).hexdigest()

    # Process all images in the input list
    base64_images = [process_image(img) for img in images]

    # Get the image size for display
    image_size = images[0].shape[:2] if image_size is None else image_size

    # Generate the HTML code for the slider
    html_code = f"""
    <div>
        <input type="range" id="imageSlider_{slider_id}" min="0" max="{len(base64_images) - 1}" value="0">
        <img id="displayedImage_{slider_id}" src="data:image/png;base64,{base64_images[0]}" style="width: {image_size[1]}px; height: {image_size[0]}px;">
        <span id="imageNumber_{slider_id}">Image 0/{len(base64_images) - 1}</span>
    </div>

    <script>
        (function() {{
            let base64_images = {base64_images};
            let current_image = 0;
    
            function updateImage() {{
                let slider = document.getElementById("imageSlider_{slider_id}");
                current_image = parseInt(slider.value);
                let displayedImage = document.getElementById("displayedImage_{slider_id}");
                displayedImage.src = "data:image/png;base64," + base64_images[current_image];
                let imageNumber = document.getElementById("imageNumber_{slider_id}");
                imageNumber.innerHTML = "Image " + current_image + "/{len(base64_images) - 1}";
            }}
            
            document.getElementById("imageSlider_{slider_id}").addEventListener("input", updateImage);
        }})();
    </script>
    """

    display(HTML(html_code))


def compute_colored_FOV(
    spatialFootprints,
    FOV_height,
    FOV_width,
    labels,
    cmap='random',
    alphas_labels=None,
    alphas_sf=None,
):
    """
    Computes a set of images of FOVs of spatial footprints, colored
     by the predicted class.

    Args:
        spatialFootprints (list of scipy.sparse.csr_matrix):
            Each element is all the spatial footprints for a given session.
        FOV_height (int):
            Height of the field of view
        FOV_width (int):
            Width of the field of view
        labels (list of arrays or array):
            Label (will be a unique color) for each spatial footprint.
            Each element is all the labels for a given session.
            Can either be a list of integer labels for each session,
             or a single array with all the labels concatenated.
        cmap (str or matplotlib.colors.ListedColormap):
            Colormap to use for the labels.
            If 'random', then a random colormap is generated.
            Else, this is passed to matplotlib.colors.ListedColormap.
        alphas_labels (np.ndarray):
            Alpha value for each label.
            shape (n_labels,) which is the same as the number of unique
             labels len(np.unique(labels))
        alphas_sf (list of np.ndarray):
            Alpha value for each spatial footprint.
            Can either be a list of alphas for each session, or a single array
             with all the alphas concatenated.
    """
    spatialFootprints = [spatialFootprints] if isinstance(spatialFootprints, np.ndarray) else spatialFootprints

    ## Check inputs
    assert all([scipy.sparse.issparse(sf) for sf in spatialFootprints]), "spatialFootprints must be a list of scipy.sparse.csr_matrix"

    n_roi = np.array([sf.shape[0] for sf in spatialFootprints], dtype=np.int64)
    n_roi_cumsum = np.concatenate([[0], np.cumsum(n_roi)]).astype(np.int64)
    n_roi_total = sum(n_roi)

    def _fix_list_of_arrays(v):
        if isinstance(v, np.ndarray) or (isinstance(v, list) and isinstance(v[0], (np.ndarray, list)) is False):
            v = [v[b_l: b_u] for b_l, b_u in zip(n_roi_cumsum[:-1], n_roi_cumsum[1:])]
        assert (isinstance(v, list) and isinstance(v[0], (np.ndarray, list))), "input must be a list of arrays or a single array of integers"
        return v
    
    labels = _fix_list_of_arrays(labels)
    alphas_sf = _fix_list_of_arrays(alphas_sf) if alphas_sf is not None else None

    labels_cat = np.concatenate(labels)
    u = np.unique(labels_cat)
    n_c = len(u)

    if alphas_labels is None:
        alphas_labels = np.ones(n_c)
    alphas_labels = np.clip(alphas_labels, a_min=0, a_max=1)
    assert len(alphas_labels) == n_c, f"len(alphas_labels)={len(alphas_labels)} != n_c={n_c}"

    if alphas_sf is None:
        alphas_sf = np.ones(len(labels_cat))
    if isinstance(alphas_sf, list):
        alphas_sf = np.concatenate(alphas_sf)
    alphas_sf = np.clip(alphas_sf, a_min=0, a_max=1)
    assert len(alphas_sf) == len(labels_cat), f"len(alphas_sf)={len(alphas_sf)} != len(labels_cat)={len(labels_cat)}"
    
    h, w = FOV_height, FOV_width

    rois = scipy.sparse.vstack(spatialFootprints)
    rois = rois.multiply(1.0/rois.max(1).A).power(1)

    if n_c > 1:
        colors = helpers.rand_cmap(nlabels=n_c, verbose=False)(np.linspace(0.,1.,n_c, endpoint=True)) if cmap=='random' else cmap(np.linspace(0.,1.,n_c, endpoint=True))
        colors = colors / colors.max(1, keepdims=True)
    else:
        colors = np.array([[0,0,0,0]])

    if np.isin(-1, labels_cat):
        colors[0] = [0,0,0,0]

    labels_squeezed = helpers.squeeze_integers(labels_cat)
    labels_squeezed -= labels_squeezed.min()

    rois_c = scipy.sparse.hstack([rois.multiply(colors[labels_squeezed, ii][:,None]) for ii in range(4)]).tocsr()
    rois_c.data = np.minimum(rois_c.data, 1)

    ## apply alpha
    rois_c = rois_c.multiply(alphas_labels[labels_squeezed][:,None] * alphas_sf[:,None]).tocsr()

    ## make session_bool
    session_bool = util.make_session_bool(n_roi)

    rois_c_bySessions = [rois_c[idx] for idx in session_bool.T]

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

def display_cropped_cluster_ims(
    spatialFootprints, 
    labels, 
    FOV_height=512, 
    FOV_width=1024,
    n_labels_to_display=100,
):
    """
    Displays the cropped cluster images.
    RH 2023

    Args:
        spatialFootprints (list):
            List of spatial footprints.
        labels (np.ndarray):
            Labels for each ROI.
        FOV_height (int, optional):
            Height of the field of view.
        FOV_width (int, optional):
            Width of the field of view.
        n_labels_to_display (int, optional):
            Number of labels to display.
    """
    import scipy.sparse

    labels_unique = np.unique(labels[labels>-1])

    ROI_ims_sparse = scipy.sparse.vstack(spatialFootprints)
    ROI_ims_sparse = ROI_ims_sparse.multiply( ROI_ims_sparse.max(1).power(-1) ).tocsr()

    labels_bool_t = scipy.sparse.vstack([scipy.sparse.csr_matrix(labels==u) for u in np.sort(np.unique(labels_unique))]).tocsr()
    labels_bool_t = labels_bool_t[:n_labels_to_display]

    def helper_crop_cluster_ims(ii):
        idx = labels_bool_t[ii].indices
        return np.concatenate(list(crop_cluster_ims(ROI_ims_sparse[idx].toarray().reshape(len(idx), FOV_height, FOV_width))), axis=1)

    labels_sfCat = [helper_crop_cluster_ims(ii) for ii in range(labels_bool_t.shape[0])]

    for sf in labels_sfCat[:n_labels_to_display]:
        plt.figure(figsize=(40,1))
        plt.imshow(sf, cmap='gray')
        plt.axis('off')


def select_region_scatterPlot(
    data, 
    path=None, 
    images_overlay=None, 
    idx_images_overlay=None, 
    size_images_overlay=None,
    figsize=(300,300),
):
    """Select a region of a scatter plot and return the indices 
     of the points in that region via a function call.

    Args:
        data (np.ndarray):
            Input data to draw a scatterplot.
        path (str, optional):
            Temporary file path that saves selected indices. Defaults to None.
        images_overlay (A 3D or 4D array, optional):
            A 3D array of grayscale images or a 4D array of RGB images,
            where the first dimension is the number of images. Defaults to None.
        idx_images_overlay (np.ndarray, optional):
            A vector of the data indices correspond to each images in images_overlay.
            Therefore, images_overlay must have the same number of images as idx_images_overlay. Defaults to None.
        size_images_overlay (tuple, optional):
            Size of each overlaid images. Defaults to None.
        figsize (tuple, optional):
            Size of the figure. Defaults to (300,300).
    """
    import holoviews as hv
    import numpy as np
    import pandas as pd

    import tempfile
    try:
        from IPython.display import display
    except:
        print('Warning: Could not import IPython.display. Cannot display plot.')
        return None, None

    hv.extension('bokeh')

    assert isinstance(data, np.ndarray), 'data must be a numpy array'
    assert data.ndim == 2, 'data must have 2 dimensions'
    assert data.shape[1] == 2, 'data must have 2 columns'

    ## Ingest inputs
    if images_overlay is not None:
        assert isinstance(images_overlay, np.ndarray), 'images_overlay must be a numpy array'
        assert (images_overlay.ndim == 3) or (images_overlay.ndim == 4), 'images_overlay must have 3 or 4 dimensions'
        assert images_overlay.shape[0] == idx_images_overlay.shape[0], 'images_overlay must have the same number of images as idx_images_overlay'

    if size_images_overlay is None:
        size_images_overlay = (data.max() - data.min()) / 30

    # Declare some points
    points = hv.Points(data)

    # Declare points as source of selection stream
    selection = hv.streams.Selection1D(source=points)

    path_tempFile = tempfile.gettempdir() + '/indices.csv' if path is None else path

    # Write function that uses the selection indices to slice points and compute stats
    def callback(index):
        ## Save the indices to a temporary file.
        ## First delete the file if it already exists.
        if os.path.exists(path_tempFile):
            os.remove(path_tempFile)
        ## Then save the indices to the file. Open in a protected way that blocks other threads from opening it
        with open(path_tempFile, 'w') as f:
            f.write(','.join([str(i) for i in index]))

        return points
        
    selection.param.watch_values(callback, 'index')
    layout = points.opts(
        tools=['lasso_select', 'box_select'],
        width=figsize[0],
        height=figsize[1],
    )

    # If images are provided, overlay them on the points
    def norm_img(image):
        """
        Normalize 2D grayscale image
        """        
        normalized_image = (image - np.min(image)) / np.max(image)
        return normalized_image

    imo = hv.RGB([])
    if images_overlay is not None and idx_images_overlay is not None:
        for image, idx in zip(images_overlay, idx_images_overlay):
            image_rgb = np.stack([norm_img(image), norm_img(image), norm_img(image)], axis=-1) if image.ndim == 2 else image
            image_rgb = hv.RGB(
                image_rgb, 
                bounds=(data[idx,0] - size_images_overlay/2, data[idx,1] - size_images_overlay/2, data[idx,0] + size_images_overlay/2, data[idx,1] + size_images_overlay/2)
            )
            imo *= image_rgb

    layout *= imo

    ## start layout with lasso tool active
    layout = layout.opts(
        active_tools=[
            'lasso_select', 
            'wheel_zoom',
        ]
    )

    # Display plot
    display(layout)

    def fn_get_indices():
        if os.path.exists(path_tempFile):
            with open(path_tempFile, 'r') as f:
                indices = f.read().split(',')
            indices = [int(i) for i in indices if i != ''] if len(indices) > 0 else None
            return indices
        else:
            return None

    return layout, path_tempFile, fn_get_indices