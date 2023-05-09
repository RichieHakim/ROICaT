import matplotlib.pyplot as plt
from ipywidgets import interact, widgets
import seaborn as sns


import numpy as np
import sparse
import scipy.sparse

import copy

from . import helpers


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
    boolSessionID=None,
    alphas=None,
):
    """
    Computes a set of images of FOVs of spatial footprints, colored
     by the predicted class.

    Args:
        spatialFootprints (list of scipy.sparse.csr_matrix or scipy.sparse.csr_matrix):
            If list, then each element is all the spatial footprints for a given session.
            If scipy.sparse.csr_matrix, then this is all the spatial footprints for all 
             sessions, and boolSessionID must be provided.
        FOV_height (int):
            Height of the field of view
        FOV_width (int):
            Width of the field of view
        labels (list of arrays or array):
            Label (will be a unique color) for each spatial footprint.
            If list, then each element is all the labels for a given session.
            If array, then this is all the labels for all sessions, and 
             boolSessionID must be provided.
        cmap (str or matplotlib.colors.ListedColormap):
            Colormap to use for the labels.
            If 'random', then a random colormap is generated.
            Else, this is passed to matplotlib.colors.ListedColormap.
        boolSessionID (np.ndarray of bool):
            Boolean array indicating which session each spatial footprint belongs to.
            Only required if spatialFootprints and labels are not lists.
            shape: (n_roi_total, n_sessions)
        alphas (np.ndarray):
            Alpha value for each label.
            shape (n_labels,) which is the same as the number of unique labels len(np.unique(labels))
    """
    labels_cat = np.concatenate(labels) if (isinstance(labels, list) and (isinstance(labels[0], list) or isinstance(labels[0], np.ndarray))) else labels.copy()
    if alphas is None:
        alphas = np.ones(len(labels_cat))
    
    h, w = FOV_height, FOV_width

    rois = scipy.sparse.vstack(spatialFootprints)
    rois = rois.multiply(1.2/rois.max(1).A).power(1)

    u = np.unique(labels_cat)

    n_c = len(u)

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
    rois_c = rois_c.multiply(alphas[labels_squeezed][:,None]).tocsr()

    boolSessionID = np.concatenate([[np.arange(len(labels))==ii]*len(labels[ii]) for ii in range(len(labels))] , axis=0) if boolSessionID is None else boolSessionID
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


def confusion_matrix(cm, **params):
    default_params = dict(
        annot=True,
        annot_kws={"size": 16},
        vmin=0.,
        vmax=1.,
        cmap=plt.get_cmap('gray')
    )
    for key in params:
        default_params[key] = params[key]
    sns.heatmap(cm, **default_params)