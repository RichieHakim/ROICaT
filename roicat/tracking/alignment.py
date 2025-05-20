from typing import List, Tuple, Union, Optional, Dict, Any, Sequence, Callable

import warnings
import functools
import PIL
from pathlib import Path
import math

import numpy as np
import scipy.optimize
import scipy.sparse
import torch
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

from .. import helpers, util

class Aligner(util.ROICaT_Module):
    """
    A class for registering ROIs to a template FOV. Currently relies on
    available OpenCV methods for rigid and non-rigid registration.
    RH 2023

    Args:
        use_match_search (bool):
            Whether to densely search all possible paths to match images to the
            template upon failure. (Default is ``True``)
        all_to_all (bool):
            Whether to start with an all-to-all matching approach using out
            match_search algorithm. Much slower (N vs. N^2) but useful if you
            know you are working with challenging data.
        radius_in (float):
            Value in micrometers used to define the maximum shift/offset between
            two images that are considered to be aligned. Use larger values for
            more lenient alignment requirements. (Default is *4*)
        radius_out (float):
            Value in micrometers used to define the minimum shift/offset between
            two images that are considered to be misaligned. Use smaller values
            for more stringent alignment requirements. (Default is *20*)
        order (int):
            The order of the Butterworth filter used to define the 'in' and 'out'
            regions of the ImageAlignmentChecker class. (Default is *5*)
        z_threshold (float):
            Z-score required to define two images as aligned. Larger values
            results in more stringent alignment requirements and possibly slower
            registration. Value is the threshold used on the 'z_in' output of
            the ImageAlignmentChecker class to determine if two images are
            properly aligned. (Default is *4.0*)
        um_per_pixel (float):
            The number of micrometers per pixel in the FOV images. (Default is *1.0*)
        device (str):
            The torch device used for various steps in the alignment process.
            (Default is ``'cpu'``)
        verbose (bool):
            Whether to print progress updates. (Default is ``True``)
    """
    def __init__(
        self,
        use_match_search: bool = True,
        all_to_all: bool = False,
        radius_in: float = 4,
        radius_out: float = 20,
        order: int = 5,
        z_threshold: float = 4.0,
        um_per_pixel: float = 1.0,
        device: str = 'cpu',
        verbose: bool = True,
    ):
        """
        Initialize the Aligner class.
        """
        super().__init__()

        ## Store parameter (but not data) args as attributes
        self.params['__init__'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'use_match_search',
                'all_to_all',
                'radius_in',
                'radius_out',
                'order',
                'z_threshold',
                'um_per_pixel',
                'device',
                'verbose',
            ],
        )

        self.use_match_search = use_match_search
        self.all_to_all = all_to_all
        self.radius_in = radius_in
        self.radius_out = radius_out
        self.order = order
        self.z_threshold = z_threshold
        self.device = device

        assert isinstance(um_per_pixel, (int, float, np.number)), 'um_per_pixel must be a single value. If the FOV images have different pixel sizes, then our approach to checking image alignment (using the ImageAlignmentChecker class) will not work smoothly. Please preprocess the images to have the same pixel size or contact the developers for a custom solution.'
        self.um_per_pixel = float(um_per_pixel)
        self._verbose = verbose
        
        self.remappingIdx_geo = None
        self.warp_matrices = None

        self.remappingIdx_nonrigid = None

        self._HW = None

    def augment_FOV_images(
        self,
        FOV_images: List[np.ndarray],
        spatialFootprints: Optional[List[scipy.sparse.csr_matrix]] = None,
        normalize_FOV_intensities: bool = True,
        roi_FOV_mixing_factor: float = 0.5,
        use_CLAHE: bool = True,
        CLAHE_grid_block_size: int = 10,
        CLAHE_clipLimit: int = 1,
        CLAHE_normalize: bool = True,
    ) -> None:
        """
        Augments the FOV images by mixing the FOV with the ROI images and
        optionally applying CLAHE.
        RH 2023

        Args:
            FOV_images (List[np.ndarray]): 
                A list of FOV images.
            spatialFootprints (Optional[List[scipy.sparse.csr_matrix]]):
                A list of spatial footprints for each ROI. If ``None``, then no
                mixing will be performed. (Default is ``None``)
            normalize_FOV_intensities (bool):
                Whether to normalize the FOV images. Setting this to ``True``
                will scale each FOV image to the same intensity range. (Default
                is ``True``)
            roi_FOV_mixing_factor (float):
                The factor by which to mix the ROI images into the FOV images.
                If 0, then no mixing will be performed. (Default is *0.5*)
            use_CLAHE (bool):
                Whether to apply CLAHE to the images. (Default is ``True``)
            CLAHE_grid_block_size (int):
                The size of the blocks in the grid for CLAHE. Used to divide the
                image into small blocks and create the grid_size parameter for
                the cv2.createCLAHE function. Smaller block sizes will result in
                more local CLAHE. (Default is *50*)
            CLAHE_clipLimit (int):
                The clip limit for CLAHE. See cv2.createCLAHE for more details.
                (Default is *1*)
            CLAHE_normalize (bool):
                Whether to normalize the CLAHE output. See alignment.clahe for
                more details. (Default is ``True``)

        Returns:
            List[np.ndarray]:
                The augmented FOV images.
        """
        ## Warn if roi_FOV_mixing_factor = 0 but spatialFootprints is not None
        if (roi_FOV_mixing_factor == 0) and (spatialFootprints is not None):
            warnings.warn("roi_FOV_mixing_factor = 0 but spatialFootprints is not None. The ROI images will not be used.")
        ## Warn if roi_FOV_mixing_factor != 0 but spatialFootprints is None
        if (roi_FOV_mixing_factor != 0) and (spatialFootprints is None):
            warnings.warn("roi_FOV_mixing_factor != 0 but spatialFootprints is None. No mixing will be performed.")

        ## Store parameter (but not data) args as attributes
        self.params['augment_FOV_images'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'normalize_FOV_intensities',
                'roi_FOV_mixing_factor',
                'use_CLAHE',
                'CLAHE_grid_block_size',
                'CLAHE_clipLimit',
                'CLAHE_normalize',
            ],
        )
        
        h,w = FOV_images[0].shape
        sf = spatialFootprints

        ## Cast to float32
        FOV_images = [im.astype(np.float32) for im in FOV_images]

        ## Do the CLAHE first
        if use_CLAHE:
            CLAHE_grid_size = (max(1, h // CLAHE_grid_block_size), max(1, w // CLAHE_grid_block_size))
            FOV_images = [clahe(im, grid_size=CLAHE_grid_size, clipLimit=CLAHE_clipLimit, normalize=CLAHE_normalize) for im in FOV_images]

        ## normalize FOV images
        if normalize_FOV_intensities:
            val_max = max([np.max(im) for im in FOV_images])
            val_min = min([np.min(im) for im in FOV_images])
            FOV_images = [(im - val_min) / (val_max - val_min) for im in FOV_images]

        ## mix ROI images into FOV images
        if spatialFootprints is not None:
            mixing_factor_final = roi_FOV_mixing_factor * np.mean(np.concatenate([im.reshape(-1) for im in FOV_images]))
            fn_mix = lambda im, sf, f: (1 - f) * im + np.array((f) * mixing_factor_final * sf.multiply(1/sf.max(1).toarray()).sum(0).reshape(h, w))
            FOV_images = [fn_mix(f, s, roi_FOV_mixing_factor) for f, s in zip(FOV_images, sf)]

        return FOV_images

    def fit_geometric(
        self,
        template: Union[int, np.ndarray],
        ims_moving: List[np.ndarray],
        template_method: str = 'sequential',
        mask_borders: Tuple[int, int, int, int] = (0, 0, 0, 0),
        method: str = 'RoMa',
        kwargs_method: dict = {
            'RoMa': {
                'model_type': 'outdoor',
                'n_points': 10000,  ## Higher values mean more points are used for the registration. Useful for larger FOV_images. Larger means slower.
                'batch_size': 1000,
            },
            'LoFTR': {
                'model_type': 'indoor_new',
                'threshold_confidence': 0.2,  ## Higher values means fewer but better matches.
            },
            'ECC_cv2': {
                'mode_transform': 'euclidean',  ## Must be one of {'translation', 'affine', 'euclidean', 'homography'}. See cv2 documentation on findTransformECC for more details.
                'n_iter': 200,
                'termination_eps': 1e-09,  ## Termination criteria for the registration algorithm. See documentation for more details.
                'gaussFiltSize': 31,  ## Size of the gaussian filter used to smooth the FOV_image before registration. Larger values mean more smoothing.
                'auto_fix_gaussFilt_step': 10,  ## If the registration fails, then the gaussian filter size is reduced by this amount and the registration is tried again.
            },
            'DISK_LightGlue': {
                'num_features': 2048,  ## Number of features to extract and match. I've seen best results around 2048 despite higher values typically being better.
                'threshold_confidence': 0.2,  ## Higher values means fewer but better matches.
            },
            'SIFT': {
                'nfeatures': 10000,
                'contrastThreshold': 0.04,
                'edgeThreshold': 10,
                'sigma': 1.6,
            },
            'ORB': {
                'nfeatures': 1000,
                'scaleFactor': 1.2,
                'nlevels': 8,
                'edgeThreshold': 31,
                'firstLevel': 0,
                'WTA_K': 2,
                'scoreType': 0,
                'patchSize': 31,
                'fastThreshold': 20,
            },
            'PhaseCorrelation': {
                'bandpass_freqs': [1, 30],
                'order': 5,
            },
        },
        constraint: str = 'affine',
        kwargs_RANSAC: dict = {
            'inl_thresh': 2.0,
            'max_iter': 10,
            'confidence': 0.99,
        },
        verbose: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Performs geometric registration of ``ims_moving`` to a template using
        the specified method. 
        RH 2023

        Args:
            template (Union[int, np.ndarray]): 
                The template image or index. If ``template_method`` is 'image',
                this should be an image (np.ndarray) or an index of the image to
                use as the template. If ``template_method`` is 'sequential',
                then template is the integer index or fractional index of the
                image to use as the template.
            ims_moving (List[np.ndarray]): 
                List of images to be aligned.
            template_method (str): 
                Method to use for template selection. * 'image': use the image
                specified by 'template'. * 'sequential': register each image to
                the previous or next image. (Default is 'sequential')
            mask_borders (Tuple[int, int, int, int]): 
                Border mask for the image. Format is (top, bottom, left, right).
                (Default is (0, 0, 0, 0))
            method (str):
                The method to use for registration. One of {'RoMa', 'LoFTR',
                'ECC_cv2', 'DISK_LightGlue', 'SIFT', 'ORB'}.\n
                * 'RoMa': Feature-based registration using the RoMa algorithm.
                * 'LoFTR': Feature-based registration using LoFTR.
                * 'ECC_cv2': Direct intensity-based registration using OpenCV's
                  findTransformECC.
                * 'DISK_LightGlue': Feature-based registration using DISK
                  features and LightGlue matcher.
                * 'SIFT': Feature-based registration using SIFT keypoints.
                * 'ORB': Feature-based registration using ORB keypoints.
                (Default is 'RoMa')
            kwargs_method (dict):
                Keyword arguments for the selected method. The keys are method
                names, and the values are dictionaries of keyword arguments.
                For example:
                    'RoMa': {
                        'model_type': 'outdoor',
                        'n_points': 10000,
                        'batch_size': 1000,
                    },
                    'LoFTR': {
                        'model_type': 'indoor_new',
                        'threshold_confidence': 0.2,
                    },
                    ...            
            constraint (str):
                The type of transformation to use for the registration. One of:
                * 'rigid': Rigid transformation (translation only)
                * 'euclidean': Euclidean transformation (translation + rotation)
                * 'affine': Affine transformation (translation + rotation + scale + shear)
                * 'homography': Homography transformation (translation + rotation + scale + shear + perspective)
            kwargs_RANSAC (dict):
                Keyword arguments for RANSAC algorithm used in homography
                estimation.
                * 'inl_thresh' (float): RANSAC inlier threshold. (Default is
                  2.0)
                * 'max_iter' (int): Maximum number of iterations for RANSAC.
                  (Default is 10)
                * 'confidence' (float): Confidence level for RANSAC. (Default is
                  0.99)
            verbose (Optional[bool]):
                Whether to print progress updates. If ``None``, the verbose
                level set during initialization will be used.

        Returns:
            np.ndarray:
                An array of shape (N, H, W, 2) representing the remap field for
                N images.
        """
        ## Store parameter (but not data) args as attributes
        self.params['fit_geometric'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'template',
                'template_method',
                'mask_borders',
                'method',
                'kwargs_method',
                'constraint',
                'kwargs_RANSAC',
                'verbose',
            ],
        )

        verbose = verbose if verbose is not None else self._verbose

        methods_lut = {
            'RoMa': RoMa,
            'LoFTR': LoFTR,
            'ECC_cv2': ECC_cv2,
            'DISK_LightGlue': DISK_LightGlue,
            'PhaseCorrelation': PhaseCorrelationRegistration,
            'SIFT': SIFT,
            'ORB': ORB,
        }
        assert method in methods_lut, f"method must be one of {methods_lut.keys()}"
        model = methods_lut[method](
            device=self.device, 
            verbose=verbose,
            **kwargs_method[method],
        )
        
        # Check if ims_moving is a non-empty list
        assert len(ims_moving) > 0, "ims_moving must be a non-empty list of images."
        # Check if all images in ims_moving have the same shape
        shape = ims_moving[0].shape
        for im in ims_moving:
            assert im.shape == shape, "All images in ims_moving must have the same shape."
        # Check if template_method is valid
        valid_template_methods = {'sequential', 'image'}
        assert template_method in valid_template_methods, f"template_method must be one of {valid_template_methods}"

        ims_moving, template = self._fix_input_images(ims_moving=ims_moving, template=template, template_method=template_method)

        H, W = ims_moving[0].shape
        self._HW = (H,W) if self._HW is None else self._HW

        def _register(
            ims_moving: List[np.ndarray],
            template: Union[int, np.ndarray],
            template_method: str,
        ):
            ims_moving = [self._crop_image(im, mask_borders) for im in ims_moving]
            template = self._crop_image(template, mask_borders) if isinstance(template, np.ndarray) else template

            warp_matrices_raw = []
            for ii, im_moving in tqdm(enumerate(ims_moving), desc='Finding geometric registration warps', total=len(ims_moving), disable=not self._verbose):
                if template_method == 'sequential':
                    ## warp images before template forward (t1->t2->t3->t4)
                    if ii < template:
                        im_template = ims_moving[ii+1]
                    ## warp template to itself
                    elif ii == template:
                        im_template = ims_moving[ii]
                    ## warp images after template backward (t4->t3->t2->t1)
                    elif ii > template:
                        im_template = ims_moving[ii-1]
                elif template_method == 'image':
                    im_template = template

                warp_matrix = model.fit_rigid(
                    im_template=im_template,
                    im_moving=im_moving,
                    constraint=constraint,
                    **kwargs_RANSAC,
                )
                warp_matrices_raw.append(warp_matrix)

            # compose warp transforms
            warp_matrices = []
            if template_method == 'sequential':
                ## compose warps before template forward (t1->t2->t3->t4)
                for ii in np.arange(0, template):
                    warp_composed = self._compose_warps(
                        warp_0=warp_matrices_raw[ii], 
                        warps_to_add=warp_matrices_raw[ii+1:template+1],
                        warpMat_or_remapIdx='warpMat',
                    )
                    warp_matrices.append(warp_composed)
                ## compose template to itself
                warp_matrices.append(warp_matrices_raw[template])
                ## compose warps after template backward (t4->t3->t2->t1)
                for ii in np.arange(template+1, len(ims_moving)):
                    warp_composed = self._compose_warps(
                        warp_0=warp_matrices_raw[ii], 
                        warps_to_add=warp_matrices_raw[template:ii][::-1],
                        warpMat_or_remapIdx='warpMat',
                    )
                    warp_matrices.append(warp_composed)
            ## no composition when template_method == 'image'
            elif template_method == 'image':
                warp_matrices = warp_matrices_raw

            ## Extend 2x3 affine warp matrix into 3x3 homography matrix if necessary
            def _extend_warp_matrix(warp_matrix):
                if warp_matrix.shape == (2, 3):
                    warp_matrix = np.vstack([warp_matrix, [0, 0, 1]])
                elif warp_matrix.shape == (3, 3):
                    pass
                else:
                    raise ValueError(f'Unexpected warp_matrix shape: {warp_matrix.shape}')
                return warp_matrix
            warp_matrices = [_extend_warp_matrix(warp_matrix) for warp_matrix in warp_matrices]
            warp_matrices = np.stack(warp_matrices, axis=0)  ## shape: (N, 3, 3)

            return warp_matrices
        
        ## Run initial registration
        warp_matrices_all_to_template = _register(ims_moving=ims_moving, template=template, template_method=template_method)  ## shape: [(3, 3) * n_images]

        # check alignment
        im_template_global = ims_moving[template] if template_method == 'sequential' else template
        ## warp the images
        remappingIdx_geo_all_to_template = [helpers.warp_matrix_to_remappingIdx(warp_matrix=warp_matrix, x=W, y=H) for warp_matrix in warp_matrices_all_to_template]
        images_warped_all_to_template = self.transform_images(ims_moving=ims_moving, remappingIdx=remappingIdx_geo_all_to_template)

        ## Prepare the ImageAlignmentChecker object
        iac_geo = helpers.ImageAlignmentChecker(
            hw=tuple(self._HW),
            radius_in=self.radius_in * self.um_per_pixel,
            radius_out=self.radius_out * self.um_per_pixel,
            order=self.order,
            device='cpu',
        )
        score_template_to_all = iac_geo.score_alignment(
            images=images_warped_all_to_template,
            images_ref=im_template_global,
        )['z_in'][:, 0]
        alignment_template_to_all = score_template_to_all > self.z_threshold
        idx_not_aligned = np.where(alignment_template_to_all == False)[0]
        print(f"Alignment z_in scores: {[float(f'{score:.1f}') for score in score_template_to_all]}. z_threshold: {self.z_threshold}.") if verbose else None

        alignment_all_to_all = None
        score_all_to_all = None
        if (len(idx_not_aligned) > 0) or self.all_to_all:  ## if any images are not aligned
            idx_toSearch = idx_not_aligned if not self.all_to_all else np.arange(len(ims_moving))
            if (len(idx_not_aligned) > 0):
                print(f'Warning: Alignment failed for some images (probability_not_aligned > probability(z_threshold)) for images idx: {idx_toSearch}.')
            if self.all_to_all:
                print(f"Performing all-to-all matching using the match_search algorithm.")
            
            
            if self.use_match_search or self.all_to_all:
                print('Attempting to find best matches using match search algorithm...')
                ## Make a function that wraps up all the above steps
                def _update_warps(
                    idx: Union[np.ndarray, List[int]],  ## indices of images to use as templates
                    warp_matrices_all_to_all: np.ndarray,  ## shape: (N, N, 3, 3). Rows are to, columns are from
                    alignment_all_to_all: np.ndarray,
                    score_all_to_all: np.ndarray,
                ):
                    ## Register all ims_moving to the template (which is one of the ims_moving defined by idx_template)
                    for idx_current in idx:
                        ## Run the registration algo
                        im_current = ims_moving[idx_current]
                        warp_matrices_all_to_all[idx_current] = (_register(ims_moving=ims_moving, template=im_current, template_method='image'))  ## returns shape: (N, 3, 3)
                        ## Make the transpose the inverse of the warp matrices
                        # warp_matrices_all_to_all[:, idx_current] = np.linalg.inv(warp_matrices_all_to_all[idx_current])  ## inv along last two axes
                        ## warp the images
                        remappingIdx_geo_all_to_current = [helpers.warp_matrix_to_remappingIdx(warp_matrix=warp_matrix, x=W, y=H) for warp_matrix in warp_matrices_all_to_all[idx_current]]
                        images_warped_all_to_current = self.transform_images(ims_moving=ims_moving, remappingIdx=remappingIdx_geo_all_to_current)
                        ## Check alignment
                        score_all_to_all[idx_current] = iac_geo.score_alignment(images=images_warped_all_to_current, images_ref=im_current)['z_in'][:, 0]  ## shape: (N,)
                        # score_all_to_all[:, idx_current] = score_all_to_all[idx_current]  ## add the transpose of the score matrix
                        alignment_all_to_all[idx_current] = score_all_to_all[idx_current] > self.z_threshold  ## returns a boolean array of shape (n_images,)
                        # alignment_all_to_all[:, idx_current] = alignment_all_to_all[idx_current]  ## add the transpose of the alignment matrix
                        
                    ## Recompute warp matrices based on shortest path between each image and the template (through the all_to_all alignment matrix)
                    ### Make a connection graph by appending the template alignment_matrix (1D) on top of the all_to_all alignment_matrix (N x N)
                    alignment_matrix_all_to_all_and_template = np.concatenate([np.concatenate([np.array(0)[None,], alignment_template_to_all])[None, :], np.concatenate([alignment_template_to_all[:, None], np.nan_to_num(alignment_all_to_all, nan=0.0)], axis=1)], axis=0)
                    ### Make a cost graph
                    cost_all_to_all_and_template = np.concatenate([np.concatenate([np.array(0)[None,], score_template_to_all])[None, :], np.concatenate([score_template_to_all[:, None], np.nan_to_num(score_all_to_all, nan=0.0)], axis=1)], axis=0)
                    cost_all_to_all_and_template[cost_all_to_all_and_template == 0] = np.inf  ## set 0s to inf
                    cost_all_to_all_and_template = (1 / cost_all_to_all_and_template) * alignment_matrix_all_to_all_and_template.astype(np.float32)  ## 0s are disconnected, 1s are connected
                    cost_all_to_all_and_template[np.arange(len(cost_all_to_all_and_template)), np.arange(len(cost_all_to_all_and_template))] = 0.0  ## set the diagonal to 0
                    ### Use dijkstra's algorithm to find the shortest path from every image to the template
                    distances, predecessors = scipy.sparse.csgraph.shortest_path(
                        csgraph=scipy.sparse.csr_matrix(cost_all_to_all_and_template.astype(np.float32)),  ## 0s are disconnected, 1s are connected
                        method='D',  ## Dijkstra's algorithm
                        directed=True,
                        return_predecessors=True,
                        unweighted=False,  ## Just count the number of connected edges
                    )

                    ## Make new warp matrices from each image to the template
                    warp_matrices_all_to_template_new = []
                    for idx_im in range(len(ims_moving)):
                        if not np.isinf(distances[0, idx_im+1]):  ## distance to idx 0 from idx_im+1 is not infinite
                            path = helpers.get_path_between_nodes(
                                idx_start=idx_im+1,  ## +1 because the first row and column are the template
                                idx_end=0,  ## 0 is the template
                                predecessors=predecessors,
                            )
                            warps_to_add = [warp_matrices_all_to_all[idx_to - 1, idx_from - 1] for idx_from, idx_to in zip(path[:-2], path[1:-1])]  ## add the warps along the path
                            warps_to_add += [warp_matrices_all_to_template[path[-2] - 1]]  ## add the warp to the template
                            warp_matrix_current_to_template = self._compose_warps(
                                warp_0=np.eye(3, 3, dtype=np.float32),  ## start with identity matrix
                                warps_to_add=warps_to_add,
                                warpMat_or_remapIdx='warpMat',
                            )
                            warp_matrices_all_to_template_new.append(warp_matrix_current_to_template)
                        else:
                            warp_matrices_all_to_template_new.append(np.eye(3, 3, dtype=np.float32))

                    ## compute warped images and check alignment
                    remappingIdx_geo_all_to_template_new = [helpers.warp_matrix_to_remappingIdx(warp_matrix=warp_matrix, x=W, y=H) for warp_matrix in warp_matrices_all_to_template_new]
                    images_warped_all_to_template_new = self.transform_images(ims_moving=ims_moving, remappingIdx=remappingIdx_geo_all_to_template_new)
                    score_template_to_all_new = iac_geo.score_alignment(images=images_warped_all_to_template_new, images_ref=im_template_global)['z_in'][:, 0]
                    alignment_template_to_all_new = score_template_to_all_new > self.z_threshold
                    idx_no_path = np.where(np.logical_not(alignment_template_to_all_new))[0]

                    ## Check if there are any failed paths to the template
                    if len(idx_no_path) > 0:
                        print(f'Warning: Could not find a path to alignment after path finding for images idx: {idx_no_path}')

                    return warp_matrices_all_to_template_new, warp_matrices_all_to_all, alignment_all_to_all, idx_no_path
                    
                warp_matrices_all_to_all = np.tile(np.eye(3, 3)[None, None, :, :], reps=(len(ims_moving), len(ims_moving), 1, 1))
                # alignment_all_to_all = np.eye(len(ims_moving), dtype=np.bool_)
                alignment_all_to_all = np.ones((len(ims_moving), len(ims_moving)), dtype=np.float32) * np.nan
                score_all_to_all = np.ones((len(ims_moving), len(ims_moving)), dtype=np.float32) * np.nan
                print(f'Finding alignment between images idx: {idx_toSearch} and all other images...')
                ## Register the images in idx to the template
                warp_matrices_all_to_template_new, warp_matrices_all_to_all, alignment_all_to_all, idx_no_path = _update_warps(
                    idx=idx_toSearch, 
                    warp_matrices_all_to_all=warp_matrices_all_to_all, 
                    alignment_all_to_all=alignment_all_to_all,
                    score_all_to_all=score_all_to_all,
                )
                warp_matrices_all_to_template = warp_matrices_all_to_template_new
                if len(idx_no_path) == 0:
                    print('All images aligned successfully after one round of path finding.') if self._verbose else None
                    warp_matrices_all_to_template = warp_matrices_all_to_template_new
                else:
                    idx_remaining = sorted(list(set(list(range(len(ims_moving)))) - set(idx_toSearch)))
                    warnings.warn(f'Warning: Could not find a path to alignment for image idx: {idx_no_path}. Now doing a dense search for alignment between all images...')
                    print(f"Finding alignment between remaining images and all other images: {idx_remaining}...") if self._verbose else None
                    ## Register the images in idx to the template
                    warp_matrices_all_to_template_new, warp_matrices_all_to_all, alignment_all_to_all, idx_no_path = _update_warps(
                        idx=idx_remaining, 
                        warp_matrices_all_to_all=warp_matrices_all_to_all, 
                        alignment_all_to_all=alignment_all_to_all,
                        score_all_to_all=score_all_to_all,
                    )
                    if len(idx_no_path) == 0:
                        print('All images aligned successfully after dense search.') if self._verbose else None
                        warp_matrices_all_to_template = warp_matrices_all_to_template_new
                    else:
                        warnings.warn(f"Warning: Could not find a path to alignment for image idx: {idx_no_path}. Some images may not be aligned.")

            else:
                warnings.warn(f"Alignment failed for some images (probability_not_aligned > probability(z_threshold)) for images idx: {idx_not_aligned}. Use 'use_match_search=True' to attempt to find best matches for misaligned images.")                
        else:
            print('All images aligned successfully!') if self._verbose else None
            alignment_all_to_all = None
        
        ### Convert warp matrices to remap indices
        self.remappingIdx_geo = [helpers.warp_matrix_to_remappingIdx(warp_matrix=warp_matrix, x=W, y=H) for warp_matrix in warp_matrices_all_to_template]

        ### Make the registered images
        self.ims_registered_geo = self.transform_images(ims_moving=ims_moving, remappingIdx=self.remappingIdx_geo)
        ### Compute the new alignment scores
        score_all_to_all_final = iac_geo.score_alignment(images=self.ims_registered_geo)['z_in']
        alignment_all_to_all_final = score_all_to_all_final > self.z_threshold
        score_template_to_all_final = iac_geo.score_alignment(images=self.ims_registered_geo, images_ref=im_template_global)['z_in'][:, 0]
        alignment_template_to_all_final = score_template_to_all_final > self.z_threshold


        ## Prepare outputs
        self.results_geometric = {
            'warp_matrices': warp_matrices_all_to_template,
            'image_alignment_checker': iac_geo,
            'direct': {
                'alignment_template_to_all': alignment_template_to_all,
                'score_template_to_all': score_template_to_all,

                'alignment_all_to_all': alignment_all_to_all,
                'score_all_to_all': score_all_to_all,
            },
            'final': {
                'alignment_all_to_all': alignment_all_to_all_final,
                'score_all_to_all': score_all_to_all_final,
                
                'alignment_template_to_all': alignment_template_to_all_final,
                'score_template_to_all': score_template_to_all_final,
            },
        }

        return self.remappingIdx_geo


    def fit_nonrigid(
        self,
        template: Union[int, np.ndarray],
        ims_moving: List[np.ndarray],
        remappingIdx_init: Optional[np.ndarray] = None,
        template_method: str = 'sequential',
        method: str = 'RoMa',
        kwargs_method: dict = {
            'RoMa': {
                'model_type': 'outdoor',
            },
            'DeepFlow': {},
            'OpticalFlowFarneback': {
                'pyr_scale': 0.7,
                'levels': 5,
                'winsize': 128,
                'iterations': 15,
                'poly_n': 5,
                'poly_sigma': 1.5,            
            },
        },
    ) -> np.ndarray:
        """
        Performs non-rigid registration of ``ims_moving`` to a template using
        the specified method.
        RH 2023

        Args:
            template (Union[int, np.ndarray]): 
                The template image or index. If ``template_method`` is 'image',
                this should be an image (np.ndarray) or an index of the image to
                use as the template. If ``template_method`` is 'sequential',
                then template is the integer index or fractional index of the
                image to use as the template.
            ims_moving (List[np.ndarray]): 
                A list of images to be aligned.
            remappingIdx_init (Optional[np.ndarray]): 
                An array of shape (N, H, W, 2) representing any initial remap
                field to apply to the images in ``ims_moving``. The output of
                this method will be composed with ``remappingIdx_init``.
                (Default is ``None``)
            template_method (str): 
                Method to use for template selection.
                * 'image': use the image specified by 'template'.
                * 'sequential': register each image to the previous or next
                  image.
                (Default is 'sequential')
            method (str):
                The method to use for registration. One of {'RoMa', 'DeepFlow',
                'OpticalFlowFarneback'}.
                * 'DeepFlow': Optical flow using OpenCV's DeepFlow algorithm.
                * 'RoMa': Non-rigid registration using the RoMa algorithm.
                * 'OpticalFlowFarneback': Optical flow using OpenCV's
                  calcOpticalFlowFarneback.
                (Default is 'RoMa')
            kwargs_method (dict):
                Keyword arguments for the selected method. The keys are method
                names, and the values are dictionaries of keyword arguments.

        Returns:
            np.ndarray:
                An array of shape (N, H, W, 2) representing the remap field for N images.
        """
        # Check if ims_moving is a non-empty list
        assert len(ims_moving) > 0, "ims_moving must be a non-empty list of images."
        # Check if all images in ims_moving have the same shape
        shape = ims_moving[0].shape
        for im in ims_moving:
            assert im.shape == shape, "All images in ims_moving must have the same shape."
        # Check if template_method is valid
        valid_template_methods = {'sequential', 'image'}
        assert template_method in valid_template_methods, f"template_method must be one of {valid_template_methods}"

        ## Store parameter (but not data) args as attributes
        self.params['fit_nonrigid'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'template',
                'template_method',
                'method',
                'kwargs_method',
            ],
        )

        methods_lut = {
            'RoMa': RoMa,
            'DeepFlow': DeepFlow,
            'OpticalFlowFarneback': OpticalFlowFarneback,
        }
        assert method in methods_lut, f"method must be one of {methods_lut.keys()}"
        model = methods_lut[method](
            device=self.device, 
            verbose=self._verbose,
            **kwargs_method[method],
        )

        # Warn if any images have values below 0 or NaN
        found_0 = np.any([np.any(im < 0) for im in ims_moving])
        found_nan = np.any([np.any(np.isnan(im)) for im in ims_moving])
        warnings.warn(f"Found images with values below 0: {found_0}. Found images with NaN values: {found_nan}") if found_0 or found_nan else None

        H, W = ims_moving[0].shape
        self._HW = (H,W) if self._HW is None else self._HW

        ims_moving, template = self._fix_input_images(ims_moving=ims_moving, template=template, template_method=template_method)
        norm_factor = np.nanmax([np.nanmax(im) for im in ims_moving])
        template_norm   = np.array(template * (template > 0) * (1/norm_factor) * 255, dtype=np.uint8) if template_method == 'image' else None
        ims_moving_norm = [np.array(im * (im > 0) * (1/np.nanmax(im)) * 255, dtype=np.uint8) for im in ims_moving]

        print(f'Finding nonrigid registration warps with mode: {method}, template_method: {template_method}') if self._verbose else None
        remappingIdx_raw = []
        for ii, im_moving in tqdm(enumerate(ims_moving_norm), desc='Finding nonrigid registration warps', total=len(ims_moving_norm), unit='image', disable=not self._verbose):
            if template_method == 'sequential':
                ## warp images before template forward (t1->t2->t3->t4)
                if ii < template:
                    im_template = ims_moving_norm[ii+1]
                ## warp template to itself
                elif ii == template:
                    im_template = ims_moving_norm[ii]
                ## warp images after template backward (t4->t3->t2->t1)
                elif ii > template:
                    im_template = ims_moving_norm[ii-1]
            elif template_method == 'image':
                im_template = template_norm

            remappingIdx_raw.append(model.fit_nonrigid(
                im_template=im_template,
                im_moving=im_moving,
            ))

        # compose warp transforms
        print('Composing nonrigid warp matrices...') if self._verbose else None
        
        self.remappingIdx_nonrigid = []
        if template_method == 'sequential':
            ## compose warps before template forward (t1->t2->t3->t4)
            for ii in np.arange(0, template):
                warp_composed = self._compose_warps(
                    warp_0=remappingIdx_raw[ii], 
                    warps_to_add=remappingIdx_raw[ii+1:template+1],
                    warpMat_or_remapIdx='remapIdx',
                )
                self.remappingIdx_nonrigid.append(warp_composed)
            ## compose template to itself
            self.remappingIdx_nonrigid.append(remappingIdx_raw[template])
            ## compose warps after template backward (t4->t3->t2->t1)
            for ii in np.arange(template+1, len(ims_moving)):
                warp_composed = self._compose_warps(
                    warp_0=remappingIdx_raw[ii], 
                    warps_to_add=remappingIdx_raw[template:ii][::-1],
                    warpMat_or_remapIdx='remapIdx',
                    )
                self.remappingIdx_nonrigid.append(warp_composed)
        ## no composition when template_method == 'image'
        elif template_method == 'image':
            self.remappingIdx_nonrigid = remappingIdx_raw

        if remappingIdx_init is not None:
            self.remappingIdx_nonrigid = [self._compose_warps(warp_0=remappingIdx_init[ii], warps_to_add=[warp], warpMat_or_remapIdx='remapIdx') for ii, warp in enumerate(self.remappingIdx_nonrigid)]

        return self.remappingIdx_nonrigid
    
        
    def transform_images_geometric(
        self, 
        ims_moving: List[np.ndarray], 
        remappingIdx: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Transforms images based on geometric registration warps.

        Args:
            ims_moving (np.ndarray): 
                The images to be transformed. *(N, H, W)*
            remappingIdx (Optional[np.ndarray]): 
                An array specifying how to remap the images. If ``None``, the
                remapping index from geometric registration is used. (Default is
                ``None``)

        Returns:
            (np.ndarray): 
                ims_registered_geo (np.ndarray):
                    The images after applying the geometric registration warps.
                    *(N, H, W)*
        """
        remappingIdx = self.remappingIdx_geo if remappingIdx is None else remappingIdx
        print('Applying geometric registration warps to images...') if self._verbose else None
        self.ims_registered_geo = self.transform_images(ims_moving=ims_moving, remappingIdx=remappingIdx)
        return self.ims_registered_geo
    
    def transform_images_nonrigid(
        self, 
        ims_moving: List[np.ndarray], 
        remappingIdx: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Transforms images based on non-rigid registration warps.

        Args:
            ims_moving (np.ndarray): 
                The images to be transformed. *(N, H, W)*
            remappingIdx (Optional[np.ndarray]): 
                An array specifying how to remap the images. If ``None``, the
                remapping index from non-rigid registration is used. (Default is
                ``None``)

        Returns:
            (np.ndarray): 
                ims_registered_nonrigid (np.ndarray): 
                    The images after applying the non-rigid registration warps.
                    *(N, H, W)*
        """
        remappingIdx = self.remappingIdx_nonrigid if remappingIdx is None else remappingIdx
        print('Applying nonrigid registration warps to images...') if self._verbose else None
        self.ims_registered_nonrigid = self.transform_images(ims_moving=ims_moving, remappingIdx=remappingIdx)

        ### Make the registered images
        #### Undo any remappingIdx_init
        self.ims_registered_nonrigid = self.transform_images(ims_moving=ims_moving, remappingIdx=self.remappingIdx_nonrigid)
        ### Compute the new alignment scores
        iac_nonrigid = helpers.ImageAlignmentChecker(
            hw=tuple(self._HW),
            radius_in=self.radius_in * self.um_per_pixel,
            radius_out=self.radius_out * self.um_per_pixel,
            order=self.order,
            device='cpu',
        )
        score_all_to_all_final = iac_nonrigid.score_alignment(images=self.ims_registered_geo)['z_in']
        alignment_all_to_all_final = score_all_to_all_final > self.z_threshold

        ## Prepare outputs
        self.results_nonrigid = {
            'image_alignment_checker': iac_nonrigid,
            'final': {
                'alignment_all_to_all': alignment_all_to_all_final,
                'score_all_to_all': score_all_to_all_final,
            },
        }

        return self.ims_registered_nonrigid
    
    def transform_images(
        self, 
        ims_moving: List[np.ndarray], 
        remappingIdx: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Transforms images using the specified remapping index.

        Args:
            ims_moving (List[np.ndarray]): 
                The images to be transformed. List of arrays with shape: *(H,
                W)* or *(H, W, C)*
            remappingIdx (List[np.ndarray]): 
                The remapping index to apply to the images. List of arrays with
                shape: *(H, W, 2)*. List length must match the number of images.

        Returns:
            (List[np.ndarray]): 
                ims_registered (List[np.ndarray]): 
                    The transformed images. *(N, H, W)*
        """
        if not isinstance(ims_moving, (list, tuple)):
            if isinstance(ims_moving, np.ndarray):
                ims_moving = [ims_moving,]
            else:
                raise ValueError('ims_moving must be a list of images.')
            squeeze_output = True
        else:
            squeeze_output = False
        if not isinstance(remappingIdx, (list, tuple)):
            if isinstance(remappingIdx, np.ndarray):
                remappingIdx = [remappingIdx,]
            else:
                raise ValueError('remappingIdx must be a list of remapping indices.')
        
        assert len(ims_moving) == len(remappingIdx), 'Number of images must match number of remapping indices.'

        ims_registered = []
        for ii, (im_moving, remapIdx) in enumerate(zip(ims_moving, remappingIdx)):
            remapper = functools.partial(
                helpers.remap_images,
                remappingIdx=remapIdx,
                backend='cv2',
                interpolation_method='linear',
                border_mode='constant',
                border_value=float(im_moving.mean()),
            )
            im_registered = np.stack([remapper(im_moving[:,:,ii]) for ii in range(im_moving.shape[2])], axis=-1) if im_moving.ndim==3 else remapper(im_moving)
            ims_registered.append(im_registered)
        return ims_registered if not squeeze_output else ims_registered[0]  

    def _crop_image(self, image: np.ndarray, borders: Tuple[int, int, int, int],) -> np.ndarray:
        """
        Crops an image based on the specified borders.

        Args:
            image (np.ndarray): 
                The image to crop. *(H, W)*
            borders (Tuple[int, int, int, int]): 
                The borders to crop. Format is (top, bottom, left, right).

        Returns:
            (np.ndarray): 
                image_cropped (np.ndarray): 
                    The cropped image.
        """
        return image[borders[0]:image.shape[0]-borders[1], borders[2]:image.shape[1]-borders[3]]
        
    def _compose_warps(
        self, 
        warp_0: np.ndarray, 
        warps_to_add: List[np.ndarray], 
        warpMat_or_remapIdx: str = 'remapIdx'
    ) -> np.ndarray:
        """
        Composes a series of warps into a single warp.
        RH 2023

        Args:
            warp_0 (np.ndarray): 
                The initial warp.
            warps_to_add (List[np.ndarray]): 
                A list of warps to add to the initial warp.
            warpMat_or_remapIdx (str): 
                Determines the function to use for composition. Can be either 
                'warpMat' or 'remapIdx'. (Default is 'remapIdx')

        Returns:
            (np.ndarray): 
                warp_out (np.ndarray): 
                    The resulting warp after composition.
        """
        if warpMat_or_remapIdx == 'warpMat':
            fn_compose = helpers.compose_transform_matrices
        elif warpMat_or_remapIdx == 'remapIdx':
            fn_compose = helpers.compose_remappingIdx
        else:
            raise ValueError(f'warpMat_or_remapIdx must be one of ["warpMat", "remapIdx"]')
        
        if len(warps_to_add) == 0:
            return warp_0
        else:
            warp_out = warp_0.copy()
            for warp_to_add in warps_to_add:
                warp_out = fn_compose(warp_out, warp_to_add)
            return warp_out

    def transform_ROIs(
        self, 
        ROIs: np.ndarray, 
        remappingIdx: Optional[np.ndarray] = None,
        normalize: bool = True,
    ) -> List[np.ndarray]:
        """
        Transforms ROIs based on remapping indices and normalization settings.
        RH 2023

        Args:
            ROIs (np.ndarray): 
                The regions of interest to transform. (shape: *(H, W)*)
            remappingIdx (Optional[np.ndarray]): 
                The indices for remapping the ROIs. If ``None``, geometric or
                nonrigid registration must be performed first. (Default is
                ``None``)
            normalize (bool): 
                If ``True``, data is normalized. (Default is ``True``)

        Returns:
            (List[np.ndarray]): 
                ROIs_aligned (List[np.ndarray]): 
                    Transformed ROIs.
        """
        ## Store parameter (but not data) args as attributes
        self.params['transform_ROIs'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'normalize',
            ],
        )

        if remappingIdx is None:
            assert (self.remappingIdx_geo is not None) or (self.remappingIdx_nonrigid is not None), 'If remappingIdx is not provided, then geometric or nonrigid registration must be performed first.'
            remappingIdx = self.remappingIdx_nonrigid if self.remappingIdx_nonrigid is not None else self.remappingIdx_geo

        H, W = remappingIdx[0].shape[:2]

        print('Registering ROIs...') if self._verbose else None
        self.ROIs_aligned = []
        for ii, (remap, rois) in tqdm(enumerate(zip(remappingIdx, ROIs)), total=len(remappingIdx), mininterval=1, disable=not self._verbose, desc='Registering ROIs', position=1):
            rois_aligned = helpers.remap_sparse_images(
                ims_sparse=[roi.reshape((H, W)) for roi in rois],
                remappingIdx=remap,
                method='cubic',
                fill_value=0,
                dtype=np.float32,
                safe=True,
                verbose=False,
            )
            rois_aligned = scipy.sparse.vstack([roi.reshape(1, -1) for roi in rois_aligned])

            if normalize:
                rois_aligned.data[rois_aligned.data < 0] = 0
                rois_aligned_sum = np.array(rois_aligned.sum(1))
                ## convert 0s to NaNs to avoid division by 0
                rois_aligned_sum[rois_aligned_sum == 0] = np.nan
                rois_aligned = rois_aligned.multiply(1/rois_aligned_sum)
                rois_aligned.data[np.isnan(rois_aligned.data)] = 0
                rois_aligned.eliminate_zeros()
            
            ## remove NaNs from ROIs
            rois_aligned = rois_aligned.tocsr()
            rois_aligned.data[np.isnan(rois_aligned.data)] = 0

            self.ROIs_aligned.append(rois_aligned)

        return self.ROIs_aligned

    def get_ROIsAligned_maxIntensityProjection(
        self, 
        H: Optional[int] = None, 
        W: Optional[int] = None,
        normalize: bool = True,
    ) -> List[np.ndarray]:
        """
        Returns the max intensity projection of the ROIs aligned to the template
        FOV.

        Args:
            H (Optional[int]): 
                The height of the output projection. If not provided and if not
                already set, an error will be thrown. (Default is ``None``)
            W (Optional[int]): 
                The width of the output projection. If not provided and if not
                already set, an error will be thrown. (Default is ``None``)
            normalize (bool):
                If ``True``, the ROIs are normalized by the maximum value.
                (Default is ``True``)

        Returns:
            (List[np.ndarray]): 
                max_projection (List[np.ndarray]): 
                    The max intensity projections of the ROIs.
        """
        if H is None:
            assert self._HW is not None, 'H and W must be provided if not already set.'
            H, W = self._HW
        return [(rois.multiply(rois.max(1).power(-1)) if normalize else rois).max(0).toarray().reshape(H, W) for rois in self.ROIs_aligned]
    
    def get_flowFields(
        self, 
        remappingIdx: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Returns the flow fields based on the remapping indices.

        Args:
            remappingIdx (Optional[np.ndarray]): 
                The indices for remapping the flow fields. 
                If ``None``, geometric or nonrigid registration must be performed first. 
                (Default is ``None``)

        Returns:
            (List[np.ndarray]): 
                flow_fields (List[np.ndarray]): 
                    The transformed flow fields.
        """
        if remappingIdx is None:
            assert (self.remappingIdx_geo is not None) or (self.remappingIdx_nonrigid is not None), 'If remappingIdx is not provided, then geometric or nonrigid registration must be performed first.'
            remappingIdx = self.remappingIdx_nonrigid if self.remappingIdx_nonrigid is not None else self.remappingIdx_geo
        return [helpers.remappingIdx_to_flowField(remap) for remap in remappingIdx]
    
    def _fix_input_images(
        self,
        ims_moving: List[np.ndarray],
        template: Union[int, np.ndarray],
        template_method: str,
    ) -> Tuple[int, List[np.ndarray]]:
        """
        Converts the input images and template to float32 dtype if they are not
        already. 

        Warnings are printed for any conversions made. The method for selecting
        the template image can either be **'image'** or **'sequential'**.

        Args:
            ims_moving (List[np.ndarray]):
                A list of input images. Images should be of type ``np.float32``,
                if not, they will be converted to it.
            template (Union[int, np.ndarray]): 
                The index or actual template image. Depending on the
                `template_method`, this could be an integer index, a float
                representing a fractional index, or a numpy array representing
                the actual template image.
            template_method (str):
                Method for selecting the template image. Either \n
                * ``'image'``: template is considered as an image
                 (``np.ndarray``) or as an index (``int`` or ``float``)
                 referring to the list of images (``ims_moving``). 
                * ``'sequential'``: template is considered as a sequential index
                  (``int`` or ``float``) referring to the list of images
                  (``ims_moving``). \n

        Returns:
            (Tuple[int, List[np.ndarray]]): tuple containing:
                template (int):
                    Index of the template in the list of images.
                ims_moving (List[np.ndarray]):
                    List of converted images.

        Example:
            .. highlight:: python
            .. code-block:: python

                ims_moving, template = _fix_input_images(ims_moving, template,
                'image')
        """
        ## convert images to float32 and warn if they are not
        print(f'WARNING: ims_moving are not all dtype: np.float32, found {np.unique([im.dtype for im in ims_moving])}, converting...') if any(im.dtype != np.float32 for im in ims_moving) else None
        ims_moving = [im.astype(np.float32) for im in ims_moving]    

        if template_method == 'image':
            if isinstance(template, int):
                assert 0 <= template < len(ims_moving), f'template must be between 0 and {len(ims_moving)-1}, not {template}'
                print(f'WARNING: template image is not dtype: np.float32, found {ims_moving[template].dtype}, converting...') if ims_moving[template].dtype != np.float32 else None
                template = ims_moving[template]
            elif isinstance(template, float):
                assert 0.0 <= template <= 1.0, f'template must be between 0.0 and 1.0, not {template}'
                idx = int(len(ims_moving) * template)
                print(f'Converting float fractional index to integer index: {template} -> {idx}')
                template = ims_moving[idx] # take the image at the specified fractional index
            elif isinstance(template, np.ndarray):
                assert template.ndim == 2, f'template must be 2D, not {template.ndim}'
            else:
                raise ValueError(f'template must be np.ndarray or int or float between 0.0-1.0, not {type(template)}')
            if template.dtype != np.float32:
                print(f'WARNING: template image is not dtype: np.float32, found {template.dtype}, converting...')
                template = template.astype(np.float32)        

        elif template_method == 'sequential':
            assert isinstance(template, (int, float)), f'template must be int or float between 0.0-1.0, not {type(template)}'
            if isinstance(template, float):
                assert 0.0 <= template <= 1.0, f'template must be between 0.0 and 1.0, not {template}'
                idx = int(len(ims_moving) * template)
                print(f'Converting float fractional index to integer index: {template} -> {idx}')
                template = idx
            assert 0 <= template < len(ims_moving), f'template must be between 0 and {len(ims_moving)-1}, not {template}'

        return ims_moving, template

    def plot_alignment_results_geometric(
        self, 
        plot_direct: bool = True,
    ) -> Tuple[plt.Figure, plt.Figure]:
        """
        Plots the alignment results for geometric registration.

        Args:
            plot_direct (bool): 
                If ``True``, plots the direct alignment results.

        Returns:
            (Tuple[plt.Figure, plt.Figure]): tuple containing:
                fig_final (plt.Figure):
                    Figure showing the final alignment results.
                fig_direct (plt.Figure):
                    Figure showing the direct alignment results.
        """
        ## Make sure that geometric alignment was done
        assert hasattr(self, 'results_geometric'), 'Missing results_geometric attribute. Geometric registration must be performed first.'
        assert hasattr(self, 'ims_registered_geo'), 'Missing ims_registered_geo attribute. Geometric registration must be performed first.'

        results_geometric = self.results_geometric

        ## Plot the results for the geometric registration
        fig_final = self._plot_results(results_geometric['final'], 'final')
        if (results_geometric['direct']['alignment_all_to_all'] is not None) and (results_geometric['direct']['score_all_to_all'] is not None) and plot_direct:
            fig_direct = self._plot_results(results_geometric['direct'], 'direct')
        else:
            fig_direct = None

        return fig_final, fig_direct
    
    def plot_alignment_results_nonrigid(
        self,
    ) -> Tuple[plt.Figure, plt.Figure]:
        """
        Plots the alignment results for non-rigid registration.

        Returns:
            (Tuple[plt.Figure, plt.Figure]): tuple containing:
                fig_final (plt.Figure):
                    Figure showing the final alignment results.
                fig_direct (plt.Figure):
                    Figure showing the direct alignment results.
        """
        ## Make sure that nonrigid alignment was done
        assert hasattr(self, 'results_nonrigid'), 'Missing results_nonrigid attribute. Nonrigid registration must be performed first.'
        assert hasattr(self, 'ims_registered_nonrigid'), 'Missing ims_registered_nonrigid attribute. Nonrigid registration must be performed first.'

        results_nonrigid = self.results_nonrigid

        ## Plot the results for the nonrigid registration
        fig_final = self._plot_results(results_nonrigid['final'], 'final')

        return fig_final, None

    def _plot_results(self, results, name):
        inv_eye = 1 - np.eye(results['alignment_all_to_all'].shape[0])
        cmap = 'viridis'
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs = axs.flatten()
        plt.colorbar(axs[0].imshow(results['score_all_to_all'] * inv_eye, cmap=cmap))
        axs[0].set_title(f'score_all_to_all ({name})')
        axs[1].imshow(results['alignment_all_to_all'] * inv_eye, cmap=cmap)
        axs[1].set_title(f'alignment_all_to_all ({name})')
        
        plt.tight_layout()
        return fig


def clahe(
    im: np.ndarray, 
    grid_size: Union[int, Tuple[int, int]] = 50,
    clipLimit: int = 0, 
    normalize: bool = True,
) -> np.ndarray:
    """
    Perform Contrast Limited Adaptive Histogram Equalization (CLAHE) on an image.

    Args:
        im (np.ndarray):
            Input image.
        grid_size (int):
            Size of the grid. See ``cv2.createCLAHE`` for more info. 
            (Default is *50*)
        clipLimit (int):
            Clip limit. See ``cv2.createCLAHE`` for more info. 
            (Default is *0*)
        normalize (bool):
            Whether to normalize the image to the maximum value (0 - 1) during
            the CLAHE process, then return the image to the original dtype and
            range.
            (Default is ``True``)
        
    Returns:
        (np.ndarray): 
            im_out (np.ndarray): 
                Output image after applying CLAHE.
    """
    assert isinstance(grid_size, (int, tuple)), 'grid_size must be int or tuple'
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)
    elif isinstance(grid_size, tuple):
        assert len(grid_size) == 2, 'grid_size must be a tuple of length 2'
        assert all(isinstance(x, int) for x in grid_size), 'grid_size must be a tuple of integers'

    dtype_in = im.dtype
    if normalize:
        val_max = np.nanmax(im)
        im_tu = (im.astype(np.float32) / val_max)*(2**8 - 1)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=grid_size)
    im_c = clahe.apply(im_tu.astype(np.uint16))
    if normalize:
        im_c = (im_c / (2**8 - 1)) * val_max
    im_c = im_c.astype(dtype_in)
    return im_c


def adaptive_brute_force_matcher(
    features_template: torch.Tensor,
    features_moving: torch.Tensor,
    thresh_prob: float = 0.05,
    metric: str = 'normalized_euclidean',
    moat_prob_ratio: float = 10,
    batch_size: int = 100,
):
    """
    Perform adaptive brute force matching between two sets of features.
    Similar to brute force matching, but converts the distance threshold to a
    probability using statistics of the distances.
    """
    ## Compute distances ('normalized_euclidean' or 'cosine')
    if metric == 'normalized_euclidean':
        precomputed_mat_template = torch.var(features_template, dim=1)
        precomputed_mat_moving   = torch.var(features_moving, dim=1)
        fn_dist = lambda x, y, var_x: (1/2) * (torch.var(x[:, None, :] - y[None, :, :], dim=2) / (var_x[:, None] + torch.var(y, dim=1)[None, :]))
    elif metric == 'cosine':
        precomputed_mat_template = torch.linalg.norm(features_template, dim=1)
        precomputed_mat_moving   = torch.linalg.norm(features_moving, dim=1)
        fn_dist = lambda x, y, norm_x: 1 - (x @ y.T) / (norm_x[:, None] @ torch.linalg.norm(y, dim=1)[None, :])
    else:
        raise ValueError(f"metric must be one of ['normalized_euclidean', 'cosine'], not {metric}")

    def _compute_distance_pvals(mat1, mat2, fn_dist, precomputed_mat):
        ## get stats for mat1 for distance calculation
        idx_top_out = []
        mask_out = []
        for batch2 in helpers.make_batches(mat2, batch_size):
            ## compute distances
            d = fn_dist(mat1, batch2, precomputed_mat)
            ## get stats for z-scoring
            d_mean = d.mean(dim=1)
            d_std  = d.std(dim=1)
            ## get smallest 2 matches using topk
            d_top2 = torch.topk(d, 2, largest=False, dim=1, sorted=True)
            ## convert to z, then p-values
            d_pval = (1 + torch.erf((d_top2.values - d_mean[:, None]) / (d_std[:, None] * np.sqrt(2)))) / 2
            ## get moat probability ratio
            moat_prob = d_pval[:, 1] / d_pval[:, 0]
            ## get matches that pass threshold
            mask_moat = moat_prob > moat_prob_ratio
            ## get indices of matches that pass absolute probability threshold
            mask_thresh = d_pval[:, 0] < thresh_prob
            ## get indices of matches that pass both thresholds
            mask = mask_moat & mask_thresh
            ## append
            idx_top_out.append(d_top2.indices[:, 0])
            mask_out.append(mask)
        idx_top_out = torch.cat(idx_top_out)
        mask_out = torch.cat(mask_out, dim=0)
        return idx_top_out, mask_out

    idx_t2m, mask_t2m = _compute_distance_pvals(features_template, features_moving, fn_dist, precomputed_mat_template)
    idx_m2t, mask_m2t = _compute_distance_pvals(features_moving, features_template, fn_dist, precomputed_mat_moving)
    ## get matches that are mutual
    bool_m2t_mutual = idx_t2m[idx_m2t] == torch.arange(len(idx_m2t))
    mask_mutual = mask_m2t & bool_m2t_mutual & mask_t2m[idx_m2t]
    idx_m2t_mutual = (torch.arange(len(idx_m2t))[mask_mutual], idx_m2t[mask_mutual])
    return idx_m2t_mutual


##########################################################################################################
########################################## REGISTRATION METHODS ##########################################
##########################################################################################################


class ImageRegistrationMethod:
    """
    Base class for image registration methods.
    RH 2024

    This class defines the interface for image registration methods, both rigid
    and non-rigid. Subclasses should implement the methods `_forward_rigid` and
    `_forward_nonrigid`.

    Args:
        device (str):
            Device to use for computations.
        verbose (bool):
            Whether to print progress updates.
    """    
    def __init__(
        self,
        device: str = 'cpu',
        verbose: bool = False,
    ):
        self.device = device
        self.verbose = verbose

    def fit_nonrigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ):
        remappingIdx = self._forward_nonrigid(im_template, im_moving, **kwargs)
        return remappingIdx
    
    @staticmethod
    def _compute_rigid_transform(
        src_pts: np.ndarray,
        dst_pts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes pure-rotation + translation that best aligns src_pts → dst_pts
        via orthogonal Procrustes (SVD) with no scaling.
        """
        # 1) center the points
        src_center = src_pts.mean(axis=0)
        dst_center = dst_pts.mean(axis=0)
        src_shifted = src_pts - src_center
        dst_shifted = dst_pts - dst_center

        # 2) SVD of covariance
        U, _, Vt = np.linalg.svd(src_shifted.T @ dst_shifted)
        R = Vt.T @ U.T
        # Reflection check
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # 3) translation
        t = dst_center - R @ src_center
        return R, t

    def fit_rigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving:  Union[np.ndarray, torch.Tensor],
        inl_thresh: float = 2.0,
        max_iter:  int    = 10,
        confidence: float = 0.99,
        constraint: str   = 'homography',
        **kwargs,
    ) -> np.ndarray:
        """
        Estimate a constrained warp between two images using matched points.
        Returns:
            warp_matrix (np.ndarray[3x3]) - the resulting transformation.
        """
        # 1) detect & match keypoints
        kptsA, kptsB = self._forward_rigid(im_template, im_moving, **kwargs)

        # 2) to numpy
        src_pts = kptsA.cpu().numpy().astype(np.float32)
        dst_pts = kptsB.cpu().numpy().astype(np.float32)

        if len(kptsA) < 3:
            # not enough for homography, fallback to identity
            warnings.warn(f"number of points is less than needed for homography transform. len(kptsA)={len(kptsA)}")
            warp_matrix = np.eye(3)

        # 3) dispatch on constraint
        elif constraint == 'rigid':
            R, t = self._compute_rigid_transform(src_pts, dst_pts)
            warp_matrix = np.eye(3, dtype=np.float32)
            warp_matrix[:2, :2] = R
            warp_matrix[:2, 2   ] = t
            return warp_matrix

        elif constraint == 'euclidean':
            warp_matrix, status = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=inl_thresh,
                maxIters       =max_iter,
                confidence     =confidence,
            )
            if warp_matrix is None:
                raise RuntimeError("Euclidean fit failed")
            # expand 2×3 → 3×3
            warp_matrix = np.vstack([warp_matrix, [0.0, 0.0, 1.0]])

        elif constraint == 'affine':
            M_affine, status = cv2.estimateAffine2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=inl_thresh,
                maxIters       =max_iter,
                confidence     =confidence,
            )
            if M_affine is None:
                raise RuntimeError("Affine fit failed")
            return np.vstack([M_affine, [0.0, 0.0, 1.0]])

        elif constraint == 'homography':
            if len(kptsA) < 4:
                # not enough for homography, fallback to identity
                warnings.warn(f"number of points is less than needed for homography transform. len(kptsA)={len(kptsA)}")
                warp_matrix = np.eye(3)
            else:
                # full projective homography
                warp_matrix, inliers = cv2.findHomography(
                    src_pts, dst_pts,
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=inl_thresh,
                    maxIters       =max_iter,
                    confidence     =confidence,
                )
            
        else:
            raise ValueError(f"Unknown constraint: {constraint}")
        
        # 4) check for failure
        if warp_matrix is None:
            raise RuntimeError("Homography fit failed")
            
        ## If verbose > 2, make a plot of corresponding points
        if self.verbose > 2:
            alpha = float(np.clip(1 - (len(src_pts) / 2000), 0.2, 1))
            outliers = np.logical_not(inliers)
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(np.hstack([im_template, im_moving]), cmap='gray')
            ## plot inliers as green dots
            ax.scatter(src_pts[inliers, 0], src_pts[inliers, 1], c='g', s=10, label='inliers', alpha=alpha)
            ax.scatter(dst_pts[inliers, 0] + im_template.shape[1], dst_pts[inliers, 1], c='g', s=10, alpha=alpha)
            ## plot outliers as red dots
            ax.scatter(src_pts[outliers, 0], src_pts[outliers, 1], c='r', s=10, label='outliers', alpha=alpha)
            ax.scatter(dst_pts[outliers, 0] + im_template.shape[1], dst_pts[outliers, 1], c='r', s=10, alpha=alpha)
            ## plot lines connecting points
            for pt in np.where(inliers)[0]:
                ## plot lines in cyan
                ax.plot([src_pts[pt, 0], dst_pts[pt, 0] + im_template.shape[1]], [src_pts[pt, 1], dst_pts[pt, 1]], 'c-', lw=1, alpha=alpha)
            for pt in np.where(outliers)[0]:
                ## plot lines in magenta
                ax.plot([src_pts[pt, 0], dst_pts[pt, 0] + im_template.shape[1]], [src_pts[pt, 1], dst_pts[pt, 1]], 'm-', lw=1, alpha=alpha)
            plt.show()

        return warp_matrix
        
    def _forward_nonrigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ):
        raise NotImplementedError(f"Method _forward_nonrigid not implemented for {self.__class__.__name__}.")

    def _forward_rigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ):
        raise NotImplementedError(f"Method _forward_rigid not implemented for {self.__class__.__name__}.")


class RoMa(ImageRegistrationMethod):
    """
    RoMa-based image registration method.
    RH 2024

    Args:
        model_type (str):
            Type of RoMa model to use. Either 'outdoor' or 'indoor'.
        n_points (int):
            Number of points to sample for matching. (Default is *10000*)
        batch_size (int):
            Batch size for processing matches. (Default is *1000*)
        device (str):
            Device to use for computations.
        verbose (bool):
            Whether to print progress updates.
    """
    def __init__(
        self,
        model_type: str = 'outdoor',
        n_points: int = 10000,
        batch_size: int = 1000,
        device: str = 'cpu',
        weight_urls: Dict[str, Dict[str, Dict[str, str]]] = {
            "romatch": {
                "outdoor": {
                    "url": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
                    "hash": "9a451dfb65745e777bf916db6ea84933",
                    "filename": "roma_outdoor.pth",
                },
                "indoor": {
                    "url": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
                    "hash": "349a17aaa21883bb164b1a5884febb21",
                    "filename": "roma_indoor.pth",
                },
            },
            "tiny_roma_v1": {
                "outdoor": {
                    "url": "https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth",
                    "hash": "b8120606c6b027a07b856d64f20bd13e",
                    "filename": "tiny_roma_v1_outdoor.pth",
                },
            },
            "dinov2": {
                "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", #hopefully this doesnt change :D
                "hash": "19a02c10947ed50096ce382b46b15662",
                "filename": "dinov2_vitl14_pretrain.pth",
            },
        },
        fallback_weight_urls = {
            "romatch": {
                "outdoor": {
                    "url": "https://osf.io/cmzpa/download",
                    "hash": "9a451dfb65745e777bf916db6ea84933",
                    "filename": "roma_outdoor.pth",
                },
                "indoor": {
                    "url": "https://osf.io/uzx64/download",
                    "hash": "349a17aaa21883bb164b1a5884febb21",
                    "filename": "roma_indoor.pth",
                },
            },
            "tiny_roma_v1": {
                "outdoor": {
                    "url": "https://osf.io/6anre/download",
                    "hash": "b8120606c6b027a07b856d64f20bd13e",
                    "filename": "tiny_roma_v1_outdoor.pth",
                },
            },
            "dinov2": {
                "url": "https://osf.io/tmj5c/download", #hopefully this doesnt change :D
                "hash": "19a02c10947ed50096ce382b46b15662",
                "filename": "dinov2_vitl14_pretrain.pth",
            },
        },
        verbose=False,
    ):
        try:
            from romatch import roma_outdoor, roma_indoor, tiny_roma_v1_outdoor
        except ImportError:
            raise ImportError("RoMa not installed. Please install romatch. See ROICaT's installation instructions for more details.")

        super().__init__(device=device, verbose=verbose)

        self.roma_model_type = model_type
        self.n_points = n_points
        self.batch_size = batch_size
        self.weight_urls = weight_urls
        self.fallback_weight_urls = fallback_weight_urls
        self.verbose = verbose

        ## Download weights and dinov2
        def download_and_check_weights(url, filename, hash):
            dir_save = str(Path(torch.hub.get_dir()) / "checkpoints")

            weights = torch.hub.load_state_dict_from_url(
                url,
                map_location=device,
                file_name=filename,
            )
            ## Check if weights are correct
            path_weights = str(Path(dir_save) / filename)

            if not hash == helpers.hash_file(path=path_weights, type_hash='MD5'):
                raise ValueError(f"RoMa weights hash mismatch. Expected: {hash}. Found: {helpers.hash_file(path=path_weights, type_hash='MD5')}. Path: {path_weights}, URL: {url}")
            else:
                return weights

        def safe_download_and_check_weights(weight_urls, fallback_weight_urls):
            try:
                weights = download_and_check_weights(
                    url=     weight_urls["url"],
                    filename=weight_urls["filename"],
                    hash=    weight_urls["hash"],
                )
            except Exception as e:
                warnings.warn(f"Download or hash check failed. Error: {e}")
                print("Using fallback weights for file: ", weight_urls["filename"])
                weights = download_and_check_weights(
                    url=     fallback_weight_urls["url"],
                    filename=fallback_weight_urls["filename"],
                    hash=    fallback_weight_urls["hash"],
                )
            return weights

        weights = safe_download_and_check_weights(
            weight_urls=weight_urls["romatch"][model_type],
            fallback_weight_urls=fallback_weight_urls["romatch"][model_type],
        )
        weights_dinov2 = safe_download_and_check_weights(
            weight_urls=weight_urls["dinov2"],
            fallback_weight_urls=fallback_weight_urls["dinov2"],
        )

        if model_type == 'outdoor':
            self.model = roma_outdoor(device=device, weights=weights, dinov2_weights=weights_dinov2)
        elif model_type == 'indoor':
            self.model = roma_indoor(device=device, weights=weights, dinov2_weights=weights_dinov2)
    
    def _match(
        self,
        im1: Union[np.ndarray, torch.Tensor],
        im2: Union[np.ndarray, torch.Tensor],
        device: Optional[str] = None,
        **kwargs,
    ):
        ff, certainty = self.model.match(
            self._prepare_image(im1),
            self._prepare_image(im2),
            device=self.device if device is None else device,
        )
        return ff, certainty
    
    def _forward_nonrigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ):
        h, w = im_moving.shape[0], im_moving.shape[1]
        
        ## Pass images through RoMa model to get flow field
        ff, certainty = self._match(im_template, im_moving, device=self.device)

        ## Convert flow field to remappingIdx
        remappingIdx = helpers.resize_remappingIdx(
            ri=helpers.pytorchFlowField_to_cv2RemappingIdx(ff.cpu()[:, :ff.shape[1]//2, 2:]),
            new_shape=(h, w),
            interpolation='BILINEAR',
        )
        return remappingIdx.cpu().numpy()
    
    def _forward_rigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ):
        h, w = im_moving.shape[0], im_moving.shape[1]
        
        ## Pass images through RoMa model to get flow field
        ff, certainty = self._match(im_template, im_moving, device=self.device)

        ## Sample matches for estimation
        def get_points(ff, certainty, num):
            matches, certainty = self.model.sample(ff, certainty, num=num)
            kptsA, kptsB = self.model.to_pixel_coordinates(matches, h, w, h, w)
            return kptsA, kptsB, certainty

        ## Batch the points
        batch_ns = [int(batch.sum()) for batch in helpers.make_batches(np.ones(self.n_points), batch_size=self.batch_size, min_batch_size=10)]
        outs = [get_points(ff, certainty, num=n) for n in batch_ns]

        kptsA, kptsB, certainty = [torch.cat([out[ii] for out in outs], dim=0) for ii in range(3)]

        return kptsA, kptsB


    def _prepare_image(
        self,
        image: np.ndarray,
    ):
        """
        Prepare FOV image for RoMa model.

        Args:

                Image to be prepared. dtype should be floating and data must be
                in range [0, 1].

        Returns:
            PIL.Image: 
                Image object.
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        return PIL.Image.fromarray(image * 255).convert("RGB")


class LoFTR(ImageRegistrationMethod):
    """
    LoFTR-based image registration method.
    RH 2024

    Args:
        model_type (str):
            Type of LoFTR model to use. Default is 'indoor_new'.
        threshold_confidence (float):
            Confidence threshold for filtering matches. (Default is *0.2*)
        device (str):
            Device to use for computations.
        verbose (bool):
            Whether to print progress updates.
    """    
    def __init__(
        self,
        model_type: str = 'indoor_new',
        threshold_confidence: float = 0.2,
        device: str = 'cpu',
        verbose: bool = False,
    ):
        import kornia

        super().__init__(device=device, verbose=verbose)
        self.verbose = verbose

        self.model_type = model_type
        self.threshold_confidence = threshold_confidence

        self.model = kornia.feature.LoFTR(pretrained=model_type)
        self.model.to(device)
    
    def _forward_rigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        threshold_confidence: Optional[float] = None,
        **kwargs,
    ):
        input_dict = {
            'image0': torch.as_tensor(im_template, dtype=torch.float32, device=self.device)[None, None, :, :],
            'image1': torch.as_tensor(im_moving, dtype=torch.float32, device=self.device)[None, None, :, :],
        }
        ## Get the keypoints and descriptors
        with torch.inference_mode():
            matches = self.model(input_dict)
        ## Clean up matches with low confidence
        bool_keep = matches["confidence"] > (self.threshold_confidence if threshold_confidence is None else threshold_confidence)
        kptsA = matches["keypoints0"][bool_keep]
        kptsB = matches["keypoints1"][bool_keep]
        return kptsA, kptsB
    

class ECC_cv2(ImageRegistrationMethod):
    """
    Image registration method using OpenCV's ECC algorithm.
    RH 2024

    Args:
        mode_transform (str):
            Type of geometric transformation. One of {'translation',
            'euclidean', 'affine', 'homography'}.
            (Default is 'euclidean')
        n_iter (int):
            Number of iterations for optimization. (Default is *200*)
        termination_eps (float):
            Convergence tolerance. (Default is *1e-09*)
        gaussFiltSize (Union[float, int]):
            Size of Gaussian blurring filter applied to images. (Default is *1*)
        auto_fix_gaussFilt_step (Optional[int]):
            Increment in gaussFiltSize after a failed optimization. If ``None``,
            no automatic fixing is performed.
            (Default is *10*)
        device (str):
            Device to use for computations.
        verbose (bool):
            Whether to print progress updates.
    """    
    def __init__(
        self,
        mode_transform='euclidean',  ## type of geometric transformation. See openCV's cv2.findTransformECC for details
        n_iter: int = 200,  ## number of iterations for optimization
        termination_eps: float = 1e-09,  ## convergence tolerance
        gaussFiltSize: Union[float, int] = 1,  ## size of gaussian blurring filter applied to all images
        auto_fix_gaussFilt_step: Optional[int] = 10,  ## increment in gaussFiltSize after a failed optimization
        device: str = 'cpu',
        verbose: bool = False,
    ):

        super().__init__(device=device, verbose=verbose)

        ## Check inputs
        ### Check if mode_transform is valid
        valid_mode_transforms = {'translation', 'euclidean', 'affine', 'homography'}
        assert mode_transform in valid_mode_transforms, f"mode_transform must be one of {valid_mode_transforms}"
        ### Check if gaussFiltSize is a number (float or int)
        assert isinstance(gaussFiltSize, (float, int)), "gaussFiltSize must be a number."
        ### Convert gaussFiltSize to an odd integer
        gaussFiltSize = int(np.round(gaussFiltSize))

        self.mode_transform, self.n_iter, self.termination_eps, self.gaussFiltSize, self.auto_fix_gaussFilt_step = \
            mode_transform, n_iter, termination_eps, gaussFiltSize, auto_fix_gaussFilt_step
    
    def fit_rigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ):
        def _recursive_closure(
            im_template: Union[np.ndarray, torch.Tensor],
            im_moving: Union[np.ndarray, torch.Tensor],
            gaussFiltSize: Union[float, int],
            depth: int = 0,
            max_depth: int = 100,
        ):
            depth += 1
            try:
                warp_matrix = helpers.find_geometric_transformation(
                    im_template=im_template,
                    im_moving=im_moving,
                    warp_mode=self.mode_transform,
                    n_iter=self.n_iter,
                    termination_eps=self.termination_eps,
                    gaussFiltSize=gaussFiltSize,
                )
            except Exception as e:
                if self.auto_fix_gaussFilt_step is not None:
                    print(f'Error finding geometric registration warp for image. Error: {e}') if self.verbose else None

                    if depth > max_depth:
                        print(f"Reached maximum depth of {max_depth}. Returning identity matrix warp.")
                        return np.eye(3)[:2,:] if self.mode_transform != 'homography' else np.eye(3)
                    return _recursive_closure(
                        im_template=im_template,
                        im_moving=im_moving,
                        gaussFiltSize=gaussFiltSize + self.auto_fix_gaussFilt_step,
                        depth=depth,
                        max_depth=max_depth,
                    )

                print(f'Failed finding geometric registration warp for image Error: {e}')
                print(f'Defaulting to identity matrix warp.')
                print(f'Consider doing one of the following:')
                print(f'  - Make better images to input. You can add the spatialFootprints images to the FOV images to make them better.')
                print(f'  - Increase the gaussFiltSize parameter. This will make the images blurrier, but may help with registration.')
                print(f'  - Decrease the termination_eps parameter. This will make the registration less accurate, but may help with registration.')
                print(f'  - Increase the mask_borders parameter. This will make the images smaller, but may help with registration.')
                warp_matrix = np.eye(3)[:2,:] if self.mode_transform != 'homography' else np.eye(3)
            return warp_matrix
        
        warp_matrix = _recursive_closure(
            im_template=im_template,
            im_moving=im_moving,
            gaussFiltSize=self.gaussFiltSize,
        )

        warp_matrix = np.concatenate([warp_matrix, np.array([[0, 0, 1]])], axis=0) if warp_matrix.shape[0] == 2 else warp_matrix
        return warp_matrix


class DeepFlow(ImageRegistrationMethod):
    """
    Image registration method using OpenCV's DeepFlow algorithm.
    RH 2024

    Args:
        device (str):
            Device to use for computations.
        verbose (bool):
            Whether to print progress updates.
    """
    def __init__(
        self,
        device: str = 'cpu',
        verbose=False,
    ):
        super().__init__(device=device, verbose=verbose)

        self.model = cv2.optflow.createOptFlow_DeepFlow()
    
    def _forward_nonrigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ):
        h, w = im_moving.shape
        x_grid, y_grid = np.meshgrid(np.arange(0., w).astype(np.float32), np.arange(0., h).astype(np.float32), indexing='xy')

        im_template, im_moving = (self._prepare_image(im) for im in (im_template, im_moving))

        remappingIdx = self.model.calc(
            im_template,
            im_moving,
            None
        ) + np.stack([x_grid, y_grid], axis=-1)

        return remappingIdx
    
    def _prepare_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Prepare FOV image for RoMa model.

        Args:
            image (np.ndarray): 
                Image to be prepared. dtype should be floating and data must be
                in range [0, 1].

        Returns:
            np.ndarray:
                image array
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        return (image * 255).astype(np.uint8)


class OpticalFlowFarneback(ImageRegistrationMethod):
    """
    Image registration method using OpenCV's calcOpticalFlowFarneback.
    RH 2024

    Args:
        pyr_scale (float):
            Parameter specifying the image scale (<1) to build pyramids for each
            image. (Default is *0.3*)
        levels (int):
            Number of pyramid layers including the initial image. (Default is
            *3*)
        winsize (int):
            Averaging window size. Larger values increase the algorithm
            robustness to noise and provide smoother motion field. (Default is
            *128*)
        iterations (int):
            Number of iterations the algorithm does at each pyramid level.
            (Default is *7*)
        poly_n (int):
            Size of the pixel neighborhood used to find polynomial expansion in
            each pixel. (Default is *5*)
        poly_sigma (float):
            Standard deviation of the Gaussian used to smooth derivatives used
            as a basis for the polynomial expansion. (Default is *1.5*)
        flags (int):
            Operation flags. (Default is cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        device (str):
            Device to use for computations.
        verbose (bool):
            Whether to print progress updates.
    """    
    def __init__(
        self,
        pyr_scale: float = 0.3,
        levels: int = 3,
        winsize: int = 128,
        iterations: int = 7,
        poly_n: int = 5,
        poly_sigma: float = 1.5,
        flags: int = cv2.OPTFLOW_FARNEBACK_GAUSSIAN,  ## == 256
        device: str = 'cpu',
        verbose=False,
    ):
        super().__init__(device=device, verbose=verbose)

        self.pyr_scale, self.levels, self.winsize, self.iterations, self.poly_n, self.poly_sigma, self.flags = \
            pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
    
    def _forward_nonrigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ):
        h, w = im_moving.shape
        x_grid, y_grid = np.meshgrid(np.arange(0., w).astype(np.float32), np.arange(0., h).astype(np.float32), indexing='xy')

        im_template, im_moving = (self._prepare_image(im) for im in (im_template, im_moving))

        remappingIdx = cv2.calcOpticalFlowFarneback(
            prev=im_template,
            next=im_moving,
            flow=None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=self.flags,
        ) + np.stack([x_grid, y_grid], axis=-1)

        return remappingIdx
    
    def _prepare_image(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Prepare FOV image for RoMa model.

        Args:
            image (np.ndarray): 
                Image to be prepared. dtype should be floating and data must be
                in range [0, 1].

        Returns:
            np.ndarray:
                image array
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        return (image * 255).astype(np.uint8)
    

class SIFT(ImageRegistrationMethod):
    """
    Image registration method using SIFT keypoints.
    RH 2024

    Args:
        nfeatures (int):
            Number of best features to retain. (Default is *500*)
        contrastThreshold (float):
            Contrast threshold used to filter out weak features. (Default is
            *0.04*)
        edgeThreshold (float):
            Threshold used to filter out edge-like features. (Default is *10*)
        sigma (float):
            Sigma of the Gaussian applied to the input image at the octave #0.
            (Default is *1.6*)
        device (str):
            Device to use for computations.
        verbose (bool):
            Whether to print progress updates.
    """    
    def __init__(
        self,
        nfeatures: int = 500,
        contrastThreshold: float = 0.04,
        edgeThreshold: float = 10,
        sigma: float = 1.6,
        device: str = 'cpu',
        verbose: bool = False,
    ):
        super().__init__(device=device, verbose=verbose)

        self.sift = cv2.SIFT_create(
            nfeatures=nfeatures,
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            sigma=sigma,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    def _forward_rigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ):
        # Prepare images
        img1 = self._prepare_image(im_template)
        img2 = self._prepare_image(im_moving)
        # Detect and compute features
        keypoints1, descriptors1 = self.sift.detectAndCompute(img1, None)
        keypoints2, descriptors2 = self.sift.detectAndCompute(img2, None)
        # Match descriptors
        matches = self.matcher.match(descriptors1, descriptors2)
        # Sort matches by distance (quality)
        matches = sorted(matches, key=lambda x: x.distance)
        # Extract matched keypoints
        kptsA = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        kptsB = np.float32([keypoints2[m.trainIdx].pt for m in matches])
        # Convert to torch tensors
        kptsA = torch.from_numpy(kptsA).to(self.device)
        kptsB = torch.from_numpy(kptsB).to(self.device)
        return kptsA, kptsB

    def _prepare_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
    ):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = (image * 255).astype(np.uint8)
        return image


class ORB(ImageRegistrationMethod):
    """
    Image registration method using ORB keypoints.
    RH 2024

    Args:
        nfeatures (int):
            Maximum number of features to retain. (Default is *500*)
        scaleFactor (float):
            Pyramid decimation ratio. (Default is *1.2*)
        nlevels (int):
            Number of pyramid levels. (Default is *8*)
        edgeThreshold (int):
            Size of the border where the features are not detected. (Default is
            *31*)
        firstLevel (int):
            The level of pyramid to put source image to. (Default is *0*)
        WTA_K (int):
            Number of points that produce each element of the oriented BRIEF
            descriptor. (Default is *2*)
        scoreType (int):
            Type of score to rank features. (Default is cv2.ORB_HARRIS_SCORE)
        patchSize (int):
            Size of the patch used by the oriented BRIEF descriptor. (Default is
            *31*)
        fastThreshold (int):
            FAST threshold. (Default is *20*)
        device (str):
            Device to use for computations.
        verbose (bool):
            Whether to print progress updates.
    """
    def __init__(
        self,
        nfeatures: int = 500,
        scaleFactor: float = 1.2,
        nlevels: int = 8,
        edgeThreshold: int = 31,
        firstLevel: int = 0,
        WTA_K: int = 2,
        scoreType: int = cv2.ORB_HARRIS_SCORE,
        patchSize: int = 31,
        fastThreshold: int = 20,
        device: str = 'cpu',
        verbose: bool = False,
    ):
        super().__init__(device=device, verbose=verbose)
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=edgeThreshold,
            firstLevel=firstLevel,
            WTA_K=WTA_K,
            scoreType=scoreType,
            patchSize=patchSize,
            fastThreshold=fastThreshold,
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def _forward_rigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ):
        # Prepare images
        img1 = self._prepare_image(im_template)
        img2 = self._prepare_image(im_moving)
        # Detect and compute features
        keypoints1, descriptors1 = self.orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(img2, None)
        # Match descriptors
        matches = self.matcher.match(descriptors1, descriptors2)
        # Sort matches by distance (quality)
        matches = sorted(matches, key=lambda x: x.distance)
        # Extract matched keypoints
        kptsA = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        kptsB = np.float32([keypoints2[m.trainIdx].pt for m in matches])
        # Convert to torch tensors
        kptsA = torch.from_numpy(kptsA).to(self.device)
        kptsB = torch.from_numpy(kptsB).to(self.device)
        return kptsA, kptsB

    def _prepare_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
    ):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = (image * 255).astype(np.uint8)
        return image


class DISK_LightGlue(ImageRegistrationMethod):
    """
    Image registration method using DISK features and LightGlue matcher.
    RH 2024

    Args:
        num_features (int):
            Number of features to extract. (Default is *2048*)
        threshold_confidence (float):
            Confidence threshold for filtering matches. (Default is *0.2*)
        window_nms (int):
            Window size for non-maximum suppression. Must be odd integer. Larger
            values will result in fewer keypoints. (Default is *5*)
        device (str):
            Device to use for computations.
        verbose (bool):
            Whether to print progress updates.
    """    
    def __init__(
        self,
        num_features: int = 2048,
        threshold_confidence: float = 0.2,
        window_nms: int = 5,
        device: str = 'cpu',
        verbose: bool = False,
    ):
        import kornia

        super().__init__(device=device, verbose=verbose)
        self.verbose = verbose
        self.threshold_confidence = threshold_confidence
        self.num_features = num_features
        self.window_nms = window_nms

        # Initialize feature extractor
        self.feature_extractor = kornia.feature.DISK.from_pretrained(checkpoint="epipolar", device=device).eval()

        # Initialize LightGlue matcher
        self.matcher = kornia.feature.LightGlue(features='disk').eval().to(device)

    def _forward_rigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ):
        # Prepare images
        img1 = self._prepare_image(im_template)
        img2 = self._prepare_image(im_moving)
        ## Warn if image shapes are not divisible by 16
        shapes_ims = (img1.shape[-2:], img2.shape[-2:])
        if any([dim % 16 != 0 for dim in np.concatenate(shapes_ims)]) and (self.verbose > 1):
            print(f"Image shapes are not divisible by 16. Will be padded using kornia.feature.DISK padding. Shapes: {shapes_ims}")


        # Extract features
        with torch.inference_mode():
            inp = torch.cat([img1, img2], dim=0)
            features = self.feature_extractor(
                images=inp, 
                n=self.num_features,
                window_size=self.window_nms,
                # score_threshold=self.threshold_confidence,
                pad_if_not_divisible=True,
            )
            features1 = features[0]
            features2 = features[1]
            kps1, descs1 = features1.keypoints, features1.descriptors
            kps2, descs2 = features2.keypoints, features2.descriptors

            # Prepare data for LightGlue
            image0 = {
                "keypoints": kps1[None],
                "descriptors": descs1[None],
                "image_size": torch.tensor(img1.shape[-2:][::-1]).view(1, 2).to(self.device),
            }
            image1 = {
                "keypoints": kps2[None],
                "descriptors": descs2[None],
                "image_size": torch.tensor(img2.shape[-2:][::-1]).view(1, 2).to(self.device),
            }

            # Match with LightGlue
            out = self.matcher({"image0": image0, "image1": image1})
            idxs = out["matches0"][0].cpu()  # matches0 from LightGlue output
            valid = (idxs > -1)
            confidences = out["matching_scores0"][0].cpu()
            valid = (valid * (confidences > self.threshold_confidence)).cpu()
            idxs = torch.stack([torch.arange(len(kps1))[valid], idxs[valid]], dim=-1)

        # Get matching keypoints
        kptsA = kps1[idxs[:, 0]]
        kptsB = kps2[idxs[:, 1]]

        return kptsA, kptsB

    def _prepare_image(self, image: Union[np.ndarray, torch.Tensor]):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if image.dim() == 2:
            image = image.unsqueeze(0)  # Add channel dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension
        if image.max() > 1.0:
            image = image / 255.0
        image = image.to(self.device).tile(1, 3, 1, 1)  # Add color channels

        return image
    

class PhaseCorrelationRegistration(ImageRegistrationMethod):
    """
    Image registration method using helpers.phase_correlation.
    RH 2024

    Args:
        device (str):
            Device to use for computations.
        verbose (bool):
            Whether to print progress updates.
    """    
    def __init__(
        self,
        device: str = 'cpu',
        bandpass_freqs: Optional[List[float]] = None,
        order: int = 5,
        verbose: bool = False,
    ):
        super().__init__(device=device, verbose=verbose)

        self.bandpass_freqs = bandpass_freqs
        self.order = order

    def fit_rigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ):
        
        filt = helpers.make_2D_frequency_filter(
            hw=im_template.shape[-2:],
            low=self.bandpass_freqs[0],
            high=self.bandpass_freqs[1],
            order=self.order,
        ) if self.bandpass_freqs is not None else None

        cc = helpers.phase_correlation(
            im_template=im_template,
            im_moving=im_moving,
            mask_fft=filt,
            return_filtered_images = False,
            eps = 1e-8,
        )

        ## get shifts
        shift_y_raw, shift_x_raw = np.unravel_index(cc.argmax(), cc.shape)
        # shifts = (float(np.floor(cc.shape[-2]/2) - shift_y_raw) , float(np.ceil(cc.shape[-1]/2) - shift_x_raw))
        shifts = (float(np.ceil(cc.shape[-1]/2) - shift_x_raw), float(np.floor(cc.shape[-2]/2) - shift_y_raw))

        ## convert shifts to warp matrix
        warp_matrix = np.eye(3, dtype=np.float32)
        warp_matrix[:2, 2] = shifts

        return warp_matrix

