from typing import List, Tuple, Union, Optional, Dict, Any, Sequence, Callable

import warnings
import functools
import copy

import numpy as np
import scipy.optimize
import scipy.sparse
import torch
from tqdm.auto import tqdm
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
            template upon failure.
        radius_in (float):
            Value in micrometers used to define the maximum shift/offset between
            two images that are considered to be aligned. Use larger values for
            more lenient alignment requirements. (Default is *4*)
        radius_out (float):
            Value in micrometers used to define the minimum shift/offset between
            two images that are considered to be misaligned. Use smaller values
            for more stringent alignment requirements. (Default is *10*)
        order (int):
            The order of the butterworth filter used to define the 'in' and 'out'
            regions of the ImageAlignmentChecker class. (Default is *5*)
        probability_threshold (float):
            Probability required to define two images as aligned. Smaller values
            results in more stringent alignment requirements and possibly slower
            registration. Value is the probability threshold used on the 'z_in'
            output of the ImageAlignmentChecker class to determine if two images
            are properly aligned. (Default is *0.01*)
        um_per_pixel (float):
            The number of micrometers per pixel in the FOV_images.
        device (str):
            The torch device used for various steps in the alignment process.
        verbose (bool):
            Whether to print progress updates. (Default is ``True``)
    """
    def __init__(
        self,
        use_match_search: bool = True,
        radius_in: float = 4,
        radius_out: float = 20,
        order: int = 5,
        probability_threshold: float = 0.01,
        um_per_pixel: float = 1.0,
        device: str = 'cpu',
        verbose: bool = True,
    ):
        super().__init__()

        ## Store parameter (but not data) args as attributes
        self.params['__init__'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'use_match_search',
                'radius_in',
                'radius_out',
                'order',
                'probability_threshold',
                'um_per_pixel',
                'device',
                'verbose',
            ],
        )

        self.use_match_search = use_match_search
        self.radius_in = radius_in
        self.radius_out = radius_out
        self.order = order
        self.probability_threshold = probability_threshold
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
        CLAHE_grid_block_size: int = 50,
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
            spatialFootprints (List[scipy.sparse.csr_matrix], optional):
                A list of spatial footprints for each ROI. If ``None``, then no
                mixing will be performed. (Default is ``None``)
            normalize_FOV_intensities (bool):
                Whether to normalize the FOV images. Setting this to ``True``
                will divide each FOV image by the norm of all the FOV images.
                (Default is ``True``)
            roi_FOV_mixing_factor (float):
                The factor by which to mix the ROI images into the FOV images.
                If 0, then no mixing will be performed. (Default is *0.5*)
            use_CLAHE (bool):
                Whether to apply CLAHE to the images. (Default is ``True``)
            CLAHE_grid_block_size (int):
                The size of the blocks in the gride for CLAHE. Used to divide
                the image into small blocks and create the grid_size parameter
                for the cv2.createCLAHE function. Smaller block sizes will
                result in more local CLAHE. (Default is *50*)
            CLAHE_clipLimit (int):
                The clip limit for CLAHE. See alignment.clahe for more details.
                (Default is *1*)
            CLAHE_normalize (bool):
                Whether to normalize the CLAHE output. See alignment.clahe for
                more details. (Default is ``True``)

        Returns:
            (List[np.ndarray]):
                FOV_images_augmented (List[np.ndarray]):
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
        algorithm: str = 'LoFTR',
        kwargs_algo_findTransformECC: Optional[Dict[str, Any]] = {
            'mode_transform': 'affine',
            'gaussFiltSize': 11,
            'n_iter': 1000,
            'termination_eps': 1e-9,
            'auto_fix_gaussFilt_step': 10,
        },
        kwargs_algo_LoFTR: Optional[Dict[str, Any]] = {
            'mode_transform': 'affine',
            'gaussFiltSize': 11,
            'confidence_LoFTR': 0.2,
            'confidence_RANSAC': 0.99,
            'ransacReprojThreshold': 3.0,
            'maxIters': 2000,
        },
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Performs geometric registration of ``ims_moving`` to a template, using 
        ``cv2.findTransformECC``. 
        RH 2023

        Args:
            template (Union[int, np.ndarray]): 
                Depends on the value of 'template_method'. 
                If 'template_method' == 'image', this should be a 2D np.ndarray image, an integer index 
                of the image to use as the template, or a float between 0 and 1 representing the fractional 
                index of the image to use as the template. 
                If 'template_method' == 'sequential', then template is the integer index or fractional index 
                of the image to use as the template.
            ims_moving (List[np.ndarray]): 
                List of images to be aligned.
            template_method (str): 
                Method to use for template selection. \n
                * 'image': use the image specified by 'template'. 
                * 'sequential': register each image to the previous or next image \n
                (Default is 'sequential')
            mask_borders (Tuple[int, int, int, int]): 
                Border mask for the image. Format is (top, bottom, left, right). 
                (Default is (0, 0, 0, 0))
            algorithm (str):
                The algorithm to use for registration. Either 'LoFTR' or 'findTransformECC'. \n
                    * 'LoFTR': Feature-based registration. Uses LoFTR for
                      feature detection and matching. Use kornia's pretrained
                      LoFTR model with pretrained='indoor_new'. \n
                    * 'findTransformECC': Direct / intensity-based registration.
                      Use cv2's findTransformECC method. \n
            kwargs_algo_findTransformECC (Optional[Dict[str, Any]]):
                Keyword arguments for the findTransformECC method. \n
                * mode_transform (str): 
                    * Mode of geometric transformation. Can be 'translation',
                      'euclidean', 'affine', or 'homography'. See
                      ``cv2.findTransformECC`` for more details. (Default is
                      'affine')
                * gaussFiltSize (int): 
                    * Size of the Gaussian filter. (Default is *11*)
                * n_iter (int): 
                    * Number of iterations for ``cv2.findTransformECC``.
                * termination_eps (float): 
                    * Termination criteria for ``cv2.findTransformECC``.
                * auto_fix_gaussFilt_step (Union[bool, int]): 
                    * Automatically fixes convergence issues by increasing the
                      gaussFiltSize. If ``False``, no automatic fixing is
                      performed. If ``True``, the gaussFiltSize is increased by
                      2 until convergence. If int, the gaussFiltSize is
                      increased by this amount until convergence.
            kwargs_algo_LoFTR (Optional[Dict[str, Any]]):
                Keyword arguments for the LoFTR method. \n
                * mode_transform (str): 
                    * Mode of geometric transformation. Can be 'euclidean',
                      'similarity', 'affine', or 'projective',
                * gaussFiltSize (int): 
                    * Size of the Gaussian filter. (Default is *11*)
                * confidence_LoFTR (float):
                    * Confidence threshold for LoFTR matches. Matches with
                      confidence values below this threshold will be discarded.
                * confidence_RANSAC (float):
                    * Confidence threshold for RANSAC inliers.
                * ransacReprojThreshold (float):
                    * Reprojection threshold for RANSAC.
                * maxIters (int):
                    * Maximum number of iterations for RANSAC.

        Returns:
            (np.ndarray): 
                remapIdx_geo (np.ndarray): 
                    An array of shape *(N, H, W, 2)* representing the remap field for N images.
        """
        ## Store parameter (but not data) args as attributes
        self.params['fit_geometric'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'template',
                'template_method',
                'mask_borders',
                'algorithm',
                'kwargs_algo_findTransformECC',
                'kwargs_algo_LoFTR',
                'device',
            ],
        )

        device = self.device if device is None else device

        ## Fill missing kwargs with defaults
        default_kwargs_algo_findTransformECC = {
            'mode_transform': 'affine',
            'gaussFiltSize': 11,
            'n_iter': 1000,
            'termination_eps': 1e-9,
            'auto_fix_gaussFilt_step': 10,
        }
        default_kwargs_algo_LoFTR = {
            'mode_transform': 'affine',
            'gaussFiltSize': 11,
            'confidence_LoFTR': 0.2,
            'confidence_RANSAC': 0.99,
            'ransacReprojThreshold': 3.0,
            'maxIters': 2000,
            'device': 'cpu',
        }
        kwargs_algo_findTransformECC = default_kwargs_algo_findTransformECC if kwargs_algo_findTransformECC is None else {**default_kwargs_algo_findTransformECC, **kwargs_algo_findTransformECC}
        kwargs_algo_LoFTR = default_kwargs_algo_LoFTR if kwargs_algo_LoFTR is None else {**default_kwargs_algo_LoFTR, **kwargs_algo_LoFTR}
        
        # Check if ims_moving is a non-empty list
        assert len(ims_moving) > 0, "ims_moving must be a non-empty list of images."
        # Check if all images in ims_moving have the same shape
        shape = ims_moving[0].shape
        for im in ims_moving:
            assert im.shape == shape, "All images in ims_moving must have the same shape."
        # Check if template_method is valid
        valid_template_methods = {'sequential', 'image'}
        assert template_method in valid_template_methods, f"template_method must be one of {valid_template_methods}"

        H, W = ims_moving[0].shape
        self._HW = (H,W) if self._HW is None else self._HW

        ims_moving, template = self._fix_input_images(ims_moving=ims_moving, template=template, template_method=template_method)

        self.mask_geo = helpers.mask_image_border(
            im=np.ones((H, W), dtype=np.uint8),
            border_outer=mask_borders,
            mask_value=0,
        )

        def _register(
            template: np.ndarray,
            template_method: str,
        ):
            print(f'Finding geometric registration warps with template_method: {template_method}, mask_borders: {mask_borders is not None}, algorithm: {algorithm}') if self._verbose else None
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
                                                
                if algorithm == 'LoFTR':
                    warp_matrix = _LoFTR_fit(
                        im_template=im_template,
                        im_moving=im_moving,
                        mask_borders=mask_borders,
                        image_id=ii,
                        verbose=self._verbose,
                        **kwargs_algo_LoFTR,
                    )
                elif algorithm == 'findTransformECC':
                    warp_matrix = _safe_find_geometric_transformation(
                        im_template=im_template,
                        im_moving=im_moving,
                        mask=self.mask_geo,
                        gaussFiltSize=kwargs_algo_findTransformECC['gaussFiltSize'],
                        image_id=ii,
                        verbose=self._verbose,
                        **kwargs_algo_findTransformECC,
                    )
                warp_matrices_raw.append(warp_matrix)

            # compose warp transforms
            print('Composing geometric warp matrices...') if self._verbose else None
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

            return warp_matrices
        
        ## Run initial registration
        warp_matrices_all_to_template = _register(template=template, template_method=template_method)

        # check alignment
        im_template_global = ims_moving[template] if template_method == 'sequential' else template
        z_threshold = helpers.pvalue_to_zscore(p=self.probability_threshold, two_tailed=False)
        ## warp the images
        remappingIdx_geo_all_to_template = [helpers.warp_matrix_to_remappingIdx(warp_matrix=warp_matrix, x=W, y=H) for warp_matrix in warp_matrices_all_to_template]
        images_warped_all_to_template = self.transform_images(ims_moving=ims_moving, remappingIdx=remappingIdx_geo_all_to_template)

        ## Prepare the ImageAlignmentChecker object
        iac = ImageAlignmentChecker(
            hw=tuple(self._HW),
            radius_in=self.radius_in * self.um_per_pixel,
            radius_out=self.radius_out * self.um_per_pixel,
            order=self.order,
            device=device,
        )
        alignment_template_to_all = iac.score_alignment(
            images=images_warped_all_to_template,
            images_ref=im_template_global,
        )['z_in'][:, 0] > z_threshold
        idx_not_aligned = np.where(alignment_template_to_all == False)[0]

        if len(idx_not_aligned) > 0:
            print(f'Warning: Alignment failed for some images (probability_not_aligned > probability_threshold) for images idx: {idx_not_aligned}')
            if self.use_match_search:
                print('Attempting to find best matches for misaligned images...')
                ## Make a function that wraps up all the above steps
                def update_warps_to_template(
                    idx: Union[np.ndarray, List[int]],
                    warp_matrices_all_to_all: np.ndarray,
                    alignment_all_to_all: np.ndarray,
                ):
                    ## Register the images in idx to the template
                    for ii in idx:
                        ## Run the registration algo
                        warp_matrices_all_to_all[ii] = _register(template=ims_moving[ii], template_method='image')
                        ## warp the images
                        remappingIdx_geo_all_to_current = [helpers.warp_matrix_to_remappingIdx(warp_matrix=warp_matrix, x=W, y=H) for warp_matrix in warp_matrices_all_to_all[ii]]
                        images_warped_all_to_current = self.transform_images(ims_moving=ims_moving, remappingIdx=remappingIdx_geo_all_to_current)
                        ## Check alignment
                        alignment_all_to_all[ii] = iac.score_alignment(images=images_warped_all_to_current, images_ref=im_template_global)['z_in'][:, 0] > z_threshold
                    ## Append the template alignment_matrix (1D) on top of the all_to_all alignment_matrix (N x N)
                    alignment_matrix_all_to_all_and_template = np.concatenate([np.concatenate([np.array(0)[None,], alignment_template_to_all])[None, :], np.concatenate([alignment_template_to_all[:, None], alignment_all_to_all], axis=1)], axis=0)
                    ## Use dijkstra's algorithm to find the shortest path from every image to the template
                    distances, predecessors = scipy.sparse.csgraph.shortest_path(
                        csgraph=scipy.sparse.csr_matrix(alignment_matrix_all_to_all_and_template.astype(np.float32)),  ## 0s are disconnected, 1s are connected
                        method='D',  ## Dijkstra's algorithm
                        directed=True,
                        return_predecessors=True,
                        unweighted=True,  ## Just count the number of connected edges
                    )
                    ## Check if there are any failed paths to the template
                    if not np.any(np.isinf(distances[0])):  ## if there are no failed paths to the template
                        ## There's a path. Now compose the warp matrices by combining the all_to_template warp and the path of warps from each image to the template
                        warp_matrices_all_to_template_new = [self._compose_warps(
                            warp_0=warp_matrices_all_to_template[idx],
                            warps_to_add=[warp_matrices_all_to_all[ii] for ii in helpers.get_path_between_nodes(idx_start=idx + 1, idx_end=0, predecessors=predecessors)],  ## idx+1 and idx_end=0 because the template is the first row
                            warpMat_or_remapIdx='warpMat',
                        ) for idx in range(len(ims_moving))]  ## compose from all images to the template
                        return warp_matrices_all_to_template_new, warp_matrices_all_to_all, alignment_all_to_all
                    else:
                        ## No path. Set the warp_matrices_all_to_all to None
                        return None, warp_matrices_all_to_all, alignment_all_to_all
                    
                warp_matrices_all_to_all = np.array([[None if ii != jj else np.eye(3, dtype=np.float32) for jj in range(len(ims_moving))] for ii in range(len(ims_moving))], dtype=object)
                alignment_all_to_all = np.eye(len(ims_moving), dtype=np.bool_)
                print(f'Finding alignment between failed match idx: {idx_not_aligned} and all other images...')
                ## Register the images in idx to the template
                warp_matrices_all_to_template_new, warp_matrices_all_to_all, alignment_all_to_all = update_warps_to_template(
                    idx=idx_not_aligned, 
                    warp_matrices_all_to_all=warp_matrices_all_to_all, 
                    alignment_all_to_all=alignment_all_to_all,
                )
                if warp_matrices_all_to_template_new is not None:
                    print('All images aligned successfully after one round of path finding.') if self._verbose else None
                    warp_matrices_all_to_template = warp_matrices_all_to_template_new
                else:
                    warnings.warn('Warning: Could not find a path to alignment for some images after one round of path finding. Now doing a dense search for alignment between all images.')
                    idx_remaining = sorted(list(set(np.arange(len(ims_moving))) - set(idx_not_aligned)))
                    print(f"Finding alignment between remaining images and all other images: {idx_remaining}...") if self._verbose else None
                    ## Register the images in idx to the template
                    warp_matrices_all_to_template_new, warp_matrices_all_to_all, alignment_all_to_all = update_warps_to_template(
                        idx=idx_remaining, 
                        warp_matrices_all_to_all=warp_matrices_all_to_all, 
                        alignment_all_to_all=alignment_all_to_all,
                    )
                    if warp_matrices_all_to_template_new is not None:
                        print('All images aligned successfully after dense search.') if self._verbose else None
                        warp_matrices_all_to_template = warp_matrices_all_to_template_new
                    else:
                        warnings.warn('Warning: Could not find a path to alignment for some images after dense search. Returning the original warp matrices directly between each image and the template. \nSee')

            else:
                warnings.warn(f"Alignment failed for some images (probability_not_aligned > probability_threshold) for images idx: {idx_not_aligned}. Use 'use_match_search=True' to attempt to find best matches for misaligned images.")                
        else:
            print('All images aligned successfully!') if self._verbose else None
            alignment_all_to_all = None
        
        ## Prepare outputs
        self.warp_matrices = warp_matrices_all_to_template
        ### Convert warp matrices to remap indices
        self.remappingIdx_geo = [helpers.warp_matrix_to_remappingIdx(warp_matrix=warp_matrix, x=W, y=H) for warp_matrix in self.warp_matrices]

        ### Other outputs
        self.alignment_all_to_all = alignment_all_to_all
        self.alignment_template_to_all = alignment_template_to_all

        return self.remappingIdx_geo

    def fit_nonrigid(
        self,
        template: Union[int, np.ndarray],
        ims_moving: List[np.ndarray],
        remappingIdx_init: Optional[np.ndarray] = None,
        template_method: str = 'sequential',
        mode_transform: str = 'createOptFlow_DeepFlow',
        kwargs_mode_transform: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Perform geometric registration of ``ims_moving`` to a **template**.
        Currently relies on ``cv2.findTransformECC``.
        RH 2023

        Args:
            template (Union[int, np.ndarray]): 
                * If ``template_method`` == ``'image'``: Then **template** is
                  either an image or an integer index or a float fractional
                  index of the image to use as the **template**.
                * If ``template_method`` == ``'sequential'``: then **template**
                  is the integer index of the image to use as the **template**.
            ims_moving (List[np.ndarray]): 
                A list of images to be aligned.
            remappingIdx_init (Optional[np.ndarray]): 
                An array of shape *(N, H, W, 2)* representing any initial remap
                field to apply to the images in ``ims_moving``. The output of
                this method will be added/composed with ``remappingIdx_init``.
                (Default is ``None``)
            template_method (str): 
                The method to use for **template** selection. Either \n
                * ``'image'``: use the image specified by 'template'.
                * ``'sequential'``: register each image to the previous or next
                  image (will be next for images before the template and
                  previous for images after the template) \n
                (Default is 'sequential')
            mode_transform (str): 
                The type of transformation to use for registration. Either
                'createOptFlow_DeepFlow' or 'calcOpticalFlowFarneback'. (Default
                is 'createOptFlow_DeepFlow')
            kwargs_mode_transform (Optional[dict]): 
                Keyword arguments for the transform chosen. See cv2 docs for
                chosen transform. (Default is ``None``)

        Returns:
            (np.ndarray): 
                remapIdx_nonrigid (np.ndarray): 
                    An array of shape *(N, H, W, 2)* representing the remap
                    field for N images.
        """
        import cv2
        # Check if ims_moving is a non-empty list
        assert len(ims_moving) > 0, "ims_moving must be a non-empty list of images."
        # Check if all images in ims_moving have the same shape
        shape = ims_moving[0].shape
        for im in ims_moving:
            assert im.shape == shape, "All images in ims_moving must have the same shape."
        # Check if template_method is valid
        valid_template_methods = {'sequential', 'image'}
        assert template_method in valid_template_methods, f"template_method must be one of {valid_template_methods}"
        # Check if mode_transform is valid
        valid_mode_transforms = {'createOptFlow_DeepFlow', 'calcOpticalFlowFarneback'}
        assert mode_transform in valid_mode_transforms, f"mode_transform must be one of {valid_mode_transforms}"

        ## Store parameter (but not data) args as attributes
        self.params['fit_nonrigid'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'template',
                'template_method',
                'mode_transform',
                'kwargs_mode_transform',
            ],
        )

        # Warn if any images have values below 0 or NaN
        found_0 = np.any([np.any(im < 0) for im in ims_moving])
        found_nan = np.any([np.any(np.isnan(im)) for im in ims_moving])
        warnings.warn(f"Found images with values below 0: {found_0}. Found images with NaN values: {found_nan}") if found_0 or found_nan else None

        H, W = ims_moving[0].shape
        self._HW = (H,W) if self._HW is None else self._HW
        x_grid, y_grid = np.meshgrid(np.arange(0., W).astype(np.float32), np.arange(0., H).astype(np.float32))

        ims_moving, template = self._fix_input_images(ims_moving=ims_moving, template=template, template_method=template_method)
        norm_factor = np.nanmax([np.nanmax(im) for im in ims_moving])
        template_norm   = np.array(template * (template > 0) * (1/norm_factor) * 255, dtype=np.uint8) if template_method == 'image' else None
        ims_moving_norm = [np.array(im * (im > 0) * (1/np.nanmax(im)) * 255, dtype=np.uint8) for im in ims_moving]

        print(f'Finding nonrigid registration warps with mode: {mode_transform}, template_method: {template_method}') if self._verbose else None
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

            if mode_transform == 'calcOpticalFlowFarneback':
                self._kwargs_method = {
                    'pyr_scale': 0.3, 
                    'levels': 3,
                    'winsize': 128, 
                    'iterations': 7,
                    'poly_n': 7, 
                    'poly_sigma': 1.5,
                    'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN, ## = 256
                } if kwargs_mode_transform is None else kwargs_mode_transform
                flow_tmp = cv2.calcOpticalFlowFarneback(
                    prev=im_template,
                    next=im_moving, 
                    flow=None, 
                    **self._kwargs_method,
                )
        
            elif mode_transform == 'createOptFlow_DeepFlow':
                flow_tmp = cv2.optflow.createOptFlow_DeepFlow().calc(
                    im_template,
                    im_moving,
                    None
                )

            remappingIdx_raw.append(flow_tmp + np.stack([x_grid, y_grid], axis=-1))

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

        self.remappingIdx_nonrigid = np.stack(self.remappingIdx_nonrigid, axis=0)

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
                rois_aligned = rois_aligned.multiply(1/rois_aligned.sum(1))
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


class PhaseCorrelationRegistration:
    """
    Performs rigid transformation using phase correlation. 
    RH 2022

    Attributes:
        mask (np.ndarray):
            Spectral mask created using `set_spectral_mask()`.
        ims_registered (np.ndarray):
            Registered images, set in `register()`.
        shifts (np.ndarray):
            Pixel shift values (y, x), set in `register()`.
        ccs (np.ndarray):
            Phase correlation coefficient images, set in `register()`.
        ims_template_filt (np.ndarray):
            Template images filtered by the spectral mask, set in `register()`.
        ims_moving_filt (np.ndarray):
            Moving images filtered by the spectral mask, set in `register()`.
    """
    def __init__(self) -> None:
        """
        Initializes the PhaseCorrelationRegistration.
        """
        self.mask = None

    def set_spectral_mask(
        self, 
        freq_highPass: float = 0.01, 
        freq_lowPass: float = 0.3,
        im_shape: Tuple[int, int] = (512, 512),
    ) -> None:
        """
        Sets the spectral mask for the phase correlation.

        Args:
            freq_highPass (float): 
                High pass frequency. (Default is *0.01*)
            freq_lowPass (float): 
                Low pass frequency. (Default is *0.3*)
            im_shape (Tuple[int, int]): 
                Shape of the image. (Default is *(512, 512)*)
        """
        self.mask = make_spectral_mask(
            freq_highPass=freq_highPass,
            freq_lowPass=freq_lowPass,
            im_shape=im_shape,
        )

    def register(
        self, 
        template: Union[np.ndarray, int], 
        ims_moving: np.ndarray,
        template_method: str = 'sequential',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Registers a set of images using phase correlation. 
        RH 2022

        Args:
            template (Union[np.ndarray, int]): 
                Template image. \n
                * If ``template_method`` is 'image', ``template`` should be a
                  single image.
                * If ``template_method`` is 'sequential', ``template`` should be
                  an integer corresponding to the index of the image to set as
                  'zero' offset.
            ims_moving (np.ndarray): 
                Images to align to the template. (shape: *(n, H, W)*)
            template_method (str): 
                Method used to register the images. \n
                * 'image': ``template`` should be a single image.
                * 'sequential': ``template`` should be an integer corresponding 
                  to the index of the image to set as 'zero' offset. \n
                (Default is 'sequential')

        Returns:
            (Tuple[np.ndarray, np.ndarray]): tuple containing:
                ims_registered (np.ndarray):
                    Registered images. (shape: *(n, H, W)*)
                shifts (np.ndarray):
                    Pixel shift values (y, x). (shape: *(n, 2)*)
        """
        self.ccs, self.ims_template_filt, self.ims_moving_filt, self.shifts, self.ims_registered = [], [], [], [], []
        shift_old = np.array([0,0])
        for ii, im_moving in enumerate(ims_moving):
            if template_method == 'sequential':
                im_template = ims_moving[ii-1] if ii > 0 else ims_moving[ii]
            elif template_method == 'image':
                im_template = template

            cc, im_template_filt, im_moving_filt = phase_correlation(
                im_template,
                im_moving,
                mask_fft=self.mask,
                return_filtered_images=True,
            )
            shift = convert_phaseCorrelationImage_to_shifts(cc) + shift_old

            self.ccs.append(cc)
            self.ims_template_filt.append(im_template_filt)
            self.ims_moving_filt.append(im_moving_filt)
            self.shifts.append(shift)

            shift_old = shift.copy() if template_method == 'sequential' else shift_old
        
        self.shifts = np.stack(self.shifts, axis=0)
        self.shifts = self.shifts - self.shifts[template,:] if template_method == 'sequential' else self.shifts

        if template_method == 'sequential':
            self.ims_registered = [shift_along_axis(shift_along_axis(im_moving, shift[0] - self.shifts[template,0], fill_val=0, axis=0), shift[1] - self.shifts[template,1], fill_val=0, axis=1) for ii, (im_moving, shift) in enumerate(zip(ims_moving, self.shifts))]
        elif template_method == 'image':
            self.ims_registered = [shift_along_axis(shift_along_axis(im_moving, shift[0], fill_val=0, axis=0), shift[1], fill_val=0, axis=1) for ii, (im_moving, shift) in enumerate(zip(ims_moving, self.shifts))]

        return self.ims_registered, self.shifts


def phase_correlation(
    im_template: np.ndarray, 
    im_moving: np.ndarray, 
    mask_fft: Optional[np.ndarray] = None, 
    return_filtered_images: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Perform phase correlation on two images. Calculation performed along the
    last two axes of the input arrays (-2, -1) corresponding to the (height,
    width) of the images.
    RH 2024

    Args:
        im_template (np.ndarray): 
            The template image.
        im_moving (np.ndarray): 
            The moving image.
        mask_fft (Optional[np.ndarray]): 
            2D array mask for the FFT. If ``None``, no mask is used. Assumes mask_fft is
            fftshifted. (Default is ``None``)
        return_filtered_images (bool): 
            If set to ``True``, the function will return filtered images in
            addition to the phase correlation coefficient. (Default is
            ``False``)
    
    Returns:
        (Tuple[np.ndarray, np.ndarray, np.ndarray]): tuple containing:
            cc (np.ndarray): 
                The phase correlation coefficient.
            fft_template (np.ndarray): 
                The filtered template image. Only returned if
                return_filtered_images is ``True``.
            fft_moving (np.ndarray): 
                The filtered moving image. Only returned if
                return_filtered_images is ``True``.
    """
    fft2, fftshift, ifft2 = torch.fft.fft2, torch.fft.fftshift, torch.fft.ifft2
    abs, conj = torch.abs, torch.conj
    axes = (-2, -1)

    return_numpy = isinstance(im_template, np.ndarray)
    im_template = torch.as_tensor(im_template)
    im_moving = torch.as_tensor(im_moving)

    fft_template = fft2(im_template, dim=axes)
    fft_moving   = fft2(im_moving, dim=axes)

    if mask_fft is not None:
        mask_fft = torch.as_tensor(mask_fft)
        # Normalize and shift the mask
        mask_fft = fftshift(mask_fft / mask_fft.sum(), dim=axes)
        mask = mask_fft[tuple([None] * (im_template.ndim - 2) + [slice(None)] * 2)]
        fft_template *= mask
        fft_moving *= mask

    # Compute the cross-power spectrum
    R = fft_template * conj(fft_moving)

    # Normalize to obtain the phase correlation function
    R /= abs(R) + 1e-8  # Add epsilon to prevent division by zero

    # Compute the magnitude of the inverse FFT to ensure symmetry
    # cc = abs(fftshift(ifft2(R, dim=axes), dim=axes))
    # Compute the real component of the inverse FFT (not symmetric)
    cc = fftshift(ifft2(R, dim=axes), dim=axes).real

    if return_filtered_images == False:
        return cc.cpu().numpy() if return_numpy else cc
    else:
        if return_numpy:
            return (
                cc.cpu().numpy(), 
                abs(ifft2(fft_template, dim=axes)).cpu().numpy(), 
                abs(ifft2(fft_moving, dim=axes)).cpu().numpy()
            )
        else:
            return cc, abs(ifft2(fft_template, dim=axes)), abs(ifft2(fft_moving, dim=axes))


def convert_phaseCorrelationImage_to_shifts(cc_im: np.ndarray) -> Tuple[int, int]:
    """
    Convert phase correlation image to pixel shift values.
    RH 2022

    Args:
        cc_im (np.ndarray):
            Phase correlation image. The middle of the image corresponds to a
            zero-shift.

    Returns:
        (Tuple[int, int]): tuple containing:
            shift_y (int):
                The pixel shift in the y-axis.
            shift_x (int):
                The pixel shift in the x-axis.
    """
    height, width = cc_im.shape
    shift_y_raw, shift_x_raw = np.unravel_index(cc_im.argmax(), cc_im.shape)
    return int(np.floor(height/2) - shift_y_raw) , int(np.ceil(width/2) - shift_x_raw)


def _helper_shift(X, shift, fill_val=0):
    X_shift = np.empty_like(X, dtype=X.dtype)
    if shift>0:
        X_shift[:shift] = fill_val
        X_shift[shift:] = X[:-shift]
    elif shift<0:
        X_shift[shift:] = fill_val
        X_shift[:shift] = X[-shift:]
    else:
        X_shift[:] = X
    return X_shift
def shift_along_axis(
    X: np.ndarray, 
    shift: int, 
    fill_val: int = 0, 
    axis: int = 0
) -> np.ndarray:
    """
    Shifts the elements of an array along a specified axis.
    RH 2023

    Args:
        X (np.ndarray): 
            Input array to be shifted.
        shift (int): 
            The number of places to shift. If the value is positive, the shift
            is to the right. If the value is negative, the shift is to the left.
        fill_val (int): 
            The value to fill in the emptied places after the shift. (Default is
            ``0``)
        axis (int): 
            The axis along which to apply the shift. (Default is ``0``)

    Returns:
        (np.ndarray): 
            shifted_array (np.ndarray):
                The array after shifting elements along the specified axis.
    """
    return np.apply_along_axis(_helper_shift, axis, np.array(X, dtype=X.dtype), shift, fill_val)


def make_spectral_mask(
    freq_highPass: float = 0.01, 
    freq_lowPass: float = 0.3, 
    im_shape: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """
    Generates a spectral mask for an image with given high pass and low pass frequencies.

    Args:
        freq_highPass (float): 
            High pass frequency to use. 
            (Default is ``0.01``)
        freq_lowPass (float): 
            Low pass frequency to use. 
            (Default is ``0.3``)
        im_shape (Tuple[int, int]): 
            Shape of the input image as a tuple *(height, width)*. 
            (Default is *(512, 512)*)

    Returns:
        (np.ndarray): 
            mask_out (np.ndarray): 
                The generated spectral mask.
    """
    height, width = im_shape[0], im_shape[1]
    
    idx_highPass = (int(np.ceil(height * freq_highPass / 2)), int(np.ceil(width * freq_highPass / 2)))
    idx_lowPass = (int(np.floor(height * freq_lowPass / 2)), int(np.floor(width * freq_lowPass / 2)))

    if freq_lowPass < 1:
        mask = np.ones((height, width))
        mask[idx_lowPass[0]:-idx_lowPass[0]+1,:] = 0
        mask[:,idx_lowPass[1]:-idx_lowPass[1]+1] = 0
    else:
        mask = np.ones((height, width))

    mask_high = np.fft.fftshift(mask)

    if (idx_highPass[0] > 0) and (idx_highPass[1] > 0):
        mask = np.zeros((height, width))
        mask[idx_highPass[0]:-idx_highPass[0],:] = 1
        mask[:,idx_highPass[1]:-idx_highPass[1]] = 1
    else:
        mask = np.ones((height, width))

    mask_low = np.fft.fftshift(mask)

    mask_out = mask_high * mask_low
    
    return mask_out


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
    import cv2
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


class ImageAlignmentChecker:
    """
    Class to check the alignment of images using phase correlation.
    RH 2024

    Args:
        hw (Tuple[int, int]): 
            Height and width of the images.
        radius_in (Union[float, Tuple[float, float]]): 
            Radius of the pixel shift / offset that can be considered as
            'aligned'. Used to create the 'in' filter which is an image of a
            small centered circle that is used as a filter and multiplied by
            the phase correlation images. If a single value is provided, the
            filter will be a circle with radius 0 to that value; it will be
            converted to a tuple representing a bandpass filter (0, radius_in).
        radius_out (Union[float, Tuple[float, float]]):
            Similar to radius_in, but for the 'out' filter, which defines the
            'null distribution' for defining what is 'aligned'. Should be a
            value larger than the expected maximum pixel shift / offset. If a
            single value is provided, the filter will be a donut / taurus
            starting at that value and ending at the edge of the smallest
            dimension of the image; it will be converted to a tuple representing
            a bandpass filter (radius_out, min(hw)).
        order (int):
            Order of the butterworth bandpass filters used to define the 'in'
            and 'out' filters. Larger values will result in a sharper edges, but
            values higher than 5 can lead to collapse of the filter.
        device (str):
            Torch device to use for computations. (Default is 'cpu')

    Attributes:
        hw (Tuple[int, int]): 
            Height and width of the images.
        order (int):
            Order of the butterworth bandpass filters used to define the 'in'
            and 'out' filters.
        device (str):
            Torch device to use for computations.
        filt_in (torch.Tensor):
            The 'in' filter used for scoring the alignment.
        filt_out (torch.Tensor):
            The 'out' filter used for scoring the alignment.
    """
    def __init__(
        self,
        hw: Tuple[int, int],
        radius_in: Union[float, Tuple[float, float]],
        radius_out: Union[float, Tuple[float, float]],
        order: int,
        device: str,
    ):
        ## Set attributes
        ### Convert to torch.Tensor
        self.hw = tuple(hw)

        ### Set other attributes
        self.order = int(order)
        self.device = str(device)
        ### Set filter attributes
        if isinstance(radius_in, (int, float, complex)):
            radius_in = (float(0.0), float(radius_in))
        elif isinstance(radius_in, (tuple, list, np.ndarray, torch.Tensor)):
            radius_in = tuple(float(r) for r in radius_in)
        else:
            raise ValueError(f'radius_in must be a float or tuple of floats. Found type: {type(radius_in)}')
        if isinstance(radius_out, (int, float, complex)):
            radius_out = (float(radius_out), float(min(self.hw)) / 2)
        elif isinstance(radius_out, (tuple, list, np.ndarray, torch.Tensor)):
            radius_out = tuple(float(r) for r in radius_out)
        else:
            raise ValueError(f'radius_out must be a float or tuple of floats. Found type: {type(radius_out)}')

        ## Make filters
        self.filt_in, self.filt_out = (torch.as_tensor(self._make_filter(
            hw=self.hw,
            low=bp[0],
            high=bp[1],
            order=order,
        ), dtype=torch.float32, device=device) for bp in [radius_in, radius_out])
    
    def _make_filter(
        self,
        hw: tuple,
        low: float = 5,
        high: float = 6,
        order: int = 3,
    ):
        ## Make a distance grid starting from the fftshifted center
        grid = helpers.make_distance_grid(shape=hw, p=2, use_fftshift_center=True)

        ## Make the number of datapoints for the kernel large
        n_x = max(hw) * 10

        fs = max(hw) * 1
        b, a = helpers.design_butter_bandpass(lowcut=low, highcut=high, fs=fs, order=order, plot_pref=False)
        w, h = scipy.signal.freqz(b, a, worN=n_x)
        x_kernel = (fs * 0.5 / np.pi) * w
        kernel = np.abs(h)

        ## Interpolate the kernel to the distance grid
        filt = np.interp(
            x=grid,
            xp=x_kernel,
            fp=kernel,
        )

        return filt
    
    def score_alignment(
        self,
        images: Union[np.ndarray, torch.Tensor],
        images_ref: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ):
        """
        Score the alignment of a set of images using phase correlation. Computes
        the stats of the center ('in') of the phase correlation image over the
        stats of the outer region ('out') of the phase correlation image.
        RH 2024

        Args:
            images (Union[np.ndarray, torch.Tensor]): 
                A 3D array of images. Shape: *(n_images, height, width)*
            images_ref (Optional[Union[np.ndarray, torch.Tensor]]):
                Reference images to compare against. If provided, the images
                will be compared against these images. If not provided, the
                images will be compared against themselves. (Default is
                ``None``)

        Returns:
            (Dict): 
                Dictionary containing the following keys:
                * 'mean_out': 
                    Mean of the phase correlation image weighted by the
                    'out' filter
                * 'mean_in': 
                    Mean of the phase correlation image weighted by the
                    'in' filter
                * 'ptile95_out': 
                    95th percentile of the phase correlation image multiplied by
                    the 'out' filter
                * 'max_in': 
                    Maximum value of the phase correlation image multiplied by
                    the 'in' filter
                * 'std_out': 
                    Standard deviation of the phase correlation image weighted by
                    the 'out' filter
                * 'std_in': 
                    Standard deviation of the phase correlation image weighted by
                    the 'in' filter
                * 'max_diff': 
                    Difference between the 'max_in' and 'ptile95_out' values
                * 'z_in': 
                    max_diff divided by the 'std_out' value
                * 'r_in': 
                    max_diff divided by the 'ptile95_out' value
        """
        def _fix_images(ims):
            assert isinstance(ims, (np.ndarray, torch.Tensor, list, tuple)), f'images must be np.ndarray, torch.Tensor, or a list/tuple of np.ndarray or torch.Tensor. Found type: {type(ims)}'
            if isinstance(ims, (list, tuple)):
                assert all(isinstance(im, (np.ndarray, torch.Tensor)) for im in ims), f'images must be np.ndarray or torch.Tensor. Found types: {set(type(im) for im in ims)}'
                assert all(im.ndim == 2 for im in ims), f'images must be 2D arrays (height, width). Found shapes: {set(im.shape for im in ims)}'
                if isinstance(ims[0], np.ndarray):
                    ims = np.stack([np.array(im) for im in ims], axis=0)
                else:
                    ims = torch.stack([torch.as_tensor(im) for im in ims], dim=0)
            else:
                if ims.ndim == 2:
                    ims = ims[None, :, :]
                assert ims.ndim == 3, f'images must be a 3D array (n_images, height, width). Found shape: {ims.shape}'
                assert ims.shape[1:] == self.hw, f'images must have shape (n_images, {self.hw[0]}, {self.hw[1]}). Found shape: {ims.shape}'

            ims = torch.as_tensor(ims, dtype=torch.float32, device=self.device)
            return ims

        images = _fix_images(images)
        images_ref = _fix_images(images_ref) if images_ref is not None else images
        
        pc = phase_correlation(images_ref[None, :, :, :], images[:, None, :, :])  ## All to all phase correlation. Shape: (n_images, n_images, height, width)

        ## metrics
        filt_in, filt_out = self.filt_in[None, None, :, :], self.filt_out[None, None, :, :]
        mean_out = (pc * filt_out).sum(dim=(-2, -1)) / filt_out.sum(dim=(-2, -1))
        mean_in =  (pc * filt_in).sum(dim=(-2, -1))  / filt_in.sum(dim=(-2, -1))
        ptile95_out = torch.quantile((pc * filt_out).reshape(pc.shape[0], pc.shape[1], -1)[:, :, filt_out.reshape(-1) > 1e-3], 0.95, dim=-1)
        max_in = (pc * filt_in).amax(dim=(-2, -1))
        std_out = torch.sqrt(torch.mean((pc - mean_out[:, :, None, None])**2 * filt_out, dim=(-2, -1)))
        std_in = torch.sqrt(torch.mean((pc - mean_in[:, :, None, None])**2 * filt_in, dim=(-2, -1)))

        max_diff = max_in - ptile95_out
        z_in = max_diff / std_out
        r_in = max_diff / ptile95_out

        outs = {
            'pc': pc.cpu().numpy(),
            'mean_out': mean_out,
            'mean_in': mean_in,
            'ptile95_out': ptile95_out,
            'max_in': max_in,
            'std_out': std_out,
            'std_in': std_in,
            'max_diff': max_diff,
            'z_in': z_in,  ## z-score of in value over out distribution
            'r_in': r_in,
        }

        outs = {k: val.cpu().numpy() if isinstance(val, torch.Tensor) else val for k, val in outs.items()}
        
        return outs
    
    def __call__(
        self,
        images: Union[np.ndarray, torch.Tensor],
    ):
        """
        Calls the `score_alignment` method. See `self.score_alignment` docstring
        for more info.
        """
        return self.score_alignment(images)
    

def _safe_find_geometric_transformation(
    im_template: np.ndarray,
    im_moving: np.ndarray,
    mask: np.ndarray,
    gaussFiltSize,
    image_id: Optional[Any] = None,
    verbose: bool = False,
    **kwargs_algo_findTransformECC,
):
    """
    Safe version of find_geometric_transformation used in Aligner class (wrapper
    for cv2.findTransformECC). If alignment fails during optimization, this
    function will increase the gaussFiltSize parameter and try again.

    Args:
        im_template (np.ndarray):
            Template image.
        im_moving (np.ndarray):
            Moving image.
        mask (np.ndarray):
            Mask for the images.
        gaussFiltSize (int):
            Size of the Gaussian filter used to blur the images before
            registration.
        image_id (Optional[Any]):
            Identifier for the image. Used for descriptive error messages.
        verbose (bool):
            Whether to print verbose output.
        **kwargs_algo_findTransformECC:
            * mode_transform (str): Mode of the transformation. One of
              ['translation', 'euclidean', 'affine', 'homography'].
            * n_iter (int): Number of iterations for the optimization.
            * termination_eps (float): Termination epsilon for the optimization.
            * auto_fix_gaussFilt_step (int): Step size to increase gaussFiltSize
              by if alignment fails.

    Returns:
        (np.ndarray): 
            Warp matrix for the geometric transformation. If the alignment
            fails, the function will return an identity matrix. Shape: *(2, 3)*
            for mode_transform in ['translation', 'euclidean', 'affine'], *(3,
            3)* for mode_transform='homography'.
    """
    mode_transform, n_iter, termination_eps, auto_fix_gaussFilt_step = [kwargs_algo_findTransformECC[key] for key in [
        'mode_transform', 'n_iter', 'termination_eps', 'auto_fix_gaussFilt_step'
    ]]

    # Check if mode_transform is valid
    valid_mode_transforms = {'translation', 'euclidean', 'affine', 'homography'}
    assert mode_transform in valid_mode_transforms, f"mode_transform must be one of {valid_mode_transforms}"
    # Check if gaussFiltSize is a number (float or int)
    assert isinstance(gaussFiltSize, (float, int)), "gaussFiltSize must be a number."
    # Convert gaussFiltSize to an odd integer
    gaussFiltSize = int(np.round(gaussFiltSize))

    try:
        warp_matrix = helpers.find_geometric_transformation(
            im_template=im_template,
            im_moving=im_moving,
            warp_mode=mode_transform,
            n_iter=n_iter,
            mask=mask,
            termination_eps=termination_eps,
            gaussFiltSize=gaussFiltSize,
        )
    except Exception as e:
        if auto_fix_gaussFilt_step:
            print(f'Error finding geometric registration warp for image {image_id}: {e}') if verbose else None
            print(f'Increasing gaussFiltSize by {auto_fix_gaussFilt_step} to {gaussFiltSize + auto_fix_gaussFilt_step}') if verbose else None
            return _safe_find_geometric_transformation(
                im_template=im_template,
                im_moving=im_moving,
                mask=mask,
                gaussFiltSize=gaussFiltSize + auto_fix_gaussFilt_step,
                image_id=image_id,
                verbose=verbose,
                **kwargs_algo_findTransformECC,
            )

        print(f'Error finding geometric registration warp for image {image_id}: {e}')
        print(f'Defaulting to identity matrix warp.')
        print(f'Consider doing one of the following:')
        print(f'  - Make better images to input. You can add the spatialFootprints images to the FOV images to make them better.')
        print(f'  - Increase the gaussFiltSize parameter. This will make the images blurrier, but may help with registration.')
        print(f'  - Decrease the termination_eps parameter. This will make the registration less accurate, but may help with registration.')
        print(f'  - Increase the mask_borders parameter. This will make the images smaller, but may help with registration.')
        warp_matrix = np.eye(3)[:2,:] if mode_transform != 'homography' else np.eye(3)
    return warp_matrix


def _LoFTR_fit(
    im_template: Union[np.ndarray, torch.Tensor],
    im_moving: Union[np.ndarray, torch.Tensor],
    mask_borders: Tuple[int, int, int, int],
    image_id: Optional[Any] = None,
    device: str = 'cpu',
    verbose: bool = False,
    **kwargs_algo_LoFTR,
):
    """
    Fit a geometric transformation using LoFTR and RANSAC.

    Args:
        im_template (Union[np.ndarray, torch.Tensor]):
            Template image.
        im_moving (Union[np.ndarray, torch.Tensor]):
            Moving image.
        mask_borders (Tuple[int, int, int, int]):
            Borders to mask from the images. Format: (top, bottom, left, right)
        image_id (Optional[Any]):
            Identifier for the image. Used for descriptive error messages.
        verbose (bool):
            Whether to print verbose output.
        **kwargs_algo_LoFTR:
            * mode_transform (str): Mode of the transformation. One of
              ['euclidean', 'similarity', 'affine', 'projective'].
            * gaussFiltSize (int): Size of the Gaussian filter used to blur the
              images before registration.
            * confidence_LoFTR (float): Confidence threshold for LoFTR matches.
            * confidence_RANSAC (float): Confidence threshold for RANSAC.
            * ransacReprojThreshold (float): RANSAC reprojection threshold.
            * maxIters (int): Maximum number of iterations for RANSAC.
    """
    import kornia
    import skimage
    import cv2

    mode_transform, gaussFiltSize, confidence_LoFTR, confidence_RANSAC, ransacReprojThreshold, maxIters = [kwargs_algo_LoFTR[key] for key in [
        'mode_transform', 'gaussFiltSize', 'confidence_LoFTR', 'confidence_RANSAC', 'ransacReprojThreshold', 'maxIters'
    ]]

    ## Confirm that the images are floats between 0 and 1
    assert all([np.issubdtype(im.dtype, np.floating) for im in [im_template, im_moving]]), 'Images must be floating dtype.'
    assert all([np.min(im) >= 0 and np.max(im) <= 1 for im in [im_template, im_moving]]), 'Images must be between 0 and 1.'
    ## Prepare input dictionary
    input_dict = {
        'image0': torch.as_tensor(im_template, dtype=torch.float32, device=device)[None, None, :, :],
        'image1': torch.as_tensor(im_moving, dtype=torch.float32, device=device)[None, None, :, :],
    }
    ## Mask the images by cropping out the borders
    b_top, b_bottom, b_left, b_right = mask_borders[0], input_dict['image0'].shape[2] - mask_borders[1], mask_borders[2], input_dict['image0'].shape[3] - mask_borders[3]
    input_dict['image0'] = input_dict['image0'][:, :, b_top:b_bottom][:, :, :, b_left:b_right]
    input_dict['image1'] = input_dict['image1'][:, :, b_top:b_bottom][:, :, :, b_left:b_right]

    # ## Normalize the images for LoFTR
    # ### LoFTR expects images to be in the range [0, 1]
    # input_dict['image0'] = input_dict['image0'] / input_dict['image0'].max()
    # input_dict['image1'] = input_dict['image1'] / input_dict['image1'].max()

    ## Blur the images
    if gaussFiltSize > 0:
        input_dict['image0'] = kornia.filters.gaussian_blur2d(input=input_dict['image0'], kernel_size=gaussFiltSize, sigma=((s:=gaussFiltSize/6), s))
        input_dict['image1'] = kornia.filters.gaussian_blur2d(input=input_dict['image1'], kernel_size=gaussFiltSize, sigma=((s:=gaussFiltSize/6), s))

    ## Load LoFTR model
    model = kornia.feature.LoFTR(pretrained="indoor_new")
    model = model.to(device)
    ## Get the keypoints and descriptors
    with torch.inference_mode():
        matches = model(input_dict)
    ## Clean up matches with low confidence
    mkpts0 = matches["keypoints0"].cpu().numpy()
    mkpts1 = matches["keypoints1"].cpu().numpy()
    print(f"Found {mkpts0.shape[0]} matches.") if verbose else None
    mkpts0 = mkpts0[(matches["confidence"] > confidence_LoFTR).cpu().numpy().astype(np.bool_)]
    mkpts1 = mkpts1[(matches["confidence"] > confidence_LoFTR).cpu().numpy().astype(np.bool_)]
    ## Throw an error if no matches are found
    print(f"Found {mkpts0.shape[0]} matches.") if verbose else None
    if mkpts0.shape[0] < 2:
        print(f'Registration Failed for image_moving: {image_id}. Not enough matches found. Setting warp matrix to identity. Found: {mkpts0.shape[0]} matches. Try adjusting one of the following: decrease confidence_LoFTR, decrease gaussFiltSize, increase mask_borders.')
        return np.eye(3)
    ## Get inliers
    try:
        fund_mat, inliers = cv2.findFundamentalMat(
            points1=mkpts0,
            points2=mkpts1,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=ransacReprojThreshold,
            confidence=confidence_RANSAC,
            maxIters=maxIters,
        )
    except Exception as e:
        print(f'Registration Failed for image_moving: {image_id}. Error finding fundamental matrix. Setting warp matrix to identity. Likely not enough matches found. Found: {mkpts0.shape[0]} matches. Try adjusting one of the following: decrease confidence_LoFTR, decrease gaussFiltSize, increase mask_borders. \nError: {e}')
        return np.eye(3)
    inliers = np.array(inliers > 0, dtype=np.bool_).ravel()
    # ## Remove outliers
    # mkpts0 = mkpts0[inliers].astype(np.float32)
    # mkpts1 = mkpts1[inliers].astype(np.float32)

    ## Use skimage's estimate_transform
    tforms = ['euclidean', 'similarity', 'affine', 'projective']
    assert mode_transform in tforms, f'Found mode_transform={mode_transform}. Must be one of {tforms.keys()}'
    warp_matrix = skimage.transform.estimate_transform(
        ttype=mode_transform,
        src=mkpts0,
        dst=mkpts1,
    ).params

    # plt.figure()
    # plt.scatter(mkpts0[:,0,0], mkpts1[:,0,0], c='r')
    # plt.scatter(mkpts0[:,0,1], mkpts1[:,0,1], c='b')
    # plt.show()

    ## Get the transformation matrix (can be homography or affine, etc.)
    # if mode_transform == 'homography':
    #     warp_matrix, inliers = cv2.findHomography(
    #         mkpts0,
    #         mkpts1,
    #         method=cv2.RANSAC,
    #         ransacReprojThreshold=ransacReprojThreshold,
    #         confidence=confidence_RANSAC,
    #         maxIters=maxIters,
    #     )
    #     # print(warp_matrix.shape)
    #     # warp_matrix = np.linalg.inv(warp_matrix)
    #     # warp_matrix = warp_matrix[:2, :].astype(np.float32)
    # elif mode_transform == 'affine':
    #     ## Don't use estimateAffinePartial2D because it doesn't work well
    #     warp_matrix, mask = cv2.estimate(
    #         mkpts0,
    #         mkpts1,
    #         method=cv2.RANSAC,
    #         ransacReprojThreshold=ransacReprojThreshold,
    #         maxIters=maxIters,
    #         confidence=confidence_RANSAC,
    #     )
    ## Use skimage's estimate_transform

    # # print(warp_matrix)
    # warp_matrix = np.linalg.inv(warp_matrix)
    # # else:
    # #     raise ValueError(f'LoFTR does not support mode_transform={mode_transform}')

    
    # print(warp_matrix)
    # warp_matrix = warp_matrix[:2, :].astype(np.float32)
    ## Warp image
    # im_moving_warped = cv2.warpAffine(
    #     im_moving,
    #     warp_matrix,
    #     (W, H),
    #     flags=cv2.INTER_LINEAR,
    #     borderMode=cv2.BORDER_CONSTANT,
    #     borderValue=float(im_moving.mean()),
    # )
    # im_moving_warped = cv2.warpPerspective(
    #     im_moving,
    #     warp_matrix,
    #     (W, H),
    #     flags=cv2.INTER_LINEAR,
    #     borderMode=cv2.BORDER_CONSTANT,
    #     borderValue=float(im_moving.mean()),
    # )
    # im_moving_warped = kornia.geometry.warp_perspective(
    #     torch.as_tensor(im_moving).view(1, 1, H, W),
    #     torch.as_tensor(warp_matrix).view(1, 3, 3),
    #     dsize=(H, W),
    #     align_corners=False,
    # ).squeeze().numpy()[]
    ## Show the images
    # from .. import visualization
    # visualization.display_toggle_image_stack([im_template, im_moving_warped],)
    # from kornia_moons.viz import draw_LAF_matches
    # draw_LAF_matches(
    #     kornia.feature.laf_from_center_scale_ori(
    #         torch.from_numpy(mkpts0).view(1, -1, 2),
    #         torch.ones(mkpts0.shape[0]).view(1, -1, 1, 1),
    #         torch.ones(mkpts0.shape[0]).view(1, -1, 1),
    #     ),
    #     kornia.feature.laf_from_center_scale_ori(
    #         torch.from_numpy(mkpts1).view(1, -1, 2),
    #         torch.ones(mkpts1.shape[0]).view(1, -1, 1, 1),
    #         torch.ones(mkpts1.shape[0]).view(1, -1, 1),
    #     ),
    #     torch.arange(mkpts0.shape[0]).view(-1, 1).repeat(1, 2),
    #     kornia.tensor_to_image(torch.as_tensor(input_dict['image0'])),
    #     kornia.tensor_to_image(torch.as_tensor(input_dict['image1'])),
    #     inliers,
    #     draw_dict={"inlier_color": (0.2, 1, 0.2), "tentative_color": None, "feature_color": (0.2, 0.5, 1), "vertical": False},
    # )

    return warp_matrix