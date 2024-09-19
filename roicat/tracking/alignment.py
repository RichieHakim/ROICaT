import copy
import typing
import warnings
from typing import List, Tuple, Union, Optional, Dict, Any, Sequence, Callable
import functools

import numpy as np
import scipy.sparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from .. import helpers, util

class Aligner(util.ROICaT_Module):
    """
    A class for registering ROIs to a template FOV. Currently relies on
    available OpenCV methods for rigid and non-rigid registration.
    RH 2023

    Args:
        verbose (bool):
            Whether to print progress updates. (Default is ``True``)
    """
    def __init__(
        self,
        verbose=True,
    ):
        ## Store parameter (but not data) args as attributes
        self.params['__init__'] = self._locals_to_params(
            locals_dict=locals(),
            keys=['verbose'],
        )

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
        CLAHE_grid_size: int = 1,
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
            CLAHE_grid_size (int):
                The grid size for CLAHE. See alignment.clahe for more details.
                (Default is *1*)
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
                'CLAHE_grid_size',
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
            FOV_images = [clahe(im, grid_size=CLAHE_grid_size, clipLimit=CLAHE_clipLimit, normalize=CLAHE_normalize) for im in FOV_images]

        ## normalize FOV images
        if normalize_FOV_intensities:
            val_norm = np.max(np.concatenate([im.reshape(-1) for im in FOV_images]))
            FOV_images = [im / val_norm for im in FOV_images]

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
        mode_transform: str = 'affine',
        gaussFiltSize: int = 11,
        mask_borders: Tuple[int, int, int, int] = (0, 0, 0, 0),
        n_iter: int = 1000,
        termination_eps: float = 1e-9,
        auto_fix_gaussFilt_step: Union[bool, int] = 10,
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
            mode_transform (str): 
                Mode of geometric transformation. Can be 'translation', 'euclidean', 
                'affine', or 'homography'. See ``cv2.findTransformECC`` for more details. 
                (Default is 'affine')
            gaussFiltSize (int): 
                Size of the Gaussian filter. (Default is *11*)
            mask_borders (Tuple[int, int, int, int]): 
                Border mask for the image. Format is (top, bottom, left, right). 
                (Default is (0, 0, 0, 0))
            n_iter (int): 
                Number of iterations for ``cv2.findTransformECC``. (Default is *1000*)
            termination_eps (float): 
                Termination criteria for ``cv2.findTransformECC``. (Default is *1e-9*)
            auto_fix_gaussFilt_step (Union[bool, int]): 
                Automatically fixes convergence issues by increasing the gaussFiltSize. 
                If ``False``, no automatic fixing is performed. If ``True``, the 
                gaussFiltSize is increased by 2 until convergence. If int, the gaussFiltSize 
                is increased by this amount until convergence. (Default is *10*)

        Returns:
            (np.ndarray): 
                remapIdx_geo (np.ndarray): 
                    An array of shape *(N, H, W, 2)* representing the remap field for N images.
        """
        ## Imports
        super().__init__()

        ## Store parameter (but not data) args as attributes
        self.params['fit_geometric'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'template' if isinstance(template, int) else None,
                'template_method',
                'mode_transform',
                'gaussFiltSize',
                'mask_borders',
                'n_iter',
                'termination_eps',
                'auto_fix_gaussFilt_step',
            ],
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
        # Check if mode_transform is valid
        valid_mode_transforms = {'translation', 'euclidean', 'affine', 'homography'}
        assert mode_transform in valid_mode_transforms, f"mode_transform must be one of {valid_mode_transforms}"
        # Check if gaussFiltSize is a number (float or int)
        assert isinstance(gaussFiltSize, (float, int)), "gaussFiltSize must be a number."
        # Convert gaussFiltSize to an odd integer
        gaussFiltSize = int(np.round(gaussFiltSize))

        H, W = ims_moving[0].shape
        self._HW = (H,W) if self._HW is None else self._HW

        ims_moving, template = self._fix_input_images(ims_moving=ims_moving, template=template, template_method=template_method)

        self.mask_geo = helpers.mask_image_border(
            im=np.ones((H, W), dtype=np.uint8),
            border_outer=mask_borders,
            mask_value=0,
        )

        print(f'Finding geometric registration warps with mode: {mode_transform}, template_method: {template_method}, mask_borders: {mask_borders is not None}') if self._verbose else None
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
            
            def _safe_find_geometric_transformation(gaussFiltSize):
                try:
                    warp_matrix = helpers.find_geometric_transformation(
                        im_template=im_template,
                        im_moving=im_moving,
                        warp_mode=mode_transform,
                        n_iter=n_iter,
                        mask=self.mask_geo,
                        termination_eps=termination_eps,
                        gaussFiltSize=gaussFiltSize,
                    )
                except Exception as e:
                    if auto_fix_gaussFilt_step:
                        print(f'Error finding geometric registration warp for image {ii}: {e}') if self._verbose else None
                        print(f'Increasing gaussFiltSize by {auto_fix_gaussFilt_step} to {gaussFiltSize + auto_fix_gaussFilt_step}') if self._verbose else None
                        return _safe_find_geometric_transformation(gaussFiltSize + auto_fix_gaussFilt_step)

                    print(f'Error finding geometric registration warp for image {ii}: {e}')
                    print(f'Defaulting to identity matrix warp.')
                    print(f'Consider doing one of the following:')
                    print(f'  - Make better images to input. You can add the spatialFootprints images to the FOV images to make them better.')
                    print(f'  - Increase the gaussFiltSize parameter. This will make the images blurrier, but may help with registration.')
                    print(f'  - Decrease the termination_eps parameter. This will make the registration less accurate, but may help with registration.')
                    print(f'  - Increase the mask_borders parameter. This will make the images smaller, but may help with registration.')
                    warp_matrix = np.eye(3)[:2,:]
                return warp_matrix
            
            warp_matrix = _safe_find_geometric_transformation(gaussFiltSize=gaussFiltSize)
            warp_matrices_raw.append(warp_matrix)

        
        # compose warp transforms
        print('Composing geometric warp matrices...') if self._verbose else None
        self.warp_matrices = []
        if template_method == 'sequential':
            ## compose warps before template forward (t1->t2->t3->t4)
            for ii in np.arange(0, template):
                warp_composed = self._compose_warps(
                    warp_0=warp_matrices_raw[ii], 
                    warps_to_add=warp_matrices_raw[ii+1:template+1],
                    warpMat_or_remapIdx='warpMat',
                )
                self.warp_matrices.append(warp_composed)
            ## compose template to itself
            self.warp_matrices.append(warp_matrices_raw[template])
            ## compose warps after template backward (t4->t3->t2->t1)
            for ii in np.arange(template+1, len(ims_moving)):
                warp_composed = self._compose_warps(
                    warp_0=warp_matrices_raw[ii], 
                    warps_to_add=warp_matrices_raw[template:ii][::-1],
                    warpMat_or_remapIdx='warpMat',
                )
                self.warp_matrices.append(warp_composed)
        ## no composition when template_method == 'image'
        elif template_method == 'image':
            self.warp_matrices = warp_matrices_raw

        self.warp_matrices = np.stack(self.warp_matrices, axis=0)

        # convert warp matrices to remap indices
        self.remappingIdx_geo = np.stack([helpers.warp_matrix_to_remappingIdx(warp_matrix=warp_matrix, x=W, y=H) for warp_matrix in self.warp_matrices], axis=0)

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
                'template' if isinstance(template, int) else None,
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
                The remapping index to apply to the images.

        Returns:
            (List[np.ndarray]): 
                ims_registered (List[np.ndarray]): 
                    The transformed images. *(N, H, W)*
        """

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
        return ims_registered    

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
    Perform phase correlation on two images.
    RH 2022

    Args:
        im_template (np.ndarray): 
            The template image.
        im_moving (np.ndarray): 
            The moving image.
        mask_fft (Optional[np.ndarray]): 
            Mask for the FFT. If ``None``, no mask is used. (Default is
            ``None``)
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
    if mask_fft is None:
        mask_fft = np.ones(im_template.shape)
    else:
        mask_fft = np.fft.fftshift(mask_fft/mask_fft.sum())

    fft_template = np.fft.fft2(im_template) * mask_fft
    fft_moving   = np.fft.fft2(im_moving) * mask_fft
    R = np.conj(fft_template) * fft_moving
    R[mask_fft != 0] /= np.abs(R)[mask_fft != 0]
    cc = np.fft.fftshift(np.fft.ifft2(R)).real
    if return_filtered_images == False:
        return cc
    else:
        return cc, np.abs(np.fft.ifft2(fft_template)), np.abs(np.fft.ifft2(fft_moving))


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
    grid_size: int = 50, 
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
    dtype_in = im.dtype
    if normalize:
        val_max = np.nanmax(im)
        im_tu = (im.astype(np.float32) / val_max)*(2**8 - 1)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid_size, grid_size))
    im_c = clahe.apply(im_tu.astype(np.uint16))
    if normalize:
        im_c = (im_c / (2**8 - 1)) * val_max
    im_c = im_c.astype(dtype_in)
    return im_c
