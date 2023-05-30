import copy
import typing
import warnings

import numpy as np
import cv2
import scipy.sparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from .. import helpers, util

class Aligner(util.ROICaT_Module):
    """
    A class for registering ROIs to a template FOV.
    Currently relies on available OpenCV methods for 
     non-rigid registration.
    RH 2023
    """
    def __init__(
        self,
        verbose=True,
    ):
        """
        Initialize the class.

        Args:
            verbose (bool):
                Whether to print progress updates.
        """
        self._verbose = verbose
        
        self.remapingIdx_geo = None
        self.warp_matrices = None

        self.remappingIdx_nonrigid = None

        self._HW = None

    def fit_geometric(
        self,
        template: typing.Union[int, np.ndarray],
        ims_moving: typing.List[np.ndarray],
        template_method: str = 'sequential',
        mode_transform: str = 'affine',
        gaussFiltSize: int = 11,
        mask_borders: typing.Tuple[int, int, int, int] = (0, 0, 0, 0),
        n_iter: int = 1000,
        termination_eps: float = 1e-9,
        auto_fix_gaussFilt_step: int = 10,
    ) -> np.ndarray:
        """
        Perform geometric registration of ims_moving to a template.
        Currently relies on cv2.findTransformECC.
        RH 2023

        Args:
            template (int or np.ndarray):
                If 'template_method' == 'image', this is either a 2D np.ndarray
                 image, an integer index of the image to use as the template, or
                 a float between 0 and 1 representing the fractional index of the
                 image to use as the template.
                If 'template_method' == 'sequential', then template is the
                 integer index or fractional index of the image to use as the
                 template.
            ims_moving (list of np.ndarray):
                A list of images to be aligned.
            template_method (str, optional):
                The method to use for template selection.
                'image': use the image specified by 'template'.
                'sequential': register each image to the previous or next image
                    (will be next for images before the template and previous for
                    images after the template)
            mode_transform (str, optional):
                The mode of geometric transformation.
                Can be 'translation', 'euclidean', 'affine', or 'homography'.
                See cv2.findTransformECC for more details.
            gaussFiltSize (int, optional):
                The size of the Gaussian filter, default is 11.
            mask_borders (tuple of int, int, optional):
                The border mask for the image, default is (0, 0, 0, 0).
            n_iter (int, optional):
                The number of iterations for cv2.findTransformECC, default is 1000.
            termination_eps (float, optional):
                The termination criteria for cv2.findTransformECC, default is 1e-9.
            auto_fix_gaussFilt_step (bool, int):
                Whether to automatically fix convergence issues by increasing the
                 gaussFiltSize.
                If False, then no automatic fixing is performed.
                If True, then the gaussFiltSize is increased by 2 until convergence.
                If int, then the gaussFiltSize is increased by this amount until
                 convergence.

        Returns:
            remapIdx_geo (np.ndarray): 
                An array of shape (N, H, W, 2) representing the remap field for N images.
        """
        ## Imports
        super().__init__()
        
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
        template: typing.Union[int, np.ndarray],
        ims_moving: typing.List[np.ndarray],
        remappingIdx_init: np.ndarray = None,
        template_method: str = 'sequential',
        mode_transform: str = 'createOptFlow_DeepFlow',
        kwargs_mode_transform: dict = None,
    ) -> np.ndarray:
        """
        Perform geometric registration of ims_moving to a template.
        Currently relies on cv2.findTransformECC.
        RH 2023

        Args:
            template (int or np.ndarray):
                If 'template_method' == 'image', then template is either
                 an image or an integer index or a float fractional index
                 of the image to use as the template.
                If 'template_method' == 'sequential', then template is the
                 integer index of the image to use as the template.
            ims_moving (list of np.ndarray):
                A list of images to be aligned.
            remappingIdx_init (np.ndarray, optional):
                An array of shape (N, H, W, 2) representing any initial
                 remap field to apply to the images in ims_moving.
                The output of this method will be added/composed with remappingIdx_init.
            template_method (str, optional):
                The method to use for template selection.
                'image': use the image specified by 'template'.
                'sequential': register each image to the previous or next image
                    (will be next for images before the template and previous for
                    images after the template)
            mode_transform (str, optional):
                The type of transformation to use for registration.
                Either 'createOptFlow_DeepFlow' or 'calcOpticalFlowFarneback'.
            kwargs_mode_transform (dict, optional):
                Keyword arguments to pass to the mode_transform function.
                See cv2 functions for more details.

        Returns:
            remapIdx_nonrigid (np.ndarray): 
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
        # Check if mode_transform is valid
        valid_mode_transforms = {'createOptFlow_DeepFlow', 'calcOpticalFlowFarneback'}
        assert mode_transform in valid_mode_transforms, f"mode_transform must be one of {valid_mode_transforms}"

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
                    'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN
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
        
    def transform_images_geometric(self, ims_moving, remappingIdx=None):
        remappingIdx = self.remappingIdx_geo if remappingIdx is None else remappingIdx
        print('Applying geometric registration warps to images...') if self._verbose else None
        self.ims_registered_geo = self._transform_images(ims_moving=ims_moving, remappingIdx=remappingIdx)
        return self.ims_registered_geo
    def transform_images_nonrigid(self, ims_moving, remappingIdx=None):
        remappingIdx = self.remappingIdx_nonrigid if remappingIdx is None else remappingIdx
        print('Applying nonrigid registration warps to images...') if self._verbose else None
        self.ims_registered_nonrigid = self._transform_images(ims_moving=ims_moving, remappingIdx=remappingIdx)
        return self.ims_registered_nonrigid
    
    def _transform_images(self, ims_moving, remappingIdx):
        ims_registered = []
        for ii, (im_moving, remapIdx) in enumerate(zip(ims_moving, remappingIdx)):
            im_registered = helpers.remap_images(
                images=im_moving,
                remappingIdx=remapIdx,
                backend='cv2',
                interpolation_method='linear',
                border_mode='constant',
                border_value=float(im_moving.mean()),
            )
            ims_registered.append(im_registered)
        return ims_registered
    

    def _compose_warps(self, warp_0, warps_to_add, warpMat_or_remapIdx='remapIdx'):
        """
        Compose a series of warps into a single warp.
        RH 2023
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
        ROIs, 
        remappingIdx=None,
        normalize=True,
    ):
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

    def get_ROIsAligned_maxIntensityProjection(self, H=None, W=None):
        """
        Returns the max intensity projection of the ROIs aligned to the template FOV.
        """
        if H is None:
            assert self._HW is not None, 'H and W must be provided if not already set.'
            H, W = self._HW
        return [rois.max(0).toarray().reshape(H, W) for rois in self.ROIs_aligned]
    
    def get_flowFields(self, remappingIdx=None):
        if remappingIdx is None:
            assert (self.remappingIdx_geo is not None) or (self.remappingIdx_nonrigid is not None), 'If remappingIdx is not provided, then geometric or nonrigid registration must be performed first.'
            remappingIdx = self.remappingIdx_nonrigid if self.remappingIdx_nonrigid is not None else self.remappingIdx_geo
        return [helpers.remappingIdx_to_flowField(remap) for remap in remappingIdx]
    
    def _fix_input_images(
        self,
        ims_moving: typing.List[np.ndarray],
        template: typing.Union[int, np.ndarray],
        template_method: str
    ) -> typing.Tuple[int, typing.List[np.ndarray]]:
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


class PhaseCorrelation_registration:
    """
    Class for performing rigid transformation using phase correlation.
    RH 2022
    """
    def __init__(
        self,
    ):
        self.mask = None

    def set_spectral_mask(
        self,
        freq_highPass=0.01,
        freq_lowPass=0.3,
        im_shape=(512, 512),
    ):
        self.mask = make_spectral_mask(
            freq_highPass=freq_highPass,
            freq_lowPass=freq_lowPass,
            im_shape=im_shape,
        )

    def register(
        self, 
        template, 
        ims_moving,
        template_method='sequential',
    ):
        """
        Register set of images using phase correlation.
        RH 2022
        
        Args:
            template (np.ndarray or int):
                Template image
            ims_moving (np.ndarray):
                Images to align to the template.
            template_method (str):
                The method used to register the images.
                Either 'image' or 'sequential'.
                If 'image':      template must be a single image.
                If 'sequential': template must be an integer corresponding 
                 to the index of the image to set as 'zero' offset.
        
        Returns:
            template (np.ndarray):
                Registered images
            shifts (np.ndarray):
                Pixel shift values (y, x).

        Attributes set:
            self.ims_registered (np.ndarray):
                Registered images
            self.shifts (np.ndarray):
                Pixel shift values (y, x).
            self.ccs (np.ndarray):
                Phase correlation coefficient images.
            self.ims_template_filt (np.ndarray):
                Template images filtered by the spectral mask.
            self.ims_moving_filt (np.ndarray):
                Moving images filtered by the spectral mask.
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


def phase_correlation(im_template, im_moving, mask_fft=None, return_filtered_images=False):
    """
    From BNPM
    Perform phase correlation on two images.
    RH 2022
    
    Args:
        im_template (np.ndarray):
            Template image
        im_moving (np.ndarray):
            Moving image
        mask_fft (np.ndarray):
            Mask for the FFT.
            If None, no mask is used.
    
    Returns:
        cc (np.ndarray):
            Phase correlation coefficient.
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


def convert_phaseCorrelationImage_to_shifts(cc_im):
    """
    Convert phase correlation image to pixel shift values.
    RH 2022

    Args:
        cc_im (np.ndarray):
            Phase correlation image.
            Middle of image is zero-shift.

    Returns:
        shifts (np.ndarray):
            Pixel shift values (y, x).
    """
    height, width = cc_im.shape
    shift_y_raw, shift_x_raw = np.unravel_index(cc_im.argmax(), cc_im.shape)
    return int(np.floor(height/2) - shift_y_raw) , int(np.ceil(width/2) - shift_x_raw)


def helper_shift(X, shift, fill_val=0):
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
def shift_along_axis(X, shift, fill_val=0, axis=0):
    return np.apply_along_axis(helper_shift, axis, np.array(X, dtype=X.dtype), shift, fill_val)


def make_spectral_mask(freq_highPass=0.01, freq_lowPass=0.3, im_shape=(512, 512)):
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


def clahe(im, grid_size=50, clipLimit=0, normalize=True):
    """
    Perform Contrast Limited Adaptive Histogram Equalization (CLAHE)
     on an image.
    RH 2022

    Args:
        im (np.ndarray):
            Input image
        grid_size (int):
            Grid size.
            See cv2.createCLAHE for more info.
        clipLimit (int):
            Clip limit.
            See cv2.createCLAHE for more info.
        normalize (bool):
            Whether to normalize the output image.
        
    Returns:
        im_out (np.ndarray):
            Output image
    """
    im_tu = (im / im.max())*(2**8) if normalize else im
    im_tu = (im_tu/10).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid_size, grid_size))
    im_c = clahe.apply(im_tu.astype(np.uint16))
    return im_c
