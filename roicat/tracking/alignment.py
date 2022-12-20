import copy

import numpy as np
import cv2
import scipy.sparse
from tqdm import tqdm
import matplotlib.pyplot as plt

class Alinger:
    """
    A class for registering ROIs to a template FOV.
    Currently relies on available OpenCV methods for 
     non-rigid registration.
    RH 2022
    """
    def __init__(
        self,
        method='createOptFlow_DeepFlow',
        kwargs_method=None,
        verbose=True,
    ):
        """
        Initialize the class.

        Args:
            method (str):
                The method to use for optical flow calculation.
                The following are currently supported:
                    'calcOpticalFlowFarneback',
                    'createOptFlow_DeepFlow',
            kwargs_method (dict):
                The keyword arguments to pass to the method.
                See the documentation for the method for the
                 required arguments.
                If None, hard-coded defaults will be used.
            verbose (bool):
                Whether to print progress updates.
        """
        self._verbose = verbose
        self._method = method
        self._kwargs_method = kwargs_method

    def register_ROIs(
        self,
        template,
        FOVs,
        ROIs,
        template_method='image',
        shifts=None,
        return_sparse=True,
        normalize=True
    ):
        """
        Perform non-rigid registration of ROIs to a template FOV.
        Currently relies on available OpenCV methods for
         non-rigid registration.
        RH 2022

        Args:
            template (numpy.ndarray):
                The template FOV to align to.
            FOVs (list of numpy.ndarray):
                The FOVs to register.
            ROIs (list of numpy.ndarray):
                The ROIs to register.
            template_method (str):
                The method used to register the images.
                Either 'image' or 'sequential'.
                If 'image':      template must be a single image.
                If 'sequential': template must be an integer corresponding 
                 to the index of the image to set as 'zero' offset.
            shifts (list of numpy.ndarray):
                The shifts to apply to the ROIs.
                If None, no shifts will be applied.
                The shifts describe the relative shift between the 
                 FOVs and the ROIs. This will be non-zero if the 
                 input FOVs have been shifted using the phase-
                 correlation shifter. 
            return_sparse (bool):
                If True, return sparse spatial footprints.
                If False, return dense spatial footprint FOVs.
            normalize (bool):
                If True, normalize the spatial footprints to have
                 a sum of 1.
                If False, do not normalize.
        
        Returns:
            self.ROIs_aligned (numpy.ndarray or scipy.sparse.csr_matrix):
                The aligned ROIs.
            self.FOVs_aligned (numpy.ndarray):
                The aligned FOVs.
            self.flows (list of numpy.ndarray):
                The optical flows between the FOVs and the ROIs.
                DOES NOT INCLUDE THE INPUT SHIFTS.
        """
    
        dims = FOVs[0].shape
        x_grid, y_grid = np.meshgrid(np.arange(0., dims[1]).astype(np.float32), np.arange(0., dims[0]).astype(np.float32))

        template_norm = np.uint8(template * (template > 0) * (1/template.max()) * 255) if template_method == 'image' else None
        FOVs_norm    = [np.uint8(FOVs[ii] * (FOVs[ii] > 0) * (1/FOVs[ii].max()) * 255) for ii in range(len(FOVs))]

        def safe_ROI_remap(img_ROI, x_remap, y_remap):
            img_ROI_remap = cv2.remap(
                img_ROI.astype(np.float32),
                x_remap,
                y_remap, 
                cv2.INTER_LINEAR
            )
            if img_ROI_remap.sum() == 0:
                img_ROI_remap = img_ROI
            return img_ROI_remap

        if shifts is None:
            shifts = [(0,0)] * len(FOVs)
        
        print(f'Finding optical flow fields between FOVs using {template_method} template...') if self._verbose else None
        self.ROIs_aligned, self.FOVs_aligned, self.flows = [], [], []
        flow_old = np.zeros((dims[0], dims[1], 2))
        flows_all = []
        for ii in tqdm(range(len(FOVs)), mininterval=60):
            if template_method == 'sequential':
                template_norm = FOVs_norm[ii-1] if ii > 0 else FOVs_norm[ii]

            if self._method == 'calcOpticalFlowFarneback':
                if self._kwargs_method is None:
                    self._kwargs_method = {
                        'pyr_scale': 0.3, 
                        'levels': 3,
                        'winsize': 128, 
                        'iterations': 7,
                        'poly_n': 7, 
                        'poly_sigma': 1.5,
                        'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN
                    }
                flow_tmp = cv2.calcOpticalFlowFarneback(
                    prev=template_norm,
                    next=FOVs_norm[ii], 
                    flow=None, 
                    **self._kwargs_method,
                )
        
            elif self._method == 'createOptFlow_DeepFlow':
                flow_tmp = cv2.optflow.createOptFlow_DeepFlow().calc(
                    template_norm,
                    FOVs_norm[ii],
                    None
                )

            flow = flow_tmp + flow_old if template_method == 'sequential' else flow_tmp
            flow_old = flow.copy() if template_method == 'sequential' else flow_old
            flows_all.append(flow)
        
        self.flows = [f - flows_all[template] for f in flows_all] if template_method == 'sequential' else flows_all

        print('Registering ROIs...') if self._verbose else None
        for ii, flow in tqdm(enumerate(self.flows), total=len(self.flows), mininterval=60):
            x_remap = (flow[:, :, 0] + x_grid).astype(np.float32)
            y_remap = (flow[:, :, 1] + y_grid).astype(np.float32)
            
            if type(ROIs[ii]) is scipy.sparse.coo_matrix:
                ROIs[ii] = ROIs[ii].tocsr()
            
            rois_toUse = ROIs[ii].toarray().astype(np.float32).reshape(ROIs[ii].shape[0], FOVs[ii].shape[0], FOVs[ii].shape[1]) if type(ROIs[ii]) is scipy.sparse.csr_matrix else ROIs[ii].astype(np.float32)
            
            
            ROI_aligned = np.stack([safe_ROI_remap(
                img, 
                x_remap - shifts[ii][1], 
                y_remap - shifts[ii][0],
            ) for img in rois_toUse], axis=0)
    #         ROI_aligned = np.stack([img.astype(np.float32) for img in ROIs[ii]], axis=0)
            FOV_aligned = cv2.remap(FOVs_norm[ii], x_remap, y_remap, cv2.INTER_NEAREST)

            if normalize:
                ROI_aligned = ROI_aligned / np.sum(ROI_aligned, axis=(1,2), keepdims=True)
            
            if return_sparse:
                self.ROIs_aligned.append(scipy.sparse.csr_matrix(ROI_aligned.reshape(ROI_aligned.shape[0], -1)))
                self.FOVs_aligned.append(FOV_aligned)
            else:
                self.ROIs_aligned.append(ROI_aligned)
                self.FOVs_aligned.append(FOV_aligned)

            ## remove NaNs from ROIs
            for ii in range(len(self.ROIs_aligned)):
                self.ROIs_aligned[ii].data[np.isnan(self.ROIs_aligned[ii].data)] = 0
                
        return self.ROIs_aligned, self.FOVs_aligned, self.flows

    def get_ROIsAligned_maxIntensityProjection(self):
        """
        Returns the max intensity projection of the ROIs aligned to the template FOV.
        """
        return [rois.max(0).toarray().reshape(self.FOVs_aligned[0].shape[0], self.FOVs_aligned[0].shape[1]) for rois in self.ROIs_aligned]


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
        Register two images using phase correlation.
        RH 2022
        
        Args:
            im_template (np.ndarray):
                Template image
            ims_moving (np.ndarray):
                Images to align to the template.
            template_method (str):
                The method used to register the images.
                Either 'image' or 'sequential'.
                If 'image':      ims_registered must be a single image.
                If 'sequential': ims_registered must be an integer corresponding 
                 to the index of the image to set as 'zero' offset.
        
        Returns:
            ims_registered (np.ndarray):
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
    X_shift = np.empty_like(X)
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
    return np.apply_along_axis(helper_shift, axis, X, shift, fill_val)


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
    im_tu = (im / im.max())*(2**16) if normalize else im
    im_tu = im_tu/10
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid_size, grid_size))
    im_c = clahe.apply(im_tu.astype(np.uint16))
    return im_c
