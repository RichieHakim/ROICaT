import numpy as np
import cv2
import scipy.sparse
from tqdm import tqdm

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
        kwargs_method=None
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
        """
        self._method = method
        self._kwargs_method = kwargs_method

    def register_ROIs(
        self,
        templateFOV,
        FOVs,
        ROIs,
        return_sparse=True,
        normalize=True
    ):
    
        dims = templateFOV.shape
        x_grid, y_grid = np.meshgrid(np.arange(0., dims[1]).astype(np.float32), np.arange(0., dims[0]).astype(np.float32))

        template_norm = np.uint8(templateFOV * (templateFOV > 0) * (1/templateFOV.max()) * 255)
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

        
        self.ROIs_aligned, self.FOVs_aligned, self.flows = [], [], []
        for ii in tqdm(range(len(FOVs)), mininterval=60):

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
                flow = cv2.calcOpticalFlowFarneback(
                    prev=template_norm,
                    next=FOVs_norm[ii], 
                    flow=None, 
                    **self._kwargs_method,
                )
        
            elif self._method == 'createOptFlow_DeepFlow':
                flow = cv2.optflow.createOptFlow_DeepFlow().calc(
                    template_norm,
                    FOVs_norm[ii],
                    None
                )
                
            x_remap = (flow[:, :, 0] + x_grid).astype(np.float32)
            y_remap = (flow[:, :, 1] + y_grid).astype(np.float32)

            rois_toUse = ROIs[ii].toarray().astype(np.float32).reshape(ROIs[ii].shape[0], FOVs[ii].shape[0], FOVs[ii].shape[1]) if type(ROIs[ii]) is scipy.sparse.csr_matrix else ROIs[ii].astype(np.float32)

            ROI_aligned = np.stack([safe_ROI_remap(img, x_remap, y_remap) for img in rois_toUse], axis=0)
    #         ROI_aligned = np.stack([img.astype(np.float32) for img in ROIs[ii]], axis=0)
            FOV_aligned = cv2.remap(FOVs_norm[ii], x_remap, y_remap, cv2.INTER_NEAREST)

            if normalize:
                ROI_aligned = ROI_aligned / np.sum(ROI_aligned, axis=(1,2), keepdims=True)
            
            if return_sparse:
                self.ROIs_aligned.append(scipy.sparse.csr_matrix(ROI_aligned.reshape(ROI_aligned.shape[0], -1)))
                self.FOVs_aligned.append(FOV_aligned)
                self.flows.append(flow)
            else:
                self.ROIs_aligned.append(ROI_aligned)
                self.FOVs_aligned.append(FOV_aligned)
                self.flows.append(flow)

            ## remove NaNs from ROIs
            for ii in range(len(self.ROIs_aligned)):
                self.ROIs_aligned[ii].data[np.isnan(self.ROIs_aligned[ii].data)] = 0
                
        return self.ROIs_aligned, self.FOVs_aligned, self.flows

    def get_ROIsAligned_maxIntensityProjection(self):
        """
        Returns the max intensity projection of the ROIs aligned to the template FOV.
        """
        return [rois.max(0).toarray().reshape(self.FOVs_aligned[0].shape[0], self.FOVs_aligned[0].shape[1]) for rois in self.ROIs_aligned]