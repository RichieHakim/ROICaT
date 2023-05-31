import gc

import torch
import numpy as np
from tqdm import tqdm

from .. import helpers, util

class SWT(util.ROICaT_Module):
    """
    Class for performing scattering wavelet transform
     using the kymatio library.
    """
    def __init__(
        self, 
        kwargs_Scattering2D={'J': 2, 'L': 8}, 
        image_shape=(36,36), 
        device='cpu',
        verbose=True,
    ):
        """
        Initialize the class.
        
        Args:
            kwargs_Scattering2D (dict):
                The keyword arguments to pass to the Scattering2D class.
                See the documentation for the kymatio's 
                 Scattering2D class for details.
            image_shape (tuple):
                The shape of the images to be transformed.
            device (str):
                The device to use for the transformation.
        """
        ## Imports
        super().__init__()

        from kymatio.torch import Scattering2D

        self._verbose = verbose
        self._device = device
        self.swt = Scattering2D(shape=image_shape, **kwargs_Scattering2D).to(device)
        print('SWT initialized') if self._verbose else None

    def transform(self, ROI_images, batch_size=100):
        """
        Transform the ROI images.

        Args:
            ROI_images (np.ndarray):
                The ROI images to transform.
                shape: (n_ROIs, height, width)
                One should probably concatenate ROI images
                 across session for passing through here.

        Returns:
            latents (np.ndarray):
                The transformed ROI images.
                shape: (n_ROIs, latent_size)
        """
        print('Starting: SWT transform on ROIs') if self._verbose else None
        # sfs = torch.as_tensor(np.ascontiguousarray(ROI_images[None,...]), device=self._device, dtype=torch.float32)
        # self.latents = self.swt(sfs[None,...]).squeeze().cpu()
        def helper_swt(ims_batch):
            sfs = torch.as_tensor(np.ascontiguousarray(ims_batch[None,...]), device=self._device, dtype=torch.float32)
            out = self.swt(sfs[None,...]).squeeze().cpu()
            if out.ndim == 3:  ## if there is only one ROI in the batch, append a dimension to the front
                out = out[None,...]
            return out
        self.latents = torch.cat([helper_swt(ims_batch) for ims_batch in tqdm(helpers.make_batches(ROI_images, batch_size=batch_size), total=ROI_images.shape[0] / batch_size, mininterval=5)], dim=0)
        self.latents = self.latents.reshape(self.latents.shape[0], -1)
        print('Completed: SWT transform on ROIs') if self._verbose else None

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

        return self.latents
