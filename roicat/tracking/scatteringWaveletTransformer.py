import gc
from typing import Any, Dict, Tuple

import torch
import numpy as np
from tqdm import tqdm

from .. import helpers, util

class SWT(util.ROICaT_Module):
    """
    Performs scattering wavelet transform using the kymatio library.
    RH 2022

    Args:
        kwargs_Scattering2D (Dict[str, Any]):
            The keyword arguments to pass to the Scattering2D class.
            (Default is ``{'J': 2, 'L': 8}``)
        image_shape (Tuple[int, int]):
            The shape of the images to be transformed. 
            (Default is ``(36,36)``)
        device (str):
            The device to use for the transformation. 
            (Default is ``'cpu'``)
        verbose (bool):
            If ``True``, print statements will be outputted. 
            (Default is ``True``)
    
    Example:
        .. highlight:: python
        .. code-block:: python

            swt = SWT(kwargs_Scattering2D={'J': 2, 'L': 8}, image_shape=(36,36), device='cpu', verbose=True)
            transformed_images = swt.transform(ROI_images, batch_size=100)
    """
    def __init__(
        self, 
        kwargs_Scattering2D: Dict[str, Any] = {'J': 2, 'L': 8}, 
        image_shape: Tuple[int, int] = (36,36), 
        device: str = 'cpu',
        verbose: bool = True,
    ):
        """
        Initializes the SWT with the given settings.
        """
        ## Imports
        super().__init__()

        ## Store parameter (but not data) args as attributes
        self.params['__init__'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'kwargs_Scattering2D',
                'image_shape',
                'device',
                'verbose',
            ],
        )

        from kymatio.torch import Scattering2D

        self._verbose = verbose
        self._device = device
        self.swt = Scattering2D(shape=image_shape, **kwargs_Scattering2D)
        self.swt = util.Model_SWT(self.swt)
        self.swt.to(device)
        print('SWT initialized') if self._verbose else None

    def transform(self, ROI_images: np.ndarray, batch_size: int = 100) -> np.ndarray:
        """
        Transforms the ROI images.

        Args:
            ROI_images (np.ndarray):
                The ROI images to transform. 
                One should probably concatenate ROI images across sessions for passing through here. 
                *(n_ROIs, height, width)*
            batch_size (int):
                The batch size to use for the transformation. 
                (Default is *100*)

        Returns:
            (np.ndarray):
                latents (np.ndarray):
                    The transformed ROI images. *(n_ROIs, latent_size)*
        """
        ## Store parameter (but not data) args as attributes
        self.params['transform'] = self._locals_to_params(
            locals_dict=locals(),
            keys=['batch_size',],)

        print('Starting: SWT transform on ROIs') if self._verbose else None
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
