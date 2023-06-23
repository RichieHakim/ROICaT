from typing import Union, List, Tuple, Optional

import scipy.sparse
import numpy as np
from tqdm import tqdm

from .. import helpers, util
    
class ROI_Blurrer(util.ROICaT_Module):
    """
    Blurs the Region of Interest (ROI).
    RH 2022

    Args:
        frame_shape (Tuple[int, int]):
            The shape of the frame/Field Of View (FOV). Product of
            ``frame_shape[0]`` and ``frame_shape[1]`` must equal the length of a
            single flattened/sparse spatialFootprint. (Default is *(512, 512)*)
        kernel_halfWidth (int):
            The half-width of the cosine kernel to use for convolutional
            blurring. (Default is *2*)
        plot_kernel (bool):
            Whether to plot an image of the kernel. (Default is ``False``)
        verbose (bool):
            Whether to print the convolutional blurring operation progress.
            (Default is ``True``)

    Attributes:
        frame_shape (Tuple[int, int]):
            The shape of the frame/Field Of View (FOV). Product of
            ``frame_shape[0]`` and ``frame_shape[1]`` must equal the length of a
            single flattened/sparse spatialFootprint.
        kernel_halfWidth (int):
            The half-width of the cosine kernel to use for convolutional
            blurring.
        plot_kernel (bool):
            Whether to plot an image of the kernel.
        verbose (bool):
            Whether to print the convolutional blurring operation progress.
    """
    def __init__(
        self, 
        frame_shape: Tuple[int, int] = (512, 512),
        kernel_halfWidth: int = 2,
        plot_kernel: bool = False,
        verbose: bool = True,
    ):
        """
        Initializes the ROI_Blurrer with the given frame shape, kernel half-width, 
        plot kernel and verbosity setting.
        """
        ## Imports
        super().__init__()

        self._frame_shape = frame_shape
        self._verbose = verbose

        self._width = kernel_halfWidth * 2
        self._kernel_size = max(int((self._width//2)*2) - 1, 1)
        kernel_tmp = helpers.cosine_kernel_2D(
            center=(self._kernel_size//2, self._kernel_size//2), 
            image_size=(self._kernel_size, self._kernel_size),
            width=self._width
        )
        self.kernel = kernel_tmp / kernel_tmp.sum()

        print('Preparing the Toeplitz convolution matrix') if self._verbose else None
        self._conv = helpers.Toeplitz_convolution2d(
            x_shape=self._frame_shape,
            k=self.kernel,
            mode='same',
            dtype=np.float32,
        )

        if plot_kernel:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(self.kernel)

    def blur_ROIs(
        self,
        spatialFootprints: List[object],
    ) -> List[object]:
        """
        Blurs the Region of Interest (ROI).

        Args:
            spatialFootprints (List[object]): 
                A list of sparse matrices corresponding to spatial footprints from each session.

        Returns:
            (List[object]): 
                ROIs_blurred (List[object]):
                    A list of blurred ROI spatial footprints.
        """
        print('Performing convolution for blurring') if self._verbose else None
        if self._width == 0:
            self.ROIs_blurred = spatialFootprints
        else:
            self.ROIs_blurred = [
                    self._conv(
                        x=sf,
                        batching=True,
                        mode='same',
                    ) for sf in spatialFootprints
            ]
        return self.ROIs_blurred
    
    def get_ROIsBlurred_maxIntensityProjection(self) -> List[object]:
        """
        Calculates the maximum intensity projection of the ROIs.

        Returns:
            (List[object]): 
                ims (List[object]):
                    The maximum intensity projection of the ROIs.
        """
        ims = [(rois.multiply(rois.max(1).power(-1))).max(0).toarray().reshape(self._frame_shape[0], self._frame_shape[1]) for rois in self.ROIs_blurred]
        return ims



# class ROI_Blurrer:
#     """
#     Class for blurring ROIs.
#     Uses the sp_conv library for fast sparse convolutions.
#      Repo here: https://github.com/traveller59/spconv
#     RH 2022
#     """
#     def __init__(
#         self,
#         frame_shape=(512, 512),
#         kernel_halfWidth=2,
#         device='cpu',
#         plot_kernel=False,
#     ):
#         """
#         Initialize the class.

#         Args:
#             frame_shape (tuple):
#                 The shape of the frame/FOV.
#                 frame_shape[0] * frame_shape[1]
#                  must equal the length of a single flattened/
#                  sparse spatialFootprint.
#             kernel_halfWidth (int):
#                 The half-width of the cosine kernel to use
#                  for convolutional blurring.
#             device (str):
#                 The device to use for the convolution.
#             plot_kernel (bool):
#                 Whether to plot an image of the kernel.
#         """
#         self._frame_shape = frame_shape
#         self._device = device

#         self._width = kernel_halfWidth * 2
#         self._kernel_size = int((self._width//2)*2) + 3
#         kernel_tmp = helpers.cosine_kernel_2D(
#             center=(self._kernel_size//2, self._kernel_size//2), 
#             image_size=(self._kernel_size, self._kernel_size),
#             width=self._width
#         )
#         self.kernel = kernel_tmp / kernel_tmp.sum()

#         ## prepare kernel
#         kernel_prep = torch.as_tensor(
#             self.kernel[:,:,None,None], 
#             dtype=torch.float32,
#             device=device
#         ).contiguous()
        
#         ## prepare convolution
#         self._conv = spconv.SparseConv2d(
#             in_channels=1, 
#             out_channels=1,
#             kernel_size=self.kernel.shape, 
#             stride=1, 
#             padding=self.kernel.shape[0]//2, 
#             dilation=1, 
#             groups=1, 
#             bias=False
#         )
        
#         self._conv.weight = torch.nn.Parameter(data=kernel_prep, requires_grad=False)

#         if plot_kernel:
#             import matplotlib.pyplot as plt
#             plt.figure()
#             plt.imshow(self.kernel)


#     def _sparse_conv2D(
#         self,
#         sf_sparseCOO, 
#     ):
#         """
#         Method to perform a 2D convolution on a sparse matrix.

#         Args:
#             sf_sparseCOO (sparse.COO):
#                 The sparse matrix to convolve.
#                 shape: (num_ROIs, frame_shape[0], frame_shape[1])
#         """
#         images_spconv = pydata_sparse_to_spconv(
#             sf_sparseCOO,
#             device=self._device
#         )

#         images_conv = self._conv(images_spconv)
#         return sparse_convert_spconv_to_scipy(images_conv)

#     def blur_ROIs(
#         self, 
#         spatialFootprints, 
#         batch_size=None, 
#         num_batches=100, 
#     ):
#         """
#         Method to blur ROIs.

#         Args:
#             spatialFootprints (list of scipy.sparse.csr_matrix):
#                 The spatialFootprints to blur.
#                 shape of each element:
#                  (num_ROIs, frame_shape[0] * frame_shape[1])
#             batch_size (int):
#                 The batch size to use for blurring.
#                 if None, then will use num_batches to determine size.
#             num_batches (int):
#                 The number of batches to use for blurring.
#         """
#         sf_coo = [sparse.as_coo(sf).reshape((sf.shape[0], self._frame_shape[0], self._frame_shape[1])) for sf in spatialFootprints]
        
#         self.ROIs_blurred = [scipy.sparse.vstack([self._sparse_conv2D(
#             sf_sparseCOO=batch, 
#         ) for batch in helpers.make_batches(sf, batch_size=batch_size, num_batches=num_batches)]) for sf in sf_coo]

#         return self.ROIs_blurred


    # def get_ROIsBlurred_maxIntensityProjection(self):
    #     """
    #     Returns the max intensity projection of the ROIs.
    #     """
    #     return [rois.max(0).toarray().reshape(self._frame_shape[0], self._frame_shape[1]) for rois in self.ROIs_blurred]


# def pydata_sparse_to_spconv(sp_array, device='cpu'):
#     coo = sparse.COO(sp_array)
#     idx_raw = torch.as_tensor(coo.coords.T, dtype=torch.int32, device=device).contiguous()
#     spconv_array = spconv.SparseConvTensor(
#         features=torch.as_tensor(coo.reshape((-1)).T.data, dtype=torch.float32, device=device)[:,None].contiguous(),
#         indices=idx_raw,
#         spatial_shape=coo.shape[1:], 
#         batch_size=coo.shape[0]
#     )
#     return spconv_array

# def sparse_convert_spconv_to_scipy(sp_arr):
#     coo = sparse.COO(
#         coords=sp_arr.indices.T.to('cpu'),
#         data=sp_arr.features.squeeze().to('cpu'),
#         shape=[sp_arr.batch_size] + sp_arr.spatial_shape
#     )
#     return coo.reshape((coo.shape[0], -1)).to_scipy_sparse().tocsr()

