# import sparse
# import torch
# import spconv.pytorch as spconv
import scipy.sparse
import numpy as np
from tqdm import tqdm

from .. import helpers
    
class ROI_Blurrer:
    """
    Class for blurring ROIs.
    RH 2022
    """
    def __init__(
        self,
        frame_shape=(512, 512),
        kernel_halfWidth=2,
        plot_kernel=False,
        verbose=True,
    ):
        """
        Initialize the class.

        Args:
            frame_shape (tuple):
                The shape of the frame/FOV.
                frame_shape[0] * frame_shape[1]
                 must equal the length of a single flattened/
                 sparse spatialFootprint.
            kernel_halfWidth (int):
                The half-width of the cosine kernel to use
                 for convolutional blurring.
            plot_kernel (bool):
                Whether to plot an image of the kernel.
        """
        self._frame_shape = frame_shape
        self._verbose = verbose

        self._width = kernel_halfWidth * 2
        self._kernel_size = int((self._width//2)*2) - 1
        kernel_tmp = helpers.cosine_kernel_2D(
            center=(self._kernel_size//2, self._kernel_size//2), 
            image_size=(self._kernel_size, self._kernel_size),
            width=self._width
        )
        self.kernel = kernel_tmp / kernel_tmp.sum()

        print('Preparing the Toeplitz convolution matrix') if self._verbose else None
        self._conv = Toeplitz_convolution2d(
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
        spatialFootprints,
    ):
        # sf_cat = helpers.scipy_sparse_csr_with_length(scipy.sparse.vstack(spatialFootprints))
        # self.ROIs_blurred = scipy.sparse.vstack([
        #     self._conv(
        #         x=batch,
        #         batching=True,
        #         mode='same',
        #     ) for batch in tqdm(helpers.make_batches(sf_cat, batch_size=batch_size), total=int(np.ceil(sf_cat.shape[0]/batch_size)))])

        print('Performing convolution for blurring') if self._verbose else None
        self.ROIs_blurred = [
                self._conv(
                    x=sf,
                    batching=True,
                    mode='same',
                ) for sf in spatialFootprints
        ]
    

    def get_ROIsBlurred_maxIntensityProjection(self):
        """
        Returns the max intensity projection of the ROIs.
        """
        return [rois.max(0).toarray().reshape(self._frame_shape[0], self._frame_shape[1]) for rois in self.ROIs_blurred]



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



class Toeplitz_convolution2d:
    """
    Convolve a 2D array with a 2D kernel using the Toeplitz matrix 
     multiplication method.
    Allows for SPARSE 'x' inputs. 'k' should remain dense.
    Ideal when 'x' is very sparse (density<0.01), 'x' is small
     (shape <(1000,1000)), 'k' is small (shape <(100,100)), and
     the batch size is large (e.g. 1000+).
    Generally faster than scipy.signal.convolve2d when convolving mutliple
     arrays with the same kernel. Maintains low memory footprint by
     storing the toeplitz matrix as a sparse matrix.

    See: https://stackoverflow.com/a/51865516 and https://github.com/alisaaalehi/convolution_as_multiplication
     for a nice illustration.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.convolution_matrix.html 
     for 1D version.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html#scipy.linalg.matmul_toeplitz 
     for potential ways to make this implementation faster.

    Test with: tests.test_toeplitz_convolution2d()
    RH 2022
    """
    def __init__(
        self,
        x_shape,
        k,
        mode='same',
        dtype=None,
    ):
        """
        Initialize the convolution object.
        Makes the Toeplitz matrix and stores it.

        Args:
            x_shape (tuple):
                The shape of the 2D array to be convolved.
            k (np.ndarray):
                2D kernel to convolve with
            mode (str):
                'full', 'same' or 'valid'
                see scipy.signal.convolve2d for details
            dtype (np.dtype):
                The data type to use for the Toeplitz matrix.
                Ideally, this matches the data type of the input array.
                If None, then the data type of the kernel is used.
        """
        self.k = k = np.flipud(k.copy())
        self.mode = mode
        self.x_shape = x_shape
        self.dtype = k.dtype if dtype is None else dtype

        if mode == 'valid':
            assert x_shape[0] >= k.shape[0] and x_shape[1] >= k.shape[1], "x must be larger than k in both dimensions for mode='valid'"

        self.so = so = size_output_array = ( (k.shape[0] + x_shape[0] -1), (k.shape[1] + x_shape[1] -1))  ## 'size out' is the size of the output array

        ## make the toeplitz matrices
        t = toeplitz_matrices = [scipy.sparse.diags(
            diagonals=np.ones((k.shape[1], x_shape[1]), dtype=self.dtype) * k_i[::-1][:,None], 
            offsets=np.arange(-k.shape[1]+1, 1), 
            shape=(so[1], x_shape[1]),
            dtype=self.dtype,
        ) for k_i in k[::-1]]  ## make the toeplitz matrices for the rows of the kernel
        tc = toeplitz_concatenated = scipy.sparse.vstack(t + [scipy.sparse.dia_matrix((t[0].shape), dtype=self.dtype)]*(x_shape[0]-1))  ## add empty matrices to the bottom of the block due to padding, then concatenate

        ## make the double block toeplitz matrix
        self.dt = double_toeplitz = scipy.sparse.hstack([self._roll_sparse(
            x=tc, 
            shift=(ii>0)*ii*(so[1])  ## shift the blocks by the size of the output array
        ) for ii in range(x_shape[0])]).tocsr()
    
    def __call__(
        self,
        x,
        batching=True,
        mode=None,
    ):
        """
        Convolve the input array with the kernel.

        Args:
            x (np.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix):
                Input array(s) (i.e. image(s)) to convolve with the kernel
                If batching==False: Single 2D array to convolve with the kernel.
                    shape: (self.x_shape[0], self.x_shape[1])
                    type: np.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
                If batching==True: Multiple 2D arrays that have been flattened
                 into row vectors (with order='C').
                    shape: (n_arrays, self.x_shape[0]*self.x_shape[1])
                    type: np.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
            batching (bool):
                If False, x is a single 2D array.
                If True, x is a 2D array where each row is a flattened 2D array.
            mode (str):
                'full', 'same' or 'valid'
                see scipy.signal.convolve2d for details
                Overrides the mode set in __init__.

        Returns:
            out (np.ndarray or scipy.sparse.csr_matrix):
                If batching==True: Multiple convolved 2D arrays that have been flattened
                 into row vectors (with order='C').
                    shape: (n_arrays, height*width)
                    type: np.ndarray or scipy.sparse.csc_matrix
                If batching==False: Single convolved 2D array of shape (height, width)
        """
        # if batching:
        #     if x.shape[0] > 9999:
        #         print("RH WARNING: scipy.sparse.lil_matrix doesn't seem to work well with arrays with large numbers of rows. Consider breaking your job into smaller batches.")
        if mode is None:
            mode = self.mode  ## use the mode that was set in the init if not specified
        issparse = scipy.sparse.issparse(x)
        
        if batching:
            x_v = x.T  ## transpose into column vectors
        else:
            x_v = x.reshape(-1, 1)  ## reshape 2D array into a column vector
        
        if issparse:
            x_v = x_v.tocsc()
        
        out_v = self.dt @ x_v  ## if sparse, then 'out_v' will be a csc matrix
            
        ## crop the output to the correct size
        if mode == 'full':
            p_t = 0
            p_b = self.so[0]+1
            p_l = 0
            p_r = self.so[1]+1
        if mode == 'same':
            p_t = (self.k.shape[0]-1)//2
            p_b = -(self.k.shape[0]-1)//2
            p_l = (self.k.shape[1]-1)//2
            p_r = -(self.k.shape[1]-1)//2

            p_b = self.x_shape[0]+1 if p_b==0 else p_b
            p_r = self.x_shape[1]+1 if p_r==0 else p_r
        if mode == 'valid':
            p_t = (self.k.shape[0]-1)
            p_b = -(self.k.shape[0]-1)
            p_l = (self.k.shape[1]-1)
            p_r = -(self.k.shape[1]-1)

            p_b = self.x_shape[0]+1 if p_b==0 else p_b
            p_r = self.x_shape[1]+1 if p_r==0 else p_r
        
        if batching:
            idx_crop = np.zeros((self.so), dtype=np.bool8)
            idx_crop[p_t:p_b, p_l:p_r] = True
            idx_crop = idx_crop.reshape(-1)
            out = out_v[idx_crop,:].T
        else:
            if issparse:
                out = out_v.reshape((self.so)).tocsc()[p_t:p_b, p_l:p_r]
            else:
                out = out_v.reshape((self.so))[p_t:p_b, p_l:p_r]  ## reshape back into 2D array and crop
        return out
    
    def _roll_sparse(
        self,
        x,
        shift,
    ):
        """
        Roll columns of a sparse matrix.
        """
        out = x.copy()
        out.row += shift
        return out