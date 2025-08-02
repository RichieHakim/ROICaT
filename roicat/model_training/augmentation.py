import torch
from torch.nn import Module
import torchvision.transforms
import time


RandomHorizontalFlip = torchvision.transforms.RandomHorizontalFlip
def RandomAffine(**kwargs):
    if 'interpolation' in kwargs:
        kwargs['interpolation'] = torchvision.transforms.InterpolationMode(kwargs['interpolation'])
    return torchvision.transforms.RandomAffine(**kwargs)


class AddGaussianNoise(Module):
    """
    Adds Gaussian noise to the input tensor.
    RH 2021
    """
    def __init__(self, mean=0., std=1., level_bounds=(0., 1.), prob=1):
        """
        Initializes the class.
        Args:
            mean (float): 
                The mean of the Gaussian noise.
            std (float):
                The standard deviation of the Gaussian 
                 noise.
            level_bounds (tuple):
                The lower and upper bound of how much
                 noise to add.
            prob (float):
                The probability of adding noise at all.
        """
        super().__init__()

        self.std = std
        self.mean = mean

        self.prob = prob
        
        self.level_bounds = level_bounds
        self.level_range = level_bounds[1] - level_bounds[0]

    def forward(self, tensor):
        if torch.rand(1) <= self.prob:
            level = torch.rand(1, device=tensor.device) * self.level_range + self.level_bounds[0]
            return (1-level)*tensor + level*(tensor + torch.randn(tensor.shape, device=tensor.device) * self.std + self.mean)
        else:
            return tensor
    def __repr__(self):
        return f"AddGaussianNoise(mean={self.mean}, std={self.std}, level_bounds={self.level_bounds}, prob={self.prob})"

class AddPoissonNoise(Module):
    """
    Adds Poisson noise to the input tensor.
    RH 2021
    """
    def __init__(self, scaler_bounds=(0.1,1.), prob=1, base=10, scaling='log'):
        """
        Initializes the class.
        Args:
            lam (float): 
                The lambda parameter of the Poisson noise.
            scaler_bounds (tuple):
                The bounds of how much to multiply the image by
                 prior to adding the Poisson noise.
            prob (float):
                The probability of adding noise at all.
            base (float):
                The base of the logarithm used if scaling
                 is set to 'log'. Larger base means more
                 noise (higher probability of scaler being
                 close to scaler_bounds[0]).
            scaling (str):
                'linear' or 'log'
        """
        super().__init__()

        self.prob = prob
        self.bounds = scaler_bounds
        self.range = scaler_bounds[1] - scaler_bounds[0]
        self.base = base
        self.scaling = scaling

    def forward(self, tensor):
        check = tensor.min()
        if check < 0:
            print(f'RH: check= {check}')
        if torch.rand(1) <= self.prob:
            if self.scaling == 'linear':
                scaler = torch.rand(1, device=tensor.device) * self.range + self.bounds[0]
                return torch.poisson(tensor * scaler) / scaler
            else:
                scaler = (((self.base**torch.rand(1, device=tensor.device) - 1)/(self.base-1)) * self.range) + self.bounds[0]
                return torch.poisson(tensor * scaler) / scaler
        else:
            return tensor
    
    def __repr__(self):
        return f"AddPoissonNoise(level_bounds={self.bounds}, prob={self.prob})"

class ScaleDynamicRange(Module):
    """
    Min-max scaling of the input tensor.
    RH 2021
    """
    def __init__(self, scaler_bounds=(0,1), epsilon=1e-9):
        """
        Initializes the class.
        Args:
            scaler_bounds (tuple):
                The bounds of how much to multiply the image by
                 prior to adding the Poisson noise.
             epsilon (float):
                 Value to add to the denominator when normalizing.
        """
        super().__init__()

        self.bounds = scaler_bounds
        self.range = scaler_bounds[1] - scaler_bounds[0]
        
        self.epsilon = epsilon
    
    def forward(self, tensor):
        tensor_minSub = tensor - tensor.min()
        return tensor_minSub * (self.range / (tensor_minSub.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]+self.epsilon))
    def __repr__(self):
        return f"ScaleDynamicRange(scaler_bounds={self.bounds})"

class TileChannels(Module):
    """
    Expand dimension dim in X_in and tile to be N channels.
    RH 2021
    """
    def __init__(self, dim=0, n_channels=3):
        """
        Initializes the class.
        Args:
            dim (int):
                The dimension to tile.
            n_channels (int):
                The number of channels to tile to.
        """
        super().__init__()
        self.dim = dim
        self.n_channels = n_channels

    def forward(self, tensor):
        dims = [1]*len(tensor.shape)
        dims[self.dim] = self.n_channels
        return torch.tile(tensor, dims)
    def __repr__(self):
        return f"TileChannels(dim={self.dim})"

class Normalize(Module):
    """
    Normalizes the input tensor by setting the 
     mean and standard deviation of each channel.
    RH 2021
    """
    def __init__(self, means=0, stds=1):
        """
        Initializes the class.
        Args:
            mean (float):
                Mean to set.
            std (float):
                Standard deviation to set.
        """
        super().__init__()
        self.means = torch.as_tensor(means)[:,None,None]
        self.stds = torch.as_tensor(stds)[:,None,None]
    def forward(self, tensor):
        tensor_means = tensor.mean(dim=(1,2), keepdim=True)
        tensor_stds = tensor.std(dim=(1,2), keepdim=True)
        tensor_z = (tensor - tensor_means) / tensor_stds
        return (tensor_z * self.stds) + self.means

class WarpPoints(Module):
    """
    Warps the input tensor at the given points by the given deltas.
    RH 2021 / JZ 2021
    """
    
    def __init__(self,  r=[0, 2],
                        cx=[-0.5, 0.5],
                        cy=[-0.5, 0.5], 
                        dx=[-0.3, 0.3], 
                        dy=[-0.3, 0.3], 
                        n_warps=1,
                        prob=0.5,
                        img_size_in=[36, 36],
                        img_size_out=[36, 36]):
        """
        Initializes the class.

        Args:
            r (list):
                The range of the radius.
            cx (list):
                The range of the center x.
            cy (list):  
                The range of the center y.
            dx (list):
                The range of the delta x.
            dy (list):
                The range of the delta y.
            n_warps (int):
                The number of warps to apply.
            prob (float):
                The probability of adding noise at all.
            img_size_in (list):
                The size of the input image.
            img_size_out (list):
                The size of the output image.
        """
        
        super().__init__()

        self.r = r
        self.cx = cx
        self.cy = cy
        self.dx = dx
        self.dy = dy
        self.n_warps = n_warps

        self.prob = prob

        self.img_size_in = img_size_in
        self.img_size_out = img_size_out

        self.r_range = r[1] - r[0]
        self.cx_range = cx[1] - cx[0]
        self.cy_range = cy[1] - cy[0]
        self.dx_range = dx[1] - dx[0]
        self.dy_range = dy[1] - dy[0]
        
        #### , indexing='ij' within torch.meshgrid call to remove warning
        
        self.meshgrid_in =  torch.tile(torch.stack(torch.meshgrid(torch.linspace(-1, 1, self.img_size_in[0]),  torch.linspace(-1, 1, self.img_size_in[1]), indexing='ij'), dim=0)[...,None], (1,1,1, n_warps))
        self.meshgrid_out = torch.tile(torch.stack(torch.meshgrid(torch.linspace(-1, 1, self.img_size_out[0]), torch.linspace(-1, 1, self.img_size_out[1]), indexing='ij'), dim=0)[...,None], (1,1,1, n_warps))
        

    def gaus2D(self, x, y, sigma):
        return torch.exp(-((torch.square(self.meshgrid_out[0] - x[None,None,:]) + torch.square(self.meshgrid_out[1] - y[None,None,:]))/(2*torch.square(sigma[None,None,:]))))        

    def forward(self, tensor):
        if tensor.ndim == 3:
            tensor = tensor[None, ...]
            flag_batched = False
        elif tensor.ndim == 4:
            flag_batched = True
        else:
            raise ValueError("Input tensor must be 3 or 4 dimensional.")
        
        ## move meshgrid to the device of the input tensor
        if self.meshgrid_in.device != tensor.device:
            self.meshgrid_in = self.meshgrid_in.to(tensor.device)
        if self.meshgrid_out.device != tensor.device:
            self.meshgrid_out = self.meshgrid_out.to(tensor.device)

        if torch.rand(1) <= self.prob:
            rands = torch.rand(5, self.n_warps, device=tensor.device)  # shape: (5, n_warps)
            cx = rands[0,:] * (self.cx_range) + self.cx[0]
            cy = rands[1,:] * (self.cy_range) + self.cy[0]
            dx = rands[2,:] * (self.dx_range) + self.dx[0]
            dy = rands[3,:] * (self.dy_range) + self.dy[0]
            r =  rands[4,:] * (self.r_range)  + self.r[0]
            im_gaus = self.gaus2D(x=cx, y=cy, sigma=r) # shape: (img_size_x, img_size_y, n_warps)
            im_disp = im_gaus[None,...] * torch.stack([dx, dy], dim=0).reshape(2, 1, 1, self.n_warps) # shape: (2(dx,dy), img_size_x, img_size_y, n_warps)
            im_disp_composite = torch.sum(im_disp, dim=3, keepdim=True) # shape: (2(dx,dy), img_size_x, img_size_y)
            im_newPos = self.meshgrid_out[...,0:1] + im_disp_composite
        else:
            im_newPos = self.meshgrid_out[...,0:1]
        
        im_newPos = torch.permute(im_newPos, [3,2,1,0]) # Requires 1/2 transpose because otherwise results are transposed from torchvision Resize
        if flag_batched:
            ## repmat for batch dimension
            im_newPos = torch.tile(im_newPos, (tensor.shape[0], 1, 1, 1))
        ret = torch.nn.functional.grid_sample( tensor, 
                                                im_newPos, 
                                                mode='bicubic',
                                                # mode='bicubic', 
                                                padding_mode='zeros', 
                                                align_corners=True)
        ret = ret[0] if not flag_batched else ret
        return ret
        
    def __repr__(self):
        return f"WarpPoints(r={self.r}, cx={self.cx}, cy={self.cy}, dx={self.dx}, dy={self.dy}, n_warps={self.n_warps}, prob={self.prob}, img_size_in={self.img_size_in}, img_size_out={self.img_size_out})"
    
    

class Horizontal_stripe_scale(Module):
    """
    Adds horizontal stripes. Can be used for for augmentation
     in images generated from raster scanning with bidirectional
     phased raster scanning.
    RH 2022
    """
    def __init__(self, alpha_min_max=(0.5,1), im_size=(36,36), prob=0.5):
        """
        Initializes the class.
        Args:
            alpha_min_max (2-tuple of floats):
                Range of scaling to apply to stripes.
                Will be pulled from uniform distribution.
        """
        super().__init__()
        
        self.alpha_min = alpha_min_max[0]
        self.alpha_max = alpha_min_max[1]
        self.alpha_range = alpha_min_max[1] - alpha_min_max[0]
        
        self.stripes_odd   = (torch.arange(im_size[0]) % 2)
        self.stripes_even = ((torch.arange(im_size[0])+1) % 2)
        
        self.prob = prob

    def forward(self, tensor):
#         assert tensor.ndim==3, "RH ERROR: Number of dimensions of input tensor should be 3: (n_images, height, width)"
        
        if torch.rand(1) < self.prob:
            n_ims = tensor.shape[0]
            alphas_odd  = (torch.rand(n_ims)*self.alpha_range) + self.alpha_min
            alphas_even = (torch.rand(n_ims)*self.alpha_range) + self.alpha_min

            stripes_mask = (self.stripes_odd[None,:]*alphas_odd[:,None]) + (self.stripes_even[None,:]*alphas_even[:,None])
            mask = torch.ones(tensor.shape[1], tensor.shape[2]) * stripes_mask[:,:,None]

            return mask*tensor
        else:
            return tensor

class Horizontal_stripe_shift(Module):
    """
    Shifts horizontal stripes. Can be used for for augmentation
     in images generated from raster scanning with bidirectional
     phased raster scanning.
    RH 2022
    """
    def __init__(self, alpha_min_max=(0,5), im_size=(36,36), prob=0.5):
        """
        Initializes the class.
        Args:
            alpha_min_max (2-tuple of ints):
                Range of absolute shift differences between
                 adjacent horizontal lines. INCLUSIVE.
                In pixels.
                Will be pulled from uniform distribution.
        """
        super().__init__()
        
        self.alpha_min = int(alpha_min_max[0])
        self.alpha_max = int(alpha_min_max[1] + 1)
        self.alpha_range = int(alpha_min_max[1] - alpha_min_max[0])
        
        self.idx_odd   = (torch.arange(im_size[0]) % 2).type(torch.bool)
        self.idx_even = ((torch.arange(im_size[0])+1) % 2).type(torch.bool)
        
        self.prob = prob
        
    def forward(self, tensor):
#         assert tensor.ndim==3, "RH ERROR: Number of dimensions of input tensor should be 3: (n_images, height, width)"
        
        if torch.rand(1) < self.prob:
            n_ims = tensor.shape[0]
            shape_im = (tensor.shape[1], tensor.shape[2])

            alpha = torch.randint(low=self.alpha_min, high=self.alpha_max, size=[n_ims]) * (torch.randint(low=0, high=2, size=[n_ims])*2 - 1)
#             alpha = (torch.randint(high=self.alpha_max-self.alpha_min, size=[n_ims]) + self.alpha_min) * (torch.randint(high=2, size=[n_ims])*2 - 1)
            alpha_half = alpha/2
            alphas_odd  =  torch.ceil(alpha_half).type(torch.int64)
            alphas_even = -torch.floor(alpha_half).type(torch.int64)

            out = torch.zeros_like(tensor)
            for ii in range(out.shape[0]):
                idx_take = slice(max(0, -alphas_odd[ii]) , min(shape_im[1], shape_im[1]-alphas_odd[ii]))
                idx_put = slice(max(0, alphas_odd[ii]) , min(shape_im[1], shape_im[1]+alphas_odd[ii]))
                out[ii, self.idx_odd, idx_put] = tensor[ii, self.idx_odd, idx_take]

                idx_take = slice(max(0, -alphas_even[ii]) , min(shape_im[1], shape_im[1]-alphas_even[ii]))
                idx_put = slice(max(0, alphas_even[ii]) , min(shape_im[1], shape_im[1]+alphas_even[ii]))
                out[ii, self.idx_even, idx_put] = tensor[ii, self.idx_even, idx_take]

            return out
        else:
            return tensor



class Scale_image_sum(Module):
    """
    Scales the entire image so that the sum is user-defined.
    RH 2022
    """
    def __init__(self, sum_val:float=1.0, epsilon=1e-9, min_sub=True):
        """
        Initializes the class.
        Args:
            sum_val (float):
                Value used to normalize the sum of each image.
            epsilon (float):
                Value added to denominator to prevent 
                 dividing by zero.
        """
        super().__init__()
        
        self.sum_val=sum_val
        self.epsilon=epsilon
        self.min_sub=min_sub

    def forward(self, tensor):
        out = self.sum_val * (tensor / (torch.sum(tensor, dim=(-2,-1), keepdim=True) + self.epsilon))
        if self.min_sub:
            out = out - torch.min(torch.min(out, dim=-1, keepdim=True)[0], dim=-1, keepdim=True)[0]
        return out
    

class Check_NaN(Module):
    """
    Checks for NaNs.
    RH 2022
    """
    def __init__(self):
        super().__init__()
        

    def forward(self, tensor):
        if tensor.isnan().any():
            print('FOUND NaN')
            
        return tensor


class Random_occlusion(Module):
    """
    Randomly occludes a slice of the entire image.
    RH 2022
    """
    def __init__(self, prob=0.5, size=(0.3, 0.5)):
        """
        Initializes the class.
        Args:
            prob (float):
                Probability of occlusion.
            size (2-tuple of floats):
                Size of occlusion.
                In percent of image size.
                Will be pulled from uniform distribution.
            seed (int):
                Seed for random number generator.
        """
        super().__init__()
        
        self.prob = prob
        self.size = size

        self.rotator = torchvision.transforms.RandomRotation(
            (-180,180),
            # interpolation='nearest', 
            expand=False, 
            center=None, 
            fill=0,
        )
        
    def forward(self, tensor):
        if torch.rand(1) < self.prob:
            size_rand = torch.rand(1) * (self.size[1] - self.size[0]) + self.size[0]
            idx_rand = ((torch.ceil(tensor.shape[1] * (1-size_rand)).int().item()) , 0)
            mask = torch.ones_like(tensor)
            mask[:, idx_rand[0]:, :] = torch.zeros(1)

            out = tensor * self.rotator(mask).type(torch.bool)
            return out
        else:
            return tensor
        
class Clip(Module):
    """
    Clips the input tensor to the specified range.
    RH 2025
    """
    def __init__(self, min_val:float=0, max_val:float=1, p:float=1.0):
        """
        Initializes the class.
        Args:
            min_val (float):
                Minimum value to clip to.
            max_val (float):
                Maximum value to clip to.
            p (float):
                Probability of applying the clipping.
        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.p = p

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) < self.p:
            return torch.clamp(tensor, self.min_val, self.max_val)
        return tensor