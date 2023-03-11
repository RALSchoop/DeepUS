"""Library containing main functionality for DeepUS project."""

import torch
import math
import torch.utils.data
import os.path
import h5py
import numpy as np
import random
import torch.nn as nn
from torch.nn.common_types import _size_2_t
from typing import Tuple, Union, List
import torch.nn.functional as F

class InspiredBlock(nn.Module):
    """Modification of ResidualBlock containing the core part.
    
    Main idea consists of 2D convolutional layers with the number of
    input channels equal to the number of output channels, with weight
    standardization and group normalization.
    """
    def __init__(self,
                 channels: int = 64,
                 kernel_size: _size_2_t = 5,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 2,
                 bias: bool = True,
                 groups: int = 8,
                 eps: float = 1e-5,
                 affine: bool = True,
                 inplace_activation: bool = False
                 ) -> None:
        """Define block of network architecture.

        See DataFKImageNetwork.__init__() for an explanation of arguments.
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        self.eps = eps
        self.affine = affine
        self.inplace_activation = inplace_activation

        self.inspired_block = nn.Sequential(
            WSConv2d(self.channels, self.channels, self.kernel_size,
                     self.stride, self.padding, bias=self.bias, eps=self.eps),
            nn.GroupNorm(self.groups, self.channels, self.eps, self.affine),
            nn.ReLU(self.inplace_activation)
        )
    
    def forward(self, x):
        return self.inspired_block(x)

class ResidualBlock(nn.Module):
    """Main part of the network which is a residual block.

    Idea is to do weight standardized convolutionals followed by group
    normalization and ReLU twice before adding a skip connection.

    Note: this block starts with the activation function.
    Reference on deep residual learning: https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self,
                 channels: int = 64,
                 kernel_size: _size_2_t = 5,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 2,
                 bias: bool = True,
                 groups: int = 8,
                 eps: float = 1e-5,
                 affine: bool = True,
                 inplace_activation: bool = False
                 ) -> None:
        """Define block of network architecture.

        See DataFKImageNetwork.__init__() for a description of arguments.
        """
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        self.eps = eps
        self.affine = affine
        self.inplace_activation = inplace_activation

        self.georgios_block = nn.Sequential(
            nn.ReLU(self.inplace_activation),
            WSConv2d(self.channels, self.channels, self.kernel_size,
                     self.stride, self.padding, bias=self.bias, eps=self.eps),
            nn.GroupNorm(self.groups, self.channels, self.eps, self.affine),
            nn.ReLU(self.inplace_activation),
            WSConv2d(self.channels, self.channels, self.kernel_size,
                     self.stride, self.padding, bias=self.bias, eps=self.eps),
            nn.GroupNorm(self.groups, self.channels, self.eps, self.affine)
        )

    def forward(self, x):
        return x + self.georgios_block(x)

class DataFKImageNetwork(nn.Module):
    """Data to image network with f-k migration as network layer.
    
    The main network consists of two 2D-CNNs with the f-k migration
    algorithm in between. This implementation uses both group normalization
    and weight standardization.
    """
    def __init__(self,
                 h_data,
                 input_dim: int = 1,
                 channels: int = 64,
                 kernel_size: _size_2_t = 5,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 2,
                 bias: bool = True,
                 groups: int = 8,
                 eps: float = 1e-5,
                 affine: bool = True,
                 residual: bool = True,
                 inplace_activation: bool = False,
                 cnn_pre: bool = True,
                 cnn_post: bool = True,
                 num_blocks: int = 3
                 ) -> None:
        """Set the important variables for the network.

        Arguments:
         - h_data: Dict used to provide metadata about the data.
            Required fields:
            - Those necessary for torch_fkmig
            - nz_cutoff
            In this version of the implementation the targets are cutoff
            at 1500 points nz, probably due to most of the acoustic power
            being in that range. Moreover it is cut off to remove horizontal
            stripe artefacts at the bottom of the image.
         - input_dim: Number of channels of the input.
         - channels: Number of filters to be used in the convolution layers.
         - kernel_size: Size of the convolution kernels. One size for all.
         - stride: Stride for the convolution layers.
         - padding: Padding for the convolutions.
         - bias: Flag for learnable bias term in convolution layers.
         - groups: Number of groups for the group normalization.
         - eps: A value for numerical stability, used for both the group
            normalization and weight standardization.
         - affine: Flag for learnable per-channel affine parameters of the
            group normalization.
         - residual: Flag to use residual version of the network architecture.
         - inplace_activation: In-place flag for the activation function of
            the core blocks.
         - cnn_pre: Flag for 2D-CNN before fk migration.
         - cnn_post: Flag for 2D-CNN after fk migration.
         - num_blocks: Specify the depth of the CNNs in terms of how many
            blocks are used. Note: residual=False, has num_blocks multiplied
            by 2 so that the depth corresponds to the depths when residual=True.
        """
        super().__init__()
        self.input_dim = input_dim
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        self.eps = eps
        self.affine = affine
        self.residual = residual
        self.inplace_activation = inplace_activation
        self.cnn_pre = cnn_pre
        self.cnn_post = cnn_post
        self.num_blocks = num_blocks

        if self.residual:
            self.blocks = num_blocks * [ResidualBlock(
                self.channels, self.kernel_size, self.stride, self.padding,
                self.bias, self.groups, self.eps, self.affine,
                self.inplace_activation)]

            self.data_to_fk = nn.Sequential(
                WSConv2d(self.input_dim, self.channels, self.kernel_size,
                         self.stride, self.padding, bias=self.bias,
                         eps=self.eps),
                nn.GroupNorm(self.groups, self.channels, self.eps,
                             self.affine),
                *self.blocks,
                nn.ReLU(self.inplace_activation),
                WSConv2d(self.channels, self.input_dim, self.kernel_size,
                         self.stride, self.padding, bias=self.bias,
                         eps=self.eps),
                nn.GroupNorm(1, self.input_dim, self.eps, self.affine),
                nn.Tanh()
            )

            self.fk_to_image = nn.Sequential(
                WSConv2d(self.input_dim, self.channels, self.kernel_size,
                         self.stride, self.padding, bias=self.bias,
                         eps=self.eps),
                nn.GroupNorm(self.groups, self.channels, self.eps,
                             self.affine),
                *self.blocks,
                nn.ReLU(self.inplace_activation),
                WSConv2d(self.channels, self.input_dim, self.kernel_size,
                         self.stride, self.padding, bias=self.bias,
                         eps=self.eps),
                nn.GroupNorm(1, self.input_dim, self.eps, self.affine),
                nn.Sigmoid()
            )
            
        else:
            # Times 2 to make InspiredBlock the same depth as ResidualBlock.
            self.blocks = 2 * self.num_blocks * [InspiredBlock(
                self.channels, self.kernel_size, self.stride, self.padding,
                self.bias, self.groups, self.eps, self.affine,
                self.inplace_activation)]

            self.data_to_fk = nn.Sequential(
                WSConv2d(self.input_dim, self.channels, self.kernel_size,
                         self.stride, self.padding, bias=self.bias,
                         eps=self.eps),
                nn.GroupNorm(self.groups, self.channels, self.eps,
                             self.affine),
                nn.ReLU(self.inplace_activation),
                *self.blocks,
                WSConv2d(self.channels, self.input_dim, self.kernel_size,
                         self.stride, self.padding, bias=self.bias,
                         eps=self.eps),
                nn.GroupNorm(1, self.input_dim, self.eps, self.affine),
                nn.Tanh()
            )

            self.fk_to_image = nn.Sequential(
                WSConv2d(self.input_dim, self.channels, self.kernel_size,
                         self.stride, self.padding, bias=self.bias,
                         eps=self.eps),
                nn.GroupNorm(self.groups, self.channels, self.eps,
                             self.affine),
                nn.ReLU(self.inplace_activation),
                *self.blocks,
                WSConv2d(self.channels, self.input_dim, self.kernel_size,
                         self.stride, self.padding, bias=self.bias,
                         eps=self.eps),
                nn.GroupNorm(1, self.input_dim, self.eps, self.affine),
                nn.Sigmoid()
            )
        
        self.h_data = h_data
        
    def forward(self, input):
        # Note: targets are cutoff in z direction, so the network output
        # is matched to this.
        o_input = input.to(torch.get_default_dtype())
        if self.cnn_pre:
            o_input = self.data_to_fk(o_input)
        o_input = torch.real(torch_fkmig_batch(o_input, self.h_data, o_input.device))
        o_input = torch_image_formation_batch(o_input)
        if self.cnn_post:
            o_input = self.fk_to_image(o_input)
        o_input = o_input[:, :, :self.h_data['nz_cutoff'], :]
        return o_input

class WSConv2d(nn.Conv2d):
    """2D convolutional layer implemented with weight standardization.

    Only use when the convolutional layer is followed by a normalization
    layer e.g. batch normalization or group normalization.

    Based on: https://github.com/joe-siyuan-qiao/WeightStandardization.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None,
                 eps: float = 1e-5
                 ) -> None:
        """Call super and initializes eps for numerical stability."""
        super(WSConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype)
        self.eps = eps
        
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = (weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1)
               + self.eps)
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
        

class UltrasoundDataset(torch.utils.data.Dataset):
    """Ultrasound Plane Wave Phantom Dataset."""

    def __init__(self,
                 root_data_dir,
                 relative_input_dir=os.path.join('Data', 'nAng1'),
                 relative_target_dir=os.path.join('Images', 'nAng75'),
                 input_transform=None,
                 target_transform=None,
                 seed=1
                 ) -> None:
        """Specifies paths and data transforms.

        Arguments:
         - root_data_dir: Root directory for data.
            File metadata.mat expected to be here with LesionIdx.nIdx
            struct field containing the size of the dataset.
         - relative_input_dir: Relative path from root_data_dir to input
            data for training. Default: "join(Data, nAng1)".
         - relative_target_dir: Relative path from root_data_dir to target
            data for training. Default: "join(Images, nAng75)".
         - input_transform: Transform to be applied to the input data.
         - target_transform: Transform to be applied to the target data.
        """
        self.root_data_dir = root_data_dir
        self.input_dir = os.path.join(self.root_data_dir, relative_input_dir)
        self.target_dir = os.path.join(self.root_data_dir, relative_target_dir)
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.h_data = dict()
        self.lesion_idx = dict()
        self.seed = seed
        random.seed(self.seed)

        fp = os.path.join(self.root_data_dir, 'metadata.mat')
        with h5py.File(fp, 'r') as mfile:
            self.size = int(list(mfile.get('LesionIdx').get('nIdx'))[0][0])
    
    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index) -> tuple:
        """Fetch a single sample from the dataset

        The transform callables are responsible for turning the samples
        into torch Tensors.
        """
        fp_data = os.path.join(self.input_dir, 'data_' + str(index) + '.mat')
        with h5py.File(fp_data, 'r') as mfile:
            input = np.swapaxes(np.array(mfile.get('data')), 0, 1)
        # Values in whole dataset range between (+/-)2**14.
        input = input / (2**14)
        
        fp_img = os.path.join(self.target_dir, 'img_' + str(index) + '.mat')
        with h5py.File(fp_img, 'r') as mfile:
            target = np.swapaxes(np.array(mfile.get('img_pp')), 0, 1)

        if self.input_transform:
            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)
        
        return input, target

    def load_header(self):
        """Load header data for f-k migration.
        
        Only supports scalar xmitAngles header field, because the
        currently implemented f-k migration algorithm only supports
        a single angle.

        Maybe add some state to the class for accessing the header.
        """
        fp = os.path.join(self.root_data_dir, 'metadata.mat')
        with h5py.File(fp, 'r') as mfile:
            self.h_data['c'] = list(mfile.get('USHEADER').get('c'))[0][0]
            self.h_data['fs'] = list(mfile.get('USHEADER').get('fs'))[0][0]
            self.h_data['pitch'] = list(
                mfile.get('USHEADER').get('pitch'))[0][0]

            self.h_data['t0'] = np.min(
                np.array(mfile.get('USHEADER').get('xmitDelay')))

            xmitAngles = np.array(mfile.get('USHEADER').get('xmitAngles'))
            if np.shape(xmitAngles) == (1, 1):
                self.h_data['tx_angle'] =  math.radians(xmitAngles[0][0])
            else:
                raise NotImplementedError('Unsupported: multiple angles, '
                                          'in: load_header.')
            
            self.h_data['nz_cutoff'] = int(list(
                mfile.get('TargetInfo').get('nz_cutoff'))[0][0])
            
        return self.h_data
    
    def create_samplers(self,
                        test_fraction: float,
                        train_fraction: float = 1.0,
                        except_lesion: List[str] = []
                        ) -> Tuple[torch.utils.data.SubsetRandomSampler,
                                   torch.utils.data.SubsetRandomSampler]:
        """Create custom samplers for the dataloader based on a test fraction.
        
        Creates custom samplers for the dataloader to get a similar fraction
        of training and testing data for each different lesion type in the
        dataset. Overall the test set will be at least: test_fraction
        * len(dataset) in size.

        Arguments:
         - test_fraction (float): Fraction of the total datset size to
           determine the test set size. The amount of test samples is
           always ceiled.
         - train_fraction (float): Fraction of the remaining samples
           (after splitting the test fraction) to use for training. If
           train_fraction < 1.0, then not all the remaining data samples
           are used and the amount is always ceiled.
         - except_lesion (list(str)): List containing lesion types either:
           'hyper_lesion', 'hypo_lesion', or 'no_lesion'. The lesion types
           contained in this list are exempted from the training and
           test set. (They may later be used for evaluation.)
        
        Return:
         - Tuple of SubsetRandomSampler: train_sampler, test sampler.
        """
        _, dsname = os.path.split(self.root_data_dir)
        if dsname == 'CIRS073_RUMC':
            lesion_idx_field_names = [
                'hyper_lesion1', 'hyper_lesion2','hyper_lesion3',
                'hypo_lesion1', 'hypo_lesion2', 'hypo_lesion3', 'no_lesion1',
                'no_lesion2']
        elif dsname == 'CIRS040GSE':
            lesion_idx_field_names = [
                'high_attenuation_hypoechoic', 'high_attenuation_wires',
                'high_attenuation_plusdb', 'high_attenuation_minusdb',
                'low_attenuation_hypoechoic', 'low_attenuation_wires',
                'low_attenuation_plusdb', 'low_attenuation_minusdb']
        else:
            raise ValueError('Unsupported root data dir, if intended: add to '
                             'UltrasoundDataset.create_sampler.')

        fp = os.path.join(self.root_data_dir, 'metadata.mat')
        with h5py.File(fp, 'r') as mfile:
            for field_name in lesion_idx_field_names:
                self.lesion_idx[field_name] = np.array(
                    list(map(int, np.ndarray.flatten(np.array(
                        mfile.get('LesionIdx').get(field_name))))))
        
        train_indices, test_indices = np.array([]), np.array([])
        for lesion, current_lesion_idx in self.lesion_idx.items():
            if except_lesion:
                for cel in except_lesion:
                    if cel in lesion:
                        skip = True
                        break
                    else:
                        skip = False
                if skip:
                    continue
            random.shuffle(current_lesion_idx)
            split_idx = math.ceil(len(current_lesion_idx) * test_fraction)
            take_train_idx = math.ceil((len(current_lesion_idx) - split_idx)
                                       * train_fraction) + split_idx
            train_indices = np.array(list(map(int, np.append(
                train_indices, current_lesion_idx[split_idx:take_train_idx]))))
            test_indices = np.array(list(map(int, np.append(
                test_indices, current_lesion_idx[:split_idx]))))

        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

        return train_sampler, test_sampler

def torch_interplin(v: torch.Tensor, ds: float, sq: torch.Tensor
                    ) -> torch.Tensor:
    """Single dimension linear interpolation using torch operations.

    Based on interpLIN function from:
     - https://github.com/rehmanali1994/Plane_Wave_Ultrasound_Stolt_F-K_Migration.github.io/blob/master/Python/fkmig.py

    Arguments:
     - v: (2D) tensor in which v(., k) is to be interpolated for each k.
     - ds: scalar in which each index step of v(., k) corresponds to a step
        in ds, for each k. In which v(i, k) actually represents v(i*ds, k),
        i = 1,...,len(v(., k)).
     - sq: 2D tensor query points to interpolate towards. 
        We wish to find: v(sq(., k), k).
    """
    vq = torch.zeros_like(v)

    # Query indices, these positions lie in-between the grid.
    iq = sq / ds

    # Extra -1 on vq.size(dim=1) because the linear interpolation requires two
    # points and in the interpolation below an index higher is always taken.
    out_of_bounds = torch.logical_not(
        within_bounds := torch.lt(iq, v.size(dim=0)-2))
    
    # It should be possible to implement this by removing those out_of_bounds
    # indices instead of correcting for them.
    iqc = iq.masked_fill(out_of_bounds, junk_index := 0)

    iqc_floor = torch.floor(iqc).to(torch.int32)

    # The weights for the linear interpolation.
    lw = iqc - iqc_floor

    # Interpolation per line in 2nd dim, kinda necessary since
    # torch.index_select requires a 1D tensor for index parameter.
    for k in torch.arange(v.size(dim=1)):
        vq[:, k] = (torch.index_select(v[:, k], 0,
                                       iqc_floor[:v.size(dim=0), k])
                    * (1 - lw[:v.size(dim=0), k])
                    + torch.index_select(v[:, k], 0,
                                         iqc_floor[:v.size(dim=0), k] + 1)
                    * lw[:v.size(dim=0), k])
    vqc = vq.masked_fill(out_of_bounds, junk_value := 0)

    return vqc


def torch_fkmig(data: torch.Tensor, h_data, device = None) -> torch.Tensor:
    """Stolt's f-k migration for plane wave ultrasound imaging using PyTorch.

    This function implements Stolt's f-k migration for plane wave ultrasound
    imaging as a transcription of using PyTorch's built in operations (instead
    of numpy):
    - https://github.com/rehmanali1994/Plane_Wave_Ultrasound_Stolt_F-K_Migration.github.io/blob/master/Python/fkmig.py
    
    This allows the fk migration to be used as a network layer with torch's
    AutoGrad feature.

    Note: this is only for a single steering angle.

    Note: returns complex tensors with imaginary parts that are very small,
    probably due to precisional errors. Cast the result to real to strictly
    keep the real part.

    Arguments:
     - data: 2D tensor of RF signals acquired using a plane wave configuration.
        Each column corresponds to the RF signals over time of that particular
        transducer element.
     - h_data: Header data dict which contains necessary parameters for the
        fk migration execution.
        Necessary keys:
          - pitch [m],
          - fs [Hz] (sampling frequency),
          - tx_angle [rad] {often 0},
          - c [m/s] (sound velocity) {often 1540},
          - t0 [s] (acquisition start time) {often 0}.
    """
    nt, nx = data.size()

    # Makes sure nt_fft is even.
    nt_shift = 2 * math.ceil(h_data['t0'] * h_data['fs'] / 2)
    # Extenstive time domain zero-padding to interpolate in Fourier domain
    # for linear interpolation.
    nt_fft = 4 * nt + nt_shift

    # Zero padding which interpolates in Fourier domain to avoid lateral edge
    # effects. Again this formula makes sure nx_fft is an even number close
    # to factor * nx. As for the factor, taken here: 4, GitHub ref: 1.5.
    nx_fft = 2 * math.ceil((factor := 4) * nx / 2)

    f_vec = (torch.arange(nt_fft / 2 + 1, device=device)
             * (h_data['fs'] / nt_fft))

    # Peculiar looking order, but it might not actually be that peculiar:
    #  - https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html#numpy.fft.ifft
    kx_vec = torch.roll(torch.arange(-nx_fft / 2, nx_fft / 2, device=device)
                        + 1, int(nx_fft / 2 + 1)) / h_data['pitch'] / nx_fft

    # Grid for spatial and temporal frequencies.
    kx, f = torch.meshgrid(kx_vec, f_vec, indexing='xy')

    o_data = torch.fft.fft(data, nt_fft, 0)

    # Exploit symmetry of real signal FFT: keeping only positive frequencies.
    # Consider using rfft above instead.
    ol_data = o_data[:int(nt_fft / 2 + 1), :]

    sin_a = torch.sin(torch.tensor(h_data['tx_angle'], device=device))

    t_delay = (sin_a * ((nx - 1) * int(h_data['tx_angle'] < 0)
                        - torch.arange(nx, device=device))
                     * (h_data['pitch'] / h_data['c']))
    
    t_trim, f_trim = torch.meshgrid(t_delay + h_data['t0'], f_vec,
                                    indexing='xy')

    # Compensate for steering angle and depth start.
    ol_data = ol_data * torch.exp(
        -2 * torch.complex(torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device)) * math.pi * t_trim * f_trim)

    ol_data = torch.fft.fft(ol_data, nx_fft, 1)

    cos_a = torch.cos(torch.tensor(h_data['tx_angle'], device=device))

    # Model values
    v_erm = h_data['c'] / torch.sqrt(1 + cos_a + sin_a**2)
    beta = ((1 + cos_a) ** 1.5) / (1 + cos_a + sin_a ** 2)

    # Spectral remapping
    kz = 2 * f / (beta * h_data['c'])
    f_kz = v_erm * torch.sqrt(kx**2 + kz**2)

    evanescent = torch.lt(
        torch.abs(f) / (torch.abs(kx)
                        + torch.finfo(torch.get_default_dtype()).eps),
        torch.tensor(h_data['c'], device=device))
    # Removal of evanescent part.
    ol_data = torch.masked_fill(ol_data, evanescent, 0)

    # Interpolation for the change of variable.
    ol_data = (torch_interplin(torch.real(ol_data), h_data['fs'] / nt_fft, f_kz)
               + torch.complex(torch.tensor(0.0, device=device),
                        torch.tensor(1.0, device=device))
               * torch_interplin(
                    torch.imag(ol_data), h_data['fs'] / nt_fft, f_kz))
    
    # Obliquity factor: not sure what this exactly is, but it looks like
    # something close to df(kz)/dkz.
    ol_data = ol_data * f / (f_kz + torch.finfo(torch.get_default_dtype()).eps)
    mq = torch.zeros_like(ol_data, dtype=torch.bool)
    mq[0] = 1
    # Non-in-place modification of first element.
    ol_data = torch.masked_fill(ol_data, mq, 0)

    # Use symmetry to get other values back again.
    oln_data = torch.conj(torch.fliplr(torch.roll(ol_data, -1, 1)))
    # Negative step (was) not possible with tensors.
    inv_idx = torch.arange(int(nt_fft / 2 - 1), 0, -1, device=device)
    ol_data = torch.cat((ol_data, torch.index_select(oln_data, 0, inv_idx)), 0)

    ol_data = torch.fft.ifft(ol_data, dim=0)

    # Model value
    gamma = sin_a / (2 - cos_a)
    # Compensation for steering angle
    dx = (-gamma * (torch.arange(nt_fft, device=device)
                    / torch.tensor(h_data['fs'], device=device))
                 * h_data['c'] / 2)
    kx_z, gamma_z = torch.meshgrid(kx_vec, dx.to(kx_vec.dtype), indexing='xy')
    ol_data = ol_data * torch.exp(
        -2 * torch.complex(torch.tensor(0.0, device=device),
                           torch.tensor(1.0, device=device))
           * math.pi * kx_z * gamma_z)

    ol_data = torch.fft.ifft(ol_data, dim=1)

    olf_data = ol_data[torch.arange(nt, device=device) + nt_shift, :nx]

    return olf_data

def torch_fkmig_batch(data: torch.Tensor, h_data, device = None
                      ) -> torch.Tensor:
    """The f-k migration algorithm for batched data.

    Works for shapes:
     - Batches, Height, Width
     - Batches, Channels, Height, Width
    """
    if data.dim() == 3:
        # Code was made for this case.
        pass
    elif data.dim() == 4 and data.size(1) == 1:
        # Squeeze channel dim to apply the implemented case, then unsqueeze.
        return torch.unsqueeze(torch_fkmig_batch(torch.squeeze(data, 1),
                                                 h_data, device=device), 1)
    elif data.dim() == 2:
        # Can easily support this.
        raise ValueError("Use non-batch version of torch_fkmig_batch.")
    else:
        raise ValueError("Unsupported dimensions.")

    for batch in range(data.size(0)):
        if batch == 0:
            tmp_data = torch_fkmig(data[batch, :, :], h_data, device=device)
            o_data = torch.zeros(data.size(0), *tmp_data.size(),
                                 dtype=tmp_data.dtype, device=tmp_data.device)
            o_data[batch, :, :] = tmp_data
            continue
        o_data[batch, :, :] = torch_fkmig(data[batch, :, :], h_data, device=device)
    return o_data

def torch_hilbert(x: torch.Tensor, N: int = None, dim: int = -1
                  ) -> torch.Tensor:
    """Computes the analytical signal, using the Hilbert transform.

    Implementation based on scipy.signaltools.hilbert.

    Arguments:
     - x: Signal data, must be real valued.
     - N: Number of Fourier components. Default: x.size(dim=axis).
     - dim: Axis/dimension along which to do the transformation. Default: -1.
    """
    if torch.is_complex(x):
        raise ValueError("x must be real.")
    if N is None:
        N = x.size(dim)
    if N <= 0:
        raise ValueError("N must be positive.")
    
    Xf = torch.fft.fft(x, N, dim)
    
    h = torch.zeros(N, device=x.device)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    
    if x.dim() > 1:
        # To apply h along axis, considering all other dims.
        ind = [None] * x.dim()
        ind[dim] = slice(None)
        h = h[tuple(ind)]
    
    x = torch.fft.ifft(Xf * h, dim=dim)

    return x

def torch_image_formation(data_mig: torch.Tensor, clip: float = -70
                          ) -> torch.Tensor:
    """Image formation to create the targets.

    Steps:
     - Takes absolute value of hilbert transform (envelope detection). 
     - Log transformation.
     - Clipping at clip dB w.r.t. max value. {default -70}
     - Shift from negative values and scaling to [0, 1].
    """
    if clip > 0:
        raise ValueError("Clipping value is required < 0 dB")

    img = torch.abs(torch_hilbert(data_mig, dim=0))
    img = 20 * torch.log10(img)
    img = img - torch.max(img)
    # Note: max value is now 0.

    # Clipping
    img = torch.masked_fill(img, torch.lt(img, clip), clip)

    # Shift and scale
    img = (img + abs(clip)) / abs(clip)

    return img

def torch_image_formation_batch(data_mig: torch.Tensor, clip: float = -70
                                ) -> torch.Tensor:
    """The torch_image_formation function for batches.
    
    Works for shapes:
     - Batches, Height, Width
     - Batches, Channels, Height, Width
    """
    if data_mig.dim() == 3:
        # Code was made with this assumption.
        pass
    elif data_mig.dim() == 4 and data_mig.size(1) == 1:
        # Squeeze channel dimension if it's an expected grayscale.
        return torch.unsqueeze(torch_image_formation_batch(
            torch.squeeze(data_mig, 1), clip), 1)
    elif data_mig.dim() == 2:
        # Not that big of an effort to combine these now.
        raise ValueError("Use non-batch version of this function.")
    else:
        raise ValueError("Unsupported dimensions.")

    for batch in range(data_mig.size(0)):
        if batch == 0:
            tmp_data_mig = torch_image_formation(data_mig[batch, :, :], clip)
            o_data_mig = torch.zeros(data_mig.size(0), *tmp_data_mig.size(),
                                     dtype=tmp_data_mig.dtype,
                                     device=tmp_data_mig.device)
            o_data_mig[batch, :, :] = tmp_data_mig
            continue
        o_data_mig[batch, :, :] = torch_image_formation(data_mig[batch, :, :],
                                                        clip)
    return o_data_mig