"""Evaluation of resolution."""

from typing import Tuple
import torch
import numpy as np
import deepus
import matplotlib.pyplot as plt
import torch.utils.data
import torchvision
from matplotlib.patches import Rectangle
import evaluation as eva
from os.path import join
from figure import get_model_output, make_figure

# Global initializations.
torch.manual_seed(seed := 1)

# Specify the root dataset folder here.
data_root      = r'D:\Files\CWI Winter Spring 2022\Data\DeepUS\\'
data_set       = 'CIRS040GSE' # Dataset for the actual current sample index.
data_set_train = 'CIRS073_RUMC' # Dataset on which models were trained.

# Initialize dataset.
deepus_dataset = deepus.UltrasoundDataset(
        join(data_root, 'TrainingData', data_set),
        input_transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]),
        target_transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]))
h_data = deepus_dataset.load_header()

# Just to get the right samplers with the config style above.
deepus_dataset_train = deepus.UltrasoundDataset(
        join(data_root, 'TrainingData', data_set_train),
        input_transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]),
        target_transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]))
train_sampler, _ = deepus_dataset_train.create_samplers(0.2)
n_samples_train = len(train_sampler.indices) # need to correct this.

# Sample choice.
# Indices 6 to 10 are wire samples for resolution evaluation from a 'high'
# attenuation region with attenuation coefficient of 0.95 dB/(cm MHz).
# Indices 26 to 30 are wire samples for resolution evaluation from a 'low'
# attentuation region with attenuation coefficient of 0.70 dB/(cm MHz).
idx = 6

# Load data of specified sample.
input, target = deepus_dataset[idx]
# Add batch dimension for model passthrough.
input = input[None, :]
target = target[None, :]
# Manual image formation for input data.
input_img = deepus.torch_image_formation_batch(
    torch.real(deepus.torch_fkmig_batch(input, h_data)))

# Specify model configuration
train_fractions = [0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# train_fractions = [0.06, 0.2, 0.5, 1]
# For each train fraction there are 16 different initializations of the model.
# Useful to check if the order of the models is as desired:
# msd_paths[(i-1)*n_init:i*n_init] has the same train_fraction.
n_init = 16
# n_init = 3

# Takes a long time to calculate these so you might want to save and load.
model_outputs_full = get_model_output('full', train_fractions, n_init, input,
                                      data_root=data_root,
                                      data_set=data_set_train, h_data=h_data)
model_outputs_pre = get_model_output('pre', train_fractions, n_init, input,
                                     data_root=data_root,
                                     data_set=data_set_train, h_data=h_data)
model_outputs_post = get_model_output('post', train_fractions, n_init, input,
                                      data_root=data_root,
                                      data_set=data_set_train, h_data=h_data)

# Specify rectangles for resolution evaluation.
# A different rectangle can be defined for axial and lateral direction,
# as well as the input, model outputs and target images.
input_roi_ax = Rectangle((65, 730), 11, 85)
models_roi_ax = input_roi_ax
# Too much noise and too little signal over the area from wire for fair
# comparison of target using same Rectangle as input with Gaussian fit
# for resolution PSF estimation. Hence tailored ROI for target.
target_roi_ax = Rectangle((68, 730), 6, 75)

input_roi_lat = Rectangle((65, 730), 11, 85)
models_roi_lat = input_roi_lat
target_roi_lat = input_roi_lat

# Resolution evaluation in terms of PSF estimation.
# Returned as (Axial FWHM, Lateral FWHM), so after zip axial is first dim.
input_psf = eva.psf(input_img, input_roi_ax, input_roi_lat)
target_psf = eva.psf(target, target_roi_ax, target_roi_lat)
model_psfs_full = tuple(zip(*(eva.psf(model_img, models_roi_ax, models_roi_lat)
                              for model_img in model_outputs_full)))
model_psfs_pre = tuple(zip(*(eva.psf(model_img, models_roi_ax, models_roi_lat)
                             for model_img in model_outputs_pre)))
model_psfs_post = tuple(zip(*(eva.psf(model_img, models_roi_ax, models_roi_lat)
                              for model_img in model_outputs_post)))

fig, ax = make_figure(model_psfs_full[0], model_psfs_pre[0], model_psfs_post[0],
                      train_fractions=train_fractions, n_init=n_init,
                      n_samples_train=n_samples_train,
                      figure_type='fill_between')
ax.axhline(y=input_psf[0], label='Input', ls='--', c='y')
ax.axhline(y=target_psf[0], label='Target', ls='--', c='m')
ax.legend(loc='lower left')
ax.set_ylabel('Axial FWHM [mm]')
ax.set_xlabel('Number of training samples')
ax.set_title('Axial FWHM for different number of training samples')
ax.set_xlim(1, n_samples_train)
ax.set_ylim(0.63, 1.23)

fig, ax = make_figure(model_psfs_full[1], model_psfs_pre[1], model_psfs_post[1],
                      train_fractions=train_fractions, n_init=n_init,
                      n_samples_train=n_samples_train,
                      figure_type='fill_between')
ax.axhline(y=input_psf[1], label='Input', ls='--', c='y')
ax.axhline(y=target_psf[1], label='Target', ls='--', c='m')
ax.legend(loc='lower left')
ax.set_ylabel('Lateral FWHM [mm]')
ax.set_xlabel('Number of training samples')
ax.set_title('Lateral FWHM for different number of training samples')
ax.set_xlim(1, n_samples_train)
ax.set_ylim(0.66, 1.08)

# No fit variant.
model_psfs_full_nf = tuple(zip(*(eva.psf(model_img, models_roi_ax,
                                         models_roi_lat, fit=False)
                                 for model_img in model_outputs_full)))
model_psfs_pre_nf = tuple(zip(*(eva.psf(model_img, models_roi_ax,
                                        models_roi_lat, fit=False)
                                for model_img in model_outputs_pre)))
model_psfs_post_nf = tuple(zip(*(eva.psf(model_img, models_roi_ax,
                                         models_roi_lat, fit=False)
                                 for model_img in model_outputs_post)))

fig, ax = make_figure(model_psfs_full_nf[0], model_psfs_pre_nf[0], model_psfs_post_nf[0],
                      train_fractions=train_fractions, n_init=n_init,
                      n_samples_train=n_samples_train,
                      figure_type='fill_between')
ax.axhline(y=input_psf[0], label='Input', ls='--', c='y')
ax.axhline(y=target_psf[0], label='Target', ls='--', c='m')
ax.legend(loc='lower left')
ax.set_ylabel('Axial FWHM [mm]')
ax.set_xlabel('Number of training samples')
ax.set_title('No fit axial FWHM for different number of training samples')
ax.set_xlim(1, n_samples_train)
ax.set_ylim(0.20, 1.1)

fig, ax = make_figure(model_psfs_full_nf[1], model_psfs_pre_nf[1], model_psfs_post_nf[1],
                      train_fractions=train_fractions, n_init=n_init,
                      n_samples_train=n_samples_train,
                      figure_type='fill_between')
ax.axhline(y=input_psf[1], label='Input', ls='--', c='y')
ax.axhline(y=target_psf[1], label='Target', ls='--', c='m')
ax.legend(loc='lower left')
ax.set_ylabel('Lateral FWHM [mm]')
ax.set_xlabel('Number of training samples')
ax.set_title('No fit lateral FWHM for different number of training samples')
ax.set_xlim(1, n_samples_train)
ax.set_ylim(0.60, 1.25)

plt.show()