"""Evaluation using global evaluation metrics."""

import torch
import numpy as np
import deepus
import matplotlib.pyplot as plt
import torch.utils.data
import torchvision
import evaluation as eva
from os.path import join
from typing import Generator, Tuple
from figure import get_model_output, make_figure

# Global initializations.
torch.manual_seed(seed := 1)

# Specify the root dataset folder here.
data_root      = r'D:\Files\CWI Winter Spring 2022\Data\DeepUS\\'
data_set       = 'CIRS073_RUMC'

# Initialize dataset.
deepus_dataset = deepus.UltrasoundDataset(
        join(data_root, 'TrainingData', data_set),
        input_transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]),
        target_transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]))
h_data = deepus_dataset.load_header()
train_sampler, _ = deepus_dataset.create_samplers(0.2)
n_samples_train = len(train_sampler.indices)

# Select a sample from the test set.
idx = 10

# Load data of specified sample.
input, target = deepus_dataset[idx]
# Add batch dimension for model passthrough.
input = input[None, :]
target = target[None, :]
# Manual image formation for input data.
input_img = deepus.torch_image_formation_batch(
    torch.real(deepus.torch_fkmig_batch(input, h_data)))

# Specify model configuration
# train_fractions = [0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
train_fractions = [0.06, 0.2, 0.5, 1]
# For each train fraction there are 16 different initializations of the model.
# Useful to check if the order of the models is as desired:
# msd_paths[(i-1)*n_init:i*n_init] has the same train_fraction.
# n_init = 16
n_init = 3

# Takes a long time to calculate these so you might want to save and load.
model_outputs_full = get_model_output('full', train_fractions, n_init, input,
                                      data_root=data_root, data_set=data_set,
                                      h_data=h_data)
model_outputs_pre = get_model_output('pre', train_fractions, n_init, input,
                                     data_root=data_root, data_set=data_set,
                                     h_data=h_data)
model_outputs_post = get_model_output('post', train_fractions, n_init, input,
                                      data_root=data_root, data_set=data_set,
                                      h_data=h_data)
    

# Evaluation metric: peak signal to noise ratio (PSNR)
input_psnr = eva.psnr(input_img, target)
model_psnrs_full = [eva.psnr(model_img, target) for model_img
                    in model_outputs_full]
model_psnrs_pre = [eva.psnr(model_img, target) for model_img
                   in model_outputs_pre]
model_psnrs_post = [eva.psnr(model_img, target) for model_img
                    in model_outputs_post]
fig, ax = make_figure(model_psnrs_full, model_psnrs_pre, model_psnrs_post,
                      train_fractions=train_fractions, n_init=n_init,
                      n_samples_train=n_samples_train,
                      figure_type='fill_between')
ax.axhline(y=input_psnr, label='Input', ls='--', c='y')
ax.legend(loc='lower left')
ax.set_ylabel('PSNR [dB]')
ax.set_xlabel('Number of training samples')
ax.set_title('PSNR for different number of training samples')
ax.set_xlim(1, n_samples_train)
# ax.set_ylim(18, 25)

# Evaluation metric: l1 loss
input_l1loss = eva.l1loss(input_img, target)
model_l1losses_full = [eva.l1loss(model_img, target) for model_img
                       in model_outputs_full]
model_l1losses_pre = [eva.l1loss(model_img, target) for model_img
                      in model_outputs_pre]
model_l1losses_post = [eva.l1loss(model_img, target) for model_img
                       in model_outputs_post]
fig, ax = make_figure(model_l1losses_full, model_l1losses_pre,
                      model_l1losses_post, train_fractions=train_fractions,
                      n_init=n_init, n_samples_train=n_samples_train,
                      figure_type='fill_between')
ax.axhline(y=input_l1loss, label='Input', ls='--', c='y')
ax.legend(loc='lower left')
ax.set_ylabel('L1 Loss')
ax.set_xlabel('Number of training samples')
ax.set_title('L1 Loss for different number of training samples')
ax.set_xlim(1, n_samples_train)

# Evaluation metric: l2 loss
input_l2loss = eva.l2loss(input_img, target)
model_l2losses_full = [eva.l2loss(model_img, target) for model_img
                       in model_outputs_full]
model_l2losses_pre = [eva.l2loss(model_img, target) for model_img
                      in model_outputs_pre]
model_l2losses_post = [eva.l2loss(model_img, target) for model_img
                       in model_outputs_post]
fig, ax = make_figure(model_l2losses_full, model_l2losses_pre,
                      model_l2losses_post, train_fractions=train_fractions,
                      n_init=n_init, n_samples_train=n_samples_train,
                      figure_type='fill_between')
ax.axhline(y=input_l2loss, label='Input', ls='--', c='y')
ax.legend(loc='lower left')
ax.set_ylabel('L2 Loss')
ax.set_xlabel('Number of training samples')
ax.set_title('L2 Loss for different number of training samples')
ax.set_xlim(1, n_samples_train)

# Evalution metric: normalized cross correlation (NCC)
input_ncc = eva.ncc(input_img, target)
model_nccs_full = [eva.ncc(model_img, target) for model_img
                   in model_outputs_full]
model_nccs_pre = [eva.ncc(model_img, target) for model_img
                  in model_outputs_pre]
model_nccs_post = [eva.ncc(model_img, target) for model_img
                   in model_outputs_post]
fig, ax = make_figure(model_nccs_full, model_nccs_pre, model_nccs_post,
                      train_fractions=train_fractions, n_init=n_init,
                      n_samples_train=n_samples_train,
                      figure_type='fill_between')
ax.axhline(y=input_ncc, label='Input', ls='--', c='y')
ax.legend(loc='lower left')
ax.set_ylabel('Normalized Cross Correlation (NCC)')
ax.set_xlabel('Number of training samples')
ax.set_title('NCC for different number of training samples')
ax.set_xlim(1, n_samples_train)

plt.show()