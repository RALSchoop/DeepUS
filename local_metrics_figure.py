"""Evaluate local evaluation metrics."""

from typing import Tuple
import torch
import numpy as np
import deepus
import matplotlib.pyplot as plt
import torch.utils.data
import torchvision
from matplotlib.patches import Ellipse
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
train_sampler, _ = deepus_dataset.create_samplers(0.2)
n_samples_train = len(train_sampler.indices)

# Sample choice
# Indices 1 to 5 contain hypoechoic regions, indices 11 to 15 have
# circular regions of +3 and +6 dB, and indices 16 to 20 have circular
# regions of -3 and -6 dB from a region with 'low' attentuation coefficient
# of 0.70 dB/(cm MHz).
# Indices 21 to 25, 31 to 35, and 36 to 40 contain the same type of
# regions respectively, but from a region with 'high' attenuation
# coefficient of 0.95 dB/(cm MHz).
idx = 22

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
                                      data_root=data_root,
                                      data_set=data_set_train, h_data=h_data)
model_outputs_pre = get_model_output('pre', train_fractions, n_init, input,
                                     data_root=data_root,
                                     data_set=data_set_train, h_data=h_data)
model_outputs_post = get_model_output('post', train_fractions, n_init, input,
                                      data_root=data_root,
                                      data_set=data_set_train, h_data=h_data)

# Specify circle and ring with common center for evaluation of local metrics.
# Specify center as an ij coordinate tuple, that is axial coordinate first.
# Given in pixel coordinates.
c_ij = (597, 62)
# Radius of circle corresponding to 'contrast' region for evaluation metrics.
# Given in mm.
r1 = 1.3
# Radii of ring corresponding to 'background' region for evaluation metrics.
# Given in mm.
r2 = (2.9, 4.0)

# Evaluation metrics: contrast to noise ratio (CNR)
input_cnr = eva.cnr(input_img, c_ij=c_ij, r1=r1, r2=r2)
target_cnr = eva.cnr(target, c_ij=c_ij, r1=r1, r2=r2)
model_cnrs_full = [eva.cnr(model_img, c_ij=c_ij, r1=r1, r2=r2)
                   for model_img in model_outputs_full]
model_cnrs_pre = [eva.cnr(model_img, c_ij=c_ij, r1=r1, r2=r2)
                  for model_img in model_outputs_pre]
model_cnrs_post = [eva.cnr(model_img, c_ij=c_ij, r1=r1, r2=r2)
                   for model_img in model_outputs_post]

fig, ax = make_figure(model_cnrs_full, model_cnrs_pre, model_cnrs_post,
                      train_fractions=train_fractions, n_init=n_init,
                      n_samples_train=n_samples_train)
ax.axhline(y=input_cnr, label='Input', ls='--', c='y')
ax.axhline(y=target_cnr, label='Target', ls='--', c='m')
ax.legend(loc='lower left')
ax.set_ylabel('CNR [dB]')
ax.set_xlabel('Number of training samples')
ax.set_title('CNR for different number of training samples')
ax.set_xlim(1, n_samples_train)

# Evaluation metrics: contrast ratio (CR)
input_cr = eva.cr(input_img, c_ij=c_ij, r1=r1, r2=r2)
target_cr = eva.cr(target, c_ij=c_ij, r1=r1, r2=r2)
model_crs_full = [eva.cr(model_img, c_ij=c_ij, r1=r1, r2=r2)
                  for model_img in model_outputs_full]
model_crs_pre = [eva.cr(model_img, c_ij=c_ij, r1=r1, r2=r2)
                 for model_img in model_outputs_pre]
model_crs_post = [eva.cr(model_img, c_ij=c_ij, r1=r1, r2=r2)
                  for model_img in model_outputs_post]

# Evaluation metrics: generalized contrast to noise ratio (gCNR)
input_gcnr = eva.gcnr(input_img, c_ij=c_ij, r1=r1, r2=r2)
target_gcnr = eva.gcnr(target, c_ij=c_ij, r1=r1, r2=r2)
model_gcnrs_full = [eva.gcnr(model_img, c_ij=c_ij, r1=r1, r2=r2)
                    for model_img in model_outputs_full]
model_gcnrs_pre = [eva.gcnr(model_img, c_ij=c_ij, r1=r1, r2=r2)
                   for model_img in model_outputs_pre]
model_gcnrs_post = [eva.gcnr(model_img, c_ij=c_ij, r1=r1, r2=r2)
                    for model_img in model_outputs_post]

plt.show()