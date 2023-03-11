"""Evaluate resolution."""

import typing
import torch
import numpy as np
import deepus
import matplotlib.pyplot as plt
import torch.utils.data
import torchvision
from matplotlib.patches import Rectangle
import evaluation as eva
from os.path import join

# Global initializations.
torch.manual_seed(seed := 1)

# Specify your path to the root data directory for resolution evaluation.
# This should be of the form: r'...\CIRS040GSE' or r'.../CIRS040GSE',
# depending on your os's seperator.
root_data_dir = r'D:\Files\CWI Winter Spring 2022\Data\DeepUS\TrainingData\CIRS040GSE'

# Initialize dataset.
deepus_dataset = deepus.UltrasoundDataset(
        root_data_dir,
        input_transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]),
        target_transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]))
h_data = deepus_dataset.load_header()

# Sample choice.
# Indices 6 to 10 and 26 to 30 are wires samples for resolution evaluation.
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
model = model = deepus.DataFKImageNetwork(h_data, residual=True, num_blocks=3)
# Specify desired model state dict paths corresponding to model configuration.
# These are essentially the model weights after training.
runs_root = r'D:\Files\CWI Winter Spring 2022\DeepUS\pythonnew'
msd_paths = [join(runs_root, 'gpu_s_runs', 'deepus_experiment_s1',
                           'model_epoch67.msd'),
             join(runs_root, 'gpu_sf_runs', 'deepus_experiment_s1f1',
                   'model_epoch69.msd'),
             join(runs_root, 'gpu_sf_runs', 'deepus_experiment_s1f2',
                   'model_epoch62.msd'),
             join(runs_root, 'gpu_sf_runs', 'deepus_experiment_s1f3',
                   'model_epoch69.msd'),
             join(runs_root, 'gpu_sf_runs', 'deepus_experiment_s1f4',
                   'model_epoch68.msd')]

# Generate model outputs.
model_outputs = [eva.model_output(model, msd_path, input)
                 for msd_path in msd_paths]

# Specify rectangles for resolution evaluation.
# A different rectangle can be defined for axial and lateral direction,
# as well as the input, model outputs and target images.
input_roi_ax = Rectangle((65, 730), 12, 85)
models_roi_ax = input_roi_ax
# Too much noise and too little signal over the area from wire for fair
# comparison of target using same Rectangle as input with Gaussian fit
# for resolution PSF estimation. Hence tailored ROI for target.
target_roi_ax = Rectangle((68, 730), 6, 75)

input_roi_lat = Rectangle((65, 730), 12, 85)
models_roi_lat = input_roi_lat
target_roi_lat = input_roi_lat

# Function for display.
def display_img(img: torch.Tensor,
                title: str = '',
                roi: Rectangle = None,
                physical_units = True,
                pdeltaz: float = 2.4640e-2,
                pdeltax: float = 1.9530e-1) -> None:
    """Display image"""
    _, axs = plt.subplots()
    axs.set_title(title)
    axs.imshow(torch.squeeze(img, 1)[0, :h_data['nz_cutoff'], :].numpy(),
               aspect=1/6, cmap='gray')
    if physical_units:
        axs.set_ylabel('Axial [mm]')
        axs.set_xlabel('Lateral [mm]')
        axs.set_xticks(np.round((np.array([-10, -5, 0, 5, 10]) + 12.4)
                                / pdeltax), ['-10', '-5', '0', '5', '10'])
        axs.set_yticks(np.round(np.array([0, 5, 10, 15, 20, 25, 30, 35])
                                / pdeltaz), ['0', '5', '10', '15', '20',
                                             '25', '30', '35'])
    if roi is not None:
        axs.add_patch(Rectangle(roi.get_xy(), roi.get_width(),
                      roi.get_height(), facecolor='none', edgecolor='g'))

# Set flag if you want to see all the images.
# Use display_img() roi argument to experiment and find the desired ROI.
# Passing physical_units as false can help finding the exact pixel coordinates.
display = True
if display:
    display_img(input_img, 'Input')
    display_img(target, 'Target')
    for i, model_img in enumerate(model_outputs):
        display_img(model_img, f'Model Output {i}')

# Resolution evaluation in terms of PSF estimation.
input_psf = eva.psf(input_img, input_roi_ax, input_roi_lat)
target_psf = eva.psf(target, target_roi_ax, target_roi_lat)
model_psfs = [eva.psf(model_img, models_roi_ax, models_roi_lat)
              for model_img in model_outputs]

# Now you can save input_psf, target_psf, and model_psfs to your liking.
# Or modify this script and use them in whatever way you'd like.

print(input_psf)
print(target_psf)
print(model_psfs)

if display:
    # Locks further code execution so show last.
    plt.show()