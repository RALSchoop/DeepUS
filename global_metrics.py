"""Evaluation using global evaluation metrics."""

import torch
import numpy as np
import deepus
import matplotlib.pyplot as plt
import torch.utils.data
import torchvision
import evaluation as eva
from os.path import join

# Global initializations.
torch.manual_seed(seed := 1)

# Specify your path to the root data directory for training and testing.
# This should be of the form: r'...\CIRS073_RUMC' or r'.../CIRS073_RUMC',
# depending on your os's seperator.
root_data_dir = r'D:\Files\CWI Winter Spring 2022\Data\DeepUS\TrainingData\CIRS073_RUMC'

# Initialize dataset.
deepus_dataset = deepus.UltrasoundDataset(
        root_data_dir,
        input_transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]),
        target_transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]))
h_data = deepus_dataset.load_header()

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
model = deepus.DataFKImageNetwork(h_data, residual=True, num_blocks=3)
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

# Function for display.
def display_img(img: torch.Tensor,
                title: str = '',
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
        # + 12.4 has to do with getting the origin in the center.
        axs.set_xticks(np.round((np.array([-10, -5, 0, 5, 10]) + 12.4)
                                / pdeltax), ['-10', '-5', '0', '5', '10'])
        axs.set_yticks(np.round(np.array([0, 5, 10, 15, 20, 25, 30, 35])
                                / pdeltaz), ['0', '5', '10', '15', '20',
                                             '25', '30', '35'])

# Set flag if you want to see all images.
display = True
if display:
    display_img(input_img, 'Input')
    display_img(target, 'Target')
    for i, model_img in enumerate(model_outputs):
        display_img(model_img, f'Model Output {i}')

# Evaluation metric: peak signal to noise ratio (PSNR)
input_psnr = eva.psnr(input_img, target)
model_psnrs = [eva.psnr(model_img, target) for model_img in model_outputs]

# Evaluation metric: l1 loss
input_l1loss = eva.l1loss(input_img, target)
model_l1losses = [eva.l1loss(model_img, target) for model_img in model_outputs]

# Evaluation metric: l2 loss
input_l2loss = eva.l2loss(input_img, target)
model_l2losses = [eva.l2loss(model_img, target) for model_img in model_outputs]

# Evalution metric: normalized cross correlation (NCC)
input_ncc = eva.ncc(input_img, target)
model_nccs = [eva.ncc(model_img, target) for model_img in model_outputs]

# Now you can save the evaluation metrics' results to your liking.
# Or modify this script to use them in whatever way you'd like.

print('PSNR:')
print(f'Input PSNR: {input_psnr}')
print(f'Model PSNRs: {model_psnrs}')
print() # Just for '\n'.
print('l1 loss:')
print(f'Input l1 loss: {input_l1loss}')
print(f'Model l1 losses: {model_l1losses}')
print()
print('l2 loss:')
print(f'Input l2 loss: {input_l2loss}')
print(f'Model l2 losses: {model_l2losses}')
print()
print('NCC:')
print(f'Input NCC: {input_ncc}')
print(f'Model NCCs: {model_nccs}')

if display:
    # Locks further code execution so show last.
    plt.show()