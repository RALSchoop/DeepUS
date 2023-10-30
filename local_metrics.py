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

# Global initializations.
torch.manual_seed(seed := 1)

# Specify the root dataset folder here.
data_root      = r'/export/scratch2/felix/Dropbox/Data/US/DeepUSData/'
data_set       = 'CIRS073_RUMC'

# Initialize dataset.
deepus_dataset = deepus.UltrasoundDataset(
        join(data_root, 'TrainingData', data_set),
        input_transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]),
        target_transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]))
h_data = deepus_dataset.load_header()

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
model_type = 'full'

if model_type == 'pre': # pre processing only
    model = deepus.DataFKImageNetwork(h_data, residual=True, cnn_pre=True, cnn_post=False, num_blocks=6)
elif model_type == 'post': # post processing only
    model = deepus.DataFKImageNetwork(h_data, residual=True, cnn_pre=False, cnn_post=True, num_blocks=6)
elif model_type == 'full': # full model
    model = deepus.DataFKImageNetwork(h_data, residual=True, cnn_pre=True, cnn_post=True, num_blocks=3)
else:
    raise ValueError("model_type needs to be 'pre', 'post' or 'full'")
    
# Specify desired model state dict paths corresponding to model configuration.
# These are essentially the model weights after training.
trained_NN_root = join(data_root, 'TrainedNetworks', data_set , model_type)
msd_paths = [join(trained_NN_root, 'trainfrac1' ,  'rnd1', 'model_best.msd'),
             join(trained_NN_root, 'trainfrac0.5' ,  'rnd1', 'model_best.msd'),
             join(trained_NN_root, 'trainfrac0.1' ,  'rnd1', 'model_best.msd'),
             ]

# Generate model outputs.
model_outputs = [eva.model_output(model, msd_path, input)
                 for msd_path in msd_paths]

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

# Function for display.
def display_img(img: torch.Tensor,
                title: str = '',
                c_ij: Tuple[float, float] = None,
                r1: float = None,
                r2: Tuple[float, float] = None,
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
    if c_ij is not None:
        c_xy = (c_ij[1], c_ij[0])
        if r1 is not None:
            axs.add_patch(Ellipse(c_xy, 2*r1/pdeltax, 2*r1/pdeltaz,
                                facecolor='none', edgecolor='g'))
        if r2 is not None:
            axs.add_patch(Ellipse(c_xy, 2*r2[0]/pdeltax, 2*r2[0]/pdeltaz,
                                facecolor='none', edgecolor='r'))
            axs.add_patch(Ellipse(c_xy, 2*r2[1]/pdeltax, 2*r2[1]/pdeltaz,
                                facecolor='none', edgecolor='r'))
        
# Set flag if you want to see all images.
# Use display_img() c_xy, r1 and r2 arguments to check the desired ROI.
display = True
if display:
    display_img(input_img, 'Input')
    display_img(target, 'Target')
    for i, model_img in enumerate(model_outputs):
        display_img(model_img, f'Model Output {i}')

# Evaluation metrics: contrast to noise ratio (CNR)
input_cnr = eva.cnr(input_img, c_ij=c_ij, r1=r1, r2=r2)
target_cnr = eva.cnr(target, c_ij=c_ij, r1=r1, r2=r2)
model_cnrs = [eva.cnr(model_img, c_ij=c_ij, r1=r1, r2=r2)
              for model_img in model_outputs]

# Evaluation metrics: contrast ratio (CR)
input_cr = eva.cr(input_img, c_ij=c_ij, r1=r1, r2=r2)
target_cr = eva.cr(target, c_ij=c_ij, r1=r1, r2=r2)
model_crs = [eva.cr(model_img, c_ij=c_ij, r1=r1, r2=r2)
             for model_img in model_outputs]

# Evaluation metrics: generalized contrast to noise ratio (gCNR)
input_gcnr = eva.gcnr(input_img, c_ij=c_ij, r1=r1, r2=r2)
target_gcnr = eva.gcnr(target, c_ij=c_ij, r1=r1, r2=r2)
model_gcnrs = [eva.gcnr(model_img, c_ij=c_ij, r1=r1, r2=r2)
               for model_img in model_outputs]

# Now you can save the evaluation metrics' results to your liking.
# Or add to this script to use them in whatever way you'd like.

print('CNR:')
print(f'Input CNR: {input_cnr}')
print(f'Target CNR: {target_cnr}')
print(f'Model CNRs: {model_cnrs}')
print() # Just for '\n'.
print('CR:')
print(f'Input CR: {input_cr}')
print(f'Target CR: {target_cr}')
print(f'Model CRs: {model_crs}')
print() # Just for '\n'.
print('gCNR:')
print(f'Input gCNR: {input_gcnr}')
print(f'Target gCNR: {target_gcnr}')
print(f'Model gCNRs: {model_gcnrs}')

if display:
    # Locks further code execution so show last.
    plt.show()