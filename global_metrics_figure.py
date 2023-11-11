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

# 0.04 and 0.06 not in pre folder atm, add them once gotten.
train_fractions = [0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# For each train fraction there are 16 different initializations of the model.
# Useful to check if the order of the models is as desired:
# msd_paths[(i-1)*n_init:i*n_init] has the same train_fraction.
n_init = 16

def get_model_output(model_type, train_fractions, n_init):
    # model_type = 'full'
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
    msd_paths = (join(
        trained_NN_root, f'trainfrac{train_frac}', f'rnd{n + 1}', 'model_best.msd')
        for train_frac in train_fractions for n in range(n_init))

    # Generate model outputs.
    model_outputs = [eva.model_output(model, msd_path, input)
                    for msd_path in msd_paths]
    return model_outputs

# Takes a long time to calculate these so you might want to save and load.
model_outputs_full = get_model_output('full', train_fractions, n_init)
model_outputs_pre = get_model_output('pre', train_fractions, n_init)
model_outputs_post = get_model_output('post', train_fractions, n_init)

def make_figure(model_metric_full: torch.Tensor,
                model_metric_pre: torch.Tensor,
                model_metric_post: torch.Tensor):
    # Note: model_metric is same ordering as msd_paths.
    def _tfpi(model_metric) -> Generator[Tuple[float, float], None, None]:
        """Get mean and std for each train fraction over the initialization."""
        for init in range(len(train_fractions)):
            # Same #initializations for each train fraction.
            tf_inits = np.array(model_metric[init * n_init:(init + 1) * n_init])
            yield (np.mean(tf_inits), np.std(tf_inits))

    me_full = tuple(zip(*_tfpi(model_metric_full)))
    me_pre = tuple(zip(*_tfpi(model_metric_pre)))
    me_post = tuple(zip(*_tfpi(model_metric_post)))

    fig, ax = plt.subplots()
    # It's actually not this because of ceiling and flooring in creating the
    # samplers. But in approximation it's ok.
    train_amount = list(n_samples_train * np.array(train_fractions))
    ax.errorbar(train_amount, me_full[0], me_full[1],
                label='Complete model', c='c', marker='H')
    ax.errorbar(train_amount, me_pre[0], me_pre[1],
                label='Pre-processing model', c='salmon', marker='H')
    ax.errorbar(train_amount, me_post[0], me_post[1],
                label='Post-processing model', c='tan', marker='H')
    
    return fig, ax
    

# Evaluation metric: peak signal to noise ratio (PSNR)
input_psnr = eva.psnr(input_img, target)
model_psnrs_full = [eva.psnr(model_img, target) for model_img
                    in model_outputs_full]
model_psnrs_pre = [eva.psnr(model_img, target) for model_img
                   in model_outputs_pre]
model_psnrs_post = [eva.psnr(model_img, target) for model_img
                    in model_outputs_post]

fig, ax = make_figure(model_psnrs_full, model_psnrs_pre, model_psnrs_post)
ax.axhline(y=input_psnr, label='Input', ls='--', c='y')
ax.legend(loc='lower left')
ax.set_ylabel('PSNR [dB]')
ax.set_xlabel('Number of training samples')
ax.set_title('PSNR for different number of training samples')
# Errorbar now indicating the standard deviation over the initializations.
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

# Evaluation metric: l2 loss
input_l2loss = eva.l2loss(input_img, target)
model_l2losses_full = [eva.l2loss(model_img, target) for model_img
                       in model_outputs_full]
model_l2losses_pre = [eva.l2loss(model_img, target) for model_img
                      in model_outputs_pre]
model_l2losses_post = [eva.l2loss(model_img, target) for model_img
                       in model_outputs_post]

# Evalution metric: normalized cross correlation (NCC)
input_ncc = eva.ncc(input_img, target)
model_nccs = [eva.ncc(model_img, target) for model_img in model_outputs_full]
model_nccs = [eva.ncc(model_img, target) for model_img in model_outputs_pre]
model_nccs = [eva.ncc(model_img, target) for model_img in model_outputs_post]
