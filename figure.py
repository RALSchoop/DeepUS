"""Contains common functionality for the creation of result figures.

NOTE: This should be used in a script-like fashion is done in
*_metrics_figure.py, because this is simply extracted code from an
initial script of that form and the code was written with certain
assumptions of the script-flow on mind. Any other use is fine but
be more careful.
"""

import torch
import numpy as np
import deepus
import matplotlib.pyplot as plt
import torch.utils.data
import torchvision
import evaluation as eva
from os.path import join
from typing import Generator, Literal, Sequence, Tuple

def get_model_output(model_type, 
                     train_fractions,
                     n_init, # Could generalize this to list instead of amount.
                     input,
                     data_root,
                     data_set,
                     h_data):
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

# Use plt.fill_between to make the nicer figures that you have in mind.
# Errorbar now indicating the standard deviation over the initializations.
def make_figure(model_metric_full: torch.Tensor,
                model_metric_pre: torch.Tensor,
                model_metric_post: torch.Tensor,
                train_fractions: Sequence[float],
                n_init: int,
                n_samples_train: int,
                figure_type: Literal['errorbar', 'fill_between'] = 'fill_between'
                ):
    # Note: model_metric is (now: is assumed) same ordering as msd_paths.
    def _tfpi(model_metric) -> Generator[Tuple[float, float], None, None]:
        """Get mean and std for each train fraction over the initialization."""
        for init in range(len(train_fractions)):
            # Same #initializations for each train fraction.
            tf_inits = np.array(model_metric[init * n_init:(init + 1) * n_init])
            yield (np.mean(tf_inits), np.std(tf_inits))

    me_full = np.array(tuple(zip(*_tfpi(model_metric_full))))
    me_pre = np.array(tuple(zip(*_tfpi(model_metric_pre))))
    me_post = np.array(tuple(zip(*_tfpi(model_metric_post))))

    fig, ax = plt.subplots()
    # It's actually not this because of ceiling and flooring in creating the
    # samplers. But in approximation it's ok.
    train_amount = list(n_samples_train * np.array(train_fractions))
    if figure_type == 'errorbar':
        ax.errorbar(train_amount, me_full[0], me_full[1],
                    label='Complete model', c='c', marker='H')
        ax.errorbar(train_amount, me_pre[0], me_pre[1],
                    label='Pre-processing model', c='salmon', marker='H')
        ax.errorbar(train_amount, me_post[0], me_post[1],
                    label='Post-processing model', c='tan', marker='H')
    elif figure_type == 'fill_between':
        ax.plot(train_amount, me_full[0], label='Complete model', c='c',
                marker='H')
        ax.fill_between(train_amount, me_full[0] + me_full[1],
                        me_full[0] - me_full[1], color='c', alpha=0.2)
        ax.plot(train_amount, me_pre[0], label='Pre-processing model',
                c='salmon', marker='H')
        ax.fill_between(train_amount, me_pre[0] + me_pre[1],
                        me_pre[0] - me_pre[1], color='salmon', alpha=0.2)
        ax.plot(train_amount, me_post[0], label='Post-processing model',
                c='tan', marker='H')
        ax.fill_between(train_amount, me_post[0] + me_post[1],
                        me_post[0] - me_post[1], color='tan', alpha=0.2)
    else:
        raise ValueError(f'Invalid figure_type specificied: {figure_type}')
    
    return fig, ax