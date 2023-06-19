# DeepUS
Ultrasound beamforming with end-to-end deep learning for single plane wave imaging. This repository contains code to preprocess raw ultrasound data, train neural networks for image reconstruction and evaluate the trained models.

## DeepUS Dataset
Associated with this code is the dataset of 220 breast phantom samples and 40 calibration phantom samples. The data can be found on [Zenodo](https://zenodo.org/record/7986407).

## Usage
### Preprocessing Data
The data associated with this repository is raw ultrasound plane wave data of 75 angles. Image reconstruction and subsampling of the amount of angles can be done with the script `prepare_radboud_data_for_training.py`.

### Main Library
This file `deepus.py` is the main library file containing the implementation details of the fk-migration reconstruction algorithm and of the network architectures.

### Training
In order to train a network use the `train.py` script. Accompanying this script is `coach.py` which contains some useful functionality which is used in the training script.

### Evaluation
When it comes to evaluation there is one library file `evaluation.py` containing functionality and implementation details for doing the evaluation. The script files actually running an evaluation are: 1. `global_metrics.py` computing PSNR, l1-loss, l2-loss, and NCC, 2. `local_metrics.py` computing CNR, CR, and gCNR, and 3. `resolution.py` which determines the PSF.

## Python Requirements - Conda Installation
Here is an example guide to install the dependencies for DeepUS using conda.
* Create a new environment with Python 3.8 `conda create -n deepus python=3.8`.
* Activate the newly created environment `conda activate deepus`.
* Check out [PyTorch start locally](https://pytorch.org/get-started/locally/) to get the right command to install PyTorch with conda, e.g. `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`.
* Install other dependencies using `conda install cudnn numpy scipy h5py matplotlib hdf5storage`.

