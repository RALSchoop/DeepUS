# DeepUS
Ultrasound beamforming with end-to-end deep learning for single plane wave imaging. This repository contains code to preprocess raw ultrasound data, simulate ultrasound data, train neural networks for image reconstruction and evaluate the trained models.

## DeepUS Dataset
Associated with this code is the dataset of 220 breast phantom samples and 40 calibration phantom samples. The data can be found on [Zenodo](https://zenodo.org/record/7986407).

## Usage
This data and code is published with the aim to facilitate creation and validation of various algorithms on the same data. Use whichever components from this publication as it suits your needs. When you do please cite [TBD PAPER DOI].

### Preprocessing Data
The data associated with this repository is raw ultrasound plane wave data of 75 angles. Image reconstruction and subsampling of the amount of angles can be done with the script `prepare_radboud_data_for_training.py`.

### Matlab Functionality
The matlab code provided here allows for simulation using the k-Wave toolbox, as well as preprocessing the data. The Matlab and Python code for preprocessing are analogous.

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
* Run `conda install cudnn numpy scipy h5py matplotlib tensorboard`
* Check out [PyTorch start locally](https://pytorch.org/get-started/locally/) to get the right command to install PyTorch with conda, e.g. `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`.
* Check that both pytorch and torchvision were actually compiled for using your GPU correctly by checking `conda list pytorch`, `conda list torchvision`. If this does not show the GPU builds, try running `conda install pytorch=*=*cuda* -c pytorch` or `conda install torchvision=*=*cuda* -c pytorch` again.
* Run `conda install -c conda-forge hdf5storage torchinfo`

## Matlab Requirements
* Download the [k-Wave toolbox](http://www.k-wave.org/).
* Download the [FelixMatlabTools](https://github.com/FelixLucka/FelixMatlabTools) toolbox.
* Modify `startup.m` to set paths to k-Wave, FelixMatlabTools and the folder in which the data is stored (variable `storage_path`).
