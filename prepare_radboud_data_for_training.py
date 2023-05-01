"""Python version of matlab script of same name.

Script does image formation on ultrasound acquisition data.

Author: Ryan A.L. Schoop
"""

from os import makedirs
from os.path import join
from glob import glob
from time import time
from datetime import timedelta
import numpy as np
from scipy.io import loadmat
from h5py import File
from torch import from_numpy, real, zeros
import hdf5storage
from deepus import torch_fkmig, torch_image_formation

# Script assumes that the data is stored in 'ExperimentalData/CIRS073_RUMC' or
# 'ExperimentalData/CIRS040GSE/' within some storage_path root directory.
storage_path = r'D:\Files\CWI Winter Spring 2022\Data\DeepUS'
data_set = 'CIRS040GSE' # Pick CIRS073_RUMC or CIRS040GSE
experimental_data_path = join(storage_path, 'ExperimentalData', data_set)

# This is where the training data will be stored.
training_data_path_root = join(storage_path, 'TrainingDataPyGen', data_set)

# Settings for reconstruction
# Sub-sampled number of steering angles used for reconstruction [1, ..., 75]
n_ang_ss = 16
# Truncation of pixels in axial direction.
# Note: this is rather replaced by proper interpolation.
nz_cutoff = 1500

# Load the header containing information about the transducer settings.
header_file = join(experimental_data_path, 'USHEADER_20220330105548.mat')
usheader = loadmat(header_file, struct_as_record=False)['USHEADER'][0][0]

n_ang = len(usheader.xmitAngles)
n_ele = usheader.xmitDelay.shape[1]

# Prepare to loop over all data sets
data_files = glob(join(experimental_data_path, '**', 'USDATA_*.mat'),
                  recursive=True)
n_data = len(data_files)
# Result directory for images.
training_data_path_img = join(training_data_path_root, 'Images',
                              f'nAng{n_ang_ss}')
makedirs(training_data_path_img, exist_ok=True)
# Result directory for training.
training_data_path_data = join(training_data_path_root, 'Data',
                               f'nAng{n_ang_ss}')
makedirs(training_data_path_data, exist_ok=True)

# Load single dataset for some overarching information.
with File(data_files[1]) as f:
    # Disregard empty dimensions, get right shape and convert to double.
    usdata = np.swapaxes(np.squeeze(np.array(f.get('USDATA'))),
                         0, 2).astype(np.float64)

# If specified, subsample the amount of angles.
if n_ang_ss == 1:
    ang_ind = np.squeeze(np.argwhere(usheader.xmitAngles == 0))[0]
else:
    ang_ss = np.linspace(np.squeeze(usheader.xmitAngles[0]),
                         np.squeeze(usheader.xmitAngles[-1]), n_ang_ss)
    ang_ind = np.zeros(n_ang_ss, dtype=np.int64)
    for i_ang in range(n_ang_ss):
        ang_ind[i_ang] = np.argmin(np.abs(np.squeeze(usheader.xmitAngles)
                                          - ang_ss[i_ang])).squeeze()

usdata = np.atleast_3d(usdata[:, :, ang_ind])
usheader.xmitAngles = np.atleast_1d(np.squeeze(usheader.xmitAngles)[ang_ind])
usheader.xmitDelay = np.squeeze(usheader.xmitDelay)[ang_ind, :]
usheader.xmitFocus = np.squeeze(usheader.xmitFocus)[ang_ind]
usheader.xmitApodFunction = np.squeeze(usheader.xmitApodFunction)[ang_ind, :]

# Speed of sound used for calculations.
sos_bgn = np.squeeze(usheader.c)
# Used to correct for general delay of the acoustic lens.
lens_delay = 96

# Main loop
for i, data_file in enumerate(data_files):
    print(f'Reconstructing data file index {i}...')
    rc_time_start = time()

    with File(data_file) as f:
        usdata = np.swapaxes(np.squeeze(np.array(f.get('USDATA'))),
                            0, 2).astype(np.float64)
        usdata = np.atleast_3d(usdata[:, :, ang_ind])
    
    # Correct general delay, i.e. lens correction
    usdata = usdata[lens_delay:, :, :]

    # Correct angle dependent delay
    n_t = usdata.shape[0]
    dt = 1 / np.squeeze(usheader.fs)
    delay = np.abs((n_ele - 1) / 2 * np.squeeze(usheader.pitch)
                   * np.sin(np.deg2rad(usheader.xmitAngles)) / sos_bgn)
    delay_ind = np.floor(delay / dt).astype(int)
    for i_ang in range(n_ang_ss):
        usdata[:(n_t - delay_ind[i_ang]), :, i_ang] = (
            usdata[delay_ind[i_ang]:, :, i_ang])
        usdata[(n_t - delay_ind[i_ang]):, :, i_ang] = 0
    
    # Correct channel 125
    usdata[:, 125, :] = 2 * usdata[:, 125, :]

    img_fkmig = zeros((usdata.shape[0], usdata.shape[1]))
    for k, xmitAngle in enumerate(usheader.xmitAngles):
        # Set up fk-migration and run it.
        fk_para = {
            'pitch': np.squeeze(usheader.pitch).astype(np.float32),
            'fs': np.squeeze(usheader.fs).astype(np.float32),
            'tx_angle': np.deg2rad(xmitAngle).astype(np.float32),
            'c': sos_bgn.astype(np.float32),
            't0': 0
        }
        # Compounding
        img_fkmig = ((k * img_fkmig
                      + torch_fkmig(from_numpy(usdata[:, :, k]), fk_para))
                     / (k + 1))

    # Image post processing
    img_pp = torch_image_formation(real(img_fkmig), -70)
    
    # Convert back to numpy.
    img_fkmig = img_fkmig.detach().cpu().numpy()
    img_pp = img_pp.detach().cpu().numpy()

    # Truncation in axial direction, though rather done by interpolation.
    img_fkmig = img_fkmig[:nz_cutoff, :]
    img_pp = img_pp[:nz_cutoff, :]

    # Saving.
    options = hdf5storage.Options(matlab_compatible=True)
    res_filename_img = join(training_data_path_img, f'img_{i + 1}.mat')
    hdf5storage.writes(
        {'/img_fkmig': img_fkmig, '/img_pp': img_pp, '/fk_para': fk_para},
        res_filename_img, options=options)

    data = usdata
    res_filename_data = join(training_data_path_data, f'data_{i + 1}.mat')
    fk_para['tx_angle'] = usheader.xmitAngles
    hdf5storage.writes({'/data': data, '/fk_para': fk_para}, res_filename_data,
                       options=options)

    rc_time_end = time()
    print('Reconstruction Duration: '
          f'{str(timedelta(seconds=rc_time_end - rc_time_start))}')
