"""Python version of matlab script of same.

Script does image formation on ultrasound acquisition data.]

Author: Ryan A.L. Schoop
"""

from os import makedirs
from os.path import join
from glob import glob
import numpy as np
from scipy.io import loadmat
from h5py import File

# Script assumes that the data is stored in 'ExperimentalData/CIRS073_RUMC' or
# 'ExperimentalData/CIRS040GSE/' within some storage_path root directory.
storage_path = r'D:\Files\CWI Winter Spring 2022\Data\DeepUS'
data_set = 'CIRS040GSE' # Pick CIRS073_RUMC or CIRS040GSE
experimental_data_path = join(storage_path, 'ExperimentalData', data_set)

# This is where the training data will be stored.
training_data_path_root = join(storage_path, 'TrainingData', data_set)

# Settings for reconstruction
# Sub-sampled number of steering angles used for reconstruction [1, ..., 75]
n_ang_ss = 1
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
    ang_ind = np.zeros(n_ang_ss)
    for i_ang in range(n_ang_ss):
        # Should only be scalar.
        ang_ind[i_ang] = np.argmin(np.abs(np.squeeze(usheader.xmitAngles)
                                          - ang_ss[i_ang])).squeeze()
    # Would check this clause one time, but not that important because only
    # 1 angle and 75 angles are used in my project.

usdata = usdata[:, :, ang_ind]
usheader.xmitAngles = np.squeeze(usheader.xmitAngles)[ang_ind]
usheader.xmitDelay = np.squeeze(usheader.xmitDelay)[ang_ind, :]
usheader.xmitFocus = np.squeeze(usheader.xmitFocus)[ang_ind]
usheader.xmitApodFunction = np.squeeze(usheader.xmitApodFunction)[ang_ind, :]

# Speed of sound used for calculations.
sos_bgn = np.squeeze(usheader.c)
# Used to correct for general delay of the acoustic lens.
lens_delay = 96

# Main loop