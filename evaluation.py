"""Library file for evaluation"""

import torch
import numpy as np
import matplotlib.patches as patches
from typing import Literal, Tuple
import os.path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def model_output(model: torch.nn.Module,
                 msd_path: str,
                 input: torch.Tensor) -> torch.Tensor:
    """Infer specified model for output
    
    Arguments:
     - model: the model configuration for inference.
     - msd_path: path to model state dict after training.
     - input: input to pass through model.
    """
    model.load_state_dict(torch.load(msd_path, map_location='cuda:0'))
    model.eval()
    model_img = model(input).detach()
    # Rescaling to [0, 1].
    model_img = ((model_img - torch.min(model_img))
                 / (torch.max(model_img) - torch.min(model_img)))
    return model_img

def get_rois(img: torch.Tensor,
             roi_c: patches.Rectangle,
             roi_n: patches.Rectangle,
             nz_cutoff: int = 1500
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return sections of img corresponding to specified rectangles.
    
    Arguments:
     - img: Image with dimensions for model passthrough, that is including
        batch dimension.
     - roi_c: ROI for 'contrast' region.
     - roi_n: ROI for 'noise'/background region.
     - nz_cutoff: Maximum axial coordinate.
    """
    eval_img = torch.squeeze(img, 1)[0, :nz_cutoff, :].numpy()
    eval_img_roi_c = eval_img[roi_c.get_y():roi_c.get_y() + roi_c.get_height(),
                              roi_c.get_x():roi_c.get_x() + roi_c.get_width()]
    eval_img_roi_n = eval_img[roi_n.get_y():roi_n.get_y() + roi_n.get_height(),
                              roi_n.get_x():roi_n.get_x() + roi_n.get_width()]
    return eval_img_roi_c, eval_img_roi_n

def get_circular_rois(img: torch.Tensor,
                      c_ij: Tuple[int, int],
                      r1: float,
                      r2: Tuple[float, float],
                      pdeltaz: float = 2.4640e-2,
                      pdeltax: float = 1.9530e-1,
                      nz_cutoff: int = 1500):
    """Return masked img sections corresponding to specified disk and ring.
    
    Arguments:
     - img: Image with dimension for model passthrough.
     - c_ij: Center of disk and ring ROIs. Specify as pixel coordinate.
     - r1: Radius of inner disk, given in mm.
     - r2: Defines ring with inner radius r2[0] and outer radius r2[1],
        given in mm.
     - pdeltaz: Distance between pixel centers for axial dimension in mm.
     - pdeltax: Distance between pixel centers for lateral dimension in mm.
     - nz_cutoff: Maximum axial coordinate.
    """
    eval_img = torch.squeeze(img, 1)[0, :nz_cutoff, :].numpy()
    mdist = dist(img, c_ij, pdeltaz, pdeltax) # [mm]
    disk_mask = mdist <= r1 # [mm]
    ring_mask = mdist <= r2[1]
    ring_mask[mdist <= r2[0]] = 0
    eval_img_roi_c = eval_img[disk_mask]
    eval_img_roi_n = eval_img[ring_mask]
    return eval_img_roi_c, eval_img_roi_n

def get_roi(img: torch.Tensor,
            roi_c: patches.Rectangle,
            nz_cutoff: int = 1500):
    """Return section of image corresponding to specified rectangle.
    
    Single version of evalution.get_rois.
    """
    eval_img = torch.squeeze(img, 1)[0, :nz_cutoff, :].numpy()
    eval_img_roi_c = eval_img[roi_c.get_y():roi_c.get_y() + roi_c.get_height(),
                              roi_c.get_x():roi_c.get_x() + roi_c.get_width()]
    return eval_img_roi_c

def dist(img: torch.Tensor,
         c_ij: Tuple[int, int],
         pdeltaz: float = 2.4640e-2,
         pdeltax: float = 1.9530e-1,
         nz_cutoff: int = 1500):
    """Calculates distance in mm for each point in img to center point c_ij."""
    eval_img = torch.squeeze(img, 1)[0, :nz_cutoff, :].numpy()
    mdist = np.zeros_like(eval_img)
    for i in range(np.shape(eval_img)[0]):
        for j in range(np.shape(eval_img)[1]):
            mdist[i, j] = np.sqrt(((i - c_ij[0]) * pdeltaz)**2
                                  + ((j - c_ij[1]) * pdeltax)**2)
    return mdist

def psf_dim(dir_avg):
    """Calculate PSF in terms half-maximum distance around peak value."""
    hm = np.amin(dir_avg) + 0.5 * (np.amax(dir_avg) - np.amin(dir_avg))
    # Get the first point that is less than half maximum starting from
    # the maximum point moving to the sides.
    fwhm = (next(x for x in range(len(dir_avg)) if dir_avg[x] <= hm
                    and x >= np.argmax(dir_avg))
                - next(x for x in range(len(dir_avg) - 1, -1, -1)
                        if dir_avg[x] <= hm and x <= np.argmax(dir_avg)))
    return fwhm

def gauss_psf_dim(dir_avg, debug=False):
    """Calculate PSF after performing Gaussian fit on the data."""
    def gauss1(x, a1, b1, c1):
        return a1 * np.exp(-((x - b1) / c1)**2)
    popt, _ = curve_fit(gauss1, range(dir_avg.size),
                        dir_avg - min(dir_avg))
    
    if debug:
        # Check, if fit matches that which is expected.
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(dir_avg)
        ax2.plot(gauss1(range(dir_avg.size), popt[0], popt[1], popt[2]))
        plt.show()

    fwhm = 2 * np.sqrt(2 * np.log(2)) * np.sqrt(1/2) * popt[2]
    return np.abs(fwhm)

def psf(img: torch.Tensor,
        roi_ax: patches.Rectangle,
        roi_lat: patches.Rectangle,
        fit: bool = True,
        avg_mode: Literal['mean', 'sum'] = 'sum',
        units: Literal['physical', 'pixels'] = 'physical',
        pdeltaz: float = 2.4640e-2,
        pdeltax: float = 1.9530e-1,
        nz_cutoff: int = 1500
        ) -> Tuple[float, float]:
    """Calculates the PSF as the FWHM within the ROI.
    
    Arguments:
     - img: image which contains regions for psf calculation.
     - roi_ax: specified region of interest for axial psf calculation.
     - roi_lat: pecified region of interest for lateral psf calculation.
     - fit: flag to determine whether to do a Gaussian fit of the intensity
        curve before determining FWHM.
     - avg_mode: desired way to reduce reduce 2D rectangle ROI to average
        intensity curve in both lateral and axial dimension. Supported: mean,
        and sum.
     - units: string indicating the units in which the PSF should be returned.
        Supported: physical, pixels. The unit is determined by pdeltaz, and
        pdeltax.
     - pdeltaz, pdeltax: if units is a physical units of distance then these
        parameters describe the physical distance between the pixels. The
        default values are in mm for the current experiments.
     - nz_cutoff: maximum actual coordinate of passed img.
    """
    eval_img_roi_ax = get_roi(img, roi_ax, nz_cutoff)
    eval_img_roi_lat = get_roi(img, roi_lat, nz_cutoff)
    
    if avg_mode == 'sum':
        lat_avg, ax_avg = (np.sum(eval_img_roi_lat, 0),
                           np.sum(eval_img_roi_ax, 1))
    elif avg_mode == 'mean':
        lat_avg, ax_avg = (np.mean(eval_img_roi_lat, 0),
                           np.mean(eval_img_roi_ax, 1))
    else:
        raise ValueError(f'Unsupported avg_mode: {avg_mode}.')

    if fit:
        psf_ax = gauss_psf_dim(ax_avg)
        psf_lat = gauss_psf_dim(lat_avg)
    else:
        psf_ax = psf_dim(ax_avg)
        psf_lat = psf_dim(lat_avg)

    if units == 'pixel':
        pass
    elif units == 'physical':
        psf_ax, psf_lat = psf_ax * pdeltaz, psf_lat * pdeltax
    else:
        raise ValueError('Unsupported unit mode.')
    return psf_ax, psf_lat

def cnr(img: torch.Tensor,
        roi_c: patches.Rectangle = None,
        roi_n: patches.Rectangle = None,
        c_ij: Tuple[int, int] = None,
        r1: float = None,
        r2: Tuple[float, float] = None,
        pdeltaz: float = 2.4640e-2,
        pdeltax: float = 1.9530e-1,
        nz_cutoff: int = 1500
        ) -> float:
    """Calculate CNR in dB in img based on specified areas.
    
    Specify areas either as two rectangles roi_c and roi_n, or by a disk
    and ring with shared center point c_ij.
    """
    if roi_c is not None and roi_n is not None:
        eval_img_roi_c, eval_img_roi_n = get_rois(img, roi_c, roi_n, nz_cutoff)
    elif c_ij is not None and r1 is not None and r2 is not None:
        eval_img_roi_c, eval_img_roi_n = get_circular_rois(img, c_ij, r1, r2, pdeltaz=pdeltaz, pdeltax=pdeltax, nz_cutoff=nz_cutoff)
    else:
        raise ValueError('Insufficient parameters to get ROI.')
    cnr = 20 * np.log10(np.abs(np.mean(eval_img_roi_c)
                               - np.mean(eval_img_roi_n))
                        / np.sqrt((np.var(eval_img_roi_c)
                                  + np.var(eval_img_roi_n)) / 2))
    return cnr

def cr(img: torch.Tensor,
       roi_c: patches.Rectangle = None,
       roi_n: patches.Rectangle = None,
       c_ij: Tuple[int, int] = None,
       r1: float = None,
       r2: Tuple[float, float] = None,
       pdeltaz: float = 2.4640e-2,
       pdeltax: float = 1.9530e-1,
       nz_cutoff: int = 1500
       ) -> float:
    """Calculate CR in dB in img based on specified areas.
    
    Specify areas either as two rectangles roi_c and roi_n, or by a disk
    and ring with shared center point c_ij.
    """
    if roi_c is not None and roi_n is not None:
        eval_img_roi_c, eval_img_roi_n = get_rois(img, roi_c, roi_n, nz_cutoff)
    elif c_ij is not None and r1 is not None and r2 is not None:
        eval_img_roi_c, eval_img_roi_n = get_circular_rois(
            img, c_ij, r1, r2, pdeltaz=pdeltaz, pdeltax=pdeltax,
            nz_cutoff=nz_cutoff)
    else:
        raise ValueError('Insufficient parameters to get ROI.')
    cr = 20 * np.log10(np.mean(eval_img_roi_c) / np.mean(eval_img_roi_n))
    return cr

def snr(img: torch.Tensor,
        roi_n: patches.Rectangle = None,
        c_ij: Tuple[int, int] = None,
        r1: float = None,
        pdeltaz: float = 2.4640e-2,
        pdeltax: float = 1.9530e-1,
        nz_cutoff: int = 1500
        ) -> float:
    """Calculates SNR as done in DOI: 10.1109/TUFFC.2020.2982848.

    Can't just always use this SNR definition. Take care in selecting an
    appropriate 'noise' area for this.
    """
    eval_img = torch.squeeze(img, 1)[0, :nz_cutoff, :].numpy()
    if roi_n is not None:
        eval_img_roi_n = eval_img[roi_n.get_y():roi_n.get_y()
                                  + roi_n.get_height(),
                                  roi_n.get_x():roi_n.get_x()
                                  + roi_n.get_width()]
    elif c_ij is not None and r1 is not None:
        mdist = dist(img, c_ij, pdeltaz, pdeltax) # [mm]
        disk_mask = mdist <= r1 # [mm]
        eval_img_roi_n = eval_img[disk_mask]
    snr = np.mean(eval_img_roi_n) / np.std(eval_img_roi_n)
    return snr

def psnr(img: torch.Tensor,
         target: torch.Tensor,
         nz_cutoff: int = 1500
         ) -> float:
    """Calculates PSNR in dB of img with respect to target."""
    eval_img = torch.squeeze(img, 1)[0, :nz_cutoff, :].numpy()
    eval_target = torch.squeeze(target, 1)[0, :nz_cutoff, :].numpy()
    psnr = 20 * np.log10(np.amax(eval_target)
                         / np.sqrt(np.mean((eval_target - eval_img)**2)))
    return psnr

def gcnr(img: torch.Tensor,
         roi_c: patches.Rectangle = None,
         roi_n: patches.Rectangle = None,
         c_ij: Tuple[int, int] = None,
         r1: float = None,
         r2: Tuple[float, float] = None,
         pdeltaz: float = 2.4640e-2,
         pdeltax: float = 1.9530e-1,
         nz_cutoff: int = 1500
         ) -> float:
    """Calculates the gCNR as described in 10.1109/TUFFC.2019.2956855."""
    if roi_c is not None and roi_n is not None:
        eval_img_roi_c, eval_img_roi_n = get_rois(img, roi_c, roi_n, nz_cutoff)
    elif c_ij is not None and r1 is not None and r2 is not None:
        eval_img_roi_c, eval_img_roi_n = get_circular_rois(
            img, c_ij, r1, r2, pdeltaz=pdeltaz, pdeltax=pdeltax,
            nz_cutoff=nz_cutoff)
    else:
        raise ValueError('Insufficient parameters to get ROI.')
    # Tip from 10.1109/TUFFC.2021.3094849, to have to same bin sequence.
    _, bins = np.histogram(np.concatenate((eval_img_roi_c, eval_img_roi_n)),
                           bins=256)
    pd_roi_c, _ = np.histogram(eval_img_roi_c, bins=bins, density=True)
    pd_roi_n, _ = np.histogram(eval_img_roi_n, bins=bins, density=True)
    gcnr = 1 - np.sum(np.minimum(pd_roi_c, pd_roi_n) * np.diff(bins))
    return gcnr

def l1loss(img: torch.Tensor,
           target: torch.Tensor,
           nz_cutoff: int = 1500
           ) -> float:
    """Calculates the l1loss between the target and the img."""
    eval_img = torch.squeeze(img, 1)[0, :nz_cutoff, :].numpy()
    eval_target = torch.squeeze(target, 1)[0, :nz_cutoff, :].numpy()
    l1loss = np.mean(np.abs(eval_target - eval_img))
    return l1loss

def l2loss(img: torch.Tensor,
           target: torch.Tensor,
           nz_cutoff: int = 1500
           ) -> float:
    """Calculates the l2loss between the target and img."""
    eval_img = torch.squeeze(img, 1)[0, :nz_cutoff, :].numpy()
    eval_target = torch.squeeze(target, 1)[0, :nz_cutoff, :].numpy()
    l2loss = np.sqrt(np.mean((eval_target - eval_img)**2))
    return l2loss

def ncc(img: torch.Tensor,
        target: torch.Tensor,
        nz_cutoff: int = 1500
        ) -> float:
    """Calculates the sample normalized cross correlation."""
    eval_img = torch.squeeze(img, 1)[0, :nz_cutoff, :].numpy()
    eval_target = torch.squeeze(target, 1)[0, :nz_cutoff, :].numpy()
    ncc = (np.sum((eval_target - np.mean(eval_target)) 
                 * (eval_img - np.mean(eval_img)))
           / np.sqrt(np.sum((eval_target - np.mean(eval_target))**2)
                     * np.sum((eval_img - np.mean(eval_img))**2)))
    return ncc