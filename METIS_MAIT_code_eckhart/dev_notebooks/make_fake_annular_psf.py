# %%
# Makes my own version of ScopeSim data, in order to debug my IMG OPT 03 psf quality
# test code. This uses two methods to make the PSF:
# 1. Using the "perfect" PSF (kernel) from ScopeSim and convolving with a pinhole
# 2. Using an annular aperture function and convolving with a pinhole

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve_fft
from scipy.ndimage import zoom, shift
import sys
from pathlib import Path

sys.path.insert(0, str(Path("..") / "main_scripts"))  # if cwd is dev_notebooks/
# or absolute:
# sys.path.insert(0, "/podman-share/metis_work/METIS_MAIT_code_eckhart/main_scripts")
from modules.helpers import intensity_annular_aperture
import ipdb


def oversample_2d(array_2d, factor):
    """
    Oversample a 2D array by ``factor`` using cubic spline interpolation.

    PARAMETERS
    ----------
    array_2d : ndarray
        2D input array.
    factor : float
        Linear oversampling factor (e.g. 4 → 4x finer sampling along each axis).

    RETURNS
    -------
    ndarray
        Oversampled 2D array of shape approximately
        ``(factor * ny, factor * nx)``.
    """
    array_2d = np.asarray(array_2d)
    if array_2d.ndim != 2:
        raise ValueError("array_2d must be 2D")
    if factor <= 0:
        raise ValueError("factor must be positive")
    return zoom(array_2d, factor, order=3)


def downsample_2d(array_2d, factor):
    """
    Downsample a 2D array by ``factor`` using cubic spline interpolation.

    The inverse of ``oversample_2d``: an array oversampled by N, then
    downsampled by N, returns to the original shape.

    PARAMETERS
    ----------
    array_2d : ndarray
        2D input array.
    factor : float
        Linear downsampling factor (e.g. 4 → 4x coarser sampling along each axis).

    RETURNS
    -------
    ndarray
        Downsampled 2D array of shape
        ``(round(ny / factor), round(nx / factor))``.
    """
    array_2d = np.asarray(array_2d)
    if array_2d.ndim != 2:
        raise ValueError("array_2d must be 2D")
    if factor <= 0:
        raise ValueError("factor must be positive")
    ny, nx = array_2d.shape
    out_ny = int(round(ny / factor))
    out_nx = int(round(nx / factor))
    if out_ny < 1 or out_nx < 1:
        raise ValueError("factor is too large for the array shape")
    return zoom(array_2d, (out_ny / ny, out_nx / nx), order=3)


def peak_yx_subpix(array_2d):
    """
    Subpixel peak (y, x) from a 1D quadratic fit through nanargmax along each axis.
    """
    array_2d = np.asarray(array_2d, dtype=float)
    y0, x0 = np.unravel_index(np.nanargmax(array_2d), array_2d.shape)

    def quad_offset(values):
        values = np.nan_to_num(values, nan=0.0)
        denom = values[0] - 2.0 * values[1] + values[2]
        if denom == 0 or not np.isfinite(denom):
            return 0.0
        delta = 0.5 * (values[0] - values[2]) / denom
        return float(np.clip(delta, -0.5, 0.5))

    ny, nx = array_2d.shape
    dy = quad_offset(array_2d[y0 - 1 : y0 + 2, x0]) if 0 < y0 < ny - 1 else 0.0
    dx = quad_offset(array_2d[y0, x0 - 1 : x0 + 2]) if 0 < x0 < nx - 1 else 0.0
    return np.array([y0 + dy, x0 + dx])


def shift_peak_to(array_2d, target_yx, n_pass=2):
    """
    Cubic-spline shift so the quadratic peak lands on ``target_yx`` (y, x).
    """
    out = np.asarray(array_2d, dtype=float)
    target_yx = np.asarray(target_yx, dtype=float)
    for _ in range(n_pass):
        out = shift(out, target_yx - peak_yx_subpix(out), order=3, mode="nearest")
    return out


# # Method 1: Use ScopeSim kernel
file_name_abs_kernel_scopesim = "/Users/eckhartspalding/Documents/git.repos/metis_work/METIS_MAIT_code_eckhart/data/kernels_scopesim/PSF_PPS-LM.fits"

# set the oversampling factor
oversampling_factor = 3

# set the wavelength
# N.b. Br-alpha is 4.052 um
wavelength_m = 4.09e-6  # wavelength of PSF [m]; see header

# read in the kernel
hdul = fits.open(file_name_abs_kernel_scopesim)
# hdul.info()

# 16  PSF_4.01um    1 ImageHDU        23   (256, 256)   float32
# 17  PSF_4.09um    1 ImageHDU        23   (256, 256)   float32
# 18  PSF_4.17um    1 ImageHDU        23   (256, 256)   float32
kernel_scopesim = hdul[17].data

# if kernel is still of even dimension, pad it with 1 pixel on each side to make it an odd number of pixels on each axis
if kernel_scopesim.shape[0] % 2 == 0:
    kernel_scopesim = np.pad(kernel_scopesim, ((1, 0), (0, 1)), mode="constant")
if kernel_scopesim.shape[1] % 2 == 0:
    kernel_scopesim = np.pad(kernel_scopesim, ((0, 1), (0, 1)), mode="constant")
# kernel_scopesim = np.pad(kernel_scopesim, ((1,0), (0,1)), mode='constant')

# define the pinhole
pinhole_diam_um = 25.0  # um (25 um for LM)
# Plate scale at WCU FP2.1 [mm/asec]; Table 2.6 in E-REP-MPIA-1203
plate_scale_mm_per_asec = 3.319
pinhole_diam_asec = pinhole_diam_um * 1e-3 / plate_scale_mm_per_asec

# detector plate scale of original array [mas/pix]
plate_scale_mas_per_pix = 5.47  # LM

# call the analytical annular PSF function
# import intensity_annular_aperture() from helpers.py

# define the annular aperture parameters
# from METIS_MAIT_code_eckhart.misc.debug_annular_aperture import D_obscuration
annulus_inner_diam_m = 12.8502
annulus_outer_diam_m = 34.6878


# shift by one pixel, to match the kernel
# annulus_analytical_psf = np.roll(annulus_analytical_psf, -1, axis=0)
# annulus_analytical_psf = np.roll(annulus_analytical_psf, -1, axis=1)

# upsample by a factor of 3
kernel_scopesim_up = oversample_2d(kernel_scopesim, factor=oversampling_factor)

# make the pinhole screen from upsampled kernel image
# Origin at (N-1)/2 so an odd axis has a pixel at the center (not N/2, which is between pixels).
ny_up, nx_up = kernel_scopesim_up.shape
center_yx = np.array([(ny_up - 1) / 2.0, (nx_up - 1) / 2.0])
xx, yy = np.meshgrid(
    np.arange(nx_up) - (nx_up - 1) / 2.0,
    np.arange(ny_up) - (ny_up - 1) / 2.0,
)
dist_from_center_pix = np.sqrt(xx**2 + yy**2)
dist_from_center_asec_up = dist_from_center_pix * (
    (plate_scale_mas_per_pix / oversampling_factor) / 1000.0
)

pinhole_up = np.zeros((ny_up, nx_up))
ipdb.set_trace()
pinhole_up[dist_from_center_asec_up < pinhole_diam_asec / 2] = 1
# distances in units of radians
dist_from_center_rad_up = dist_from_center_asec_up / 206265.0

# make the annular aperture function
annulus_analytical_psf_up = intensity_annular_aperture(
    r_rad_array=dist_from_center_rad_up,
    wavel=wavelength_m,
    D_obscuration=annulus_inner_diam_m,
    D_aperture=annulus_outer_diam_m,
    pinhole_diam_rad=None,
)

# Shift PSF peaks onto the center pixel. Do not spline-shift the binary pinhole;
# it is already centered by the (N-1)/2 origin.
annulus_analytical_psf_up = shift_peak_to(annulus_analytical_psf_up, center_yx)
kernel_scopesim_up = shift_peak_to(kernel_scopesim_up, center_yx)

print("center_yx:", center_yx)
print("kernel peak - center:", peak_yx_subpix(kernel_scopesim_up) - center_yx)
print(
    "analytical peak - center:", peak_yx_subpix(annulus_analytical_psf_up) - center_yx
)
print(
    "kernel peak - analytical peak:",
    peak_yx_subpix(kernel_scopesim_up) - peak_yx_subpix(annulus_analytical_psf_up),
)

kernel_up_n = kernel_scopesim_up / np.nanmax(kernel_scopesim_up)
annulus_up_n = annulus_analytical_psf_up / np.nanmax(annulus_analytical_psf_up)
"""
# debug
plt.clf()
plt.imshow(kernel_up_n - annulus_up_n)
plt.colorbar()
plt.title('Resids between kernel and analytical (oversampled)')
plt.show()
"""

# convolve the annular aperture function with the pinhole
# psf_analytical_conv_pinhole = convolve_fft(annulus_analytical_psf, pinhole)
psf_analytical_conv_pinhole_up = convolve_fft(annulus_analytical_psf_up, pinhole_up)
# convolve the pinhole screen with the kernel
# psf_kernel_conv_pinhole = convolve_fft(kernel_scopesim, pinhole)
psf_kernel_conv_pinhole_up = convolve_fft(kernel_scopesim_up, pinhole_up)

# sample back down
psf_analytical_conv_pinhole = downsample_2d(
    psf_analytical_conv_pinhole_up, factor=oversampling_factor
)
psf_kernel_conv_pinhole = downsample_2d(
    psf_kernel_conv_pinhole_up, factor=oversampling_factor
)
annulus_analytical_psf = downsample_2d(
    annulus_analytical_psf_up, factor=oversampling_factor
)
kernel_scopesim = downsample_2d(kernel_scopesim_up, factor=oversampling_factor)

# normalize for comparison
psf_analytical_conv_pinhole = psf_analytical_conv_pinhole / np.max(
    psf_analytical_conv_pinhole
)
kernel_scopesim = kernel_scopesim / np.max(kernel_scopesim)
annulus_analytical_psf = annulus_analytical_psf / np.max(annulus_analytical_psf)
psf_kernel_conv_pinhole = psf_kernel_conv_pinhole / np.max(psf_kernel_conv_pinhole)

plt.clf()
plt.imshow(pinhole_up)
plt.colorbar()
plt.title("Pinhole (upsampled)")
plt.show()

plt.clf()
plt.imshow(annulus_analytical_psf)
plt.colorbar()
plt.title("Analytical annulus PSF (native sampling)")
plt.show()

plt.clf()
plt.imshow(kernel_scopesim)
plt.colorbar()
plt.title("Scopesim kernel PSF (native sampling)")
plt.show()

plt.clf()
plt.imshow(kernel_scopesim - annulus_analytical_psf)
plt.colorbar()
plt.title("Resids between kernel and analytical (no pinhole) PSF (native sampling)")
plt.show()
# plt.savefig('junk_kernel_and_analytical_psf_2d_residuals.png')

plt.clf()
plt.imshow(psf_kernel_conv_pinhole - psf_analytical_conv_pinhole)
plt.colorbar()
plt.title("Resids between kernel and analytical (with pinhole) PSF (native sampling)")
plt.show()

plt.imshow(kernel_scopesim - psf_kernel_conv_pinhole)
plt.colorbar()
plt.title("Resids between kernel and (kernel * pinhole) (native sampling)")
plt.show()

plt.clf()
plt.imshow(annulus_analytical_psf - psf_analytical_conv_pinhole)
plt.colorbar()
plt.title("Resids between analytical and (analytical * pinhole) (native sampling)")
plt.show()

# make 1D plots through the centers of kernel_scopesim and annulus_analytical_psf
plt.clf()
# just plot the central 100 pixels
span = 50
idx1 = kernel_scopesim.shape[1] // 2 - int(span / 2)
idx2 = kernel_scopesim.shape[1] // 2 + int(span / 2)
plt.plot(kernel_scopesim[kernel_scopesim.shape[0] // 2, idx1:idx2])
plt.plot(annulus_analytical_psf[annulus_analytical_psf.shape[0] // 2, idx1:idx2])
plt.yscale("log")
plt.legend(["ScopeSim PSF_PPS-LM kernel", "analytical (annular aperture)"])
plt.title("Cross-section (native sampling; without pinholes)")
# plt.savefig('junk_kernel_and_analytical_psf_1d_comparison.png')
plt.show()

# make 1D plots through the centers of kernel_scopesim and annulus_analytical_psf
plt.clf()
# just plot the central N pixels
idx1 = kernel_scopesim.shape[1] // 2 - int(span / 2)
idx2 = kernel_scopesim.shape[1] // 2 + int(span / 2)
plt.plot(psf_kernel_conv_pinhole[psf_kernel_conv_pinhole.shape[0] // 2, idx1:idx2])
plt.plot(
    psf_analytical_conv_pinhole[psf_analytical_conv_pinhole.shape[0] // 2, idx1:idx2]
)
plt.yscale("log")
plt.legend(["kernel * pinhole", "analytical * pinhole"])
plt.title("Cross-section (native sampling; with pinholes)")
plt.show()

print("native kernel peak:", peak_yx_subpix(kernel_scopesim))
print("native analytical peak:", peak_yx_subpix(annulus_analytical_psf))
print(
    "native kernel - analytical peak:",
    peak_yx_subpix(kernel_scopesim) - peak_yx_subpix(annulus_analytical_psf),
)
