import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.modeling.models import AiryDisk2D
from astropy.visualization import ZScaleInterval
from scipy.signal import convolve2d
from scipy.optimize import curve_fit
import ipdb
from astropy.visualization import ZScaleInterval

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.ndimage import zoom

from modules.helpers import load_config_and_pipe
from modules.amoeba import amoeba_minimize
from modules.helpers import (
    angle_from_center_2d,
    model_for_fit_fixed,
    mtf_arrays
)

# simulate a PSF using an analytical expression
cookie_cut_out_sci_original = np.ones((101, 101))
baseline_shape_original = cookie_cut_out_sci_original.shape

# upsample
fac_oversamp = 1
cookie_cut_out_sci_oversamp = zoom(cookie_cut_out_sci_original, fac_oversamp, order=3)
baseline_shape_oversamp = cookie_cut_out_sci_oversamp.shape

# centers
y_center_cookie_original = cookie_cut_out_sci_original.shape[0] // 2
x_center_cookie_original = cookie_cut_out_sci_original.shape[1] // 2
y_center_cookie_oversamp = cookie_cut_out_sci_oversamp.shape[0] // 2
x_center_cookie_oversamp = cookie_cut_out_sci_oversamp.shape[1] // 2

# set parameters
config_observing = {'pixel_scales': {'img_lm': 5.47}}
D_aperture = 35
D_obscuration = 10
ampl = 1

# ersatz mask
valid_mask = np.ones(cookie_cut_out_sci_original.shape, dtype=bool)

# read in all the data states (one filter curve for each file)
stem = '/podman-share/metis_work/playing_with_scopesim/'
data_states_config_file = stem + 'config/config_file_IMG_04_strehl_runs.yaml'
data_states_config = load_config_and_pipe(config_file_choice=data_states_config_file, print_one_line=False)

# config file with the observing parameters
observing_config_file = stem + 'config/config_file_IMG_04_observing.yaml'
observing_config = load_config_and_pipe(config_file_choice=observing_config_file, print_one_line=False)


# make simulated data (no oversampling)
for state in data_states_config['runs']:

    filter_name = state['filter_name']
    filter_file = observing_config['polychromatic_observing_filters_lm'][filter_name]

    # get the central wavelength (for FYI purposes only, if using a polychromatic PSF)
    central_wavelength = observing_config['monochromatic_observing_filters_lm'][filter_name]

    pinhole_size = None

    r_rad_2d_original = angle_from_center_2d(array_passed_in=cookie_cut_out_sci_original, 
                            y_center=y_center_cookie_original, 
                            x_center=x_center_cookie_original, 
                            pixel_scale_mas=config_observing['pixel_scales']['img_lm'], 
                            fac_oversamp=fac_oversamp, 
                            units='radians')

    r_rad_1d_original = r_rad_2d_original.flatten()
    valid_mask_1d = valid_mask.flatten()

    data_empirical_1d_original = model_for_fit_fixed(
                r_rad_1d_original,
                D_aperture,
                D_obscuration,
                ampl,
                shape_original_2d=baseline_shape_original,
                shape_oversampled_2d=baseline_shape_oversamp,
                valid_mask=valid_mask_1d,
                fac_oversamp=fac_oversamp,
                #wavel=config_observing['observing_filters_lm'][filter_name],
                filter_file=filter_file
            )

    data_empirical_2d_original = data_empirical_1d_original.reshape(baseline_shape_original)
    data_empirical_2d_original_noconv_nonoise = np.copy(data_empirical_2d_original)
    ipdb.set_trace()

    # make a pinhole, if we're using one of finite size
    pinhole_size = 3e-8  # units rad
    if pinhole_size is not None:
        # pixel scale in radians (same units as r_rad_2d)
        rad_per_pix = (config_observing['pixel_scales']['img_lm'] / 1000.0) / 206265.0 / fac_oversamp
        # fractional pixels: linear ramp at boundary so edge pixels get value in (0, 1)
        pinhole_array = np.clip(
            (pinhole_size - r_rad_2d_original + rad_per_pix / 2) / rad_per_pix,
            0, 1
        )
        # convolve with the empirical data pre-noise
        data_empirical_2d_original_conv = convolve2d(data_empirical_2d_original, pinhole_array, mode='same')
        data_empirical_1d_original_conv = data_empirical_2d_original_conv.flatten()
    ipdb.set_trace()

    # renormalize
    data_empirical_2d_conv_norm = (data_empirical_2d_original_conv / np.sum(data_empirical_2d_original_conv)) * np.sum(data_empirical_2d_original_noconv_nonoise)
    # add some noise
    noise = np.random.normal(0, 0.002, data_empirical_2d_conv_norm.shape)
    data_empirical_2d_conv_norm += noise
    ipdb.set_trace()
    #data_empirical_2d = data_empirical_1d.reshape(baseline_shape)

    # make subplots of the empirical, pinhole, convolved data, and cross-sections
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))
    for ax in axs:
        ax.set_box_aspect(1)

    # Empirical
    
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(data_empirical_2d_conv_norm)
    im0 = axs[0].imshow(data_empirical_2d_original_noconv_nonoise, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
    axs[0].set_title('Perfect PSF, no pinhole')
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # Pinhole
    im1 = axs[1].imshow(pinhole_array, origin='lower', cmap='gray_r')
    axs[1].set_title('Pinhole')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # Convolved
    #vmin, vmax = zscale.get_limits(test)
    im2 = axs[2].imshow(data_empirical_2d_conv_norm, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
    axs[2].set_title('Empirical\n(Convolved, renormalized, noise added)')
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    # Cross-section
    mid_row = data_empirical_2d_original_noconv_nonoise.shape[0] // 2
    x_pixels = np.arange(data_empirical_2d_original_noconv_nonoise.shape[1])
    line_empirical, = axs[3].plot(x_pixels, data_empirical_2d_original_noconv_nonoise[mid_row, :], label='Perfect, no pinhole')
    line_convolved, = axs[3].plot(x_pixels, data_empirical_2d_conv_norm[mid_row, :], label='Convolved', linestyle='--')
    axs[3].set_yscale('log')
    axs[3].set_xlabel('Pixel')
    axs[3].set_ylabel('Intensity')
    axs[3].set_title('Cross-section (center row)')
    axs[3].legend()

    '''
    # Optionally, add a dummy colorbar for the 1D plot to be consistent (though not common)
    norm = Normalize(vmin=min(data_empirical_2d[mid_row, :].min(), data_empirical_2d_noconv_nonoise[mid_row, :].min()),
                    vmax=max(data_empirical_2d[mid_row, :].max(), data_empirical_2d_noconv_nonoise[mid_row, :].max()))
    sm = ScalarMappable(norm=norm, cmap='gray_r')
    plt.colorbar(sm, ax=axs[3], fraction=0.046, pad=0.04)
    '''
    plt.tight_layout()
    plt.show()

    #########################################################
    # fit it
    model_wrapper = lambda r_rad_1d, D_aperture, D_obscuration, ampl: \
            model_for_fit_fixed(
                r_rad_1d,
                D_aperture,
                D_obscuration,
                ampl,
                baseline_shape,
                valid_mask_1d,
                #wavel=config_observing['observing_filters_lm'][filter_name],
                filter_file=filter_file
            )

    lower_bounds = [25., 2.0, 1e-3]
    upper_bounds = [60.0, 20., 1e3]
    initial_guess = [30, 15, 2]
    popt, pcov = curve_fit(
        model_wrapper,
        r_rad_1d,
        data_empirical_1d,
        p0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        method='trf'  # Trust Region Reflective algorithm supports bounds
    )

    best_fit_2d = model_wrapper(r_rad_1d, popt[0], popt[1], popt[2]).reshape(baseline_shape)
    residuals = data_empirical_2d - best_fit_2d

    # plot the empirical, best-fit, residuals, and cross-sections (2x3 layout)
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    for ax in axs.flat:
        ax.set_box_aspect(1)

    im0 = axs[0, 0].imshow(data_empirical_2d, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)


    axs[0, 0].set_title('Empirical')
    axs[0, 0].set_xlabel('Pixel')
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

    im1 = axs[0, 1].imshow(best_fit_2d, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('Best-fit')
    axs[0, 1].set_xlabel('Pixel')
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    im2 = axs[0, 2].imshow(residuals, origin='lower', cmap='gray_r')
    axs[0, 2].set_title('Residuals')
    axs[0, 2].set_xlabel('Pixel')
    plt.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

    # cross-section through center row (bottom left)
    mid_row = data_empirical_2d.shape[0] // 2
    mid_col = data_empirical_2d.shape[1] // 2
    x_pixels = np.arange(data_empirical_2d.shape[1])
    axs[1, 0].plot(x_pixels, data_empirical_2d[mid_row, :], label='Empirical')
    axs[1, 0].plot(x_pixels, best_fit_2d[mid_row, :], label='Best-fit', linestyle='--')
    axs[1, 0].set_xlabel('Pixel')
    axs[1, 0].set_ylabel('Intensity')
    axs[1, 0].set_title('Cross-section (center row)')
    axs[1, 0].legend()

    # cross-section through center column (bottom middle)
    y_pixels = np.arange(data_empirical_2d.shape[0])
    axs[1, 1].plot(x_pixels, data_empirical_2d[mid_row, :], label='Empirical')
    axs[1, 1].plot(x_pixels, best_fit_2d[mid_row, :], label='Best-fit', linestyle='--')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_xlabel('Pixel')
    axs[1, 1].set_ylabel('Intensity')
    axs[1, 1].set_title('Cross-section (center row)')
    axs[1, 1].legend()

    axs[1, 2].axis('off')
    fig.suptitle(
        f"Residuals\n"
        f"Injected: D_aperture = {D_aperture:.2f}, D_obscuration = {D_obscuration:.2f}, ampl = {ampl:.2f}\n"
        f"Initial guess: D_aperture = {initial_guess[0]:.2f}, D_obscuration = {initial_guess[1]:.2f}, ampl = {initial_guess[2]:.2f}\n"
        f"Retrieved: D_aperture = {popt[0]:.2f} ± {np.sqrt(pcov[0,0]):.2f}, "
        f"D_obscuration = {popt[1]:.2f} ± {np.sqrt(pcov[1,1]):.2f}, "
        f"ampl = {popt[2]:.2f} ± {np.sqrt(pcov[2,2]):.2f}"
    )
    #fig.tight_layout()
    #plt.show()

    plt.savefig('debug_annular_aperture_' + state['filter_name'] + '.png')