import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.modeling.models import AiryDisk2D
from astropy.visualization import ZScaleInterval
from scipy.optimize import curve_fit
import ipdb

from .amoeba import amoeba_minimize
from .helpers import (
    angle_from_center_2d,
    model_for_fit_fixed,
    mtf_arrays,
)


def strehl_from_annular_aperture_fixed(cookie_cut_out_sci, data_empirical_original, filter_name, plot_string, x_center_final_cookie_oversamp, y_center_final_cookie_oversamp, config_observing, fac_oversamp, polychromatic=True):
    '''
    Calculate the Strehl ratio from an annular aperture.

    INPUTS:
    cookie_cut_out_sci: the empirical PSF
    data_empirical_original: the original empirical data
    filter_name: the name of the observing filter
    plot_string: the string to add to the plot file name
    x_center_final_cookie_oversamp: the x-center of the PSF (i.e., no more centroiding will be done here); in coordinates of the cookie cut-out
    y_center_final_cookie_oversamp: the y-center of the PSF; in coordinates of the cookie cut-out
    config_observing: the config object containing the observing parameters
    fac_oversamp: the oversampling factor
    polychromatic: whether to use a polychromatic PSF; if true, read in a filter curve; if false, use a single wavelength set in a config file

    OUTPUTS:
    strehl_results: a dictionary containing the Strehl ratios from the different methods
    '''

    logging.info('--------------------------------')
    logging.info('Calculating Strehl from annular aperture, with fixed aperture radii')

    r_rad_2d = angle_from_center_2d(array_passed_in=cookie_cut_out_sci, 
                    y_center=y_center_final_cookie_oversamp, 
                    x_center=x_center_final_cookie_oversamp, 
                    pixel_scale_mas=config_observing['pixel_scales']['img_lm'], 
                    fac_oversamp=fac_oversamp, 
                    units='radians')
                    
    test_empirical_2d = np.where(np.isnan(cookie_cut_out_sci), np.nanmedian(cookie_cut_out_sci), cookie_cut_out_sci)

    size = cookie_cut_out_sci.shape[0] 
    baseline_shape = (size, size)

    # Flatten both arrays first
    r_rad_1d_full = r_rad_2d.flatten()
    test_empirical_1d_full = test_empirical_2d.flatten()

    # Create a SINGLE mask for valid (non-NaN, finite) data points
    # Apply the SAME mask to both arrays to keep them aligned
    mask_valid = np.isfinite(test_empirical_1d_full) & np.isfinite(r_rad_1d_full)

    # Apply the SAME mask to both arrays
    r_rad_1d = r_rad_1d_full[mask_valid]
    test_empirical_1d = test_empirical_1d_full[mask_valid]

    valid_mask = mask_valid.copy()

    # generate a fixed model PSF (output in 1D, for vestigial reasons)
    if polychromatic:
        filter_file = config_observing['polychromatic_observing_filters_lm'][filter_name]
        logging.info(f'Making a polychromatic PSF for filter file: {filter_file}')
        intensity_1d_full_1d = model_for_fit_fixed(r_rad_1d, 
                                                    D_aperture=config_observing['D_aperture']['full'], 
                                                    D_obscuration=config_observing['D_aperture']['D_obscuration'], 
                                                    ampl=1, 
                                                    baseline_shape=baseline_shape, 
                                                    valid_mask=valid_mask, 
                                                    filter_file=filter_file)
    else: # monochromatic
        wavel_mono = config_observing['monochromatic_observing_filters_lm'][filter_name]
        logging.info(f'Making a monochromatic PSF for wavelength: {wavel_mono} um')
        intensity_1d_full_1d = model_for_fit_fixed(r_rad_1d, 
                                                    D_aperture=config_observing['D_aperture']['full'], 
                                                    D_obscuration=config_observing['D_aperture']['D_obscuration'], 
                                                    ampl=1, 
                                                    baseline_shape=baseline_shape, 
                                                    valid_mask=valid_mask, 
                                                    wavel=wavel_mono)
    model_annular_2d_full = intensity_1d_full_1d.reshape(baseline_shape)

    # normalize the model PSF to the empirical PSF, so that they have the same total power
    model_annular_2d_full_norm = (model_annular_2d_full / np.sum(model_annular_2d_full)) * np.sum(cookie_cut_out_sci)

    # make mask corresponding to first dark ring for an Airy (but not annular) aperture, so as to see how much power is in the central region
    # note this is just the first dark Airy ring for a monochromatic PSF, but this should be good enough for the polychromatic case as well
    dark_ring_loc_rad = 1.22 * (config_observing['monochromatic_observing_filters_lm'][filter_name] / config_observing['D_aperture']['full'])
    mask_central = r_rad_2d < dark_ring_loc_rad

    # strehl given the max values of the normalized model and empirical PSF
    strehl_from_fixed_annular_aperture_max = np.max(cookie_cut_out_sci) / np.max(model_annular_2d_full_norm)

    # strehl given the enclosed power in the central region
    model_annular_2d_full_norm_masked = model_annular_2d_full_norm * mask_central
    test_empirical_2d_masked = test_empirical_2d * mask_central
    strehl_from_fixed_annular_aperture_power_enclosed = np.sum(test_empirical_2d_masked) / np.sum(model_annular_2d_full_norm_masked)

    #test_factor = np.copy(strehl_from_fixed_annular_aperture_power_enclosed) # use this as a normalization factor downstream
    #ipdb.set_trace()

    ############################################################
    # another method: Fourier transform and use the ratios of the MTF
    # (note this masks based on max frequency, but not based on the dark rings in physical space)

    fft_model_power_cutoff, fft_empirical_power_cutoff, cutoff_freq, fx, fy, n_fft = mtf_arrays(array_empirical=cookie_cut_out_sci, array_model=model_annular_2d_full_norm, config_observing=config_observing, fac_oversamp=fac_oversamp, size=size, filter_name=filter_name)

    # normalize the powers so that power at zero freq is the same in both
    fft_model_power_cutoff_norm = fft_model_power_cutoff * np.nanmax(fft_empirical_power_cutoff) / np.nanmax(fft_model_power_cutoff)
    strehl_from_fixed_annular_aperture_mtf = np.sum(fft_empirical_power_cutoff) / np.sum(fft_model_power_cutoff_norm)
    #ipdb.set_trace()

    # plots subplots of the empirical, normalized model PSF, and residuals
    plt.clf()
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cookie_cut_out_sci, origin='lower', cmap='gray_r')
    plt.title('Empirical')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(model_annular_2d_full_norm, origin='lower', cmap='gray_r')
    plt.title('Normalized Model')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(cookie_cut_out_sci - model_annular_2d_full_norm, origin='lower', cmap='gray_r')
    plt.title('Residuals')
    plt.suptitle(
        f"Fixed annular aperture\n"
        f"Fixed: D_aperture={config_observing['D_aperture']['full']:.2f} m, "
        f"D_obscuration={config_observing['D_aperture']['D_obscuration']:.2f} m\n"
        f"Strehl from max: {strehl_from_fixed_annular_aperture_max:.2f}\n"
        f"Strehl from enclosed power: {strehl_from_fixed_annular_aperture_power_enclosed:.2f}"
    )
    plt.colorbar()
    #plt.show()
    file_name_plot = f'figs_dump/intensity_1d_full_2d_{plot_string}.png'
    plt.savefig(file_name_plot)
    logging.info(f'Saved {file_name_plot} to figs_dump/')

    # plot a cross-section through the FTs of the empirical and model PSFs
    plt.clf()
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 1, 1)
    # Restrict x-range to ±2 * cutoff_freq
    x_mask = (fx >= -2*cutoff_freq) & (fx <= 2*cutoff_freq)
    plt.plot(fx[x_mask], fft_empirical_power_cutoff[n_fft//2][x_mask], label='Empirical')
    plt.plot(fx[x_mask], fft_model_power_cutoff_norm[n_fft//2][x_mask], label='Model')
    plt.xlabel('Frequency (cycles per radian)')
    plt.ylabel('Power (units TBD)')
    plt.axvline(x=cutoff_freq, color='k', linestyle='--', label='Cutoff frequency', alpha=0.5)
    plt.axvline(x=-cutoff_freq, color='k', linestyle='--', alpha=0.5)
    plt.legend()
    plt.title(f'Cross-sections of MTFs\nStrehl from MTF: {strehl_from_fixed_annular_aperture_mtf:.2f}')
    file_name_plot = f'figs_dump/mtf_fixed_ann_ap_{plot_string}.png'
    plt.savefig(file_name_plot)
    logging.info(f'Saved {file_name_plot} to figs_dump/')

    logging.info(f"Strehl from fixed annular aperture, max: {strehl_from_fixed_annular_aperture_max}")
    logging.info(f"Strehl from fixed annular aperture, enclosed power: {strehl_from_fixed_annular_aperture_power_enclosed}")
    logging.info(f"Strehl from fixed annular aperture, MTF: {strehl_from_fixed_annular_aperture_mtf}")

    # strehls based on 
    # 1. max of the empirical and model PSFs
    # 2. enclosed power in the central region
    # 3. MTF
    # INSERT_YOUR_CODE
    strehl_results = {
        'strehl_fix_ann_ap_max': strehl_from_fixed_annular_aperture_max,
        'strehl_fix_ann_ap_pow': strehl_from_fixed_annular_aperture_power_enclosed,
        'strehl_fix_ann_ap_mtf': strehl_from_fixed_annular_aperture_mtf
    }
    return strehl_results


def fit_airy_psf(cookie_cut_out_sci, data_empirical_original, obs_filter, x_center_pix_gaussian_best_fit_oversamp, y_center_pix_gaussian_best_fit_oversamp, fac_oversamp, config_observing, plot_string=None):
    '''
    Generate an Airy PSF with the same total power as the empirical PSF, then compare the peak fluxes.  

    INPUTS:
    cookie_cut_out_sci: the empirical PSF
    data_empirical_original: the original empirical data
    obs_filter: the observing filter
    x_center_pix_gaussian_best_fit_oversamp: the x-center of the Gaussian-best-fit PSF
    y_center_pix_gaussian_best_fit_oversamp: the y-center of the Gaussian-best-fit PSF
    fac_oversamp: the oversampling factor
    config_observing: config object containing the observing parameters
    plot_string: the string to add to the plot file name

    OUTPUTS:
    total_power_empirical: the total power of the empirical PSF
    total_power_gaussian_best_fit: the total power of the Gaussian-best-fit PSF
    '''

    total_power_empirical = np.sum(cookie_cut_out_sci)
    # generate an Airy PSF with the same total power as the empirical PSF
    r_rad_2d = angle_from_center_2d(array_passed_in=cookie_cut_out_sci, 
                    y_center=y_center_pix_gaussian_best_fit_oversamp, 
                    x_center=x_center_pix_gaussian_best_fit_oversamp, 
                    pixel_scale_mas=config_observing['pixel_scales']['img_lm'], 
                    fac_oversamp=fac_oversamp, 
                    units='radians')
    # radius of first zero in pixel space of cookie_cut_out_sci
    rad_per_pix = ((config_observing['pixel_scales']['img_lm'] / 1000.0) / 206265.0) / fac_oversamp # radians per pixel, where pixels are those of the input array, not the detector
    radius_pix = (1.22 * (config_observing['observing_filters_lm'][obs_filter] / float(config_observing['D_aperture']['full']))) / rad_per_pix
    airy_model = AiryDisk2D(amplitude=1,
                                x_0=x_center_pix_gaussian_best_fit_oversamp, 
                                y_0=y_center_pix_gaussian_best_fit_oversamp, 
                                radius=radius_pix)
    yy, xx = np.mgrid[0:cookie_cut_out_sci.shape[0], 0:cookie_cut_out_sci.shape[1]]
    airy_psf = airy_model(xx, yy)

    # normalize the power to that of the empirical PSF
    airy_psf = (airy_psf / np.sum(airy_psf)) * total_power_empirical

    # compare the peak fluxes
    peak_flux_empirical = np.max(cookie_cut_out_sci) ## ## TO DO: MAKE THIS ROBUST TO BAD PIXELS
    peak_flux_airy = np.max(airy_psf)

    # make plots comparing the empirical and airy PSFs with log color scaling
    fig, axs = plt.subplots(1, 4, figsize=(23, 5), constrained_layout=True)  # use the axs array for all plotting

    padding_colorbars = 0.1
    # Empirical PSF
    im0 = axs[0].imshow(cookie_cut_out_sci, origin='lower', cmap='gray_r',
                       norm=LogNorm(vmin=np.maximum(np.nanmin(cookie_cut_out_sci[cookie_cut_out_sci > 0]), 1e-3),
                                    vmax=np.nanmax(cookie_cut_out_sci)))
    axs[0].set_title('Empirical PSF')
    axs[0].set_xlabel('Pixel')
    axs[0].set_ylabel('Pixel')
    divider0 = make_axes_locatable(axs[0])
    cax0 = divider0.append_axes("right", size="5%", pad=padding_colorbars)
    fig.colorbar(im0, cax=cax0)

    # Airy PSF
    #ipdb.set_trace(context=10)
    im1 = axs[1].imshow(airy_psf, origin='lower', cmap='gray_r',
                       norm=LogNorm(vmin=np.maximum(np.nanmin(airy_psf[airy_psf > 0]), 1e-3),
                                    vmax=np.nanmax(airy_psf)))
    axs[1].set_title('Airy PSF')
    axs[1].set_xlabel('Pixel')
    axs[1].set_ylabel('Pixel')
    divider1 = make_axes_locatable(axs[1])
    cax1 = divider1.append_axes("right", size="5%", pad=padding_colorbars)
    fig.colorbar(im1, cax=cax1)

    # Middle row cross-section
    mid_row = cookie_cut_out_sci.shape[0] // 2
    y_pixels = np.arange(cookie_cut_out_sci.shape[1])
    axs[2].plot(y_pixels, cookie_cut_out_sci[mid_row, :], label='Empirical', color='blue')
    axs[2].plot(y_pixels, airy_psf[mid_row, :], label='Model (Airy)', color='orange', linestyle='--')
    axs[2].set_title('Cross-section')
    axs[2].set_xlabel('Pixel')
    axs[2].set_ylabel('Counts')
    axs[2].legend()

    # Residuals
    im3 = axs[3].imshow(cookie_cut_out_sci - airy_psf, origin='lower', cmap='gray_r')
    axs[3].set_title('Residuals')
    axs[3].set_xlabel('Pixel')
    axs[3].set_ylabel('Pixel')
    divider3 = make_axes_locatable(axs[3])
    cax3 = divider3.append_axes("right", size="5%", pad=padding_colorbars)
    fig.colorbar(im3, cax=cax3)
    plot_filename = f'total_power_comparison_{plot_string}.png'
    #plt.show()
    plt.savefig(plot_filename)
    logging.info(f'Saved {plot_filename} to figs_dump/')

    strehl_airy_max = peak_flux_empirical / peak_flux_airy
    logging.info(f'Strehl from unobstructed circular aperture (-> Airy), max vals: {strehl_airy_max}')

    strehl_results = {
        'strehl_airy_max': strehl_airy_max
    }
    return strehl_results


def fit_annular_aperture_free_parameters(cookie_cut_out_sci, data_empirical_original, filter_name, plot_string, x_center_final_cookie_oversamp, y_center_final_cookie_oversamp, fac_oversamp, config_observing, fit_method):
    '''
    Fit a 2D analytical PSF to a given frame.

    INPUTS:
    cookie_cut_out_sci: 2D array of the cookie cut our from the full science frame
    data_empirical_original: the original empirical data
    filter_name: name of the observing filter
    plot_string: string to add to the plot file name
    x_center_final_cookie_oversamp: final x-center of the PSF (i.e., no more centroiding will be done here); in coordinates of the cookie cut-out
    y_center_final_cookie_oversamp: final y-center of the PSF; in coordinates of the cookie cut-out
    fac_oversamp: oversampling factor
    config_observing: config object containing the observing parameters
    fit_method: 'curve_fit' (default) or 'amoeba' - optimizer to use for finding best fit
    pinhole_size: size of the pinhole in pixels (if None, the analytical expression for the PSF alone is used; this is equivalent to a pinhole delta function)

    OUTPUTS:
    '''

    # make the cutout from the full array
    #psf_perfect_cutout = psf_perfect_oversamp[int(y_center_final_oversamp-0.5*cookie_cut_out_sci.shape[0]):int(y_center_final_oversamp+0.5*cookie_cut_out_sci.shape[0]), \
    #        int(x_center_final_oversamp-0.5*cookie_cut_out_sci.shape[1]):int(x_center_final_oversamp+0.5*cookie_cut_out_sci.shape[1])]

    r_rad_2d = angle_from_center_2d(array_passed_in=cookie_cut_out_sci, 
                        y_center=y_center_final_cookie_oversamp, 
                        x_center=x_center_final_cookie_oversamp, 
                        pixel_scale_mas=config_observing['pixel_scales']['img_lm'], 
                        fac_oversamp=fac_oversamp, 
                        units='radians')

    # replace nans with median
    test_empirical_2d = np.where(np.isnan(cookie_cut_out_sci), np.nanmedian(cookie_cut_out_sci), cookie_cut_out_sci)
    #test_perfect_2d = np.where(np.isnan(test_empirical_2d), np.nanmedian(test_empirical_2d), test_empirical_2d)

    # Flatten both arrays first
    r_rad_1d_full = r_rad_2d.flatten()
    test_empirical_1d_full = test_empirical_2d.flatten()

    # Create a SINGLE mask for valid (non-NaN, finite) data points
    # Apply the SAME mask to both arrays to keep them aligned
    mask = np.isfinite(test_empirical_1d_full) & np.isfinite(r_rad_1d_full)

    # Apply the SAME mask to both arrays
    r_rad_1d = r_rad_1d_full[mask]
    test_empirical_1d = test_empirical_1d_full[mask]

    valid_mask = mask.copy()

    logging.info(f"Original data points: {len(r_rad_1d_full)}")
    logging.info(f"Valid data points after masking: {len(r_rad_1d)}")
    logging.info(f"Arrays are aligned: {len(r_rad_1d) == len(test_empirical_1d)}")

    # Initial parameter guesses
    # [D_aperture, D_obscuration, ampl]
    initial_guess = [36., 12., 7e5]
    #initial_guess = [36., 12., 1e5]  

    # Create a wrapper function that binds the fixed parameters (baseline_shape and valid_mask)
    # This ensures curve_fit only optimizes D_aperture, D_obscuration, and ampl
    size = cookie_cut_out_sci.shape[0] 
    baseline_shape = (size, size)
    pinhole_size = 1e-8  # units radians (fixed, not a fit parameter)
    #pinhole_size = None
    model_wrapper = lambda r_rad_1d, D_aperture, D_obscuration, ampl: \
        model_for_fit_fixed(
            r_rad_1d,
            D_aperture,
            D_obscuration,
            ampl,
            baseline_shape,
            valid_mask,
            filter_file=config_observing['polychromatic_observing_filters_lm'][filter_name],
            pinhole_size=pinhole_size
            )

    # Set bounds for parameters: [D_aperture, D_obscuration, ampl]
    lower_bounds = [25., 2.0, 10.0]
    upper_bounds = [60.0, 20., 1e6]
    

    if fit_method == 'amoeba':
        # Use Nelder-Mead simplex (amoeba) - objective is sum of squared residuals
        logging.info('Fitting PSF with amoeba algorithm')
        def chi_sq(params):
            model = model_wrapper(r_rad_1d, params[0], params[1], params[2])
            return np.sum((model - test_empirical_1d) ** 2)

        popt, fopt = amoeba_minimize(
            chi_sq,
            initial_guess,
            delta=[0.01, 0.01, 500.0],  # step sizes for D_aperture, D_obscuration, ampl
            ftol=1e-8,
            nmax=50000,
            bounds=(lower_bounds, upper_bounds),
        )
        pcov = None  # amoeba does not provide covariance
        logging.info(f"Fit method: amoeba (Nelder-Mead), chi_sq = {fopt:.2f}")
    elif fit_method == 'curve_fit':
        # Perform the fit with curve_fit (Trust Region Reflective)
        logging.info('Fitting PSF with curve_fit algorithm')
        popt, pcov = curve_fit(
            model_wrapper,
            r_rad_1d,
            test_empirical_1d,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            method='trf'  # Trust Region Reflective algorithm supports bounds
        )

    # Extract best-fit parameters and uncertainties
    D_aperture_fit = popt[0]
    D_obscuration_fit = popt[1]
    ampl_fit = popt[2]

    # Calculate parameter uncertainties from covariance matrix (curve_fit only)
    if pcov is not None:
        param_errors = np.sqrt(np.diag(pcov))
        D_aperture_err = param_errors[0]
        D_obscuration_err = param_errors[1]
        ampl_err = param_errors[2]
    else:
        D_aperture_err = D_obscuration_err = ampl_err = np.nan

    # Print results
    logging.info('--------------------------------')
    logging.info("Fixed observing parameters, annular aperture:")
    logging.info(f"filter: {filter_name}, λ={config_observing['monochromatic_observing_filters_lm'][filter_name]*1e6:.2f}μm, ps={config_observing['pixel_scales']['img_lm']:.2f}mas", )
    logging.info('--------------------------------')
    logging.info("Best-fit parameters, annular aperture:")
    logging.info(f"D_aperture = {D_aperture_fit:.2f} ± {D_aperture_err:.2f} meters")
    logging.info(f"D_obscuration = {D_obscuration_fit:.2f} ± {D_obscuration_err:.2f} meters")
    logging.info(f"ampl = {ampl_fit:.2f} ± {ampl_err:.2f}")

    # Check if covariance matrix has infs (curve_fit only)
    if pcov is not None:
        if np.any(np.isinf(pcov)):
            logging.warning("WARNING: Covariance matrix contains infinities!")
            logging.warning("This usually means the fit didn't converge properly.")
        else:
            logging.info("Covariance matrix is finite - fit MAY have converged successfully")

    # generate the best-fit model based on the fit parameters
    # Note: r_rad_2d needs to be flattened and masked for model_for_fit_fixed
    r_rad_1d = r_rad_2d.flatten()
    r_rad_1d_masked = r_rad_1d[valid_mask]
    
    initial_guess_model_1d = model_for_fit_fixed(
        r_rad_1d_masked,
        initial_guess[0],
        initial_guess[1],
        initial_guess[2],
        baseline_shape,
        valid_mask,
        filter_file=config_observing['polychromatic_observing_filters_lm'][filter_name],
        pinhole_size=pinhole_size,
        fac_oversamp=fac_oversamp
    )
    best_fit_model_1d = model_for_fit_fixed(
        r_rad_1d_masked,
        D_aperture_fit,
        D_obscuration_fit,
        ampl_fit,
        baseline_shape,
        valid_mask,
        filter_file=config_observing['polychromatic_observing_filters_lm'][filter_name],
        pinhole_size=pinhole_size,
        fac_oversamp=fac_oversamp
    )
    
    # Reshape to 2D (though these are already 1D masked arrays, we need to reconstruct the full 2D)
    # Actually, model_for_fit_fixed returns masked 1D, so we need to reconstruct the full 2D array
    initial_guess_model_2d_full = np.full(baseline_shape, np.nan).flatten()
    initial_guess_model_2d_full[valid_mask] = initial_guess_model_1d
    initial_guess_model_2d = initial_guess_model_2d_full.reshape(baseline_shape)
    
    best_fit_model_2d_full = np.full(baseline_shape, np.nan).flatten()
    best_fit_model_2d_full[valid_mask] = best_fit_model_1d
    best_fit_model_2d = best_fit_model_2d_full.reshape(baseline_shape)

    # Calculate chi-squared
    # Both test_empirical_1d and best_fit_model_1d are already masked 1D arrays
    chi_squared = np.sum((test_empirical_1d - best_fit_model_1d)**2 / (0.01**2))  # assuming noise std = 0.01
    dof = len(test_empirical_1d) - 3  # degrees of freedom (data points - number of parameters)
    reduced_chi_squared = chi_squared / dof

    # best_fit_model_2d is already created above

    logging.info(f"\nChi-squared = {chi_squared:.2f}")
    logging.info(f"Degrees of freedom = {dof}")
    logging.info(f"Reduced chi-squared = {reduced_chi_squared:.6f}")

    ############################################################
    # Find the Strehl from the MTF, like in fit_annular_aperture_fixed
    fft_model_power_cutoff, fft_empirical_power_cutoff, cutoff_freq, fx, fy, n_fft = mtf_arrays(array_empirical=cookie_cut_out_sci, array_model=best_fit_model_2d, config_observing=config_observing, fac_oversamp=fac_oversamp, size=size, filter_name=filter_name)

    # normalize the powers so that zero freq is equal
    fft_model_power_cutoff_norm = (
        fft_model_power_cutoff
        * np.nanmax(fft_empirical_power_cutoff)
        / np.nanmax(fft_model_power_cutoff)
    )
    strehl_from_free_annular_aperture_mtf = np.sum(fft_empirical_power_cutoff) / np.sum(fft_model_power_cutoff_norm)
    logging.info(f"Strehl from free annular aperture, MTF: {strehl_from_free_annular_aperture_mtf}")

    # plot a cross-section through the FTs of the empirical and model PSFs
    plt.clf()
    plt.figure(figsize=(30, 5))
    plt.subplot(1, 1, 1)
    x_mask = (fx >= -2 * cutoff_freq) & (fx <= 2 * cutoff_freq)
    plt.plot(fx[x_mask], fft_empirical_power_cutoff[n_fft // 2][x_mask], label='Empirical')
    plt.plot(fx[x_mask], fft_model_power_cutoff_norm[n_fft // 2][x_mask], label='Model')
    plt.xlabel('Frequency (cycles per radian)')
    plt.ylabel('Power (units TBD)')
    plt.axvline(x=cutoff_freq, color='k', linestyle='--', label='Cutoff frequency', alpha=0.5)
    plt.axvline(x=-cutoff_freq, color='k', linestyle='--', alpha=0.5)
    plt.legend()
    plt.title(f'Cross-sections of MTFs\nStrehl from MTF: {strehl_from_free_annular_aperture_mtf:.2f}')
    file_name_plot = f'figs_dump/mtf_free_ann_ap_{plot_string}.png'
    plt.savefig(file_name_plot)
    logging.info(f'Saved {file_name_plot} to figs_dump/')
    plt.close()

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(test_empirical_2d)

    fig, axs = plt.subplots(3, 2, figsize=(20, 15), constrained_layout=True,
                            gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1, 1]})
    for ax in axs.flat:
        ax.set_box_aspect(1)

    # Panel 1: Empirical data
    im0 = axs[0,0].imshow(test_empirical_2d, vmin=vmin, vmax=vmax)
    axs[0,0].set_title("Empirical")

    # Panel 2: Best fit
    im1 = axs[0,1].imshow(best_fit_model_2d, vmin=vmin, vmax=vmax)
    axs[0,1].set_title("Best fit")

    # Panel 3: Cross-section between empirical and best-fit PSF
    center_y, center_x = np.array(test_empirical_2d.shape) // 2
    cross_empirical = test_empirical_2d[center_y, :]
    cross_best_fit = best_fit_model_2d[center_y, :]
    axs[1,0].plot(cross_empirical, label="Empirical")
    axs[1,0].plot(cross_best_fit, label="Best fit")
    axs[1,0].set_title("Cross-section")
    axs[1,0].legend()

    # Panel 3: Cross-section between empirical and best-fit PSF
    axs[1,1].plot(cross_empirical, label="Empirical")
    axs[1,1].plot(cross_best_fit, label="Best fit")
    axs[1,1].set_yscale('log')
    axs[1,1].set_title("Cross-section")
    axs[1,1].legend()

    # Panel 3: Initial guess
    im2 = axs[2,0].imshow(initial_guess_model_2d, vmin=vmin, vmax=vmax)
    axs[2,0].set_title("Initial guess")

    # Panel 4: Residuals
    residuals = test_empirical_2d - best_fit_model_2d
    im2 = axs[2,1].imshow(residuals, vmin=vmin, vmax=vmax)
    axs[2,1].set_title("Empirical - Best fit")

    # degbug: write FITS file
    #fits.writeto(f'junk_resids.fits', residuals, overwrite=True)


    plt.suptitle(
        f"Filter: {filter_name}, λ={config_observing['monochromatic_observing_filters_lm'][filter_name]*1e6:.2f}μm, pix={config_observing['pixel_scales']['img_lm']:.2f}mas, \n"
        f"Best fits: D_aper={D_aperture_fit:.2f}±{D_aperture_err:.2f}m, "
        f'D_obsc={D_obscuration_fit:.2f}±{D_obscuration_err:.2f}m, '
        f'Amp={ampl_fit:.2f}±{ampl_err:.2f}',
        fontsize=10
    )


    # Add one colorbar for all
    fig.colorbar(im0, ax=axs, orientation='vertical', fraction=0.04, pad=0.04).set_label('Color scale is the same')


    file_name_plot = f'figs_dump/free_ann_ap_best_fit_{plot_string}.png'
    plt.savefig(file_name_plot)
    logging.info(f"Saved {file_name_plot} to figs_dump/")
    plt.close()

    # return dict of Strehl ratios found with different methods
    strehl_results = {
        'strehl_free_ann_ap_mtf': strehl_from_free_annular_aperture_mtf
    }

    return strehl_results