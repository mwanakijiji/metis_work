import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")  # put this before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.modeling.models import AiryDisk2D
from astropy.visualization import ZScaleInterval
from scipy.optimize import curve_fit
from scipy.ndimage import zoom
import ipdb
from skimage.measure import block_reduce

from .amoeba import amoeba_minimize
from .helpers import (
    angle_from_center_2d,
    model_for_fit_fixed,
    mtf_arrays,
)

def fit_airy_psf(cookie_cut_out_sci, data_empirical_original, obs_filter, x_center_pix_gaussian_best_fit_oversamp, y_center_pix_gaussian_best_fit_oversamp, fac_oversamp, config_observing, plot_string=None):
    '''
    Strehl based on Airy PSF fit with the same total power as the empirical PSF

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


def fit_annular_aperture_fixed_parameters(cookie_cut_out_sci_oversamp, data_cookie_empirical_original, filter_name, plot_string, 
        x_center_2nd_pass_cookie_oversamp, y_center_2nd_pass_cookie_oversamp, config_observing, fac_oversamp, polychromatic=True):
    '''
    Strehl based on PSF fit, using model with fixed aperture dimensions

    INPUTS:
    cookie_cut_out_sci_oversamp: the empirical PSF (oversampled)
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

    r_rad_2d_oversamp = angle_from_center_2d(array_passed_in=cookie_cut_out_sci_oversamp, 
                    y_center=y_center_2nd_pass_cookie_oversamp, 
                    x_center=x_center_2nd_pass_cookie_oversamp, 
                    pixel_scale_mas=config_observing['pixel_scales']['img_lm'], 
                    fac_oversamp=fac_oversamp, 
                    units='radians')
                    
    # replace nans in cookie cut-out
    nonan_empirical_2d_oversamp = np.where(np.isnan(cookie_cut_out_sci_oversamp), np.nanmedian(cookie_cut_out_sci_oversamp), cookie_cut_out_sci_oversamp)

    shape_oversamp = cookie_cut_out_sci_oversamp.shape

    # Flatten both arrays first
    r_rad_1d_oversamp = r_rad_2d_oversamp.flatten()
    test_empirical_1d_oversamp = nonan_empirical_2d_oversamp.flatten()

    # Create a SINGLE mask for valid (non-NaN, finite) data points
    # Apply the SAME mask to both arrays to keep them aligned
    mask_valid = np.isfinite(test_empirical_1d_oversamp) & np.isfinite(r_rad_1d_oversamp)

    # Apply the SAME mask to both arrays
    r_rad_1d_oversamp_masked = r_rad_1d_oversamp[mask_valid]
    test_empirical_1d_oversamp_masked = test_empirical_1d_oversamp[mask_valid]

    # to avoid interfering updates downstream
    valid_mask = mask_valid.copy()

    # generate a fixed model PSF (output in 1D, for vestigial reasons); note this involves oversampled images
    if polychromatic:
        filter_file = config_observing['polychromatic_observing_filters_lm'][filter_name]
        logging.info(f'Making a polychromatic PSF for filter file: {filter_file}')
        intensity_1d_full_1d = model_for_fit_fixed(r_rad_1d_oversamp_masked, 
                                                    D_aperture=config_observing['D_aperture']['full'], 
                                                    D_obscuration=config_observing['D_aperture']['D_obscuration'], 
                                                    ampl=1, 
                                                    centroid_yx_oversamp=(y_center_2nd_pass_cookie_oversamp, x_center_2nd_pass_cookie_oversamp),
                                                    shape_oversamp=shape_oversamp, 
                                                    pixel_scale_mas=config_observing['pixel_scales']['img_lm'],
                                                    fac_oversamp=fac_oversamp,
                                                    valid_mask=valid_mask, 
                                                    filter_file=filter_file, 
                                                    save_fyi_plot=True)
    else: # monochromatic
        wavel_mono = config_observing['monochromatic_observing_filters_lm'][filter_name]
        logging.info(f'Making a monochromatic PSF for wavelength: {wavel_mono} um')
        intensity_1d_full_1d = model_for_fit_fixed(r_rad_1d_oversamp_masked, 
                                                    D_aperture=config_observing['D_aperture']['full'], 
                                                    D_obscuration=config_observing['D_aperture']['D_obscuration'], 
                                                    ampl=1, 
                                                    shape_oversamp=shape_oversamp, 
                                                    pixel_scale_mas=config_observing['pixel_scales']['img_lm'],
                                                    fac_oversamp=fac_oversamp,
                                                    valid_mask=valid_mask, 
                                                    wavel=wavel_mono)
    model_annular_2d_oversamp = intensity_1d_full_1d.reshape(shape_oversamp)

    # normalize the model PSF to the empirical PSF, so that they have the same total power
    model_annular_2d_oversamp_norm = (model_annular_2d_oversamp / np.sum(model_annular_2d_oversamp)) * np.sum(cookie_cut_out_sci_oversamp)

    

    # make mask corresponding to first dark ring for an Airy (but not annular) aperture, so as to see how much power is in the central region
    # note this is just the first dark Airy ring for a monochromatic PSF, but this should be good enough for the polychromatic case as well
    dark_ring_loc_rad = 1.22 * (config_observing['monochromatic_observing_filters_lm'][filter_name] / config_observing['D_aperture']['full'])
    mask_central = r_rad_2d_oversamp < dark_ring_loc_rad

    # strehl given the max values of the normalized model and empirical PSF
    # downsample the model PSF to the native scale
    model_annular_2d_oversamp_norm_native = block_reduce(
        model_annular_2d_oversamp_norm,
        block_size=(fac_oversamp, fac_oversamp),
        func=np.mean,
    )

    # take the max values of the empirica and model PSFs, at the native detector scale
    strehl_from_fixed_annular_aperture_max = np.max(data_cookie_empirical_original) / np.max(model_annular_2d_oversamp_norm_native)

    # strehl given the enclosed power in the central region (note this uses the oversampled model PSF; is that the best way?)
    model_annular_2d_full_norm_masked = model_annular_2d_oversamp_norm * mask_central
    test_empirical_2d_masked = nonan_empirical_2d_oversamp * mask_central
    strehl_from_fixed_annular_aperture_power_enclosed = np.sum(test_empirical_2d_masked) / np.sum(model_annular_2d_full_norm_masked)
    #test_factor = np.copy(strehl_from_fixed_annular_aperture_power_enclosed) # use this as a normalization factor downstream
    #ipdb.set_trace()

    ############################################################
    # another method: Fourier transform and use the ratios of the MTF
    # (note this masks based on max frequency, but not based on the dark rings in physical space)

    # Compute the Modulation Transfer Function (MTF) arrays for both empirical and model PSFs
    (
        fft_model_power_cutoff,
        fft_empirical_power_cutoff,
        cutoff_freq,
        fx,
        fy,
        n_fft
    ) = mtf_arrays(
        array_empirical=cookie_cut_out_sci_oversamp,
        array_model=model_annular_2d_oversamp_norm,
        config_observing=config_observing,
        fac_oversamp=fac_oversamp,
        size=shape_oversamp[0],
        filter_name=filter_name
    )

    # normalize the powers so that power at zero freq is the same in both
    fft_model_power_cutoff_norm = fft_model_power_cutoff * np.nanmax(fft_empirical_power_cutoff) / np.nanmax(fft_model_power_cutoff)
    strehl_from_fixed_annular_aperture_mtf = np.sum(fft_empirical_power_cutoff) / np.sum(fft_model_power_cutoff_norm)
    #ipdb.set_trace()

    # plots subplots of the empirical, normalized model PSF, and residuals
    plt.clf()
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cookie_cut_out_sci_oversamp, origin='lower', cmap='gray_r')
    plt.title('Empirical')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(model_annular_2d_oversamp_norm, origin='lower', cmap='gray_r')
    plt.title('Normalized Model')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(cookie_cut_out_sci_oversamp - model_annular_2d_oversamp_norm, origin='lower', cmap='gray_r')
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

    strehl_results = {
        'strehl_fix_ann_ap_max': strehl_from_fixed_annular_aperture_max,
        'strehl_fix_ann_ap_pow': strehl_from_fixed_annular_aperture_power_enclosed,
        'strehl_fix_ann_ap_mtf': strehl_from_fixed_annular_aperture_mtf
    }
    return strehl_results


def fit_annular_aperture_free_parameters(cookie_cut_out_sci_oversamp, cookie_cut_out_sci_original, filter_name, plot_string, 
        x_center_final_cookie_oversamp, y_center_final_cookie_oversamp, fac_oversamp, config_observing, fit_method, pinhole_diam_rad=1e-8):
    '''
    Fit a 2D analytical PSF to a given frame.

    INPUTS:
    cookie_cut_out_sci: 2D array of the cookie cut our from the full science frame
    data_cookie_empirical_original: the original empirical data (the cookie cut-out array)
    filter_name: name of the observing filter
    plot_string: string to add to the plot file name
    x_center_final_cookie_oversamp: final x-center of the PSF (i.e., no more centroiding will be done here); in coordinates of the cookie cut-out
    y_center_final_cookie_oversamp: final y-center of the PSF; in coordinates of the cookie cut-out
    fac_oversamp: oversampling factor
    config_observing: config object containing the observing parameters
    fit_method: 'curve_fit' (default) or 'amoeba' - optimizer to use for finding best fit
    pinhole_diam_rad: size of the pinhole in pixels (if None, the analytical expression for the PSF alone is used; this is equivalent to a pinhole delta function); units radians

    OUTPUTS:
    '''

    # Build the model directly on the oversampled grid using the fitted
    # oversampled centroid, then rebin once to compare on the native grid.
    nonan_empirical_2d_oversamp = np.where(np.isnan(cookie_cut_out_sci_oversamp), np.nanmedian(cookie_cut_out_sci_oversamp), cookie_cut_out_sci_oversamp)

    # flatten empirical image (note that this has never been resampled)
    data_empirical_original_1d = cookie_cut_out_sci_original.flatten()
    shape_original_2d = cookie_cut_out_sci_original.shape
    shape_oversampled_2d = nonan_empirical_2d_oversamp.shape
    dummy_x_native = np.arange(data_empirical_original_1d.size, dtype=float)
    dummy_x_oversamp = np.arange(np.prod(shape_oversampled_2d), dtype=float)

    logging.info(f"Original data points: {dummy_x_native.size}")
    logging.info(f"Oversampled data points: {dummy_x_oversamp.size}")

    save_fyi_plot_state = {"done": False}

    def model_wrapper(xdata, D_aperture, D_obscuration, ampl):
        save_fyi_plot = not save_fyi_plot_state["done"]
        save_fyi_plot_state["done"] = True
        return model_for_fit_fixed(
            xdata,
            D_aperture,
            D_obscuration,
            ampl,
            centroid_yx_oversamp=(y_center_final_cookie_oversamp, x_center_final_cookie_oversamp),
            shape_original_2d=shape_original_2d,
            fac_oversamp=fac_oversamp,
            pixel_scale_mas=config_observing['pixel_scales']['img_lm'],
            filter_file=config_observing['polychromatic_observing_filters_lm'][filter_name],
            pinhole_diam_rad=pinhole_diam_rad,
            save_fyi_plot=save_fyi_plot,
        )

    # Initial parameter guesses
    # [D_aperture, D_obscuration, ampl]
    initial_guess = [35., 13., 1.2e6]
    #initial_guess = [36., 12., 1e5] 
    # Set bounds for parameters: [D_aperture, D_obscuration, ampl]
    lower_bounds = [25., 2.0, 10.0]
    upper_bounds = [60.0, 20., 2e6]
    

    if fit_method == 'amoeba':
        # Use Nelder-Mead simplex (amoeba) - objective is sum of squared residuals
        logging.info('Fitting PSF with amoeba algorithm')
        def chi_sq(params):
            model = model_wrapper(dummy_x_native, params[0], params[1], params[2])
            return np.sum((model - data_empirical_original_1d) ** 2)

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
        # Perform the fit with curve_fit
        logging.info('Fitting PSF with curve_fit algorithm')

        #########################################################
        # Begin Example call to test model_for_fit_fixed in one line
        # loop over a few test pinhole sizes
        '''
        for pinhole_diam_rad_test in [None, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]:
            test = model_for_fit_fixed(
                r_rad_1d_original=r_rad_1d_original, 
                D_aperture=34.6878,             # D_aperture [example value in meters]
                D_obscuration=12.8502,              # D_obscuration [example value in meters]
                ampl=1.2e6,              # ampl [example value]
                centroid_yx_original=(y_center_final_cookie_oversamp,x_center_final_cookie_oversamp),
                shape_original_2d=shape_original_2d,
                fac_oversamp=fac_oversamp,
                filter_file=config_observing['polychromatic_observing_filters_lm'][filter_name],
                pinhole_diam_rad=pinhole_diam_rad_test
            )

            test_2d = test.reshape(shape_original_2d)
            plt.imshow(test_2d)
            plt.savefig(f'Figure_1_test_model_pinhole_{pinhole_diam_rad_test}.png')
            print(f'Saved Figure_1_test_model_pinhole_{pinhole_diam_rad_test}.png')
            plt.close()
            '''

        # End example call
        #########################################################
        plate_scale_wcu_fp2pt1 = 3.319 # [mm/asec]; Table 2.6 in E-REP-MPIA-1203, 'METIS WCU radiometric model'
        pinhole_diam_um = 25. # 25 um for LM, 66 um for N
        # convert to radians
        pinhole_diam_asec = pinhole_diam_um * 10**-3 / plate_scale_wcu_fp2pt1
        pinhole_diam_rad = pinhole_diam_asec * (1./206265.) # [rad]; 3.2e-8 is pretty close, with debugged centering; was 1.5e-6 previously
        popt, pcov = curve_fit(
            model_wrapper,
            xdata = dummy_x_native,
            ydata = data_empirical_original_1d,
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

    # generate the best-fit model based on the fit parameters (same r_rad grid as curve_fit)
    initial_guess_model_1d = model_for_fit_fixed(
        dummy_x_native,
        initial_guess[0],
        initial_guess[1],
        initial_guess[2],
        centroid_yx_oversamp=(y_center_final_cookie_oversamp, x_center_final_cookie_oversamp),
        shape_original_2d=shape_original_2d,
        fac_oversamp=fac_oversamp,
        pixel_scale_mas=config_observing['pixel_scales']['img_lm'],
        filter_file=config_observing['polychromatic_observing_filters_lm'][filter_name],
        pinhole_diam_rad=pinhole_diam_rad,
        save_fyi_plot=False,
    )
    best_fit_model_1d = model_for_fit_fixed(
        dummy_x_native,
        D_aperture_fit,
        D_obscuration_fit,
        ampl_fit,
        centroid_yx_oversamp=(y_center_final_cookie_oversamp, x_center_final_cookie_oversamp),
        shape_original_2d=shape_original_2d,
        fac_oversamp=fac_oversamp,
        pixel_scale_mas=config_observing['pixel_scales']['img_lm'],
        filter_file=config_observing['polychromatic_observing_filters_lm'][filter_name],
        pinhole_diam_rad=pinhole_diam_rad,
        save_fyi_plot=False,
    )

    
    initial_guess_model_2d = initial_guess_model_1d.reshape(shape_original_2d)
    best_fit_model_2d = best_fit_model_1d.reshape(shape_original_2d)

    data_original_2d = np.asarray(cookie_cut_out_sci_original, dtype=float)
    residuals_fit_native_2d = data_original_2d - best_fit_model_2d
    zscale_native = ZScaleInterval()
    vmin_n, vmax_n = zscale_native.get_limits(data_original_2d)

    # Native-grid summary: original, model, residuals, and 1D cross-section
    fig_triple, ax_triple = plt.subplots(
        2, 2, figsize=(10, 8), constrained_layout=True
    )
    for ax in ax_triple.flat:
        ax.set_box_aspect(1)

    # Top-left: original data
    im_data = ax_triple[0, 0].imshow(
        data_original_2d, origin="lower", cmap="gray_r", vmin=vmin_n, vmax=vmax_n
    )
    ax_triple[0, 0].set_title("Original data")

    # Top-right: best-fit model
    ax_triple[0, 1].imshow(
        best_fit_model_2d, origin="lower", cmap="gray_r", vmin=vmin_n, vmax=vmax_n
    )
    ax_triple[0, 1].set_title("Best-fit model")

    # Bottom-left: residuals image
    rmax = np.nanmax(np.abs(residuals_fit_native_2d))
    if not np.isfinite(rmax) or rmax == 0:
        rmax = 1e-15
    im_res = ax_triple[1, 0].imshow(
        residuals_fit_native_2d,
        origin="lower",
        cmap="RdBu_r",
        vmin=-25000,
        vmax=25000,
    )
    ax_triple[1, 0].set_title("Residuals (data − model)")

    # Bottom-right: 1D cross-section through the center row
    ny, nx = data_original_2d.shape
    center_y = ny // 2
    x_pix = np.arange(nx)
    cross_data = data_original_2d[center_y, :]
    cross_model = best_fit_model_2d[center_y, :]
    ax_triple[1, 1].plot(x_pix, cross_data, label="Data")
    ax_triple[1, 1].plot(x_pix, cross_model, label="Model", linestyle="--")
    ax_triple[1, 1].set_xlabel("Pixel")
    ax_triple[1, 1].set_ylabel("Counts")
    ax_triple[1, 1].set_title("Center-row cross-section")
    ax_triple[1, 1].legend()

    # Colorbars
    cbar_data = fig_triple.colorbar(
        im_data,
        ax=[ax_triple[0, 0], ax_triple[0, 1]],
        fraction=0.035,
        pad=0.02,
        shrink=0.8,
    )
    cbar_data.set_label("Counts")
    cbar_res = fig_triple.colorbar(
        im_res, ax=ax_triple[1, 0], fraction=0.035, pad=0.02, shrink=0.8
    )
    cbar_res.set_label("Δ counts")

    pinhole_label = (
        "None (delta pinhole)" if pinhole_diam_rad is None else f"{pinhole_diam_rad:.2e} rad"
    )
    fig_triple.suptitle(
        f"Native fit grid — {filter_name} "
        f"(λ={config_observing['monochromatic_observing_filters_lm'][filter_name]*1e6:.2f} μm)\n"
        f"Init: D_aper={initial_guess[0]:.2f}, D_obsc={initial_guess[1]:.2f}, ampl={initial_guess[2]:.2e}\n"
        f"Fit:  D_aper={D_aperture_fit:.2f}±{D_aperture_err:.2f}, "
        f"D_obsc={D_obscuration_fit:.2f}±{D_obscuration_err:.2f}, "
        f"ampl={ampl_fit:.2e}±{ampl_err:.2e}\n"
        f"pinhole_diam={pinhole_label}",
        fontsize=11,
    )
    
    file_triple = f"figs_dump/free_ann_ap_data_model_resid_native_{plot_string}.png"
    fig_triple.savefig(file_triple)
    logging.info(f"Saved {file_triple} to figs_dump/")
    plt.close(fig_triple)

    fig_cs, (ax_lin, ax_log) = plt.subplots(
        1, 2, figsize=(12, 4), sharex=True, constrained_layout=True
    )
    # Linear
    ax_lin.plot(x_pix, cross_data, label="Data")
    ax_lin.plot(x_pix, cross_model, label="Model", linestyle="--")
    ax_lin.set_xlabel("Pixel")
    ax_lin.set_ylabel("Counts")
    ax_lin.legend()
    # Log: avoid non-positive values
    cross_data_log = cross_data.copy()
    cross_model_log = cross_model.copy()
    cross_data_log[cross_data_log <= 0] = np.nan
    cross_model_log[cross_model_log <= 0] = np.nan
    ax_log.plot(x_pix, cross_data_log, label="Data")
    ax_log.plot(x_pix, cross_model_log, label="Model", linestyle="--")
    ax_log.set_xlabel("Pixel")
    ax_log.set_ylabel("Counts (log)")
    ax_log.set_yscale("log")
    ax_log.legend()
    file_cs = f"figs_dump/free_ann_ap_cross_sections_lin_log_{plot_string}.png"
    print(f"Saved {file_cs} to figs_dump/")
    fig_cs.savefig(file_cs)
    plt.close(fig_cs)
    print(f"Saved {file_cs} to figs_dump/")


    # Build oversampled models directly on the oversampled grid for diagnostics/MTF.
    initial_guess_model_2d_oversamp = model_for_fit_fixed(
        dummy_x_oversamp,
        initial_guess[0],
        initial_guess[1],
        initial_guess[2],
        centroid_yx_oversamp=(y_center_final_cookie_oversamp, x_center_final_cookie_oversamp),
        shape_oversamp=shape_oversampled_2d,
        fac_oversamp=fac_oversamp,
        pixel_scale_mas=config_observing['pixel_scales']['img_lm'],
        filter_file=config_observing['polychromatic_observing_filters_lm'][filter_name],
        pinhole_diam_rad=pinhole_diam_rad,
        save_fyi_plot=False,
    ).reshape(shape_oversampled_2d)
    best_fit_model_2d_oversamp = model_for_fit_fixed(
        dummy_x_oversamp,
        D_aperture_fit,
        D_obscuration_fit,
        ampl_fit,
        centroid_yx_oversamp=(y_center_final_cookie_oversamp, x_center_final_cookie_oversamp),
        shape_oversamp=shape_oversampled_2d,
        fac_oversamp=fac_oversamp,
        pixel_scale_mas=config_observing['pixel_scales']['img_lm'],
        filter_file=config_observing['polychromatic_observing_filters_lm'][filter_name],
        pinhole_diam_rad=pinhole_diam_rad,
        save_fyi_plot=False,
    ).reshape(shape_oversampled_2d)

    # Calculate chi-squared
    chi_squared = np.sum((data_empirical_original_1d - best_fit_model_1d) ** 2 / (0.01**2))  # assuming noise std = 0.01
    dof = len(data_empirical_original_1d) - 3  # degrees of freedom (data points - number of parameters)
    reduced_chi_squared = chi_squared / dof

    # best_fit_model_2d is already created above

    logging.info(f"\nChi-squared = {chi_squared:.2f}")
    logging.info(f"Degrees of freedom = {dof}")
    logging.info(f"Reduced chi-squared = {reduced_chi_squared:.6f}")




    ############################################################
    # Find the Strehl from the MTF, like in fit_annular_aperture_fixed
    fft_model_power_cutoff, fft_empirical_power_cutoff, cutoff_freq, fx, fy, n_fft = mtf_arrays(
        array_empirical=cookie_cut_out_sci_oversamp,
        array_model=best_fit_model_2d_oversamp,
        config_observing=config_observing,
        fac_oversamp=fac_oversamp,
        size=shape_oversampled_2d[0],
        filter_name=filter_name,
    )
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
    vmin, vmax = zscale.get_limits(nonan_empirical_2d_oversamp)

    fig, axs = plt.subplots(3, 2, figsize=(20, 15), constrained_layout=True,
                            gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 1, 1]})
    for ax in axs.flat:
        ax.set_box_aspect(1)

    # Panel 1: Empirical data
    im0 = axs[0,0].imshow(nonan_empirical_2d_oversamp, vmin=vmin, vmax=vmax)
    axs[0,0].set_title("Empirical")

    # Panel 2: Best fit
    im1 = axs[0,1].imshow(best_fit_model_2d_oversamp, vmin=vmin, vmax=vmax)
    axs[0,1].set_title("Best fit")

    # Panel 3: Cross-section between empirical and best-fit PSF
    center_y, center_x = np.array(nonan_empirical_2d_oversamp.shape) // 2
    cross_empirical = nonan_empirical_2d_oversamp[center_y, :]
    cross_best_fit = best_fit_model_2d_oversamp[center_y, :]
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
    im2 = axs[2,0].imshow(initial_guess_model_2d_oversamp, vmin=vmin, vmax=vmax)
    axs[2,0].set_title("Initial guess")

    # Panel 4: Residuals
    residuals = nonan_empirical_2d_oversamp - best_fit_model_2d_oversamp
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