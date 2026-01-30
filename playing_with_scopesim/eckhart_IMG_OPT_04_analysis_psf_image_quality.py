# Does some simple analysis of simulated images written out by the sim notebook.
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS

import scipy
from scipy.spatial import distance_matrix
from scipy.special import j0, j1
from itertools import combinations
import glob
import os
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.ndimage import zoom, shift, center_of_mass

from matplotlib import pyplot as plt
from matplotlib import colors
from astropy.visualization import ZScaleInterval

from photutils.centroids import centroid_sources, centroid_com, centroid_2dg

import pandas as pd

import ipdb

import scopesim as sim
from skimage import measure
from scipy.special import j1
import yaml


def jinc(x):
    x = np.asarray(x)
    y = np.empty_like(x, dtype=float)
    mask = x != 0
    y[mask] = j1(x[mask]) / x[mask]
    y[~mask] = 0.5
    return y

def intensity_annular_aperture(r_rad_array, wavel, D_aperture, D_obscuration, ampl=1):
    '''
    Calculate the intensity through an aperture with a central obscuration
    Ref. 'E-REP-MPIA-1203 0-1 xx-10-2024', Sec. 4.4

    INPUTS:
    - r_rad_array: 2D array of radial distances from the center (units radians)
    - wavel: wavelength (units meters)
    - D_aperture: aperture diameter (units meters)
    - D_obscuration: obscuration diameter (units meters)

    OUTPUTS:
    - I_r_array: 2D array of intensity on the detector
    '''


    nu_ = np.pi * r_rad_array * D_aperture / wavel # unitless

    eps_ = D_obscuration / D_aperture # unitless
    
    # see Eqn. 43 in 'E-REP-MPIA-1203 0-1 xx-10-2024'
    I_r = (1/(1-eps_**2)**2) * ( (2*jinc(nu_)) - eps_**2 * (2*jinc(nu_*eps_)) ) ** 2

    # normalize to the amplitude
    I_r = ampl * I_r / np.nanmax(I_r)

    return I_r


def gaussian_2d(xy_mesh, amplitude, xo, yo, sigma_x_pix, sigma_y_pix, theta):
    x, y = xy_mesh
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2 * sigma_x_pix**2) + (np.sin(theta)**2) / (2 * sigma_y_pix**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x_pix**2) + (np.sin(2 * theta)) / (4 * sigma_y_pix**2)
    c = (np.sin(theta)**2) / (2 * sigma_x_pix**2) + (np.cos(theta)**2) / (2 * sigma_y_pix**2)
    g = amplitude * np.exp(-(a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2)))
    return g.ravel()


# Define a wrapper function for curve_fit
# curve_fit expects: func(x, *params) where x is the independent variable
# and params are the parameters to fit
def model_for_fit_fixed(r_rad_1d, D_aperture, D_obscuration, ampl, baseline_shape, valid_mask, wavel):
    """
    Fixed wrapper function for intensity_annular_aperture to use with curve_fit.
    
    Parameters:
    - r_rad_1d: 1D array of radial distances (masked, only valid points)
    - D_aperture: aperture diameter (meters)
    - D_obscuration: obscuration diameter (meters)
    - ampl: amplitude
    - baseline_shape: tuple, shape of the 2D array (fixed, not optimized)
    - valid_mask: boolean array, mask for valid data points (fixed, not optimized)
    - fac_oversamp: oversampling factor
    
    Returns:
    - 1D array of intensity values (masked, same length as input)
    """
    # Reconstruct the full 2D array by inserting masked values back into their original positions
    r_rad_2d_full = np.full(baseline_shape, np.nan).flatten()
    r_rad_2d_full[valid_mask] = r_rad_1d
    r_rad_2d = r_rad_2d_full.reshape(baseline_shape)
    
    # Calculate intensity using the model function
    intensity_2d = intensity_annular_aperture(
        r_rad_array=r_rad_2d, 
        wavel=wavel, 
        D_aperture=D_aperture, 
        D_obscuration=D_obscuration, 
        ampl=ampl
    )
    
    # Flatten and apply the same mask to return only valid points
    intensity_1d_full = intensity_2d.flatten()
    return intensity_1d_full[valid_mask]


def fit_analytical_psfs(cookie_cut_out_sci, filter_name, plot_string, x_center_final_cookie_oversamp, y_center_final_cookie_oversamp, fac_oversamp, wavel, pixel_scale_mas):
    '''
    Fit a 2D analytical PSF to a given frame.

    INPUTS:
    cookie_cut_out_sci: 2D array of the cookie cut our from the full science frame
    filter_name: name of the observing filter
    plot_string: string to add to the plot file name
    x_center_final_cookie_oversamp: final x-center of the PSF (i.e., no more centroiding will be done here); in coordinates of the cookie cut-out
    y_center_final_cookie_oversamp: final y-center of the PSF; in coordinates of the cookie cut-out
    fac_oversamp: oversampling factor
    wavel: wavelength in meters
    pixel_scale_mas: pixel scale in mas

    OUTPUTS:
    '''

    # make the cutout from the full array
    #psf_perfect_cutout = psf_perfect_oversamp[int(y_center_final_oversamp-0.5*cookie_cut_out_sci.shape[0]):int(y_center_final_oversamp+0.5*cookie_cut_out_sci.shape[0]), \
    #        int(x_center_final_oversamp-0.5*cookie_cut_out_sci.shape[1]):int(x_center_final_oversamp+0.5*cookie_cut_out_sci.shape[1])]

    # Create a 2D array of distances from the center in arcseconds, with each pixel 6 mas
    size = cookie_cut_out_sci.shape[0] 
    baseline_shape = (size, size)
    pixel_scale_arcsec = pixel_scale_mas / 1000.0  # arcseconds per pixel
    y, x = np.indices((size, size))
    center = (y_center_final_cookie_oversamp, x_center_final_cookie_oversamp)
    r_pix = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    test_array_arcsec_2d = r_pix * pixel_scale_arcsec
    # convert to radians
    r_rad_2d = test_array_arcsec_2d / ( 3600 * (180/np.pi ) )

    # rescale based on the oversampling factor (this effectively makes the plate scale smaller)
    r_rad_2d = r_rad_2d / fac_oversamp

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

    # Store the mask as a global variable for use in model_for_fit
    valid_mask = mask.copy()

    print(f"Original data points: {len(r_rad_1d_full)}")
    print(f"Valid data points after masking: {len(r_rad_1d)}")
    print(f"Arrays are aligned: {len(r_rad_1d) == len(test_empirical_1d)}")

    # Initial parameter guesses
    # [D_aperture, D_obscuration, ampl]
    initial_guess = [36., 12., 3500.] 

    # Create a wrapper function that binds the fixed parameters (baseline_shape and valid_mask)
    # This ensures curve_fit only optimizes D_aperture, D_obscuration, and ampl
    
    model_wrapper = lambda r_rad_1d, D_aperture, D_obscuration, ampl: \
        model_for_fit_fixed(r_rad_1d, D_aperture, D_obscuration, ampl, baseline_shape, valid_mask, wavel)

    # Set bounds for parameters: [D_aperture, D_obscuration, ampl]
    # D_aperture: between 1 and 50
    # D_obscuration: no bounds (use -inf to +inf, but should be positive and < D_aperture)
    # ampl: no bounds (use -inf to +inf, but should be positive)
    lower_bounds = [1.0, 0.0, 0.0]  # D_aperture >= 1, D_obscuration >= 0, ampl >= 0
    upper_bounds = [50.0, np.inf, np.inf]  # D_aperture <= 50, no upper bounds for others
    
    # Perform the fit with the fixed function
    # Note: 'trf' method supports bounds, 'lm' does not
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

    # Calculate parameter uncertainties from covariance matrix
    param_errors = np.sqrt(np.diag(pcov))
    D_aperture_err = param_errors[0]
    D_obscuration_err = param_errors[1]
    ampl_err = param_errors[2]

    # Print results
    print('--------------------------------')
    print("Fixed observing parameters:")
    print(f"filter: {filter_name}, λ={wavel*1e6:.2f}μm, ps={pixel_scale_mas:.2f}mas", )
    print('--------------------------------')
    print("Best-fit parameters:")
    print(f"D_aperture = {D_aperture_fit:.2f} ± {D_aperture_err:.2f} meters")
    print(f"D_obscuration = {D_obscuration_fit:.2f} ± {D_obscuration_err:.2f} meters")
    print(f"ampl = {ampl_fit:.2f} ± {ampl_err:.2f}")

    # Check if covariance matrix has infs
    if np.any(np.isinf(pcov)):
        print("\nWARNING: Covariance matrix contains infinities!")
        print("This usually means the fit didn't converge properly.")
    else:
        print("\nCovariance matrix is finite - fit converged successfully!")

    # generate the best-fit model based on the fit parameters
    # Note: r_rad_2d needs to be flattened and masked for model_for_fit_fixed
    r_rad_1d = r_rad_2d.flatten()
    r_rad_1d_masked = r_rad_1d[valid_mask]
    
    initial_guess_model_1d = model_for_fit_fixed(r_rad_1d_masked, initial_guess[0], initial_guess[1], initial_guess[2], baseline_shape, valid_mask, wavel)
    best_fit_model_1d = model_for_fit_fixed(r_rad_1d_masked, D_aperture_fit, D_obscuration_fit, ampl_fit, baseline_shape, valid_mask, wavel)
    
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

    print(f"\nChi-squared = {chi_squared:.2f}")
    print(f"Degrees of freedom = {dof}")
    print(f"Reduced chi-squared = {reduced_chi_squared:.6f}")

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(test_empirical_2d)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Panel 1: Empirical data
    im0 = axs[0].imshow(test_empirical_2d, vmin=vmin, vmax=vmax)
    axs[0].set_title("Empirical")

    # Panel 2: Best fit
    im1 = axs[1].imshow(best_fit_model_2d, vmin=vmin, vmax=vmax)
    axs[1].set_title("Best fit")

    # Panel 3: Initial guess
    im2 = axs[2].imshow(initial_guess_model_2d, vmin=vmin, vmax=vmax)
    axs[2].set_title("Initial guess")

    # Panel 4: Residuals
    im2 = axs[3].imshow(test_empirical_2d - best_fit_model_2d, vmin=vmin, vmax=vmax)
    axs[3].set_title("Empirical - Best fit")


    plt.suptitle(
        f'λ={wavel*1e6:.2f}μm, pix={pixel_scale_mas:.2f}mas, \n'
        f'Best fits: D_aper={D_aperture_fit:.2f}±{D_aperture_err:.2f}m, '
        f'D_obsc={D_obscuration_fit:.2f}±{D_obscuration_err:.2f}m, '
        f'Amp={ampl_fit:.2f}±{ampl_err:.2f}',
        fontsize=10
    )


    # Add one colorbar for all
    fig.colorbar(im0, ax=axs, orientation='vertical', fraction=0.04, pad=0.04).set_label('Color scale is the same')

    plt.savefig('test.png')
    plt.show()

    return r_rad_2d


def fit_empirical_fwhm(frame, plot_string):
    '''
    Take the data as-is, find where the intensity is 50% of the peak intensity, and then calculate the FWHM in x and y.

    INPUTS:
    frame: 2D array of the frame
    plot_string: string to add to the plot file name
    '''

    # find the peak intensity
    #ipdb.set_trace()
    peak_intensity = np.max(frame)
    # find where the intensity is 50% of the peak intensity
    # Find the positions of the maximum value as the initial guess for the center
    y_peak, x_peak = np.unravel_index(np.argmax(frame), frame.shape)
    # Create x and y coordinate arrays
    y, x = np.indices(frame.shape)
    # Define a threshold for 50% of the peak
    half_max = 0.5 * peak_intensity

    # fit an oval to the region above half-max
    mask_half = frame >= half_max
    labeled = measure.label(mask_half)
    props = measure.regionprops(labeled)
    if len(props) > 0:
        # Select the largest region by area
        prop_biggest = [max(props, key=lambda p: p.area)]
    #ipdb.set_trace()
    if len(props) == 0:
        prop_biggest_dims = np.nan, np.nan

    # use bounding box to get the dims in x and y (instead of just major and minor axis lengths)
    min_row, min_col, max_row, max_col = prop_biggest[0].bbox
    height_y = max_row - min_row   # axis-aligned y length
    width_x  = max_col - min_col   # axis-aligned x length

    # Plot the frame
    plt.figure()
    plt.imshow(frame, origin='lower', cmap='gray')
    # Plot the bounding box if prop_biggest was found
    if len(props) > 0:
        rect = plt.Rectangle(
            (min_col, min_row), width_x, height_y,
            edgecolor='red', facecolor='none', linewidth=2, linestyle='--'
        )
        plt.gca().add_patch(rect)
    plt.title(f'Frame with Bounding Box at 50% Peak\nFWHM in x (pix): {width_x:.2f}, FWHM in y (pix): {height_y:.2f}')
    # save the plot to file
    plot_filename = 'empirical_fwhm_' + plot_string + '.png'
    #ipdb.set_trace()
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f'Figure saved as {plot_filename}')
    #plt.show()

    plt.close()

    return height_y, width_x


def fit_gaussian(frame, center_guess):
    """
    Fit a 2D Gaussian function to a given frame.

    Parameters:
    frame (ndarray): 2D array representing the frame.
    center_guess (list): List containing the initial guess for the center coordinates.

    Returns:
    fitted_array (ndarray): 2D array representing the fitted Gaussian function.
    fwhm_x_pix (float): Full Width at Half Maximum (FWHM) in the x-direction.
    fwhm_y_pix (float): Full Width at Half Maximum (FWHM) in the y-direction.
    sigma_x_pix (float): Standard deviation in the x-direction.
    sigma_y_pix (float): Standard deviation in the y-direction.
    amplitude_counts (float): Amplitude of the Gaussian function in counts.
    """
    y, x = np.indices(frame.shape)
    xy_mesh = (x, y)
    p0 = [np.max(frame), center_guess[0], center_guess[1], 1, 1, 0]
    popt, pcov = curve_fit(gaussian_2d, xy_mesh, frame.ravel(), p0=p0)
    fitted_array = gaussian_2d(xy_mesh, *popt).reshape(frame.shape)
    fwhm_x_pix = 2 * np.sqrt(2 * np.log(2)) * np.abs(popt[3])
    fwhm_y_pix = 2 * np.sqrt(2 * np.log(2)) * np.abs(popt[4])
    amplitude_counts = popt[0]
    x_center_pix = popt[1]
    y_center_pix = popt[2]
    sigma_x_pix = popt[3]
    sigma_y_pix = popt[4]
    angle_theta_deg = popt[5]
    
    return fitted_array, x_center_pix, y_center_pix, fwhm_x_pix, fwhm_y_pix, sigma_x_pix, sigma_y_pix, angle_theta_deg, amplitude_counts


def fyi_plot_centroiding(array_to_plot, coords_to_plot, title_string=None, zscale=False):
    # INSERT_YOUR_CODE

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(array_to_plot)
    plt.clf()
    plt.imshow(array_to_plot, origin='lower', vmin=vmin, vmax=vmax, cmap='gray')
    plt.scatter(coords_to_plot[:, 1], coords_to_plot[:, 0], color='red', s=10)
    plt.title(title_string)
    plt.show()
    plt.close()


def fit_gaussian_fwhm(cookie_cut_out_sci, obs_filter, fp_mask, pp_mask, coords_guess, plot_string, fac_oversamp):
    '''
    Find FWHM of Gaussian-best-fit to empirical; all fit parameters are free

    INPUTS:
    cookie_cut_out_sci: 2D array of the science frame
    obs_filter: observing filter (string)
    fp_mask: focal plane mask (string)
    pp_mask: pupil plane mask (string)
    coords_guess: 2D array of the centroided coordinates (one coordinate pair)
    plot_string: string to add to the plot file name
    fac_oversamp: oversampling factor

    OUTPUTS:
    fwhm_y_pix: FWHM in y-direction
    fwhm_x_pix: FWHM in x-direction
    '''

    ## ## TO DO: ARE THE INDEXES RIGHT HERE?
    cookie_cut_out_best_fit, x_center_pix_oversamp_cutout, y_center_pix_oversamp_cutout, fwhm_x_pix, fwhm_y_pix, sigma_x_pix, sigma_y_pix, angle_theta_deg, amplitude_counts = fit_gaussian(cookie_cut_out_sci, \
        center_guess = coords_guess)
    residuals = cookie_cut_out_sci - cookie_cut_out_best_fit

    # strehl based on the Gaussian fit
    gaussian_based_strehl = np.max(cookie_cut_out_sci) / np.max(cookie_cut_out_best_fit)
    print('--------------------------------')
    print(f'Observing filter: {obs_filter}')
    print(f'PSF ID: {plot_string}')
    print(f'Focal plane mask: {fp_mask}')
    print(f'Pupil plane mask: {pp_mask}')
    print(f'Gaussian-based Strehl: {gaussian_based_strehl:.2f}')

    ipdb.set_trace()

    # plot four subplots: 2D science, 2D best-fit, 2D residuals, and 1D overplotting of a cross-section of the science and best-fit
    plt.clf()
    # Determine vmin and vmax for consistent color scaling across all 2D plots
    vmin = min(np.nanmin(cookie_cut_out_sci), np.nanmin(cookie_cut_out_best_fit), np.nanmin(residuals))
    vmax = max(np.nanmax(cookie_cut_out_sci), np.nanmax(cookie_cut_out_best_fit), np.nanmax(residuals))
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # 2D Science image
    im0 = axs[0, 0].imshow(cookie_cut_out_sci, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('Science')
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)
    # 2D Best-fit image
    im1 = axs[0, 1].imshow(cookie_cut_out_best_fit, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('Best-fit')
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)
    # 2D Residuals image
    im2 = axs[1, 0].imshow(residuals, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title('Residuals')
    plt.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)
    # Plot a cross-section through the maximum of the PSF (along the row/col with the peak)
    max_index = np.unravel_index(np.argmax(cookie_cut_out_sci), cookie_cut_out_sci.shape)
    # Extract the row and column through the peak
    sci_row = cookie_cut_out_sci[max_index[0], :]
    best_fit_row = cookie_cut_out_best_fit[max_index[0], :]
    axs[1, 1].plot(sci_row, label='Empirical')
    axs[1, 1].plot(best_fit_row, label='Best-fit')
    # Annotate plot with FWHM in x and y
    fwhm_text = f'FWHM x = {fwhm_x_pix:.2f} pix\nFWHM y = {fwhm_y_pix:.2f} pix'
    axs[1, 1].text(
        0.95, 0.05, fwhm_text,
        transform=axs[1, 1].transAxes,
        fontsize=10, color='black',
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3')
    )
    axs[1, 1].legend()
    axs[1, 1].set_title('1D cross-section, empirical vs best-fit')
    plt.suptitle(f'PSF, oversampling factor: {fac_oversamp:.2f} \n Found coord (y,x): ({y_center_pix_oversamp_cutout:.2f}, {x_center_pix_oversamp_cutout:.2f}) \n Found FWHM x: {fwhm_x_pix:.2f} pix, FWHM y: {fwhm_y_pix:.2f} pix,\nFound amplitude: {amplitude_counts:.2f} counts')
    plt.tight_layout()
    #plt.show()
    # Save the plot to file with num_coord as a 2-digit zero-padded string
    plot_filename = f'psf_gaussian_best_fit_'+plot_string+'.png'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f'Figure saved as {plot_filename}')
    plt.close()
    #ipdb.set_trace()

    return x_center_pix_oversamp_cutout, y_center_pix_oversamp_cutout, fwhm_x_pix, fwhm_y_pix, amplitude_counts, gaussian_based_strehl


def fit_simmed_psfs(cookie_cut_out_sci, plot_string, obs_filter, fp_mask, pp_mask, x_center_final_oversamp, y_center_final_oversamp, fac_oversamp):
    '''
    Find FWHM of a PSF using a perfect PSF from ScopeSim
    
    INPUTS:
    cookie_cut_out_sci: 2D array of the science frame
    plot_string: string to add to the plot file name
    obs_filter: observing filter (string)
    fp_mask: focal plane mask (string)
    pp_mask: pupil plane mask (string)
    x_center_final_oversamp: final x-center of the PSF (i.e., no more centroiding will be done here); in coordinates of the entire array
    y_center_final_oversamp: final y-center of the PSF; in coordinates of the entire array
    fac_oversamp: oversampling factor

    OUTPUTS:
    psf_perfect_cutout_best_fit: cutout around the best-fit simulated PSF
    '''

    # set up instrument
    ## ## TO DO: MAKE THIS MORE GENERAL AND FLEXIBLE, FOR MULT OBSERVING MODES
    cmd = sim.UserCommands(use_instrument='METIS', set_modes=['wcu_img_lm'])
    metis = sim.OpticalTrain(cmd)

    wcu = metis['wcu_source']

    # set the filter
    metis["filter_wheel"].change_filter(obs_filter)

    wcu.set_fpmask(fp_mask)

    #pp_mask = metis['pupil_masks'].meta['current_mask'] # just one mask for now (Open)
    ipdb.set_trace()

    metis.effects.pprint_all()

    bb_temp = 1000 * u.K
    NDIT, EXPTIME = 1, 0.2


    print('--------------------------------')
    print('Current Observing filter:', obs_filter)
    print('Current WCU FP mask:', wcu.fpmask)
    print('Current WCU PP mask:', pp_mask)
    #ipdb.set_trace()
    # background
    print('Closing WCU BB aperture first to get a background ...')
    # background
    wcu.set_bb_aperture(value = 0.0)
    metis.observe()
    outhdul_off = metis.readout(ndit = NDIT, exptime = EXPTIME)[0]
    background = outhdul_off[1].data

    print('Re-opening WCU BB aperture to get a PSF ...')
    wcu.set_bb_aperture(value = 1.0) # open BB source

    #metis["filter_wheel"].change_filter(obs_filter)

    print('--------------------------------')
    print('Current Observing filter:', obs_filter)
    print('Current WCU FP mask:', wcu.fpmask)
    print('Current WCU PP mask:', pp_mask)
    print('Opening WCU BB aperture...')

    metis.observe()
    outhdul_on = metis.readout(ndit = NDIT, exptime = EXPTIME)[0]
    sci = outhdul_on[1].data
    #ipdb.set_trace()
    # Get perfect, background-subtracted PSF - no detector noise
    psf_perfect = sci - background

    print('!!! --- ARTIFICIALLY SUBTRACTING OFF A BACKGROUND RESIDUAL; FIX LATER --- !!')
    psf_perfect -= np.nanmean(psf_perfect)

    # Oversample the background-subtracted PSF to match the cookie_cut_out_sci oversampling
    psf_perfect_oversamp = zoom(psf_perfect, fac_oversamp, order=3)

    # for debugging
    fits.writeto("psf_perfect_oversamp.fits", psf_perfect_oversamp, overwrite=True)
    print("Saved psf_perfect_oversamp.fits for checking.")


    #ipdb.set_trace()

    # take a cutout of the PSF at the exact same coordinates as the cookie cut-out
    psf_perfect_cutout = psf_perfect_oversamp[int(y_center_final_oversamp-0.5*cookie_cut_out_sci.shape[0]):int(y_center_final_oversamp+0.5*cookie_cut_out_sci.shape[0]), \
        int(x_center_final_oversamp-0.5*cookie_cut_out_sci.shape[1]):int(x_center_final_oversamp+0.5*cookie_cut_out_sci.shape[1])]

    # cut out the central region the same size as the cookie cut-out
    #psf_perfect_cutout = psf_perfect[int(psf_perfect.shape[0]/2-0.5*cookie_cut_out_sci.shape[0]):int(psf_perfect.shape[0]/2+0.5*cookie_cut_out_sci.shape[0]), \
    #    int(psf_perfect.shape[1]/2-0.5*cookie_cut_out_sci.shape[1]):int(psf_perfect.shape[1]/2+0.5*cookie_cut_out_sci.shape[1])]

    # multiply psf_perfect_cutout by a coefficient to make it a best-fit to cookie_cut_out_sci
    coefficient = np.sum(cookie_cut_out_sci) / np.sum(psf_perfect_cutout)
    psf_perfect_cutout_best_fit = psf_perfect_cutout * coefficient

    # Make subplots of cookie_cut_out_sci, psf_perfect_cutout_best_fit, and the residuals
    plt.figure(figsize=(12, 4))
    
    # Panel 1: cookie_cut_out_sci
    plt.subplot(1, 3, 1)
    zscale1 = ZScaleInterval()
    vmin1, vmax1 = zscale1.get_limits(cookie_cut_out_sci)
    plt.imshow(cookie_cut_out_sci, origin="lower", cmap="viridis", vmin=vmin1, vmax=vmax1)
    plt.title("cookie_cut_out_sci")
    plt.colorbar(shrink=0.7, label="Counts")
    
    # Panel 2: psf_perfect_cutout_best_fit
    plt.subplot(1, 3, 2)
    zscale2 = ZScaleInterval()
    vmin2, vmax2 = zscale2.get_limits(psf_perfect_cutout_best_fit)
    plt.imshow(psf_perfect_cutout_best_fit, origin="lower", cmap="viridis", vmin=vmin2, vmax=vmax2)
    plt.title("psf_perfect_cutout_best_fit")
    plt.colorbar(shrink=0.7, label="Counts")
    
    # Panel 3: Residuals
    residuals = cookie_cut_out_sci - psf_perfect_cutout_best_fit
    plt.subplot(1, 3, 3)
    zscale3 = ZScaleInterval()
    vmin3, vmax3 = zscale3.get_limits(residuals)
    plt.imshow(residuals, origin="lower", cmap="RdBu", vmin=vmin3, vmax=vmax3)
    plt.title("Residuals (sci - best_fit)")
    plt.colorbar(shrink=0.7, label="Counts")
    
    plt.tight_layout()
    plt.show()

    return psf_perfect_cutout_best_fit


def strehl_psfs(file_name, fp_mask, pp_mask, filter_name=None, wavel=None, pixel_scale_mas=None, fit_simmed_psf=False, fit_analytical_psf=False, psfs_subset='all', config_file_name=None):
    '''
    Find the Strehl ratio of a grid of PSFs
    
    INPUTS:
    file_name: name of the file containing the grid of PSFs
    fp_mask: focal plane mask (string)
    pp_mask: pupil plane mask (string)
    filter_name: name of the observing filter
    wavel: wavelength in meters
    pixel_scale_mas: pixel scale in mas
    fit_simmed_psf: whether to fit a ScopeSim-simulated PSF
    fit_analytical_psf: whether to fit an analytical PSF
    psfs_subset: 'all' to process all PSFs, or an integer to process only the first N PSFs
    config_file_name: name of the configuration file containing the coordinates of the PSFs


    OUTPUTS:
    None; writes out plots and data
    '''

    # return the locations and other data for each PSF

    grid_frame = fits.open(file_name)
    grid_data = grid_frame[1].data
    grid_header = grid_frame[1].header

    # read in coordinate guesses
    if config_file_name is None:
        raise ValueError("config_file_name is required to load PSF coordinates.")
    with open(config_file_name, "r") as config_file:
        config_data = yaml.safe_load(config_file)
    coords_entries = config_data.get("coordinates", [])
    if not coords_entries:
        raise ValueError(f"No coordinates found in {config_file_name}")
    coords_guesses_all = np.array([(entry["y"], entry["x"]) for entry in coords_entries])
    coords_guesses_y_all = coords_guesses_all[:, 0]
    coords_guesses_x_all = coords_guesses_all[:, 1]

    #guesses_grid = np.array(())

    # for debugging
    '''
    print('! ---------- debugging --------- !')
    x_grid = np.array([3.0, 3.0, 3.1, 3.3])
    y_grid = np.array([13.0, 13.0, 13.1, 13.3])
    '''

    # oversample the image to find centroids, FWHM
    oversample_factor = 4
    # Step 1: Oversample the PSFs by a factor of 4 using bicubic interpolation
    grid_data_oversamp = zoom(grid_data, oversample_factor, order=3)
    #psf_simmed_oversamp = zoom(psf_simmed, oversample_factor, order=3)
    coords_guesses_x_all_oversamp = coords_guesses_x_all * oversample_factor
    coords_guesses_y_all_oversamp = coords_guesses_y_all * oversample_factor
    coords_guesses_all_oversamp = np.vstack((coords_guesses_y_all_oversamp, coords_guesses_x_all_oversamp)).T

    # find the PSF centroids, first pass
    x_pos_pix_oversamp, y_pos_pix_oversamp = centroid_sources(grid_data_oversamp, 
                                    xpos=coords_guesses_x_all_oversamp, 
                                    ypos=coords_guesses_y_all_oversamp, 
                                    box_size=41,
                                    centroid_func=centroid_2dg)

    # zip into one array
    coords_centroided_all_oversamp = np.vstack((y_pos_pix_oversamp, x_pos_pix_oversamp)).T

    # FYI
    #fyi_plot_centroiding(grid_data_oversamp, coords_centroided_all_oversamp, title_string="PSF first guesses", zscale=False)

    # make a cut-out of each psf and make a best-fit 2D Gaussian
    raw_cutout_size = 20 * oversample_factor
    num_coord = 0

    #cookie_cut_out_best_fit_list = []
    coord_x_array = np.zeros(len(y_pos_pix_oversamp))
    coord_y_array = np.zeros(len(y_pos_pix_oversamp))
    fwhm_x_pix_array = np.zeros(len(y_pos_pix_oversamp))
    fwhm_y_pix_array = np.zeros(len(y_pos_pix_oversamp))
    sigma_x_pix_array = np.zeros(len(y_pos_pix_oversamp))
    sigma_y_pix_array = np.zeros(len(y_pos_pix_oversamp))
    angle_theta_array = np.zeros(len(y_pos_pix_oversamp))
    amplitude_counts_array = np.zeros(len(y_pos_pix_oversamp))
    gaussian_based_strehl_array = np.zeros(len(y_pos_pix_oversamp))

    # make a copy from which we will subtract the PSFs to see the residuals
    canvas_grid_data = np.copy(grid_data)

    # Determine how many PSFs to process based on psfs_subset parameter
    total_psfs = len(y_pos_pix_oversamp)
    if psfs_subset == 'all':
        num_psfs_to_process = total_psfs
    elif isinstance(psfs_subset, int):
        num_psfs_to_process = min(psfs_subset, total_psfs)  # Don't exceed available PSFs
    else:
        raise ValueError(f"psfs_subset must be 'all' or an integer, got {psfs_subset}")
    
    print(f"Processing {num_psfs_to_process} out of {total_psfs} PSFs")

    # loop over each centroided PSF
    for num_coord in range(num_psfs_to_process):

        # is a cutout even necessary?
        cookie_edge_size = raw_cutout_size
        idx_x_start = int(x_pos_pix_oversamp[num_coord]-0.5*cookie_edge_size)
        idx_x_end = int(x_pos_pix_oversamp[num_coord]+0.5*cookie_edge_size)
        idx_y_start = int(y_pos_pix_oversamp[num_coord]-0.5*cookie_edge_size)
        idx_y_end = int(y_pos_pix_oversamp[num_coord]+0.5*cookie_edge_size)
        cookie_cut_out_sci_oversamp = grid_data_oversamp[idx_y_start:idx_y_end, idx_x_start:idx_x_end]
        #ipdb.set_trace()

        # FYI plot
        plt.clf()
        plt.imshow(cookie_cut_out_sci_oversamp, origin='lower', cmap='gray_r')
        # Convert scatter coordinates to cutout-relative coordinates
        # this point is that found with centroid_sources()
        x_scatter = x_pos_pix_oversamp[num_coord] - idx_x_start
        y_scatter = y_pos_pix_oversamp[num_coord] - idx_y_start
        plt.scatter(x_scatter, y_scatter, color='red', s=10)
        plt.title(f'Cookie cut-out sci at coord (y,x): {y_pos_pix_oversamp[num_coord]}, {x_pos_pix_oversamp[num_coord]}')
        plt.colorbar()
        plt.show()
        plt.close()
        #ipdb.set_trace()

        # Adjust the centroid coordinate for the cut-out: subtract the cutout starting indices to get cutout-relative coordinates
        coords_guess_this_cutout = np.array([
            coords_centroided_all_oversamp[num_coord][0] - idx_y_start,
            coords_centroided_all_oversamp[num_coord][1] - idx_x_start
        ])


        # find FWHM and Streof Gaussian-best-fit to empirical, and 
        # centroid the PSF, second pass
        x_center_pix_gaussian_best_fit_oversamp, y_center_pix_gaussian_best_fit_oversamp, fwhm_x_pix_gaussian_best_fit_oversamp, fwhm_y_pix_gaussian_best_fit_oversamp, amplitude_counts_gaussian_best_fit_oversamp, gaussian_based_strehl = fit_gaussian_fwhm(cookie_cut_out_sci_oversamp, 
                                                                                                        obs_filter=filter_name,
                                                                                                        fp_mask=fp_mask,
                                                                                                        pp_mask=pp_mask,
                                                                                                        coords_guess=coords_guess_this_cutout, 
                                                                                                        plot_string=f'num_coord_{num_coord}', 
                                                                                                        fac_oversamp=oversample_factor)


        # convert the coordinates of the cutout back to those of the entire oversampled image
        x_center_pix_gaussian_best_fit_oversamp_fullarray = x_center_pix_gaussian_best_fit_oversamp + idx_x_start
        y_center_pix_gaussian_best_fit_oversamp_fullarray = y_center_pix_gaussian_best_fit_oversamp + idx_y_start

        # fit a 2D Gaussian

        # find FWHM of empirical 
        '''
        fwhm_y_pix_empirical, fwhm_x_pix_empirical = fit_empirical_fwhm(cookie_cut_out_sci, plot_string=f'num_coord_{num_coord}')
        '''

 

        # fit a ScopeSim PSF
        if fit_simmed_psf:
            best_fit_cutout_oversamp = fit_simmed_psfs(cookie_cut_out_sci_oversamp, 
                                            plot_string=f'num_coord_{num_coord}', 
                                            obs_filter=filter_name,
                                            fp_mask=fp_mask,
                                            pp_mask=pp_mask,
                                            x_center_final_oversamp=x_pos_pix_oversamp[num_coord], 
                                            y_center_final_oversamp=y_pos_pix_oversamp[num_coord], 
                                            fac_oversamp=oversample_factor)


        # fit an analytical PSF
        if fit_analytical_psf:
            best_fit_analytical_oversamp = fit_analytical_psfs(cookie_cut_out_sci_oversamp, 
                                            filter_name=filter_name,
                                            plot_string=f'num_coord_{num_coord}', 
                                            x_center_final_cookie_oversamp=x_center_pix_gaussian_best_fit_oversamp, 
                                            y_center_final_cookie_oversamp=y_center_pix_gaussian_best_fit_oversamp, 
                                            wavel=wavel,
                                            pixel_scale_mas=pixel_scale_mas,
                                            fac_oversamp=oversample_factor)

        
        # resample back to the original size
        x_center_pix_gaussian_best_fit_normsamp = x_center_pix_gaussian_best_fit_oversamp_fullarray / oversample_factor
        y_center_pix_gaussian_best_fit_normsamp = y_center_pix_gaussian_best_fit_oversamp_fullarray / oversample_factor
        fwhm_x_pix_gaussian_best_fit_normsamp = fwhm_x_pix_gaussian_best_fit_oversamp / oversample_factor
        fwhm_y_pix_gaussian_best_fit_normsamp = fwhm_y_pix_gaussian_best_fit_oversamp / oversample_factor
     

        #ipdb.set_trace()
        # make cutout around the model (for plot)

        # save cookie_cut_out_sci and cookie_cut_out_best_fit as fits files
        '''
        file_name_sci = 'cookie_cut_out_sci.fits'
        file_name_best_fit = 'cookie_cut_out_best_fit.fits'
        fits.writeto(file_name_sci, cookie_cut_out_sci, overwrite=True)
        #fits.writeto(file_name_best_fit, cookie_cut_out_best_fit, overwrite=True)
        print(f'Saved {file_name_sci} and \n{file_name_best_fit}')
        '''

        # update arrays/lists
        #cookie_cut_out_best_fit_list.append(best_fit_cutout_oversamp)

        coord_x_array[num_coord] = x_center_pix_gaussian_best_fit_normsamp
        coord_y_array[num_coord] = y_center_pix_gaussian_best_fit_normsamp
        fwhm_x_pix_array[num_coord] = fwhm_x_pix_gaussian_best_fit_normsamp
        fwhm_y_pix_array[num_coord] = fwhm_y_pix_gaussian_best_fit_normsamp
        amplitude_counts_array[num_coord] = amplitude_counts_gaussian_best_fit_oversamp # note the amplitude doesn't need to be resampled
        gaussian_based_strehl_array[num_coord] = gaussian_based_strehl
        #sigma_x_pix_array[num_coord] = sigma_x_pix
        #sigma_y_pix_array[num_coord] = sigma_y_pix
        #angle_theta_array[num_coord] = angle_theta


    ipdb.set_trace()
    # plot the grid_data and annotate it with the best-fit fwhm in x and y for each PSF
    plt.clf()
    plt.imshow(grid_data, origin='lower', cmap='gray_r')
    for num_coord in range(len(coord_x_array)):
        # Draw a line from the text location to the PSF's actual (x, y) coordinate
        text_x = coord_x_array[num_coord] - 125
        text_y = coord_y_array[num_coord] + 10
        plt.text(
            text_x,
            text_y,
            f'x: {fwhm_x_pix_array[num_coord]:.2f}, \n y: {fwhm_y_pix_array[num_coord]:.2f}, \n theta: {angle_theta_array[num_coord]:.2f}, \n amp: {amplitude_counts_array[num_coord]:.2f}, \n strehl: {gaussian_based_strehl_array[num_coord]:.2f}',
            color='k',
            fontsize=7, rotation=20
        )
    plt.title('FWHM in x and y (pix), amplitude (counts)')
    plt.show()
    plt.close()

    ipdb.set_trace()

    return


def main():

    stem = '/podman-share/metis_work/playing_with_scopesim/'

    # dictionary of observing filters and their average wavelengths
    observing_filters_lm = {
        'Lp': 3.82e-6, 
        'short-L': 3.29e-6, 
        'L_spec': 3.56e-6, 
        'Mp': 4.76e-6, 
        'M_spec': 4.87e-6,
        'Br_alpha': 4.06e-6, 
        'Br_alpha_ref': 3.93e-6, 
        'PAH_3.3': 3.31e-6, 
        'PAH_3.3_ref': 3.45e-6, 
        'CO_1-0_ice': 4.66e-6,
        'CO_ref': 4.94e-6, 
        'H2O-ice': 3.09e-6
    }

    # dictionary of pixel scales (units mas)
    pixel_scales = {'img_lm': 5.47, 'img_n': 6.79}

    # pp mask choices
    # 'APP-LMS', 'APP-LM', 'CLS-LMS', 'CLS-LM', 'CLS-N', 'PPS-LMS', 'PPS-LM', 'PPS-N', 'PPS-CFO2', 'RLS-LMS', 'RLS-LM', 'SPM-LMS', 'SPM-LM', 'SPM-N', 'open'

    config_file_name = stem + 'config/config_file_IMG_04_coords.yaml'

    # file of sample data
    file_name = stem + 'IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_grid_lm_pupil_mask_open_filter_Br_alpha_clocking_angle_0.fits'
    # Pass both the filter name (key) and wavelength (value) as separate parameters
    filter_name = 'Br_alpha'
    config_file_name = stem + 'config/config_file_IMG_04_coords.yaml'
    strehl_psfs(file_name, fp_mask='grid_lm', pp_mask='open', filter_name=filter_name, wavel=observing_filters_lm[filter_name], pixel_scale_mas=pixel_scales['img_lm'], fit_simmed_psf=False, fit_analytical_psf=True, psfs_subset=1, config_file_name=config_file_name)

    # rinse and repeat
    file_name = stem + 'IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_grid_lm_pupil_mask_open_filter_Br_alpha_ref_clocking_angle_0.fits'
    filter_name = 'Br_alpha_ref'
    config_file_name = stem + 'config/config_file_IMG_04_coords.yaml'
    strehl_psfs(file_name, fp_mask='grid_lm', pp_mask='open', filter_name=filter_name, wavel=observing_filters_lm[filter_name], pixel_scale_mas=pixel_scales['img_lm'], fit_simmed_psf=False, fit_analytical_psf=True, psfs_subset=1, config_file_name=config_file_name)

    # rinse and repeat
    #file_name = stem + 'strehl/IMG_OPT_04_wcu_focal_mask_grid_lm_pupil_mask_open_filter_CO_1-0_ice_clocking_angle_0.fits'
    #filter_name = 'CO_1-0_ice'
    #config_file_name = stem + 'config/config_file_IMG_04_coords.yaml'
    #strehl_psfs(file_name, fp_mask='grid_lm', pp_mask='open', filter_name=filter_name, wavel=observing_filters_lm[filter_name], pixel_scale_mas=pixel_scales['img_lm'], fit_simmed_psf=False, fit_analytical_psf=True, psfs_subset=1, config_file_name=config_file_name)

    # rinse and repeat
    file_name = stem + 'IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_grid_lm_pupil_mask_open_filter_H2O-ice_clocking_angle_0.fits'
    filter_name = 'H2O-ice'
    config_file_name = stem + 'config/config_file_IMG_04_coords.yaml'
    strehl_psfs(file_name, fp_mask='grid_lm', pp_mask='open', filter_name=filter_name, wavel=observing_filters_lm[filter_name], pixel_scale_mas=pixel_scales['img_lm'], fit_simmed_psf=False, fit_analytical_psf=True, psfs_subset=1, config_file_name=config_file_name)

    # rinse and repeat
    #file_name = stem + 'strehl/IMG_OPT_04_wcu_focal_mask_grid_lm_pupil_mask_open_filter_L_spec_clocking_angle_0.fits'
    #filter_name = 'L_spec'
    #config_file_name = stem + 'config/config_file_IMG_04_coords.yaml'
    #strehl_psfs(file_name, fp_mask='grid_lm', pp_mask='open', filter_name=filter_name, wavel=observing_filters_lm[filter_name], pixel_scale_mas=pixel_scales['img_lm'], fit_simmed_psf=False, fit_analytical_psf=True, psfs_subset=1, config_file_name=config_file_name)

    # rinse and repeat
    file_name = stem + 'IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_grid_lm_pupil_mask_open_filter_Lp_clocking_angle_0.fits'
    filter_name = 'Lp'
    config_file_name = stem + 'config/config_file_IMG_04_coords.yaml'
    strehl_psfs(file_name, fp_mask='grid_lm', pp_mask='open', filter_name=filter_name, wavel=observing_filters_lm[filter_name], pixel_scale_mas=pixel_scales['img_lm'], fit_simmed_psf=False, fit_analytical_psf=True, psfs_subset=1, config_file_name=config_file_name)

    # rinse and repeat
    file_name = stem + 'IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_grid_lm_pupil_mask_open_filter_PAH_3.3_clocking_angle_0.fits'
    filter_name = 'PAH_3.3'
    config_file_name = stem + 'config/config_file_IMG_04_coords.yaml'
    strehl_psfs(file_name, fp_mask='grid_lm', pp_mask='open', filter_name=filter_name, wavel=observing_filters_lm[filter_name], pixel_scale_mas=pixel_scales['img_lm'], fit_simmed_psf=False, fit_analytical_psf=True, psfs_subset=1, config_file_name=config_file_name)

    # rinse and repeat
    file_name = stem + 'IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_grid_lm_pupil_mask_open_filter_PAH_3.3_ref_clocking_angle_0.fits'
    filter_name = 'PAH_3.3_ref'
    strehl_psfs(file_name, fp_mask='grid_lm', pp_mask='open', filter_name=filter_name, wavel=observing_filters_lm[filter_name], pixel_scale_mas=pixel_scales['img_lm'], fit_simmed_psf=False, fit_analytical_psf=True, psfs_subset=1, config_file_name=config_file_name)

    # rinse and repeat
    file_name = stem + 'IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_grid_lm_pupil_mask_open_filter_short-L_clocking_angle_0.fits'
    filter_name = 'short-L'
    strehl_psfs(file_name, fp_mask='grid_lm', pp_mask='open', filter_name=filter_name, wavel=observing_filters_lm[filter_name], pixel_scale_mas=pixel_scales['img_lm'], fit_simmed_psf=False, fit_analytical_psf=True, psfs_subset=1, config_file_name=config_file_name)


if __name__ == "__main__":
    main()