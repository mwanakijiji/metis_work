import io
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.visualization import ZScaleInterval
from scipy.ndimage import zoom
from scipy.special import j1
from scipy.optimize import curve_fit
from skimage import measure
import scopesim as sim
import yaml
from scipy.signal import convolve2d

import ipdb

def load_config_and_pipe(config_file_choice, print_one_line=False):
    '''
    Load a config file and print its contents to the log

    INPUTS:
    config_file_choice: the path to the config file
    print_one_line: whether to print everything on one line in the log

    OUTPUTS:
    config_this: the config dictionary
    '''

    with open(config_file_choice, "r") as config_file:

        logging.info('--------------------------------')
        config_this = yaml.safe_load(config_file)

        logging.info(f'Loading config file: {config_file_choice}')

        # print stuff to log
        if print_one_line:
            # print everything on one line
            logging.info(f'Config data: {config_this}')
        else:
            # print in more readable format
            logging.info("Config data:")
            for key, value in config_this.items():
                if isinstance(value, (list, tuple, dict)):
                    logging.info(f"\t{key}:")
                    if isinstance(value, dict):
                        for subkey, subval in value.items():
                            logging.info(f"\t  {subkey}: {subval}")
                    else:  # It's a list or tuple
                        for item in value:
                            logging.info(f"\t  {item}")
                else:
                    logging.info(f"\t{key}: {value}")

    return config_this


def pipe_2_log(callable_func, msg="Output"):
    '''
    Capture stdout from any ScopeSim callable and write each line to the log.

    INPUTS:
    - callable_func: callable (no args) that prints to stdout when invoked
    - msg: string header to add to the log

    OUTPUTS:
    - None; writes out to log

    Example:
        pipe_2_log(lambda m=metis: m.effects.pprint_all(), msg="Optical train effects")
    '''
    buffer = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buffer
        callable_func()
        output = buffer.getvalue()
    finally:
        sys.stdout = old_stdout
    logging.info('--------------------------------')
    logging.info(msg)
    for line in output.rstrip().splitlines():
        logging.info(line)


def jinc(x):
    x = np.asarray(x)
    y = np.empty_like(x, dtype=float)
    mask = x != 0
    y[mask] = j1(x[mask]) / x[mask]
    y[~mask] = 0.5
    return y


def intensity_annular_aperture(r_rad_array, wavel, D_aperture, D_obscuration, ampl=1, pinhole_size=None):
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

    # convolve with a pinhole, if we're using one of finite size
    if pinhole_size is not None:
        # pixel scale in radians (same units as r_rad_array)
        rad_per_pix = r_rad_array[0, 1] - r_rad_array[0, 0] ## ## TODO: MAKE THIS MORE ELEGANT
        # fractional pixels: linear ramp at boundary so edge pixels get value in (0, 1)
        pinhole_array = np.clip(
            (pinhole_size - r_rad_array + rad_per_pix / 4) / rad_per_pix,
            0, 1
        )
        # convolve with the pinhole array
        I_r = convolve2d(I_r, pinhole_array, mode='same')

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
def _filter_curve_from_filter_file(filter_file):
    '''
    Reads in a filter curve and returns a DataFrame with a decimation to sample that curve

    INPUTS:
    filter_file: the absolutepath to the filter curve file

    OUTPUTS:
    decimated_df: a DataFrame with a decimation to sample that curve
    '''

    df_filter = pd.read_csv(
        filter_file,
        comment='#',
        sep='\s+'
        )

    # rename columns to wavel_um and trans
    df_filter.rename(columns={"wavelength": "wavel_um", "transmission": "trans"}, inplace=True)

    # find region where transmission is greater than 0.2 of the maximum transmission
    max_trans = df_filter["trans"].max()
    mask_trans_gt_05_max = df_filter["trans"] > 0.2 * max_trans
    df_filter = df_filter[mask_trans_gt_05_max]

    # choose N evenly-spaced rows from df_filter (as a smaller DataFrame)
    N_slices = 5
    slice_idx = np.linspace(0, len(df_filter) - 1, N_slices, dtype=int)
    decimated_df = df_filter.iloc[slice_idx].reset_index(drop=True)
    
    return decimated_df


def model_for_fit_fixed(r_rad_1d, D_aperture, D_obscuration, ampl, baseline_shape, valid_mask, *, wavel=None, filter_file=None, pinhole_size=None, fac_oversamp=1):
    """
    Wrapper function for intensity_annular_aperture to use with curve_fit.
    
    Parameters:
    - r_rad_1d: 1D array of radial distances (masked, only valid points)
    - D_aperture: aperture diameter (meters)
    - D_obscuration: obscuration diameter (meters)
    - ampl: amplitude
    - baseline_shape: tuple, shape of the 2D array (fixed, not optimized)
    - valid_mask: boolean array, mask for valid data points (fixed, not optimized)
    - fac_oversamp: oversampling factor
    - wavel: wavelength (meters); for monochromatic PSFs
    - filter_file: filter curve file to make polychromatic PSFs; for more realistic PSFs
    - pinhole_size: size of the pinhole in pixels (if None, the analytical expression for the PSF alone is used; this is equivalent to a pinhole delta function)
    
    Returns:
    - 1D array of intensity values (masked, same length as input)
    """

    # Reconstruct the full 2D array by inserting masked values back into their original positions
    r_rad_2d_full = np.full(baseline_shape, np.nan).flatten()
    r_rad_2d_full[valid_mask] = r_rad_1d
    r_rad_2d = r_rad_2d_full.reshape(baseline_shape)

    # read in the filter curve    
    if wavel: # monochromatic PSF
        intensity_2d = intensity_annular_aperture(
            r_rad_array=r_rad_2d, 
            wavel=wavel, 
            D_aperture=D_aperture, 
            D_obscuration=D_obscuration, 
            ampl=ampl, 
            pinhole_size=pinhole_size
        )
    else: # polychromatic; requires a filter curve
        decimated_filter_curve_df = _filter_curve_from_filter_file(filter_file)
        intensity_2d = np.zeros_like(r_rad_2d) # init
        for idx, row in decimated_filter_curve_df.iterrows():

            wavel_um = row['wavel_um'] * 1e-6
            trans = row['trans']

            # make a PSF for this wavelength and multiply by the transmission before adding
            intensity_2d += trans * intensity_annular_aperture(
                r_rad_array=r_rad_2d, 
                wavel=wavel_um, 
                D_aperture=D_aperture, 
                D_obscuration=D_obscuration, 
                ampl=ampl, 
                pinhole_size=pinhole_size
            )
            # debug: save as FITS file
            #fits.writeto(f'junk.fits', intensity_2d, overwrite=True)
        # renormalize to the desired amplitude
        intensity_2d = ampl * intensity_2d / np.nanmax(intensity_2d) 
    
    # Flatten and apply the same mask to return only valid points
    intensity_1d_full = intensity_2d.flatten()
    return intensity_1d_full[valid_mask]


def angle_from_center_2d(array_passed_in, y_center, x_center, pixel_scale_mas, fac_oversamp, units='radians'):
    '''
    Create a 2D array of distances from the center in radians or arcseconds
    N.b. the input array is already assumed to be oversampled; the fac_oversamp here just is for rescaling the pixel values

    array_passed_in: the array to create the 2D array of distances from the center in arcseconds from
    y_center: the y-center of the array
    x_center: the x-center of the array
    pixel_scale_mas: the pixel scale in mas
    fac_oversamp: the oversampling factor

    OUTPUTS:
    r_rad_2d: the 2D array of distances from the center in arcseconds
    '''

    size = array_passed_in.shape[0] 
    baseline_shape = (size, size)
    pixel_scale_arcsec = pixel_scale_mas / 1000.0  # arcseconds per pixel
    y, x = np.indices((size, size))
    center = (y_center, x_center)
    r_pix = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    test_array_arcsec_2d = r_pix * pixel_scale_arcsec # note this pixel scale is the physical one

    if units == 'radians':
        # convert to radians
        r_rad_2d = test_array_arcsec_2d / ( 3600 * (180/np.pi ) )
    elif units == 'arcseconds':
        r_rad_2d = test_array_arcsec_2d
    else:
        raise ValueError(f"Invalid units: {units}. Must be 'radians' or 'arcseconds'.")

    # rescale based on the oversampling factor to fit the input array (this effectively makes the plate scale smaller)
    r_rad_2d = r_rad_2d / fac_oversamp

    return r_rad_2d


def mtf_arrays(array_empirical, array_model, config_observing, fac_oversamp, size, filter_name):

    pad_width = size // 2
    model_annular_2d_full_norm_padded = np.pad(
        array_model,
        pad_width=pad_width,
        mode="constant",
        constant_values=0.0
    )
    cookie_cut_out_sci_padded = np.pad(
        array_empirical,
        pad_width=pad_width,
        mode="constant",
        constant_values=0.0
    )
    #ipdb.set_trace()
    fft_model = np.fft.fftshift(np.fft.fft2(model_annular_2d_full_norm_padded))
    fft_model_power = np.abs(fft_model)
    fft_empirical = np.fft.fftshift(np.fft.fft2(cookie_cut_out_sci_padded))
    fft_empirical_power = np.abs(fft_empirical)
    #ipdb.set_trace()
    # Build frequency grid (cycles per radian) and apply diffraction cutoff
    rad_per_pix = ((config_observing["pixel_scales"]["img_lm"] / 1000.0) / 206265.0) / fac_oversamp
    n_fft = model_annular_2d_full_norm_padded.shape[0]
    fy = np.fft.fftshift(np.fft.fftfreq(n_fft, d=rad_per_pix))
    fx = np.fft.fftshift(np.fft.fftfreq(n_fft, d=rad_per_pix))
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    f_r = np.sqrt(fx_grid**2 + fy_grid**2)
    cutoff_freq = config_observing["D_aperture"]["full"] / config_observing["monochromatic_observing_filters_lm"][filter_name] ## ## TODO: IS THIS RIGHT?
    mtf_cutoff_mask = f_r <= cutoff_freq
    #ipdb.set_trace()
    fft_model_power_cutoff = fft_model_power * mtf_cutoff_mask
    fft_empirical_power_cutoff = fft_empirical_power * mtf_cutoff_mask

    return fft_model_power_cutoff, fft_empirical_power_cutoff, cutoff_freq, fx, fy, n_fft

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
    plt.savefig(f"figs_dump/{plot_filename}", bbox_inches='tight')
    plt.close()
    logging.info(f'Figure saved as {plot_filename}')
    #plt.show()

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
    plot_filename = f"fyi_plot_centroiding_{title_string}.png"
    plt.savefig(f"figs_dump/{plot_filename}", bbox_inches='tight')
    logging.info(f"Saved {plot_filename} to figs_dump/")
    plt.close()


def fit_gaussian_psf(cookie_cut_out_sci, obs_filter, fp_mask, pp_mask, coords_guess, plot_string, fac_oversamp):
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

    logging.info('--------------------------------')
    logging.info('Calculating coordinates and Strehl from Gaussian best-fit')

    ## ## TO DO: ARE THE INDEXES RIGHT HERE?
    cookie_cut_out_best_fit, x_center_pix_oversamp_cutout, y_center_pix_oversamp_cutout, fwhm_x_pix, fwhm_y_pix, sigma_x_pix, sigma_y_pix, angle_theta_deg, amplitude_counts = fit_gaussian(cookie_cut_out_sci, \
        center_guess = coords_guess)
    residuals = cookie_cut_out_sci - cookie_cut_out_best_fit

    # strehl based on the Gaussian fit
    gaussian_based_strehl = np.max(cookie_cut_out_sci) / np.max(cookie_cut_out_best_fit)
    #print(f'Observing filter: {obs_filter}')
    #print(f'PSF ID: {plot_string}')
    #print(f'Focal plane mask: {fp_mask}')
    #print(f'Pupil plane mask: {pp_mask}')
    logging.info(f'Strehl from Gaussian best-fit: {gaussian_based_strehl:.2f}')


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
    plt.savefig(f"figs_dump/{plot_filename}", bbox_inches='tight')
    logging.info(f'Figure saved as {plot_filename}')
    plt.close()
    #ipdb.set_trace()

    return x_center_pix_oversamp_cutout, y_center_pix_oversamp_cutout, fwhm_x_pix, fwhm_y_pix, amplitude_counts, gaussian_based_strehl


def fit_simmed_psfs(cookie_cut_out_sci, data_empirical_original, plot_string, obs_filter, fp_mask, pp_mask, x_center_final_oversamp, y_center_final_oversamp, fac_oversamp):
    '''
    Find FWHM of a PSF using a perfect PSF from ScopeSim
    
    INPUTS:
    cookie_cut_out_sci: 2D array of the science frame
    data_empirical_original: 2D array of the original empirical data
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

    metis.effects.pprint_all()

    bb_temp = 1000 * u.K
    NDIT, EXPTIME = 1, 0.2


    logging.info('--------------------------------')
    logging.info('Current Observing filter:', obs_filter)
    logging.info('Current WCU FP mask:', wcu.fpmask)
    logging.info('Current WCU PP mask:', pp_mask)
    #ipdb.set_trace()
    # background
    logging.info('Closing WCU BB aperture first to get a background ...')
    # background
    wcu.set_bb_aperture(value = 0.0)
    metis.observe()
    outhdul_off = metis.readout(ndit = NDIT, exptime = EXPTIME)[0]
    background = outhdul_off[1].data

    logging.info('Re-opening WCU BB aperture to get a PSF ...')
    wcu.set_bb_aperture(value = 1.0) # open BB source

    #metis["filter_wheel"].change_filter(obs_filter)

    logging.info('--------------------------------')
    logging.info('Current Observing filter:', obs_filter)
    logging.info('Current WCU FP mask:', wcu.fpmask)
    logging.info('Current WCU PP mask:', pp_mask)
    logging.info('Opening WCU BB aperture...')

    metis.observe()
    outhdul_on = metis.readout(ndit = NDIT, exptime = EXPTIME)[0]
    sci = outhdul_on[1].data
    #ipdb.set_trace()
    # Get perfect, background-subtracted PSF - no detector noise
    psf_perfect = sci - background

    # Oversample the background-subtracted PSF to match the cookie_cut_out_sci oversampling
    psf_perfect_oversamp = zoom(psf_perfect, fac_oversamp, order=3)

    # for debugging
    file_name_plot = "psf_perfect_oversamp.fits"
    fits.writeto(file_name_plot, psf_perfect_oversamp, overwrite=True)
    logging.info("Saved " + file_name_plot + " for checking.")


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

    strehl_from_simmed_psf = np.max(cookie_cut_out_sci) / np.max(psf_perfect_cutout_best_fit)
    logging.info(f'Strehl from Scopesim simmed PSF: {strehl_from_simmed_psf:.2f}')

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
    plot_filename = "junk_psf_perfect_cutout_best_fit.png"
    plt.savefig(f"figs_dump/{plot_filename}", bbox_inches="tight")
    logging.info(f"Saved {plot_filename}")

    return psf_perfect_cutout_best_fit, strehl_from_simmed_psf