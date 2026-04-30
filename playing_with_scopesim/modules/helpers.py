import io
import logging
import os
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
from skimage.measure import block_reduce
from scipy.ndimage import shift
import ipdb



def setup_logging(log_dir, log_file_name, now=None):
    '''
    Initialize logging with a file + console handler and pipe basic metadata.

    INPUTS
    ----------
    log_dir : str
        Directory where the log file should be written.
    log_file_name : str
        Full path to the output log file.
    now : datetime.datetime, optional
        Timestamp used for the "created at" log line. If None, that line is skipped.

    OUTPUTS
    -------
    None
        Configures the root logger and writes startup log lines.
    '''
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_name),
            logging.StreamHandler()
        ],
        force=True
    )
    if now is not None:
        logging.info(f'Log file created at {now.strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info(f'Log file name: {log_file_name}')
    logging.info(f'Log file directory: {log_dir}')


def load_config_and_pipe(config_file_choice, print_one_line=False):
    '''
    Load a YAML configuration file and pipe its contents to the log.

    INPUTS
    ----------
    config_file_choice : str
        Path to the configuration file.
    print_one_line : bool, optional
        If True, log the entire config on a single line. If False (default), log
        in a more readable, multiline format.

    OUTPUTS
    -------
    config_this : dict
        The configuration dictionary loaded from file.
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
    Capture stdout from any ScopeSim callable and pipe each line to the log.

    Parameters
    ----------
    callable_func : callable
        A zero-argument callable that prints to stdout when invoked.
    msg : str, optional
        String header to add to the log. Default is "Output".

    Returns
    -------
    None
        Writes output to the logging system.

    Example
    -------
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
    '''
    Calculate the Jinc function.

    INPUTS
    ----------
    x : array-like
        The input array.

    OUTPUTS
    -------
    y : array-like
        The Jinc function evaluated at the input array.
    '''
    x = np.asarray(x)
    y = np.empty_like(x, dtype=float)
    mask = x != 0
    y[mask] = j1(x[mask]) / x[mask]
    y[~mask] = 0.5
    return y


def intensity_annular_aperture(
    r_rad_array,
    wavel,
    D_aperture,
    D_obscuration,
    ampl=1,
    pinhole_diam_rad=None,
    pixel_scale_mas=5.47,
    fac_oversamp=1,
    save_fyi_plot=True,
    results_write_dir="figs_dump",
):
    '''
    Calculate the intensity through an aperture with a central obscuration
    Ref. 'E-REP-MPIA-1203 0-1 xx-10-2024', Sec. 4.4

    PARAMETERS
    ----------
    r_rad_array : ndarray
        2D array of radial distances from the center (in radians).
    wavel : float
        Wavelength (in meters).
    D_aperture : float
        Aperture diameter (in meters).
    D_obscuration : float
        Central obscuration diameter (in meters).
    ampl : float, optional
        Amplitude scaling factor for the intensity (default 1).
    pinhole_diam_rad : float or None, optional
        Size of the pinhole in radians (if None, only the analytical PSF is used; equivalent to a pinhole delta function).
    pixel_scale_mas : float, optional
        Detector pixel scale in mas/pixel (default 5.47).
    fac_oversamp : float or int, optional
        Oversampling factor for the model grid (default 1).
    save_fyi_plot : bool, optional
        If True, save an FYI plot when using a finite pinhole (default True).
    results_write_dir : str, optional
        Directory to store output plots (default "figs_dump").

    RETURNS
    -------
    I_r_array : ndarray
        2D array of intensity on the detector.
    '''

    nu_ = np.pi * r_rad_array * D_aperture / wavel # unitless

    eps_ = D_obscuration / D_aperture # unitless
    
    # see Eqn. 43 in 'E-REP-MPIA-1203 0-1 xx-10-2024'
    I_r = (1/(1-eps_**2)**2) * ( (2*jinc(nu_)) - eps_**2 * (2*jinc(nu_*eps_)) ) ** 2

    # convolve with a pinhole, if we're using one of finite size
    if pinhole_diam_rad is not None:
        # pixel scale in radians (same units as r_rad_array)
        rad_per_pix = ((pixel_scale_mas / 1000.0) / 206265.0) / fac_oversamp # this is for 'pixels' in the oversampled grid

        # Shift I_r so that its maximum is exactly at the center of the array
        
        # Find the coordinates of the maximum of I_r
        max_pos = np.unravel_index(np.nanargmax(I_r), I_r.shape)
        center_pos = ((I_r.shape[0] - 1) / 2.0, (I_r.shape[1] - 1) / 2.0)
        # Compute the shift required to move the max to the center
        shift_vals = (center_pos[0] - max_pos[0], center_pos[1] - max_pos[1])
        # Perform the shift
        I_r = shift(I_r, shift_vals, order=3, mode='nearest')

        ny, nx = r_rad_array.shape
        # Pixel coordinate grids with origin at array center
        y = np.arange(ny) - (ny - 1) / 2.0
        x = np.arange(nx) - (nx - 1) / 2.0
        # Radius in pixels, then convert to radians using rad_per_pix
        r_pix = np.sqrt(x[None, :]**2 + y[:, None]**2)
        r_rad_centered = r_pix * rad_per_pix
        # Hard pinhole (delta-function-like on the grid)
        # = (r_rad_centered <= pinhole_diam_rad).astype(float)

        # Approximate the circle-square overlap with subpixel sampling.
        radius_pix = 0.5 * pinhole_diam_rad / rad_per_pix
        n_sub = 8
        subpix_offsets = (np.arange(n_sub) + 0.5) / n_sub - 0.5
        x_sub = x[None, :, None, None] + subpix_offsets[None, None, None, :]
        y_sub = y[:, None, None, None] + subpix_offsets[None, None, :, None]
        r_sub_sq = x_sub**2 + y_sub**2
        pinhole_array = (r_sub_sq <= radius_pix**2).mean(axis=(-1, -2))

        if save_fyi_plot:
            # make an FYI plot, superimposing a circle with radius pinhole_diam_rad
            os.makedirs(results_write_dir, exist_ok=True)
            fig, ax = plt.subplots()
            im = ax.imshow(pinhole_array, origin='lower', cmap='gray_r')
            circle = plt.Circle(
                ((nx - 1) / 2.0, (ny - 1) / 2.0),
                0.5 * pinhole_diam_rad / rad_per_pix,
                fill=False,
                color='red',
                linewidth=1.5,
            )
            # Overplot dashed vertical and horizontal lines every N=fac_oversamp pixels
            for v in np.arange(2.5, nx+2.5, fac_oversamp):
                ax.axvline(x=v, color='blue', linestyle='--', linewidth=0.7, alpha=0.5)
            for h in np.arange(2.5, nx+2.5, fac_oversamp):
                ax.axhline(y=h, color='blue', linestyle='--', linewidth=0.7, alpha=0.5)
            plt.xlabel('x (oversampled pixels)')
            plt.xlabel('y (oversampled pixels)')
            ax.add_patch(circle)
            fig.colorbar(im, ax=ax)
            ax.set_title(
                f'Pinhole footprint weights on oversampled PSF grid\n'
                f'Blue: native-pixel boundaries; red: pinhole diameter'
            )
            file_name_plot = os.path.join(results_write_dir, 'pinhole_array_FYI.png')
            plt.savefig(file_name_plot)
            logging.info('Saved plot of pinhole array as ' + file_name_plot)

        # convolve with the pinhole array
        I_r = convolve2d(I_r, pinhole_array, mode='same')

        # shift I_r back to the original position
        I_r = shift(I_r, -np.array(shift_vals), order=3, mode='nearest')

    # normalize to the amplitude
    I_r = ampl * I_r / np.nanmax(I_r)

    return I_r


def gaussian_2d(xy_mesh, amplitude, xo, yo, sigma_x_pix, sigma_y_pix, theta):
    '''
    Calculate a 2D Gaussian function.

    INPUTS
    ----------
    xy_mesh : ndarray
        2D array of x and y coordinates.
    amplitude : float
        Amplitude of the Gaussian.
    xo : float
        X-coordinate of the center of the Gaussian.
    yo : float
        Y-coordinate of the center of the Gaussian.
    sigma_x_pix : float
        Standard deviation of the Gaussian in the x-direction (in pixels).
    sigma_y_pix : float
        Standard deviation of the Gaussian in the y-direction (in pixels).
    theta : float
        Rotation angle of the Gaussian (in radians).

    OUTPUTS
    -------
    g : ndarray
        The Gaussian function, flattened to 1D, evaluated at the input coordinates.
    '''
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
    Load a filter transmission curve from file and return a decimated DataFrame for sampling.

    Parameters
    ----------
    filter_file : str
        Absolute path to the filter curve file.

    Returns
    -------
    decimated_df : pandas.DataFrame
        DataFrame containing a decimated sample of the filter curve.
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


def model_for_fit_fixed(
    r_rad_1d_original,
    D_aperture,
    D_obscuration,
    ampl,
    centroid_yx_original=None,
    shape_original_2d=None,
    fac_oversamp=None,
    pixel_scale_mas=None,
    *,
    wavel=None,
    filter_file=None,
    pinhole_diam_rad=None,
    centroid_yx_oversamp=None,
    shape_oversamp=None,
    valid_mask=None,
    save_fyi_plot=True,
    results_write_dir="figs_dump",
):
    '''
    Generates a 1D model intensity array suitable for fitting routines,
    by calling intensity_annular_aperture on an oversampled PSF model.

    INPUTS
    ----------
    r_rad_1d_original : array-like
        Placeholder radial distance or flat index array for compatibility with curve_fit.
        Not used directly; required by curve_fit interface.
    D_aperture : float
        Aperture diameter in meters.
    D_obscuration : float
        Central obscuration diameter in meters.
    ampl : float
        Overall intensity normalization.
    centroid_yx_original : tuple of float, optional
        (y, x) centroid coordinates in native image/cookie-cutout pixel grid.
        Used to compute the oversampled centroid if centroid_yx_oversamp not given.
    shape_original_2d : tuple of int, optional
        Shape of the native image/cookie-cutout array (before oversampling).
        If provided, the model is rebinned to this shape before flattening for fitting.
    fac_oversamp : float or int
        Oversampling factor (oversampled pixels per detector pixel).
    pixel_scale_mas : float
        Pixel scale in milliarcseconds (mas).
    wavel : float, optional
        Wavelength in meters (for monochromatic PSFs).
    filter_file : str, optional
        Path to filter transmission curve file (for polychromatic PSFs).
    pinhole_diam_rad : float, optional
        Pinhole diameter in radians. If None, a delta-function pinhole is assumed.
    centroid_yx_oversamp : tuple of float, optional
        (y, x) centroid coordinates in the oversampled pixel grid.
        If not provided, computed from centroid_yx_original and fac_oversamp.
    shape_oversamp : tuple of int, optional
        Explicit shape for the oversampled array.
        Used if shape_original_2d is not provided. In this case, output is not rebinned.
    valid_mask : array-like, optional
        (Legacy/unused argument for interface compatibility.)
    save_fyi_plot : bool, optional
        If True, enables diagnostic plotting. Default is True.
    results_write_dir : str, optional
        Directory for outputting diagnostic plots.

    OUTPUTS
    -------
    model_1d : numpy.ndarray
        A 1D array of model PSF intensity values mapped on the requested output grid,
        suitable for passing to fitting routines.
    '''

    if shape_original_2d is not None:
        shape_oversamp_2d = tuple(int(dim * fac_oversamp) for dim in shape_original_2d)
    elif shape_oversamp is not None:
        shape_oversamp_2d = tuple(shape_oversamp)
    else:
        raise ValueError("Need shape_original_2d or shape_oversamp.")

    if centroid_yx_oversamp is None:
        if centroid_yx_original is None:
            raise ValueError("Need centroid_yx_original or centroid_yx_oversamp.")
        centroid_yx_oversamp = tuple(
            ((coord + 0.5) * fac_oversamp) - 0.5 for coord in centroid_yx_original
        )

    r_rad_2d_oversamp = angle_from_center_2d(
        array_passed_in=np.zeros(shape_oversamp_2d, dtype=float),
        y_center=centroid_yx_oversamp[0],
        x_center=centroid_yx_oversamp[1],
        pixel_scale_mas=pixel_scale_mas,
        fac_oversamp=fac_oversamp,
        units='radians',
    )



    # generate model intensities on oversampled array
    if wavel: # monochromatic PSF
        intensity_2d = intensity_annular_aperture(
            r_rad_array=r_rad_2d_oversamp, 
            wavel=wavel, 
            D_aperture=D_aperture, 
            D_obscuration=D_obscuration, 
            ampl=ampl, 
            pinhole_diam_rad=pinhole_diam_rad,
            pixel_scale_mas=pixel_scale_mas,
            fac_oversamp=fac_oversamp,
            save_fyi_plot=save_fyi_plot,
            results_write_dir=results_write_dir,
        )
    else: # polychromatic; reads in a filter curve
        decimated_filter_curve_df = _filter_curve_from_filter_file(filter_file)
        intensity_2d = np.zeros_like(r_rad_2d_oversamp) # init
        for idx, row in decimated_filter_curve_df.iterrows():

            wavel_um = row['wavel_um'] * 1e-6
            trans = row['trans']

            # make a PSF for this wavelength and multiply by the transmission before adding
            intensity_2d += trans * intensity_annular_aperture(
                r_rad_array=r_rad_2d_oversamp, 
                wavel=wavel_um, 
                D_aperture=D_aperture, 
                D_obscuration=D_obscuration, 
                ampl=ampl, 
                pinhole_diam_rad=pinhole_diam_rad,
                pixel_scale_mas=pixel_scale_mas,
                fac_oversamp=fac_oversamp,
                save_fyi_plot=save_fyi_plot and (idx == 0),
                results_write_dir=results_write_dir,
            )
            # debug: save as FITS file
            #fits.writeto(f'junk.fits', intensity_2d, overwrite=True)
        # renormalize to the desired amplitude
        intensity_2d = ampl * intensity_2d / np.nanmax(intensity_2d) 

    if shape_original_2d is None:
        return intensity_2d.flatten()

    intensity_2d_model_original_scale = block_reduce(
        intensity_2d, block_size=(fac_oversamp, fac_oversamp), func=np.mean
    )

    return intensity_2d_model_original_scale.flatten()


def angle_from_center_2d(array_passed_in, y_center, x_center, pixel_scale_mas, fac_oversamp, units='radians'):
    '''
    Generate a 2D array giving the distance from a specified center for each element,
    in radians or arcseconds.

    Parameters
    ----------
    array_passed_in : ndarray
        Input array used to define the output shape (2D; should be oversampled).
    y_center : float
        Y-coordinate of the central position of the PSF.
    x_center : float
        X-coordinate of the central position of the PSF.
    pixel_scale_mas : float
        Pixel scale in milliarcseconds (mas).
    fac_oversamp : float or int
        Oversampling factor used for the array.
    units : str, optional
        Output units; either "radians" (default) or "arcseconds".

    Returns
    -------
    r_rad_2d : ndarray
        2D array of distances from the center in the chosen units.
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
    '''
    Compute the Modulation Transfer Function (MTF) arrays for the empirical and model PSFs.

    INPUTS
    ----------
    array_empirical : ndarray
        2D empirical PSF array.
    array_model : ndarray
        2D model PSF array.
    config_observing : dict
        Dictionary with observing configuration parameters.
    fac_oversamp : float or int
        Oversampling factor used.
    size : int
        Array size (assumed square).
    filter_name : str
        Observing filter key to use for wavelength info.

    OUTPUTS
    -------
    fft_model_power_cutoff : ndarray
        Power spectrum of the model, masked by cutoff frequency.
    fft_empirical_power_cutoff : ndarray
        Power spectrum of the empirical PSF, masked by cutoff frequency.
    cutoff_freq : float
        Diffraction cutoff frequency (cycles per radian).
    fx : ndarray
        X axis values (frequency, cycles per radian).
    fy : ndarray
        Y axis values (frequency, cycles per radian).
    n_fft : int
        FFT grid size.
    '''

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
    fft_model = np.fft.fftshift(np.fft.fft2(model_annular_2d_full_norm_padded))
    fft_model_power = np.abs(fft_model)
    fft_empirical = np.fft.fftshift(np.fft.fft2(cookie_cut_out_sci_padded))
    fft_empirical_power = np.abs(fft_empirical)
    # Build frequency grid (cycles per radian) and apply diffraction cutoff
    rad_per_pix = ((config_observing["pixel_scales"]["img_lm"] / 1000.0) / 206265.0) / fac_oversamp
    n_fft = model_annular_2d_full_norm_padded.shape[0]
    fy = np.fft.fftshift(np.fft.fftfreq(n_fft, d=rad_per_pix))
    fx = np.fft.fftshift(np.fft.fftfreq(n_fft, d=rad_per_pix))
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    f_r = np.sqrt(fx_grid**2 + fy_grid**2)
    cutoff_freq = config_observing["D_aperture"]["full"] / config_observing["monochromatic_observing_filters_lm"][filter_name] ## ## TODO: IS THIS RIGHT?
    mtf_cutoff_mask = f_r <= cutoff_freq
    fft_model_power_cutoff = fft_model_power * mtf_cutoff_mask
    fft_empirical_power_cutoff = fft_empirical_power * mtf_cutoff_mask

    return fft_model_power_cutoff, fft_empirical_power_cutoff, cutoff_freq, fx, fy, n_fft

def fit_empirical_fwhm(frame, plot_string, results_write_dir="figs_dump"):
    '''
    Calculate the FWHM in x and y by finding the region where the intensity is at least 50% of the peak value.

    INPUTS
    ----------
    frame : np.ndarray
        2D array of the frame.
    plot_string : str
        String to append to the output plot filename.
    results_write_dir : str, optional
        Directory to write the output plot to. Default is "figs_dump".

    OUTPUTS
    -------
    height_y : float
        FWHM size in the y direction (in pixels).
    width_x : float
        FWHM size in the x direction (in pixels).
    '''

    # find the peak intensity
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
    plt.title(
        f'Empirical PSF half-maximum bounding box\n'
        f'FWHM_x={width_x:.2f} pix, FWHM_y={height_y:.2f} pix'
    )
    # save the plot to file
    plot_filename = 'empirical_fwhm_' + plot_string + '.png'
    os.makedirs(results_write_dir, exist_ok=True)
    plt.savefig(os.path.join(results_write_dir, plot_filename), bbox_inches='tight')
    plt.close()
    logging.info(f'Figure saved as {plot_filename}')

    return height_y, width_x


def fit_gaussian(frame, center_guess):
    '''
    Fit a 2D Gaussian to a given frame and return the fitted model and parameters. (This is the '2nd-pass' centroiding.)

    INPUTS
    ----------
    frame : np.ndarray
        2D input array to fit.
    center_guess : list or tuple
        Initial guess [y0, x0] for the center of the Gaussian.

    OUTPUTS
    -------
    fitted_array : np.ndarray
        2D array of the best-fit Gaussian.
    x_center_pix : float
        Fitted Gaussian x-center coordinate [pixels].
    y_center_pix : float
        Fitted Gaussian y-center coordinate [pixels].
    fwhm_x_pix : float
        FWHM of the Gaussian in the x direction [pixels].
    fwhm_y_pix : float
        FWHM of the Gaussian in the y direction [pixels].
    sigma_x_pix : float
        Standard deviation of the Gaussian along x [pixels].
    sigma_y_pix : float
        Standard deviation of the Gaussian along y [pixels].
    angle_theta_deg : float
        Rotation angle of Gaussian [degrees].
    amplitude_counts : float
        Fitted peak amplitude [counts].
    '''
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


def fit_psf_gaussian_from_native_array(
    original_array,
    oversample_factor,
    coords_xy_1st_pass_normsamp=None,
    edge_size_oversamp=30,
):
    '''
    Oversample a PSF image, fit a 2D Gaussian on an oversampled cutout, and map
    fitted centroid/FWHM values back to native sampling.

    INPUTS
    ----------
    original_array : np.ndarray
        Native-sampled 2D PSF image.
    oversample_factor : int or float
        Oversampling factor.
    coords_xy_1st_pass_normsamp : sequence of 2 floats, optional
        First-pass centroid guess in native sampling as [x, y]. If None, the image
        center is used.
    edge_size_oversamp : int or None, optional
        Cutout size (in oversampled pixels). If None, the full oversampled array is
        used for fitting.

    OUTPUTS
    -------
    dict
        Dictionary containing oversampled arrays, fitted Gaussian products, and
        centroid/FWHM/amplitude values in both oversampled and native sampling.
    '''
    oversampled_array = zoom(original_array, oversample_factor, order=3)

    if coords_xy_1st_pass_normsamp is None:
        coords_xy_1st_pass_normsamp = [
            original_array.shape[1] / 2,
            original_array.shape[0] / 2,
        ]
    coords_xy_1st_pass_oversamp_fullarray = oversample_factor * np.array(
        [coords_xy_1st_pass_normsamp[0], coords_xy_1st_pass_normsamp[1]]
    )

    if edge_size_oversamp is None:
        idx_cutout_oversamp_x1 = 0
        idx_cutout_oversamp_y1 = 0
        cookie_cut_out_sci_oversamp = oversampled_array
        coords_guess_xy_cutout_oversamp = [
            coords_xy_1st_pass_oversamp_fullarray[0],
            coords_xy_1st_pass_oversamp_fullarray[1],
        ]
    else:
        idx_cutout_oversamp_x1 = int(
            coords_xy_1st_pass_oversamp_fullarray[0] - edge_size_oversamp / 2
        )
        idx_cutout_oversamp_x2 = int(
            coords_xy_1st_pass_oversamp_fullarray[0] + edge_size_oversamp / 2
        )
        idx_cutout_oversamp_y1 = int(
            coords_xy_1st_pass_oversamp_fullarray[1] - edge_size_oversamp / 2
        )
        idx_cutout_oversamp_y2 = int(
            coords_xy_1st_pass_oversamp_fullarray[1] + edge_size_oversamp / 2
        )
        cookie_cut_out_sci_oversamp = oversampled_array[
            idx_cutout_oversamp_y1:idx_cutout_oversamp_y2,
            idx_cutout_oversamp_x1:idx_cutout_oversamp_x2,
        ]
        coords_guess_xy_cutout_oversamp = [
            coords_xy_1st_pass_oversamp_fullarray[0] - idx_cutout_oversamp_x1,
            coords_xy_1st_pass_oversamp_fullarray[1] - idx_cutout_oversamp_y1,
        ]

    (
        cookie_cut_out_best_fit,
        x_center_pix_oversamp_cutout,
        y_center_pix_oversamp_cutout,
        fwhm_x_pix_oversamp_cutout,
        fwhm_y_pix_oversamp_cutout,
        sigma_x_pix_oversamp_cutout,
        sigma_y_pix_oversamp_cutout,
        angle_theta_deg,
        amplitude_counts_oversamp_cutout,
    ) = fit_gaussian(
        cookie_cut_out_sci_oversamp,
        center_guess=coords_guess_xy_cutout_oversamp,
    )

    x_center_pix_fullarray_oversamp = x_center_pix_oversamp_cutout + idx_cutout_oversamp_x1
    y_center_pix_fullarray_oversamp = y_center_pix_oversamp_cutout + idx_cutout_oversamp_y1

    x_center_pix_fullarray_normsamp = x_center_pix_fullarray_oversamp / oversample_factor
    y_center_pix_fullarray_normsamp = y_center_pix_fullarray_oversamp / oversample_factor
    fwhm_x_pix_fullarray_normsamp = fwhm_x_pix_oversamp_cutout / oversample_factor
    fwhm_y_pix_fullarray_normsamp = fwhm_y_pix_oversamp_cutout / oversample_factor

    return {
        "oversampled_array": oversampled_array,
        "coords_xy_1st_pass_oversamp_fullarray": coords_xy_1st_pass_oversamp_fullarray,
        "cookie_cut_out_sci_oversamp": cookie_cut_out_sci_oversamp,
        "cookie_cut_out_best_fit": cookie_cut_out_best_fit,
        "x_center_pix_fullarray_oversamp": x_center_pix_fullarray_oversamp,
        "y_center_pix_fullarray_oversamp": y_center_pix_fullarray_oversamp,
        "fwhm_x_pix_cookie_oversamp": fwhm_x_pix_oversamp_cutout,
        "fwhm_y_pix_cookie_oversamp": fwhm_y_pix_oversamp_cutout,
        "amplitude_counts_cookie_oversamp": amplitude_counts_oversamp_cutout,
        "x_center_pix_fullarray_normsamp": x_center_pix_fullarray_normsamp,
        "y_center_pix_fullarray_normsamp": y_center_pix_fullarray_normsamp,
        "fwhm_x_pix_fullarray_normsamp": fwhm_x_pix_fullarray_normsamp,
        "fwhm_y_pix_fullarray_normsamp": fwhm_y_pix_fullarray_normsamp,
        "sigma_x_pix_oversamp_cutout": sigma_x_pix_oversamp_cutout,
        "sigma_y_pix_oversamp_cutout": sigma_y_pix_oversamp_cutout,
        "angle_theta_deg": angle_theta_deg,
    }


def fyi_plot_centroiding(array_to_plot, coords_to_plot, title_string=None, zscale=False, results_write_dir="figs_dump"):
    '''
    Make a FYI plot of the centroiding results.

    INPUTS
    ----------
    array_to_plot : np.ndarray
        2D array to plot.
    coords_to_plot : np.ndarray
        2D array of the centroided coordinates.
    title_string : str, optional
        Title of the plot. Default is None.
    zscale : bool, optional
        If True, use a zscale interval for the colorbar. Default is False.
    results_write_dir : str, optional
        Directory to write the output plot to. Default is "figs_dump".
        
    OUTPUTS
    -------
    None
        Saves the plot to the specified directory.
    '''
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(array_to_plot)
    plt.clf()
    plt.imshow(array_to_plot, origin='lower', vmin=vmin, vmax=vmax, cmap='gray')
    plt.scatter(coords_to_plot[:, 1], coords_to_plot[:, 0], color='red', s=10)
    if title_string is None:
        plot_title = "First-pass centroiding"
        plot_file_stub = "unnamed"
    else:
        plot_title = f"First-pass centroiding\n{title_string}"
        plot_file_stub = title_string
    plt.title(plot_title)
    plot_filename = f"fyi_plot_centroiding_{plot_file_stub}.png"
    os.makedirs(results_write_dir, exist_ok=True)
    file_path = os.path.join(results_write_dir, plot_filename)
    plt.savefig(file_path, bbox_inches='tight')
    logging.info(f"Saved {file_path}")
    plt.close()


def fit_gaussian_psf(cookie_cut_out_sci, obs_filter, fp_mask, pp_mask, coords_guess, plot_string, fac_oversamp, results_write_dir="figs_dump"):
    '''
    Fits a 2D Gaussian to the empirical PSF cutout and computes its FWHM and Strehl ratio.

    INPUTS
    ----------
    cookie_cut_out_sci : np.ndarray
        2D array of the science frame ("empirical" PSF cutout), oversampled.
    obs_filter : str
        Observing filter name (used for labeling/diagnostic purposes).
    fp_mask : str
        Focal plane mask identifier.
    pp_mask : str
        Pupil plane mask identifier.
    coords_guess : np.ndarray
        2D array containing the starting guess for the PSF centroid ([y, x]).
    plot_string : str
        String to append to plot file names for identification.
    fac_oversamp : float
        Oversampling factor for PSF/model relative to native pixel grid.
    results_write_dir : str, optional
        Directory to write the output plot to. Default is "figs_dump".

    OUTPUTS
    -------
    fwhm_y_pix_oversamp_cutout : float
        FWHM (oversampled pixels) along the y-axis from the best-fit Gaussian.
    fwhm_x_pix_oversamp_cutout : float
        FWHM (oversampled pixels) along the x-axis from the best-fit Gaussian.
    amplitude_counts_oversamp_cutout : float
        Amplitude of the best-fit Gaussian (in counts, oversampled).
    gaussian_based_strehl : float
        Estimated Strehl ratio: ratio of peaks, empirical / best-fit Gaussian, both oversampled.
    '''

    logging.info('--------------------------------')
    logging.info('Calculating coordinates and Strehl from Gaussian best-fit')

    ## ## TO DO: ARE THE INDEXES RIGHT HERE?
    cookie_cut_out_best_fit, x_center_pix_oversamp_cutout, y_center_pix_oversamp_cutout, fwhm_x_pix_oversamp_cutout, fwhm_y_pix_oversamp_cutout, sigma_x_pix_oversamp_cutout, sigma_y_pix_oversamp_cutout, angle_theta_deg, amplitude_counts_oversamp_cutout = fit_gaussian(cookie_cut_out_sci, \
        center_guess = coords_guess)
    residuals = cookie_cut_out_sci - cookie_cut_out_best_fit

    # strehl based on the Gaussian fit
    gaussian_based_strehl = np.max(cookie_cut_out_sci) / np.max(cookie_cut_out_best_fit)
    #print(f'Observing filter: {obs_filter}')
    #print(f'PSF ID: {plot_string}')
    #print(f'Focal plane mask: {fp_mask}')
    #print(f'Pupil plane mask: {pp_mask}')
    logging.info(f'Strehl from Gaussian best-fit: {gaussian_based_strehl:.2f}')


    plot_context = (
        f"Filter={obs_filter}, FP mask={fp_mask}, PP mask={pp_mask}, "
        f"Oversampling={fac_oversamp:.2f}, Gaussian Strehl={gaussian_based_strehl:.3f}"
    )

    # plot four subplots: 2D science, 2D best-fit, 2D residuals, and 1D overplotting of a cross-section of the science and best-fit
    plt.clf()
    # Determine vmin and vmax for consistent color scaling across all 2D plots
    vmin = min(np.nanmin(cookie_cut_out_sci), np.nanmin(cookie_cut_out_best_fit), np.nanmin(residuals))
    vmax = max(np.nanmax(cookie_cut_out_sci), np.nanmax(cookie_cut_out_best_fit), np.nanmax(residuals))
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # 2D Science image
    im0 = axs[0, 0].imshow(cookie_cut_out_sci, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('Empirical PSF cutout')
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)
    # 2D Best-fit image
    im1 = axs[0, 1].imshow(cookie_cut_out_best_fit, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('Best-fit 2D Gaussian model')
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)
    # 2D Residuals image
    im2 = axs[1, 0].imshow(residuals, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title('Residuals: empirical minus Gaussian')
    plt.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)
    # Plot a cross-section through the maximum of the PSF (along the row/col with the peak)
    max_index = np.unravel_index(np.argmax(cookie_cut_out_sci), cookie_cut_out_sci.shape)
    # Extract the row and column through the peak
    sci_row = cookie_cut_out_sci[max_index[0], :]
    best_fit_row = cookie_cut_out_best_fit[max_index[0], :]
    axs[1, 1].plot(sci_row, label='Empirical')
    axs[1, 1].plot(best_fit_row, label='Best-fit')
    # Annotate plot with FWHM in x and y
    fwhm_text = f'FWHM x = {fwhm_x_pix_oversamp_cutout:.2f} pix\nFWHM y = {fwhm_y_pix_oversamp_cutout:.2f} pix'
    axs[1, 1].text(
        0.95, 0.05, fwhm_text,
        transform=axs[1, 1].transAxes,
        fontsize=10, color='black',
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3')
    )
    axs[1, 1].legend()
    axs[1, 1].set_title('Peak-row cross-section: empirical vs Gaussian')
    plt.suptitle(
        f'Gaussian fit to oversampled PSF cutout\n'
        f'{plot_context}\n'
        f'Centroid (y, x)=({y_center_pix_oversamp_cutout:.2f}, {x_center_pix_oversamp_cutout:.2f}), '
        f'FWHM_x={fwhm_x_pix_oversamp_cutout:.2f} pix, '
        f'FWHM_y={fwhm_y_pix_oversamp_cutout:.2f} pix, '
        f'Peak amplitude={amplitude_counts_oversamp_cutout:.2f} counts'
    )
    plt.tight_layout()
    #plt.show()
    # Save the plot to file with num_coord as a 2-digit zero-padded string
    plot_filename = f'psf_gaussian_best_fit_'+plot_string+'.png'
    os.makedirs(results_write_dir, exist_ok=True)
    plt.savefig(os.path.join(results_write_dir, plot_filename), bbox_inches='tight')
    logging.info(f'Figure saved as {plot_filename}')
    plt.close()

    return x_center_pix_oversamp_cutout, y_center_pix_oversamp_cutout, fwhm_x_pix_oversamp_cutout, fwhm_y_pix_oversamp_cutout, amplitude_counts_oversamp_cutout, gaussian_based_strehl


def fit_simmed_psfs(cookie_cut_out_sci_oversamp, obs_filter, fp_mask, pp_mask, x_center_final_oversamp, y_center_final_oversamp, fac_oversamp, config_observing=None, results_write_dir="figs_dump"):
    '''
    Generate and analyze a perfect PSF using ScopeSim for direct comparison with empirical data.

    INPUTS
    ----------
    cookie_cut_out_sci_oversamp : numpy.ndarray
        2D array of the empirical (science) PSF at oversampled scale.
    obs_filter : str
        Observing filter to use (e.g., 'Lp').
    fp_mask : str
        Focal plane mask name to use in the simulation.
    pp_mask : str
        Pupil plane mask name to use in the simulation.
    x_center_final_oversamp : float
        X coordinate (in oversampled array) of center for cutout and analysis (no centroiding).
    y_center_final_oversamp : float
        Y coordinate (in oversampled array) of center for cutout and analysis (no centroiding).
    fac_oversamp : float or int
        Oversampling factor applied to the empirical data.
    config_observing : dict, optional
        Observing configuration dictionary (vestigial).
    results_write_dir : str, optional
        Directory to write the output plot to. Default is "figs_dump".

    OUTPUTS
    -------
    psf_perfect_cutout_best_fit : numpy.ndarray
        2D array of the cutout around the best-fit (simulated, "perfect") PSF generated by ScopeSim.
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
    logging.info('Generating a ScopeSim simmed PSF to compare to the empirical input')
    logging.info('Current Observing filter:', obs_filter)
    logging.info('Current WCU FP mask:', wcu.fpmask)
    logging.info('Current WCU PP mask:', pp_mask)

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
    # Get perfect, background-subtracted PSF - no detector noise
    psf_perfect = sci - background

    # Oversample the background-subtracted PSF to match the cookie_cut_out_sci_oversamp oversampling
    psf_perfect_oversamp = zoom(psf_perfect, fac_oversamp, order=3)

    # for debugging
    file_name_plot = "psf_perfect_oversamp.fits"
    fits.writeto(file_name_plot, psf_perfect_oversamp, overwrite=True)
    logging.info("Saved " + file_name_plot + " for checking.")


    # take a cutout of the PSF at the exact same coordinates as the cookie cut-out
    psf_perfect_cutout = psf_perfect_oversamp[int(y_center_final_oversamp-0.5*cookie_cut_out_sci_oversamp.shape[0]):int(y_center_final_oversamp+0.5*cookie_cut_out_sci_oversamp.shape[0]), \
        int(x_center_final_oversamp-0.5*cookie_cut_out_sci_oversamp.shape[1]):int(x_center_final_oversamp+0.5*cookie_cut_out_sci_oversamp.shape[1])]

    # cut out the central region the same size as the cookie cut-out
    #psf_perfect_cutout = psf_perfect[int(psf_perfect.shape[0]/2-0.5*cookie_cut_out_sci_oversamp.shape[0]):int(psf_perfect.shape[0]/2+0.5*cookie_cut_out_sci_oversamp.shape[0]), \
    #    int(psf_perfect.shape[1]/2-0.5*cookie_cut_out_sci_oversamp.shape[1]):int(psf_perfect.shape[1]/2+0.5*cookie_cut_out_sci_oversamp.shape[1])]

    # multiply psf_perfect_cutout by a coefficient to make it a best-fit to cookie_cut_out_sci_oversamp
    coefficient = np.sum(cookie_cut_out_sci_oversamp) / np.sum(psf_perfect_cutout)
    psf_perfect_cutout_best_fit = psf_perfect_cutout * coefficient

    strehl_from_simmed_psf = np.max(cookie_cut_out_sci_oversamp) / np.max(psf_perfect_cutout_best_fit)
    logging.info(f'Strehl from Scopesim simmed PSF: {strehl_from_simmed_psf:.2f}')

    plot_context = (
        f"Filter={obs_filter}, FP mask={fp_mask}, PP mask={pp_mask}, "
        f"Oversampling={fac_oversamp:.2f}, ScopeSim Strehl={strehl_from_simmed_psf:.3f}"
    )

    # Make subplots of cookie_cut_out_sci_oversamp, psf_perfect_cutout_best_fit, and the residuals
    plt.figure(figsize=(12, 4))
    
    # Panel 1: cookie_cut_out_sci_oversamp
    plt.subplot(1, 3, 1)
    zscale1 = ZScaleInterval()
    vmin1, vmax1 = zscale1.get_limits(cookie_cut_out_sci_oversamp)
    plt.imshow(cookie_cut_out_sci_oversamp, origin="lower", cmap="viridis", vmin=vmin1, vmax=vmax1)
    plt.title("Empirical oversampled PSF cutout")
    plt.colorbar(shrink=0.7, label="Counts")
    
    # Panel 2: psf_perfect_cutout_best_fit
    plt.subplot(1, 3, 2)
    zscale2 = ZScaleInterval()
    vmin2, vmax2 = zscale2.get_limits(psf_perfect_cutout_best_fit)
    plt.imshow(psf_perfect_cutout_best_fit, origin="lower", cmap="viridis", vmin=vmin2, vmax=vmax2)
    plt.title("Best-fit ScopeSim reference PSF")
    plt.colorbar(shrink=0.7, label="Counts")
    
    # Panel 3: Residuals
    residuals = cookie_cut_out_sci_oversamp - psf_perfect_cutout_best_fit
    plt.subplot(1, 3, 3)
    zscale3 = ZScaleInterval()
    vmin3, vmax3 = zscale3.get_limits(residuals)
    plt.imshow(residuals, origin="lower", cmap="RdBu", vmin=vmin3, vmax=vmax3)
    plt.title("Residuals: empirical minus ScopeSim model")
    plt.colorbar(shrink=0.7, label="Counts")
    
    plt.suptitle(
        f"ScopeSim reference-PSF comparison\n{plot_context}",
        fontsize=11,
    )
    plt.tight_layout()
    plot_filename = "junk_psf_perfect_cutout_best_fit.png"
    os.makedirs(results_write_dir, exist_ok=True)
    plt.savefig(os.path.join(results_write_dir, plot_filename), bbox_inches="tight")
    logging.info(f"Saved {plot_filename}")

    return psf_perfect_cutout_best_fit, strehl_from_simmed_psf