import logging
import os
from dataclasses import dataclass
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from .helpers import fit_gaussian, fit_simmed_psfs, load_config_and_pipe
from .psf_grid_prep import load_grid_data_from_fits, prepare_psf_grid
from .strehl_fcns import fit_annular_aperture_fixed_parameters, fit_annular_aperture_free_parameters
from scipy.ndimage import zoom
from photutils.centroids import centroid_2dg, centroid_sources
import pickle
from scipy.special import j0, j1



@dataclass(frozen=True)
class SinglePsfFitResult:
    '''Per-PSF outputs from Gaussian fit (native pixel coords) and optional Strehl dicts.'''

    #coord_x_fullarray_normsamp: float
    #coord_y_fullarray_normsamp: float
    #fwhm_x_fullarray_normsamp: float
    #fwhm_y_fullarray_normsamp: float
    #amplitude_counts: float
    centroid_results: dict


def process_one_psf_stray_light(
    num_coord: int,
    num_psfs_to_process: int,
    *,
    original_array: np.ndarray,
    oversample_factor: int,
    coords_xy_1st_pass_normsamp: list[float, float],
    filter_name: str,
    fp_mask: str,
    pp_mask: str,
    config_observing: dict,
    results_write_dir: str,
    fit_method: str,
    fit_simmed_psf: bool,
    fit_annular_aperture_fixed: bool,
    fit_annular_aperture_free: bool,
) -> SinglePsfFitResult:
    '''
    Process one PSF cutout by oversampling it, centroiding it with a Gaussian fit,
    and optionally evaluating additional Strehl estimators.

    INPUTS
    ----------
    num_coord : int
        Zero-based index of the PSF currently being processed.
    num_psfs_to_process : int
        Total number of PSFs being processed from this detector image.
    original_array : np.ndarray
        Native-sampling 2D array
    oversample_factor : int
        Factor used to oversample the PSF cutout before centroiding and model fitting.
    coords_1st_pass_oversamp : list[float, float]
        X and Y coordinates of the 1st-pass centroid in oversampled pixel coordinates (using the full 2D array)
    filter_name : str
        Name of the observing filter associated with the PSF.
    fp_mask : str
        Focal-plane mask label used for bookkeeping and plot naming.
    pp_mask : str
        Pupil-plane mask label used for bookkeeping and plot naming.
    config_observing : dict
        Observing configuration dictionary passed to downstream fitting routines.
    results_write_dir : str
        Directory where diagnostic plots and fit products are written.
    fit_method : str
        Name of the fitting backend to use for the free annular-aperture fit.
    fit_simmed_psf : bool
        Whether to evaluate the ScopeSim-based Strehl workflow if enabled.
    fit_annular_aperture_fixed : bool
        Whether to evaluate the fixed-geometry annular-aperture Strehl fit.
    fit_annular_aperture_free : bool
        Whether to evaluate the free-geometry annular-aperture Strehl fit.

    OUTPUTS
    -------
    SinglePsfFitResult
        Dataclass containing the Gaussian-fit centroid/FWHM/amplitude values in native
        pixel units, the Gaussian-based Strehl estimate, and any optional Strehl metrics.
    '''
    logging.info(f"Processing PSF {num_coord} of {num_psfs_to_process}")

    #cookie_edge_size_original = original_array.shape[0]

    # oversample the entire array
    oversampled_array = zoom(original_array, oversample_factor, order=3) # kidn of redundant, but needed for debugging plots
    coords_xy_1st_pass_oversamp_fullarray = oversample_factor * np.array([coords_xy_1st_pass_normsamp[0], coords_xy_1st_pass_normsamp[1]])

    # make a cutout around the oversampled PSF so rest of array is neglected
    edge_size_oversamp = 30
    idx_cutout_oversamp_x1 = int(coords_xy_1st_pass_oversamp_fullarray[0] - edge_size_oversamp/2)
    idx_cutout_oversamp_x2 = int(coords_xy_1st_pass_oversamp_fullarray[0] + edge_size_oversamp/2)
    idx_cutout_oversamp_y1 = int(coords_xy_1st_pass_oversamp_fullarray[1] - edge_size_oversamp/2)
    idx_cutout_oversamp_y2 = int(coords_xy_1st_pass_oversamp_fullarray[1] + edge_size_oversamp/2)
    cookie_cut_out_sci_oversamp = oversampled_array[idx_cutout_oversamp_y1:idx_cutout_oversamp_y2, idx_cutout_oversamp_x1:idx_cutout_oversamp_x2]
    # fit_gaussian currently expects center_guess ordered as [x, y]
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
        center_guess=coords_guess_xy_cutout_oversamp
    )

    # convert cutout fit outputs back to full-array oversampled coordinates
    x_center_pix_gaussian_best_fit_fullarray_oversamp = x_center_pix_oversamp_cutout + idx_cutout_oversamp_x1
    y_center_pix_gaussian_best_fit_fullarray_oversamp = y_center_pix_oversamp_cutout + idx_cutout_oversamp_y1
    fwhm_x_pix_gaussian_best_fit_cookie_oversamp = fwhm_x_pix_oversamp_cutout
    fwhm_y_pix_gaussian_best_fit_cookie_oversamp = fwhm_y_pix_oversamp_cutout
    amplitude_counts_gaussian_best_fit_cookie_oversamp = amplitude_counts_oversamp_cutout
    #gaussian_based_strehl = np.max(cookie_cut_out_sci_oversamp) / np.max(cookie_cut_out_best_fit)

    # scale back to native sampling
    x_center_pix_gaussian_best_fit_fullarray_normsamp = x_center_pix_gaussian_best_fit_fullarray_oversamp / oversample_factor
    y_center_pix_gaussian_best_fit_fullarray_normsamp = y_center_pix_gaussian_best_fit_fullarray_oversamp / oversample_factor
    fwhm_x_pix_gaussian_best_fit_fullarray_normsamp = fwhm_x_pix_gaussian_best_fit_cookie_oversamp / oversample_factor
    fwhm_y_pix_gaussian_best_fit_fullarray_normsamp = fwhm_y_pix_gaussian_best_fit_cookie_oversamp / oversample_factor

    # consider the center of the frame to be the first guess for the 2nd-pass centroid 
    # (remember, the 1st-pass was used to cut out the PSF in the first place)
    #x_cen_oversamp = cookie_cutout_this_psf_oversamp.shape[1] / 2
    #y_cen_oversamp = cookie_cutout_this_psf_oversamp.shape[0] / 2

    logging.info(f"Gaussian-fit FWHM (x, y) (native sampling): ({fwhm_x_pix_gaussian_best_fit_fullarray_normsamp:.2f}, {fwhm_y_pix_gaussian_best_fit_fullarray_normsamp:.2f})")
    logging.info(f"Gaussian-fit centroid (x, y) (native sampling): ({x_center_pix_gaussian_best_fit_fullarray_normsamp:.2f}, {y_center_pix_gaussian_best_fit_fullarray_normsamp:.2f})")


    # save an FYI plot of oversampled region around PSF
    plt.clf()
    edge_size_oversamp = 20
    idx_cutout_x1 = int(x_center_pix_gaussian_best_fit_fullarray_oversamp - edge_size_oversamp/2)
    idx_cutout_x2 = int(x_center_pix_gaussian_best_fit_fullarray_oversamp + edge_size_oversamp/2)
    idx_cutout_y1 = int(y_center_pix_gaussian_best_fit_fullarray_oversamp - edge_size_oversamp/2)
    idx_cutout_y2 = int(y_center_pix_gaussian_best_fit_fullarray_oversamp + edge_size_oversamp/2)
    plt.figure(figsize=(12, 12))
    plt.imshow(oversampled_array[idx_cutout_y1:idx_cutout_y2, idx_cutout_x1:idx_cutout_x2], origin="lower", cmap="gray_r")
    plt.scatter(
        coords_xy_1st_pass_oversamp_fullarray[0] - idx_cutout_x1,
        coords_xy_1st_pass_oversamp_fullarray[1] - idx_cutout_y1,
        color="red",
        s=50,
        marker='x',
        label='1st pass',
        alpha=1,
    )
    plt.scatter(
        x_center_pix_gaussian_best_fit_fullarray_oversamp - idx_cutout_x1,
        y_center_pix_gaussian_best_fit_fullarray_oversamp - idx_cutout_y1,
        color="green",
        s=50,
        marker='+',
        label='2nd pass',
        alpha=1,
    )
    plt.title(f"Oversampled region around PSF {num_coord} of {num_psfs_to_process}")
    plt.legend()
    file_name_plot = os.path.join(results_write_dir, f"fyi_plot_oversampled_region_around_psf_{num_coord}.png")
    plt.savefig(file_name_plot)
    logging.info(f"Saved {file_name_plot}")
    plt.close()

    centroid_results = {
        "coord_x_fullarray_normsamp": float(x_center_pix_gaussian_best_fit_fullarray_normsamp),
        "coord_y_fullarray_normsamp": float(y_center_pix_gaussian_best_fit_fullarray_normsamp),
        "fwhm_x_fullarray_normsamp": float(fwhm_x_pix_gaussian_best_fit_fullarray_normsamp),
        "fwhm_y_fullarray_normsamp": float(fwhm_y_pix_gaussian_best_fit_fullarray_normsamp),
        "amplitude_counts": float(amplitude_counts_gaussian_best_fit_cookie_oversamp),
    }

    '''
    return SinglePsfFitResult(
        coord_x_fullarray_normsamp=float(x_center_pix_gaussian_best_fit_fullarray_normsamp),
        coord_y_fullarray_normsamp=float(y_center_pix_gaussian_best_fit_fullarray_normsamp),
        fwhm_x_fullarray_normsamp=float(fwhm_x_pix_gaussian_best_fit_fullarray_normsamp),
        fwhm_y_fullarray_normsamp=float(fwhm_y_pix_gaussian_best_fit_fullarray_normsamp),
        amplitude_counts=float(amplitude_counts_gaussian_best_fit_cookie_oversamp),
    )
    '''
    return centroid_results


def expected_light_exterior(radius_arcsec, wavelength=3.3e-6, diameter=39.0, plate_scale=0.00547):
    """
    Calculate the fraction of the total PSF (Airy pattern) energy outside a given radius [arcsec]
    Parameters
    ----------
    radius_arcsec : float or array-like
        Radius or radii (in arcseconds) at which to compute the exterior energy fraction.
    wavelength : float, optional
        Wavelength in meters (default: 3.3e-6 m).
    diameter : float, optional
        Telescope pupil diameter, in meters (default: 39.0 m).
    plate_scale : float, optional
        Arcsec per pixel (default: 5.47 mas/pix for LM-band).
        
    Returns
    -------
    energy_outside : float or np.ndarray
        Fraction of the total energy lying outside of specified radius. Value between 0 and 1.
    """    

    # Convert radius in arcsec to radians
    radius_arcsec = np.atleast_1d(radius_arcsec)
    radius_rad = np.deg2rad(radius_arcsec / 3600.)

    lambda_over_d = (wavelength / diameter) * 206265

    # Airy pattern argument
    # alpha = (pi * D / lambda) * sin(theta) ≈ (pi * D / lambda) * theta, for small angles
    alpha = (np.pi * diameter / wavelength) * radius_rad

    # Fractional encircled energy within radius r:
    # E(<r) = 1 - J0^2(alpha) - J1^2(alpha)
    J0 = j0(alpha)
    J1 = j1(alpha)
    encircled = 1 - J0**2 - J1**2

    # The fraction of energy outside radius is 1 - encircled
    energy_outside = 1 - encircled

    # Example usage:
    # At 0.1 arcsec from the center (e.g.)
    # exterior_fraction = expected_light_exterior(0.1)
    # print(f"Fraction of PSF energy outside 0.1 arcsec: {exterior_fraction:.4f}")

    return energy_outside


def measure_light_exterior(array_input, 
                            center_xy_pix=[1024, 1024], 
                            wavelength=3.3e-6, 
                            diameter_pupil=39.0, 
                            plate_scale_mas=5.47,
                            results_write_dir="figs_dump"):
    '''
    Measure the fraction of light exterior to range of radii in an array

    INPUTS
    ----------
    array_input : np.ndarray
        Array to measure the light exterior to a given radius in.
    center_xy_pix : list[float, float], optional
        X and Y coordinates of the center of the array, in pixels (default: [1024, 1024]).
    wavelength : float, optional
        Wavelength in meters (default: 3.3e-6 m).
    diameter_pupil : float, optional
        Telescope pupil diameter, in meters (default: 39.0 m).
    plate_scale_mas : float, optional
        Mas per pixel (default: 5.47 mas/pix for LM-band).
    
    OUTPUTS
    -------
    ratio_exterior_measured_over_expected : float
       TBD
    '''
    # dark rings in units of lambda/D
    dark_ring_arcsec_array_units_ld = np.array([1.21967, 
                                                2.233131, 
                                                3.238315, 
                                                4.241063, 
                                                5.242764, 
                                                6.243922, 
                                                7.244760, 
                                                8.245395, 
                                                9.245893, 
                                                10.246293])

    # based on the wavelength of light, where should the first dark Airy rings be?
    dark_ring_arcsec_array = (wavelength / diameter_pupil) * 206265 * dark_ring_arcsec_array_units_ld
    dark_ring_pix_array = dark_ring_arcsec_array / (1e-3 * plate_scale_mas) # [pix]

    # find fractions of PSF light to be expected outside each of the dark Airy rings
    exterior_fraction = expected_light_exterior(dark_ring_arcsec_array, \
        wavelength=wavelength, \
            diameter=diameter_pupil, \
                plate_scale=1e-3 * plate_scale_mas)

    # FYI plot to see if function is working right
    '''
    plt.clf()
    lambda_over_d = (wavelength / diameter_pupil) * 206265
    steps_subarray = np.linspace(0, 10, 200)
    radius_arcsec_array = lambda_over_d * steps_subarray
    test_exterior_fraction = expected_light_exterior(radius_arcsec_array, \
        wavelength=wavelength, \
            diameter=diameter_pupil, \
                plate_scale=1e-3 * plate_scale_mas)
    plt.plot(steps_subarray, test_exterior_fraction)
    plt.xlabel('Radius [lamdbda/D]')
    plt.ylabel('Fraction of energy')
    plt.yscale('log')
    plt.title('Fraction of energy exterior to radius, circular Airy pattern')
    file_name_plot = os.path.join(results_write_dir, f"fyi_plot_exterior_fraction_vs_radius.png")
    plt.savefig(file_name_plot)
    logging.info(f"Saved {file_name_plot}")
    plt.close()
    '''

    # sum over all the pixels in the array
    sum_pixels_unmasked = np.nansum(array_input)

    # make a circular mask in the data frame, where all the pixels within the dark ring are nans
    # Define a circular mask centered at (x_center, y_center) with radius dark_ring_rad [pixels]
    # Build a coordinate grid centered on center_xy_pix so (0, 0) is the PSF center.
    y_indices, x_indices = np.ogrid[
        -center_xy_pix[1] : array_input.shape[0] - center_xy_pix[1],
        -center_xy_pix[0] : array_input.shape[1] - center_xy_pix[0],
    ]

    ratio_exterior_measured_array = []

    ipdb.set_trace()

    for num_ring in range(0, len(dark_ring_pix_array)):

        mask_circle = x_indices**2 + y_indices**2 <= dark_ring_pix_array[num_ring]**2
        data_copy = np.copy(array_input)

        # mask the central region of the PSF and add pixels
        data_copy[mask_circle] = np.nan
        sum_pixels_masked = np.nansum(data_copy)

        ratio_exterior_measured = sum_pixels_masked / sum_pixels_unmasked
        ratio_exterior_measured_array.append(ratio_exterior_measured) # append to array

        ratio_exterior_expected = exterior_fraction[num_ring]

        print(f'Fraction of irradiance measured exterior to radius: {ratio_exterior_measured:.4f}')
        print(f'Fraction of irradiance expected exterior to radius: {ratio_exterior_expected:.4f}')
        print(f'Ratio of exterior pixels measured to expected: {ratio_exterior_measured:.4f} / {ratio_exterior_expected:.4f}')
        
        # FYI plot
        '''
        plt.clf()
        plt.imshow(data_copy, origin='lower', cmap='gray')
        circle = plt.Circle((x_center, y_center), dark_ring_pix_array[num_ring], color='red', fill=False, linewidth=2)
        plt.gca().add_patch(circle)
        plt.colorbar()
        plt.show()
        plt.close()
        '''

    ipdb.set_trace()

    ## ## CONTINUE HERE: IF RADIUS INCREASES LINEARLY AND THERE IS AN IMPERFECT BACKGROUND SUBTRACTION THAT LEAVES 
    # A CONSTANT BACKGROUND OFFSET, THE RATE AT WHICH RESIDUALS INCREASE SHOULD BE **2 (?).
    # SO, CHECK THE SLOPE OF THE RESIDUALS INCREASE, AS WELL AS ITS LINEARITY
    #######  ... AND ANOTHER FUNCTION TO LOCATE DIFFUSE SHAPES

    ratio_exterior_measured_over_expected = np.divide(ratio_exterior_measured_array, exterior_fraction)
    print(f'Net ratio of exterior pixels measured to expected: {ratio_exterior_measured_over_expected:.4f}')

    return 



def stray_light(
    file_name,
    fp_mask,
    pp_mask,
    filter_name=None,
    fit_simmed_psf=False,
    fit_annular_aperture_free=False,
    fit_annular_aperture_fixed=False,
    psfs_subset="all",
    config_coords_guesses_file_name=None,
    config_observing=None,
    results_write_dir="figs_dump",
    fit_method="curve_fit",
):
    '''
    Measure strehl light within the array, using an image with a single PSF in the middle

    INPUTS
    ----------
    file_name : str
        Path to the FITS file containing the PSF grid to analyze.
    fp_mask : str
        Focal-plane mask label used for bookkeeping and output naming.
    pp_mask : str
        Pupil-plane mask label used for bookkeeping and output naming.
    filter_name : str, optional
        Name of the observing filter associated with the PSF grid.
    fit_simmed_psf : bool, optional
        Whether to evaluate the ScopeSim-based Strehl workflow if enabled downstream.
    fit_annular_aperture_free : bool, optional
        Whether to evaluate the free-geometry annular-aperture Strehl fit.
    fit_annular_aperture_fixed : bool, optional
        Whether to evaluate the fixed-geometry annular-aperture Strehl fit.
    psfs_subset : str or int, optional
        Either ``"all"`` to process the full grid or an integer giving how many PSFs
        from the start of the centroid list to process.
    config_coords_guesses_file_name : str, optional
        Path to the configuration file containing initial coordinate guesses.
    config_observing : dict, optional
        Observing configuration dictionary passed to downstream PSF-fitting routines.
    results_write_dir : str, optional
        Directory where pickled results and diagnostic figures are saved.
    fit_method : str, optional
        Name of the fitting backend to use for the free annular-aperture fit.

    OUTPUTS
    -------
    None
        Writes a pickle containing per-PSF Strehl results and a summary pass/fail flag,
        and saves a detector-frame diagnostic plot annotated with Strehl values.
    '''

    #edge_size_original = 21 # pixels along one side of the cutout, original pixel sampling
    oversample_factor = 3  # try to keep odd to facilitate centering
    logging.info(f"PSF oversampling factor: {oversample_factor}")

    # retrieve coord guesses as a starting point
    config_coords_guesses_config = load_config_and_pipe(
        config_file_choice=config_coords_guesses_file_name, print_one_line=False
    )

    # retrieve data, oversample, and do 1st-pass centroiding
    # (note oversampled empirical frame is only used for centroiding; the cost function for fitting later on just uses the frame as-is)
    grid_data, grid_header = load_grid_data_from_fits(file_name, hdu_index=1)
    prep = prepare_psf_grid(
        grid_data,
        config_coords_guesses_config,
        psfs_subset=psfs_subset,
        oversample_factor=oversample_factor,
        grid_header=grid_header,
    )

    # unpack quantities
    grid_data = prep.grid_data # original data (native pixel scale)
    grid_data_original = prep.grid_data_original # original data (native pixel scale)
    grid_data_oversamp = prep.grid_data_oversamp # oversampled data
    x_pos_pix_oversamp_1st_pass = prep.x_pos_pix_oversamp # x-positions of the centroids (oversampled)
    y_pos_pix_oversamp_1st_pass = prep.y_pos_pix_oversamp # y-positions of the centroids (oversampled)
    coords_centroided_1st_pass_all_oversamp = prep.coords_centroided_1st_pass_all_oversamp # coordinates of the centroids (oversampled)
    x_pos_pix_native_1st_pass = prep.x_pos_pix_native # x-positions of the centroids (native pixel scale)
    y_pos_pix_native_1st_pass = prep.y_pos_pix_native # y-positions of the centroids (native pixel scale)
    coords_centroided_1st_pass_all_native = prep.coords_centroided_1st_pass_all_native # coordinates of the centroids (native pixel scale)
    raw_cutout_size_oversampled = prep.raw_cutout_size_oversampled # size of the raw cutout in oversampled pixels (note no cutout has been made yet)
    num_psfs_to_process = prep.num_psfs_to_process # number of PSFs to process
    total_psfs = prep.total_psfs # total number of PSFs in the grid

    logging.info("Finding PSF centroids, first pass (via prepare_psf_grid)")
    logging.info(f"Raw PSF cutout size (oversampled): {raw_cutout_size_oversampled}")
    logging.info(f"Total PSFs: {total_psfs}")
    if psfs_subset == "all":
        logging.info(f"Processing all {total_psfs} PSFs")
    elif isinstance(psfs_subset, int):
        logging.info(f"Processing {num_psfs_to_process} out of {total_psfs} PSFs")
    logging.info(f"Processing {num_psfs_to_process} out of {total_psfs} PSFs")

    # initialize arrays/dicts to store results
    (
        coord_x_array,
        coord_y_array,
        fwhm_x_pix_array,
        fwhm_y_pix_array,
        sigma_x_pix_array,
        sigma_y_pix_array,
        angle_theta_array,
        amplitude_counts_array,
        gaussian_based_strehl_array,
    ) = (np.zeros(total_psfs) for _ in range(9))
    centroid_results_all = {} # to contain info from all the PSFs

    # oversample the grid data
    #grid_data_oversamp = zoom(grid_data, oversample_factor, order=3)

    # loop over all PSFs that we want to process from this one detector readout
    for num_coord in range(num_psfs_to_process):

        #centroid_results_this_psf = {} # to contain info from this PSF alone
        # make cutout of the PSF from the original array, using the closest int to the 1st pass centroids

        # 1-st pass coords of the PSF in the original array
        x_cen_1st_pass_native = coords_centroided_1st_pass_all_native[num_coord][1]
        y_cen_1st_pass_native = coords_centroided_1st_pass_all_native[num_coord][0]

        # convert to oversampled coordinates
        #x_cen_1st_pass_oversamp = x_cen_1st_pass_native * oversample_factor
        #y_cen_1st_pass_oversamp = y_cen_1st_pass_native * oversample_factor

        
        
        # find coordinates (and later other things?)
        centroid_results_this_psf = process_one_psf_stray_light(
            num_coord,
            num_psfs_to_process,
            original_array=grid_data,
            oversample_factor=oversample_factor,
            coords_xy_1st_pass_normsamp=[x_cen_1st_pass_native, y_cen_1st_pass_native],
            filter_name=filter_name,
            fp_mask=fp_mask,
            pp_mask=pp_mask,
            config_observing=config_observing,
            results_write_dir=results_write_dir,
            fit_method=fit_method,
            fit_simmed_psf=fit_simmed_psf,
            fit_annular_aperture_fixed=fit_annular_aperture_fixed,
            fit_annular_aperture_free=fit_annular_aperture_free,
        )

        '''
        # for each coord value in result, put it in centroid_results_this_psf as a key-value pair
        for key, value in result.centroid_results.items():
            centroid_results_this_psf[key] = value
        # also include the 1st-pass centroid coordinates
        centroid_results_this_psf['x_cen_1st_pass_native'] = x_cen_1st_pass_native
        centroid_results_this_psf['y_cen_1st_pass_native'] = y_cen_1st_pass_native
        '''
        # put the results from this PSF into the overall dictionary
        centroid_results_all[f'psf_num_{num_coord:02d}'] = centroid_results_this_psf

        # based on the center of the PSF, measure the stray light outside of it
        ## ## TODO: ENABLE POLYCHROMATIC PSFS IN THE BELOW
        ipdb.set_trace()
        measure_light_exterior(
            array_input=grid_data,
            center_xy_pix=[x_cen_1st_pass_native, y_cen_1st_pass_native],
            wavelength=config_observing['monochromatic_observing_filters_lm'][filter_name],
            diameter_pupil=config_observing['D_aperture']['full'],
            plate_scale_mas=config_observing['pixel_scales']['img_lm'],
            results_write_dir=results_write_dir,
        )
    
        ipdb.set_trace()

    # pass/fail placeholder: criteria for distortion requirement TBD
    pass_fail_list = []
    criterion_key = 'PLACEHOLDER'
    for psf_results in centroid_results_all.values():
        #pass_fail = True if psf_results[criterion_key] >= 0.8 else False
        #pass_fail_list.append(pass_fail)
        pass_fail_list.append(False)
    pass_fail_all = all(pass_fail_list)

    logging.info("--------------------------------")
    logging.info("--------------------------------")
    logging.info(
        "Reqs (Ref. Overleaf doc IMG_OPT_03_Test_Description_In_Field_Straylight_and_Ghosts):\n"
        "1) METIS-1189:\n"
        "   Max stray-light irradiance from an in-field source < 0.1% of the\n"
        "   peak irradiance in IMG focal planes.\n"
        "   (Stray light here includes scattering from METIS opto-mechanical surfaces.)\n"
        "2) METIS-1429:\n"
        "   Max stray-light irradiance in the CFO-FP2 plane, from an in-field\n"
        "   source positioned in the METIS input focal plane, < 0.06% of peak irradiance.\n"
        "3) METIS-9522:\n"
        "   After data reduction and calibration, flux in optical artefacts and ghosts\n"
        "   must be below the 3-sigma thermal background noise (1 hour observation),\n"
        "   at the respective ghost spatial scale:\n"
        "   - point-source-like ghosts: below point-source sensitivity limit\n"
        "   - extended ghosts: below surface-brightness limit for that extension\n"
        "   This must hold when the source brightness causing the artefact(s) matches\n"
        "   the saturation limit in the fastest full-frame operation."
    )
    logging.info("--------------------------------")
    logging.info("--------------------------------")
    logging.info(f"PIXEL SCALE CALCULATION MACHINERY TBD")
    logging.info("--------------------------------")
    logging.info("--------------------------------")
    logging.info(f"DISTORTION CALCULATION MACHINERY TBD")
    logging.info("--------------------------------")
    logging.info("--------------------------------")
    logging.info(f"CENTERING CALCULATION MACHINERY TBD")
    logging.info("--------------------------------")
    logging.info("--------------------------------")
    logging.info(f"PASS/FAIL TBD: {all(pass_fail_list)}")
    logging.info("--------------------------------")
    logging.info("--------------------------------")

    # pickle the results
    basename_file_name_pickle = f"centroid_results_all_{fp_mask}_{pp_mask}_{filter_name}.pkl"
    abs_file_name_pickle = os.path.join(results_write_dir, basename_file_name_pickle)
    with open(abs_file_name_pickle, 'wb') as f:
        pickle.dump({"centroid_results_all": centroid_results_all, "pass_fail_all": pass_fail_all}, f)
    logging.info(f"Saved centroid results to {abs_file_name_pickle}")


    return  # centroid_results_all
