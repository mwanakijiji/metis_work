import logging
import os
from dataclasses import dataclass
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from .helpers import fit_gaussian_psf, fit_simmed_psfs, load_config_and_pipe
from .psf_grid_prep import load_grid_data_from_fits, prepare_psf_grid
from .strehl_fcns import fit_annular_aperture_fixed_parameters, fit_annular_aperture_free_parameters
from scipy.ndimage import zoom
from photutils.centroids import centroid_2dg, centroid_sources
import pickle

# Fixed canvas for oversampled vs native FYI PNGs (same pixel size for blink comparison).
_COOKIE_FYI_FIGSIZE_INCH = (7.0, 6.25)
_COOKIE_FYI_DPI = 120


def _save_blinkable_cookie_fyi_plot(
    image_2d: np.ndarray,
    scatter_x: float,
    scatter_y: float,
    title: str,
    out_path: str,
) -> None:
    '''
    Save a plot of a PSF with a scatter point and a title.

    INPUTS
    ----------
    image_2d : np.ndarray
        2D array of the image to plot.
    scatter_x : float
        X coordinate of the scatter point.
    scatter_y : float
        Y coordinate of the scatter point.
    title : str
        Title of the plot.
    out_path : str
        Path to save the plot to.

    OUTPUTS
    -------
    None
        Saves the plot to the specified path.
    '''
    
    fig, ax = plt.subplots(
        figsize=_COOKIE_FYI_FIGSIZE_INCH,
        dpi=_COOKIE_FYI_DPI,
        constrained_layout=True,
    )
    im = ax.imshow(image_2d, origin="lower", cmap="gray_r")
    ax.scatter(scatter_x, scatter_y, color="red", s=10)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.055, pad=0.02)
    # No bbox_inches="tight" — keeps identical width×height in pixels across plots.
    fig.savefig(out_path, dpi=_COOKIE_FYI_DPI)
    plt.close(fig)


@dataclass(frozen=True)
class SinglePsfFitResult:
    '''Per-PSF outputs from Gaussian fit (native pixel coords) and optional Strehl dicts.'''

    coord_x_normsamp: float
    coord_y_normsamp: float
    fwhm_x_normsamp: float
    fwhm_y_normsamp: float
    amplitude_counts: float
    gaussian_based_strehl: float
    strehl_updates: dict


def process_one_psf(
    num_coord: int,
    num_psfs_to_process: int,
    *,
    cookie_cutout_original_this_psf: np.ndarray,
    oversample_factor: int,
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
    cookie_cutout_original_this_psf : np.ndarray
        Native-sampling square cutout containing the PSF of interest.
    oversample_factor : int
        Factor used to oversample the PSF cutout before centroiding and model fitting.
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

    cookie_edge_size_original = cookie_cutout_original_this_psf.shape[0]

    # now oversample the cutout
    cookie_cutout_this_psf_oversamp = zoom(cookie_cutout_original_this_psf, oversample_factor, order=3)

    # consider the center of the frame to be the first guess for the 2nd-pass centroid 
    # (remember, the 1st-pass was used to cut out the PSF in the first place)
    x_cen_oversamp = cookie_cutout_this_psf_oversamp.shape[1] / 2
    y_cen_oversamp = cookie_cutout_this_psf_oversamp.shape[0] / 2

    # centroid the oversampled cutout with a Gaussian fit
    (
        x_center_pix_gaussian_best_fit_cookie_oversamp,
        y_center_pix_gaussian_best_fit_cookie_oversamp,
        fwhm_x_pix_gaussian_best_fit_cookie_oversamp,
        fwhm_y_pix_gaussian_best_fit_cookie_oversamp,
        amplitude_counts_gaussian_best_fit_cookie_oversamp,
        gaussian_based_strehl,
    ) = fit_gaussian_psf(
        cookie_cutout_this_psf_oversamp,
        obs_filter=filter_name,
        fp_mask=fp_mask,
        pp_mask=pp_mask,
        coords_guess=[x_cen_oversamp, y_cen_oversamp],
        plot_string=f"num_coord_{num_coord}_fpmask_{fp_mask}_ppmask_{pp_mask}_filter_{filter_name}",
        fac_oversamp=oversample_factor,
        results_write_dir=results_write_dir,
    )


    strehl_updates = {}

    # fit a ScopeSim PSF functionality currently disabled; can reinsert later if needed
    '''
    if fit_simmed_psf:
        logging.info(f"Fitting ScopeSim PSF {num_coord} of {num_psfs_to_process}")
        strehl_simmed_psf = fit_simmed_psfs(
            cookie_cut_out_sci_oversamp = cookie_cutout_this_psf_oversamp,
            obs_filter=filter_name,
            fp_mask=fp_mask,
            pp_mask=pp_mask,
            x_center_final_oversamp = x_center_pix_gaussian_best_fit_cookie_oversamp,
            y_center_final_oversamp = y_center_pix_gaussian_best_fit_cookie_oversamp,
            fac_oversamp=oversample_factor,
            config_observing=config_observing,
            results_write_dir=results_write_dir,
        )
        strehl_updates.update(strehl_simmed_psf)
    '''

    # fit an annular aperture model with fixed aperture
    if fit_annular_aperture_fixed:
        logging.info(
            f"Calculating Strehl from annular aperture {num_coord} of {num_psfs_to_process}"
        )

        strehl_annular_aperture_fixed = fit_annular_aperture_fixed_parameters(
            cookie_cut_out_sci_oversamp = cookie_cutout_this_psf_oversamp,
            data_cookie_empirical_original = cookie_cutout_original_this_psf,
            filter_name=filter_name,
            plot_string=f"num_coord_{num_coord}_fpmask_{fp_mask}_ppmask_{pp_mask}_filter_{filter_name}",
            x_center_2nd_pass_cookie_oversamp = x_center_pix_gaussian_best_fit_cookie_oversamp,
            y_center_2nd_pass_cookie_oversamp = y_center_pix_gaussian_best_fit_cookie_oversamp,
            config_observing=config_observing,
            fac_oversamp=oversample_factor,
            polychromatic=True,
            results_write_dir=results_write_dir,
        )
        strehl_updates.update(strehl_annular_aperture_fixed)

    # fit an annular aperture model with free aperture radii
    if fit_annular_aperture_free:
        logging.info(f"Fitting analytical PSF {num_coord} of {num_psfs_to_process}")
        strehl_annular_aperture_free = fit_annular_aperture_free_parameters(
            cookie_cut_out_sci_oversamp = cookie_cutout_this_psf_oversamp,
            cookie_cut_out_sci_original = cookie_cutout_original_this_psf,
            filter_name = filter_name,
            plot_string = f"num_coord_{num_coord}_fpmask_{fp_mask}_ppmask_{pp_mask}_filter_{filter_name}",
            x_center_final_cookie_oversamp = x_center_pix_gaussian_best_fit_cookie_oversamp,
            y_center_final_cookie_oversamp = y_center_pix_gaussian_best_fit_cookie_oversamp,
            config_observing = config_observing,
            fac_oversamp = oversample_factor,
            fit_method = fit_method,
            results_write_dir=results_write_dir,
        )
        strehl_updates.update(strehl_annular_aperture_free)

    x_center_pix_gaussian_best_fit_normsamp = x_center_pix_gaussian_best_fit_cookie_oversamp / oversample_factor
    y_center_pix_gaussian_best_fit_normsamp = y_center_pix_gaussian_best_fit_cookie_oversamp / oversample_factor
    fwhm_x_pix_gaussian_best_fit_normsamp = fwhm_x_pix_gaussian_best_fit_cookie_oversamp / oversample_factor
    fwhm_y_pix_gaussian_best_fit_normsamp = fwhm_y_pix_gaussian_best_fit_cookie_oversamp / oversample_factor

    return SinglePsfFitResult(
        coord_x_normsamp=float(x_center_pix_gaussian_best_fit_normsamp),
        coord_y_normsamp=float(y_center_pix_gaussian_best_fit_normsamp),
        fwhm_x_normsamp=float(fwhm_x_pix_gaussian_best_fit_normsamp),
        fwhm_y_normsamp=float(fwhm_y_pix_gaussian_best_fit_normsamp),
        amplitude_counts=float(amplitude_counts_gaussian_best_fit_cookie_oversamp),
        gaussian_based_strehl=float(gaussian_based_strehl),
        strehl_updates=strehl_updates,
    )


def strehl_psfs(
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
    Measure Strehl-related quantities for a grid of PSFs, save the per-PSF results,
    and generate a summary diagnostic plot over the detector frame.

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

    edge_size_original = 21 # pixels along one side of the cutout, original pixel sampling
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
    strehl_results_all = {} # to contain info from all the PSFs

    # loop over all PSFs that we want to process from this one detector readout
    for num_coord in range(num_psfs_to_process):

        strehl_results_this_psf = {} # to contain info from this PSF alone
        # make cutout of the PSF from the original array, using the closest int to the 1st pass centroids

        # 1-st pass coords of the PSF in the original array
        x_cen_1st_pass_native = coords_centroided_1st_pass_all_native[num_coord][1]
        y_cen_1st_pass_native = coords_centroided_1st_pass_all_native[num_coord][0]

        # cut out the PSF
        grid_data_original_cutout_this_psf = grid_data_original[
            int(x_cen_1st_pass_native - 0.5*edge_size_original):int(x_cen_1st_pass_native + 0.5*edge_size_original),
            int(y_cen_1st_pass_native - 0.5*edge_size_original):int(y_cen_1st_pass_native + 0.5*edge_size_original)
        ]
        
        # find strehls
        result = process_one_psf(
            num_coord,
            num_psfs_to_process,
            cookie_cutout_original_this_psf=grid_data_original_cutout_this_psf,
            oversample_factor=oversample_factor,
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

        # for each strehl value in result, put it in strehl_results_this_psf as a key-value pair
        for key, value in result.strehl_updates.items():
            strehl_results_this_psf[key] = value
        # also include the 1st-pass centroid coordinates
        strehl_results_this_psf['x_cen_1st_pass_native'] = x_cen_1st_pass_native
        strehl_results_this_psf['y_cen_1st_pass_native'] = y_cen_1st_pass_native

        # put the results from this PSF into the overall dictionary
        strehl_results_all[f'psf_num_{num_coord:02d}'] = strehl_results_this_psf
    
    # pass/fail: for each PSF, is the Strehl values greater than 0.8?
    # user criterion: strehl_free_ann_ap_mtf
    pass_fail_list = []
    criterion_key = 'strehl_free_ann_ap_mtf'
    for psf_results in strehl_results_all.values():
        pass_fail = True if psf_results[criterion_key] >= 0.8 else False
        pass_fail_list.append(pass_fail)
    logging.info("--------------------------------")
    pass_fail_all = all(pass_fail_list)
    logging.info(f"Pass/fail for each PSF: {pass_fail_list}")
    logging.info(f"PASS/FAIL FOR ALL PSFs: {all(pass_fail_list)}")
    logging.info("--------------------------------")

    
    # pickle the results
    basename_file_name_pickle = f"strehl_results_all_{fp_mask}_{pp_mask}_{filter_name}.pkl"
    abs_file_name_pickle = os.path.join(results_write_dir, basename_file_name_pickle)
    with open(abs_file_name_pickle, 'wb') as f:
        pickle.dump({'strehl_results_all': strehl_results_all, 'pass_fail_all': pass_fail_all}, f)
    logging.info(f"Saved strehl results to {abs_file_name_pickle}")

    # plot the grid_data and annotate it with the best-fit fwhm in x and y for each PSF
    plt.clf()
    plt.figure(figsize=(18, 12))
    plt.imshow(grid_data, origin="lower", cmap="gray_r")
    for psf_results in strehl_results_all.values():
        x_cen = psf_results['x_cen_1st_pass_native']
        y_cen = psf_results['y_cen_1st_pass_native']
        strehl_free_ann_ap_mtf = psf_results.get('strehl_free_ann_ap_mtf', np.nan)
        plt.scatter(x_cen, y_cen, color="red", s=10)
        plt.text(
            x_cen - 125,
            y_cen + 10,
            f"{strehl_free_ann_ap_mtf:.3f}",
            color="k",
            fontsize=7,
            rotation=20,
        )
    plt.title(
        "First-pass PSF centroids, Strehl from MTF of free-parameter annular aperture\n"
        f"Filter={filter_name}, FP mask={fp_mask}, PP mask={pp_mask} "    
        )
    os.makedirs(results_write_dir, exist_ok=True)
    plot_file_name = os.path.join(
        results_write_dir,
        f"fyi_plot_strehl_free_ann_ap_mtf_{fp_mask}_{pp_mask}_{filter_name}.png",
    )
    plt.savefig(plot_file_name, bbox_inches="tight")
    logging.info(f"Saved {plot_file_name}")
    plt.close()


    return  # strehl_results_all
