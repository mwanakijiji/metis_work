import logging
from dataclasses import dataclass
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from .helpers import fit_gaussian_psf, fit_simmed_psfs, load_config_and_pipe
from .psf_grid_prep import load_grid_data_from_fits, prepare_psf_grid
from .strehl_fcns import fit_annular_aperture_fixed_parameters, fit_annular_aperture_free_parameters
from scipy.ndimage import zoom
from photutils.centroids import centroid_2dg, centroid_sources

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
    """Per-PSF outputs from Gaussian fit (native pixel coords) and optional Strehl dicts."""

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
    #grid_data_oversamp: np.ndarray,
    #grid_data_original: np.ndarray,
    #x_pos_pix_oversamp_1st_pass: np.ndarray,
    #y_pos_pix_oversamp_1st_pass: np.ndarray,
    #coords_centroided_1st_pass_all_oversamp: np.ndarray,
    #x_pos_pix_native_1st_pass: np.ndarray,
    #y_pos_pix_native_1st_pass: np.ndarray,
    #coords_centroided_1st_pass_all_native: np.ndarray,
    #raw_cutout_size_oversampled: int,
    oversample_factor: int,
    filter_name: str,
    fp_mask: str,
    pp_mask: str,
    config_observing: dict,
    fit_method: str,
    fit_simmed_psf: bool,
    fit_annular_aperture_fixed: bool,
    fit_annular_aperture_free: bool,
) -> SinglePsfFitResult:
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
    )

    #idx_x_start_oversamp = int(x_pos_pix_oversamp_1st_pass[num_coord] - 0.5 * cookie_edge_size)
    #idx_x_end_oversamp = int(x_pos_pix_oversamp_1st_pass[num_coord] + 0.5 * cookie_edge_size)
    #idx_y_start_oversamp = int(y_pos_pix_oversamp_1st_pass[num_coord] - 0.5 * cookie_edge_size)
    #idx_y_end_oversamp = int(y_pos_pix_oversamp_1st_pass[num_coord] + 0.5 * cookie_edge_size)

    # Native bounds from oversampled slice so windows match (avoids int(center±half) on two scales).
    #fac_i = oversample_factor
    #fac_f = float(fac_i)
    #idx_x_start_cookie_original = idx_x_start_oversamp // fac_i
    #idx_y_start_cookie_original = idx_y_start_oversamp // fac_i
    #idx_x_end_cookie_original = (idx_x_end_oversamp + fac_i - 1) // fac_i
    #idx_y_end_cookie_original = (idx_y_end_oversamp + fac_i - 1) // fac_i

    # make the cutout from the full array (oversampled)
    '''
    cookie_cut_out_sci_oversamp = grid_data_oversamp[
        idx_y_start_oversamp:idx_y_end_oversamp, idx_x_start_oversamp:idx_x_end_oversamp
    ]
    grid_data_cookie_original = grid_data_original[
        idx_y_start_cookie_original:idx_y_end_cookie_original,
        idx_x_start_cookie_original:idx_x_end_cookie_original,
    ]
    cookie_cut_out_sci_original = grid_data_cookie_original
    '''

    # plot the oversampled cutout
    '''
    #x_scatter_oversamp = x_pos_pix_oversamp_1st_pass[num_coord] - idx_x_start_oversamp
    #y_scatter_oversamp = y_pos_pix_oversamp_1st_pass[num_coord] - idx_y_start_oversamp
    plot_filename = f"junk_cookie_cut_out_sci_oversamp_{num_coord}.png"
    _save_blinkable_cookie_fyi_plot(
        cookie_cut_out_sci_oversamp,
        x_scatter_oversamp,
        y_scatter_oversamp,
        f"Oversampled cookie cut-out sci (1st pass centroid) at coord (y,x): "
        f"{y_pos_pix_oversamp_1st_pass[num_coord]}, {x_pos_pix_oversamp_1st_pass[num_coord]}",
        f"figs_dump/{plot_filename}",
    )
    logging.info(f"Saved {plot_filename} to figs_dump/")

    # Same offset as oversampled plot, in native pixels: (x_ov - idx_start) / fac
    x_scatter_native = (
        x_pos_pix_oversamp_1st_pass[num_coord] - idx_x_start_oversamp
    ) / fac_f
    y_scatter_native = (
        y_pos_pix_oversamp_1st_pass[num_coord] - idx_y_start_oversamp
    ) / fac_f
    plot_filename = f"junk_cookie_cut_out_sci_original_{num_coord}.png"
    _save_blinkable_cookie_fyi_plot(
        cookie_cut_out_sci_original,
        x_scatter_native,
        y_scatter_native,
        f"Original cookie cut-out sci (1st pass centroid) at coord (y,x): "
        f"{y_pos_pix_native_1st_pass[num_coord]}, {x_pos_pix_native_1st_pass[num_coord]}",
        f"figs_dump/{plot_filename}",
    )
    logging.info(f"Saved {plot_filename} to figs_dump/")
    ipdb.set_trace()

    # convert 1st pass centroid to cutout pixel coordinates
    coords_this_cutout_oversamp_1st_pass = np.array(
        [
            coords_centroided_1st_pass_all_oversamp[num_coord][0] - idx_y_start_oversamp,
            coords_centroided_1st_pass_all_oversamp[num_coord][1] - idx_x_start_oversamp,
        ]
    )

    logging.info(f"Fitting Gaussian to PSF {num_coord} of {num_psfs_to_process}")
    (
        x_center_pix_gaussian_best_fit_cookie_oversamp,
        y_center_pix_gaussian_best_fit_cookie_oversamp,
        fwhm_x_pix_gaussian_best_fit_cookie_oversamp,
        fwhm_y_pix_gaussian_best_fit_cookie_oversamp,
        amplitude_counts_gaussian_best_fit_cookie_oversamp,
        gaussian_based_strehl,
    ) = fit_gaussian_psf(
        cookie_cut_out_sci_oversamp,
        obs_filter=filter_name,
        fp_mask=fp_mask,
        pp_mask=pp_mask,
        coords_guess=coords_this_cutout_oversamp_1st_pass,
        plot_string=f"num_coord_{num_coord}_fpmask_{fp_mask}_ppmask_{pp_mask}_filter_{filter_name}",
        fac_oversamp=oversample_factor,
    )
    

    # convert 2nd-pass centroid (from Gaussian centroid best-fit) to oversampled pixel coordinates in the FULL array
    x_center_2nd_pass_oversamp_fullarray = (
        x_center_pix_gaussian_best_fit_cookie_oversamp + idx_x_start_oversamp
    )
    y_center_2nd_pass_oversamp_fullarray = (
        y_center_pix_gaussian_best_fit_cookie_oversamp + idx_y_start_oversamp
    )

    # Native cutout position (Gaussian), aligned with cookie bounds above
    x_center_2nd_pass_original_samp_cookie_cutout = (
        (idx_x_start_oversamp + x_center_pix_gaussian_best_fit_cookie_oversamp) / fac_f
        - idx_x_start_cookie_original
    )
    y_center_2nd_pass_original_samp_cookie_cutout = (
        (idx_y_start_oversamp + y_center_pix_gaussian_best_fit_cookie_oversamp) / fac_f
        - idx_y_start_cookie_original
    )
    '''

    strehl_updates = {}

    '''
    if fit_simmed_psf:
        logging.info(f"Fitting ScopeSim PSF {num_coord} of {num_psfs_to_process}")
        fit_simmed_psfs(
            cookie_cut_out_sci_oversamp,
            data_empirical_original=grid_data_original,
            plot_string=f"num_coord_{num_coord}_fpmask_{fp_mask}_ppmask_{pp_mask}_filter_{filter_name}",
            obs_filter=filter_name,
            fp_mask=fp_mask,
            pp_mask=pp_mask,
            x_center_final_oversamp=x_center_2nd_pass_oversamp_fullarray,
            y_center_final_oversamp=y_center_2nd_pass_oversamp_fullarray,
            fac_oversamp=oversample_factor,
        )
    '''
    
    '''
    if fit_annular_aperture_fixed:
        logging.info(
            f"Calculating Strehl from annular aperture {num_coord} of {num_psfs_to_process}"
        )

        strehl_annular_aperture_fixed = fit_annular_aperture_fixed_parameters(
            cookie_cut_out_sci_oversamp,
            data_cookie_empirical_original=grid_data_cookie_original,
            filter_name=filter_name,
            plot_string=f"num_coord_{num_coord}_fpmask_{fp_mask}_ppmask_{pp_mask}_filter_{filter_name}",
            x_center_2nd_pass_cookie_oversamp=x_center_2nd_pass_oversamp_fullarray,
            y_center_2nd_pass_cookie_oversamp=y_center_2nd_pass_oversamp_fullarray,
            config_observing=config_observing,
            fac_oversamp=oversample_factor,
            polychromatic=True,
        )
        strehl_updates.update(strehl_annular_aperture_fixed)
    '''

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
        )
        strehl_updates.update(strehl_annular_aperture_free)

    x_center_pix_gaussian_best_fit_normsamp = x_center_2nd_pass_oversamp_fullarray / oversample_factor
    y_center_pix_gaussian_best_fit_normsamp = y_center_2nd_pass_oversamp_fullarray / oversample_factor
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
    fit_method="curve_fit",
):
    '''
    Find the Strehl ratio of a grid of PSFs

    INPUTS:
    file_name: name of the file containing the grid of PSFs
    fp_mask: focal plane mask (string)
    pp_mask: pupil plane mask (string)
    filter_name: name of the observing filter
    fit_simmed_psf: whether to fit a ScopeSim-simulated PSF
    fit_analytical_psf: whether to fit an analytical PSF
    psfs_subset: 'all' to process all PSFs, or an integer to process only the first N PSFs
    config_coords_guesses_file_name: name of the configuration file containing the coordinates of the PSFs
    config_observing: name of the configuration file containing the observing parameters


    OUTPUTS:
    None; writes out plots and data
    '''

    edge_size_original = 21 # pixels along one side of the cutout, original pixel sampling
    oversample_factor = 3  # try to keep odd to facilitate centering
    logging.info(f"PSF oversampling factor: {oversample_factor}")

    # retrieve coord guesses as a starting point
    config_coords_guesses_config = load_config_and_pipe(
        config_file_choice=config_coords_guesses_file_name, print_one_line=False
    )

    # retrieve data and do 1st-pass centroiding
    grid_data, grid_header = load_grid_data_from_fits(file_name, hdu_index=1)
    ## ## TO DO: DON'T OVERSAMPLE YET
    prep = prepare_psf_grid(
        grid_data,
        config_coords_guesses_config,
        psfs_subset=psfs_subset,
        oversample_factor=oversample_factor,
        grid_header=grid_header,
    )

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

    strehl_results_all = {}

    # loop over PSFs in the readout
    for num_coord in range(num_psfs_to_process):
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
            #grid_data_oversamp=grid_data_oversamp,
            #grid_data_original=grid_data_original,
            #x_pos_pix_oversamp_1st_pass=x_pos_pix_oversamp_1st_pass, # redundant
            #y_pos_pix_oversamp_1st_pass=y_pos_pix_oversamp_1st_pass, # redundant
            #coords_centroided_1st_pass_all_oversamp=coords_centroided_1st_pass_all_oversamp,
            #x_pos_pix_native_1st_pass=x_pos_pix_native_1st_pass,
            #y_pos_pix_native_1st_pass=y_pos_pix_native_1st_pass,
            #coords_centroided_1st_pass_all_native=coords_centroided_1st_pass_all_native,
            #raw_cutout_size_oversampled=raw_cutout_size_oversampled, # along one side
            oversample_factor=oversample_factor,
            filter_name=filter_name,
            fp_mask=fp_mask,
            pp_mask=pp_mask,
            config_observing=config_observing,
            fit_method=fit_method,
            fit_simmed_psf=fit_simmed_psf,
            fit_annular_aperture_fixed=fit_annular_aperture_fixed,
            fit_annular_aperture_free=fit_annular_aperture_free,
        )

        coord_x_array[num_coord] = result.coord_x_normsamp
        coord_y_array[num_coord] = result.coord_y_normsamp
        fwhm_x_pix_array[num_coord] = result.fwhm_x_normsamp
        fwhm_y_pix_array[num_coord] = result.fwhm_y_normsamp
        amplitude_counts_array[num_coord] = result.amplitude_counts
        gaussian_based_strehl_array[num_coord] = result.gaussian_based_strehl
        strehl_results_all.update(result.strehl_updates)

    logging.info("Strehl results:")
    for k, v in strehl_results_all.items():
        logging.info(f"\t{k}:\t{v:.3f}")

    # plot the grid_data and annotate it with the best-fit fwhm in x and y for each PSF
    plt.clf()
    plt.imshow(grid_data, origin="lower", cmap="gray_r")
    for num_coord in range(len(coord_x_array)):
        text_x = coord_x_array[num_coord] - 125
        text_y = coord_y_array[num_coord] + 10
        plt.text(
            text_x,
            text_y,
            f"x: {fwhm_x_pix_array[num_coord]:.2f}, \n y: {fwhm_y_pix_array[num_coord]:.2f}, \n theta: {angle_theta_array[num_coord]:.2f}, \n amp: {amplitude_counts_array[num_coord]:.2f}, \n strehl: {gaussian_based_strehl_array[num_coord]:.2f}",
            color="k",
            fontsize=7,
            rotation=20,
        )
    plt.title("FWHM in x and y (pix), amplitude (counts)")
    plot_file_name = "fyi_plot_fwhm_and_amp.png"
    plt.savefig(f"figs_dump/{plot_file_name}", bbox_inches="tight")
    logging.info(f"Saved {plot_file_name} to figs_dump/")
    plt.close()

    return  # strehl_results_all
