import logging
from dataclasses import dataclass
import ipdb
import numpy as np
import matplotlib.pyplot as plt

from .helpers import fit_gaussian_psf, fit_simmed_psfs, load_config_and_pipe
from .psf_grid_prep import load_grid_data_from_fits, prepare_psf_grid
from .strehl_fcns import strehl_from_annular_aperture_fixed, fit_annular_aperture_free_parameters


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
    grid_data_oversamp: np.ndarray,
    grid_data_original: np.ndarray,
    x_pos_pix_oversamp: np.ndarray,
    y_pos_pix_oversamp: np.ndarray,
    coords_centroided_1st_pass_all_oversamp: np.ndarray,
    raw_cutout_size_oversampled: int,
    oversample_factor: int,
    filter_name,
    fp_mask,
    pp_mask,
    config_observing,
    fit_method: str,
    fit_simmed_psf: bool,
    fit_annular_aperture_fixed: bool,
    fit_annular_aperture_free: bool,
) -> SinglePsfFitResult:
    logging.info(f"Processing PSF {num_coord} of {num_psfs_to_process}")

    cookie_edge_size = raw_cutout_size_oversampled
    idx_x_start = int(x_pos_pix_oversamp[num_coord] - 0.5 * cookie_edge_size)
    idx_x_end = int(x_pos_pix_oversamp[num_coord] + 0.5 * cookie_edge_size)
    idx_y_start = int(y_pos_pix_oversamp[num_coord] - 0.5 * cookie_edge_size)
    idx_y_end = int(y_pos_pix_oversamp[num_coord] + 0.5 * cookie_edge_size)
    cookie_cut_out_sci_oversamp = grid_data_oversamp[
        idx_y_start:idx_y_end, idx_x_start:idx_x_end
    ]

    plt.clf()
    plt.imshow(cookie_cut_out_sci_oversamp, origin="lower", cmap="gray_r")
    x_scatter = x_pos_pix_oversamp[num_coord] - idx_x_start
    y_scatter = y_pos_pix_oversamp[num_coord] - idx_y_start
    plt.scatter(x_scatter, y_scatter, color="red", s=10)
    plt.title(
        f"Cookie cut-out sci at coord (y,x): {y_pos_pix_oversamp[num_coord]}, {x_pos_pix_oversamp[num_coord]}"
    )
    plt.colorbar()
    plot_filename = f"junk_cookie_cut_out_sci_oversamp_{num_coord}.png"
    plt.savefig(f"figs_dump/{plot_filename}", bbox_inches="tight")
    plt.close()
    logging.info(f"Saved {plot_filename} to figs_dump/")

    coords_guess_this_cutout = np.array(
        [
            coords_centroided_1st_pass_all_oversamp[num_coord][0] - idx_y_start,
            coords_centroided_1st_pass_all_oversamp[num_coord][1] - idx_x_start,
        ]
    )

    logging.info(f"Fitting Gaussian to PSF {num_coord} of {num_psfs_to_process}")
    (
        x_center_pix_gaussian_best_fit_oversamp,
        y_center_pix_gaussian_best_fit_oversamp,
        fwhm_x_pix_gaussian_best_fit_oversamp,
        fwhm_y_pix_gaussian_best_fit_oversamp,
        amplitude_counts_gaussian_best_fit_oversamp,
        gaussian_based_strehl,
    ) = fit_gaussian_psf(
        cookie_cut_out_sci_oversamp,
        obs_filter=filter_name,
        fp_mask=fp_mask,
        pp_mask=pp_mask,
        coords_guess=coords_guess_this_cutout,
        plot_string=f"num_coord_{num_coord}_fpmask_{fp_mask}_ppmask_{pp_mask}_filter_{filter_name}",
        fac_oversamp=oversample_factor,
    )
    x_center_pix_gaussian_best_fit_oversamp_fullarray = (
        x_center_pix_gaussian_best_fit_oversamp + idx_x_start
    )
    y_center_pix_gaussian_best_fit_oversamp_fullarray = (
        y_center_pix_gaussian_best_fit_oversamp + idx_y_start
    )

    strehl_updates = {}

    if fit_simmed_psf:
        logging.info(f"Fitting ScopeSim PSF {num_coord} of {num_psfs_to_process}")
        fit_simmed_psfs(
            cookie_cut_out_sci_oversamp,
            data_empirical_original=grid_data_original,
            plot_string=f"num_coord_{num_coord}_fpmask_{fp_mask}_ppmask_{pp_mask}_filter_{filter_name}",
            obs_filter=filter_name,
            fp_mask=fp_mask,
            pp_mask=pp_mask,
            x_center_final_oversamp=x_pos_pix_oversamp[num_coord],
            y_center_final_oversamp=y_pos_pix_oversamp[num_coord],
            fac_oversamp=oversample_factor,
        )

    if fit_annular_aperture_fixed:
        logging.info(
            f"Calculating Strehl from annular aperture {num_coord} of {num_psfs_to_process}"
        )
        strehl_annular_aperture_fixed = strehl_from_annular_aperture_fixed(
            cookie_cut_out_sci_oversamp,
            data_empirical_original=grid_data_original,
            filter_name=filter_name,
            plot_string=f"num_coord_{num_coord}_fpmask_{fp_mask}_ppmask_{pp_mask}_filter_{filter_name}",
            x_center_final_cookie_oversamp=x_center_pix_gaussian_best_fit_oversamp,
            y_center_final_cookie_oversamp=y_center_pix_gaussian_best_fit_oversamp,
            config_observing=config_observing,
            fac_oversamp=oversample_factor,
            polychromatic=True,
        )
        strehl_updates.update(strehl_annular_aperture_fixed)

    if fit_annular_aperture_free:
        logging.info(f"Fitting analytical PSF {num_coord} of {num_psfs_to_process}")
        strehl_annular_aperture_free = fit_annular_aperture_free_parameters(
            cookie_cut_out_sci_oversamp,
            data_empirical_original=grid_data_original,
            filter_name=filter_name,
            plot_string=f"num_coord_{num_coord}_fpmask_{fp_mask}_ppmask_{pp_mask}_filter_{filter_name}",
            x_center_final_cookie_oversamp=x_center_pix_gaussian_best_fit_oversamp,
            y_center_final_cookie_oversamp=y_center_pix_gaussian_best_fit_oversamp,
            config_observing=config_observing,
            fac_oversamp=oversample_factor,
            fit_method=fit_method,
        )
        strehl_updates.update(strehl_annular_aperture_free)

    x_center_pix_gaussian_best_fit_normsamp = (
        x_center_pix_gaussian_best_fit_oversamp_fullarray / oversample_factor
    )
    y_center_pix_gaussian_best_fit_normsamp = (
        y_center_pix_gaussian_best_fit_oversamp_fullarray / oversample_factor
    )
    fwhm_x_pix_gaussian_best_fit_normsamp = (
        fwhm_x_pix_gaussian_best_fit_oversamp / oversample_factor
    )
    fwhm_y_pix_gaussian_best_fit_normsamp = (
        fwhm_y_pix_gaussian_best_fit_oversamp / oversample_factor
    )

    return SinglePsfFitResult(
        coord_x_normsamp=float(x_center_pix_gaussian_best_fit_normsamp),
        coord_y_normsamp=float(y_center_pix_gaussian_best_fit_normsamp),
        fwhm_x_normsamp=float(fwhm_x_pix_gaussian_best_fit_normsamp),
        fwhm_y_normsamp=float(fwhm_y_pix_gaussian_best_fit_normsamp),
        amplitude_counts=float(amplitude_counts_gaussian_best_fit_oversamp),
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

    oversample_factor = 3  # try to keep odd to facilitate centering
    logging.info(f"PSF oversampling factor: {oversample_factor}")

    config_coords_guesses_config = load_config_and_pipe(
        config_file_choice=config_coords_guesses_file_name, print_one_line=False
    )

    grid_data, grid_header = load_grid_data_from_fits(file_name, hdu_index=1)
    prep = prepare_psf_grid(
        grid_data,
        config_coords_guesses_config,
        psfs_subset=psfs_subset,
        oversample_factor=oversample_factor,
        grid_header=grid_header,
    )

    grid_data = prep.grid_data
    grid_data_original = prep.grid_data_original
    grid_data_oversamp = prep.grid_data_oversamp
    x_pos_pix_oversamp = prep.x_pos_pix_oversamp
    y_pos_pix_oversamp = prep.y_pos_pix_oversamp
    coords_centroided_1st_pass_all_oversamp = prep.coords_centroided_1st_pass_all_oversamp
    raw_cutout_size_oversampled = prep.raw_cutout_size_oversampled
    num_psfs_to_process = prep.num_psfs_to_process
    total_psfs = prep.total_psfs

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
        result = process_one_psf(
            num_coord,
            num_psfs_to_process,
            grid_data_oversamp=grid_data_oversamp,
            grid_data_original=grid_data_original,
            x_pos_pix_oversamp=x_pos_pix_oversamp,
            y_pos_pix_oversamp=y_pos_pix_oversamp,
            coords_centroided_1st_pass_all_oversamp=coords_centroided_1st_pass_all_oversamp,
            raw_cutout_size_oversampled=raw_cutout_size_oversampled,
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
        ipdb.set_trace()
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
