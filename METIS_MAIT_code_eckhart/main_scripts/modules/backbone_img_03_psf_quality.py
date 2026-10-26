"""Analysis backbone for the IMG-OPT-03 PSF image quality (Strehl) test."""

import logging
import os
import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from pipeline_registry import CLUSTER_PSF, COLOR_PSF, pipeline_stage

from .helpers import fit_psf_gaussian_from_native_array, load_config_and_pipe
from .psf_grid_prep import (
    load_grid_data_from_fits,
    oversample_1st_pass_centroid,
)
from .strehl_fcns import (
    fit_annular_aperture_fixed_parameters,
    fit_annular_aperture_free_parameters,
    imaging_band_from_fp_mask,
)

# Fixed canvas for oversampled vs native FYI PNGs (same pixel size for blink
# comparison).
_COOKIE_FYI_FIGSIZE_INCH = (7.0, 6.25)
_COOKIE_FYI_DPI = 120

# METIS-1408/-1409/-2864: minimum Strehl ratio
STREHL_MIN = 0.8
# Strehl estimator used for pass/fail
STREHL_CRITERION_KEY = "strehl_free_ann_ap_mtf"


def _save_blinkable_cookie_fyi_plot(
    image_2d: np.ndarray,
    scatter_x: float,
    scatter_y: float,
    title: str,
    out_path: str,
) -> None:
    """
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
    """

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


def resolve_config_for_masks(
    config_observing: dict, fp_mask: str, pp_mask: str
) -> dict:
    """
    Return a shallow copy of the observing config with mask-dependent values resolved.

    INPUTS
    ----------
    config_observing : dict
        Observing configuration dictionary (left unmodified).
    fp_mask : str
        Focal-plane mask label (e.g. ``grid_lm``, ``pinhole_n``); sets the LM/N band.
    pp_mask : str
        Cold-stop (pupil-plane mask) label; selects the reference aperture geometry.

    OUTPUTS
    -------
    dict
        Copy of ``config_observing`` with ``D_aperture`` taken from
        ``D_aperture_by_pupil_mask[pp_mask]`` (if listed), plus the band-neutral keys
        ``band``, ``pixel_scale_mas``, ``monochromatic_observing_filters``,
        ``polychromatic_observing_filters_leaf_name`` and ``cutout_edge_native``.
    """
    config_resolved = dict(config_observing)

    # reference aperture geometry for this cold stop
    d_aperture_by_pupil_mask = config_observing.get("D_aperture_by_pupil_mask", {})
    if pp_mask in d_aperture_by_pupil_mask:
        config_resolved["D_aperture"] = d_aperture_by_pupil_mask[pp_mask]
        logging.info(
            "Reference aperture for pp_mask=%s: D_aperture=%s m, D_obscuration=%s m",
            pp_mask,
            config_resolved["D_aperture"]["full"],
            config_resolved["D_aperture"]["D_obscuration"],
        )
    else:
        logging.warning(
            "pp_mask=%s not in D_aperture_by_pupil_mask; using default D_aperture=%s "
            "m, D_obscuration=%s m",
            pp_mask,
            config_resolved["D_aperture"]["full"],
            config_resolved["D_aperture"]["D_obscuration"],
        )

    # LM vs N band settings
    band = imaging_band_from_fp_mask(fp_mask)
    config_resolved["band"] = band
    config_resolved["pixel_scale_mas"] = float(
        config_observing["pixel_scales"][f"img_{band}"]
    )
    config_resolved["monochromatic_observing_filters"] = config_observing[
        f"monochromatic_observing_filters_{band}"
    ]
    config_resolved["polychromatic_observing_filters_leaf_name"] = config_observing[
        f"polychromatic_observing_filters_{band}_leaf_name"
    ]
    config_resolved["cutout_edge_native"] = int(
        config_observing.get("cutout_edge_native", {}).get(band, 31)
    )
    logging.info(
        "Band for fp_mask=%s: %s, pixel scale=%s mas, cutout edge=%s px",
        fp_mask,
        band,
        config_resolved["pixel_scale_mas"],
        config_resolved["cutout_edge_native"],
    )

    return config_resolved


@dataclass(frozen=True)
class SinglePsfFitResult:
    """
    Per-PSF outputs from Gaussian fit (native pixel coords) and optional Strehl dicts.
    """

    coord_x_normsamp: float
    coord_y_normsamp: float
    fwhm_x_normsamp: float
    fwhm_y_normsamp: float
    amplitude_counts: float
    gaussian_based_strehl: float
    strehl_updates: dict


@dataclass
class StrehlRunResult:
    """
    Strehl results for all PSFs of one FITS file (from strehl_psfs), plus the
    requirement verdicts (from log_test_outcome).
    """

    file_name_abs: str
    filter_name: str
    fp_mask: str
    pp_mask: str
    band: str  # 'lm' or 'n'
    # per PSF ('psf_num_NN'): Strehl estimates and 1st-pass centroid coordinates
    strehl_results_all: dict
    total_psfs: int  # number of PSFs in the grid
    num_psfs_processed: int  # number of PSFs analysed (< total_psfs for a subset)

    # requirement ID -> 'PASS', 'FAIL', 'INCONCLUSIVE', 'NOT EVALUATED' or 'N/A'
    verdicts: dict = field(default_factory=dict)
    verdict_overall: str | None = None


@pipeline_stage(
    name="process_one_psf",
    depends_on=("strehl_psfs",),
    cluster=CLUSTER_PSF,
    cluster_color=COLOR_PSF,
    label="Fit one PSF",
)
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
    """
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
        pixel units, the Gaussian-based Strehl estimate, and any optional Strehl
        metrics.
    """
    del fit_simmed_psf  # ScopeSim PSF fit is disabled for now (see below)
    logging.info("Processing PSF %s of %s", num_coord, num_psfs_to_process)

    _, gaussian_fit_outputs = fit_psf_gaussian_from_native_array(
        original_array=cookie_cutout_original_this_psf,
        oversample_factor=oversample_factor,
        coords_xy_1st_pass_normsamp=None,
        edge_size_oversamp=None,
    )
    cookie_cutout_this_psf_oversamp = gaussian_fit_outputs[
        "cookie_cut_out_sci_oversamp"
    ]
    cookie_cutout_best_fit = gaussian_fit_outputs["cookie_cut_out_best_fit"]
    x_center_pix_gaussian_best_fit_cookie_oversamp = gaussian_fit_outputs[
        "x_center_pix_fullarray_oversamp"
    ]
    y_center_pix_gaussian_best_fit_cookie_oversamp = gaussian_fit_outputs[
        "y_center_pix_fullarray_oversamp"
    ]
    fwhm_x_pix_gaussian_best_fit_cookie_oversamp = gaussian_fit_outputs[
        "fwhm_x_pix_cookie_oversamp"
    ]
    fwhm_y_pix_gaussian_best_fit_cookie_oversamp = gaussian_fit_outputs[
        "fwhm_y_pix_cookie_oversamp"
    ]
    amplitude_counts_gaussian_best_fit_cookie_oversamp = gaussian_fit_outputs[
        "amplitude_counts_cookie_oversamp"
    ]
    gaussian_based_strehl = np.max(cookie_cutout_this_psf_oversamp) / np.max(
        cookie_cutout_best_fit
    )

    logging.info(
        "Gaussian-fit FWHM (x, y) (oversampled): (%.2f, %.2f)",
        fwhm_x_pix_gaussian_best_fit_cookie_oversamp,
        fwhm_y_pix_gaussian_best_fit_cookie_oversamp,
    )

    strehl_updates = {}

    # fit a ScopeSim PSF functionality currently disabled; can reinsert later if needed
    # if fit_simmed_psf:
    #     logging.info(f"Fitting ScopeSim PSF {num_coord} of {num_psfs_to_process}")
    #     strehl_simmed_psf = fit_simmed_psfs(
    #         cookie_cut_out_sci_oversamp = cookie_cutout_this_psf_oversamp,
    #         obs_filter=filter_name,
    #         fp_mask=fp_mask,
    #         pp_mask=pp_mask,
    #         x_center_final_oversamp = x_center_pix_gaussian_best_fit_cookie_oversamp,
    #         y_center_final_oversamp = y_center_pix_gaussian_best_fit_cookie_oversamp,
    #         fac_oversamp=oversample_factor,
    #         config_observing=config_observing,
    #         results_write_dir=results_write_dir,
    #     )
    #     strehl_updates.update(strehl_simmed_psf)
    #

    # shorthands for the Strehl fits below
    plot_string = (
        f"num_coord_{num_coord}_fpmask_{fp_mask}_"
        f"ppmask_{pp_mask}_filter_{filter_name}"
    )
    x_center_gauss_cookie_oversamp = x_center_pix_gaussian_best_fit_cookie_oversamp
    y_center_gauss_cookie_oversamp = y_center_pix_gaussian_best_fit_cookie_oversamp

    # fit an annular aperture model with fixed aperture
    if fit_annular_aperture_fixed:
        logging.info(
            "Calculating Strehl from annular aperture %s of %s",
            num_coord,
            num_psfs_to_process,
        )

        strehl_annular_aperture_fixed = fit_annular_aperture_fixed_parameters(
            cookie_cut_out_sci_oversamp=cookie_cutout_this_psf_oversamp,
            data_cookie_empirical_original=cookie_cutout_original_this_psf,
            filter_name=filter_name,
            plot_string=plot_string,
            x_center_2nd_pass_cookie_oversamp=x_center_gauss_cookie_oversamp,
            y_center_2nd_pass_cookie_oversamp=y_center_gauss_cookie_oversamp,
            config_observing=config_observing,
            fac_oversamp=oversample_factor,
            polychromatic=True,
            results_write_dir=results_write_dir,
        )
        strehl_updates.update(strehl_annular_aperture_fixed)

    # fit an annular aperture model with free aperture radii
    if fit_annular_aperture_free:
        logging.info("Fitting analytical PSF %s of %s", num_coord, num_psfs_to_process)
        strehl_annular_aperture_free = fit_annular_aperture_free_parameters(
            cookie_cut_out_sci_oversamp=cookie_cutout_this_psf_oversamp,
            cookie_cut_out_sci_original=cookie_cutout_original_this_psf,
            filter_name=filter_name,
            plot_string=plot_string,
            x_center_final_cookie_oversamp=x_center_gauss_cookie_oversamp,
            y_center_final_cookie_oversamp=y_center_gauss_cookie_oversamp,
            config_observing=config_observing,
            fac_oversamp=oversample_factor,
            fit_method=fit_method,
            fp_mask=fp_mask,
            results_write_dir=results_write_dir,
            pp_mask=pp_mask,
        )
        strehl_updates.update(strehl_annular_aperture_free)

    x_center_pix_gaussian_best_fit_normsamp = gaussian_fit_outputs[
        "x_center_pix_fullarray_normsamp"
    ]
    y_center_pix_gaussian_best_fit_normsamp = gaussian_fit_outputs[
        "y_center_pix_fullarray_normsamp"
    ]
    fwhm_x_pix_gaussian_best_fit_normsamp = gaussian_fit_outputs[
        "fwhm_x_pix_fullarray_normsamp"
    ]
    fwhm_y_pix_gaussian_best_fit_normsamp = gaussian_fit_outputs[
        "fwhm_y_pix_fullarray_normsamp"
    ]

    return SinglePsfFitResult(
        coord_x_normsamp=float(x_center_pix_gaussian_best_fit_normsamp),
        coord_y_normsamp=float(y_center_pix_gaussian_best_fit_normsamp),
        fwhm_x_normsamp=float(fwhm_x_pix_gaussian_best_fit_normsamp),
        fwhm_y_normsamp=float(fwhm_y_pix_gaussian_best_fit_normsamp),
        amplitude_counts=float(amplitude_counts_gaussian_best_fit_cookie_oversamp),
        gaussian_based_strehl=float(gaussian_based_strehl),
        strehl_updates=strehl_updates,
    )


@pipeline_stage(
    name="strehl_psfs",
    depends_on=("generate_psf_quality_sim",),
    cluster=CLUSTER_PSF,
    cluster_color=COLOR_PSF,
    label="Measure Strehl",
)
def strehl_psfs(
    file_name_abs,
    fp_mask,
    pp_mask,
    filter_name=None,
    fit_simmed_psf=False,
    fit_annular_aperture_free=False,
    fit_annular_aperture_fixed=False,
    psfs_subset="all",
    config_coords_guesses_file_name_abs=None,
    config_observing=None,
    results_write_dir="figs_dump",
    fit_method="curve_fit",
):
    """
    Measure Strehl-related quantities for a grid of PSFs, and generate summary
    diagnostic plots over the detector frame. Pass/fail is decided separately, by
    log_test_outcome.

    INPUTS
    ----------
    file_name_abs : str
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
    StrehlRunResult
        Per-PSF Strehl results and PSF counts for this file (verdicts still empty).
        Also saves detector-frame diagnostic plots annotated with Strehl values.
    """

    # resolve cold-stop geometry and LM/N band settings for this mask combination
    config_observing = resolve_config_for_masks(config_observing, fp_mask, pp_mask)

    edge_size_original = config_observing[
        "cutout_edge_native"
    ]  # pixels along one side of the cutout, original pixel sampling
    oversample_factor = 3  # try to keep odd to facilitate centering
    logging.info("PSF oversampling factor: %s", oversample_factor)

    # retrieve coord guesses as a starting point
    config_coords_guesses_config = load_config_and_pipe(
        config_file_choice=config_coords_guesses_file_name_abs,
        print_one_line=False,
    )

    # retrieve data, oversample, and do 1st-pass centroiding
    # (note oversampled empirical frame is only used for centroiding; the cost function
    # for fitting later on just uses the frame as-is)
    grid_data, grid_header = load_grid_data_from_fits(file_name_abs, hdu_index=1)

    prep = oversample_1st_pass_centroid(
        grid_data,
        config_coords_guesses_config,
        psfs_subset=psfs_subset,
        oversample_factor=oversample_factor,
        grid_header=grid_header,
    )

    # unpack quantities
    grid_data = prep.grid_data  # original data (native pixel scale)
    grid_data_original = prep.grid_data_original  # original data (native pixel scale)
    coords_centroided_1st_pass_all_native = (
        prep.coords_centroided_1st_pass_all_native
    )  # coordinates of the centroids (native pixel scale)
    raw_cutout_size_oversampled = (
        prep.raw_cutout_size_oversampled
    )  # size of the raw cutout in oversampled pixels (note no cutout has been made yet)
    num_psfs_to_process = prep.num_psfs_to_process  # number of PSFs to process
    total_psfs = prep.total_psfs  # total number of PSFs in the grid

    logging.info("Finding PSF centroids, first pass (via oversample_1st_pass_centroid)")
    logging.info("Raw PSF cutout size (oversampled): %s", raw_cutout_size_oversampled)
    logging.info("Total PSFs: %s", total_psfs)
    if psfs_subset == "all":
        logging.info("Processing all %s PSFs", total_psfs)
    elif isinstance(psfs_subset, int):
        logging.info("Processing %s out of %s PSFs", num_psfs_to_process, total_psfs)
    logging.info("Processing %s out of %s PSFs", num_psfs_to_process, total_psfs)

    # initialize dict to store results
    strehl_results_all = {}  # to contain info from all the PSFs

    # loop over all PSFs that we want to process from this one detector readout
    for num_coord in range(num_psfs_to_process):
        # ipdb.set_trace(context=10)

        strehl_results_this_psf = {}  # to contain info from this PSF alone
        # make cutout of the PSF from the original array, using the closest int to the
        # 1st pass centroids

        # 1-st pass coords of the PSF in the original array
        x_cen_1st_pass_native = coords_centroided_1st_pass_all_native[num_coord][1]
        y_cen_1st_pass_native = coords_centroided_1st_pass_all_native[num_coord][0]

        # cut out the PSF
        grid_data_original_cutout_this_psf = grid_data_original[
            int(x_cen_1st_pass_native - 0.5 * edge_size_original) : int(
                x_cen_1st_pass_native + 0.5 * edge_size_original
            ),
            int(y_cen_1st_pass_native - 0.5 * edge_size_original) : int(
                y_cen_1st_pass_native + 0.5 * edge_size_original
            ),
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

        # for each strehl value in result, put it in strehl_results_this_psf as a
        # key-value pair
        for key, value in result.strehl_updates.items():
            strehl_results_this_psf[key] = value
        # also include the 1st-pass centroid coordinates
        strehl_results_this_psf["x_cen_1st_pass_native"] = x_cen_1st_pass_native
        strehl_results_this_psf["y_cen_1st_pass_native"] = y_cen_1st_pass_native

        # put the results from this PSF into the overall dictionary
        strehl_results_all[f"psf_num_{num_coord:02d}"] = strehl_results_this_psf

    # pickle the results (skipped for now)
    # basename_file_name_pickle = (
    #     f"strehl_results_all_{fp_mask}_{pp_mask}_{filter_name}.pkl"
    # )
    # abs_file_name_pickle = os.path.join(results_write_dir, basename_file_name_pickle)
    # with open(abs_file_name_pickle, "wb") as f:
    #     pickle.dump(
    #         {
    #             "strehl_results_all": strehl_results_all,
    #         },
    #         f,
    #     )
    # logging.info(f"Saved strehl results to {abs_file_name_pickle}")
    #

    # plot the grid_data and annotate it with the best-fit fwhm in x and y for each PSF
    plt.clf()
    plt.figure(figsize=(18, 12))
    plt.imshow(grid_data, origin="lower", cmap="gray_r")

    # plot strehls of all the PSFs
    method_choice = "strehl_fix_ann_ap_eso"
    for psf_results in strehl_results_all.values():

        x_cen = psf_results["x_cen_1st_pass_native"]
        y_cen = psf_results["y_cen_1st_pass_native"]
        strehl_here = psf_results.get(method_choice, np.nan)
        plt.scatter(x_cen, y_cen, color="red", s=10)
        plt.text(
            x_cen - 125,
            y_cen + 10,
            f"{strehl_here:.3f}",
            color="k",
            fontsize=7,
            rotation=20,
        )
    plt.title(
        f"First-pass PSF centroids, Strehl from {method_choice}\n"
        f"Filter={filter_name}, FP mask={fp_mask}, PP mask={pp_mask} "
    )
    os.makedirs(results_write_dir, exist_ok=True)
    plot_file_name_all_psfs = os.path.join(
        results_write_dir,
        f"fyi_plot_strehl_free_ann_ap_mtf_{fp_mask}_{pp_mask}_{filter_name}.png",
    )
    plt.savefig(plot_file_name_all_psfs, bbox_inches="tight")
    logging.info(
        "Saved plot of all PSFs, Strehl from %s: %s",
        method_choice,
        plot_file_name_all_psfs,
    )
    plt.close()

    # pick one random PSF and plot strehls from all the methods
    psf_result_random = random.choice(list(strehl_results_all.values()))
    x_cen_random = psf_result_random["x_cen_1st_pass_native"]
    y_cen_random = psf_result_random["y_cen_1st_pass_native"]
    strehl_fix_ann_ap_eso_random = psf_result_random.get(
        "strehl_fix_ann_ap_eso", np.nan
    )
    strehl_free_ann_ap_mtf_random = psf_result_random.get(
        "strehl_free_ann_ap_mtf", np.nan
    )
    strehl_gaussian_fit_random = psf_result_random.get("strehl_gaussian_fit", np.nan)
    strehl_simmed_psf_random = psf_result_random.get("strehl_simmed_psf", np.nan)
    plt.clf()
    plt.figure(figsize=(18, 12))
    plt.imshow(grid_data, origin="lower", cmap="gray_r")
    plt.scatter(x_cen_random, y_cen_random, color="red", s=10)
    strehl_methods = [
        ("strehl_fix_ann_ap_eso", strehl_fix_ann_ap_eso_random),
        ("strehl_free_ann_ap_mtf", strehl_free_ann_ap_mtf_random),
        ("strehl_gaussian_fit", strehl_gaussian_fit_random),
        ("strehl_simmed_psf", strehl_simmed_psf_random),
        # Uncomment/add more strehl methods here if needed, e.g.:
        # ("strehl_other_method", strehl_other_method_random),
    ]
    # Annotate with method name and value, spaced vertically
    dy = 18  # vertical spacing between annotations
    for i, (method, value) in enumerate(strehl_methods):
        plt.text(
            x_cen_random - 125,
            y_cen_random + 10 + i * dy,
            f"{method}: {value:.3f}",
            color="k",
            fontsize=8,
            rotation=20,
        )
    plt.title(
        f"Random PSF, Strehl from mult methods\n"
        f"Filter={filter_name}, FP mask={fp_mask}, PP mask={pp_mask} "
    )
    plot_file_name_random_psf_all_methods = os.path.join(
        results_write_dir,
        f"fyi_plot_strehl_mult_methods_{fp_mask}_{pp_mask}_{filter_name}.png",
    )
    plt.savefig(plot_file_name_random_psf_all_methods, bbox_inches="tight")
    logging.info(
        "Saved plot of all Strehl methods, random PSF: %s",
        plot_file_name_random_psf_all_methods,
    )
    plt.close()

    return StrehlRunResult(
        file_name_abs=file_name_abs,
        filter_name=filter_name,
        fp_mask=fp_mask,
        pp_mask=pp_mask,
        band=config_observing["band"],
        strehl_results_all=strehl_results_all,
        total_psfs=int(total_psfs),
        num_psfs_processed=int(num_psfs_to_process),
    )


@pipeline_stage(
    name="img_03_test_outcome",
    depends_on=("strehl_psfs",),
    cluster=CLUSTER_PSF,
    cluster_color=COLOR_PSF,
    label="IMG-OPT-03 pass/fail",
)
def log_test_outcome(
    run_result, criterion_key=STREHL_CRITERION_KEY, strehl_min=STREHL_MIN
):
    """
    Decide and log the pass/fail of each IMG-OPT-03 requirement, and of the whole test.

    - METIS-1408 (Strehl >= 80% at lambda > 3 um over the entire FOV) and METIS-1409
      (Strehl > 0.8 at 3 um, IMG-LM and -NQ, over the full FOV): every PSF has
      Strehl (criterion_key) >= strehl_min. FAIL if any PSF is below; INCONCLUSIVE if
      all analysed PSFs pass but only a subset of the grid was analysed (so not the
      entire FOV).
    - METIS-2864 (Strehl > 80% at 3.3 um over the entire FOV, WCU+CFO+IMG-LM path):
      as above, for LM data only; N/A for N-band data.
    - METIS-3503 (PSF shape characterised across the entire FoV using the WCU): PASS if
      every PSF in the grid was analysed and has a finite Strehl estimate; FAIL if any
      analysed PSF has a non-finite one; INCONCLUSIVE if only a subset of the grid was
      analysed.
    - If criterion_key is missing for some PSF (e.g. that fit was switched off), all
      four requirements are NOT EVALUATED.

    Overall: PASS if all applicable requirements pass; FAIL if any fails; else
    INCONCLUSIVE.

    INPUTS:
    - run_result (StrehlRunResult): from strehl_psfs
    - criterion_key (str): Strehl estimator used for pass/fail
    - strehl_min (float): minimum Strehl ratio

    OUTPUTS:
    - run_result (StrehlRunResult), with verdicts and verdict_overall set
    """
    strehls = {
        psf_name: psf_results.get(criterion_key)
        for psf_name, psf_results in run_result.strehl_results_all.items()
    }
    full_grid = run_result.num_psfs_processed >= run_result.total_psfs
    missing = [name for name, strehl in strehls.items() if strehl is None]
    finite = {
        name: float(strehl)
        for name, strehl in strehls.items()
        if strehl is not None and np.isfinite(strehl)
    }

    logging.info("--------------------------------------------------")
    logging.info("--------------------------------------------------")
    logging.info("File: %s", run_result.file_name_abs)
    logging.info(
        "Filter %s, FP mask %s, PP mask %s (band %s); %s of %s PSFs analysed",
        run_result.filter_name,
        run_result.fp_mask,
        run_result.pp_mask,
        run_result.band,
        run_result.num_psfs_processed,
        run_result.total_psfs,
    )

    if missing:
        # the pass/fail estimator was not computed (e.g. that fit was switched off)
        logging.error(
            "No %s for %s PSFs (%s); requirements not evaluated",
            criterion_key,
            len(missing),
            ", ".join(missing),
        )
        verdict_strehl = "NOT EVALUATED"
        verdict_3503 = "NOT EVALUATED"
    else:
        # Strehl criterion (METIS-1408, -1409, -2864)
        below = []
        for psf_name, strehl in strehls.items():
            ok = np.isfinite(strehl) and strehl >= strehl_min
            if not ok:
                below.append(psf_name)
            logging.info(
                "  %s: %s = %.3f %s %.2f",
                psf_name,
                criterion_key,
                strehl,
                ">=" if ok else "<",
                strehl_min,
            )
        if below:
            verdict_strehl = "FAIL"
            logging.info(
                "%s of %s PSFs below Strehl %.2f: %s",
                len(below),
                len(strehls),
                strehl_min,
                ", ".join(below),
            )
        elif full_grid:
            verdict_strehl = "PASS"
        else:
            verdict_strehl = "INCONCLUSIVE"
            logging.info(
                "All analysed PSFs have Strehl >= %.2f, but only %s of %s PSFs were "
                "analysed (not the entire FOV)",
                strehl_min,
                run_result.num_psfs_processed,
                run_result.total_psfs,
            )

        # PSF characterisation across the FoV (METIS-3503)
        if len(finite) < len(strehls):
            verdict_3503 = "FAIL"
        elif full_grid:
            verdict_3503 = "PASS"
        else:
            verdict_3503 = "INCONCLUSIVE"

    run_result.verdicts = {
        "METIS-1408": verdict_strehl,
        "METIS-1409": verdict_strehl,
        "METIS-2864": verdict_strehl if run_result.band == "lm" else "N/A",
        "METIS-3503": verdict_3503,
    }

    logging.info("--------------------------------------------------")
    logging.info(
        "METIS-1408 (Strehl >= %.0f%% at lambda > 3 um, entire FOV): %s",
        100 * strehl_min,
        run_result.verdicts["METIS-1408"],
    )
    logging.info(
        "METIS-1409 (Strehl > %.1f at 3 um, IMG-LM and -NQ, full FOV): %s",
        strehl_min,
        run_result.verdicts["METIS-1409"],
    )
    logging.info(
        "METIS-2864 (Strehl > %.0f%% at 3.3 um, WCU+CFO+IMG-LM path, entire FOV): %s",
        100 * strehl_min,
        run_result.verdicts["METIS-2864"],
    )
    logging.info(
        "METIS-3503 (PSF shape characterised across the entire FoV with the WCU): %s "
        "(%s of %s PSFs with a finite %s)",
        run_result.verdicts["METIS-3503"],
        len(finite),
        run_result.total_psfs,
        criterion_key,
    )
    logging.info("--------------------------------------------------")

    applicable = [v for v in run_result.verdicts.values() if v != "N/A"]
    if all(v == "PASS" for v in applicable):
        run_result.verdict_overall = "PASS"
    elif "FAIL" in applicable:
        run_result.verdict_overall = "FAIL"
    else:
        run_result.verdict_overall = "INCONCLUSIVE"
    logging.info("IMG 03 PSF QUALITY TEST: %s", run_result.verdict_overall)
    logging.info("--------------------------------------------------")

    return run_result
