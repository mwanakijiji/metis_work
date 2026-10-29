"""
Analysis backbone for the IMG-OPT-03 PSF image quality (Strehl) test.

Stages, called in this order by the ANALYSIS script:
prepare_psf_grid -> process_one_psf (once per PSF, on cutout_psf) -> collect_strehls
-> plot_strehl_fov, and the requirement checks metis_1408_1409_check,
metis_2864_check, metis_3503_check -> log_test_outcome
"""

import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pipeline_registry import CLUSTER_PSF, COLOR_PSF, pipeline_stage

from .helpers import fit_psf_gaussian_from_native_array, load_config_and_pipe
from .paths import irdb_dir
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
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
        ``polychromatic_observing_filters_leaf_name`` (absolute filter curve paths,
        under the IRDB) and ``cutout_edge_native``.
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
    # filter curve paths in the config are relative to the IRDB; make them absolute
    config_resolved["polychromatic_observing_filters_leaf_name"] = {
        name: os.path.join(irdb_dir(), leaf)
        for name, leaf in config_observing[
            f"polychromatic_observing_filters_{band}_leaf_name"
        ].items()
    }
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
class PsfGrid:
    """
    One FITS file of PSFs, loaded and 1st-pass centroided (from prepare_psf_grid):
    everything needed to cut out and fit each PSF. (Not to be confused with
    psf_grid_prep.PsfGridPrep, the lower-level centroiding output it is built from.)
    """

    file_name_abs: str
    filter_name: str
    fp_mask: str
    pp_mask: str
    config_observing: dict  # with mask-dependent values resolved
    grid_data: np.ndarray  # frame (native pixel scale), for plotting
    grid_data_original: np.ndarray  # frame (native pixel scale), for the cutouts
    coords_centroided_1st_pass_all_native: Any  # per PSF: (y, x), native pixels
    num_psfs_to_process: int
    total_psfs: int
    oversample_factor: int
    edge_size_original: int  # cutout edge, native pixels


@dataclass
class StrehlRunResult:
    """
    Strehl results for all PSFs of one FITS file (from collect_strehls), plus the
    requirement verdicts (from the requirement checks) and the overall verdict (from
    log_test_outcome).
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
    name="prepare_psf_grid",
    depends_on=("generate_psf_quality_sim",),
    cluster=CLUSTER_PSF,
    cluster_color=COLOR_PSF,
    label="Load frame, 1st-pass centroids",
)
def prepare_psf_grid(
    file_name_abs,
    fp_mask,
    pp_mask,
    filter_name=None,
    psfs_subset="all",
    config_coords_guesses_file_name_abs=None,
    config_observing=None,
    oversample_factor=3,
):
    """
    Load a FITS file with a grid of PSFs, resolve the mask-dependent observing config,
    and find the 1st-pass centroids of the PSFs.

    INPUTS
    ----------
    file_name_abs : str
        Path to the FITS file containing the PSF grid to analyze.
    fp_mask : str
        Focal-plane mask label (sets the LM/N band).
    pp_mask : str
        Pupil-plane mask label (sets the reference aperture geometry).
    filter_name : str, optional
        Name of the observing filter associated with the PSF grid.
    psfs_subset : str or int, optional
        Either ``"all"`` to process the full grid or an integer giving how many PSFs
        from the start of the centroid list to process.
    config_coords_guesses_file_name_abs : str, optional
        Path to the configuration file containing initial coordinate guesses.
    config_observing : dict, optional
        Observing configuration dictionary.
    oversample_factor : int, optional
        Oversampling factor for centroiding and fitting (keep odd to facilitate
        centering).

    OUTPUTS
    -------
    PsfGrid
        The frame, the resolved observing config, the 1st-pass centroids and PSF
        counts; input to cutout_psf / process_one_psf, collect_strehls and
        plot_strehl_fov.
    """

    # resolve cold-stop geometry and LM/N band settings for this mask combination
    config_observing = resolve_config_for_masks(config_observing, fp_mask, pp_mask)

    edge_size_original = config_observing[
        "cutout_edge_native"
    ]  # pixels along one side of the cutout, original pixel sampling
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

    logging.info("Finding PSF centroids, first pass (via oversample_1st_pass_centroid)")
    logging.info(
        "Raw PSF cutout size (oversampled): %s", prep.raw_cutout_size_oversampled
    )
    logging.info("Total PSFs: %s", prep.total_psfs)
    logging.info(
        "Processing %s out of %s PSFs", prep.num_psfs_to_process, prep.total_psfs
    )

    return PsfGrid(
        file_name_abs=file_name_abs,
        filter_name=filter_name,
        fp_mask=fp_mask,
        pp_mask=pp_mask,
        config_observing=config_observing,
        grid_data=prep.grid_data,
        grid_data_original=prep.grid_data_original,
        coords_centroided_1st_pass_all_native=(
            prep.coords_centroided_1st_pass_all_native
        ),
        num_psfs_to_process=int(prep.num_psfs_to_process),
        total_psfs=int(prep.total_psfs),
        oversample_factor=oversample_factor,
        edge_size_original=int(edge_size_original),
    )


def cutout_psf(psf_grid, num_coord):
    """
    Native-sampling square cutout of PSF num_coord, around the closest ints to its
    1st-pass centroid (input to process_one_psf).

    INPUTS
    ----------
    psf_grid : PsfGrid
        From prepare_psf_grid.
    num_coord : int
        Zero-based index of the PSF.

    OUTPUTS
    -------
    np.ndarray
        Cutout of side psf_grid.edge_size_original (native pixels).
    """
    x_cen_1st_pass_native = psf_grid.coords_centroided_1st_pass_all_native[num_coord][1]
    y_cen_1st_pass_native = psf_grid.coords_centroided_1st_pass_all_native[num_coord][0]
    half_edge = 0.5 * psf_grid.edge_size_original
    return psf_grid.grid_data_original[
        int(x_cen_1st_pass_native - half_edge) : int(x_cen_1st_pass_native + half_edge),
        int(y_cen_1st_pass_native - half_edge) : int(y_cen_1st_pass_native + half_edge),
    ]


@pipeline_stage(
    name="process_one_psf",
    depends_on=("prepare_psf_grid",),
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
        Native-sampling square cutout containing the PSF of interest (e.g. from
        cutout_psf).
    oversample_factor : int
        Factor used to oversample the PSF cutout before centroiding and model fitting.
    filter_name : str
        Name of the observing filter associated with the PSF.
    fp_mask : str
        Focal-plane mask label used for bookkeeping and plot naming.
    pp_mask : str
        Pupil-plane mask label used for bookkeeping and plot naming.
    config_observing : dict
        Observing configuration dictionary with mask-dependent values resolved (e.g.
        PsfGrid.config_observing), passed to downstream fitting routines.
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
        metrics (input to collect_strehls).
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
    name="collect_strehls",
    depends_on=("process_one_psf",),
    cluster=CLUSTER_PSF,
    cluster_color=COLOR_PSF,
    label="Collect Strehl over FOV",
)
def collect_strehls(psf_grid, fit_results):
    """
    Gather the per-PSF fit results (in the order of the 1st-pass centroids) into one
    result for the file.

    INPUTS
    ----------
    psf_grid : PsfGrid
        From prepare_psf_grid.
    fit_results : list[SinglePsfFitResult]
        From process_one_psf, one per PSF processed (index = num_coord).

    OUTPUTS
    -------
    StrehlRunResult
        Per-PSF Strehl estimates (+ 1st-pass centroids) and PSF counts; verdicts still
        empty (see the requirement checks and log_test_outcome).
    """
    strehl_results_all = {}
    for num_coord, fit_result in enumerate(fit_results):
        strehl_results_this_psf = dict(fit_result.strehl_updates)
        # also include the 1st-pass centroid coordinates
        coords = psf_grid.coords_centroided_1st_pass_all_native[num_coord]
        strehl_results_this_psf["x_cen_1st_pass_native"] = coords[1]
        strehl_results_this_psf["y_cen_1st_pass_native"] = coords[0]
        strehl_results_all[f"psf_num_{num_coord:02d}"] = strehl_results_this_psf

    # pickle the results (skipped for now)
    # basename_file_name_pickle = (
    #     f"strehl_results_all_{fp_mask}_{pp_mask}_{filter_name}.pkl"
    # )
    # abs_file_name_pickle = os.path.join(results_write_dir, basename_file_name_pickle)
    # with open(abs_file_name_pickle, "wb") as f:
    #     pickle.dump({"strehl_results_all": strehl_results_all}, f)
    # logging.info(f"Saved strehl results to {abs_file_name_pickle}")

    return StrehlRunResult(
        file_name_abs=psf_grid.file_name_abs,
        filter_name=psf_grid.filter_name,
        fp_mask=psf_grid.fp_mask,
        pp_mask=psf_grid.pp_mask,
        band=psf_grid.config_observing["band"],
        strehl_results_all=strehl_results_all,
        total_psfs=psf_grid.total_psfs,
        num_psfs_processed=len(fit_results),
    )


@pipeline_stage(
    name="plot_strehl_fov",
    depends_on=("collect_strehls",),
    cluster=CLUSTER_PSF,
    cluster_color=COLOR_PSF,
    label="Plot Strehl over FOV",
)
def plot_strehl_fov(run_result, grid_data, results_write_dir):
    """
    FYI plots on the detector frame: the Strehl of each PSF, and the Strehl from all
    methods for one randomly chosen PSF.

    INPUTS
    ----------
    run_result : StrehlRunResult
        From collect_strehls.
    grid_data : np.ndarray
        The detector frame (native pixel scale), e.g. PsfGrid.grid_data.
    results_write_dir : str
        Directory where the plots are saved.

    OUTPUTS
    -------
    None
        Saves the plots.
    """
    strehl_results_all = run_result.strehl_results_all
    filter_name = run_result.filter_name
    fp_mask = run_result.fp_mask
    pp_mask = run_result.pp_mask

    # plot the grid_data and annotate it with the Strehl of each PSF
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


def _criterion_strehls(run_result, criterion_key):
    """
    Per PSF, the Strehl estimate used for pass/fail (None where it was not computed),
    and the PSFs where it is missing.
    """
    strehls = {
        psf_name: psf_results.get(criterion_key)
        for psf_name, psf_results in run_result.strehl_results_all.items()
    }
    missing = [name for name, strehl in strehls.items() if strehl is None]
    return strehls, missing


def _full_grid(run_result):
    """Whether every PSF of the grid was analysed (i.e. the entire FOV)."""
    return run_result.num_psfs_processed >= run_result.total_psfs


def _strehl_verdict(run_result, criterion_key, strehl_min, log_per_psf=True):
    """
    Verdict of the Strehl criterion (every PSF has Strehl >= strehl_min) over the FOV:
    FAIL if any PSF is below; PASS if none is and the whole grid was analysed;
    INCONCLUSIVE if none is but only a subset of the grid was analysed; NOT EVALUATED
    if criterion_key is missing for some PSF (e.g. that fit was switched off).
    """
    strehls, missing = _criterion_strehls(run_result, criterion_key)
    if missing:
        logging.error(
            "No %s for %s PSFs (%s); Strehl criterion not evaluated",
            criterion_key,
            len(missing),
            ", ".join(missing),
        )
        return "NOT EVALUATED"

    below = []
    for psf_name, strehl in strehls.items():
        ok = np.isfinite(strehl) and strehl >= strehl_min
        if not ok:
            below.append(psf_name)
        if log_per_psf:
            logging.info(
                "  %s: %s = %.3f %s %.2f",
                psf_name,
                criterion_key,
                strehl,
                ">=" if ok else "<",
                strehl_min,
            )
    if below:
        logging.info(
            "%s of %s PSFs below Strehl %.2f: %s",
            len(below),
            len(strehls),
            strehl_min,
            ", ".join(below),
        )
        return "FAIL"
    if _full_grid(run_result):
        return "PASS"
    logging.info(
        "All analysed PSFs have Strehl >= %.2f, but only %s of %s PSFs were analysed "
        "(not the entire FOV)",
        strehl_min,
        run_result.num_psfs_processed,
        run_result.total_psfs,
    )
    return "INCONCLUSIVE"


@pipeline_stage(
    name="img_03_metis_1408_1409_check",
    depends_on=("collect_strehls",),
    cluster=CLUSTER_PSF,
    cluster_color=COLOR_PSF,
    label="METIS-1408/-1409: Strehl >= 0.8 over FOV",
)
def metis_1408_1409_check(
    run_result, criterion_key=STREHL_CRITERION_KEY, strehl_min=STREHL_MIN
):
    """
    Decide METIS-1408 (Strehl >= 80% at lambda > 3 um in all modes over the entire FOV)
    and METIS-1409 (Strehl > 0.8 at 3 um for IMG-LM and -NQ over the full FOV): every
    PSF has Strehl (criterion_key) >= strehl_min (see _strehl_verdict for FAIL /
    INCONCLUSIVE / NOT EVALUATED).

    INPUTS:
    - run_result (StrehlRunResult): from collect_strehls
    - criterion_key (str): Strehl estimator used for pass/fail
    - strehl_min (float): minimum Strehl ratio

    OUTPUTS:
    - run_result (StrehlRunResult), with verdicts['METIS-1408'] and
      verdicts['METIS-1409'] set
    """
    verdict = _strehl_verdict(run_result, criterion_key, strehl_min)
    run_result.verdicts["METIS-1408"] = verdict
    run_result.verdicts["METIS-1409"] = verdict
    logging.info(
        "METIS-1408 (Strehl >= %.0f%% at lambda > 3 um, entire FOV): %s",
        100 * strehl_min,
        verdict,
    )
    logging.info(
        "METIS-1409 (Strehl > %.1f at 3 um, IMG-LM and -NQ, full FOV): %s",
        strehl_min,
        verdict,
    )
    return run_result


@pipeline_stage(
    name="img_03_metis_2864_check",
    depends_on=("collect_strehls",),
    cluster=CLUSTER_PSF,
    cluster_color=COLOR_PSF,
    label="METIS-2864: Strehl > 0.8, WCU+CFO+IMG-LM path",
)
def metis_2864_check(
    run_result, criterion_key=STREHL_CRITERION_KEY, strehl_min=STREHL_MIN
):
    """
    Decide METIS-2864 (minimum Strehl of the WCU+CFO+IMG-LM optical path > 80% at
    3.3 um over the entire FOV): the same Strehl criterion as metis_1408_1409_check,
    for LM data only; N/A for N-band data.

    INPUTS:
    - run_result (StrehlRunResult): from collect_strehls
    - criterion_key (str): Strehl estimator used for pass/fail
    - strehl_min (float): minimum Strehl ratio

    OUTPUTS:
    - run_result (StrehlRunResult), with verdicts['METIS-2864'] set
    """
    if run_result.band == "lm":
        # per-PSF values are already logged by metis_1408_1409_check
        verdict = _strehl_verdict(
            run_result, criterion_key, strehl_min, log_per_psf=False
        )
    else:
        verdict = "N/A"
    run_result.verdicts["METIS-2864"] = verdict
    logging.info(
        "METIS-2864 (Strehl > %.0f%% at 3.3 um, WCU+CFO+IMG-LM path, entire FOV): %s",
        100 * strehl_min,
        verdict,
    )
    return run_result


@pipeline_stage(
    name="img_03_metis_3503_check",
    depends_on=("collect_strehls",),
    cluster=CLUSTER_PSF,
    cluster_color=COLOR_PSF,
    label="METIS-3503: PSF characterised across FoV",
)
def metis_3503_check(run_result, criterion_key=STREHL_CRITERION_KEY):
    """
    Decide METIS-3503 (the shape of the instrument PSF can be characterised across the
    entire FoV using the WCU): PASS if every PSF in the grid was analysed and has a
    finite Strehl estimate (criterion_key); FAIL if any analysed PSF has a non-finite
    one; INCONCLUSIVE if only a subset of the grid was analysed; NOT EVALUATED if
    criterion_key is missing for some PSF.

    INPUTS:
    - run_result (StrehlRunResult): from collect_strehls
    - criterion_key (str): Strehl estimator whose fit must have succeeded

    OUTPUTS:
    - run_result (StrehlRunResult), with verdicts['METIS-3503'] set
    """
    strehls, missing = _criterion_strehls(run_result, criterion_key)
    n_finite = sum(
        1 for strehl in strehls.values() if strehl is not None and np.isfinite(strehl)
    )
    if missing:
        verdict = "NOT EVALUATED"
    elif n_finite < len(strehls):
        verdict = "FAIL"
    elif _full_grid(run_result):
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
    run_result.verdicts["METIS-3503"] = verdict
    logging.info(
        "METIS-3503 (PSF shape characterised across the entire FoV with the WCU): %s "
        "(%s of %s PSFs with a finite %s)",
        verdict,
        n_finite,
        run_result.total_psfs,
        criterion_key,
    )
    return run_result


@pipeline_stage(
    name="img_03_test_outcome",
    depends_on=(
        "img_03_metis_1408_1409_check",
        "img_03_metis_2864_check",
        "img_03_metis_3503_check",
    ),
    cluster=CLUSTER_PSF,
    cluster_color=COLOR_PSF,
    label="IMG-OPT-03 pass/fail",
)
def log_test_outcome(run_result):
    """
    Log the outcome of each IMG-OPT-03 requirement (as decided by
    metis_1408_1409_check, metis_2864_check and metis_3503_check) and of the whole
    test: PASS if all applicable requirements pass; FAIL if any fails; else
    INCONCLUSIVE.

    INPUTS:
    - run_result (StrehlRunResult): with verdicts set by the requirement checks

    OUTPUTS:
    - run_result (StrehlRunResult), with verdict_overall set
    """
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
    for requirement in ("METIS-1408", "METIS-1409", "METIS-2864", "METIS-3503"):
        logging.info(
            "%s: %s", requirement, run_result.verdicts.get(requirement, "NOT EVALUATED")
        )
    logging.info("--------------------------------------------------")

    applicable = [v for v in run_result.verdicts.values() if v != "N/A"]
    if applicable and all(v == "PASS" for v in applicable):
        run_result.verdict_overall = "PASS"
    elif "FAIL" in applicable:
        run_result.verdict_overall = "FAIL"
    else:
        run_result.verdict_overall = "INCONCLUSIVE"
    logging.info("IMG 03 PSF QUALITY TEST: %s", run_result.verdict_overall)
    logging.info("--------------------------------------------------")

    return run_result
