"""Analysis backbone for the IMG-OPT-02 geometric distortion test."""

import logging
import os
import pickle
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from pipeline_registry import (
    CLUSTER_DISTORTION,
    COLOR_DISTORTION,
    pipeline_stage,
)

from .helpers import fit_psf_gaussian_from_native_array, load_config_and_pipe
from .psf_grid_prep import (
    load_grid_data_from_fits,
    oversample_1st_pass_centroid,
)


@dataclass(frozen=True)
class SinglePsfFitResult:
    """
    Per-PSF outputs from Gaussian fit (native pixel coords) and optional Strehl dicts.
    """

    # coord_x_fullarray_normsamp: float
    # coord_y_fullarray_normsamp: float
    # fwhm_x_fullarray_normsamp: float
    # fwhm_y_fullarray_normsamp: float
    # amplitude_counts: float
    centroid_results: dict


def process_one_psf_image_distortion(
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
    """
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
        X and Y coordinates of the 1st-pass centroid in oversampled pixel coordinates
        (using the full 2D array)
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
    # not used yet: accepted so the caller can pass the same settings as IMG-03
    del filter_name, fp_mask, pp_mask, config_observing, fit_method
    del fit_simmed_psf, fit_annular_aperture_fixed, fit_annular_aperture_free
    logging.info("Processing PSF %s of %s", num_coord, num_psfs_to_process)

    # cookie_edge_size_original = original_array.shape[0]

    _, gaussian_fit_outputs = fit_psf_gaussian_from_native_array(
        original_array=original_array,
        oversample_factor=oversample_factor,
        coords_xy_1st_pass_normsamp=coords_xy_1st_pass_normsamp,
        edge_size_oversamp=30,
    )
    oversampled_array = gaussian_fit_outputs["oversampled_array"]
    coords_xy_1st_pass_oversamp_fullarray = gaussian_fit_outputs[
        "coords_xy_1st_pass_oversamp_fullarray"
    ]
    x_center_pix_gaussian_best_fit_fullarray_oversamp = gaussian_fit_outputs[
        "x_center_pix_fullarray_oversamp"
    ]
    y_center_pix_gaussian_best_fit_fullarray_oversamp = gaussian_fit_outputs[
        "y_center_pix_fullarray_oversamp"
    ]
    amplitude_counts_gaussian_best_fit_cookie_oversamp = gaussian_fit_outputs[
        "amplitude_counts_cookie_oversamp"
    ]
    x_center_pix_gaussian_best_fit_fullarray_normsamp = gaussian_fit_outputs[
        "x_center_pix_fullarray_normsamp"
    ]
    y_center_pix_gaussian_best_fit_fullarray_normsamp = gaussian_fit_outputs[
        "y_center_pix_fullarray_normsamp"
    ]
    fwhm_x_pix_gaussian_best_fit_fullarray_normsamp = gaussian_fit_outputs[
        "fwhm_x_pix_fullarray_normsamp"
    ]
    fwhm_y_pix_gaussian_best_fit_fullarray_normsamp = gaussian_fit_outputs[
        "fwhm_y_pix_fullarray_normsamp"
    ]

    # consider the center of the frame to be the first guess for the 2nd-pass centroid
    # (remember, the 1st-pass was used to cut out the PSF in the first place)
    # x_cen_oversamp = cookie_cutout_this_psf_oversamp.shape[1] / 2
    # y_cen_oversamp = cookie_cutout_this_psf_oversamp.shape[0] / 2

    logging.info(
        "Gaussian-fit FWHM (x, y) (native sampling): (%.2f, %.2f)",
        fwhm_x_pix_gaussian_best_fit_fullarray_normsamp,
        fwhm_y_pix_gaussian_best_fit_fullarray_normsamp,
    )
    logging.info(
        "Gaussian-fit centroid (x, y) (native sampling): (%.2f, %.2f)",
        x_center_pix_gaussian_best_fit_fullarray_normsamp,
        y_center_pix_gaussian_best_fit_fullarray_normsamp,
    )

    # save an FYI plot of oversampled region around PSF
    plt.clf()
    edge_size_oversamp = 20
    idx_cutout_x1 = int(
        x_center_pix_gaussian_best_fit_fullarray_oversamp - edge_size_oversamp / 2
    )
    idx_cutout_x2 = int(
        x_center_pix_gaussian_best_fit_fullarray_oversamp + edge_size_oversamp / 2
    )
    idx_cutout_y1 = int(
        y_center_pix_gaussian_best_fit_fullarray_oversamp - edge_size_oversamp / 2
    )
    idx_cutout_y2 = int(
        y_center_pix_gaussian_best_fit_fullarray_oversamp + edge_size_oversamp / 2
    )
    plt.figure(figsize=(12, 12))
    plt.imshow(
        oversampled_array[idx_cutout_y1:idx_cutout_y2, idx_cutout_x1:idx_cutout_x2],
        origin="lower",
        cmap="gray_r",
    )
    plt.scatter(
        coords_xy_1st_pass_oversamp_fullarray[0] - idx_cutout_x1,
        coords_xy_1st_pass_oversamp_fullarray[1] - idx_cutout_y1,
        color="red",
        s=50,
        marker="x",
        label="1st pass",
        alpha=1,
    )
    plt.scatter(
        x_center_pix_gaussian_best_fit_fullarray_oversamp - idx_cutout_x1,
        y_center_pix_gaussian_best_fit_fullarray_oversamp - idx_cutout_y1,
        color="green",
        s=50,
        marker="+",
        label="2nd pass",
        alpha=1,
    )
    plt.title(f"Oversampled region around PSF {num_coord} of {num_psfs_to_process}")
    plt.legend()
    file_name_plot = os.path.join(
        results_write_dir,
        f"fyi_plot_oversampled_region_around_psf_{num_coord}.png",
    )
    plt.savefig(file_name_plot)
    logging.info("Saved %s", file_name_plot)
    plt.close()

    centroid_results = {
        "coord_x_fullarray_normsamp": float(
            x_center_pix_gaussian_best_fit_fullarray_normsamp
        ),
        "coord_y_fullarray_normsamp": float(
            y_center_pix_gaussian_best_fit_fullarray_normsamp
        ),
        "fwhm_x_fullarray_normsamp": float(
            fwhm_x_pix_gaussian_best_fit_fullarray_normsamp
        ),
        "fwhm_y_fullarray_normsamp": float(
            fwhm_y_pix_gaussian_best_fit_fullarray_normsamp
        ),
        "amplitude_counts": float(amplitude_counts_gaussian_best_fit_cookie_oversamp),
    }

    # return SinglePsfFitResult(
    #     coord_x_fullarray_normsamp=float(
    #         x_center_pix_gaussian_best_fit_fullarray_normsamp
    #     ),
    #     coord_y_fullarray_normsamp=float(
    #         y_center_pix_gaussian_best_fit_fullarray_normsamp
    #     ),
    # fwhm_x_fullarray_normsamp=float(fwhm_x_pix_gaussian_best_fit_fullarray_normsamp),
    # fwhm_y_fullarray_normsamp=float(fwhm_y_pix_gaussian_best_fit_fullarray_normsamp),
    #     amplitude_counts=float(amplitude_counts_gaussian_best_fit_cookie_oversamp),
    # )
    #
    return centroid_results


@pipeline_stage(
    name="image_distortion",
    depends_on=("generate_distortion_sim",),
    cluster=CLUSTER_DISTORTION,
    cluster_color=COLOR_DISTORTION,
    label="Measure distortion",
)
def image_distortion(
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
    """
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
    """

    # edge_size_original = 21 # pixels along one side of the cutout, original pixel
    # sampling
    oversample_factor = 3  # try to keep odd to facilitate centering
    logging.info("PSF oversampling factor: %s", oversample_factor)

    # retrieve coord guesses as a starting point
    config_coords_guesses_config = load_config_and_pipe(
        config_file_choice=config_coords_guesses_file_name,
        print_one_line=False,
    )

    # retrieve data, oversample, and do 1st-pass centroiding
    # (note oversampled empirical frame is only used for centroiding; the cost function
    # for fitting later on just uses the frame as-is)
    grid_data, grid_header = load_grid_data_from_fits(file_name, hdu_index=1)
    prep = oversample_1st_pass_centroid(
        grid_data,
        config_coords_guesses_config,
        psfs_subset=psfs_subset,
        oversample_factor=oversample_factor,
        grid_header=grid_header,
    )

    # unpack quantities
    grid_data = prep.grid_data  # original data (native pixel scale)
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
    centroid_results_all = {}  # to contain info from all the PSFs

    # oversample the grid data
    # grid_data_oversamp = zoom(grid_data, oversample_factor, order=3)

    # loop over all PSFs that we want to process from this one detector readout
    for num_coord in range(num_psfs_to_process):

        # centroid_results_this_psf = {} # to contain info from this PSF alone
        # make cutout of the PSF from the original array, using the closest int to the
        # 1st pass centroids

        # 1-st pass coords of the PSF in the original array
        x_cen_1st_pass_native = coords_centroided_1st_pass_all_native[num_coord][1]
        y_cen_1st_pass_native = coords_centroided_1st_pass_all_native[num_coord][0]

        # convert to oversampled coordinates
        # x_cen_1st_pass_oversamp = x_cen_1st_pass_native * oversample_factor
        # y_cen_1st_pass_oversamp = y_cen_1st_pass_native * oversample_factor

        # find coordinates (and later other things?)
        centroid_results_this_psf = process_one_psf_image_distortion(
            num_coord,
            num_psfs_to_process,
            original_array=grid_data,
            oversample_factor=oversample_factor,
            coords_xy_1st_pass_normsamp=[
                x_cen_1st_pass_native,
                y_cen_1st_pass_native,
            ],
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

        # # for each coord value in result, put it in centroid_results_this_psf as a
        # key-value pair
        # for key, value in result.centroid_results.items():
        #     centroid_results_this_psf[key] = value
        # # also include the 1st-pass centroid coordinates
        # centroid_results_this_psf['x_cen_1st_pass_native'] = x_cen_1st_pass_native
        # centroid_results_this_psf['y_cen_1st_pass_native'] = y_cen_1st_pass_native
        #
        # put the results from this PSF into the overall dictionary
        centroid_results_all[f"psf_num_{num_coord:02d}"] = centroid_results_this_psf

    # pass/fail placeholder: criteria for distortion requirement TBD
    # (eventually: pass_fail = psf_results[criterion_key] >= 0.8 for each PSF)
    pass_fail_list = [False for _ in centroid_results_all.values()]
    pass_fail_all = all(pass_fail_list)

    logging.info("--------------------------------")
    logging.info("--------------------------------")
    logging.info(
        "Reqs (Ref. Overleaf doc IMG_OPT_02_Test_Geometric_Distortion):\n"
        "1) METIS-1097: The Imager shall provide a pixel scale of 5.47 +0.26/-0.26 "
        "mas/pix "
        "for the LM-band and 6.79 +0.25/-0.50 mas/pix for the N-band.\n"
        "2) METIS-3502: After calibration, the distortions introduced by METIS shall "
        "be "
        "removed to better than 0.5 mas (ca. 1/10 px for the L-band imager) over the "
        "full "
        "field of view.\n"
        "3) METIS-8222: The center of the H2RG chip within IMG-LM-DET shall be offset "
        "from "
        "the METIS optical axis by 175 mas +/- 25 mas on-sky PtV in the 'across H2RG "
        "stripe' "
        "direction (i.e., perpendicular to the orientation of the 32 stripes in the "
        "H2RG detector).\n"
        "4) METIS-9920: Image scale and distortion of METIS shall be constant (for "
        "each optical "
        "configuration, even after change of observing modes) to an accuracy of 1e-3 "
        "(goal: 1e-4) "
        "at L/M-band and 2e-3 (goal: 2e-4) at N-band with respect to the full field of "
        "view."
    )
    logging.info("--------------------------------")
    logging.info("--------------------------------")

    # pickle the results
    basename_file_name_pickle = (
        f"centroid_results_all_{fp_mask}_{pp_mask}_{filter_name}.pkl"
    )
    abs_file_name_pickle = os.path.join(results_write_dir, basename_file_name_pickle)
    with open(abs_file_name_pickle, "wb") as f:
        pickle.dump(
            {
                "centroid_results_all": centroid_results_all,
                "pass_fail_all": pass_fail_all,
            },
            f,
        )
    logging.info("Saved centroid results to %s", abs_file_name_pickle)

    return centroid_results_all


# Requirement checks: stubs until the distortion machinery is in place. Each returns a
# verdict (PASS, FAIL or INCONCLUSIVE); for now always INCONCLUSIVE.


@pipeline_stage(
    name="metis_1097_pixel_scale",
    depends_on=("image_distortion",),
    cluster=CLUSTER_DISTORTION,
    cluster_color=COLOR_DISTORTION,
    label="METIS-1097: pixel scale",
)
def metis_1097_pixel_scale(centroid_results_all, filter_name, config_observing):
    """
    METIS-1097: the Imager shall provide a pixel scale of 5.47 +0.26/-0.26 mas/pix for
    the LM-band and 6.79 +0.25/-0.50 mas/pix for the N-band.

    STUB: pixel scale calculation machinery TBD.

    INPUTS:
    - centroid_results_all (dict): per-PSF centroid results from image_distortion
    - filter_name (str): observing filter (selects the LM- or N-band limits)
    - config_observing (dict): observing configuration

    OUTPUTS:
    - verdict (str): PASS, FAIL or INCONCLUSIVE
    """
    # not used yet
    del centroid_results_all, filter_name, config_observing
    logging.info("METIS-1097: PIXEL SCALE CALCULATION MACHINERY TBD")
    return "INCONCLUSIVE"


@pipeline_stage(
    name="metis_3502_residual_distortion",
    depends_on=("image_distortion",),
    cluster=CLUSTER_DISTORTION,
    cluster_color=COLOR_DISTORTION,
    label="METIS-3502: residual distortion after calibration",
)
def metis_3502_residual_distortion(centroid_results_all, filter_name, config_observing):
    """
    METIS-3502: after calibration, the distortions introduced by METIS shall be removed
    to better than 0.5 mas (ca. 1/10 px for the L-band imager) over the full field of
    view.

    STUB: distortion calculation machinery TBD.

    INPUTS:
    - centroid_results_all (dict): per-PSF centroid results from image_distortion
    - filter_name (str): observing filter
    - config_observing (dict): observing configuration

    OUTPUTS:
    - verdict (str): PASS, FAIL or INCONCLUSIVE
    """
    # not used yet
    del centroid_results_all, filter_name, config_observing
    logging.info("METIS-3502: DISTORTION CALCULATION MACHINERY TBD")
    return "INCONCLUSIVE"


@pipeline_stage(
    name="metis_8222_detector_offset",
    depends_on=("image_distortion",),
    cluster=CLUSTER_DISTORTION,
    cluster_color=COLOR_DISTORTION,
    label="METIS-8222: H2RG offset from optical axis",
)
def metis_8222_detector_offset(centroid_results_all, filter_name, config_observing):
    """
    METIS-8222: the center of the H2RG chip within IMG-LM-DET shall be offset from the
    METIS optical axis by 175 mas +/- 25 mas on-sky PtV in the 'across H2RG stripe'
    direction (i.e., perpendicular to the orientation of the 32 stripes in the H2RG
    detector). Applies to the LM-band detector only.

    STUB: centering calculation machinery TBD.

    INPUTS:
    - centroid_results_all (dict): per-PSF centroid results from image_distortion
    - filter_name (str): observing filter
    - config_observing (dict): observing configuration

    OUTPUTS:
    - verdict (str): PASS, FAIL or INCONCLUSIVE
    """
    # not used yet
    del centroid_results_all, filter_name, config_observing
    logging.info("METIS-8222: CENTERING CALCULATION MACHINERY TBD")
    return "INCONCLUSIVE"


@pipeline_stage(
    name="metis_9920_scale_stability",
    depends_on=("image_distortion",),
    cluster=CLUSTER_DISTORTION,
    cluster_color=COLOR_DISTORTION,
    label="METIS-9920: stability of scale and distortion",
)
def metis_9920_scale_stability(centroid_results_all, filter_name, config_observing):
    """
    METIS-9920: image scale and distortion of METIS shall be constant (for each optical
    configuration, even after change of observing modes) to an accuracy of 1e-3 (goal:
    1e-4) at L/M-band and 2e-3 (goal: 2e-4) at N-band with respect to the full field of
    view.

    STUB: stability calculation machinery TBD (needs measurements of the same
    configuration before and after a change of observing mode).

    INPUTS:
    - centroid_results_all (dict): per-PSF centroid results from image_distortion
    - filter_name (str): observing filter (selects the L/M- or N-band limit)
    - config_observing (dict): observing configuration

    OUTPUTS:
    - verdict (str): PASS, FAIL or INCONCLUSIVE
    """
    # not used yet
    del centroid_results_all, filter_name, config_observing
    logging.info("METIS-9920: STABILITY CALCULATION MACHINERY TBD")
    return "INCONCLUSIVE"


@pipeline_stage(
    name="img_02_test_outcome",
    depends_on=(
        "metis_1097_pixel_scale",
        "metis_3502_residual_distortion",
        "metis_8222_detector_offset",
        "metis_9920_scale_stability",
    ),
    cluster=CLUSTER_DISTORTION,
    cluster_color=COLOR_DISTORTION,
    label="IMG-OPT-02 pass/fail",
)
def log_test_outcome(verdicts, file_name):
    """
    Decide and log the pass/fail of each IMG-OPT-02 requirement, and of the whole test.

    Overall: PASS only if all requirements pass; FAIL if any fails; else INCONCLUSIVE.

    STUB: the requirement checks are stubs, so the outcome is INCONCLUSIVE for now.

    INPUTS:
    - verdicts (dict): requirement ID (e.g. 'METIS-1097') -> verdict (PASS, FAIL or
      INCONCLUSIVE)
    - file_name (str): FITS file the verdicts are for

    OUTPUTS:
    - verdict_overall (str): PASS, FAIL or INCONCLUSIVE
    """
    logging.info("--------------------------------------------------")
    logging.info("--------------------------------------------------")
    logging.info("File: %s", file_name)
    for req_id, verdict in verdicts.items():
        logging.info("%s: %s", req_id, verdict)
    logging.info("--------------------------------------------------")
    if all(v == "PASS" for v in verdicts.values()):
        verdict_overall = "PASS"
    elif any(v == "FAIL" for v in verdicts.values()):
        verdict_overall = "FAIL"
    else:
        verdict_overall = "INCONCLUSIVE"
    logging.info("IMG 02 DISTORTION TEST: %s", verdict_overall)
    logging.info("--------------------------------------------------")
    return verdict_overall
