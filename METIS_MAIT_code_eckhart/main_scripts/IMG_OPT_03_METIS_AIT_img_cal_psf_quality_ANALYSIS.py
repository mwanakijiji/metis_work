"""
Does some simple analysis of simulated images written out by the sim notebook.
"""

import datetime
import logging
import os

import numpy as np
from astropy.io import fits

import modules.backbone_img_03_psf_quality as pq
from modules.helpers import load_config_and_pipe, setup_logging
from modules.strehl_fcns import imaging_band_from_fp_mask


def check_saturation(file_name_abs, fp_mask, config_observing, hdu_index=2, frac=0.75):
    """
    Log an error if the max counts in image extension ``hdu_index`` (the raw readout,
    before
    background subtraction) exceed ``frac`` of the detector saturation level. Does not
    stop the analysis.
    """
    band = imaging_band_from_fp_mask(fp_mask)
    saturation_adu = float(config_observing["saturation_adu"][f"img_{band}"])
    with fits.open(file_name_abs) as hdul:
        max_counts = float(np.nanmax(hdul[hdu_index].data))
    if max_counts > frac * saturation_adu:
        logging.error(
            "! ------- %s: max counts in extension %s = %.0f ADU > %s x saturation "
            "(%.0f of %.0f ADU, img_%s)",
            os.path.basename(file_name_abs),
            hdu_index,
            max_counts,
            frac,
            frac * saturation_adu,
            saturation_adu,
            band,
        )
    else:
        logging.info(
            "Saturation check OK for %s: max counts in extension %s = %.0f ADU <= %s x "
            "saturation (%.0f ADU, img_%s)",
            os.path.basename(file_name_abs),
            hdu_index,
            max_counts,
            frac,
            frac * saturation_adu,
            band,
        )


# Reqs.:
# - Ref. Overleaf doc IMG_OPT_04_Test_Description_PSF_Image_Quality
#
# 1. METIS-1408: Quality and alignment of the optical components within Mid-infrared ELT
# Imager and
# Spectrograph (METIS) shall provide diffraction limited performance (Strehl ≥ 80 %)
# at λ > 3μm in all modes over the entire FOV.
# 2. METIS-1409: The Instrument Wavefront Error (WFE) shall satisfy the diffraction
# limit requirement
# (Strehl>0.8) at lambda = 3 μm for IMG (both LM and NQ) and IMG. The minimum
# RMS WFE below shall be satisfied over the full Field Of View (FOV) relevant to the
# given optical path.
# 3. METIS-2864: The minimum Strehl ratio of the WCU+CFO+IMG-LM optical path shall be
# >80% at
# 3.3μm over the entire field of view.
# 4. METIS-3503: METIS shall be able to characterise the shape of the instrument PSF
# across the entire
# FoV using the WCU.
#
# Pass/fail of each requirement is decided in one function per requirement
# (backbone_img_03_psf_quality.metis_1408_1409_check, .metis_2864_check,
# .metis_3503_check), and the outcome of the whole test in
# backbone_img_03_psf_quality.log_test_outcome


def main():
    """Run the IMG-OPT-03 Strehl analysis for each data state."""

    stem = "/podman-share/metis_work/METIS_MAIT_code_eckhart/"
    # config file with the observing parameters
    observing_config_file = (
        stem + "config/config_file_IMG_OPT_03_psf_quality_observing_params.yaml"
    )
    # config file with the data states (i.e., how to analyze each PSF)
    data_states_config_file = (
        stem + "config/config_file_IMG_OPT_03_psf_quality_data_states.yaml"
    )

    now = datetime.datetime.now()

    # initialize logging
    log_dir = stem + "logs/IMG_OPT_03_psf_quality_logs/"
    log_file_name = (
        log_dir
        + "log_IMG_OPT_03_analysis_psf_image_quality_"
        + now.strftime("%Y-%m-%d_%H-%M-%S")
        + ".txt"
    )
    setup_logging(log_dir=log_dir, log_file_name=log_file_name, now=now)

    # read in main config file
    observing_config = load_config_and_pipe(
        config_file_choice=observing_config_file, print_one_line=False
    )

    # load data states (defaults + per-run overrides)
    data_states_config = load_config_and_pipe(
        config_file_choice=data_states_config_file, print_one_line=False
    )
    defaults = data_states_config.get("defaults", {})
    runs = data_states_config.get("runs", [])

    # to set up the data states, merge config data state defaults with overrides that
    # are specific for each run
    data_states = []

    for entry in runs:
        merged = {**defaults, **entry}
        # Prepend stem to relative file paths
        fname = merged["file_name_abs"]
        if not fname.startswith("/"):
            merged["file_name_abs"] = stem + fname
        results_write_dir = merged["results_write_dir"]
        if not results_write_dir.startswith("/"):
            merged["results_write_dir"] = stem + results_write_dir
        data_states.append(merged)

    # loop over each data state (which likely means a single input FITS file; but that
    # file can include multiple PSFs)
    for state in data_states:  # data_states[0:1]: if just for a small test
        check_saturation(
            state["file_name_abs"],
            fp_mask=state["fp_mask"],
            config_observing=observing_config,
        )

        # load the frame and find the 1st-pass centroids of the PSFs
        grid_prep = pq.prepare_psf_grid(
            state["file_name_abs"],
            fp_mask=state["fp_mask"],
            pp_mask=state["pp_mask"],
            filter_name=state["filter_name"],
            psfs_subset=state["psfs_subset"],
            config_coords_guesses_file_name_abs=state[
                "config_coords_guesses_file_name_abs"
            ],
            config_observing=observing_config,
        )

        # fit each PSF
        fit_results = []
        for num_coord in range(grid_prep.num_psfs_to_process):
            fit_results.append(
                pq.process_one_psf(
                    num_coord,
                    grid_prep.num_psfs_to_process,
                    cookie_cutout_original_this_psf=pq.cutout_psf(grid_prep, num_coord),
                    oversample_factor=grid_prep.oversample_factor,
                    filter_name=state["filter_name"],
                    fp_mask=state["fp_mask"],
                    pp_mask=state["pp_mask"],
                    config_observing=grid_prep.config_observing,
                    results_write_dir=state["results_write_dir"],
                    fit_method="curve_fit",
                    fit_simmed_psf=state["fit_simmed_psf"],
                    fit_annular_aperture_fixed=state["fit_annular_aperture_fixed"],
                    fit_annular_aperture_free=state["fit_annular_aperture_free"],
                )
            )

        # collect the Strehls over the FOV and plot them
        run_result = pq.collect_strehls(grid_prep, fit_results)
        pq.plot_strehl_fov(run_result, grid_prep.grid_data, state["results_write_dir"])

        # pass/fail of each requirement
        run_result = pq.metis_1408_1409_check(run_result)
        run_result = pq.metis_2864_check(run_result)
        run_result = pq.metis_3503_check(run_result)

        # outcome of the whole test
        pq.log_test_outcome(run_result)


if __name__ == "__main__":
    main()
