"""
Does some simple analysis of simulated images written out by the sim notebook.

!!! TBD MORE DISTORTION INFO FROM DD NEEDED FIRST (warned in the log and stamped on
every plot this script saves)
"""

import datetime
import logging
import os

from modules.backbone_img_opt_02_distortion import (
    image_distortion,
    log_test_outcome,
    metis_1097_pixel_scale,
    metis_3502_residual_distortion,
    metis_8222_detector_offset,
    metis_9920_scale_stability,
)
from modules.config_common import load_observing_config
from modules.helpers import load_config_and_pipe, setup_logging
from modules.paths import project_root
from modules.tbd_warnings import log_tbd_distortion, stamp_all_saved_figures

# Make simulated data for the IMG-OPT-02 image distorition test

# Reqs.:
# - Ref. Overleaf doc IMG_OPT_02_Test_Geometric_Distortion
#
# 1. METIS-1097: The Imager shall provide a pixel scale of 5.47 +0.26/-0.26 mas/pix for
# the LM-band and
#               6.79+0.25/-0.50 mas/pix for the N-band.
# METIS-3502: After calibration, the distortions introduced by METIS shall be removed to
# better than
#               0.5 mas (ca. 1/10 px for the L band imager) over the full field of view.
# METIS-8222: The center of the H2RG chip within IMG-LM-DET shall be offset from the
# METIS optical axis
#               by 175 mas ± 25 mas on-sky PtV in the "across H2RG stripe" direction
# (i.e., perpendicularly
#               to the orientation of the 32 stripes in the H2RG detector).
# METIS-9920: Image scale and distortion of METIS shall be constant (for each optical
# configuration, even after
#               change of observing modes) to an accuracy of 10-3 (goal: 10-4) at
# L/M-band and 2×10-3 (goal: 2×10-4)
#               at N-band with respect to the full field of view.


def main():
    """Run the IMG-OPT-02 distortion analysis for each data state."""

    stem = project_root()
    # config file with the data states (i.e., how to analyze each PSF)
    data_states_config_file = stem + "config/config_file_IMG_02_strehl_runs.yaml"

    now = datetime.datetime.now()

    # initialize logging
    log_dir = stem + "logs/IMG_02_logs/"
    log_file_name = (
        log_dir
        + "log_IMG_02_analysis_psf_image_quality_"
        + now.strftime("%Y-%m-%d_%H-%M-%S")
        + ".txt"
    )
    setup_logging(log_dir=log_dir, log_file_name=log_file_name, now=now)
    log_tbd_distortion("IMG-OPT-02 ANALYSIS")
    # the plots are made deep in the backbone and helpers: stamp the warning banner on
    # every figure this run saves
    stamp_all_saved_figures()

    # observing parameters (common to all tests)
    observing_config = load_observing_config(stem)

    # load data states (defaults + per-run overrides)
    data_states_config = load_config_and_pipe(
        config_file_choice=data_states_config_file, print_one_line=False
    )
    defaults = data_states_config.get("defaults", {})
    runs = data_states_config.get("runs", [])

    # to set up the data states, merge config data state defaults with overrides that
    # are specific for each run
    # 'data state' is defined as the set of parameters and the collection of images
    # corresponding to that set of parameters
    # (the only difference between the images is the dither position)
    # paths in the config are relative to the stem (image_distortion uses them as given)
    path_keys = ("file_name", "config_coords_guesses_file_name", "results_write_dir")
    data_states = []
    for entry in runs:
        merged = {**defaults, **entry}
        for key in path_keys:
            if merged.get(key) and not os.path.isabs(merged[key]):
                merged[key] = stem + merged[key]
        data_states.append(merged)

    # loop over each data state
    for state in data_states:  # [0:1]: # if just for a small test
        logging.info("==================================================")
        centroid_results_all = image_distortion(
            state["file_name"],
            fp_mask=state["fp_mask"],
            pp_mask=state["pp_mask"],
            filter_name=state["filter_name"],
            fit_simmed_psf=state["fit_simmed_psf"],
            fit_annular_aperture_free=state["fit_annular_aperture_free"],
            fit_annular_aperture_fixed=state["fit_annular_aperture_fixed"],
            psfs_subset=state["psfs_subset"],
            config_coords_guesses_file_name=state["config_coords_guesses_file_name"],
            config_observing=observing_config,
            results_write_dir=state["results_write_dir"],
            fit_method="curve_fit",
        )
        # requirement checks (stubs for now), then the pass/fail of the whole test
        checks = {
            "METIS-1097": metis_1097_pixel_scale,
            "METIS-3502": metis_3502_residual_distortion,
            "METIS-8222": metis_8222_detector_offset,
            "METIS-9920": metis_9920_scale_stability,
        }
        verdicts = {
            req_id: check(centroid_results_all, state["filter_name"], observing_config)
            for req_id, check in checks.items()
        }
        log_test_outcome(verdicts, state["file_name"])
        log_tbd_distortion(f"IMG-OPT-02 ANALYSIS, {state['filter_name']}")

    log_tbd_distortion("IMG-OPT-02 ANALYSIS")


if __name__ == "__main__":
    main()
