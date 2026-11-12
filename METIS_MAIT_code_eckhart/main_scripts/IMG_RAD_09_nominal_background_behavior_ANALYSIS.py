"""
Analysis of the IMG-RAD-09 nominal background behaviour data written by
IMG_RAD_09_nominal_background_behavior_SIM.py.

Reqs.:
- Ref. IMG-RAD-09 Nominal background behaviour

1. (requirement ID TBD) Verify sensititity requirement on point source in a 10000
second exposure.

Measured: the 1-sigma point-source noise of the background-subtracted 10000 s
frame (BCKGD_SUBTED = exposure A - exposure B of a uniform background; in the SIM a WCU
black-body flat with no source), from the robust spread
of the sums in apertures of radius aperture_radius_lambda_over_d x lambda/D, in absolute
flux units via the counts-to-flux calibration. The calibration and the noise limit are
PLACEHOLDERS (config/config_file_IMG_RAD_09_nominal_background_observing.yaml).

Pass/fail is decided in backbone_img_rad_09_nom.sens_10000s_check (noise-aware: the
uncertainty of the noise estimate is included), and logged per run (requirement,
instrument, case) with its consistency with the injected truth in
backbone_img_rad_09_nom.log_test_outcome; log_summary tabulates all runs.
"""

import datetime
import logging

import modules.backbone_img_rad_09_nom as nom
from modules import analysis_common
from modules.config_common import load_observing_config
from modules.helpers import setup_logging
from modules.paths import project_root

CHECKS = {
    "SENS-10000S": nom.sens_10000s_check,
}


def main():
    """Run the IMG-RAD-09 analysis for each data state (requirement, instrument,
    case)."""

    stem = project_root()
    now = datetime.datetime.now()

    # initialize logging
    log_dir = stem + "logs/IMG_RAD_09_nom_bkgd_logs/"
    log_file_name = (
        log_dir
        + "log_IMG_RAD_09_analysis_nominal_background_"
        + now.strftime("%Y-%m-%d_%H-%M-%S")
        + ".txt"
    )
    setup_logging(log_dir=log_dir, log_file_name=log_file_name, now=now)

    # observing parameters: common, then IMG-RAD-09's (calibration, apertures, limits)
    observing_config = load_observing_config(
        stem, "config/config_file_IMG_RAD_09_nominal_background_observing.yaml"
    )
    saturation_frac = float(observing_config["img_rad_09"]["saturation_frac"])

    # data states: one per run (defaults + per-run overrides)
    data_states = analysis_common.load_data_states(
        stem,
        "config/config_file_IMG_RAD_09_nominal_background_data_states.yaml",
        path_keys=("file_name_abs", "results_write_dir"),
    )

    results = []
    for state in data_states:
        logging.info("==================================================")
        result = nom.populate_run(state, observing_config)

        # raw readout below saturation (SAT_ADU written by the SIM)
        analysis_common.check_saturation(
            result.file_name, float(result.header["SAT_ADU"]), frac=saturation_frac
        )

        # point-source noise in flux units, then the requirement's check
        result = nom.measure_noise(result, observing_config)
        result = CHECKS[result.req_id](result)
        nom.plot_run(result)

        # outcome of the run
        results.append(nom.log_test_outcome(result))

    nom.log_summary(results)


if __name__ == "__main__":
    main()
