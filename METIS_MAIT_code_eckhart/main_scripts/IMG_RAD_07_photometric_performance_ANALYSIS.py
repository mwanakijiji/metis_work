"""
Analysis of the IMG-RAD-07 photometric performance flats written by
IMG_RAD_07_photometric_performance_SIM.py (DRAFT).

Reqs.:
- Ref. IMG-RAD-07 Photometric performance, E-XXX-NOVA-MET-NNNN 0-1, Table 1-1

4. METIS-2883 (verified by test at subsystem level): The source used for detector
linearity calibration METIS-3289 shall provide an illumination that is spatially
homogeneous within +/- 3% peak-to-valley over the science field of view of METIS.

6. METIS-3300 (verified by test at System level, applicable to WCU): The level of flux
of the flat field shall be tunable to 15 discrete values and shall be known at each step
to 0.3%. The resulting spatial illumination pattern shall remain constant across the
field within 0.5% (0.3% as goal) after tuning.

Not analysed: METIS-2594 (SIM is a stub), METIS-2595 (twice), METIS-3291 and METIS-3301
(not simulated).

Pass/fail of each requirement is decided in one function per requirement
(backbone_img_rad_07_phot.metis_2883_check, .metis_3300_check), and logged per run
(requirement, instrument, case) with its consistency with the injected truth in
backbone_img_rad_07_phot.log_test_outcome; log_summary tabulates all runs.
"""

import datetime
import logging

import modules.backbone_img_rad_07_phot as phot
from modules import analysis_common
from modules.config_common import load_observing_config
from modules.helpers import setup_logging
from modules.paths import project_root

CHECKS = {
    "METIS-2883": phot.metis_2883_check,
    "METIS-3300": phot.metis_3300_check,
}


def main():
    """Run the IMG-RAD-07 analysis for each data state (requirement, instrument,
    case)."""

    stem = project_root()
    now = datetime.datetime.now()

    # initialize logging
    log_dir = stem + "logs/IMG_RAD_07_phot_logs/"
    log_file_name = (
        log_dir
        + "log_IMG_RAD_07_analysis_photometric_performance_"
        + now.strftime("%Y-%m-%d_%H-%M-%S")
        + ".txt"
    )
    setup_logging(log_dir=log_dir, log_file_name=log_file_name, now=now)

    # observing parameters: common, then IMG-RAD-07's (analysis parameters, limits)
    observing_config = load_observing_config(
        stem, "config/config_file_IMG_RAD_07_phot_performance_observing.yaml"
    )
    saturation_frac = float(observing_config["img_rad_07"]["saturation_frac"])

    # data states: one per run (defaults + per-run overrides)
    data_states = analysis_common.load_data_states(
        stem,
        "config/config_file_IMG_RAD_07_phot_performance_data_states.yaml",
        path_keys=("file_name_abs", "file_name_pattern", "results_write_dir"),
    )

    results = []
    for state in data_states:
        logging.info("==================================================")
        result = phot.populate_run(state, observing_config)

        # raw readouts below saturation (SAT_ADU written by the SIM)
        for file_name, header in zip(result.file_names, result.headers):
            analysis_common.check_saturation(
                file_name, float(header["SAT_ADU"]), frac=saturation_frac
            )

        # fluxes and block maps in the field, then the requirement's check
        result = phot.reduce_flats(result)
        result = CHECKS[result.req_id](result)
        phot.plot_run(result)

        # outcome of the run
        results.append(phot.log_test_outcome(result))

    phot.log_summary(results)


if __name__ == "__main__":
    main()
