"""
Analysis of the IMG-OPT-05 optomechanical stability pupil-image sequences written by
IMG_OPT_05_optomechanical_stability_SIM.py.

Reqs.:
- Ref. IMG-OPT-05 Optomechanical stability, Table 1-1 (requirements verified by
  IMG-OPT-05)

2. METIS-5422 (verified by test at System level, applicable to IMG): The position of
the Vortex Phase Mask (VPM) located in CFO-FP2 shall be stable with respect to the IMG
detectors, with the following accuracy:
On timescales between 10 sec and 10 min (goal: up to 1h)
- 0.2 mas RMS on sky for the L- and M-band VPMs with respect to the IMG-LM detector
- 0.5 mas RMS on sky for the N-band VPMs with respect to the IMG-N detector
On timescales below 10 sec:
- 1 mas RMS on sky for the L- and M-band VPMs with respect to the IMG-LM detector
- 2.5 mas RMS on sky for the N-band VPMs with respect to the IMG-N detector

Tested with pupil imaging: one run per sequence of frames (timescale, instrument,
case); the pupil-image centre is measured in each frame, and the 2D RMS of the centres
about their mean, in mas on sky (x the detector's pixel scale), is compared with the
limit of the run's detector (backbone_img_opt_05_optomech_stability).

Not analysed: METIS-1230 (on-sky flux-calibration interval; SIM is a stub).

Pass/fail of each run is decided in backbone_img_opt_05_optomech_stability
.metis_5422_check, and logged per run with its consistency with the injected truth in
.log_test_outcome; log_summary tabulates all runs.
"""

import datetime
import logging

import modules.backbone_img_opt_05_optomech_stability as stab
from modules import analysis_common
from modules.config_common import load_observing_config
from modules.helpers import setup_logging
from modules.paths import project_root

CHECKS = {
    "METIS-5422": stab.metis_5422_check,
}


def main():
    """Run the IMG-OPT-05 analysis for each data state (one frame sequence each)."""

    stem = project_root()
    now = datetime.datetime.now()

    # initialize logging
    log_dir = stem + "logs/IMG_OPT_05_optomechanical_stability_logs/"
    log_file_name = (
        log_dir
        + "log_IMG_OPT_05_analysis_optomechanical_stability_"
        + now.strftime("%Y-%m-%d_%H-%M-%S")
        + ".txt"
    )
    setup_logging(log_dir=log_dir, log_file_name=log_file_name, now=now)

    # observing parameters: common, then IMG-OPT-05's (analysis parameters, limits)
    observing_config = load_observing_config(
        stem, "config/config_file_IMG_OPT_05_optomechanical_stability_observing.yaml"
    )
    saturation_frac = float(observing_config["img_opt_05"]["saturation_frac"])

    # data states: one per frame sequence (defaults + per-run overrides)
    data_states = analysis_common.load_data_states(
        stem,
        "config/config_file_IMG_OPT_05_optomechanical_stability_data_states.yaml",
        path_keys=("file_name_pattern", "results_write_dir"),
    )

    results = []
    for state in data_states:
        logging.info("==================================================")
        result = stab.populate_run(state, observing_config)

        # raw readouts below saturation (SAT_ADU written by the SIM)
        for file_name, header in zip(result.file_names, result.headers):
            analysis_common.check_saturation(
                file_name, float(header["SAT_ADU"]), frac=saturation_frac
            )

        # pupil centre per frame, then the requirement's check
        result = stab.measure_positions(result)
        result = CHECKS[result.req_id](result)
        stab.plot_run(result)

        # outcome of the run
        results.append(stab.log_test_outcome(result))

    stab.log_summary(results)


if __name__ == "__main__":
    main()
