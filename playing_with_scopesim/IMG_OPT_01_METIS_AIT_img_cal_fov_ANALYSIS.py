# Analysis of simulated images written out by IMG_OPT_01_METIS_AIT_img_cal_fov_SIM.py

import datetime
import logging

from modules.helpers import load_config_and_pipe, setup_logging
from modules.backbone_img_01 import fov_calc


# Reqs.:
# - Ref. Overleaf doc IMG_OPT_01_Test_Description_Field_of_View_and_Ghost_Measurement.pdf
#
# 1. METIS-1095: The FoV shall be 10.0 +1.0/-0.0 arcsec for the LM-arm and
#    13.47 +0.50/-0.99 arcsec for the N-arm of the IMG.


def main():

    stem = '/podman-share/metis_work/playing_with_scopesim/'
    analysis_states_config_file = stem + 'config/config_file_IMG_01_METIS_AIT_img_cal_fov_ANALYSIS.yaml'

    now = datetime.datetime.now()
    log_dir = stem + 'IMG_OPT_01_METIS_AIT_img_cal_fov_SIM_logs/'
    log_file_name = (
        log_dir
        + 'log_IMG_OPT_01_METIS_AIT_img_cal_fov_ANALYSIS_'
        + now.strftime('%Y-%m-%d_%H-%M-%S')
        + '.txt'
    )
    setup_logging(log_dir=log_dir, log_file_name=log_file_name, now=now)

    analysis_states_config = load_config_and_pipe(
        config_file_choice=analysis_states_config_file, print_one_line=False
    )
    defaults = analysis_states_config.get("defaults", {})
    runs = analysis_states_config.get("runs", [])

    analysis_states = []
    for entry in runs:
        merged = {**defaults, **entry}
        results_write_dir = merged.get("results_write_dir", "figs_dump")
        if stem and not results_write_dir.startswith("/"):
            merged["results_write_dir"] = stem + results_write_dir
        analysis_states.append(merged)

    logging.info(f"Number of analysis states (filters): {len(analysis_states)}")

    for state in analysis_states:
        fov_calc(run_state=state, stem=stem)


if __name__ == "__main__":
    main()
