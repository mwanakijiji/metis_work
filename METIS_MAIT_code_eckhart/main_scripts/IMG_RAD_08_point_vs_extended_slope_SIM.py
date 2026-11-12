"""
Make simulated data for the IMG-RAD-08 point vs. extended source slope test (STUB)

Reqs.:
- Ref. IMG-RAD-08 Point vs. extended slope

1. (requirement ID and text TBD)

STUB: the requirement text(s) and the WCU set-ups (focal-plane masks for the point and
the extended source, pupil-plane masks) are not known yet (null = TBD in
config/config_file_IMG_RAD_08_point_vs_extended_slope_SIM.yaml). This script loads the
scenario config, logs what is missing, and writes nothing; no ScopeSim calls are made.

To implement (following the other SIMs):
- set fp_mask / pp_mask (and filters, DITs) per instrument in the scenario config, and
  the requirement(s) with their limits and 'pass' / 'fail' cases (simulate: true);
- write one simulate_<requirement> function per requirement, injecting the quantity
  the ANALYSIS will measure, and list it in SIMULATORS;
- run them with sim_common.run_scenario (WCU optical train, DIT tuned to stay below
  saturation, darks / illuminated frame, FITS files with the injected truth in the
  header), and add a test in tests/test_sim_common.py.
"""

import datetime
import logging

from modules import sim_common
from modules.helpers import load_config_and_pipe
from modules.paths import project_root
from pipeline_registry import (  # pylint: disable=import-error,wrong-import-order
    CLUSTER_PTEXT,
    COLOR_PTEXT,
    pipeline_stage,
)

TEST_PREFIX = "IMG_RAD_08"


@pipeline_stage(
    name="simulate_rad_08_point_vs_extended",
    cluster=CLUSTER_PTEXT,
    cluster_color=COLOR_PTEXT,
    planned=True,  # stub: drawn dashed in the pipeline flow diagram
    label="SIM point vs extended slope (stub: reqs TBD)",
)
def simulate_point_vs_extended(ctx, req_key, req_cfg):
    """
    Stub for the IMG-RAD-08 requirement(s): writes nothing (requirement texts and
    focal- and pupil-plane masks TBD).

    OUTPUTS:
    - list[str]: written files (empty)
    """
    del ctx  # not used yet
    logging.info(
        "%s (%s): stub, no files written", req_key, req_cfg.get("req_id", req_key)
    )
    return []


# requirement key (in the scenario config) -> function writing its simulated data
SIMULATORS = {
    "RAD-08-TBD": simulate_point_vs_extended,
}


@pipeline_stage(
    name="generate_point_vs_extended_sim",
    depends_on=("simulate_rad_08_point_vs_extended",),
    cluster=CLUSTER_PTEXT,
    cluster_color=COLOR_PTEXT,
    planned=True,  # stub: drawn dashed in the pipeline flow diagram
    label="Generate point-vs-extended sims",
)
def main():
    """
    IMG-RAD-08 point vs. extended slope simulations (STUB): load the scenario config and
    log what is still TBD; nothing is simulated or written.
    """
    stem = project_root()
    out_dir = stem + "data/IMG_RAD_08_point_vs_extended_slope_simmed_data/"

    now = datetime.datetime.now()
    sim_common.start_sim_logging(
        stem, "IMG_RAD_08_pt_ext_logs", "IMG_RAD_08_point_vs_extended_slope", now
    )
    logging.info(
        "IMG-RAD-08 point vs extended slope SIM is a STUB: requirements and masks TBD; "
        "nothing is simulated (output directory once implemented: %s)",
        out_dir,
    )
    scenario = load_config_and_pipe(
        config_file_choice=stem
        + "config/config_file_IMG_RAD_08_point_vs_extended_slope_SIM.yaml",
        print_one_line=False,
    )
    n_tbd = sim_common.log_stub_inputs(scenario)
    logging.info(
        "%s instrument(s) with focal-/pupil-plane masks TBD; 0 files written", n_tbd
    )


if __name__ == "__main__":
    main()
