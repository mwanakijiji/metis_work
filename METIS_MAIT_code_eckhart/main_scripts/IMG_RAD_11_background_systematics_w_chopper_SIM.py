"""
Make simulated data for the IMG-RAD-11 background systematics with chopper test (STUB)

Reqs.:
- Ref. IMG-RAD-11 Background systematics with chopper

1. (requirement ID and text TBD)

STUB: the requirement text(s) and the WCU set-ups (focal- and pupil-plane masks,
chopper positions) are not known yet (null = TBD in
config/config_file_IMG_RAD_11_background_systematics_SIM.yaml). This script loads the
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
    CLUSTER_CHOP,
    COLOR_CHOP,
    pipeline_stage,
)

TEST_PREFIX = "IMG_RAD_11"


@pipeline_stage(
    name="simulate_rad_11_chopper_background",
    cluster=CLUSTER_CHOP,
    cluster_color=COLOR_CHOP,
    planned=True,  # stub: drawn dashed in the pipeline flow diagram
    label="SIM chopped background (stub: reqs TBD)",
)
def simulate_chopper_background(ctx, req_key, req_cfg):
    """
    Stub for the IMG-RAD-11 requirement(s): writes nothing (requirement texts, focal-
    and pupil-plane masks and chopper set-up TBD).

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
    "RAD-11-TBD": simulate_chopper_background,
}


@pipeline_stage(
    name="generate_background_systematics_sim",
    depends_on=("simulate_rad_11_chopper_background",),
    cluster=CLUSTER_CHOP,
    cluster_color=COLOR_CHOP,
    planned=True,  # stub: drawn dashed in the pipeline flow diagram
    label="Generate chopper background sims",
)
def main():
    """
    IMG-RAD-11 background systematics with chopper simulations (STUB): load the
    scenario config and log what is still TBD; nothing is simulated or written.
    """
    stem = project_root()
    out_dir = stem + "data/IMG_RAD_11_background_systematics_simmed_data/"

    now = datetime.datetime.now()
    sim_common.start_sim_logging(
        stem, "IMG_RAD_11_bkgd_sys_logs", "IMG_RAD_11_background_systematics", now
    )
    logging.info(
        "IMG-RAD-11 background systematics SIM is a STUB: requirements, masks and "
        "chopper set-up TBD; nothing is simulated (output directory once implemented: "
        "%s)",
        out_dir,
    )
    scenario = load_config_and_pipe(
        config_file_choice=stem
        + "config/config_file_IMG_RAD_11_background_systematics_SIM.yaml",
        print_one_line=False,
    )
    n_tbd = sim_common.log_stub_inputs(scenario)
    logging.info(
        "%s instrument(s) with focal-/pupil-plane masks TBD; 0 files written", n_tbd
    )


if __name__ == "__main__":
    main()
