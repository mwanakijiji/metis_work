"""
Make simulated data for the IMG-OPT-07 pupil imaging test (STUB)

Reqs.:
- Ref. IMG-OPT-07 Pupil imaging

1. METIS-2472 (verified by test at System level, applicable to IMG): METIS shall
provide pupil viewing capability in the IMG-LM and IMG-N cameras. The ELT entrance
pupil shall be imaged onto the IMG-LM and N detectors and the following requirements
shall be satisfied. (1) The maximum ELT M1 diameter (39146 mm) shall be imaged onto
at least 75% and less then the 95% of the useful detector length, (2) the PSF FWHM in
pupil imaging mode shall be less then 1% of the maximum ELT M1 diameter using the
wavelength range corresponding to the filters below, (3) pupil imaging requirements
shall be satisfied for the wavelength ranges: L band: 4.00 - 4.10 um (IB 4.05
filter) and N band: 8.95-9.25 um ("PAH 8.6 ref" filter).

STUB: pupil imaging is not yet possible in ScopeSim (the capability will be built in
later), so the WCU focal-plane and pupil-plane masks for it are not known yet (null =
TBD in config/config_file_IMG_OPT_07_pupil_imaging_SIM.yaml). This script loads the
scenario config, logs what is missing, and writes nothing; no ScopeSim calls are made.

To implement, once ScopeSim can image the pupil (following the other SIMs):
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
    CLUSTER_PUPIL,
    COLOR_PUPIL,
    pipeline_stage,
)

TEST_PREFIX = "IMG_OPT_07"


@pipeline_stage(
    name="simulate_img_07_pupil_imaging",
    cluster=CLUSTER_PUPIL,
    cluster_color=COLOR_PUPIL,
    planned=True,  # stub: drawn dashed in the pipeline flow diagram
    label="SIM pupil images (stub: masks TBD)",
)
def simulate_pupil_imaging(ctx, req_key, req_cfg):
    """
    Stub for the IMG-OPT-07 pupil imaging requirement(s): writes nothing (pupil
    imaging not yet in ScopeSim; focal- and pupil-plane masks TBD).

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
    "PUPIL-TBD": simulate_pupil_imaging,
}


@pipeline_stage(
    name="generate_pupil_imaging_sim",
    depends_on=("simulate_img_07_pupil_imaging",),
    cluster=CLUSTER_PUPIL,
    cluster_color=COLOR_PUPIL,
    planned=True,  # stub: drawn dashed in the pipeline flow diagram
    label="Generate pupil-imaging sims",
)
def main():
    """
    IMG-OPT-07 pupil imaging simulations (STUB): load the scenario config and log what
    is still TBD; nothing is simulated or written.
    """
    stem = project_root()
    out_dir = stem + "data/IMG_OPT_07_pupil_imaging_simmed_data/"

    now = datetime.datetime.now()
    sim_common.start_sim_logging(
        stem, "IMG_OPT_07_pupil_imaging_logs", "IMG_OPT_07_pupil_imaging", now
    )
    logging.info(
        "IMG-OPT-07 pupil imaging SIM is a STUB: pupil imaging is not yet possible in "
        "ScopeSim; nothing is simulated (output directory once implemented: %s)",
        out_dir,
    )
    scenario = load_config_and_pipe(
        config_file_choice=stem
        + "config/config_file_IMG_OPT_07_pupil_imaging_SIM.yaml",
        print_one_line=False,
    )
    n_tbd = sim_common.log_stub_inputs(scenario)
    logging.info(
        "%s instrument(s) with focal-/pupil-plane masks TBD; 0 files written", n_tbd
    )


if __name__ == "__main__":
    main()
