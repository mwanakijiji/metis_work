"""
Make simulated data for the IMG-OPT-05 optomechanical stability test (STUB)

Reqs.:
- Ref. IMG-OPT-05 Optomechanical stability, Table 1-1 (requirements verified by
  IMG-OPT-05)

1. METIS-5422 (verified by test at System level): METIS shall require an on-sky flux
calibration not more often than once per hour per instrument mode, so long as the
instrument is not powered down or disturbed in a manner outside its nominal operations
concept during this period.

2. METIS-5422 (verified by test at System level, applicable to IMG): The position of
the Vortex Phase Mask (VPM) located in CFO-FP2 shall be stable with respect to the IMG
detectors, with the following accuracy:
On timescales between 10 sec and 10 min (goal: up to 1h)
- 0.2 mas RMS on sky for the L- and M-band VPMs with respect to the IMG-LM detector
- 0.5 mas RMS on sky for the N-band VPMs with respect to the IMG-N detector
On timescales below 10 sec:
- 1 mas RMS on sky for the L- and M-band VPMs with respect to the IMG-LM detector
- 2.5 mas RMS on sky for the N-band VPMs with respect to the IMG-N detector

N.b. both requirements carry the ID METIS-5422 in the table (likely an error in the
table; to be checked); their keys in the scenario config are METIS-5422-FLUXCAL and
METIS-5422-VPM.

STUB: the VPM is not yet in ScopeSim (the capability will be built in later), so the
focal-plane mask (the VPM in CFO-FP2) and the pupil-plane mask are not known yet (null =
TBD in config/config_file_IMG_OPT_05_optomechanical_stability_SIM.yaml). This script
loads the scenario config, logs what is missing, and writes nothing; no ScopeSim calls
are made.

To implement, once ScopeSim has the VPM (following the other SIMs):
- set fp_mask / pp_mask (and filters, DITs) per instrument in the scenario config, and
  per requirement a limit with 'pass' / 'fail' cases (simulate: true);
- write the simulate_* functions, injecting the quantity the ANALYSIS will measure
  (e.g. the VPM position RMS on sky per timescale), and list them in SIMULATORS;
- run them with sim_common.run_scenario, and add a test in tests/test_sim_common.py.
"""

import datetime
import logging

from modules import sim_common
from modules.helpers import load_config_and_pipe
from modules.paths import project_root
from pipeline_registry import (  # pylint: disable=import-error,wrong-import-order
    CLUSTER_OPTOMECH,
    COLOR_OPTOMECH,
    pipeline_stage,
)

TEST_PREFIX = "IMG_OPT_05"


def _stub(req_key, req_cfg):
    """Log that a requirement's simulation is a stub; nothing is written."""
    logging.info(
        "%s (%s): stub, no files written", req_key, req_cfg.get("req_id", req_key)
    )
    return []


@pipeline_stage(
    name="simulate_metis_5422_flux_calibration",
    cluster=CLUSTER_OPTOMECH,
    cluster_color=COLOR_OPTOMECH,
    planned=True,  # stub: drawn dashed in the pipeline flow diagram
    label="SIM METIS-5422: flux calibration interval (stub)",
)
def simulate_metis_5422_flux_calibration(ctx, req_key, req_cfg):
    """
    Stub for METIS-5422 (on-sky flux calibration not more often than once per hour per
    instrument mode): writes nothing.

    OUTPUTS:
    - list[str]: written files (empty)
    """
    del ctx  # not used yet
    return _stub(req_key, req_cfg)


@pipeline_stage(
    name="simulate_metis_5422_vpm_stability",
    cluster=CLUSTER_OPTOMECH,
    cluster_color=COLOR_OPTOMECH,
    planned=True,  # stub: drawn dashed in the pipeline flow diagram
    label="SIM METIS-5422: VPM position stability (stub: no VPM in ScopeSim)",
)
def simulate_metis_5422_vpm_stability(ctx, req_key, req_cfg):
    """
    Stub for METIS-5422 (VPM position in CFO-FP2 stable w.r.t. the IMG detectors):
    writes nothing (the VPM is not yet in ScopeSim).

    OUTPUTS:
    - list[str]: written files (empty)
    """
    del ctx  # not used yet
    return _stub(req_key, req_cfg)


# requirement key (in the scenario config) -> function writing its simulated data
SIMULATORS = {
    "METIS-5422-FLUXCAL": simulate_metis_5422_flux_calibration,
    "METIS-5422-VPM": simulate_metis_5422_vpm_stability,
}


@pipeline_stage(
    name="generate_optomechanical_stability_sim",
    depends_on=(
        "simulate_metis_5422_flux_calibration",
        "simulate_metis_5422_vpm_stability",
    ),
    cluster=CLUSTER_OPTOMECH,
    cluster_color=COLOR_OPTOMECH,
    planned=True,  # stub: drawn dashed in the pipeline flow diagram
    label="Generate optomechanical-stability sims",
)
def main():
    """
    IMG-OPT-05 optomechanical stability simulations (STUB): load the scenario config
    and log what is still TBD; nothing is simulated or written.
    """
    stem = project_root()
    out_dir = stem + "data/IMG_OPT_05_optomechanical_stability_simmed_data/"

    now = datetime.datetime.now()
    sim_common.start_sim_logging(
        stem,
        "IMG_OPT_05_optomechanical_stability_logs",
        "IMG_OPT_05_optomechanical_stability",
        now,
    )
    logging.info(
        "IMG-OPT-05 optomechanical stability SIM is a STUB: the VPM is not yet in "
        "ScopeSim; nothing is simulated (output directory once implemented: %s)",
        out_dir,
    )
    scenario = load_config_and_pipe(
        config_file_choice=stem
        + "config/config_file_IMG_OPT_05_optomechanical_stability_SIM.yaml",
        print_one_line=False,
    )
    n_tbd = sim_common.log_stub_inputs(scenario)
    logging.info(
        "%s instrument(s) with focal-/pupil-plane masks TBD; 0 files written", n_tbd
    )


if __name__ == "__main__":
    main()
