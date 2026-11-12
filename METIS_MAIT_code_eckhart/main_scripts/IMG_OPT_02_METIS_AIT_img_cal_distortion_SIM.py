"""
Make simulated data for the IMG-OPT-02 image distorition test

Reqs.:
- Ref. Overleaf doc IMG_OPT_02_Test_Geometric_Distortion

1. METIS-1097: The Imager shall provide a pixel scale of 5.47 +0.26/-0.26 mas/pix for
   the LM-band and 6.79+0.25/-0.50 mas/pix for the N-band.
2. METIS-3502: After calibration, the distortions introduced by METIS shall be removed
   to better than 0.5 mas (ca. 1/10 px for the L band imager) over the full field of
   view.
3. METIS-8222: The center of the H2RG chip within IMG-LM-DET shall be offset from the
   METIS optical axis by 175 mas ± 25 mas on-sky PtV in the "across H2RG stripe"
   direction (i.e., perpendicularly to the orientation of the 32 stripes in the H2RG
   detector).
4. METIS-9920: Image scale and distortion of METIS shall be constant (for each optical
   configuration, even after change of observing modes) to an accuracy of 10-3 (goal:
   10-4) at L/M-band and 2×10-3 (goal: 2×10-4) at N-band with respect to the full field
   of view.

!!! TBD MORE DISTORTION INFO FROM DD NEEDED FIRST (warned in the log)

Structure: driven by the scenario in config/config_file_IMG_OPT_02_distortion_SIM.yaml
(one instrument per LM filter, all with the pinhole grid grid_lm and PPS-CFO2) and run
by modules.sim_common.run_scenario. No requirement cases are simulated yet (the
requirements are in the scenario with simulate: false until the distortion is better
known); the scenario entry GRID-FRAMES writes one unmodified pinhole-grid frame per
instrument, IMG_OPT_02_GRID-FRAMES_<instrument>_reference.fits, read by the ANALYSIS
(config/config_file_IMG_02_strehl_runs.yaml).

Integration: the DITs in the scenario are starting values; run_scenario shortens the
DIT of each set-up if needed (sim_common.limit_dit_to_saturation) so that the raw
readout peaks at <= sim_common.MAX_RAW_FRAC of the detector saturation level
(saturation_adu in the common observing config). The tuned and configured DITs and the
raw peak / saturation are in the header (DIT, DIT_CFG, PEAKFRAC, SAT_ADU), and a DIT
summary of all set-ups is logged at the end.
"""

import datetime
import logging

from modules import sim_common
from modules.config_common import load_observing_config
from modules.helpers import load_config_and_pipe
from modules.paths import project_root
from modules.tbd_warnings import log_tbd_distortion
from pipeline_registry import (  # pylint: disable=import-error,wrong-import-order
    CLUSTER_DISTORTION,
    COLOR_DISTORTION,
    pipeline_stage,
)

TEST_PREFIX = "IMG_OPT_02"


@pipeline_stage(
    name="sim_img_02_grid_frames",
    cluster=CLUSTER_DISTORTION,
    cluster_color=COLOR_DISTORTION,
    planned=True,  # stub until the distortion map is better known
    label="SIM pinhole-grid frames (no injection)",
)
def simulate_grid_frames(ctx, req_key, req_cfg):
    """
    Unmodified pinhole-grid frame of one WCU set-up (no distortion injected yet), as
    IMG_OPT_02_<req_key>_<instrument>_reference.fits (sim_common.write_reference_frame).

    OUTPUTS:
    - list[str]: written files
    """
    return sim_common.write_reference_frame(ctx, req_key, req_cfg)


# requirement key (in the scenario config) -> function writing its simulated data
SIMULATORS = {
    "GRID-FRAMES": simulate_grid_frames,
}


@pipeline_stage(
    name="generate_distortion_sim",
    depends_on=("sim_img_02_grid_frames",),
    cluster=CLUSTER_DISTORTION,
    cluster_color=COLOR_DISTORTION,
    planned=True,  # stub until the distortion map is better known
    label="Generate distortion sims",
)
def main():
    """
    Generate IMG-OPT-02 distortion simulations, as set in the scenario config.
    """
    stem = project_root()
    # write simulated data here, where the ANALYSIS reads it
    # (config/config_file_IMG_02_strehl_runs.yaml)
    out_dir = stem + "data/IMG_OPT_02_distortion_simmed_data/"

    now = datetime.datetime.now()
    sim_common.start_sim_logging(
        stem, "IMG_02_logs", "IMG_02_simulation_psf_image_quality", now
    )
    log_tbd_distortion("IMG-OPT-02 SIM")
    logging.info("Simmed file output directory: %s", out_dir)
    # uncomment to (re-)download the IRDB packages (overwrites them in the IRDB)
    # sim.download_packages(
    #     ["METIS", "ELT", "Armazones"], save_dir=sim_common.IRDB_DIR
    # )
    sim_common.setup_scopesim()

    observing = load_observing_config(stem)
    scenario = load_config_and_pipe(
        config_file_choice=stem + "config/config_file_IMG_OPT_02_distortion_SIM.yaml",
        print_one_line=False,
    )
    sim_common.run_scenario(scenario, SIMULATORS, observing, out_dir, TEST_PREFIX)

    log_tbd_distortion("IMG-OPT-02 SIM")


if __name__ == "__main__":
    main()
