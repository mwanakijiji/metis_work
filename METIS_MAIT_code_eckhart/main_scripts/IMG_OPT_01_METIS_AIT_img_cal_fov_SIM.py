"""
Make simulated data for the IMG-OPT-01 FOV measurement

Reqs.:
- Ref. Overleaf doc IMG_OPT_01_Test_Description_Field_of_View_and_Ghost_Measurement.pdf

1. METIS-1095: The FoV shall be 10.0 +1.0/-0.0 arcsec for the LM-arm and 13.47
+0.50/-0.99 arcsec for the N-arm of the IMG.

Structure: one function per requirement, driven by the scenario in
config/config_file_IMG_01_METIS_AIT_img_cal_fov_SIM.yaml and run by
modules.sim_common.run_scenario. Each case is four dithers of the WCU pinhole, named
IMG_OPT_01_<requirement>_<instrument>_<case>_dither<N>.fits, with the requirement and
the injected truth in the primary header (REQ_ID, CASE, EXPECTED, REQ_QTY, REQ_LIM,
PASS_IF, INJ_VAL, REQ_PLH, SIMVER; see modules/sim_common.py) and the dither position
(DITHX, DITHY). There is a case that clearly passes, and one that clearly fails on
either side of the FoV range.

The FoV is injected through the dither offsets (see simulate_metis_1095). Each dither is
a ScopeSim readout; BACKGROUND is an independent dark.
"""

import datetime
import logging

import numpy as np

from modules import sim_common
from modules.config_common import load_observing_config
from modules.helpers import load_config_and_pipe
from pipeline_registry import (  # pylint: disable=import-error,wrong-import-order
    CLUSTER_FOV,
    COLOR_FOV,
    pipeline_stage,
)

TEST_PREFIX = "IMG_OPT_01"

# corners of the dither square, in units of dither_offset_arcsec; consecutive dithers
# are adjacent corners, as the ANALYSIS assumes
DITHER_CORNERS = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])


def dither_positions(dither_offset_arcsec, fov_true, fov_injected):
    """
    Dither positions (x, y) in arcsec, scaled by fov_true / fov_injected so that an
    analysis assuming the nominal offsets measures fov_injected.
    """
    return DITHER_CORNERS * dither_offset_arcsec * (fov_true / fov_injected)


@pipeline_stage(
    name="sim_img_01_metis_1095",
    cluster=CLUSTER_FOV,
    cluster_color=COLOR_FOV,
    label="SIM METIS-1095: FoV",
)
def simulate_metis_1095(ctx, req_key, req_cfg):
    """
    Simulated data for METIS-1095: the FoV shall be 10.0 +1.0/-0.0 arcsec (LM) and
    13.47 +0.50/-0.99 arcsec (N).

    The ANALYSIS derives the pixel scale from the pinhole positions in four dithers it
    takes to be 2 * dither_offset_arcsec apart, and the FoV as that pixel scale times
    the number of illuminated pixels. Per case, the dithers are scaled by FoV_true /
    FoV_injected (FoV_true = pixel scale * illuminated pixels), so the FoV it measures
    is the injected one.

    OUTPUTS:
    - list[str]: written files
    """
    expected = sim_common.check_cases(req_key, req_cfg)
    _, dark_2 = ctx.darks()
    fov_true = (
        float(ctx.observing["pixel_scales"][ctx.det_key])
        / 1000.0
        * float(req_cfg["illuminated_pixels"])
    )
    offset = float(req_cfg["dither_offset_arcsec"])
    logging.info(
        "%s: FoV_true %.3f arcsec; nominal dither offset %.2f arcsec",
        req_key,
        fov_true,
        offset,
    )

    written = []
    for case, case_cfg in req_cfg["cases"].items():
        inject = case_cfg["inject"]
        cards = sim_common.requirement_cards(
            req_key, req_cfg, case, inject, expected[case], ctx.simver
        )
        positions = dither_positions(offset, fov_true, float(inject["fov"]))
        for i, pos in enumerate(positions):
            sim_common.set_fpmask_shift(ctx.metis, pos)
            raw, header = sim_common.readout_frame(ctx.metis, 1.0, ctx.instrument)
            written.append(
                sim_common.write_case_file(
                    ctx.out_dir,
                    ctx.test_prefix,
                    req_key,
                    ctx.instrument,
                    case,
                    f"_dither{i}",
                    raw,
                    dark_2,
                    header,
                    cards
                    + [
                        ("DITHER", i, "Dither index"),
                        ("DITHX", float(pos[0]), "WCU FP mask shift x (arcsec)"),
                        ("DITHY", float(pos[1]), "WCU FP mask shift y (arcsec)"),
                        ("DITHNOM", offset, "Nominal dither offset (arcsec)"),
                        ("FOV_TRUE", fov_true, "Pixel scale * illum. pixels (arcsec)"),
                    ],
                )
            )
    sim_common.set_fpmask_shift(ctx.metis, (0.0, 0.0))
    return written


# requirement key (in the scenario config) -> function writing its simulated data
SIMULATORS = {
    "METIS-1095": simulate_metis_1095,
}


@pipeline_stage(
    name="generate_fov_sim",
    depends_on=("sim_img_01_metis_1095",),
    cluster=CLUSTER_FOV,
    cluster_color=COLOR_FOV,
    label="Generate FOV sims",
)
def main():
    """
    Generate IMG-OPT-01 FoV simulations, one set per requirement, as set in the
    scenario config.
    """
    stem = "/podman-share/metis_work/METIS_MAIT_code_eckhart/"
    out_dir = stem + "data/IMG_OPT_01_fov_simmed_data/"

    now = datetime.datetime.now()
    sim_common.start_sim_logging(
        stem, "IMG_01_fov_sim_logs", "IMG_OPT_01_METIS_AIT_img_cal_fov_SIM", now
    )
    logging.info("Simmed file output directory: %s", out_dir)
    sim_common.setup_scopesim()

    observing = load_observing_config(stem)
    scenario = load_config_and_pipe(
        config_file_choice=stem
        + "config/config_file_IMG_01_METIS_AIT_img_cal_fov_SIM.yaml",
        print_one_line=False,
    )
    sim_common.run_scenario(scenario, SIMULATORS, observing, out_dir, TEST_PREFIX)


if __name__ == "__main__":
    main()
