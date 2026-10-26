"""
Make simulated data for the IMG-OPT-06 out-of-field stray light test

Reqs. (Table 1-1 of the IMG-OPT-06 test case description):

1. METIS-3114 (verified by test at System level): METIS shall be designed and built to
be insensitive to outside illumination so that all calibration procedures needed by the
instrument can be performed without being affected by routine observatory operations.
-> not simulated; verified at observatory level.

2. METIS-5838 (verified by analysis at System level, applicable to IMG): Irradiance
from sources outside the input beam envelope (defined by the 'FOV' and the max ELT
pupil area), propagating between the specified 'input plane' and 'output plane'
through scattering shall be attenuated by a factor indicated below for the indicated
'wavelength ranges'.

Subsystem | Input plane | Output plane    | FOV            | Band | Attenuation
IMG       | CFO FP2     | IMG-N detector  | 14"x14" square | N    | 1e4
IMG       | CFO FP2     | IMG-LM detector | 11"x11" square | L, M | 1e4

3. METIS-9522: the flux in optical artefacts and ghosts shall be less than the 3-sigma
thermal background noise (...). -> not simulated here; simulated for IMG-OPT-04.

The WCU pinhole in ScopeSim is centred, so a truly out-of-field source cannot be
simulated. Instead, as for IMG-OPT-04, a single PSF is simulated in the middle of the
detector (its peak is the unattenuated irradiance), and synthetic scattered light from
a virtual out-of-field source is added, scaled so that PSF peak / max of the scattered
light is the injected attenuation.

Structure: one function per requirement, driven by the scenario in
config/config_file_IMG_OPT_06_outoffield_stray_light_SIM.yaml and run by
modules.sim_common.run_scenario. Each writes a case that clearly passes and one that
clearly fails, named IMG_OPT_06_<requirement>_<instrument>_<case>.fits, with the
requirement and the injected truth in the primary header (REQ_ID, CASE, EXPECTED,
REQ_QTY, REQ_LIM, PASS_IF, INJ_VAL, REQ_PLH, SIMVER; see modules/sim_common.py), plus
the cards the ANALYSIS reads (ATT_REQ, ATT_INJ, REF_PEAK, ...).

The exposures are deep (many NDIT at an unsaturated DIT), as in the test description,
so that an attenuation of 1e4 is measurable above the noise. The scattered light is
added to an illuminated ScopeSim frame; BACKGROUND is an independent dark (dark_2), and
BACKGROUND_2 is another one (dark_1), so the analysis can measure the noise from a null
frame (BACKGROUND_2 - BACKGROUND).
"""

import datetime
import logging

import numpy as np
from astropy.io import fits

import modules.backbone_img_opt_06_out_of_field as backbone
from modules import sim_common
from modules.config_common import load_observing_config
from modules.helpers import load_config_and_pipe
from pipeline_registry import (  # pylint: disable=import-error,wrong-import-order
    CLUSTER_OOF,
    COLOR_OOF,
    pipeline_stage,
)

TEST_PREFIX = "IMG_OPT_06"


def scaled_out_of_field_scatter(shape, ref_peak, attenuation, direction):
    """
    Scattered light from a virtual source beyond the 'direction' edge of the detector
    (shape from backbone.make_out_of_field_scatter), scaled so that
    ref_peak / max(scatter) = attenuation.
    """
    scatter = backbone.make_out_of_field_scatter(
        shape=shape, peak=1.0, attenuation=1.0, direction=direction
    )
    return scatter * (ref_peak / attenuation) / np.max(scatter)


@pipeline_stage(
    name="sim_img_06_metis_5838",
    cluster=CLUSTER_OOF,
    cluster_color=COLOR_OOF,
    label="SIM METIS-5838: out-of-field attenuation",
)
def simulate_metis_5838(ctx, req_key, req_cfg):
    """
    Simulated data for METIS-5838: out-of-field irradiance attenuated by >= 1e4.

    Per case, the illuminated ScopeSim frame (one PSF in the middle) plus scattered
    light scaled so that PSF peak / max of the scattered light = injected attenuation
    (the PSF peak from the signal, illuminated frame - dark_1).

    OUTPUTS:
    - list[str]: written files
    """
    expected = sim_common.check_cases(req_key, req_cfg)
    dark_1, _ = ctx.darks()
    flat, _ = ctx.lit_frame()
    ref_peak = float(np.max(flat - dark_1))
    direction = req_cfg.get("oof_direction", "+x")
    logging.info("%s: PSF peak (signal) %.4g counts", req_key, ref_peak)

    written = []
    for case, case_cfg in req_cfg["cases"].items():
        inject = case_cfg["inject"]
        attenuation = float(inject["attenuation"])
        scatter = scaled_out_of_field_scatter(
            flat.shape, ref_peak, attenuation, direction
        )
        logging.info(
            "%s case %s: injected attenuation %.1e; scattered irradiance max %.4g, "
            "mean %.4g counts",
            req_key,
            case,
            attenuation,
            float(np.max(scatter)),
            float(np.mean(scatter)),
        )
        cards = sim_common.requirement_cards(
            req_key, req_cfg, case, inject, expected[case], ctx.simver
        ) + [
            (
                "ATT_REQ",
                float(req_cfg["limit"]["attenuation"]["value"]),
                "Required out-of-field attenuation",
            ),
            ("ATT_INJ", attenuation, "Injected attenuation (peak / max scatter)"),
            (
                "FOV_REQ",
                float(ctx.observing["fov_req_arcsec"][ctx.det_key]),
                "Required square FOV side (arcsec)",
            ),
            ("OOF_DIR", direction, "Side of virtual out-of-field source"),
            ("REF_PEAK", ref_peak, "Peak of bckgd-subted PSF"),
        ]
        written.append(
            ctx.write_case(
                req_key,
                case,
                "",
                flat + scatter,
                cards,
                extra_hdus=[
                    fits.ImageHDU(data=dark_1.astype(np.float32), name="BACKGROUND_2")
                ],
            )
        )
    return written


# requirement key (in the scenario config) -> function writing its simulated data
SIMULATORS = {
    "METIS-5838": simulate_metis_5838,
}


@pipeline_stage(
    name="generate_out_of_field_sim",
    depends_on=("sim_img_06_metis_5838",),
    cluster=CLUSTER_OOF,
    cluster_color=COLOR_OOF,
    label="Generate out-of-field sims",
)
def main():
    """
    Generate IMG-OPT-06 out-of-field simulations, one set per requirement, as set in the
    scenario config.
    """
    stem = "/podman-share/metis_work/METIS_MAIT_code_eckhart/"
    # write simulated data here, where the ANALYSIS script reads it
    out_dir = stem + "data/IMG_OPT_06_out_of_field_simmed_data/"

    now = datetime.datetime.now()
    sim_common.start_sim_logging(
        stem, "IMG_06_simmed_out_of_field_logs", "IMG_06_simmed_out_of_field", now
    )
    logging.info("Simmed file output directory: %s", out_dir)
    sim_common.setup_scopesim()

    # IMG-OPT-06 observing config (fov_req_arcsec, ...) on top of IMG-OPT-04's
    observing = load_observing_config(
        stem,
        "config/config_file_IMG_04_stray_light_observing.yaml",
        "config/config_file_IMG_06_out_of_field_observing.yaml",
    )
    scenario = load_config_and_pipe(
        config_file_choice=stem
        + "config/config_file_IMG_OPT_06_outoffield_stray_light_SIM.yaml",
        print_one_line=False,
    )
    sim_common.run_scenario(scenario, SIMULATORS, observing, out_dir, TEST_PREFIX)


if __name__ == "__main__":
    main()
