"""
Make simulated data for the IMG-RAD-09 nominal background behaviour test

Reqs.:
- Ref. IMG-RAD-09 Nominal background behaviour

1. (requirement ID TBD) Verify sensititity requirement on point source in a 10000
second exposure.

What is simulated: per instrument (config/config_file_IMG_RAD_09_nominal_background_
SIM.yaml; L' on the LM detector and N2 on the N detector), a WCU flat field (FP mask
'open', black-body aperture fully open, no source) standing in for a uniform
background, read out as two independent exposures, each of total_exptime_s (10000 s):
the raw frame A and the background frame B, as for a nod pair, so BCKGD_SUBTED = A - B
is the background-limited noise. The DIT is the configured one, shortened if needed by
sim_common.run_scenario so the raw readout stays below saturation, and NDIT =
total_exptime_s / (DIT x n_exposures); ScopeSim integrates NDIT x DIT in one readout
(shot and read noise of the whole integration), and the frames are per DIT (ScopeSim
averages over NDIT).

Requirement-driven, as the other SIMs (run by sim_common.run_scenario): the quantity is
the 1-sigma point-source noise of BCKGD_SUBTED in absolute flux units
(backbone_img_rad_09_nom: robust spread of the sums in apertures of radius
aperture_radius_lambda_over_d x lambda/D, divided by the DIT and times the
counts-to-flux calibration, a PLACEHOLDER in
config/config_file_IMG_RAD_09_nominal_background_observing.yaml); the limit is a
PLACEHOLDER too. Per case, the noise of A - B is scaled to the injected level:
raw = B + (injected / natural noise) x (A - B), written with B as BACKGROUND, so the
ANALYSIS measures the injected noise. Each case file
IMG_RAD_09_<requirement>_<instrument>_<case>.fits has the requirement and the injected
truth in the primary header (REQ_ID, CASE, EXPECTED, REQ_LIM, INJ_VAL, REQ_PLH, ...),
and the natural (unscaled) noise and the scale factor (NAT_NOIS, NOISFAC).
"""

import datetime
import logging
import math

import numpy as np

from modules import analysis_common, sim_common
from modules import backbone_img_rad_09_nom as nom
from modules.config_common import load_observing_config
from modules.helpers import load_config_and_pipe
from modules.paths import project_root
from pipeline_registry import (  # pylint: disable=import-error,wrong-import-order
    CLUSTER_NOMBKG,
    COLOR_NOMBKG,
    pipeline_stage,
)

TEST_PREFIX = "IMG_RAD_09"
OBSERVING_CONFIG = "config/config_file_IMG_RAD_09_nominal_background_observing.yaml"


def take_long_exposure(metis, instrument, n_exposures):
    """
    One exposure of n_exposures readouts of NDIT x DIT (instrument's), averaged (per
    DIT), with the WCU black-body aperture fully open.

    OUTPUTS:
    - np.ndarray: frame (ADU per DIT)
    - astropy header: primary header of the first readout
    """
    frames = []
    header = None
    for _ in range(int(n_exposures)):
        frame, hdr = sim_common.readout_frame(metis, 1.0, instrument)
        frames.append(frame)
        header = header if header is not None else hdr
    return np.mean(frames, axis=0), header


@pipeline_stage(
    name="simulate_rad_09_sensitivity",
    cluster=CLUSTER_NOMBKG,
    cluster_color=COLOR_NOMBKG,
    label="SIM 10000 s WCU flat: point-source noise",
)
def simulate_sensitivity(ctx, req_key, req_cfg):  # pylint: disable=too-many-locals
    """
    Simulated data for the point-source sensitivity in a 10000 s exposure: two
    independent long WCU flat exposures A and B (total_exptime_s each, at the DIT tuned
    by run_scenario), and the 1-sigma point-source noise of A - B (aperture sums, in
    flux units) scaled per case to the injected level (see the module docstring).

    OUTPUTS:
    - list[str]: written files
    """
    expected = sim_common.check_cases(req_key, req_cfg)
    cfg = ctx.observing["img_rad_09"]
    calib = float(cfg["flux_per_adu_per_s"][ctx.det_key])

    dit = float(ctx.instrument["dit"])
    n_exposures = int(req_cfg["n_exposures"])
    ndit = int(math.ceil(float(req_cfg["total_exptime_s"]) / (dit * n_exposures)))
    inst = {**ctx.instrument, "ndit": ndit}
    logging.info(
        "%s: DIT %.4g s, NDIT %s, %s exposure(s) -> %.6g s per frame",
        req_key,
        dit,
        ndit,
        n_exposures,
        dit * ndit * n_exposures,
    )
    raw_a, header = take_long_exposure(ctx.metis, inst, n_exposures)
    background, _ = take_long_exposure(ctx.metis, inst, n_exposures)

    mask = analysis_common.field_mask(
        raw_a, cfg["mask_frac_of_median"], cfg["mask_erosion_px"]
    )
    radius = nom.aperture_radius_px(
        nom.filter_wavelength_m(
            ctx.observing, ctx.instrument["obs_filter"], ctx.det_key
        ),
        float(cfg["telescope_diameter_m"]),
        float(ctx.observing["pixel_scales"][ctx.det_key]),
        float(cfg["aperture_radius_lambda_over_d"]),
    )
    spacing = radius * float(cfg["aperture_spacing_radii"])
    diff = raw_a - background
    noise_adu, n_apertures = nom.aperture_noise(diff, mask, radius, spacing)
    natural = nom.noise_to_flux(noise_adu, dit, calib)
    limit = float(req_cfg["limit"]["noise_flux"]["value"])
    logging.info(
        "%s: aperture radius %.2f px (%s lambda/D), %s apertures; natural noise %.6g "
        "ADU/DIT -> %.6g %s (calibration %s per ADU/s%s); limit %s -> natural verdict "
        "%s",
        req_key,
        radius,
        cfg["aperture_radius_lambda_over_d"],
        n_apertures,
        noise_adu,
        natural,
        cfg["flux_unit"],
        calib,
        ", PLACEHOLDER" if cfg.get("flux_calibration_is_placeholder") else "",
        limit,
        "PASS" if natural <= limit else "FAIL",
    )

    written = []
    for case, case_cfg in req_cfg["cases"].items():
        inject = case_cfg["inject"]
        factor = float(inject["noise_flux"]) / natural
        if not 0.5 <= factor <= 2.0:
            logging.warning(
                "! ------- %s case %s: noise scaled by %.3g (far from the natural "
                "noise; placeholder limit/injection values to be revised?)",
                req_key,
                case,
                factor,
            )
        raw = background + factor * diff
        written.append(
            sim_common.write_case_file(
                ctx.out_dir,
                ctx.test_prefix,
                req_key,
                inst,
                case,
                "",
                raw,
                background,
                header,
                sim_common.requirement_cards(
                    req_key, req_cfg, case, inject, expected[case], ctx.simver
                )
                + [
                    ("TARGET", "WCU BB flat", "Uniform background, no source"),
                    ("EXPTIME", dit * ndit * n_exposures, "Per frame (s)"),
                    ("NEXP", n_exposures, "Readouts averaged per frame"),
                    ("AP_RAD", radius, "Aperture radius (px)"),
                    ("N_APER", n_apertures, "Apertures in the noise estimate"),
                    ("FLUXCAL", calib, "Flux per ADU/s in the aperture"),
                    (
                        "FLUXPLH",
                        bool(cfg.get("flux_calibration_is_placeholder", False)),
                        "Flux calibration is a placeholder",
                    ),
                    ("NAT_NOIS", natural, "Natural 1-sigma point-source noise"),
                    ("NOISFAC", factor, "Injected / natural noise"),
                ],
                saturation_adu=ctx.saturation_adu,
            )
        )
        logging.info(
            "%s case %s: noise %.6g (x %.4g of natural) -> %s",
            req_key,
            case,
            float(inject["noise_flux"]),
            factor,
            expected[case],
        )
    return written


# requirement key (in the scenario config) -> function writing its simulated data
SIMULATORS = {
    "SENS-10000S": simulate_sensitivity,
}


@pipeline_stage(
    name="generate_nominal_background_sim",
    depends_on=("simulate_rad_09_sensitivity",),
    cluster=CLUSTER_NOMBKG,
    cluster_color=COLOR_NOMBKG,
    label="Generate nominal background sims",
)
def main():
    """
    Generate IMG-RAD-09 nominal background simulations, one set per requirement and
    instrument, as set in the scenario config.
    """
    stem = project_root()
    out_dir = stem + "data/IMG_RAD_09_nominal_background_simmed_data/"

    now = datetime.datetime.now()
    sim_common.start_sim_logging(
        stem, "IMG_RAD_09_nom_bkgd_logs", "IMG_RAD_09_nominal_background", now
    )
    logging.info("Simmed file output directory: %s", out_dir)
    sim_common.setup_scopesim()

    observing = load_observing_config(stem, OBSERVING_CONFIG)
    scenario = load_config_and_pipe(
        config_file_choice=stem
        + "config/config_file_IMG_RAD_09_nominal_background_SIM.yaml",
        print_one_line=False,
    )
    sim_common.run_scenario(scenario, SIMULATORS, observing, out_dir, TEST_PREFIX)


if __name__ == "__main__":
    main()
