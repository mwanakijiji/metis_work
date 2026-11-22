"""
Make simulated data for the IMG-RAD-09 nominal background behaviour test

Reqs.:
- Ref. IMG-RAD-09 Nominal background behaviour

1. (requirement ID TBD) Verify sensititity requirement on point source in a 10000
second exposure.

What is simulated: per instrument set-up (config/config_file_IMG_RAD_09_nominal_
background_SIM.yaml), i.e. per band (Lp and Mp on the LM detector, N2 on the N
detector) and pupil-plane mask (each mask with its own ScopeSim PSF kernel: PPS-CFO2,
PPS-LM, SPM-LM for LM; SPM-N for N), one readout of total_exptime_s (10000 s) of the
WCU pinhole (FP mask pinhole_lm / pinhole_n, black-body aperture fully open), a
stand-in for a point source: one pinhole PSF in the middle of the frame, on the WCU
and instrument background. Nothing is subtracted for the measurement. The DIT is the
configured one, shortened if needed by sim_common.run_scenario so the raw readout
stays below saturation, and NDIT = total_exptime_s / (DIT x n_exposures); ScopeSim
integrates NDIT x DIT in one readout (shot and read noise of the whole integration),
and the frames are per DIT (ScopeSim averages over NDIT).

The quantity of the requirement is the 1-sigma point-source noise of RAW_READOUT in
absolute flux units (backbone_img_rad_09_nom: robust spread of the sums in apertures
of radius aperture_radius_lambda_over_d x lambda/D away from the PSF (noise_mask),
about their median, so the background level drops out; divided by the DIT and times
the counts-to-flux calibration, a PLACEHOLDER in
config/config_file_IMG_RAD_09_nominal_background_observing.yaml); the limit is a
PLACEHOLDER too. Nothing is injected: one natural readout per set-up, written as
IMG_RAD_09_<requirement>_<instrument>_natural.fits (instrument = <band>_<pupil mask>),
with the requirement in the primary header (REQ_ID, REQ_QTY, REQ_LIM, PASS_IF,
REQ_PLH; no EXPECTED), the natural noise as the SIM measures it (NAT_NOIS), the pupil
mask (WCU_PP, PSF_KERN), and the PSF position, exclusion radius and the point source's
flux and S/N (SRC_X, SRC_Y, EXCL_RAD, SRC_FLUX, SRC_SNR). For the usual file layout,
BACKGROUND is an independent dark readout of the same integration (black-body aperture
closed) and BCKGD_SUBTED = RAW_READOUT - BACKGROUND; neither is used for the noise.
"""

import datetime
import logging
import math

import numpy as np

from modules import sim_common
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


def take_long_exposure(metis, instrument, n_exposures, bb_aperture=1.0):
    """
    One exposure of n_exposures readouts of NDIT x DIT (instrument's), averaged (per
    DIT), with the WCU black-body aperture at bb_aperture (1: open, 0: closed, a dark).

    OUTPUTS:
    - np.ndarray: frame (ADU per DIT)
    - astropy header: primary header of the first readout
    """
    frames = []
    header = None
    for _ in range(int(n_exposures)):
        frame, hdr = sim_common.readout_frame(metis, bb_aperture, instrument)
        frames.append(frame)
        header = header if header is not None else hdr
    return np.mean(frames, axis=0), header


def requirement_header_cards(req_key, req_cfg, simver):
    """
    Requirement cards for a natural readout (nothing injected, so no CASE truth,
    EXPECTED or INJ_VAL): REQ_ID, CASE 'natural', REQ_PLH, SIMVER and, per limited
    quantity, REQ_QTY / REQ_LIM / PASS_IF (numbered if more than one).
    """
    cards = [
        ("REQ_ID", req_cfg.get("req_id", req_key), "Requirement simulated"),
        ("CASE", "natural", "Natural readout, nothing injected"),
        (
            "REQ_PLH",
            bool(req_cfg.get("limit_is_placeholder", False)),
            "Requirement limit is a placeholder",
        ),
        ("SIMVER", simver, "ScopeSim, IRDB, code versions"),
    ]
    limit = req_cfg["limit"]
    numbered = len(limit) > 1
    for i, (quantity, spec) in enumerate(limit.items(), start=1):
        n = str(i) if numbered else ""
        cards += [
            (f"REQ_QTY{n}", quantity, f"Quantity ({spec.get('unit', '')})"),
            (f"REQ_LIM{n}", float(spec["value"]), "Requirement limit"),
            (f"PASS_IF{n}", spec["pass_if"], "Pass if measured <op> REQ_LIM"),
        ]
    return cards


@pipeline_stage(
    name="simulate_rad_09_sensitivity",
    cluster=CLUSTER_NOMBKG,
    cluster_color=COLOR_NOMBKG,
    label="SIM 10000 s WCU pinhole readout: point-source noise",
)
def simulate_sensitivity(ctx, req_key, req_cfg):  # pylint: disable=too-many-locals
    """
    Simulated data for the point-source sensitivity in a 10000 s exposure: one long
    WCU pinhole readout (total_exptime_s, at the DIT tuned by run_scenario) with one
    pinhole PSF in the middle; its natural 1-sigma point-source noise (aperture sums
    away from the PSF, in flux units) is logged and written in the header. Nothing is
    injected (see the module docstring).

    OUTPUTS:
    - list[str]: written files
    """
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
    raw, header = take_long_exposure(ctx.metis, inst, n_exposures)
    # independent dark of the same integration: BACKGROUND, for the file layout only
    background, _ = take_long_exposure(ctx.metis, inst, n_exposures, bb_aperture=0.0)

    radius = nom.aperture_radius_px(
        nom.filter_wavelength_m(
            ctx.observing, ctx.instrument["obs_filter"], ctx.det_key
        ),
        float(cfg["telescope_diameter_m"]),
        float(ctx.observing["pixel_scales"][ctx.det_key]),
        float(cfg["aperture_radius_lambda_over_d"]),
    )
    spacing = radius * float(cfg["aperture_spacing_radii"])
    mask, positions, excl_radius = nom.noise_mask(raw, raw, cfg, radius)
    noise_adu, n_apertures = nom.aperture_noise(raw, mask, radius, spacing)
    natural = nom.noise_to_flux(noise_adu, dit, calib)
    src_flux = nom.noise_to_flux(
        nom.source_sum(raw, mask, positions[0], radius), dit, calib
    )
    logging.info(
        "%s (%s, pupil mask %s): PSF at %s, excluded within %.1f px; source flux in "
        "the aperture (above the background) %.6g, S/N %.4g",
        req_key,
        sim_common.instrument_tag(ctx.instrument),
        ctx.instrument["pp_mask"],
        positions[0],
        excl_radius,
        src_flux,
        src_flux / natural,
    )
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

    return [
        sim_common.write_case_file(
            ctx.out_dir,
            ctx.test_prefix,
            req_key,
            inst,
            "natural",
            "",
            raw,
            background,
            header,
            requirement_header_cards(req_key, req_cfg, ctx.simver)
            + [
                ("TARGET", "WCU pinhole", "Point-source stand-in, PSF in the middle"),
                ("SRC_X", positions[0][0], "Point source x (px)"),
                ("SRC_Y", positions[0][1], "Point source y (px)"),
                ("EXCL_RAD", excl_radius, "Source exclusion radius (px)"),
                ("SRC_FLUX", src_flux, "Source flux in the aperture"),
                ("SRC_SNR", src_flux / natural, "Source S/N in the aperture"),
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
            ],
            saturation_adu=ctx.saturation_adu,
        )
    ]


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
