"""
Make simulated data for the IMG-OPT-04 in-field stray light test

Reqs.:
- Ref. Overleaf doc
  IMG_OPT_04_Test_Description_In_Field_Straylight_and_Ghosts

1. METIS-1189: The maximum allowed stray light irradiance from an
in-field source shall be less than 0.1 % of the peak irradiance in the
focal planes of the IMG. Hereby, stray light contains scattering from
opto-mechanical surfaces in Mid-infrared ELT Imager and Spectrograph
(METIS).

2. METIS-1429: The maximum allowed stray light irradiance in the
CFO-FP2 plane from an in-field source positioned in the METIS input
focal plane shall be less than 0.06 % of the peak irradiance. The
maximum allowed stray light irradiance in the IMG-LM and IMG-N
detector planes from an in-field source positioned in the CFO-FP2
plane shall be less than 0.04% of the peak irradiance.

3. METIS-9522: After data reduction and calibration, the flux in
optical artefacts and ghosts shall be less than the 3-sigma thermal
background noise for one hour of observations and for the respective
spatial scale of the ghost, i.e. point-source-like ghosts shall
contain less flux than the point-source sensitivity limit; extended
ghosts shall contain less flux than the surface brightness limit for
that extension. This shall hold when the brightness of the celestial
source causing the artefact(s) corresponds to the saturation limit in
the fastest full-frame operation.

Structure: one function per requirement, driven by the scenario in
config/config_file_IMG_OPT_04_stray_light_SIM.yaml and run by
modules.sim_common.run_scenario. Each writes a case that clearly passes and one that
clearly fails, named IMG_OPT_04_<requirement>_<instrument>_<case>.fits, with the
requirement and the injected truth in the primary header (REQ_ID, CASE, EXPECTED,
REQ_QTY, REQ_LIM, PASS_IF, INJ_VAL, REQ_PLH, SIMVER; see modules/sim_common.py) and the
true artefact regions in a TRUTH_LABELS extension.

A single PSF is simulated in the middle of the detector, and synthetic artefacts
(random contiguous shapes, the same in every case) are added to the illuminated
ScopeSim frame; BACKGROUND is an independent dark.
"""

import datetime
import logging

import numpy as np
from astropy.io import fits
from scipy.ndimage import uniform_filter

import modules.backbone_img_opt_04_stray_light as backbone
from modules import sim_common
from modules.config_common import load_observing_config
from modules.helpers import load_config_and_pipe
from pipeline_registry import (  # pylint: disable=import-error,wrong-import-order
    CLUSTER_STRAY,
    COLOR_STRAY,
    pipeline_stage,
)

TEST_PREFIX = "IMG_OPT_04"


def lambda_over_d_pix(observing, obs_filter, det_key):
    """lambda/D in pixels, from the filter's central wavelength and D_aperture full."""
    band = det_key.split("_")[-1]  # 'lm' or 'n'
    wavel = float(observing[f"monochromatic_observing_filters_{band}"][obs_filter])
    lambda_over_d_mas = wavel / float(observing["D_aperture"]["full"]) * 206265e3
    return lambda_over_d_mas / float(observing["pixel_scales"][det_key])


def artefact_labels(shape, artefacts_cfg, psf_yx, exclusion_pix):
    """
    Label map (0: none, 1..N) of random contiguous artefact shapes, with the shapes
    that come within exclusion_pix of the PSF dropped and the rest renumbered.
    """
    _, labels = backbone.make_random_contiguous_stray_light(
        shape,
        n_shapes=int(artefacts_cfg["n_shapes"]),
        seed=int(artefacts_cfg["seed"]),
        pixels_per_shape=tuple(artefacts_cfg["pixels_per_shape"]),
        intensity_range=(1.0, 1.0),
        smooth_edges=False,
    )
    yy, xx = np.indices(shape)
    near_psf = (yy - psf_yx[0]) ** 2 + (xx - psf_yx[1]) ** 2 < exclusion_pix**2
    dropped = sorted(set(np.unique(labels[near_psf])) - {0})
    if dropped:
        logging.info(
            "Dropping artefacts %s: within %.0f pix of the PSF", dropped, exclusion_pix
        )
    renumbered = np.zeros(shape, dtype=np.int32)
    for new, old in enumerate(sorted(set(np.unique(labels)) - {0} - set(dropped)), 1):
        renumbered[labels == old] = new
    return renumbered


def artefact_image(labels, amplitudes):
    """
    Artefacts with constant irradiance amplitudes[k - 1] over label k, edges softened
    with a 3x3 box filter (which conserves the flux).
    """
    amp_map = np.zeros(labels.shape, dtype=float)
    for k, amplitude in enumerate(amplitudes, 1):
        amp_map[labels == k] = amplitude
    return uniform_filter(amp_map, size=3, mode="constant")


def _truth_inputs(ctx, req_cfg):
    """Darks, illuminated frame, PSF peak and position, and the artefact labels."""
    dark_1, _ = ctx.darks()
    flat, _ = ctx.lit_frame()
    signal = flat - dark_1
    psf_yx = np.unravel_index(np.argmax(signal), signal.shape)
    exclusion_pix = float(
        req_cfg["artefacts"]["psf_exclusion_lambda_over_d"]
    ) * lambda_over_d_pix(ctx.observing, ctx.instrument["obs_filter"], ctx.det_key)
    labels = artefact_labels(signal.shape, req_cfg["artefacts"], psf_yx, exclusion_pix)
    logging.info(
        "PSF peak (signal) %.4g counts at (y, x) = %s; %s artefacts",
        float(signal[psf_yx]),
        psf_yx,
        int(labels.max()),
    )
    return flat, float(signal[psf_yx]), labels


def _truth_hdu(labels):
    return fits.ImageHDU(data=labels.astype(np.int32), name="TRUTH_LABELS")


@pipeline_stage(
    name="sim_img_04_metis_1189_1429",
    cluster=CLUSTER_STRAY,
    cluster_color=COLOR_STRAY,
    label="SIM METIS-1189/-1429: stray light vs. PSF peak",
)
def simulate_metis_1189_1429(ctx, req_key, req_cfg):
    """
    Simulated data for METIS-1189 and -1429: stray light irradiance < 0.04% of the
    peak irradiance.

    Per case, the illuminated ScopeSim frame plus artefacts whose peak irradiance is
    the injected fraction peak_frac of the PSF peak (the PSF peak from the signal,
    illuminated frame - dark_1).

    OUTPUTS:
    - list[str]: written files
    """
    expected = sim_common.check_cases(req_key, req_cfg)
    flat, ref_peak, labels = _truth_inputs(ctx, req_cfg)
    n_artefacts = int(labels.max())

    written = []
    for case, case_cfg in req_cfg["cases"].items():
        inject = case_cfg["inject"]
        amplitude = float(inject["peak_frac"]) * ref_peak
        artefacts = artefact_image(labels, [amplitude] * n_artefacts)
        written.append(
            ctx.write_case(
                req_key,
                case,
                "",
                flat + artefacts,
                sim_common.requirement_cards(
                    req_key, req_cfg, case, inject, expected[case], ctx.simver
                )
                + [
                    ("N_ART", n_artefacts, "Number of artefacts injected"),
                    ("ART_PEAK", amplitude, "Peak irradiance of each artefact"),
                    ("REF_PEAK", ref_peak, "Peak of bckgd-subted PSF"),
                ],
                extra_hdus=[_truth_hdu(labels)],
            )
        )
    return written


@pipeline_stage(
    name="sim_img_04_metis_9522",
    cluster=CLUSTER_STRAY,
    cluster_color=COLOR_STRAY,
    label="SIM METIS-9522: artefacts vs. 3-sigma background",
)
def simulate_metis_9522(ctx, req_key, req_cfg):
    """
    Simulated data for METIS-9522: the flux of each artefact < 3-sigma of the
    background noise for its spatial scale.

    Per case, the illuminated ScopeSim frame plus artefacts each with total flux =
    flux_over_3sigma * 3 * sigma_pix * sqrt(area), where sigma_pix is the per-pixel
    noise of a background-subtracted frame (std of dark_2 - dark_1).

    OUTPUTS:
    - list[str]: written files
    """
    expected = sim_common.check_cases(req_key, req_cfg)
    flat, ref_peak, labels = _truth_inputs(ctx, req_cfg)
    dark_1, dark_2 = ctx.darks()
    sigma_pix = float(np.std(dark_2 - dark_1))
    areas = np.bincount(labels.ravel())[1:]
    logging.info(
        "%s: per-pixel noise %.4g counts; artefact areas %s pix",
        req_key,
        sigma_pix,
        areas.tolist(),
    )

    written = []
    for case, case_cfg in req_cfg["cases"].items():
        inject = case_cfg["inject"]
        ratio = float(inject["flux_over_3sigma"])
        # total flux ratio * 3 sigma_pix sqrt(A), spread evenly over the A pixels
        amplitudes = [ratio * 3.0 * sigma_pix / np.sqrt(area) for area in areas]
        artefacts = artefact_image(labels, amplitudes)
        written.append(
            ctx.write_case(
                req_key,
                case,
                "",
                flat + artefacts,
                sim_common.requirement_cards(
                    req_key, req_cfg, case, inject, expected[case], ctx.simver
                )
                + [
                    ("N_ART", int(labels.max()), "Number of artefacts injected"),
                    ("SIG_PIX", sigma_pix, "Per-pixel noise of bckgd-subted frame"),
                    ("REF_PEAK", ref_peak, "Peak of bckgd-subted PSF"),
                ],
                extra_hdus=[_truth_hdu(labels)],
            )
        )
    return written


# requirement key (in the scenario config) -> function writing its simulated data
SIMULATORS = {
    "METIS-1189-1429": simulate_metis_1189_1429,
    "METIS-9522": simulate_metis_9522,
}


@pipeline_stage(
    name="generate_stray_light_sim",
    depends_on=("sim_img_04_metis_1189_1429", "sim_img_04_metis_9522"),
    cluster=CLUSTER_STRAY,
    cluster_color=COLOR_STRAY,
    label="Generate stray-light sims",
)
def main():
    """
    Generate IMG-OPT-04 stray-light simulations, one set per requirement, as set in the
    scenario config.
    """
    stem = "/podman-share/metis_work/METIS_MAIT_code_eckhart/"
    # write simulated data here, where the ANALYSIS script reads it
    out_dir = stem + "data/IMG_OPT_04_stray_light_simmed_data/"

    now = datetime.datetime.now()
    sim_common.start_sim_logging(
        stem, "IMG_04_simmed_stray_light_logs", "IMG_04_simmed_stray_light", now
    )
    logging.info("Simmed file output directory: %s", out_dir)
    sim_common.setup_scopesim()

    observing = load_observing_config(
        stem, "config/config_file_IMG_04_stray_light_observing.yaml"
    )
    scenario = load_config_and_pipe(
        config_file_choice=stem + "config/config_file_IMG_OPT_04_stray_light_SIM.yaml",
        print_one_line=False,
    )
    sim_common.run_scenario(scenario, SIMULATORS, observing, out_dir, TEST_PREFIX)


if __name__ == "__main__":
    main()
