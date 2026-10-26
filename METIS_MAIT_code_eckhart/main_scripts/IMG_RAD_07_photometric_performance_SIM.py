"""
Make simulated data for the IMG-RAD-07 photometric performance test

Reqs.:
- Ref. IMG-RAD-07 Photometric performance, E-XXX-NOVA-MET-NNNN 0-1,
  Table 1-1

1. METIS-2594 (verified by test at System level): METIS internal
calibration shall provide a range of flux levels matching that
encountered in scientific operation as defined in METIS-3137 -
E-TNT-MPIA-MET-1004.

2. METIS-2595 (verified by test at Observatory level, applicable to
IMG): Relative photometric accuracy shall be better than 0.97% rms.

3. METIS-2595 (verified by test at Observatory level, applicable to
IMG): Uncertainty in color correction should be less than 1% rms.

4. METIS-2883 (verified by test at subsystem level): The source used
for detector linearity calibration METIS-3289 shall provide an
illumination that is spatially homogeneous within +/- 3%
peak-to-valley over the science field of view of METIS.
------> Procedure: see Calibration Plan E-PLA-NOVA-MET-1066 4-0 24-08-2022, Sec. 2.3

5. METIS-3291 (verified by test at System level, applicable to WCU and
PIP): METIS shall allow the characterisation and correction of the
relative spectral response function (RSRF) with 3% precision over the
wavelength range specified in METIS-1090.
------> Procedure: see Calibration Plan E-PLA-NOVA-MET-1066 4-0 24-08-2022, p. 23-4

6. METIS-3300 (verified by test at System level, applicable to WCU):
The level of flux of the flat field shall be tunable to 15 discrete
values and shall be known at each step to 0.3%. The resulting spatial
illumination pattern shall remain constant across the field within
0.5% (0.3% as goal) after tuning.

7. METIS-3301 (verified by test at subsystem level, applicable to
WCU): After calibration, the distortions introduced by METIS shall be
removed to better than 0.5 mas (ca. 1/10 px for the L band imager) over
the full field of view.

8. Stretch goal 1 (not an FDR req.): Aperture correction factor shall
be known.

Structure: one function per requirement (simulate_metis_NNNN), driven by the scenario in
config/config_file_IMG_RAD_07_phot_performance_SIM.yaml (limits and injected values per
case) and run by modules.sim_common.run_scenario. Each function writes at least one case
that clearly passes and one that clearly fails, named
IMG_RAD_07_<requirement>_<filter>_<case>[_...].fits, with the requirement and the
injected truth in the primary header:
  REQ_ID    requirement ID, e.g. 'METIS-2594'
  CASE      case name from the config ('pass', 'fail', 'near', ...)
  EXPECTED  'PASS' or 'FAIL' (injected value vs. limit)
  REQ_QTY, REQ_LIM, PASS_IF, INJ_VAL   quantity, limit, sense and injected value
            (numbered REQ_QTY1, REQ_LIM1, ... if a requirement has several quantities)
  REQ_PLH   True if the limit is a placeholder
  SIMVER    ScopeSim and IRDB METIS versions, and the git commit of this code
so the ANALYSIS can check its verdict against the injected truth.

The injection is applied to the ScopeSim signal (flat - dark):
raw = flat + (factor - 1) * (flat - dark_1), and the written BACKGROUND is an
independent dark (dark_2), so BCKGD_SUBTED = raw - dark_2 has realistic noise. Images
are written as float32. All raw readouts peak at <= sim_common.MAX_RAW_FRAC of
saturation (DIT tuned by sim_common.run_scenario; METIS-2594 levels capped below that).
"""

import datetime
import logging

import numpy as np

from modules import sim_common
from modules.config_common import load_observing_config
from modules.helpers import load_config_and_pipe
from pipeline_registry import (  # pylint: disable=import-error,wrong-import-order
    CLUSTER_PHOT,
    COLOR_PHOT,
    pipeline_stage,
)

TEST_PREFIX = "IMG_RAD_07"


def signal_ceiling(dark, signal, max_raw):
    """
    Largest scaling k_c of the signal such that no pixel of dark + k * signal exceeds
    max_raw, and the median raw level at that scaling (the ceiling of the calibration
    flux levels).

    OUTPUTS:
    - float: k_c
    - float: median(dark + k_c * signal) (ADU)

    Raises ValueError if there is no positive signal, or if the dark alone already
    reaches max_raw where there is signal.
    """
    dark = np.asarray(dark, dtype=float)
    signal = np.asarray(signal, dtype=float)
    positive = signal > 0
    if not np.any(positive):
        raise ValueError("No positive signal to scale")
    k_c = float(np.min((max_raw - dark[positive]) / signal[positive]))
    if k_c <= 0:
        raise ValueError(
            f"The dark alone reaches the raw ceiling {max_raw:.4g} ADU: no headroom "
            "for calibration flux levels"
        )
    return k_c, float(np.median(dark + k_c * signal))


def raw_target_level(background, ceiling, frac):
    """
    Raw level (ADU) a fraction frac (0..1) of the way from the background up to the
    ceiling: background + frac * (ceiling - background).

    Raises ValueError if the background is at or above the ceiling (no headroom), or
    if frac is outside [0, 1].
    """
    background, ceiling, frac = float(background), float(ceiling), float(frac)
    if background >= ceiling:
        raise ValueError(
            f"Background {background:.4g} ADU is at or above the ceiling "
            f"({ceiling:.4g} ADU): no headroom for calibration flux levels"
        )
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"Level fraction {frac} is outside [0, 1]")
    return background + frac * (ceiling - background)


def signal_factor_for_raw_level(dark, signal, target, n_iter=5):
    """
    Factor k such that the median of the raw frame dark + k * signal is target (ADU).
    The median is not linear, so k is refined iteratively from the first guess
    (target - median(dark)) / median(signal).

    Raises ValueError if target is below the median of the dark (not reachable by
    scaling the signal).
    """
    median_dark = float(np.median(dark))
    median_signal = float(np.median(signal))
    if target < median_dark:
        raise ValueError(
            f"Target raw level {target:.4g} ADU is below the median of the dark "
            f"({median_dark:.4g} ADU): not reachable by scaling the signal"
        )
    factor = (target - median_dark) / median_signal
    for _ in range(n_iter):
        factor += (target - float(np.median(dark + factor * signal))) / median_signal
    return max(factor, 0.0)


@pipeline_stage(
    name="simulate_metis_2594",
    cluster=CLUSTER_PHOT,
    cluster_color=COLOR_PHOT,
    label="SIM METIS-2594: calibration flux range",
)
def simulate_metis_2594(ctx, req_key, req_cfg):  # pylint: disable=too-many-locals
    """
    Simulated data for METIS-2594: METIS internal calibration shall provide a range of
    flux levels matching that encountered in scientific operation as defined in
    METIS-3137.

    Per case, two flats at the faintest and brightest flux the calibration provides.
    The injected flux_min, flux_max are fractions (0..1) of the headroom between the
    closed-aperture background B (median of dark_1) and the ceiling C: the median raw
    level at the largest scaling of the signal for which no pixel of the raw readout
    exceeds sim_common.MAX_RAW_FRAC of saturation (signal_ceiling). The raw level
    (median of the raw frame, background included) is B + frac * (C - B). Each flat is
    made by scaling the signal (flat - dark) of a full-aperture ScopeSim flat to that
    median; the noise of the scaled signal is scaled with it, so it is not shot-noise
    correct for large factors. Writes ..._<case>_min.fits and ..._<case>_max.fits for
    one PASS case and one FAIL case on each side.

    OUTPUTS:
    - list[str]: written files
    """
    expected = sim_common.check_cases(req_key, req_cfg)
    dark_1, _ = ctx.darks()
    flat, _ = ctx.lit_frame()
    signal = flat - dark_1
    saturation = float(ctx.observing["saturation_adu"][ctx.det_key])
    max_raw = sim_common.MAX_RAW_FRAC * saturation
    background = float(np.median(dark_1))
    k_ceiling, ceiling = signal_ceiling(dark_1, signal, max_raw)
    logging.info(
        "%s: background (dark median) %.4g ADU = %.3g of saturation (%.0f ADU, %s); "
        "ceiling (raw median with max raw = %.2f of saturation) %.4g ADU; headroom "
        "%.4g ADU",
        req_key,
        background,
        background / saturation,
        saturation,
        ctx.det_key,
        sim_common.MAX_RAW_FRAC,
        ceiling,
        ceiling - background,
    )

    written = []
    for case, case_cfg in req_cfg["cases"].items():
        inject = case_cfg["inject"]
        cards = sim_common.requirement_cards(
            req_key, req_cfg, case, inject, expected[case], ctx.simver
        )
        for end in ("min", "max"):
            frac = float(inject[f"flux_{end}"])
            target = raw_target_level(background, ceiling, frac)
            # never past the ceiling, whatever the median iteration does
            factor = min(
                signal_factor_for_raw_level(dark_1, signal, target), k_ceiling
            )
            raw = dark_1 + factor * signal
            logging.info(
                "%s case %s %s: %.3g of the headroom -> target raw median %.4g ADU; "
                "signal factor %.4g; raw median %.4g, max %.4g ADU",
                req_key,
                case,
                end,
                frac,
                target,
                factor,
                float(np.median(raw)),
                float(np.max(raw)),
            )
            written.append(
                ctx.write_case(
                    req_key,
                    case,
                    f"_{end}",
                    raw,
                    cards
                    + [
                        ("LEVEL", end.upper(), "Faintest/brightest calibration flux"),
                        ("FLUXFRAC", frac, "(raw - bckgd) / (ceiling - bckgd)"),
                        ("FLUXINJ", target, "Injected raw level (ADU/DIT, median)"),
                        ("BKG_ADU", background, "Background: dark median (ADU/DIT)"),
                        ("CEIL_ADU", ceiling, "Ceiling: raw median at max cap"),
                        ("SIGFAC", factor, "Signal scaling factor"),
                    ],
                )
            )
    return written


@pipeline_stage(
    name="simulate_metis_2883",
    cluster=CLUSTER_PHOT,
    cluster_color=COLOR_PHOT,
    label="SIM METIS-2883: flat homogeneity",
)
def simulate_metis_2883(ctx, req_key, req_cfg):
    """
    Simulated data for METIS-2883: the source used for detector linearity calibration
    shall provide an illumination that is spatially homogeneous within +/- 3%
    peak-to-valley over the science field of view.

    Per case, a full-aperture ScopeSim flat with a smooth illumination pattern injected
    so that max |pixel / mean - 1| = injected half_ptv (on top of whatever
    inhomogeneity ScopeSim itself produces). Writes one PASS case and one FAIL case
    (and e.g. one near the limit, if configured).

    OUTPUTS:
    - list[str]: written files
    """
    expected = sim_common.check_cases(req_key, req_cfg)
    dark_1, _ = ctx.darks()
    flat, _ = ctx.lit_frame()
    signal = flat - dark_1
    pattern = sim_common.smooth_pattern(signal.shape, "tilt_and_radial")

    written = []
    for case, case_cfg in req_cfg["cases"].items():
        inject = case_cfg["inject"]
        raw = flat + float(inject["half_ptv"]) * pattern * signal
        written.append(
            ctx.write_case(
                req_key,
                case,
                "",
                raw,
                sim_common.requirement_cards(
                    req_key, req_cfg, case, inject, expected[case], ctx.simver
                ),
            )
        )
    return written


@pipeline_stage(
    name="simulate_metis_3300",
    cluster=CLUSTER_PHOT,
    cluster_color=COLOR_PHOT,
    label="SIM METIS-3300: 15 flat-field flux steps",
)
def simulate_metis_3300(ctx, req_key, req_cfg):
    """
    Simulated data for METIS-3300: the level of flux of the flat field shall be tunable
    to 15 discrete values and shall be known at each step to 0.3%; the resulting
    spatial illumination pattern shall remain constant across the field within 0.5%
    after tuning.

    Flux steps: ScopeSim flats at n_steps WCU black-body apertures (header BB_APER, the
    nominal relative flux). Per case and step, injected:
    - flux error: the flux is off the nominal by +/- flux_error (alternating sign),
    - pattern change: a smooth pattern with amplitude ramping from -pattern_change to
      +pattern_change over the steps (so each step's normalised flat differs from the
      step-averaged one by at most pattern_change).
    Writes ..._<case>_stepNN.fits for one PASS case and one FAIL case (more if
    configured), one ScopeSim readout per step shared by all cases.

    OUTPUTS:
    - list[str]: written files
    """
    expected = sim_common.check_cases(req_key, req_cfg)
    n_steps = int(req_cfg["n_steps"])
    apertures = np.linspace(
        float(req_cfg["bb_aperture_min"]), float(req_cfg["bb_aperture_max"]), n_steps
    )
    dark_1, dark_2 = ctx.darks()
    pattern = None
    signs = np.where(np.arange(n_steps) % 2 == 0, 1.0, -1.0)
    ramp = np.linspace(-1.0, 1.0, n_steps)

    written = []
    for step, aperture in enumerate(apertures):
        # one ScopeSim readout per step, injected differently for each case
        flat, header = sim_common.readout_frame(ctx.metis, aperture, ctx.instrument)
        signal = flat - dark_1
        if pattern is None:
            pattern = sim_common.smooth_pattern(signal.shape, "saddle")
        for case, case_cfg in req_cfg["cases"].items():
            inject = case_cfg["inject"]
            flux_factor = 1.0 + float(inject["flux_error"]) * signs[step]
            pattern_factor = (
                1.0 + float(inject["pattern_change"]) * ramp[step] * pattern
            )
            raw = flat + (flux_factor * pattern_factor - 1.0) * signal
            written.append(
                sim_common.write_case_file(
                    ctx.out_dir,
                    ctx.test_prefix,
                    req_key,
                    ctx.instrument,
                    case,
                    f"_step{step + 1:02d}",
                    raw,
                    dark_2,
                    header,
                    sim_common.requirement_cards(
                        req_key, req_cfg, case, inject, expected[case], ctx.simver
                    )
                    + [
                        ("STEP", step + 1, "Flux step"),
                        ("N_STEPS", n_steps, "Number of flux steps"),
                        ("BB_APER", float(aperture), "WCU BB aperture (nominal flux)"),
                        ("FLUXFAC", flux_factor, "Injected flux / nominal"),
                        (
                            "PATTAMP",
                            float(inject["pattern_change"]) * ramp[step],
                            "Injected pattern amplitude",
                        ),
                    ],
                    saturation_adu=ctx.saturation_adu,
                )
            )
    return written


# requirement key (in the scenario config) -> function writing its simulated data
SIMULATORS = {
    "METIS-2594": simulate_metis_2594,
    "METIS-2883": simulate_metis_2883,
    "METIS-3300": simulate_metis_3300,
}


def main():
    """
    Generate IMG-RAD-07 photometric performance simulations, one set per requirement,
    as set in the scenario config.
    """
    stem = "/podman-share/metis_work/METIS_MAIT_code_eckhart/"
    out_dir = stem + "data/IMG_OPT_07_photometric_performance_simmed_data/"

    now = datetime.datetime.now()
    sim_common.start_sim_logging(
        stem, "IMG_07_phot_logs", "IMG_07_photometric_performance", now
    )
    logging.info("Simmed file output directory: %s", out_dir)
    sim_common.setup_scopesim()

    observing = load_observing_config(stem)
    scenario = load_config_and_pipe(
        config_file_choice=stem
        + "config/config_file_IMG_RAD_07_phot_performance_SIM.yaml",
        print_one_line=False,
    )
    sim_common.run_scenario(scenario, SIMULATORS, observing, out_dir, TEST_PREFIX)


if __name__ == "__main__":
    main()
