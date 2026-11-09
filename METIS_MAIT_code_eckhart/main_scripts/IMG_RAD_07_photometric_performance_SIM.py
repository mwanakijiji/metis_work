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
IMG_RAD_07_<requirement>_<instrument>_<case>[_...].fits (instruments: Br_alpha on
the LM detector, N2 on the N detector), with the requirement and the
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
saturation: each base set-up has an ND filter (LM: ND_OD2, N: ND_OD1) and the
detector's minimum DIT, the DIT is
checked by sim_common.run_scenario, METIS-2594 levels are capped below that, and the
METIS-3300 flux steps (fixed BB aperture) are made with ND filters and DITs whose
product T_ND x DIT is stepped linearly up to that limit.
"""

import datetime
import logging

import numpy as np
import scopesim as sim

from modules import sim_common
from modules.config_common import load_observing_config
from modules.helpers import load_config_and_pipe
from modules.paths import project_root
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


def max_exposure(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    peak_raw, t_probe, dit_probe, max_raw, margin
):
    """
    Largest exposure E = T_ND x DIT (s) for which the raw peak (background included,
    which also scales with T_ND x DIT) stays at max_raw / (1 + margin), from a probe
    readout peaking at peak_raw with ND transmission t_probe and DIT dit_probe.

    Raises ValueError if the probe has no positive peak.
    """
    if peak_raw <= 0:
        raise ValueError(f"Probe raw peak {peak_raw:.4g} ADU is not positive")
    peak_per_exposure = float(peak_raw) / (float(t_probe) * float(dit_probe))
    return float(max_raw) / (1.0 + float(margin)) / peak_per_exposure


def exposure_steps(e_max, n_steps):
    """Linearly spaced exposures e_max * k / n_steps, k = 1..n_steps."""
    return float(e_max) * np.arange(1, int(n_steps) + 1) / int(n_steps)


def choose_nd_and_dit(exposure, nd_transmissions, mindit):
    """
    ND filter and DIT giving an exposure T_ND x DIT: the least attenuating ND (highest
    transmission) for which DIT = exposure / T_ND is >= mindit, so the DIT stays legal.

    INPUTS:
    - exposure (float): T_ND x DIT (s)
    - nd_transmissions (dict): ND name -> transmission
    - mindit (float): minimum DIT (s)

    OUTPUTS:
    - str: ND name
    - float: DIT (s)

    Raises ValueError if even the most attenuating ND needs DIT < mindit.
    """
    for name, trans in sorted(nd_transmissions.items(), key=lambda item: -item[1]):
        dit = float(exposure) / float(trans)
        if dit >= mindit:
            return name, dit
    raise ValueError(
        f"Exposure {exposure:.4g} s needs a DIT below the minimum ({mindit} s) with "
        f"every ND filter {sorted(nd_transmissions)}: add a denser ND filter"
    )


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
            factor = min(signal_factor_for_raw_level(dark_1, signal, target), k_ceiling)
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
    label="SIM METIS-3300: 15 ND+DIT flux steps",
)
def simulate_metis_3300(  # pylint: disable=too-many-locals,too-many-statements
    ctx, req_key, req_cfg
):
    """
    Simulated data for METIS-3300: the level of flux of the flat field shall be tunable
    to 15 discrete values and shall be known at each step to 0.3%; the resulting
    spatial illumination pattern shall remain constant across the field within 0.5%
    after tuning.

    Flux steps: the WCU black-body aperture is fixed (bb_aperture); the exposure
    E = T_ND x DIT is stepped linearly, E_k = E_max * k / n_steps (header NOMFLUX = k /
    n_steps, the nominal relative flux). E_max puts the raw peak (background included,
    which scales with E too) at sim_common.MAX_RAW_FRAC of saturation / (1 +
    peak_margin), scaled from the probe ctx.lit_frame() (base ND and DIT). Per step,
    the least attenuating ND of nd_filters with DIT = E_k / T_ND >= the detector's
    minimum DIT is used (choose_nd_and_dit), with two darks and a flat read out at that
    ND and DIT (the background scales with it). Per case and step, injected:
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
    bb_aperture = float(req_cfg["bb_aperture"])
    base_nd = ctx.instrument.get("nd_filter")
    if base_nd is None:
        raise ValueError(
            f"{req_key}: the instrument needs a base nd_filter (the probe flat's ND)"
        )
    mindit = sim_common.detector_mindit(ctx.metis)
    nd_trans = {
        name: sim_common.nd_transmission(ctx.metis, name)
        for name in req_cfg["nd_filters"]
    }
    max_raw = sim_common.MAX_RAW_FRAC * ctx.saturation_adu

    # probe: a flat at the base ND and DIT and the steps' BB aperture (the cached base
    # flat if that is 1.0)
    if bb_aperture == 1.0:
        probe, _ = ctx.lit_frame()
    else:
        probe, _ = sim_common.readout_frame(ctx.metis, bb_aperture, ctx.instrument)
    t_probe = sim_common.nd_transmission(ctx.metis, base_nd)
    dit_probe = float(ctx.instrument["dit"])
    e_max = max_exposure(
        float(np.max(probe)),
        t_probe,
        dit_probe,
        max_raw,
        float(req_cfg["peak_margin"]),
    )
    plan = [
        choose_nd_and_dit(exposure, nd_trans, mindit)
        for exposure in exposure_steps(e_max, n_steps)
    ]
    peak_per_exposure = float(np.max(probe)) / (t_probe * dit_probe)
    logging.info(
        "%s: probe raw peak %.4g ADU (%s, T %.3g, DIT %.4g s) -> E_max %.4g s; "
        "mindit %.3g s",
        req_key,
        float(np.max(probe)),
        base_nd,
        t_probe,
        dit_probe,
        e_max,
        mindit,
    )
    for step, (nd_name, dit) in enumerate(plan):
        logging.info(
            "%s step %02d: %s (T %.3g), DIT %.4g s -> expected raw peak %.3f of "
            "saturation",
            req_key,
            step + 1,
            nd_name,
            nd_trans[nd_name],
            dit,
            peak_per_exposure * nd_trans[nd_name] * dit / ctx.saturation_adu,
        )

    pattern = None
    signs = np.where(np.arange(n_steps) % 2 == 0, 1.0, -1.0)
    ramp = np.linspace(-1.0, 1.0, n_steps)

    written = []
    try:
        for step, (nd_name, dit) in enumerate(plan):
            # one ScopeSim readout per step (darks at the same ND and DIT, as the
            # background scales with both), injected differently for each case
            step_inst = {**ctx.instrument, "nd_filter": nd_name, "dit": dit}
            sim_common.set_nd_filter(ctx.metis, nd_name)
            dark_1, dark_2 = sim_common.take_darks(ctx.metis, step_inst)
            flat, header = sim_common.readout_frame(ctx.metis, bb_aperture, step_inst)
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
                        step_inst,
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
                            ("BB_APER", bb_aperture, "WCU BB aperture (fixed)"),
                            ("ND_TRANS", nd_trans[nd_name], "ND filter transmission"),
                            ("EXPOSURE", nd_trans[nd_name] * dit, "T_ND x DIT (s)"),
                            ("NOMFLUX", (step + 1) / n_steps, "Nominal relative flux"),
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
    finally:
        sim_common.set_nd_filter(ctx.metis, base_nd)
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
    stem = project_root()
    out_dir = stem + "data/IMG_RAD_07_photometric_performance_simmed_data/"

    now = datetime.datetime.now()
    sim_common.start_sim_logging(
        stem, "IMG_RAD_07_phot_logs", "IMG_RAD_07_photometric_performance", now
    )
    logging.info("Simmed file output directory: %s", out_dir)
    # uncomment to (re-)download the IRDB packages (overwrites them in the IRDB)
    # sim.download_packages(
    #     ["METIS", "ELT", "Armazones"], save_dir=sim_common.IRDB_DIR
    # )
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
