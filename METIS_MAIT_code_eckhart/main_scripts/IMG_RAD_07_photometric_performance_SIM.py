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
case) and run by modules.sim_common.run_scenario. Each function (except the stub for
METIS-2594, which writes nothing) writes at least one case that clearly passes and one
that clearly fails, named
IMG_RAD_07_<requirement>_<instrument>_<case>[_...].fits (instruments: Br_alpha on
the LM detector, N2 on the N detector, and for METIS-3291 the long-slit set-ups
L_spec, M_spec (wcu_lss_l, wcu_lss_m; LM detector) and N_spec (wcu_lss_n; N detector),
whose files also have an 'RSRF' table extension), with the requirement
and the injected truth in the primary header:
  REQ_ID    requirement ID, e.g. 'METIS-3300'
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
detector's minimum DIT, the DIT is checked by sim_common.run_scenario, and the
METIS-3300 flux steps (fixed BB aperture) are made with ND filters and DITs whose
product T_ND x DIT is stepped linearly up to that limit. METIS-3291 (RSRF) multiplies
the signal by a smooth function of wavelength (see simulate_metis_3291 and
modules/rsrf_common.py); with plot_dir set, it also writes an FYI plot there.
"""

import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

from modules import rsrf_common, sim_common
from modules.config_common import load_observing_config
from modules.helpers import load_config_and_pipe
from modules.paths import project_root
from pipeline_registry import (  # pylint: disable=import-error,wrong-import-order
    CLUSTER_PHOT,
    COLOR_PHOT,
    pipeline_stage,
)

TEST_PREFIX = "IMG_RAD_07"


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
    planned=True,  # stub: drawn dashed in the pipeline flow diagram
    label="SIM METIS-2594: calibration flux range (stub)",
)
def simulate_metis_2594(ctx, req_key, req_cfg):
    """
    Stub for METIS-2594: METIS internal calibration shall provide a range of flux
    levels matching that encountered in scientific operation as defined in METIS-3137.

    Writes nothing (the scenario's cases are still checked, so the config stays
    valid for a future implementation).

    OUTPUTS:
    - list[str]: written files (empty)
    """
    sim_common.check_cases(req_key, req_cfg)
    logging.info(
        "%s (%s): stub, no files written",
        req_key,
        sim_common.instrument_tag(ctx.instrument),
    )
    return []


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


def rsrf_error_pattern(lam_um, lam_eval_um):
    """
    Smooth pattern P(lambda) for the METIS-3291 injection: a tilt, a curvature and a
    slow wave in x = lambda scaled to [-1, 1] over lam_eval_um (the rows the RSRF is
    evaluated on; x is clipped outside them), with mean 0 and max |P| = 1 over
    lam_eval_um, so that an RSRF times (1 + a * P) deviates from the original by at
    most a there.

    INPUTS:
    - lam_um (ndarray): wavelengths to evaluate P at [um] (any shape)
    - lam_eval_um (ndarray): wavelengths of the evaluated rows [um]

    OUTPUTS:
    - ndarray: P, the shape of lam_um
    """
    lo, hi = float(np.min(lam_eval_um)), float(np.max(lam_eval_um))

    def shape(lam):
        x = np.clip(2.0 * (np.asarray(lam, float) - lo) / (hi - lo) - 1.0, -1.0, 1.0)
        return 0.5 * x + 0.3 * x**2 + 0.4 * np.sin(2.5 * np.pi * x)

    ref = shape(lam_eval_um)
    offset = float(np.mean(ref))
    scale = float(np.max(np.abs(ref - offset)))
    return (shape(lam_um) - offset) / scale


def rsrf_deviation(rsrf_case, rsrf_ref, evaluate):
    """
    The quantity METIS-3291 is tested on: max |rsrf / rsrf_ref - 1| over the evaluated
    rows (what the SIM injects as rsrf_error).
    """
    return float(np.max(np.abs(rsrf_case[evaluate] / rsrf_ref[evaluate] - 1.0)))


def rsrf_table(spec, evaluate, model, pattern_rows, rsrf_error):
    """
    RSRF extension (one row per detector row): the wavelength solution, the
    un-injected S_empirical, the model (S_perfect, alpha_fcn), the reference RSRF and
    the injected (true) RSRF of the case.

    INPUTS:
    - spec (rsrf_common.SlitSpectrum): of the un-injected signal
    - evaluate (ndarray): bool per row, RSRF evaluated there
    - model (dict): s_perfect, alpha, rsrf_ref (per row)
    - pattern_rows (ndarray): rsrf_error_pattern per row
    - rsrf_error (float): injected amplitude

    OUTPUTS:
    - fits.BinTableHDU: EXTNAME 'RSRF'
    """
    columns = {
        "ROW": np.arange(spec.lam_rows.size),
        "WAVELENGTH": spec.lam_rows,
        "IN_RANGE": spec.in_range,
        "EVAL": evaluate,
        "S_EMP_REF": spec.spectrum,
        "S_PERFECT": model["s_perfect"],
        "ALPHA": model["alpha"],
        "RSRF_REF": model["rsrf_ref"],
        "PATTERN": pattern_rows,
        "RSRF_TRUE": model["rsrf_ref"] * (1.0 + rsrf_error * pattern_rows),
    }
    table = Table(columns)
    table["WAVELENGTH"].unit = "um"
    for name in ("S_EMP_REF", "S_PERFECT"):
        table[name].unit = "adu"
    hdu = fits.table_to_hdu(table)
    hdu.name = "RSRF"
    return hdu


def fyi_plot_rsrf(plot_dir, label, spec, model):
    """
    FYI plot of the METIS-3291 simulation: left, S_empirical (un-injected), S_perfect
    and S_perfect * alpha_fcn; right, the reference RSRF (RSRF_REF).

    INPUTS:
    - plot_dir (str): where to write fyi_sim_<label>.png
    - label (str): e.g. 'METIS-3291_L_spec'
    - spec (rsrf_common.SlitSpectrum), model (dict): see rsrf_table

    OUTPUTS:
    - str: path of the written plot
    """
    lam = spec.lam_rows
    rows = spec.in_range
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    axes[0].plot(lam[rows], spec.spectrum[rows], label="S_empirical (un-injected)")
    axes[0].plot(lam[rows], model["s_perfect"][rows], label="S_perfect")
    axes[0].plot(
        lam[rows],
        (model["s_perfect"] * model["alpha"])[rows],
        label="S_perfect * alpha_fcn",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("wavelength [um]")
    axes[0].set_ylabel("ADU per row")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Spectra (sum over the slit)", fontsize=10)

    axes[1].plot(lam[rows], model["rsrf_ref"][rows], "k-", lw=0.8, label="RSRF_REF")
    axes[1].set_xlabel("wavelength [um]")
    axes[1].set_ylabel("rsrf = S_empirical / (S_perfect * alpha_fcn)")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].set_title("Reference RSRF (un-injected)", fontsize=10)

    fig.suptitle(f"IMG-RAD-07 SIM {label}", fontsize=11)
    fig.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    file_name = os.path.join(plot_dir, f"fyi_sim_{label}.png")
    fig.savefig(file_name)
    plt.close(fig)
    logging.info("Saved FYI plot %s", file_name)
    return file_name


@pipeline_stage(
    name="simulate_metis_3291",
    cluster=CLUSTER_PHOT,
    cluster_color=COLOR_PHOT,
    label="SIM METIS-3291: RSRF precision",
)
def simulate_metis_3291(ctx, req_key, req_cfg):  # pylint: disable=too-many-locals
    """
    Simulated data for METIS-3291: METIS shall allow the characterisation and
    correction of the relative spectral response function (RSRF) with 3% precision
    over the wavelength range specified in METIS-1090.

    Needs an LSS instrument set-up (wcu_lss_l/m/n). From the un-injected ScopeSim
    signal (lit frame - dark_1), with the naming of modules/rsrf_common.py:
    - wavelength per pixel from ScopeSim's spectral trace, and S_empirical, the sum
      over the illuminated slit columns per row,
    - S_perfect: the BB at the WCU temperature (unit throughput and QE), scaled so its
      integral is src_scale_factor x that of S_empirical,
    - alpha_fcn: the observing filter's smoothed ripple, scaled so that
      integral(S_perfect * alpha_fcn) = alpha_integral_frac x integral(S_perfect),
    - the reference RSRF, RSRF_REF = S_empirical / (S_perfect * alpha_fcn).
    The RSRF is evaluated on the rows inside the trace's range and the filter passband
    (standing in for the METIS-1090 range). Per case, the signal is multiplied by
    (1 + rsrf_error * P(lambda)), P a smooth pattern in wavelength with mean 0 and max
    |P| = 1 over the evaluated rows (rsrf_error_pattern), so the RSRF measured from
    the case file deviates from RSRF_REF by max |rsrf / RSRF_REF - 1| = rsrf_error
    (plus noise; rsrf_deviation). Each case file gets an 'RSRF' table extension
    (rsrf_table) with the wavelength solution, the model and the reference and true
    RSRFs, for the ANALYSIS. Writes one PASS case and one FAIL case (more if
    configured).

    OUTPUTS:
    - list[str]: written files
    """
    expected = sim_common.check_cases(req_key, req_cfg)
    rsrf_cfg = req_cfg["rsrf"]
    dark_1, dark_2 = ctx.darks()
    frame, _ = ctx.lit_frame()
    signal = frame - dark_1

    # wavelength solution and the un-injected S_empirical
    lam_map, _, trace = rsrf_common.lss_wavelength_map(
        ctx.metis, signal.shape, ctx.det_key, sim_common.IRDB_DIR
    )
    spec = rsrf_common.slit_spectrum(
        signal, lam_map, trace.wave_min, trace.wave_max, rsrf_cfg["slit_frac"]
    )
    rows = spec.in_range

    # evaluated rows: trace range and filter passband
    filter_file = rsrf_common.filter_file_of(
        ctx.observing, ctx.det_key, ctx.instrument["obs_filter"], sim_common.IRDB_DIR
    )
    filt_lam, ripple, filt_in_band = rsrf_common.filter_ripple(
        filter_file,
        rsrf_cfg["alpha_ripple_sigma_um"],
        rsrf_cfg["alpha_trend_sigma_um"],
        rsrf_cfg["alpha_in_band_frac"],
    )
    evaluate = rsrf_common.eval_rows(spec.lam_rows, rows, filt_lam, filt_in_band)
    if np.count_nonzero(evaluate) < 2:
        raise ValueError(
            f"{req_key}: fewer than 2 rows in the trace range and the "
            f"{filter_file.name} passband"
        )

    # S_perfect (BB at the WCU temperature, scaled) and alpha_fcn
    bb_temp = u.Quantity(ctx.metis["wcu_source"].meta["bb_temp"], u.K)
    s_unit = rsrf_common.perfect_bb_spectrum(
        spec.lam_rows,
        bb_temp,
        ctx.metis.cmds["!TEL.area"],
        ctx.metis.cmds["!INST.pixel_scale"],
        float(ctx.metis.cmds["!DET.gain"]),
        np.count_nonzero(spec.slit_cols),
        float(ctx.instrument["dit"]) * float(ctx.instrument["ndit"]),
    )
    _, src_norm = rsrf_common.scale_to_integral(
        s_unit[rows], spec.spectrum[rows], rsrf_cfg["src_scale_factor"]
    )
    s_perfect = s_unit * src_norm
    alpha_fcn, alpha_scale = rsrf_common.make_alpha_fcn(
        filt_lam,
        ripple,
        spec.lam_rows[rows],
        s_perfect[rows],
        rsrf_cfg["alpha_integral_frac"],
    )
    alpha = alpha_fcn(spec.lam_rows)
    rsrf_ref = np.full(spec.lam_rows.shape, np.nan)
    rsrf_ref[rows] = rsrf_common.rsrf(spec.spectrum[rows], s_perfect[rows], alpha[rows])
    model = {"s_perfect": s_perfect, "alpha": alpha, "rsrf_ref": rsrf_ref}
    lam_eval = spec.lam_rows[evaluate]
    logging.info(
        "%s: trace %s %.3f-%.3f um, %d slit columns, %d rows in range, RSRF "
        "evaluated on %d rows (%.3f-%.3f um, %s passband); BB %.0f K; S_perfect "
        "scaled by %.4g "
        "(src_scale_factor %g); alpha_fcn scale %.4g (alpha_integral_frac %g); "
        "RSRF_REF median %.4g",
        req_key,
        trace.trace_id,
        trace.wave_min,
        trace.wave_max,
        np.count_nonzero(spec.slit_cols),
        np.count_nonzero(rows),
        lam_eval.size,
        lam_eval.min(),
        lam_eval.max(),
        filter_file.name,
        bb_temp.value,
        src_norm,
        rsrf_cfg["src_scale_factor"],
        alpha_scale,
        rsrf_cfg["alpha_integral_frac"],
        float(np.nanmedian(rsrf_ref[evaluate])),
    )

    pattern_rows = rsrf_error_pattern(spec.lam_rows, lam_eval)
    pattern_map = rsrf_error_pattern(lam_map, lam_eval)
    del lam_map

    written = []
    for case, case_cfg in req_cfg["cases"].items():
        inject = case_cfg["inject"]
        rsrf_error = float(inject["rsrf_error"])
        raw = frame + rsrf_error * pattern_map * signal

        # as the ANALYSIS measures it: RSRF of the written BCKGD_SUBTED vs. RSRF_REF
        s_case = (raw - dark_2)[:, spec.slit_cols].sum(axis=1)
        rsrf_case = np.full(spec.lam_rows.shape, np.nan)
        rsrf_case[rows] = rsrf_common.rsrf(s_case[rows], s_perfect[rows], alpha[rows])
        measured = rsrf_deviation(rsrf_case, rsrf_ref, evaluate)
        logging.info(
            "%s case %s: injected rsrf_error %g, measured max |rsrf / RSRF_REF - 1| "
            "%.4g",
            req_key,
            case,
            rsrf_error,
            measured,
        )
        written.append(
            ctx.write_case(
                req_key,
                case,
                "",
                raw,
                sim_common.requirement_cards(
                    req_key, req_cfg, case, inject, expected[case], ctx.simver
                )
                + [
                    ("BBTEMP_K", bb_temp.value, "WCU BB temperature used (K)"),
                    ("TRACE_ID", str(trace.trace_id), "ScopeSim spectral trace"),
                    ("WAVEMIN", float(trace.wave_min), "Trace wavelength min (um)"),
                    ("WAVEMAX", float(trace.wave_max), "Trace wavelength max (um)"),
                    ("EVALMIN", float(lam_eval.min()), "RSRF evaluated from (um)"),
                    ("EVALMAX", float(lam_eval.max()), "RSRF evaluated to (um)"),
                    ("NSLITCOL", int(np.count_nonzero(spec.slit_cols)), "Slit cols"),
                    ("FILTFILE", filter_file.name, "Filter curve for alpha_fcn"),
                    ("SRCSCALE", float(rsrf_cfg["src_scale_factor"]), "S_perf int."),
                    ("SRCNORM", src_norm, "S_perfect / unit-throughput BB"),
                    ("ALPHAINT", float(rsrf_cfg["alpha_integral_frac"]), "alpha int."),
                    ("ALPHASCL", alpha_scale, "alpha_fcn / filter ripple"),
                    ("MEAS_SIM", measured, "rsrf_error as measured by the SIM"),
                ],
                extra_hdus=[
                    rsrf_table(spec, evaluate, model, pattern_rows, rsrf_error)
                ],
            )
        )

    if ctx.plot_dir:
        fyi_plot_rsrf(
            ctx.plot_dir,
            f"{req_key}_{sim_common.instrument_tag(ctx.instrument)}",
            spec,
            model,
        )
    return written


# requirement key (in the scenario config) -> function writing its simulated data
# (i.e., these are the reqs that can be simulated)
SIMULATORS = {
    "METIS-2594": simulate_metis_2594,
    "METIS-2883": simulate_metis_2883,
    "METIS-3300": simulate_metis_3300,
    "METIS-3291": simulate_metis_3291,
}


@pipeline_stage(
    name="generate_phot_performance_sim",
    depends_on=(
        "simulate_metis_2594",
        "simulate_metis_2883",
        "simulate_metis_3300",
        "simulate_metis_3291",
    ),
    cluster=CLUSTER_PHOT,
    cluster_color=COLOR_PHOT,
    label="Generate photometric-performance sims",
)
def main():
    """
    Generate IMG-RAD-07 photometric performance simulations, one set per requirement,
    as set in the scenario config.
    """
    stem = project_root()
    out_dir = stem + "data/IMG_RAD_07_photometric_performance_simmed_data/"
    plot_dir = sim_common.sim_plot_dir(stem, __file__)

    now = datetime.datetime.now()
    sim_common.start_sim_logging(
        stem, "IMG_RAD_07_phot_logs", "IMG_RAD_07_photometric_performance", now
    )
    logging.info("Simmed file output directory: %s", out_dir)
    logging.info("FYI plot directory: %s", plot_dir)
    logging.warning(
        "! ------- IMG-RAD-07 needs a rotator capability in ScopeSim (not available "
        "yet) to cycle through rotation angles; all frames are simulated at a single, "
        "fixed rotation angle"
    )
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

    sim_common.run_scenario(
        scenario,
        SIMULATORS,
        observing,
        out_dir,
        TEST_PREFIX,
        plot_dir=plot_dir,
    )


if __name__ == "__main__":
    main()
