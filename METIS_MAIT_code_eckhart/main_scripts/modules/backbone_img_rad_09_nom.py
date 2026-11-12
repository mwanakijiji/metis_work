"""
IMG-RAD-09 nominal background behaviour: the point-source noise of a long
background exposure (simulated as a WCU flat field), in absolute flux units.

Shared by the SIM (which injects a noise level, so it must be the quantity measured
here) and the ANALYSIS (IMG_RAD_09_nominal_background_behavior_ANALYSIS.py):
- the 1-sigma point-source noise is the robust spread (1.4826 x MAD) of the sums in
  circular apertures (radius in lambda/D) on a grid over the field of the
  background-subtracted frame (ADU per DIT),
- converted to an absolute flux with a counts-to-flux calibration (flux per ADU/s in
  the aperture; a PLACEHOLDER in
  config/config_file_IMG_RAD_09_nominal_background_observing.yaml until the
  calibration is known).

ANALYSIS stages, called in order by the ANALYSIS script: populate_run ->
measure_noise -> sens_10000s_check -> plot_run -> log_test_outcome; log_summary over
all runs. The verdict is noise-aware: the MAD-based noise estimate has a relative
uncertainty of MAD_SIGMA_REL_ERR / sqrt(number of apertures).
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from pipeline_registry import CLUSTER_NOMBKG, COLOR_NOMBKG, pipeline_stage

from . import analysis_common
from .psf_grid_prep import load_fits_data

MAD_TO_SIGMA = 1.4826
# relative standard error of 1.4826 x MAD as a Gaussian sigma estimate, x sqrt(N)
# (asymptotic efficiency ~37% of the sample standard deviation)
MAD_SIGMA_REL_ERR = 1.167


def aperture_radius_px(wavelength_m, diameter_m, pixel_scale_mas, n_lambda_over_d=1.0):
    """Aperture radius (px) of n_lambda_over_d x lambda / D."""
    lambda_over_d_mas = wavelength_m / diameter_m * 180.0 / np.pi * 3600.0e3
    return n_lambda_over_d * lambda_over_d_mas / pixel_scale_mas


def aperture_sums(frame, mask, radius_px, spacing_px):
    """
    Sums of frame in circular apertures of radius_px centred on a square grid with
    spacing spacing_px, keeping only apertures entirely inside mask.

    OUTPUTS:
    - np.ndarray: aperture sums
    - int: number of pixels per aperture
    """
    ny, nx = frame.shape
    r_int = int(np.ceil(radius_px))
    offsets = np.arange(-r_int, r_int + 1)
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    disc = xx**2 + yy**2 <= radius_px**2
    step = max(int(round(spacing_px)), 2 * r_int + 1)
    sums = []
    for yc in range(r_int, ny - r_int, step):
        for xc in range(r_int, nx - r_int, step):
            box = (slice(yc - r_int, yc + r_int + 1), slice(xc - r_int, xc + r_int + 1))
            if np.all(mask[box][disc]):
                sums.append(float(np.sum(frame[box][disc])))
    return np.array(sums), int(disc.sum())


def aperture_noise(frame, mask, radius_px, spacing_px):
    """
    1-sigma noise of an aperture sum: the robust spread 1.4826 x MAD of the aperture
    sums (same units as frame, summed over the aperture).

    OUTPUTS:
    - float: noise
    - int: number of apertures used
    """
    sums, _ = aperture_sums(frame, mask, radius_px, spacing_px)
    if sums.size < 10:
        raise ValueError(f"Only {sums.size} apertures in the field: too few")
    mad = np.median(np.abs(sums - np.median(sums)))
    return float(MAD_TO_SIGMA * mad), int(sums.size)


def noise_to_flux(noise_adu_per_dit, dit_s, flux_per_adu_per_s):
    """
    Aperture noise in ADU per DIT (ScopeSim averages over NDIT) to an absolute flux:
    noise / DIT gives ADU/s, times the counts-to-flux calibration.
    """
    return noise_adu_per_dit / dit_s * flux_per_adu_per_s


def filter_wavelength_m(observing, obs_filter, det_key):
    """Central wavelength (m) of a filter from the common observing config."""
    band = det_key.split("_")[-1]  # img_lm -> lm
    return float(observing[f"monochromatic_observing_filters_{band}"][obs_filter])


# ---------------------------------------------------------------- ANALYSIS


@dataclass
class NoiseRunResult:  # pylint: disable=too-many-instance-attributes
    """One IMG-RAD-09 run (requirement, instrument, case), filled in by the stages."""

    run_name: str
    req_id: str
    instrument: str
    case: str
    file_name: str
    header: Any
    det_key: str
    results_dir: str
    limits: dict
    analysis_cfg: dict
    aperture_radius_px: float = np.nan
    aperture_sums_flux: np.ndarray = None
    measured: dict = field(default_factory=dict)
    sigma: dict = field(default_factory=dict)
    verdicts: dict = field(default_factory=dict)
    verdict_overall: str = "NOT EVALUATED"
    consistent: Any = None


def populate_run(state, observing):
    """
    Set up one run from a data state: its file, primary header, detector (from the
    header's filter: the instrument's obs_filter), and the limits and analysis
    parameters from the IMG-RAD-09 observing config (warning if the header's limit
    differs).

    OUTPUTS:
    - NoiseRunResult
    """
    file_name = state["file_name_abs"]
    header = fits.getheader(file_name, 0)
    req_id = state["req_id"]
    cfg = observing["img_rad_09"]
    limits = dict(cfg["limits"].get(req_id) or {})
    run_name = (
        state.get("run_name") or f"{req_id}_{state['instrument']}_{state['case']}"
    )
    logging.info("Run %s: %s", run_name, os.path.basename(file_name))
    analysis_common.log_injected_truth(header)
    for quantity, header_limit in analysis_common.injected_limits(header).items():
        if quantity in limits and header_limit != limits[quantity]:
            logging.warning(
                "! ------- %s: limit on %s is %s in the SIM header, %s in the config",
                run_name,
                quantity,
                header_limit,
                limits[quantity],
            )
    return NoiseRunResult(
        run_name=run_name,
        req_id=req_id,
        instrument=state["instrument"],
        case=state["case"],
        file_name=file_name,
        header=header,
        det_key=state["det_key"],
        results_dir=state["results_write_dir"],
        limits=limits,
        analysis_cfg=cfg,
    )


def measure_noise(result, observing):
    """
    Measure the 1-sigma point-source noise of the background-subtracted frame
    (BCKGD_SUBTED) in absolute flux units: aperture sums (radius
    aperture_radius_lambda_over_d x lambda/D, on a grid in the field mask of the raw
    frame), robust spread, / DIT x counts-to-flux calibration; and its uncertainty
    MAD_SIGMA_REL_ERR / sqrt(N apertures).

    OUTPUTS:
    - NoiseRunResult, with measured/sigma['noise_flux'] and the aperture sums (flux)
    """
    cfg = result.analysis_cfg
    raw, _ = load_fits_data(result.file_name, hdu_index="RAW_READOUT")
    frame, _ = load_fits_data(result.file_name, hdu_index="BCKGD_SUBTED")
    mask = analysis_common.field_mask(
        raw.astype(float), cfg["mask_frac_of_median"], cfg["mask_erosion_px"]
    )
    del raw
    radius = aperture_radius_px(
        filter_wavelength_m(observing, result.header["FILTER"], result.det_key),
        float(cfg["telescope_diameter_m"]),
        float(observing["pixel_scales"][result.det_key]),
        float(cfg["aperture_radius_lambda_over_d"]),
    )
    spacing = radius * float(cfg["aperture_spacing_radii"])
    dit = float(result.header["DIT"])
    calib = float(cfg["flux_per_adu_per_s"][result.det_key])
    noise_adu, n_apertures = aperture_noise(frame.astype(float), mask, radius, spacing)
    sums, _ = aperture_sums(frame.astype(float), mask, radius, spacing)
    noise = noise_to_flux(noise_adu, dit, calib)

    result.aperture_radius_px = radius
    result.aperture_sums_flux = noise_to_flux(sums, dit, calib)
    result.measured["noise_flux"] = noise
    result.sigma["noise_flux"] = noise * MAD_SIGMA_REL_ERR / np.sqrt(n_apertures)
    result.measured["n_apertures"] = n_apertures
    logging.info(
        "%s: aperture radius %.2f px (%s lambda/D), %s apertures, DIT %.4g s, "
        "EXPTIME %s s: noise %.6g ADU/DIT -> %.6g +/- %.3g %s (calibration %s per "
        "ADU/s%s)",
        result.run_name,
        radius,
        cfg["aperture_radius_lambda_over_d"],
        n_apertures,
        dit,
        result.header.get("EXPTIME"),
        noise_adu,
        noise,
        result.sigma["noise_flux"],
        cfg["flux_unit"],
        calib,
        ", PLACEHOLDER" if cfg.get("flux_calibration_is_placeholder") else "",
    )
    return result


@pipeline_stage(
    name="img_rad_09_sens_10000s_check",
    depends_on=("generate_nominal_background_sim",),
    cluster=CLUSTER_NOMBKG,
    cluster_color=COLOR_NOMBKG,
    label="Point-source sensitivity in 10000 s: noise vs. limit",
)
def sens_10000s_check(result):
    """
    Decide the point-source sensitivity requirement in a 10000 s exposure: the
    1-sigma point-source noise (flux units) must not exceed the limit, with n_sigma of
    the uncertainty of the noise estimate (verdict_with_noise). Warns if the exposure
    is not 10000 s or the limit or calibration is a placeholder.

    OUTPUTS:
    - NoiseRunResult, with verdicts[req_id]
    """
    cfg = result.analysis_cfg
    exptime = float(result.header.get("EXPTIME", np.nan))
    required = float(result.limits["exptime_s"])
    if not abs(exptime - required) <= 0.01 * required:
        logging.warning(
            "! ------- %s: exposure %.6g s, not the required %.6g s",
            result.run_name,
            exptime,
            required,
        )
    limit = float(result.limits["noise_flux"])
    value, sigma = result.measured["noise_flux"], result.sigma["noise_flux"]
    verdict = analysis_common.verdict_with_noise(
        value, sigma, limit, float(cfg["n_sigma"])
    )
    result.verdicts[result.req_id] = verdict
    logging.info(
        "%s (1-sigma point-source noise <= %s %s in %.6g s): %.6g +/- %.3g -> %s%s",
        result.req_id,
        limit,
        cfg["flux_unit"],
        exptime,
        value,
        sigma,
        verdict,
        (
            " (PLACEHOLDER limit/calibration)"
            if result.limits.get("is_placeholder")
            or cfg.get("flux_calibration_is_placeholder")
            else ""
        ),
    )
    return result


@pipeline_stage(
    name="img_rad_09_fyi_plots",
    depends_on=("img_rad_09_sens_10000s_check",),
    cluster=CLUSTER_NOMBKG,
    cluster_color=COLOR_NOMBKG,
    fyi=True,  # not needed for the pass/fail
    label="FYI plots",
)
def plot_run(result):
    """
    FYI plot in result.results_dir: histogram of the aperture sums (flux units) with
    the Gaussian of the measured noise and +/- the limit.

    OUTPUTS:
    - str: written plot file
    """
    os.makedirs(result.results_dir, exist_ok=True)
    sums = result.aperture_sums_flux - np.median(result.aperture_sums_flux)
    noise = result.measured["noise_flux"]
    limit = float(result.limits["noise_flux"])
    fig, ax = plt.subplots(figsize=(7, 4))
    edges = np.linspace(-5 * max(noise, limit), 5 * max(noise, limit), 81)
    ax.hist(sums, bins=edges, density=True, histtype="step", label="aperture sums")
    xs = np.linspace(edges[0], edges[-1], 400)
    ax.plot(
        xs,
        np.exp(-0.5 * (xs / noise) ** 2) / (noise * np.sqrt(2 * np.pi)),
        label=f"measured sigma {noise:.4g}",
    )
    for sign in (-1, 1):
        ax.axvline(
            sign * limit, color="tab:red", ls="--", label="limit" if sign > 0 else None
        )
    ax.set_xlabel(f"aperture sum - median ({result.analysis_cfg['flux_unit']})")
    ax.set_ylabel("density")
    ax.set_title(
        f"{result.run_name}: point-source noise ({result.verdicts.get(result.req_id)})"
    )
    ax.legend(fontsize=8)
    file_name = os.path.join(result.results_dir, f"{result.run_name}_noise.png")
    fig.savefig(file_name, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", file_name)
    return file_name


@pipeline_stage(
    name="img_rad_09_test_outcome",
    depends_on=("img_rad_09_sens_10000s_check",),
    cluster=CLUSTER_NOMBKG,
    cluster_color=COLOR_NOMBKG,
    label="IMG-RAD-09 pass/fail",
)
def log_test_outcome(result):
    """
    Log the outcome of the run's requirement and compare it with the injected truth
    (EXPECTED).

    OUTPUTS:
    - NoiseRunResult, with verdict_overall and consistent set
    """
    logging.info("--------------------------------------------------")
    logging.info("--------------------------------------------------")
    logging.info(
        "Run %s: %s, instrument %s, case %s",
        result.run_name,
        result.req_id,
        result.instrument,
        result.case,
    )
    logging.info(
        "  noise_flux = %.6g (sigma %.3g; limit %s; %s apertures of %.2f px)",
        result.measured["noise_flux"],
        result.sigma["noise_flux"],
        result.limits.get("noise_flux"),
        result.measured["n_apertures"],
        result.aperture_radius_px,
    )
    result.verdict_overall = result.verdicts.get(result.req_id, "NOT EVALUATED")
    result.consistent = analysis_common.compare_with_expected(
        result.req_id, result.verdict_overall, result.header
    )
    logging.info("IMG RAD 09 %s: %s", result.req_id, result.verdict_overall)
    logging.info("--------------------------------------------------")
    return result


def log_summary(results):
    """
    Log a table of all runs (requirement, instrument, case, noise, verdict, expected,
    consistency with the injected truth).

    OUTPUTS:
    - int: number of runs inconsistent with the injected truth
    """
    logging.info("==================================================")
    logging.info("IMG-RAD-09 SUMMARY")
    logging.info(
        "%-12s %-10s %-6s %-12s %-13s %-9s %s",
        "requirement",
        "instrument",
        "case",
        "noise",
        "verdict",
        "expected",
        "consistent",
    )
    for res in results:
        logging.info(
            "%-12s %-10s %-6s %-12.5g %-13s %-9s %s",
            res.req_id,
            res.instrument,
            res.case,
            res.measured.get("noise_flux", np.nan),
            res.verdict_overall,
            res.header.get("EXPECTED", "-"),
            {True: "yes", False: "NO", None: "-"}[res.consistent],
        )
    n_bad = sum(1 for res in results if res.consistent is False)
    logging.info(
        "%s of %s runs inconsistent with the injected truth", n_bad, len(results)
    )
    logging.info("==================================================")
    return n_bad
