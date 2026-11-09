"""
IMG-RAD-07 photometric performance: analysis of the WCU flat fields for
- METIS-2883: spatial homogeneity of the flat (+/- 3% peak-to-valley over the FOV),
- METIS-3300: the flat-field flux is tunable to 15 discrete values, known at each step
  to 0.3%, with the illumination pattern constant within 0.5% after tuning.

Stages (called in order by IMG_RAD_07_photometric_performance_ANALYSIS.py): populate_run
-> reduce_flats -> metis_2883_check / metis_3300_check -> plots -> log_test_outcome; and
log_summary over all runs. Everything is measured on the background-subtracted frames
(BCKGD_SUBTED), within a field mask (the illuminated detector area without the
reference-pixel border), averaged over square blocks so that the per-pixel noise does
not dominate the max over the field. Verdicts are noise-aware
(analysis_common.verdict_with_noise).
"""

import glob
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from pipeline_registry import CLUSTER_PHOT, COLOR_PHOT, pipeline_stage
from scipy import ndimage

from . import analysis_common
from .psf_grid_prep import load_fits_data

REQUIREMENTS = ("METIS-2594", "METIS-2883", "METIS-3300")  # METIS-2594: stub


@dataclass
class PhotRunResult:  # pylint: disable=too-many-instance-attributes
    """One IMG-RAD-07 run (requirement, instrument, case), filled in by the stages."""

    run_name: str
    req_id: str
    instrument: str
    case: str
    file_names: list
    headers: list
    results_dir: str
    limits: dict
    analysis_cfg: dict
    exposure: np.ndarray = None
    flux: np.ndarray = None
    flux_sigma: np.ndarray = None
    block_means: list = field(default_factory=list)
    block_sigmas: list = field(default_factory=list)
    measured: dict = field(default_factory=dict)
    sigma: dict = field(default_factory=dict)
    verdicts: dict = field(default_factory=dict)
    verdict_overall: str = "NOT EVALUATED"
    consistent: Any = None


# ---------------------------------------------------------------- pure helpers


def field_mask(frame, frac=0.5, erosion_px=4):
    """
    Illuminated field: pixels above frac x the median of the frame (drops the
    reference-pixel border and anything unilluminated), eroded by erosion_px to avoid
    the edge (inter-pixel capacitance leaks into the border).
    """
    mask = frame > frac * np.median(frame)
    if erosion_px > 0:
        mask = ndimage.binary_erosion(mask, iterations=int(erosion_px))
    return mask


def block_stats(frame, mask, block_px, min_fill=0.5):
    """
    Mean and standard error of the mean of the masked pixels of frame in square blocks
    of block_px; NaN for blocks with less than min_fill of their pixels in the mask
    (incomplete blocks at the frame edge are dropped).

    OUTPUTS:
    - np.ndarray: block means
    - np.ndarray: block standard errors (std / sqrt(n))
    """
    b = int(block_px)
    ny, nx = frame.shape[0] // b, frame.shape[1] // b
    cube = np.where(mask, frame.astype(float), np.nan)[: ny * b, : nx * b]
    cube = cube.reshape(ny, b, nx, b)
    num = np.sum(np.isfinite(cube), axis=(1, 3))
    good = num >= min_fill * b * b
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN (unmasked) blocks
        mean = np.nanmean(cube, axis=(1, 3))
        sem = np.nanstd(cube, axis=(1, 3)) / np.sqrt(np.maximum(num, 1))
    return np.where(good, mean, np.nan), np.where(good, sem, np.nan)


def half_ptv(block_mean, block_sigma):
    """
    Half peak-to-valley inhomogeneity max |B / mean(B) - 1| over the valid blocks, and
    its noise sigma (the relative standard error of the block where the max is).

    OUTPUTS:
    - float: max |B / mean - 1|
    - float: sigma
    - np.ndarray: map of B / mean - 1
    """
    rel = block_mean / np.nanmean(block_mean) - 1.0
    idx = np.nanargmax(np.abs(rel))
    sigma = (block_sigma / np.nanmean(block_mean)).flat[idx]
    return float(np.abs(rel).flat[idx]), float(sigma), rel


def flux_errors(flux, flux_sigma, exposure):
    """
    Relative flux knowledge over the steps: the flux per unit exposure (T_ND x DIT) of
    each step relative to its mean over the steps, minus 1 (the absolute flux scale is
    not known, so it is taken from the data).

    OUTPUTS:
    - np.ndarray: per-step errors
    - np.ndarray: per-step sigmas
    """
    flux = np.asarray(flux, dtype=float)
    rate = flux / np.asarray(exposure, dtype=float)
    scale = np.mean(rate)
    return rate / scale - 1.0, (rate / scale) * np.asarray(flux_sigma) / flux


def pattern_changes(block_means, block_sigmas):
    """
    Change of the illumination pattern over the steps: each step's normalised block map
    (B / mean(B)) divided by the step-averaged normalised map, minus 1.

    OUTPUTS:
    - float: max |change| over steps and blocks
    - float: its sigma (from the block standard errors of that step and of the average)
    - int: index of the step with the max
    - np.ndarray: per-step change maps (n_steps, ny, nx)
    """
    means = np.asarray(block_means, dtype=float)
    sigmas = np.asarray(block_sigmas, dtype=float)
    norm_mean = np.nanmean(means, axis=(1, 2))[:, None, None]
    norm = means / norm_mean
    rel_sig = sigmas / norm_mean
    avg = np.mean(norm, axis=0)
    avg_sig = np.sqrt(np.sum(rel_sig**2, axis=0)) / len(norm)
    change = norm / avg - 1.0
    change_sig = np.sqrt(rel_sig**2 + avg_sig**2) / avg
    idx = np.nanargmax(np.abs(change))
    step = int(np.unravel_index(idx, change.shape)[0])
    return float(np.abs(change).flat[idx]), float(change_sig.flat[idx]), step, change


# ---------------------------------------------------------------- stages


def populate_run(state, observing):
    """
    Set up one run from a data state: its files (file_name_abs, or the
    file_name_pattern glob of the METIS-3300 steps, sorted by the STEP header card),
    their primary headers, and the limits and analysis parameters from the IMG-RAD-07
    observing config (warning if a header's limit differs).

    OUTPUTS:
    - PhotRunResult
    """
    if state.get("file_name_pattern"):
        files = sorted(glob.glob(state["file_name_pattern"]))
    else:
        files = [state["file_name_abs"]]
    if not files:
        raise FileNotFoundError(
            f"No files for run {state.get('run_name')}: "
            f"{state.get('file_name_pattern') or state.get('file_name_abs')}"
        )
    headers = [fits.getheader(f, 0) for f in files]
    order = np.argsort([h.get("STEP", 0) for h in headers])
    files = [files[i] for i in order]
    headers = [headers[i] for i in order]

    req_id = state["req_id"]
    cfg = observing["img_rad_07"]
    limits = dict(cfg["limits"].get(req_id) or {})
    run_name = (
        state.get("run_name") or f"{req_id}_{state['instrument']}_{state['case']}"
    )
    logging.info("Run %s: %s file(s)", run_name, len(files))
    analysis_common.log_injected_truth(headers[0])
    for quantity, header_limit in analysis_common.injected_limits(headers[0]).items():
        if quantity in limits and header_limit != limits[quantity]:
            logging.warning(
                "! ------- %s: limit on %s is %s in the SIM header, %s in the config",
                run_name,
                quantity,
                header_limit,
                limits[quantity],
            )
    return PhotRunResult(
        run_name=run_name,
        req_id=req_id,
        instrument=state["instrument"],
        case=state["case"],
        file_names=files,
        headers=headers,
        results_dir=state["results_write_dir"],
        limits=limits,
        analysis_cfg=cfg,
    )


def reduce_flats(result):
    """
    Reduce the run's flats (BCKGD_SUBTED) one at a time: the field mask from the
    brightest (last) file, then per file the mean flux in the mask and its standard
    error, and the block mean / standard error maps.

    OUTPUTS:
    - PhotRunResult, with exposure, flux, flux_sigma, block_means, block_sigmas
    """
    cfg = result.analysis_cfg
    last, _ = load_fits_data(result.file_names[-1], hdu_index="BCKGD_SUBTED")
    mask = field_mask(
        last.astype(float), cfg["mask_frac_of_median"], cfg["mask_erosion_px"]
    )
    logging.info(
        "%s: field mask %.1f%% of the frame", result.run_name, 100 * mask.mean()
    )
    del last

    flux, flux_sigma = [], []
    for file_name in result.file_names:
        frame, _ = load_fits_data(file_name, hdu_index="BCKGD_SUBTED")
        frame = frame.astype(float)
        values = frame[mask]
        flux.append(float(values.mean()))
        flux_sigma.append(float(values.std() / np.sqrt(values.size)))
        mean, sem = block_stats(frame, mask, cfg["block_px"], cfg["min_block_fill"])
        result.block_means.append(mean)
        result.block_sigmas.append(sem)
    result.flux = np.array(flux)
    result.flux_sigma = np.array(flux_sigma)
    result.exposure = np.array([h.get("EXPOSURE", np.nan) for h in result.headers])
    return result


@pipeline_stage(
    name="img_rad_07_metis_2594_check",
    depends_on=("generate_phot_performance_sim",),
    cluster=CLUSTER_PHOT,
    cluster_color=COLOR_PHOT,
    planned=True,  # stub: drawn dashed in the pipeline flow diagram
    label="METIS-2594: calibration flux range (stub)",
)
def metis_2594_check(result):
    """
    Stub for METIS-2594 (METIS internal calibration shall provide a range of flux
    levels matching that encountered in scientific operation as defined in
    METIS-3137): not evaluated yet (its SIM is a stub and writes no data).

    OUTPUTS:
    - PhotRunResult, with verdicts['METIS-2594'] = 'NOT EVALUATED'
    """
    result.verdicts["METIS-2594"] = "NOT EVALUATED"
    logging.info("METIS-2594 (calibration flux range): stub, NOT EVALUATED")
    return result


@pipeline_stage(
    name="img_rad_07_metis_2883_check",
    depends_on=("generate_phot_performance_sim",),
    cluster=CLUSTER_PHOT,
    cluster_color=COLOR_PHOT,
    label="METIS-2883: flat homogeneity",
)
def metis_2883_check(result):
    """
    Decide METIS-2883 (illumination homogeneous within +/- 3% peak-to-valley over the
    FOV): half_ptv = max |B / mean - 1| of the block-averaged flat must not exceed the
    limit, with n_sigma of the block noise (verdict_with_noise).

    OUTPUTS:
    - PhotRunResult, with measured/sigma['half_ptv'] and verdicts['METIS-2883']
    """
    value, sigma, _ = half_ptv(result.block_means[0], result.block_sigmas[0])
    limit = float(result.limits["half_ptv"])
    n_sigma = float(result.analysis_cfg["n_sigma"])
    verdict = analysis_common.verdict_with_noise(value, sigma, limit, n_sigma)
    result.measured["half_ptv"], result.sigma["half_ptv"] = value, sigma
    result.verdicts["METIS-2883"] = verdict
    logging.info(
        "METIS-2883 (max |B/mean - 1| <= %.3f, %s px blocks): %.4f +/- %.5f -> %s",
        limit,
        result.analysis_cfg["block_px"],
        value,
        sigma,
        verdict,
    )
    return result


@pipeline_stage(
    name="img_rad_07_metis_3300_check",
    depends_on=("generate_phot_performance_sim",),
    cluster=CLUSTER_PHOT,
    cluster_color=COLOR_PHOT,
    label="METIS-3300: 15 flux steps, flux and pattern",
)
def metis_3300_check(result):
    """
    Decide METIS-3300 from the run's flux steps:
    - n_steps: the number of distinct, increasing exposure levels (T_ND x DIT) must be
      the required number (15), else FAIL,
    - flux_error: max over the steps of |flux per exposure / mean - 1| <= limit,
    - pattern_change: max over the steps and blocks of |normalised map / step-averaged
      normalised map - 1| <= limit (goal logged too),
    both with n_sigma of their noise. METIS-3300 is FAIL if any part fails, PASS if all
    pass, else INCONCLUSIVE.

    OUTPUTS:
    - PhotRunResult, with measured/sigma and verdicts['METIS-3300'] (and the parts)
    """
    limits = result.limits
    n_sigma = float(result.analysis_cfg["n_sigma"])

    n_levels = len(np.unique(result.exposure))
    increasing = bool(np.all(np.diff(result.exposure) > 0))
    steps_ok = n_levels == int(limits["n_steps"]) and increasing
    result.measured["n_steps"] = n_levels
    result.verdicts["METIS-3300 n_steps"] = "PASS" if steps_ok else "FAIL"
    logging.info(
        "METIS-3300 flux steps: %s distinct levels (required %s), increasing: %s -> %s",
        n_levels,
        limits["n_steps"],
        increasing,
        result.verdicts["METIS-3300 n_steps"],
    )

    errors, error_sigmas = flux_errors(result.flux, result.flux_sigma, result.exposure)
    idx = int(np.argmax(np.abs(errors)))
    value, sigma = float(abs(errors[idx])), float(error_sigmas[idx])
    result.measured["flux_error"], result.sigma["flux_error"] = value, sigma
    result.measured["flux_errors"] = errors
    result.verdicts["METIS-3300 flux_error"] = analysis_common.verdict_with_noise(
        value, sigma, float(limits["flux_error"]), n_sigma
    )
    logging.info(
        "METIS-3300 flux knowledge (max |flux/nominal - 1| <= %.4f): %.5f +/- %.6f "
        "(step %s) -> %s",
        limits["flux_error"],
        value,
        sigma,
        idx + 1,
        result.verdicts["METIS-3300 flux_error"],
    )

    value, sigma, step, change = pattern_changes(
        result.block_means, result.block_sigmas
    )
    result.measured["pattern_change"], result.sigma["pattern_change"] = value, sigma
    result.measured["pattern_change_maps"] = change
    result.measured["pattern_change_step"] = step + 1
    result.verdicts["METIS-3300 pattern_change"] = analysis_common.verdict_with_noise(
        value, sigma, float(limits["pattern_change"]), n_sigma
    )
    logging.info(
        "METIS-3300 pattern constancy (max |change| <= %.4f; goal %.4f): %.5f +/- "
        "%.6f (step %s) -> %s (goal: %s)",
        limits["pattern_change"],
        limits["pattern_change_goal"],
        value,
        sigma,
        step + 1,
        result.verdicts["METIS-3300 pattern_change"],
        analysis_common.verdict_with_noise(
            value, sigma, float(limits["pattern_change_goal"]), n_sigma
        ),
    )

    result.verdicts["METIS-3300"] = analysis_common.combine_verdicts(
        result.verdicts[f"METIS-3300 {part}"]
        for part in ("n_steps", "flux_error", "pattern_change")
    )
    logging.info("METIS-3300: %s", result.verdicts["METIS-3300"])
    return result


@pipeline_stage(
    name="img_rad_07_fyi_plots",
    depends_on=("img_rad_07_metis_2883_check", "img_rad_07_metis_3300_check"),
    cluster=CLUSTER_PHOT,
    cluster_color=COLOR_PHOT,
    fyi=True,  # not needed for the pass/fail
    label="FYI plots",
)
def plot_run(result):
    """
    FYI plots in result.results_dir: METIS-2883 the B/mean - 1 block map; METIS-3300
    the flux error per step (with the limit band) and the pattern-change map of the
    worst step.

    OUTPUTS:
    - list[str]: written plot files
    """
    os.makedirs(result.results_dir, exist_ok=True)
    written = []
    if result.req_id == "METIS-2883":
        _, _, rel = half_ptv(result.block_means[0], result.block_sigmas[0])
        written.append(
            _plot_map(
                rel,
                f"{result.run_name}: B/mean - 1 "
                f"(max |.| {result.measured['half_ptv']:.4f}, "
                f"limit {result.limits['half_ptv']})",
                os.path.join(result.results_dir, f"{result.run_name}_homogeneity.png"),
            )
        )
    elif result.req_id == "METIS-3300":
        fig, ax = plt.subplots(figsize=(7, 4))
        steps = np.arange(1, len(result.flux) + 1)
        limit = float(result.limits["flux_error"])
        ax.axhspan(-limit, limit, color="tab:green", alpha=0.2, label="limit")
        ax.errorbar(steps, result.measured["flux_errors"], fmt="o", label="measured")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("flux step")
        ax.set_ylabel("flux / nominal - 1")
        ax.set_title(f"{result.run_name}: flux knowledge")
        ax.legend(loc="center right")
        file_name = os.path.join(result.results_dir, f"{result.run_name}_flux.png")
        fig.savefig(file_name, bbox_inches="tight")
        plt.close(fig)
        written.append(file_name)
        step = result.measured["pattern_change_step"]
        written.append(
            _plot_map(
                result.measured["pattern_change_maps"][step - 1],
                f"{result.run_name}: pattern change, step {step} "
                f"(max |.| {result.measured['pattern_change']:.4f}, "
                f"limit {result.limits['pattern_change']})",
                os.path.join(result.results_dir, f"{result.run_name}_pattern.png"),
            )
        )
    for file_name in written:
        logging.info("Saved %s", file_name)
    return written


def _plot_map(rel_map, title, file_name):
    """Diverging image of a relative-deviation block map."""
    vmax = float(np.nanmax(np.abs(rel_map))) or 1e-6
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(rel_map, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(image, ax=ax)
    ax.set_xlabel("block x")
    ax.set_ylabel("block y")
    ax.set_title(title, fontsize=9)
    fig.savefig(file_name, bbox_inches="tight")
    plt.close(fig)
    return file_name


@pipeline_stage(
    name="img_rad_07_log_test_outcome",
    depends_on=(
        "img_rad_07_metis_2594_check",
        "img_rad_07_metis_2883_check",
        "img_rad_07_metis_3300_check",
    ),
    cluster=CLUSTER_PHOT,
    cluster_color=COLOR_PHOT,
    label="IMG-RAD-07 pass/fail",
)
def log_test_outcome(result):
    """
    Log the outcome of the run's requirement (as decided by metis_2883_check or
    metis_3300_check) and compare it with the injected truth (EXPECTED).

    OUTPUTS:
    - PhotRunResult, with verdict_overall and consistent set
    """
    logging.info("--------------------------------------------------")
    logging.info("--------------------------------------------------")
    logging.info(
        "Run %s: %s, instrument %s, case %s, %s file(s)",
        result.run_name,
        result.req_id,
        result.instrument,
        result.case,
        len(result.file_names),
    )
    for quantity in ("half_ptv", "n_steps", "flux_error", "pattern_change"):
        if quantity in result.measured:
            logging.info(
                "  %s = %s (sigma %s; limit %s)",
                quantity,
                result.measured[quantity],
                result.sigma.get(quantity, "-"),
                result.limits.get(quantity, "-"),
            )
    result.verdict_overall = result.verdicts.get(result.req_id, "NOT EVALUATED")
    result.consistent = analysis_common.compare_with_expected(
        result.req_id, result.verdict_overall, result.headers[0]
    )
    logging.info("IMG RAD 07 %s: %s", result.req_id, result.verdict_overall)
    logging.info("--------------------------------------------------")
    return result


def log_summary(results):
    """
    Log a table of all runs (requirement, instrument, case, verdict, expected,
    consistency with the injected truth); METIS-2594 is NOT EVALUATED (no SIM data).

    OUTPUTS:
    - int: number of runs inconsistent with the injected truth
    """
    logging.info("==================================================")
    logging.info("IMG-RAD-07 SUMMARY")
    logging.info(
        "%-12s %-10s %-6s %-13s %-9s %s",
        "requirement",
        "instrument",
        "case",
        "verdict",
        "expected",
        "consistent",
    )
    for res in results:
        logging.info(
            "%-12s %-10s %-6s %-13s %-9s %s",
            res.req_id,
            res.instrument,
            res.case,
            res.verdict_overall,
            res.headers[0].get("EXPECTED", "-"),
            {True: "yes", False: "NO", None: "-"}[res.consistent],
        )
    logging.info("METIS-2594: NOT EVALUATED (SIM is a stub)")
    n_bad = sum(1 for res in results if res.consistent is False)
    logging.info(
        "%s of %s runs inconsistent with the injected truth", n_bad, len(results)
    )
    logging.info("==================================================")
    return n_bad
