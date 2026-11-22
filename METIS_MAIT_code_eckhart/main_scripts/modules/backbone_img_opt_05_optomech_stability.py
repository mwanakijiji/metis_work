"""
IMG-OPT-05 optomechanical stability: analysis of the pupil-image sequences for
- METIS-5422: the position of the mask in CFO-FP2 stable w.r.t. the IMG detectors,
  as RMS on sky over a timescale (0.2 mas IMG-LM, 0.5 mas IMG-N, on 10 s - 10 min;
  goal up to 1 h), tested with pupil imaging.
METIS-1230 (flux-calibration interval) is not analysed (its SIM is a stub).

Stages (called in order by IMG_OPT_05_optomechanical_stability_ANALYSIS.py):
populate_run -> measure_positions -> metis_5422_check -> plot_run -> log_test_outcome;
and log_summary over all runs. One run is one sequence of frames (one timescale,
instrument and case), read from the background-subtracted frames (BCKGD_SUBTED).

The pupil centre of each frame (pupil_centre) is the centroid of the pupil's coverage
map, clip(frame / pupil level, 0, 1), in a circular window around the pupil, iterated
from a threshold-based first guess. Clipping makes it insensitive to the illumination
level inside the pupil and to outliers; the window is re-centred at each iteration.
Its noise sigma is propagated from the per-pixel noise (MAD inside and outside the
pupil), counting every window pixel (conservative). The sequence's RMS motion is the
2D RMS of the centres about their mean, debiased for the measurement noise, converted
to mas on sky with the detector's pixel scale (pixel_scales in the common observing
config); the frame times come from DATE-OBS. Verdicts are noise-aware
(analysis_common.verdict_with_noise).
"""

import glob
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.time import Time
from pipeline_registry import CLUSTER_OPTOMECH, COLOR_OPTOMECH, pipeline_stage

from . import analysis_common
from .psf_grid_prep import load_fits_data

REQUIREMENTS = ("METIS-5422",)  # METIS-1230: SIM is a stub


@dataclass
class StabilityRunResult:  # pylint: disable=too-many-instance-attributes
    """One IMG-OPT-05 run (requirement, instrument, timescale, case): a sequence."""

    run_name: str
    req_id: str
    instrument: str
    case: str
    timescale_s: float
    det_key: str
    pix_scale: float  # mas / px
    file_names: list
    headers: list
    results_dir: str
    limits: dict
    analysis_cfg: dict
    times: np.ndarray = None  # s since the first frame (DATE-OBS)
    centres: np.ndarray = None  # (n, 2) measured pupil centres (x, y) [px]
    centre_sigmas: np.ndarray = None  # (n, 2) their noise sigmas [px]
    radii: np.ndarray = None  # (n,) measured pupil radii [px]
    measured: dict = field(default_factory=dict)
    sigma: dict = field(default_factory=dict)
    verdicts: dict = field(default_factory=dict)
    verdict_overall: str = "NOT EVALUATED"
    consistent: Any = None


# ---------------------------------------------------------------- pure helpers


def detector_of_filter(observing, filter_name):
    """'img_lm' or 'img_n': the detector whose filter list has filter_name."""
    for det_key, band in (("img_lm", "lm"), ("img_n", "n")):
        if filter_name in (
            observing.get(f"polychromatic_observing_filters_{band}_leaf_name") or {}
        ):
            return det_key
    raise ValueError(f"Filter {filter_name!r} is in neither the LM nor the N list")


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def _window_centroid(frame, level, centre, radius, sigma_in, sigma_out):
    """
    Centroid of the coverage map clip(frame / level, 0, 1) in the circle of radius
    radius around centre, and its noise sigma (x, y), propagated from a per-pixel
    noise sigma_out + (sigma_in - sigma_out) x coverage (in variance) over all window
    pixels.
    """
    ny, nx = frame.shape
    x0, x1 = max(int(centre[0] - radius) - 1, 0), min(int(centre[0] + radius) + 2, nx)
    y0, y1 = max(int(centre[1] - radius) - 1, 0), min(int(centre[1] + radius) + 2, ny)
    yy, xx = np.meshgrid(
        np.arange(y0, y1, dtype=float), np.arange(x0, x1, dtype=float), indexing="ij"
    )
    window = np.hypot(xx - centre[0], yy - centre[1]) <= radius
    cover = np.where(window, np.clip(frame[y0:y1, x0:x1] / level, 0.0, 1.0), 0.0)
    total = cover.sum()
    cx, cy = (xx * cover).sum() / total, (yy * cover).sum() / total
    var = np.where(
        window, (sigma_out**2 + (sigma_in**2 - sigma_out**2) * cover) / level**2, 0.0
    )
    sig_x = np.sqrt(((xx - cx) ** 2 * var).sum()) / total
    sig_y = np.sqrt(((yy - cy) ** 2 * var).sum()) / total
    return np.array([cx, cy]), np.array([sig_x, sig_y])


def pupil_centre(  # pylint: disable=too-many-locals
    frame, margin_px=20.0, inner_frac=0.8, n_iter=3
):
    """
    Centre of a sharp pupil image in a background-subtracted frame.

    1. First guess: pixels above half the 99th percentile; their centroid, and the
       radius from their area (R = sqrt(N / pi)).
    2. Pupil level: median inside inner_frac x R; per-pixel noise: 1.4826 x MAD
       inside (sigma_in) and in an annulus just outside the pupil (sigma_out).
    3. n_iter times: centroid of clip(frame / level, 0, 1) in the circle of radius
       R + margin_px around the current centre (_window_centroid).

    OUTPUTS:
    - ndarray: centre (x, y) [px, 0-based pixel centres]
    - ndarray: its noise sigma (x, y) [px]
    - float: radius [px]
    - float: pupil level [ADU]
    """
    frame = np.asarray(frame, dtype=float)
    bright = frame > 0.5 * np.percentile(frame, 99)
    yy, xx = np.nonzero(bright)
    centre = np.array([xx.mean(), yy.mean()])
    radius = float(np.sqrt(bright.sum() / np.pi))

    ys, xs = np.indices(frame.shape)
    dist = np.hypot(xs - centre[0], ys - centre[1])
    inner = frame[dist < inner_frac * radius]
    outer = frame[(dist > radius + margin_px) & (dist < radius + 3 * margin_px)]
    level = float(np.median(inner))

    def robust_sigma(values):
        return float(1.4826 * np.median(np.abs(values - np.median(values))))

    sigma_in, sigma_out = robust_sigma(inner), robust_sigma(outer)
    sigma = np.full(2, np.nan)
    for _ in range(int(n_iter)):
        centre, sigma = _window_centroid(
            frame, level, centre, radius + margin_px, sigma_in, sigma_out
        )
    return centre, sigma, radius, level


def rms_motion(centres, sigmas):
    """
    2D RMS of positions about their mean, and its noise:
    - raw: sqrt(mean(|c_k - <c>|^2)),
    - debiased: sqrt(max(raw^2 - (n-1)/n x mean(sx^2 + sy^2), 0)) (the noise adds
      (n-1)/n of its variance to the mean squared offset),
    - sigma: propagated from the per-frame sigmas,
      sqrt(sum_k (dx_k^2 sx_k^2 + dy_k^2 sy_k^2)) / (n raw), but at least
      sqrt(mean(sx^2 + sy^2) / n) (when the motion is within the noise, the
      propagation underestimates it).

    OUTPUTS:
    - float: debiased RMS [same unit as the input]
    - float: its sigma
    - float: raw RMS
    - ndarray: (n, 2) offsets from the mean
    """
    centres = np.asarray(centres, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    n = len(centres)
    offsets = centres - centres.mean(axis=0)
    raw = float(np.sqrt(np.mean(np.sum(offsets**2, axis=1))))
    noise_var = float(np.mean(np.sum(sigmas**2, axis=1)))
    debiased = float(np.sqrt(max(raw**2 - (n - 1) / n * noise_var, 0.0)))
    floor = float(np.sqrt(noise_var / n))
    prop = (
        float(np.sqrt(np.sum(offsets**2 * sigmas**2)) / (n * raw)) if raw > 0 else 0.0
    )
    return debiased, max(prop, floor), raw, offsets


def frame_times(headers):
    """Seconds since the first frame, from the DATE-OBS cards."""
    times = Time([h["DATE-OBS"] for h in headers], format="isot", scale="utc")
    return (times - times[0]).sec


# ---------------------------------------------------------------- stages


def populate_run(state, observing):
    """
    Set up one run from a data state: its sequence of files (the file_name_pattern
    glob), sorted by DATE-OBS, their primary headers, the detector (from the FILTER
    card), its pixel scale (pixel_scales in the common observing config; warning if
    the header's PIXSCALE differs) and the limits and analysis parameters from the
    IMG-OPT-05 observing config (warning if a header's limit differs).

    OUTPUTS:
    - StabilityRunResult
    """
    files = sorted(glob.glob(state["file_name_pattern"]))
    if not files:
        raise FileNotFoundError(
            f"No files for run {state.get('run_name')}: {state['file_name_pattern']}"
        )
    headers = [fits.getheader(f, 0) for f in files]
    order = np.argsort(Time([h["DATE-OBS"] for h in headers]).mjd)
    files = [files[i] for i in order]
    headers = [headers[i] for i in order]

    req_id = state["req_id"]
    cfg = observing["img_opt_05"]
    det_key = detector_of_filter(observing, headers[0]["FILTER"])
    pix_scale = float(observing["pixel_scales"][det_key])
    limits = dict((cfg["limits"].get(req_id) or {}).get(det_key) or {})
    timescale = float(state["timescale_s"])
    run_name = state.get("run_name") or (
        f"{req_id}_{state['instrument']}_{timescale:g}s_{state['case']}"
    )
    logging.info(
        "Run %s: %s file(s), %s, pixel scale %.3f mas/px",
        run_name,
        len(files),
        det_key,
        pix_scale,
    )
    analysis_common.log_injected_truth(headers[0])
    if "PIXSCALE" in headers[0] and float(headers[0]["PIXSCALE"]) != pix_scale:
        logging.warning(
            "! ------- %s: PIXSCALE %s in the header, %s in the config",
            run_name,
            headers[0]["PIXSCALE"],
            pix_scale,
        )
    for quantity, header_limit in analysis_common.injected_limits(headers[0]).items():
        if quantity in limits and header_limit != limits[quantity]:
            logging.warning(
                "! ------- %s: limit on %s is %s in the SIM header, %s in the config",
                run_name,
                quantity,
                header_limit,
                limits[quantity],
            )
    return StabilityRunResult(
        run_name=run_name,
        req_id=req_id,
        instrument=state["instrument"],
        case=state["case"],
        timescale_s=timescale,
        det_key=det_key,
        pix_scale=pix_scale,
        file_names=files,
        headers=headers,
        results_dir=state["results_write_dir"],
        limits=limits,
        analysis_cfg=cfg,
    )


@pipeline_stage(
    name="img_opt_05_measure_pupil_positions",
    depends_on=("generate_optomechanical_stability_sim",),
    cluster=CLUSTER_OPTOMECH,
    cluster_color=COLOR_OPTOMECH,
    label="Pupil-image centre per frame",
)
def measure_positions(result):
    """
    Measure the pupil centre (pupil_centre) of each frame of the run (BCKGD_SUBTED),
    one frame at a time, and the frame times (DATE-OBS). Warns if the sequence's span
    differs from the run's timescale by more than span_tolerance.

    OUTPUTS:
    - StabilityRunResult, with times, centres, centre_sigmas, radii
    """
    cfg = result.analysis_cfg["pupil"]
    centres, sigmas, radii = [], [], []
    for file_name in result.file_names:
        frame, _ = load_fits_data(file_name, hdu_index="BCKGD_SUBTED")
        centre, sigma, radius, level = pupil_centre(
            frame, cfg["margin_px"], cfg["inner_frac"], cfg["n_iter"]
        )
        del frame
        centres.append(centre)
        sigmas.append(sigma)
        radii.append(radius)
        logging.debug(
            "%s: centre (%.4f, %.4f) +/- (%.4f, %.4f) px, R %.1f px, level %.4g",
            os.path.basename(file_name),
            centre[0],
            centre[1],
            sigma[0],
            sigma[1],
            radius,
            level,
        )
    result.centres = np.array(centres)
    result.centre_sigmas = np.array(sigmas)
    result.radii = np.array(radii)
    result.times = frame_times(result.headers)

    span = float(result.times[-1])
    tol = float(result.analysis_cfg["span_tolerance"])
    if abs(span - result.timescale_s) > tol * result.timescale_s:
        logging.warning(
            "! ------- %s: frames span %.4g s, the timescale tested is %.4g s",
            result.run_name,
            span,
            result.timescale_s,
        )
    logging.info(
        "%s: %s frames over %.4g s; pupil radius %.1f px; median centre noise "
        "(%.4f, %.4f) px",
        result.run_name,
        len(result.file_names),
        span,
        float(np.median(result.radii)),
        float(np.median(result.centre_sigmas[:, 0])),
        float(np.median(result.centre_sigmas[:, 1])),
    )
    return result


@pipeline_stage(
    name="img_opt_05_metis_5422_check",
    depends_on=("img_opt_05_measure_pupil_positions",),
    cluster=CLUSTER_OPTOMECH,
    cluster_color=COLOR_OPTOMECH,
    label="METIS-5422: RMS motion per timescale",
)
def metis_5422_check(result):
    """
    Decide METIS-5422 for the run's timescale: the 2D RMS of the pupil centres about
    their mean, debiased for the measurement noise (rms_motion), in mas on sky (x the
    pixel scale), must not exceed the detector's limit, with n_sigma of its noise
    (verdict_with_noise). Timescales above requirement_max_s are tested against the
    goal (same limit); this is logged. For simulated data, the measured centres are
    also compared with the injected offsets (SHIFT_X/Y cards), for information.

    OUTPUTS:
    - StabilityRunResult, with measured/sigma['position_rms'] and
      verdicts['METIS-5422']
    """
    rms_px, sigma_px, raw_px, offsets = rms_motion(result.centres, result.centre_sigmas)
    value, sigma = rms_px * result.pix_scale, sigma_px * result.pix_scale
    limit = float(result.limits["position_rms"])
    n_sigma = float(result.analysis_cfg["n_sigma"])
    verdict = analysis_common.verdict_with_noise(value, sigma, limit, n_sigma)
    limtype = (
        "requirement"
        if result.timescale_s <= float(result.analysis_cfg["requirement_max_s"])
        else "goal"
    )
    result.measured["position_rms"], result.sigma["position_rms"] = value, sigma
    result.measured["position_rms_raw"] = raw_px * result.pix_scale
    result.measured["offsets_mas"] = offsets * result.pix_scale
    result.measured["limtype"] = limtype
    result.verdicts["METIS-5422"] = verdict
    logging.info(
        "METIS-5422 (%s, %s, timescale %.4g s, %s): RMS motion %.4f +/- %.4f mas "
        "(raw %.4f) vs. limit %.3f mas -> %s",
        result.instrument,
        result.det_key,
        result.timescale_s,
        limtype,
        value,
        sigma,
        result.measured["position_rms_raw"],
        limit,
        verdict,
    )

    if all("SHIFT_X" in h and "SHIFT_Y" in h for h in result.headers):
        true = np.array([[h["SHIFT_X"], h["SHIFT_Y"]] for h in result.headers])
        true = true - true.mean(axis=0)
        error = np.hypot(*(offsets - true).T)
        result.measured["offset_error_max_px"] = float(error.max())
        logging.info(
            "  vs. injected offsets: max |measured - injected| %.4f px (%.4f mas), "
            "rms %.4f px",
            float(error.max()),
            float(error.max()) * result.pix_scale,
            float(np.sqrt(np.mean(error**2))),
        )
    return result


@pipeline_stage(
    name="img_opt_05_fyi_plots",
    depends_on=("img_opt_05_metis_5422_check",),
    cluster=CLUSTER_OPTOMECH,
    cluster_color=COLOR_OPTOMECH,
    fyi=True,  # not needed for the pass/fail
    label="FYI plots",
)
def plot_run(result):
    """
    FYI plot in result.results_dir: the measured pupil offsets (mas on sky) vs. time,
    with their noise, and the injected offsets (SHIFT_X/Y) if there; and the offsets
    in the x-y plane with the limit as a circle.

    OUTPUTS:
    - list[str]: written plot files
    """
    os.makedirs(result.results_dir, exist_ok=True)
    offsets = result.measured["offsets_mas"]
    sig = result.centre_sigmas * result.pix_scale
    fig, (ax_t, ax_xy) = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, (axis, marker) in enumerate((("x", "o"), ("y", "s"))):
        ax_t.errorbar(
            result.times,
            offsets[:, i],
            yerr=sig[:, i],
            fmt=marker,
            label=f"{axis} measured",
        )
    if all("SHIFT_X" in h for h in result.headers):
        true = np.array([[h["SHIFT_X"], h["SHIFT_Y"]] for h in result.headers])
        true = (true - true.mean(axis=0)) * result.pix_scale
        ax_t.plot(result.times, true[:, 0], "k+", label="x injected")
        ax_t.plot(result.times, true[:, 1], "kx", label="y injected")
    ax_t.axhline(0.0, color="k", lw=0.5)
    ax_t.set_xlabel("time since first frame [s]")
    ax_t.set_ylabel("offset from mean position [mas on sky]")
    ax_t.legend(fontsize=8)
    limit = float(result.limits["position_rms"])
    ax_xy.plot(offsets[:, 0], offsets[:, 1], "o-")
    ax_xy.add_patch(plt.Circle((0, 0), limit, fill=False, ls="--", label="limit (RMS)"))
    rms = result.measured["position_rms"]
    ax_xy.add_patch(plt.Circle((0, 0), rms, fill=False, color="tab:red", label="RMS"))
    lim = 1.3 * max(limit, float(np.max(np.abs(offsets))))
    ax_xy.set_xlim(-lim, lim)
    ax_xy.set_ylim(-lim, lim)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x offset [mas]")
    ax_xy.set_ylabel("y offset [mas]")
    ax_xy.legend(fontsize=8)
    fig.suptitle(
        f"{result.run_name}: RMS {rms:.4f} +/- {result.sigma['position_rms']:.4f} mas "
        f"(limit {limit:g}, {result.measured['limtype']}) -> "
        f"{result.verdicts['METIS-5422']}",
        fontsize=10,
    )
    fig.tight_layout()
    file_name = os.path.join(result.results_dir, f"{result.run_name}_motion.png")
    fig.savefig(file_name, bbox_inches="tight")
    plt.close(fig)
    logging.info("Saved %s", file_name)
    return [file_name]


@pipeline_stage(
    name="img_opt_05_log_test_outcome",
    depends_on=("img_opt_05_metis_5422_check",),
    cluster=CLUSTER_OPTOMECH,
    cluster_color=COLOR_OPTOMECH,
    label="IMG-OPT-05 pass/fail",
)
def log_test_outcome(result):
    """
    Log the outcome of the run's requirement (as decided by metis_5422_check) and
    compare it with the injected truth (EXPECTED).

    OUTPUTS:
    - StabilityRunResult, with verdict_overall and consistent set
    """
    logging.info("--------------------------------------------------")
    logging.info("--------------------------------------------------")
    logging.info(
        "Run %s: %s, instrument %s, timescale %.4g s, case %s, %s frame(s)",
        result.run_name,
        result.req_id,
        result.instrument,
        result.timescale_s,
        result.case,
        len(result.file_names),
    )
    for quantity in ("position_rms", "position_rms_raw", "offset_error_max_px"):
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
    logging.info("IMG OPT 05 %s: %s", result.req_id, result.verdict_overall)
    logging.info("--------------------------------------------------")
    return result


def log_summary(results):
    """
    Log a table of all runs (requirement, instrument, timescale, case, RMS, verdict,
    expected, consistency with the injected truth); METIS-1230 is NOT EVALUATED (no
    SIM data).

    OUTPUTS:
    - int: number of runs inconsistent with the injected truth
    """
    logging.info("==================================================")
    logging.info("IMG-OPT-05 SUMMARY")
    logging.info(
        "%-11s %-6s %-8s %-5s %-17s %-12s %-9s %s",
        "requirement",
        "instr.",
        "timesc.",
        "case",
        "RMS [mas]",
        "verdict",
        "expected",
        "consistent",
    )
    for res in results:
        rms = res.measured.get("position_rms", np.nan)
        logging.info(
            "%-11s %-6s %-8s %-5s %-17s %-12s %-9s %s",
            res.req_id,
            res.instrument,
            f"{res.timescale_s:g}s",
            res.case,
            f"{rms:.4f} +/- {res.sigma.get('position_rms', np.nan):.4f}",
            res.verdict_overall,
            res.headers[0].get("EXPECTED", "-"),
            {True: "yes", False: "NO", None: "-"}[res.consistent],
        )
    logging.info("METIS-1230: NOT EVALUATED (SIM is a stub)")
    n_bad = sum(1 for res in results if res.consistent is False)
    logging.info(
        "%s of %s runs inconsistent with the injected truth", n_bad, len(results)
    )
    logging.info("==================================================")
    return n_bad
