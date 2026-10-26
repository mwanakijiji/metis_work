"""
Backbone functions for the IMG-OPT-06 out-of-field stray light test

The result object, centroiding, PSF masking, segmentation and noise machinery are
shared with IMG-OPT-04 and live in stray_light_common; this module adds the checks of
METIS-5838 (out-of-field attenuation) and METIS-9522 (artefacts and ghosts), plus the
synthetic scattered light used by the IMG-OPT-06 SIM script.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pipeline_registry import pipeline_stage  # pylint: disable=import-error

from . import stray_light_common as slc

# no IMG-OPT-06 cluster in pipeline_registry yet; move these there when there is one
CLUSTER_OOF = "IMG-OPT-06 Out-of-field stray light"
COLOR_OOF = "#8c564b"

# where IMG-OPT-06 analysis plots are written
# (METIS_MAIT_code_eckhart/results/IMG_06_out_of_field_analysis_results/)
RESULTS_DIR_IMG_06 = str(
    Path(__file__).resolve().parents[2]
    / "results"
    / "IMG_06_out_of_field_analysis_results"
)


@dataclass
class OutOfFieldResult(slc.StrayLightResult):
    """
    Shared stray light result, plus the IMG-OPT-06 out-of-field quantities.
    """

    # from the SIM primary header (for checking the analysis against the injected truth)
    expected: str | None = None  # 'PASS' or 'FAIL' w.r.t. METIS-5838
    attenuation_injected: float | None = None
    ref_peak_header: float | None = None

    # null frame: difference of two independent background frames (same noise as the
    # background-subtracted image, but no signal); None if not available
    null_image: np.ndarray | None = None

    # METIS-5838
    psf_peak: float | None = None
    stray_peak_irradiance: float | None = None
    stray_mean_irradiance: float | None = None
    stray_irradiance_noise: float | None = None  # 1-sigma noise of the smoothed map
    detection_threshold: float | None = None  # detection_nsigma * that noise
    attenuation_measured: float | None = None
    attenuation_is_lower_limit: bool | None = None
    attenuation_req: float | None = None
    verdict_metis_5838: str | None = None  # 'PASS', 'FAIL' or 'INCONCLUSIVE'


def make_out_of_field_scatter(
    shape,
    peak,
    attenuation,
    direction="+x",
    scale_length_frac=0.25,
    pedestal_frac=0.2,
):
    """
    Make a synthetic scattered-light map from a source just outside the FOV.

    The map is an exponential glow falling off from the detector edge on the
    side of the (virtual) out-of-field source, plus a weak uniform pedestal.
    It is normalised so that its mean irradiance is peak / attenuation.

    INPUTS:
    - shape: (ny, nx) detector shape
    - peak: peak irradiance of the in-field reference PSF (ADU)
    - attenuation: attenuation factor to inject (peak / mean scattered irradiance)
    - direction: side of the detector the source sits beyond; '+x', '-x', '+y', '-y'
    - scale_length_frac: e-folding length of the glow, as a fraction of the
      detector width along the falloff direction
    - pedestal_frac: fraction of the total mean irradiance in the uniform pedestal

    OUTPUTS:
    - scatter: 2D array of scattered light (ADU)
    """
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    # distance from the edge nearest the virtual source
    if direction == "+x":
        dist, length = (nx - 1) - xx, nx
    elif direction == "-x":
        dist, length = xx, nx
    elif direction == "+y":
        dist, length = (ny - 1) - yy, ny
    elif direction == "-y":
        dist, length = yy, ny
    else:
        raise ValueError(f"Unknown direction: {direction}")

    glow = np.exp(-dist / (scale_length_frac * length))
    glow /= np.mean(glow)

    shape_norm = (1.0 - pedestal_frac) * glow + pedestal_frac  # mean = 1

    return (peak / attenuation) * shape_norm


def _radius_from_psf_pix(result_obj):
    """2D array of distances (pixels) from the PSF centroid."""
    ny, nx = result_obj.image.shape
    xx, yy = np.meshgrid(
        np.arange(nx) - result_obj.centroids["x_center_pix_fullarray_normsamp"],
        np.arange(ny) - result_obj.centroids["y_center_pix_fullarray_normsamp"],
    )
    return np.sqrt(xx**2 + yy**2)


def _attenuation_verdict(psf_peak, stray_peak, threshold, attenuation_req):
    """
    METIS-5838 verdict from the peak scattered irradiance and the detection threshold.

    - Scattered light detected (stray_peak > threshold): attenuation = psf_peak /
      stray_peak; PASS if >= attenuation_req, else FAIL.
    - Not detected: the attenuation is only known to be > psf_peak / threshold; PASS if
      that lower limit >= attenuation_req, else INCONCLUSIVE (data not deep enough to
      show the requirement either way).

    OUTPUTS:
    - attenuation (float): measured attenuation, or its lower limit
    - is_lower_limit (bool)
    - verdict (str): 'PASS', 'FAIL' or 'INCONCLUSIVE'
    """
    if stray_peak > threshold:
        attenuation = psf_peak / stray_peak
        return attenuation, False, "PASS" if attenuation >= attenuation_req else "FAIL"
    attenuation = psf_peak / threshold if threshold > 0 else np.inf
    return (
        attenuation,
        True,
        "PASS" if attenuation >= attenuation_req else "INCONCLUSIVE",
    )


@pipeline_stage(
    name="metis_5838_attenuation",
    depends_on=("stray_light_mask_real",),
    cluster=CLUSTER_OOF,
    cluster_color=COLOR_OOF,
    label="METIS-5838: out-of-field attenuation",
)
def metis_5838_attenuation(result_obj, observing_config):
    """
    Measure the attenuation of out-of-field irradiance (METIS-5838).

    The attenuation is the ratio of the peak irradiance of the source (the PSF peak) to
    the peak scattered irradiance on the detector. The scattered irradiance is measured
    on a box-car smoothed version of the background-subtracted image (to beat down the
    pixel noise), excluding everything within n_lambda_over_d_exclusion * lambda/D of
    the PSF so that the PSF wings are not mistaken for scattered light.

    The scattered light only counts as detected if its peak exceeds detection_nsigma
    times the noise of the smoothed map. That noise is measured from the null frame
    (difference of two independent background frames, smoothed the same way), so that
    correlated noise is included. If nothing is detected, the attenuation is a lower
    limit (see _attenuation_verdict).

    N.b. in the real test the source is outside the FOV and the PSF peak comes from a
    separate in-field reference exposure; in the simulated data the PSF is in the frame.

    INPUTS:
    - result_obj (OutOfFieldResult): result object, with centroids, real_psf_mask and
      (preferably) null_image
    - observing_config (dict): observing config, incl. the 'metis_5838' block

    OUTPUTS:
    - result_obj (OutOfFieldResult), updated attributes:
        - psf_peak, stray_peak_irradiance, stray_mean_irradiance,
          stray_irradiance_noise, detection_threshold, attenuation_measured,
          attenuation_is_lower_limit, attenuation_req, verdict_metis_5838
    """
    cfg = observing_config["metis_5838"]
    attenuation_req = float(cfg["attenuation_req"])
    n_excl = float(cfg["n_lambda_over_d_exclusion"])
    box_pix = int(cfg["smoothing_box_pix"])
    detection_nsigma = float(cfg["detection_nsigma"])

    image = result_obj.image.astype(float)

    # irradiance of the source: peak of the PSF
    psf_peak = float(np.nanmax(np.where(result_obj.real_psf_mask, image, np.nan)))
    if result_obj.ref_peak_header is not None:
        logging.info(
            "PSF peak: %.4g (measured); %.4g (REF_PEAK in SIM header)",
            psf_peak,
            result_obj.ref_peak_header,
        )

    # exclude the PSF and its wings
    excl_radius_pix = n_excl * slc.lambda_over_d_pix(result_obj, observing_config)
    valid = _radius_from_psf_pix(result_obj) > excl_radius_pix
    logging.info(
        "METIS-5838: excluding r < %.0f lambda/D = %.1f pix around the PSF; "
        "smoothing box %d pix",
        n_excl,
        excl_radius_pix,
        box_pix,
    )

    smoothed = slc.nan_box_mean(image, valid, box_pix)
    stray_peak = float(np.nanmax(smoothed))
    stray_mean = float(np.nanmean(smoothed))

    # noise of the smoothed map, from the null frame smoothed the same way
    if result_obj.null_image is not None:
        smoothed_null = slc.nan_box_mean(
            result_obj.null_image.astype(float), valid, box_pix
        )
        smoothed_noise = slc.robust_std(smoothed_null)
        logging.info(
            "METIS-5838: noise of the smoothed map from the null frame: %.4g counts "
            "(max |smoothed null| = %.4g)",
            smoothed_noise,
            float(np.nanmax(np.abs(smoothed_null))),
        )
    else:
        # no null frame: the spread of the smoothed image itself; any real scattered
        # light inflates it, so the threshold is conservatively high
        smoothed_noise = slc.robust_std(smoothed)
        logging.warning(
            "METIS-5838: no null frame (two independent backgrounds); noise of the "
            "smoothed map estimated from the image itself (%.4g counts), which "
            "overestimates it if there is scattered light",
            smoothed_noise,
        )
    threshold = detection_nsigma * smoothed_noise

    attenuation, is_lower_limit, verdict = _attenuation_verdict(
        psf_peak, stray_peak, threshold, attenuation_req
    )

    result_obj.psf_peak = psf_peak
    result_obj.stray_peak_irradiance = stray_peak
    result_obj.stray_mean_irradiance = stray_mean
    result_obj.stray_irradiance_noise = smoothed_noise
    result_obj.detection_threshold = threshold
    result_obj.attenuation_measured = attenuation
    result_obj.attenuation_is_lower_limit = is_lower_limit
    result_obj.attenuation_req = attenuation_req
    result_obj.verdict_metis_5838 = verdict

    att_str = f"{'> ' if is_lower_limit else ''}{attenuation:.3g}"
    logging.info(
        "METIS-5838: peak (mean) smoothed irradiance %.4g (%.4g) counts; detection "
        "threshold %.0f-sigma = %.4g counts -> scattered light %s",
        stray_peak,
        stray_mean,
        detection_nsigma,
        threshold,
        "NOT detected" if is_lower_limit else "detected",
    )
    logging.info(
        "METIS-5838: attenuation %s (req. >= %.1e) -> %s",
        att_str,
        attenuation_req,
        verdict,
    )
    if verdict == "INCONCLUSIVE":
        logging.info(
            "METIS-5838: the data are not deep enough to show an attenuation of "
            "%.1e (would need the %.0f-sigma threshold <= %.4g counts)",
            attenuation_req,
            detection_nsigma,
            psf_peak / attenuation_req,
        )

    # plot the scattered irradiance relative to the source, with the requirement and
    # the detection threshold
    fits_stem = slc.fits_stem_of(result_obj)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    rel = smoothed / psf_peak
    rel_filled = np.nan_to_num(rel, nan=0.0)
    im = ax.imshow(rel, origin="lower", cmap="viridis")
    fig.colorbar(
        im, ax=ax, fraction=0.046, pad=0.04, label="Smoothed irradiance / PSF peak"
    )
    for level, color, label in (
        (1.0 / attenuation_req, "red", f"1/{attenuation_req:.0e} of PSF peak"),
        (
            threshold / psf_peak,
            "white",
            f"Detection threshold ({detection_nsigma:.0f} sigma)",
        ),
    ):
        if np.nanmax(rel) > level:
            ax.contour(rel_filled, levels=[level], colors=color, linewidths=1.5)
        ax.plot([], [], color=color, label=label)
    ax.add_patch(
        plt.Circle(
            (
                result_obj.centroids["x_center_pix_fullarray_normsamp"],
                result_obj.centroids["y_center_pix_fullarray_normsamp"],
            ),
            excl_radius_pix,
            fill=False,
            color="white",
            linestyle="--",
            label=f"Exclusion: {n_excl:.0f} lambda/D",
        )
    )
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title(
        f"METIS-5838 {verdict}: attenuation {att_str} (req. >= {attenuation_req:.0e})"
        f"\n{fits_stem}",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    plot_file_name = os.path.join(
        slc.results_dir_of(result_obj), f"metis_5838_{fits_stem}.png"
    )
    fig.savefig(plot_file_name)
    logging.info("Saved plot of METIS-5838 scattered irradiance to %s", plot_file_name)
    plt.close(fig)

    return result_obj


@pipeline_stage(
    name="metis_9522_check",
    # either segmentation stage (single method, or weighted combination) feeds this
    depends_on=("stray_light_segmentation", "stray_light_segmentation_combined"),
    cluster=CLUSTER_OOF,
    cluster_color=COLOR_OOF,
    label="METIS-9522: artefacts vs. 3-sigma background",
)
def metis_9522_check(result_obj, observing_config):
    """
    Measure the segmented stray light regions for METIS-9522, as for IMG-OPT-04: each
    region's total flux and its limit, 3-sigma of the background noise of the summed
    flux over the same area (i.e. for the spatial scale of the region). Pass/fail is
    decided in log_test_outcome; the point/extended-source sensitivities are plotted
    FYI only.

    INPUTS:
    - result_obj (OutOfFieldResult): result object, with real_psf_mask and
      segment_map (integer label map of the stray light regions)
    - observing_config (dict): observing config

    OUTPUTS:
    - result_obj (OutOfFieldResult), updated attributes:
        - regions (with metis_9522_limit set), background_rms
    """
    if result_obj.segment_map is None:
        msg = "metis_9522_check missing required attributes"
        logging.error(msg)
        raise ValueError(msg)

    image = result_obj.image.astype(float)

    # flux statistics of each region (within its bounding box)
    result_obj.regions, _ = slc.region_stats(image, result_obj.segment_map)
    if not result_obj.regions:
        logging.info("METIS-9522: no stray light regions detected")
        return result_obj

    # background: everything outside the PSF and the stray light regions
    background_valid = ~(
        (result_obj.segment_map > 0) | result_obj.real_psf_mask.astype(bool)
    )
    result_obj.background_rms = float(np.nanstd(image[background_valid]))
    logging.info(
        "Background rms per pixel (outside PSF and stray regions): %.4g",
        result_obj.background_rms,
    )

    # METIS-9522 limit for each region: 3-sigma of the noise for its area
    noise_cache = {}
    for region_this in result_obj.regions:
        region_this.metis_9522_limit = slc.METIS_9522_NSIGMA * slc.area_noise_sigma(
            image, background_valid, region_this.area_pix, noise_cache
        )

    # FYI only: point/extended-source sensitivities
    (
        point_limit_counts,
        extended_limit_counts_per_pix,
        sensitivity_band,
        is_placeholder,
    ) = slc.metis_9522_sensitivity_limits(result_obj, observing_config)
    lambda_over_d_pix = slc.lambda_over_d_pix(result_obj, observing_config)
    placeholder_note = " [placeholder Jy-to-counts]" if is_placeholder else ""
    max_radius = max(
        [lambda_over_d_pix * 2] + [r.spatial_scale * 1.1 for r in result_obj.regions]
    )

    # brightness spectrum plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i, region in enumerate(result_obj.regions):
        ax.scatter(
            region.spatial_scale,
            region.surface_brightness,
            color=f"C{i % 10}",
            s=150,
            edgecolors="black",
            zorder=3,
            label=f"Region {region.label}" if i < 20 else None,
        )
    ax.set_yscale("log")
    ax.axvline(lambda_over_d_pix, color="red", linestyle="--", label="lambda/D")
    radii_limit = np.geomspace(1.0, max_radius, 30)
    ax.plot(
        radii_limit,
        slc.metis_9522_surface_brightness_limit(
            image, background_valid, radii_limit, noise_cache
        ),
        color="green",
        linestyle="--",
        label="3-sigma of background noise for the region's area (METIS-9522 "
        "pass/fail)",
    )
    radii_point = np.linspace(0.5, lambda_over_d_pix, 100)
    ax.plot(
        radii_point,
        point_limit_counts / (np.pi * radii_point**2),
        color="purple",
        linestyle="--",
        label=f"FYI: point-source sensitivity ({sensitivity_band}){placeholder_note}",
    )
    ax.hlines(
        extended_limit_counts_per_pix,
        xmin=lambda_over_d_pix,
        xmax=max_radius,
        color="purple",
        label=f"FYI: extended-source sensitivity ({sensitivity_band}){placeholder_note}",
    )
    ax.set_xlabel("Radius of equiv. circle (pixels)")
    ax.set_ylabel("Avg. surf. brightness (counts)")
    ax.legend(fontsize=8)
    fits_stem = slc.fits_stem_of(result_obj)
    ax.set_title(f"METIS-9522: stray light fluxes vs. radius\n{fits_stem}", fontsize=10)
    plot_file_name = os.path.join(
        slc.results_dir_of(result_obj),
        f"metis_9522_brightness_spectrum_{fits_stem}.png",
    )
    fig.savefig(plot_file_name)
    logging.info("Saved plot of brightness spectrum to %s", plot_file_name)
    plt.close(fig)

    return result_obj


@pipeline_stage(
    name="img_06_test_outcome",
    depends_on=("metis_5838_attenuation", "metis_9522_check"),
    cluster=CLUSTER_OOF,
    cluster_color=COLOR_OOF,
    label="IMG-OPT-06 pass/fail",
)
def log_test_outcome(result_obj):
    """
    Decide and log the pass/fail of each IMG-OPT-06 requirement, and of the whole test.

    - METIS-5838: the verdict from metis_5838_attenuation (PASS, FAIL or INCONCLUSIVE).
    - METIS-9522: the total flux of every stray light region is below 3-sigma of the
      background noise of the summed flux over its area (StrayLightRegion
      .metis_9522_limit, from metis_9522_check); no regions: PASS.
    Overall: PASS only if both pass; INCONCLUSIVE if METIS-5838 is inconclusive and
    METIS-9522 passes; else FAIL. If the data are simulated, also checks the METIS-5838
    outcome against the injected truth.

    OUTPUTS:
    - result_obj (OutOfFieldResult), with pass_metis_9522 and verdict_overall set
    """
    checks_9522 = []
    for region_this in result_obj.regions:
        ok = region_this.total_flux < region_this.metis_9522_limit
        checks_9522.append(ok)
        logging.info(
            "Stray light region %s: flux %.4g %s 3-sigma of background for its area "
            "(%.4g over %s pix)",
            region_this.label,
            region_this.total_flux,
            "<" if ok else ">=",
            region_this.metis_9522_limit,
            region_this.area_pix,
        )
    result_obj.pass_metis_9522 = all(checks_9522)

    att_str = (
        f"{'> ' if result_obj.attenuation_is_lower_limit else ''}"
        f"{result_obj.attenuation_measured:.3g}"
    )
    logging.info("--------------------------------------------------")
    logging.info("--------------------------------------------------")
    logging.info("File: %s", result_obj.file_absname)
    logging.info(
        "METIS-5838: %s (attenuation %s, req. >= %.1e)",
        result_obj.verdict_metis_5838,
        att_str,
        result_obj.attenuation_req,
    )
    logging.info(
        "METIS-9522: %s (%s of %s regions above 3-sigma for their area)",
        "PASS" if result_obj.pass_metis_9522 else "FAIL",
        checks_9522.count(False),
        len(checks_9522),
    )
    logging.info("(METIS-3114 is verified at observatory level, not here)")
    logging.info("--------------------------------------------------")
    if result_obj.verdict_metis_5838 == "PASS" and result_obj.pass_metis_9522:
        result_obj.verdict_overall = "PASS"
    elif result_obj.verdict_metis_5838 == "INCONCLUSIVE" and result_obj.pass_metis_9522:
        result_obj.verdict_overall = "INCONCLUSIVE"
    else:
        result_obj.verdict_overall = "FAIL"
    logging.info("IMG 06 OUT-OF-FIELD STRAY LIGHT TEST: %s", result_obj.verdict_overall)
    logging.info("--------------------------------------------------")

    if result_obj.expected is not None:
        logging.info(
            "Simulated data: injected attenuation %.1e, expected METIS-5838 %s; "
            "measured %s -> %s",
            result_obj.attenuation_injected,
            result_obj.expected,
            result_obj.verdict_metis_5838,
            (
                "consistent"
                if result_obj.verdict_metis_5838 == result_obj.expected
                else "INCONSISTENT"
            ),
        )
        logging.info("--------------------------------------------------")

    return result_obj
