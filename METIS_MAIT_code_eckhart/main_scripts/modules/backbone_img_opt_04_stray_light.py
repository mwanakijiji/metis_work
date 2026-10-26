"""
Analysis backbone for the IMG-OPT-04 in-field stray light and ghosts test.

The result object, centroiding, PSF masking, segmentation and noise machinery are
shared with IMG-OPT-06 and live in stray_light_common; this module has the IMG-OPT-04
stages run after them:
stray_light_brightness_spectrum -> metis_1189_1429_check and metis_9522_check
-> log_test_outcome,
plus the synthetic artefacts used by the IMG-OPT-04 SIM script.
"""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pipeline_registry import CLUSTER_STRAY, COLOR_STRAY, pipeline_stage

from . import stray_light_common as slc

# where IMG-OPT-04 analysis plots are written
# (METIS_MAIT_code_eckhart/results/IMG_OPT_04_stray_light_analysis_results/)
RESULTS_DIR_IMG_04 = str(
    Path(__file__).resolve().parents[2]
    / "results"
    / "IMG_OPT_04_stray_light_analysis_results"
)

# METIS-1189 and -1429: stray light irradiance must be below this fraction of the
# peak irradiance (the stricter 0.04% of METIS-1429, in the IMG detector planes)
METIS_1189_1429_PEAK_FRAC = 0.0004


def _overplot_method_brightnesses(ax, result_obj, final_labels):
    """
    Overplot surface brightness vs. radius of the area-equivalent circle for each region
    as found by
    each segmentation method (combined segmentation only; no-op otherwise). One hollow
    marker shape
    and colour per method; each point is annotated with its group label, marked if the
    group is not in
    'final_labels' (i.e. rejected by the weighted vote).
    """
    method_label_maps = getattr(result_obj, "method_label_maps", None)
    if not method_label_maps:
        return
    weights = getattr(result_obj, "segmentation_weights", None) or {}
    image = result_obj.image.astype(float)
    markers = ["o", "s", "^", "D", "v", "P"]

    for i, (method, label_map) in enumerate(method_label_maps.items()):
        color = slc.SEGMENTATION_METHOD_COLORS.get(method, "gray")
        marker = markers[i % len(markers)]
        for lab, slices, inside in slc.iter_regions(label_map):
            area = int(inside.sum())
            radius = np.sqrt(area / np.pi)
            surface_brightness = float(np.nansum(image[slices][inside])) / area
            ax.scatter(
                radius,
                surface_brightness,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                s=60,
                linewidths=1.5,
                zorder=4,
            )
            is_final = int(lab) in final_labels
            ax.annotate(
                str(lab) if is_final else f"{lab} (rejected)",
                (radius, surface_brightness),
                textcoords="offset points",
                xytext=(5, -10),
                fontsize=7,
                color=color if is_final else "gray",
            )
        ax.scatter(
            [],
            [],
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=1.5,
            label=f"{method} (w={weights.get(method, float('nan')):g})",
        )


@pipeline_stage(
    name="stray_light_brightness_spectrum",
    # either segmentation stage (single method, or weighted combination) feeds this
    depends_on=("stray_light_segmentation", "stray_light_segmentation_combined"),
    cluster=CLUSTER_STRAY,
    cluster_color=COLOR_STRAY,
    label="Arrange segments into brightnesses as function of spatial scales",
)
def stray_light_brightness_spectrum(result_obj, observing_config):
    """
    Sort out the segments into brightnesses as function of spatial scales, and measure
    the quantities the requirements are tested against (pass/fail itself is decided in
    metis_1189_1429_check and metis_9522_check).

    INPUTS:
    - result_obj: result object, which should contain
        - image (ndarray): science image
        - segment_map (ndarray[int]): integer region labels (0 = not stray light)
    - observing_config: observing config


    OUTPUTS:
    - result_obj (StrayLightResult), updated attributes:
        - regions (list[StrayLightRegion]): flux statistics and METIS-9522 limit of
          each region
        - psf_max (float): peak of the real PSF
        - background_rms (float): per-pixel rms outside the PSF and the regions
    """

    # required attributes are there?
    if (result_obj.image is None) or (result_obj.segment_map is None):
        msg = "stray_light_brightness_spectrum missing required attributes"
        logging.error(msg)
        raise ValueError(msg)

    results_dir = slc.results_dir_of(result_obj)
    fits_stem = slc.fits_stem_of(result_obj)

    # extract the stray light fluxes from inside the regions
    result_obj.regions = []

    # segmentation found no stray light regions: nothing to measure, so return an empty
    # spectrum
    if not np.any(result_obj.segment_map):
        logging.warning(
            "No stray light regions from segmentation; returning an empty brightness "
            "spectrum"
        )
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_xlabel("Radius of equiv. circle (pixels)")
        ax.set_ylabel("Avg. surf. brightness (counts)")
        _overplot_method_brightnesses(ax, result_obj, final_labels=[])
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        ax.set_title("Stray Light Fluxes vs. Radius (no stray light regions found)")
        plot_file_name = os.path.join(
            results_dir, f"stray_light_brightness_spectrum_{fits_stem}.png"
        )
        fig.savefig(plot_file_name)
        logging.info("Saved plot of (empty) brightness spectrum to %s", plot_file_name)
        plt.close(fig)
        return result_obj

    image = result_obj.image.astype(float)

    # flux statistics of each stray light region (within the region's bounding box);
    # keep the cutouts for the regions plot below
    result_obj.regions, region_cutouts = slc.region_stats(image, result_obj.segment_map)

    # mask the PSF and all stray light regions to get the background level
    masked = (result_obj.segment_map > 0) | result_obj.real_psf_mask.astype(bool)
    background_valid = ~masked

    # the background-only image
    background_only = np.where(masked, np.nan, image)

    # the psf-only image
    psf_only = np.where(result_obj.real_psf_mask.astype(bool), image, np.nan)

    # get the stdev of the background (per pixel; FYI)
    stdev_background_only = np.nanstd(background_only)
    result_obj.background_rms = float(stdev_background_only)

    # get the max of the psf
    psf_max = np.nanmax(psf_only)
    result_obj.psf_max = float(psf_max)

    logging.info(
        "Background rms per pixel (outside PSF and stray regions): %.4g",
        stdev_background_only,
    )
    logging.info("PSF peak: %.4g", psf_max)

    # METIS-9522 limit for each region: 3-sigma of the noise of the summed flux over
    # its area, measured from the background pixels (cached per box size, as many
    # regions have similar areas)
    noise_cache = {}
    for region_this in result_obj.regions:
        region_this.metis_9522_limit = slc.METIS_9522_NSIGMA * slc.area_noise_sigma(
            image, background_valid, region_this.area_pix, noise_cache
        )

    # point/extended source sensitivities, for an FYI comparison only
    (
        point_limit_counts,
        extended_limit_counts_per_pix,
        sensitivity_band,
        jy_to_counts_is_placeholder,
    ) = slc.metis_9522_sensitivity_limits(result_obj, observing_config)
    placeholder_note = (
        " [placeholder Jy-to-counts]" if jy_to_counts_is_placeholder else ""
    )

    # dividing line between point-like (i.e., size of PSF) and extended stray light
    lambda_over_d_pix = slc.lambda_over_d_pix(result_obj, observing_config)

    # display the brightness spectrum
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for i, region in enumerate(result_obj.regions):
        # flux vs. radius (colors match the regions plot below)
        plt.scatter(
            region.spatial_scale,
            region.surface_brightness,
            color=f"C{i % 10}",
            s=150,
            edgecolors="black",
            zorder=3,
            label=f"Region {region.label}",
        )
        ax.annotate(
            str(region.label),
            (region.spatial_scale, region.surface_brightness),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=11,
            fontweight="bold",
        )
    # combined segmentation: also each region as found by each method
    _overplot_method_brightnesses(
        ax, result_obj, final_labels=[r.label for r in result_obj.regions]
    )
    # log scale: the METIS-9522 limits can be orders of magnitude below the regions
    plt.yscale("log")
    plt.axvline(
        x=lambda_over_d_pix, color="red", linestyle="--", label="Size scale: lambda/D"
    )
    # METIS-9522 pass/fail: 3-sigma of the noise for the spatial scale of the region
    max_radius = max(
        [lambda_over_d_pix * 2] + [r.spatial_scale * 1.1 for r in result_obj.regions]
    )
    radii_limit = np.geomspace(1.0, max_radius, 30)
    plt.plot(
        radii_limit,
        slc.metis_9522_surface_brightness_limit(
            image, background_valid, radii_limit, noise_cache
        ),
        color="green",
        linestyle="--",
        label="3-sigma of background noise for the region's area (METIS-9522 "
        "pass/fail)",
    )
    # METIS-9522 limits as surface brightness: point-like regions are limited in total
    # flux (so flux / area of the region), extended regions in surface brightness
    radii_point = np.linspace(0.5, lambda_over_d_pix, 100)
    plt.plot(
        radii_point,
        point_limit_counts / (np.pi * radii_point**2),
        color="purple",
        linestyle="--",
        label=(
            f"FYI: point-source sensitivity ({sensitivity_band}), flux / area"
            f"{placeholder_note}"
        ),
    )
    plt.hlines(
        y=extended_limit_counts_per_pix,
        xmin=lambda_over_d_pix,
        xmax=max_radius,
        color="purple",
        linestyle="-",
        label=(
            f"FYI: extended-source sensitivity ({sensitivity_band}){placeholder_note}"
        ),
    )
    plt.axhline(
        y=METIS_1189_1429_PEAK_FRAC * psf_max,
        color="orange",
        linestyle="--",
        label=f"Counts: {100 * METIS_1189_1429_PEAK_FRAC:g}% of PSF peak",
    )
    plt.axhline(y=psf_max, color="blue", linestyle="--", label="Counts: PSF peak")
    plt.xlabel("Radius of equiv. circle (pixels)")
    plt.ylabel("Avg. surf. brightness (counts)")
    plt.legend()
    plt.title("Stray Light Fluxes vs. Radius")
    plot_file_name = os.path.join(
        results_dir, f"stray_light_brightness_spectrum_{fits_stem}.png"
    )
    fig.savefig(plot_file_name)
    logging.info("Saved plot of brightness spectrum to %s", plot_file_name)
    plt.close(fig)

    # display where each region is on the 2D readout (colors match the brightness
    # spectrum)
    vmin, vmax = np.nanpercentile(image, [1.0, 99.5])
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Counts")
    ax.contour(
        result_obj.real_psf_mask.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=1.0,
        linestyles="--",
    )
    ax.plot([], [], color="red", linestyle="--", label="PSF mask")
    for i, (region, (slices, inside)) in enumerate(
        zip(result_obj.regions, region_cutouts)
    ):
        color = f"C{i % 10}"
        slc.contour_region(ax, slices, inside, colors=color, linewidths=1.5)
        # label just above-right of the region, so it doesn't hide the outline
        ax.text(
            slices[1].stop + 9,
            slices[0].stop + 9,
            str(region.label),
            color=color,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax.plot([], [], color=color, label=f"Region {region.label}")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title(
        f"Stray light regions ({result_obj.segmentation_method})\n{fits_stem}",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    regions_plot_file_name = os.path.join(
        results_dir, f"stray_light_regions_{fits_stem}.png"
    )
    fig.savefig(regions_plot_file_name)
    logging.info(
        "Saved plot of stray light region locations to %s", regions_plot_file_name
    )
    plt.close(fig)

    # FYI only (not used for pass/fail): comparison with the point/extended-source
    # sensitivities. Point-like (PSF-sized: equivalent radius <= lambda/D) regions:
    # total flux vs. the point-source sensitivity. Extended (larger) regions: mean
    # surface brightness vs. the extended-source sensitivity.
    logging.info("--------------------------------------------------")
    logging.info(
        "FYI: comparison with METIS-9522 sensitivity limits (not used for pass/fail)"
    )
    if jy_to_counts_is_placeholder:
        logging.info(
            "NOTE: the Jy-to-counts conversions are based on PLACEHOLDER values, so "
            "this comparison is not yet meaningful"
        )
    for region_this in result_obj.regions:
        if region_this.spatial_scale <= lambda_over_d_pix:
            kind, value, limit, unit = (
                "point-like",
                region_this.total_flux,
                point_limit_counts,
                "counts",
            )
        else:
            kind, value, limit, unit = (
                "extended",
                region_this.surface_brightness,
                extended_limit_counts_per_pix,
                "counts/pixel",
            )
        logging.info(
            "  Region %s (%s, r_equiv = %.1f pix vs. lambda/D = %.1f pix): "
            "%.4g %s %s sensitivity limit %.4g %s",
            region_this.label,
            kind,
            region_this.spatial_scale,
            lambda_over_d_pix,
            value,
            unit,
            "<" if value < limit else ">=",
            limit,
            unit,
        )

    return result_obj


@pipeline_stage(
    name="img_04_metis_1189_1429_check",
    depends_on=("stray_light_brightness_spectrum",),
    cluster=CLUSTER_STRAY,
    cluster_color=COLOR_STRAY,
    label="METIS-1189/-1429: stray light vs. 0.04% of PSF peak",
)
def metis_1189_1429_check(result_obj):
    """
    Decide METIS-1189 (stray light < 0.1% of the peak irradiance in the IMG focal
    planes) and METIS-1429 (< 0.04% in the IMG-LM and IMG-N detector planes): the peak
    irradiance of every stray light region is below METIS_1189_1429_PEAK_FRAC (the
    stricter 0.04%) of the PSF peak. No stray light regions: PASS.

    INPUTS:
    - result_obj (StrayLightResult): from stray_light_brightness_spectrum (regions,
      psf_max)

    OUTPUTS:
    - result_obj (StrayLightResult), with pass_metis_1189_1429 set
    """
    checks = []
    for region_this in result_obj.regions:
        limit = METIS_1189_1429_PEAK_FRAC * result_obj.psf_max
        ok = region_this.peak_irradiance < limit
        checks.append(ok)
        logging.info(
            "Stray light region %s: peak irradiance %.4g %s %g%% of PSF peak (%.4g)",
            region_this.label,
            region_this.peak_irradiance,
            "<" if ok else ">=",
            100 * METIS_1189_1429_PEAK_FRAC,
            limit,
        )
    result_obj.pass_metis_1189_1429 = all(checks)
    logging.info(
        "METIS-1189 and -1429 (stray light < %g%% of peak irradiance): %s "
        "(%s of %s regions above)",
        100 * METIS_1189_1429_PEAK_FRAC,
        "PASS" if result_obj.pass_metis_1189_1429 else "FAIL",
        checks.count(False),
        len(checks),
    )
    return result_obj


@pipeline_stage(
    name="img_04_metis_9522_check",
    depends_on=("stray_light_brightness_spectrum",),
    cluster=CLUSTER_STRAY,
    cluster_color=COLOR_STRAY,
    label="METIS-9522: artefacts vs. 3-sigma background",
)
def metis_9522_check(result_obj):
    """
    Decide METIS-9522: the total flux of every stray light region is below 3-sigma of
    the background noise of the summed flux over the region's area (i.e. for its
    spatial scale; StrayLightRegion.metis_9522_limit). No stray light regions: PASS.

    INPUTS:
    - result_obj (StrayLightResult): from stray_light_brightness_spectrum (regions,
      with metis_9522_limit)

    OUTPUTS:
    - result_obj (StrayLightResult), with pass_metis_9522 set
    """
    checks = []
    for region_this in result_obj.regions:
        ok = region_this.total_flux < region_this.metis_9522_limit
        checks.append(ok)
        logging.info(
            "Stray light region %s: flux %.4g %s 3-sigma of background for its area "
            "(%.4g over %s pix)",
            region_this.label,
            region_this.total_flux,
            "<" if ok else ">=",
            region_this.metis_9522_limit,
            region_this.area_pix,
        )
    result_obj.pass_metis_9522 = all(checks)
    logging.info(
        "METIS-9522 (artefact flux < 3-sigma of background for its spatial scale): %s "
        "(%s of %s regions above)",
        "PASS" if result_obj.pass_metis_9522 else "FAIL",
        checks.count(False),
        len(checks),
    )
    return result_obj


@pipeline_stage(
    name="img_04_test_outcome",
    depends_on=("img_04_metis_1189_1429_check", "img_04_metis_9522_check"),
    cluster=CLUSTER_STRAY,
    cluster_color=COLOR_STRAY,
    label="IMG-OPT-04 pass/fail",
)
def log_test_outcome(result_obj):
    """
    Log the outcome of each IMG-OPT-04 requirement (as decided by metis_1189_1429_check
    and metis_9522_check) and of the whole test: PASS if both pass, else FAIL.

    INPUTS:
    - result_obj (StrayLightResult): with pass_metis_1189_1429 and pass_metis_9522 set

    OUTPUTS:
    - result_obj (StrayLightResult), with verdict_overall set
    """
    result_obj.verdict_overall = (
        "PASS"
        if result_obj.pass_metis_1189_1429 and result_obj.pass_metis_9522
        else "FAIL"
    )

    logging.info("--------------------------------------------------")
    logging.info("--------------------------------------------------")
    logging.info("File: %s", result_obj.file_absname)
    if not result_obj.regions:
        logging.info("(no stray light regions detected)")
    logging.info(
        "METIS-1189 and -1429: %s",
        "PASS" if result_obj.pass_metis_1189_1429 else "FAIL",
    )
    logging.info("METIS-9522: %s", "PASS" if result_obj.pass_metis_9522 else "FAIL")
    logging.info("--------------------------------------------------")
    logging.info("IMG 04 STRAY LIGHT TEST: %s", result_obj.verdict_overall)
    logging.info("--------------------------------------------------")

    return result_obj


def make_random_contiguous_stray_light(
    shape,
    n_shapes=(3, 8),
    seed=None,
    pixels_per_shape=(80, 600),
    growth_p=0.65,
    intensity_range=(5.0, 60.0),
    smooth_edges=False,
):
    """
    Generate random contiguous stray-light shapes on a 2D detector frame.

    Parameters
    ----------
    shape : tuple[int, int]
        (ny, nx) detector shape.
    n_shapes : int or tuple[int, int]
        Number of shapes, or (min, max) inclusive.
    seed : int or None
        RNG seed.
    pixels_per_shape : tuple[int, int]
        Target number of pixels per shape (min, max).
    growth_p : float
        Probability to grow from existing frontier pixel vs random frontier pixel.
    intensity_range : tuple[float, float]
        Per-shape constant intensity (ADU) sampled uniformly in this range.
    smooth_edges : bool
        If True, apply a tiny averaging blur to soften jagged boundaries.

    Returns
    -------
    stray : ndarray
        2D array with random contiguous shapes.
    label_map : ndarray[int]
        Integer map of shape IDs (0 background, 1..N shapes).
    """
    rng = np.random.default_rng(seed)
    ny, nx = shape
    stray = np.zeros((ny, nx), dtype=float)
    label_map = np.zeros((ny, nx), dtype=int)

    if isinstance(n_shapes, int):
        n_obj = n_shapes
    else:
        n_obj = int(rng.integers(n_shapes[0], n_shapes[1] + 1))

    # 8-neighborhood
    nbrs = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    occupied = np.zeros((ny, nx), dtype=bool)

    for obj_id in range(1, n_obj + 1):
        target = int(rng.integers(pixels_per_shape[0], pixels_per_shape[1] + 1))
        intensity = float(rng.uniform(*intensity_range))

        # pick seed in free space
        free = np.argwhere(~occupied)
        if free.size == 0:
            break
        sy, sx = free[rng.integers(len(free))]

        pixels = {(int(sy), int(sx))}
        frontier = {(int(sy), int(sx))}

        while len(pixels) < target and frontier:
            # choose growth source
            if rng.random() < growth_p:
                cy, cx = list(frontier)[rng.integers(len(frontier))]
            else:
                cy, cx = list(frontier)[rng.integers(len(frontier))]

            # collect candidate neighbors
            cands = []
            for dy, dx in nbrs:
                yy, xx = cy + dy, cx + dx
                if (
                    0 <= yy < ny
                    and 0 <= xx < nx
                    and not occupied[yy, xx]
                    and (yy, xx) not in pixels
                ):
                    cands.append((yy, xx))

            if not cands:
                frontier.discard((cy, cx))
                continue

            yy, xx = cands[rng.integers(len(cands))]
            pixels.add((yy, xx))
            frontier.add((yy, xx))

            # if source is boxed in now, drop it
            has_free_nbr = any(
                0 <= cy + dy < ny
                and 0 <= cx + dx < nx
                and (not occupied[cy + dy, cx + dx])
                and ((cy + dy, cx + dx) not in pixels)
                for dy, dx in nbrs
            )
            if not has_free_nbr:
                frontier.discard((cy, cx))

        # paint shape
        for yy, xx in pixels:
            stray[yy, xx] += intensity
            label_map[yy, xx] = obj_id
            occupied[yy, xx] = True

    if smooth_edges:
        # small 3x3 mean filter without scipy dependency
        pad = np.pad(stray, 1, mode="edge")
        out = np.zeros_like(stray)
        for j in range(ny):
            for i in range(nx):
                out[j, i] = pad[j : j + 3, i : i + 3].mean()
        stray = out

    return stray, label_map


# crescent shape
def make_crescent(shape, center, width, height, angle, amplitude=0.5):
    """
    Add a crescent shape to the detector array.

    The crescent is the set difference of two disks of equal radius
    (moon-phase geometry).

    Parameters
    ----------
    detector_array : ndarray
        2D image to which the crescent is added (not modified in place).
    center : tuple[float, float]
        (x, y) pixel coordinates of the crescent center.
    width : float
        Outer radius of the crescent in pixels.
    height : float
        Approximate thickness of the bright crescent arc in pixels.
    angle : float
        Orientation of the crescent opening, in radians
        (0 opens toward +x).
    amplitude : float
        Constant intensity added inside the crescent mask.

    Returns
    -------
    ndarray
        Copy of ``detector_array`` with the crescent added.
    """
    out = np.zeros(shape, dtype=float)
    ny, nx = out.shape
    cx, cy = float(center[0]), float(center[1])

    yy, xx = np.indices((ny, nx))
    r_outer = float(width)
    thickness = max(1.0, float(height))

    # Cutting disk: same radius as outer, shifted so the leftover arc
    # has characteristic thickness ~height.
    offset = max(0.0, r_outer - thickness)
    cut_cx = cx + offset * np.cos(angle)
    cut_cy = cy + offset * np.sin(angle)

    in_outer = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_outer**2
    in_cut = (xx - cut_cx) ** 2 + (yy - cut_cy) ** 2 <= r_outer**2
    mask = in_outer & ~in_cut

    out[mask] += amplitude

    return out
