"""
Stray-light machinery shared by the IMG-OPT-04 (in-field stray light) and IMG-OPT-06
(out-of-field stray light) analyses: the result object, centroiding, masking of the real
PSF, segmentation of the remaining light, and the noise per spatial scale used for
METIS-9522.

Shared stages, called in this order by both ANALYSIS scripts:
populate_result_obj_info -> centroid_2passes_oversample -> stray_light_mask_real
-> stray_light_segmentation(_combined)
after which each test runs its own stages (backbone_img_04_stray_light,
backbone_img_06_out_of_field).
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from astropy.io import fits
from photutils.centroids import centroid_2dg, centroid_sources
from pipeline_registry import pipeline_stage  # pylint: disable=import-error
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import active_contour

from . import helpers, psf_grid_prep

# flow-chart cluster for the stages shared by IMG-OPT-04 and -06 (no '/' in the label:
# it becomes part of a Graphviz subgraph name)
CLUSTER_STRAY_SHARED = "Stray-light common (IMG-OPT-04 and -06)"
COLOR_STRAY_SHARED = "#7f7f7f"

# where shared-stage plots go if the result object does not set results_dir
# (METIS_MAIT_code_eckhart/results/stray_light_common_results/)
RESULTS_DIR_STRAY_COMMON = str(
    Path(__file__).resolve().parents[2] / "results" / "stray_light_common_results"
)

# METIS-9522: artefacts must contain less flux than this many sigma of the background
# noise, for the spatial scale of the artefact
METIS_9522_NSIGMA = 3.0


# class for containing information about a stray light region
@dataclass
class StrayLightRegion:
    """Flux statistics of one stray light region."""

    label: int
    spatial_scale: str  # e.g. "point", "extended", "large"
    peak_irradiance: float
    total_flux: float
    area_pix: int
    surface_brightness: float
    # METIS-9522 limit on total_flux: 3-sigma of the background noise of the summed
    # flux over area_pix
    metis_9522_limit: float | None = None
    # optional: bbox, centroid, ...


# class for containing information about a stray light result from a single FITS file
@dataclass
class StrayLightResult:
    """
    Image, masks, segmentation and region statistics for one stray light FITS file.
    """

    # identity
    file_absname: str
    filter_name: str
    detector: str
    image: np.ndarray

    # where plots and FITS products of this file are written (set by each test's
    # ANALYSIS script; RESULTS_DIR_STRAY_COMMON if None)
    results_dir: str | None = None

    # derived from observing config
    wavel_central: float | None = None  # meters
    pixel_scale: float | None = None  # mas/pixel

    # images
    # mask of the real PSF
    real_psf_mask: np.ndarray | None = None
    # one integer label map of the stray light regions (0 = not stray light; each
    # region has its own label > 0); see iter_regions to loop over the regions
    segment_map: np.ndarray | None = None

    # segmentation choice (see stray_light_segmentation); can be set per data state
    segmentation_method: str | None = None
    segmentation_params: dict | None = None
    # combined segmentation (see stray_light_segmentation_combined)
    segmentation_weights: dict | None = None  # weight in [0,1] per method
    segmentation_vote_fraction: float | None = (
        None  # N: pixel is in if summed weight > N * sum(all weights)
    )
    method_label_maps: dict | None = (
        None  # per method: cross-method group labels of its detections
    )
    final_label_map: np.ndarray | None = (
        None  # group labels of the weighted-vote regions
    )

    # bookkeeping
    centroids: Any | None = None  # from centroid_2passes_oversample
    regions: list[StrayLightRegion] = field(default_factory=list)

    # global quantities
    background_level: float | None = None
    background_rms: float | None = None
    psf_max: float | None = None  # peak of the real PSF

    # verdicts (from each test's log_test_outcome)
    pass_metis_1189_1429: bool | None = None
    pass_metis_9522: bool | None = None
    verdict_overall: str | None = None


def results_dir_of(result_obj):
    """Results directory of a result object (created if needed)."""
    results_dir = getattr(result_obj, "results_dir", None) or RESULTS_DIR_STRAY_COMMON
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def fits_stem_of(result_obj):
    """Input FITS file name without directory and extension (for output file names)."""
    return os.path.splitext(os.path.basename(result_obj.file_absname))[0]


def iter_regions(label_map):
    """
    Loop over the regions of an integer label map without making a full-frame mask per
    region.

    INPUTS:
    - label_map (ndarray[int]): 0 = no region; each region has its own label > 0

    OUTPUTS (yielded, one per region present):
    - label (int): label of the region
    - slices (tuple[slice, slice]): bounding box of the region in label_map
    - inside (ndarray[bool]): mask of the region within that bounding box
    """
    for i, slices in enumerate(ndimage.find_objects(label_map)):
        if slices is None:  # label not present
            continue
        label = i + 1
        yield label, slices, label_map[slices] == label


def region_stats(image, label_map):
    """
    Flux statistics of each region of a label map (within its bounding box).

    OUTPUTS:
    - regions (list[StrayLightRegion])
    - cutouts (list[tuple[slices, inside]]): bounding box and in-box mask of each region,
      in the same order (e.g. for plotting with contour_region)
    """
    regions = []
    cutouts = []
    for lab, slices, inside in iter_regions(label_map):
        vals = image[slices][inside]  # the stray light pixels of this region
        area = int(inside.sum())
        regions.append(
            StrayLightRegion(
                label=int(lab),  # >0 to avoid confusion with non-stray-light regions
                spatial_scale=np.sqrt(area / np.pi),  # radius of area-equiv. circle
                peak_irradiance=float(np.nanmax(vals)),  # max irradiance in the region
                total_flux=float(np.nansum(vals)),  # total flux in the region
                area_pix=area,  # area in pixels
                surface_brightness=float(np.nansum(vals)) / area,  # mean per pixel
            )
        )
        cutouts.append((slices, inside))
    return regions, cutouts


def contour_region(ax, slices, inside, **contour_kwargs):
    """
    Outline one region on a full-frame plot, from its bounding-box cutout (padded by 1
    pixel so the contour closes at the edge of the box).
    """
    padded = np.pad(inside.astype(float), 1)
    ys = np.arange(slices[0].start - 1, slices[0].stop + 1)
    xs = np.arange(slices[1].start - 1, slices[1].stop + 1)
    ax.contour(xs, ys, padded, levels=[0.5], **contour_kwargs)


def nan_box_mean(image, valid, box_pix, min_valid_frac=0.5):
    """
    Box-car mean of an image using only the 'valid' (and finite) pixels; NaN where
    fewer than min_valid_frac of the pixels in the box are valid.
    """
    valid = valid & np.isfinite(image)
    num = ndimage.uniform_filter(
        np.where(valid, image, 0.0), size=box_pix, mode="nearest"
    )
    den = ndimage.uniform_filter(valid.astype(float), size=box_pix, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = num / den
    mean[den < min_valid_frac] = np.nan
    return mean


def robust_std(values):
    """
    Standard deviation from the median absolute deviation (Gaussian-equivalent),
    ignoring non-finite values.
    """
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


def area_noise_sigma(image, valid, area_pix, cache=None, min_samples=20):
    """
    1-sigma noise of the flux summed over area_pix pixels, measured from the image's own
    background pixels: means over (almost) fully valid boxes of side round(sqrt(area)),
    sampled on a grid with that spacing (so the samples are roughly independent). This
    includes correlated noise (e.g. from inter-pixel capacitance), unlike
    sigma_pix * sqrt(area).

    INPUTS:
    - image (ndarray): background-subtracted image
    - valid (ndarray[bool]): background pixels (i.e. not PSF, not stray light)
    - area_pix (float): area over which the flux is summed (pixels)
    - cache (dict or None): per box side, the noise of the box mean (re-used across
      calls for the same image and mask)
    - min_samples (int): below this many boxes, fall back to sigma_pix * sqrt(area)

    OUTPUTS:
    - sigma_sum (float): 1-sigma noise of the summed flux (counts)
    """
    side = max(1, int(round(np.sqrt(area_pix))))
    if cache is not None and side in cache:
        return cache[side] * area_pix

    valid = valid & np.isfinite(image)
    if side == 1:
        samples = image[valid]
    else:
        means = nan_box_mean(image, valid, side, min_valid_frac=0.9)
        samples = means[side // 2 :: side, side // 2 :: side]
        samples = samples[np.isfinite(samples)]

    if samples.size >= min_samples:
        sigma_mean = robust_std(samples)
    else:
        # too few independent boxes of this size: assume white noise
        sigma_mean = robust_std(image[valid]) / side
        logging.warning(
            "Only %s background boxes of side %s pix; assuming white noise for the "
            "noise of a %.0f-pixel area",
            samples.size,
            side,
            area_pix,
        )

    if cache is not None:
        cache[side] = sigma_mean
    return sigma_mean * area_pix


def metis_9522_surface_brightness_limit(image, valid, radii_pix, cache=None):
    """
    METIS-9522 pass/fail limit as a mean surface brightness (counts/pixel), for regions
    with the area of a circle of each radius in radii_pix: METIS_9522_NSIGMA * (noise of
    the summed flux over that area) / area.
    """
    areas = np.pi * np.asarray(radii_pix, dtype=float) ** 2
    return np.array(
        [
            METIS_9522_NSIGMA * area_noise_sigma(image, valid, area, cache) / area
            for area in areas
        ]
    )


def metis_9522_sensitivity_limits(result_obj, observing_config):
    """
    METIS-9522 sensitivity limits in counts, from the sensitivities in the observing
    config (for an FYI comparison; pass/fail uses the measured background).

    OUTPUTS:
    - point_limit_counts (float): total flux limit of a point-like (PSF-sized) region
    - extended_limit_counts_per_pix (float): mean surface brightness limit of an
      extended region, per pixel
    - band (str): sensitivity band used (e.g. 'lprime')
    - is_placeholder (bool): whether the Jy-to-counts conversion is a placeholder
    """
    scale_key = observing_config["scope_sim_to_analysis"][result_obj.detector]
    band = observing_config["sensitivity_band"][scale_key]
    counts_per_jy = float(observing_config["jy_to_counts"][scale_key])
    is_placeholder = bool(observing_config.get("jy_to_counts_is_placeholder", False))
    if is_placeholder:
        logging.warning(
            "Jy-to-counts conversion (%s counts/Jy for %s) is a PLACEHOLDER",
            counts_per_jy,
            scale_key,
        )

    # solid angle seen by one pixel
    pixel_solid_angle_arcsec2 = (result_obj.pixel_scale / 1000.0) ** 2

    point_limit_jy = float(observing_config["mode_sensitivity_point"][band])
    extended_limit_jy_per_arcsec2 = float(
        observing_config["mode_sensitivity_extended"][band]
    )
    point_limit_counts = point_limit_jy * counts_per_jy
    extended_limit_counts_per_pix = (
        extended_limit_jy_per_arcsec2 * pixel_solid_angle_arcsec2 * counts_per_jy
    )
    logging.info(
        "METIS-9522 sensitivity limits, FYI (%s): point source %.4g Jy = %.4g "
        "counts; extended source %.4g Jy/arcsec**2 = %.4g counts/pixel (%.4g arcsec**2/pixel)",
        band,
        point_limit_jy,
        point_limit_counts,
        extended_limit_jy_per_arcsec2,
        extended_limit_counts_per_pix,
        pixel_solid_angle_arcsec2,
    )
    return point_limit_counts, extended_limit_counts_per_pix, band, is_placeholder


def lambda_over_d_pix(result_obj, observing_config):
    """lambda/D in pixels, for the central wavelength of the filter."""
    lambda_over_d_asec = (
        206265.0
        * float(result_obj.wavel_central)
        / float(observing_config["D_aperture"]["full"])
    )
    return lambda_over_d_asec / (result_obj.pixel_scale / 1000.0)


@pipeline_stage(
    name="populate_result_obj_info",
    depends_on=("generate_stray_light_sim", "generate_out_of_field_sim"),
    cluster=CLUSTER_STRAY_SHARED,
    cluster_color=COLOR_STRAY_SHARED,
    label="Populate result object",
)
def populate_result_obj_info(result_obj, data_state, observing_config):
    """
    Populate the result object with information from the data state.

    INPUTS:
    - result_obj (StrayLightResult): result object
    - data_state (dict): data state
    - observing_config (ObservingConfig): observing config

    OUTPUTS:
    - result_obj (StrayLightResult): result object
    """

    # add the things from the data state
    for key, value in data_state.items():
        setattr(result_obj, key, value)

    # detector label (LM/N) → analysis key (img_lm/img_n)
    detector_to_scale_key = observing_config["scope_sim_to_analysis"]
    try:
        scale_key = detector_to_scale_key[result_obj.detector]
    except KeyError as exc:
        raise KeyError(
            f"No ScopeSim→analysis mapping for detector {result_obj.detector!r} "
            f"in scope_sim_to_analysis"
        ) from exc

    # add the central wavelength for the filter, from the table for this band (img_lm →
    # _lm, img_n → _n)
    filters_key = "monochromatic_observing_filters_" + scale_key.removeprefix("img_")
    try:
        result_obj.wavel_central = float(
            observing_config[filters_key][result_obj.filter_name]
        )  # in m
    except KeyError as exc:
        raise KeyError(
            f"No central wavelength for filter {result_obj.filter_name!r} "
            f"in {filters_key}"
        ) from exc

    # pixel scale: analysis key → mas/pixel
    try:
        result_obj.pixel_scale = float(observing_config["pixel_scales"][scale_key])
    except KeyError as exc:
        raise KeyError(f"No pixel scale for key {scale_key!r} in pixel_scales") from exc

    return result_obj


@pipeline_stage(
    name="centroid_2passes_oversample",
    depends_on=("populate_result_obj_info",),
    cluster=CLUSTER_STRAY_SHARED,
    cluster_color=COLOR_STRAY_SHARED,
    label="Centroid PSFs",
)
def centroid_2passes_oversample(
    result_obj,
    config_coords_guesses_file_name,
    psfs_subset="all",
    oversample_factor=3,
    grid_header=None,
    centroid_box_size=41,
    zoom_order=3,
    centroid_func=centroid_2dg,
    centroid_sources_impl=centroid_sources,
):
    """
    Centroid the PSF in the image of the result object, starting from first guesses of
    the positions. This uses 2 passes for accuracy.

    INPUTS:
    - result_obj: result object
    - config_coords_guesses_file_name: path to the config file with the coordinates
      guesses
    - psfs_subset: subset of PSFs to use, "all" or an int
    - oversample_factor: oversample factor
    - grid_header: header of the grid
    - centroid_box_size: size of the centroid box
    - zoom_order: order of the zoom
    - centroid_func: function to use for centroiding
    - centroid_sources_impl: implementation of the centroid_sources function

    OUTPUTS:
    - result_obj, with centroids set (2nd-pass centroid dict, see
      psf_grid_prep.refine_2nd_pass_centroids)
    """

    data_original = result_obj.image

    # load the config file with the coordinates guesses
    coords_guesses = helpers.load_config_and_pipe(
        config_file_choice=config_coords_guesses_file_name,
    )

    # 1st pass: centroid with photutils
    prep = psf_grid_prep.oversample_1st_pass_centroid(
        data_original,
        coords_guesses,
        psfs_subset=psfs_subset,
        oversample_factor=oversample_factor,
        centroid_box_size=centroid_box_size,
        zoom_order=zoom_order,
        centroid_func=centroid_func,
        centroid_sources_impl=centroid_sources_impl,
        grid_header=grid_header,
    )

    # 2nd pass: centroid with Gaussian fit
    centroid_post_2nd_pass = psf_grid_prep.refine_2nd_pass_centroids(
        data_original, prep, oversample_factor=oversample_factor
    )

    # now attach this to the result object
    result_obj.centroids = centroid_post_2nd_pass

    return result_obj


@pipeline_stage(
    name="stray_light_mask_real",
    depends_on=("centroid_2passes_oversample",),
    cluster=CLUSTER_STRAY_SHARED,
    cluster_color=COLOR_STRAY_SHARED,
    label="Mask the real PSF",
)
def stray_light_mask_real(result_obj, observing_config):
    """
    Mask the real PSF, so we can find the stray light.

    The mask is everything within psf_mask_n_lambda_over_d * lambda/D of the PSF
    centroid, with psf_mask_n_lambda_over_d set in the observing config.

    INPUTS:
    - result_obj (StrayLightResult): result object
    - observing_config (ObservingConfig): observing config

    OUTPUTS:
    - result_obj, with real_psf_mask set
    """

    # mask over N*lambda/D, with N from the observing config
    try:
        n_lambda_over_d = float(observing_config["psf_mask_n_lambda_over_d"])
    except KeyError as exc:
        raise KeyError(
            "No 'psf_mask_n_lambda_over_d' (radius of the PSF mask, in lambda/D) in "
            "the observing config"
        ) from exc
    logging.info("Masking the real PSF out to %g lambda/D", n_lambda_over_d)
    lambda_over_d = (
        206265.0
        * float(result_obj.wavel_central)
        / float(observing_config["D_aperture"]["full"])
    )  # in arcsec

    # make a 2D array of the distances from the centroid, units pixels
    del_x_pix = (
        np.arange(result_obj.image.shape[1])
        - result_obj.centroids["x_center_pix_fullarray_normsamp"]
    )
    del_y_pix = (
        np.arange(result_obj.image.shape[0])
        - result_obj.centroids["y_center_pix_fullarray_normsamp"]
    )
    xx_pix, yy_pix = np.meshgrid(del_x_pix, del_y_pix)

    xx_arcsec = (
        xx_pix * result_obj.pixel_scale / 1000
    )  # /1000 because pixel scale is in mas/pixel
    yy_arcsec = yy_pix * result_obj.pixel_scale / 1000

    angular_distances = np.sqrt(xx_arcsec**2 + yy_arcsec**2)
    mask = angular_distances < n_lambda_over_d * lambda_over_d

    # add the mask to the result object
    result_obj.real_psf_mask = mask

    return result_obj


# options for how to segment the stray light (methods from
# dev_notebooks/TP_image_segmentation.ipynb)
SegmentationMethod = Literal[
    "threshold", "hough_circle", "active_contour", "watershed", "kmeans", "meanshift"
]
SEGMENTATION_METHODS = (
    "threshold",
    "hough_circle",
    "active_contour",
    "watershed",
    "kmeans",
    "meanshift",
)

# default knobs for each segmentation method; override via the 'params' argument
SEGMENTATION_DEFAULT_PARAMS = {
    "threshold": {
        "threshold_kind": "otsu",  # 'otsu', 'mean' or 'median'
        "threshold_factor": 1.0,  # threshold = factor * (otsu|mean|median)
        "open_kernel": 3,  # pixels; morphological opening to remove speckles
        "min_area": 50,  # pixels; drop smaller regions
    },
    "hough_circle": {
        "edge_percentile": 99.8,  # keep Sobel edges above this percentile
        "min_area": 50,
        "min_circularity": 0.4,
    },
    "active_contour": {
        "edge_percentile": 99.8,
        "min_area": 50,
        "min_circularity": 0.4,
        "expand_px": 5,  # dilate each snake by this many pixels
        "alpha": 0.05,  # snake length shape
        "beta": 3,  # snake smoothness
        "gamma": 0.1,  # snake stepping parameter
    },
    "watershed": {
        "threshold_kind": "otsu",
        "threshold_factor": 1.0,
        "open_kernel": 3,
        "min_area": 50,
        # pixels; ellipse used to erode the foreground before the distance transform
        "erode_kernel": 7,
        "erode_iterations": 3,
        "min_distance": 10,  # pixels; between watershed seeds
        "marker_radius": 5,  # pixels; radius of each seed marker
    },
    "kmeans": {
        "k": 3,  # number of intensity clusters
        "n_bright": 1,  # how many of the brightest clusters count as stray light
        "min_area": 50,
    },
    "meanshift": {
        "sp": 10,  # spatial window radius
        "sr": 20,  # colour (intensity, 0-255) window radius
        "downsample": 1.0,  # <1 speeds up pyrMeanShiftFiltering on large frames
        "min_area": 50,
    },
}

# combined segmentation: default N in 'pixel is stray light if summed weight > N *
# sum(all weights)'
SEGMENTATION_DEFAULT_VOTE_FRACTION = 0.5

# fixed colour per method in the combined FYI plot (skips red, used for the PSF mask,
# and gray)
SEGMENTATION_METHOD_COLORS = {
    "threshold": "C0",
    "hough_circle": "C1",
    "active_contour": "C2",
    "watershed": "C4",
    "kmeans": "C6",
    "meanshift": "C9",
}


def _prepare_segmentation_image(result_obj):
    """
    Fill the masked real PSF with the background level and TV-denoise the image.

    OUTPUTS:
    - filled (ndarray): image with the PSF region replaced by the background median
    - tv (ndarray): TV-denoised version of 'filled', in the original units
    - psf_mask (ndarray[bool]): mask of the real PSF
    """
    psf_mask = result_obj.real_psf_mask.astype(bool)
    filled = result_obj.image.astype(float, copy=True)
    bg = float(np.nanmedian(filled[~psf_mask])) if np.any(~psf_mask) else 0.0
    filled[psf_mask] = bg
    filled[~np.isfinite(filled)] = bg

    # TV denoise on normalized data (smaller weight => more denoise)
    vmin = float(np.min(filled))
    vmax = float(np.max(filled))
    scale = vmax - vmin if vmax > vmin else 1.0
    tv = ski.restoration.denoise_tv_bregman((filled - vmin) / scale, weight=1.0)
    tv = tv * scale + vmin

    return filled, tv, psf_mask


def _to_uint8(image):
    """Min-max normalize an image to 0-255 uint8 (as OpenCV wants)."""
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _threshold_binary(tv, psf_mask, threshold_kind, threshold_factor, open_kernel):
    """
    Global threshold of the denoised image (statistics from pixels outside the PSF
    only).
    Returns a uint8 binary foreground (1 = above threshold).
    """
    vals = tv[~psf_mask]
    if threshold_kind == "otsu":
        thr = float(ski.filters.threshold_otsu(vals))
    elif threshold_kind == "mean":
        thr = float(np.mean(vals))
    elif threshold_kind == "median":
        thr = float(np.median(vals))
    else:
        raise ValueError(
            f"threshold_kind must be 'otsu', 'mean' or 'median', got {threshold_kind!r}"
        )
    thr *= threshold_factor
    logging.info(
        "Segmentation threshold (%s x %s): %.4g", threshold_kind, threshold_factor, thr
    )

    binary = ((tv > thr) & ~psf_mask).astype(np.uint8)
    if open_kernel and open_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return binary


def _find_circles(tv, edge_percentile, min_area, min_circularity):
    """
    Fit circles to bright blobs: Sobel edges -> strong-edge binary -> open/close ->
    connected components -> circularity cut -> minimum enclosing circle.
    (Edge-Sobel + Hough on a zero-filled PSF hole finds noise and hole-boundary rings,
    so circles are fitted to connected components instead.)

    OUTPUTS:
    - circles_xyr (list): (cx, cy, radius, area, circularity) per circle
    - binary (ndarray[uint8]): cleaned edge binary (for plotting)
    """
    # Sobel edge detection
    gx = cv2.Sobel(tv, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(tv, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # Keep only strong edges so the fit isn't swamped by noise
    thr = float(np.percentile(mag, edge_percentile))
    edges = (mag > thr).astype(np.uint8) * 255

    # Remove speckles ('erosion followed by dilation'), then fill small gaps in real
    # circles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = (binary > 0).astype(np.uint8)

    # Fit a circle to each connected component (within its bounding box, so as not to
    # make a full-frame mask per component)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    circles_xyr = []
    for lab in range(1, n_lab):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x0 = int(stats[lab, cv2.CC_STAT_LEFT])
        y0 = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        comp = (labels[y0 : y0 + h, x0 : x0 + w] == lab).astype(np.uint8)
        contours, _ = cv2.findContours(
            comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x0, y0)
        )
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circularity = 4.0 * np.pi * float(cv2.contourArea(cnt)) / (peri * peri)
        if circularity < min_circularity:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        circles_xyr.append((float(cx), float(cy), float(radius), area, circularity))

    logging.info("Detected %s circular regions (edge thr=%.4g)", len(circles_xyr), thr)
    for cx, cy, radius, area, circ in circles_xyr:
        logging.info(
            "  center=(%.1f, %.1f) r=%.1f area=%s circ=%.2f", cx, cy, radius, area, circ
        )

    return circles_xyr, binary


def _relabel_min_area(labels, psf_mask, min_area):
    """
    Renumber the positive labels in 'labels' as 1..n, with the PSF excluded and regions
    of fewer than min_area pixels (outside the PSF) dropped.

    OUTPUTS:
    - label_map (ndarray[int32]): 0 = no region; regions labelled 1..n
    """
    labels = np.where(psf_mask, 0, labels).astype(np.int32, copy=False)
    labels[labels < 0] = 0
    counts = np.bincount(labels.ravel())
    keep = counts >= max(min_area, 1)
    keep[0] = False
    new_labels = np.zeros(len(counts), dtype=np.int32)
    new_labels[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return new_labels[labels]


def _binary_to_label_map(binary, psf_mask, min_area):
    """Label map (1..n) of the connected components of 'binary' (PSF excluded)."""
    binary = ((binary > 0) & ~psf_mask).astype(np.uint8)
    _, labels = cv2.connectedComponents(binary, connectivity=8, ltype=cv2.CV_32S)
    return _relabel_min_area(labels, psf_mask, min_area)


def _plot_segmentation(tv, foreground, label_map, method, plot_file_name):
    """
    FYI plot: denoised image, foreground used by the method, and the final region
    outlines; saved to plot_file_name.
    """
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axs[0].imshow(tv, origin="lower", cmap="gray")
    axs[0].set_title("Denoised")
    axs[1].imshow(foreground, origin="lower", cmap="gray")
    axs[1].set_title("Foreground")
    axs[2].imshow(tv, origin="lower", cmap="gray")
    if np.any(label_map):
        axs[2].contour(
            (label_map > 0).astype(np.uint8), levels=[0.5], colors="lime", linewidths=1.5
        )
    axs[2].set_title(f"Regions ({int(label_map.max())})")
    for ax in axs:
        ax.set_aspect("equal")
    fig.suptitle(f"Stray light segmentation: {method}")
    fig.savefig(plot_file_name)
    logging.info("Saved plot of stray light segmentation to %s", plot_file_name)
    plt.close(fig)


def _merge_method_params(method, params):
    """
    Merge user overrides 'params' over SEGMENTATION_DEFAULT_PARAMS[method], warning
    about unused keys.
    """
    unknown = set(params) - set(SEGMENTATION_DEFAULT_PARAMS[method])
    if unknown:
        logging.warning(
            "Ignoring segmentation params not used by %r: %s", method, sorted(unknown)
        )
    return {**SEGMENTATION_DEFAULT_PARAMS[method], **params}


def _segment_single_method(tv, psf_mask, method, p):
    """
    Run one segmentation method on the denoised image (see stray_light_segmentation for
    the methods).

    INPUTS:
    - tv (ndarray): TV-denoised image, PSF filled with the background (from
      _prepare_segmentation_image)
    - psf_mask (ndarray[bool]): mask of the real PSF
    - method (str): one of SEGMENTATION_METHODS
    - p (dict): full set of knobs for this method (defaults merged with overrides)

    OUTPUTS:
    - foreground (ndarray[uint8]): foreground used by the method (for plotting)
    - label_map (ndarray[int32]): 0 = no region; regions labelled 1..n, PSF excluded
      (where regions of 'hough_circle' or 'active_contour' overlap, the later one wins)
    """

    if method == "threshold":

        foreground = _threshold_binary(
            tv, psf_mask, p["threshold_kind"], p["threshold_factor"], p["open_kernel"]
        )
        label_map = _binary_to_label_map(foreground, psf_mask, p["min_area"])

    elif method == "hough_circle":

        circles_xyr, foreground = _find_circles(
            tv, p["edge_percentile"], p["min_area"], p["min_circularity"]
        )
        # draw each circle into the label map
        label_map = np.zeros(tv.shape, dtype=np.int32)
        for i, (cx, cy, radius, *_) in enumerate(circles_xyr):
            cv2.circle(label_map, (int(cx), int(cy)), int(radius), i + 1, -1)

    elif method == "active_contour":

        circles_xyr, foreground = _find_circles(
            tv, p["edge_percentile"], p["min_area"], p["min_circularity"]
        )
        expand_px = p["expand_px"]
        expand_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * expand_px + 1, 2 * expand_px + 1)
        )
        tv_blurred = ski.filters.gaussian(tv, sigma=0.5)
        label_map = np.zeros(tv.shape, dtype=np.int32)
        comp = np.zeros(tv.shape, dtype=np.uint8)  # scratch, re-used for each snake
        n_regions = 0

        for cx, cy, radius, *_ in circles_xyr:
            # Generate an initial circular contour
            theta = np.linspace(0, 2 * np.pi, 100)
            init = np.array(
                [cy + radius * np.sin(theta), cx + radius * np.cos(theta)]
            ).T

            # Apply active contour model (snake)
            snake = active_contour(
                tv_blurred, init, alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"]
            )

            # Convert the snake to integer (x, y) for OpenCV and fill it
            snake_int = np.array(
                [(int(point[1]), int(point[0])) for point in snake], dtype=np.int32
            )
            comp[:] = 0
            cv2.fillPoly(comp, [snake_int], 255)
            # expand the contour, to make sure we capture all of the stray light
            comp_exp = cv2.dilate(comp, expand_kernel, iterations=1)

            cnts, _ = cv2.findContours(
                comp_exp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            # one region per expanded contour
            for contour in cnts:
                n_regions += 1
                cv2.fillPoly(label_map, [contour], n_regions)

    elif method == "watershed":

        foreground = _threshold_binary(
            tv, psf_mask, p["threshold_kind"], p["threshold_factor"], p["open_kernel"]
        )
        # erode, so that touching blobs separate in the distance transform
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (p["erode_kernel"], p["erode_kernel"])
        )
        eroded = cv2.morphologyEx(
            foreground, cv2.MORPH_ERODE, kernel, iterations=p["erode_iterations"]
        )
        dist_transform = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)

        # seeds at the local maxima of the distance transform
        peaks = peak_local_max(
            dist_transform,
            min_distance=p["min_distance"],
            labels=(eroded > 0).astype(int),
        )
        logging.info("Watershed seeds: %s", len(peaks))

        markers = np.zeros(tv.shape, dtype=np.int32)
        r = p["marker_radius"]
        for i, (py, px) in enumerate(peaks):
            y0, y1 = max(0, py - r), min(tv.shape[0], py + r + 1)
            x0, x1 = max(0, px - r), min(tv.shape[1], px + r + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            disk = (yy - py) ** 2 + (xx - px) ** 2 <= r**2
            markers[y0:y1, x0:x1][disk] = i + 1

        if len(peaks) > 0:
            # watershed on the blurred foreground image (8-bit, 3-channel for OpenCV)
            watershed_src = cv2.GaussianBlur(tv * foreground, (5, 5), 0)
            watershed_src = cv2.cvtColor(_to_uint8(watershed_src), cv2.COLOR_GRAY2BGR)
            labels = cv2.watershed(watershed_src, markers)
            # keep the foreground only; -1 marks watershed boundaries
            labels = np.where((foreground > 0) & (labels > 0), labels, 0)
        else:
            labels = np.zeros(tv.shape, dtype=np.int32)
        label_map = _relabel_min_area(labels, psf_mask, p["min_area"])

    elif method == "kmeans":

        # cluster the intensities of the pixels outside the PSF
        vals = tv[~psf_mask].reshape(-1, 1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels_1d, centers = cv2.kmeans(
            vals, p["k"], None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        bright = np.argsort(centers.ravel())[::-1][: p["n_bright"]]
        logging.info(
            "k-means centers: %s; stray light = brightest %s",
            np.sort(centers.ravel()),
            p["n_bright"],
        )
        foreground = np.zeros(tv.shape, dtype=np.uint8)
        foreground[~psf_mask] = np.isin(labels_1d.ravel(), bright).astype(np.uint8)
        label_map = _binary_to_label_map(foreground, psf_mask, p["min_area"])

    elif method == "meanshift":

        src = _to_uint8(tv)
        if p["downsample"] != 1.0:
            src = cv2.resize(src, (0, 0), fx=p["downsample"], fy=p["downsample"])
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        filtered = cv2.pyrMeanShiftFiltering(src, p["sp"], p["sr"], termcrit=criteria)
        filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        if filtered.shape != tv.shape:
            filtered = cv2.resize(
                filtered, (tv.shape[1], tv.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        thr = float(ski.filters.threshold_otsu(filtered[~psf_mask]))
        foreground = ((filtered > thr) & ~psf_mask).astype(np.uint8)
        label_map = _binary_to_label_map(foreground, psf_mask, p["min_area"])

    else:
        raise ValueError(
            f"Unknown segmentation method {method!r}; choose one of "
            f"{SEGMENTATION_METHODS}"
        )

    # never count the real PSF as stray light; drop regions that were entirely inside it
    label_map = _relabel_min_area(label_map, psf_mask, 1)
    n_regions = int(label_map.max())
    if n_regions == 0:
        logging.warning("Segmentation method %r found no stray light regions", method)
    else:
        logging.info("Segmentation method %r found %s regions", method, n_regions)

    return foreground, label_map


@pipeline_stage(
    name="stray_light_segmentation",
    depends_on=("stray_light_mask_real",),
    cluster=CLUSTER_STRAY_SHARED,
    cluster_color=COLOR_STRAY_SHARED,
    label="Segment stray light",
)
def stray_light_segmentation(
    result_obj,
    method: SegmentationMethod | None = None,
    params: dict | None = None,
):
    """
    Segment the stray light.

    Ideas and methods herein are from Eric Pantin's notebook for instructional purposes
    (dev_notebooks/TP_image_segmentation.ipynb). Many knobs remain to be tuned with real
    data.

    INPUTS:
    - result_obj: result object (needs image and real_psf_mask)
    - method: segmentation method; one of SEGMENTATION_METHODS:
        - 'threshold': global threshold (otsu/mean/median) of the denoised image, then
          connected components
        - 'hough_circle': circles fitted to bright blobs; each region is a filled circle
        - 'active_contour': as 'hough_circle', but each circle is refined with an active
          contour (snake)
            and dilated by expand_px
        - 'watershed': thresholded foreground split into touching blobs by
          distance-transform seeds + watershed
        - 'kmeans': k-means clustering of pixel intensities; the brightest cluster(s)
          are stray light
        - 'meanshift': mean-shift filtering, then an Otsu threshold of the filtered
          image
      If None, uses result_obj.segmentation_method (e.g. from the data state), else
      'active_contour'.
    - params: dict overriding SEGMENTATION_DEFAULT_PARAMS[method]; if None, uses
        result_obj.segmentation_params if set

    OUTPUTS:
    - result_obj (StrayLightResult), updated attributes:
        - segment_map (np.ndarray[int32]): integer region labels (0 = not stray light),
          regions labelled 1..n
        - segmentation_method (str): the method used
    - FYI plot in the result object's results_dir
    """

    if method is None:
        method = getattr(result_obj, "segmentation_method", None) or "active_contour"
    if method not in SEGMENTATION_METHODS:
        raise ValueError(
            f"Unknown segmentation method {method!r}; choose one of "
            f"{SEGMENTATION_METHODS}"
        )
    if params is None:
        params = getattr(result_obj, "segmentation_params", None) or {}
    p = _merge_method_params(method, params)
    logging.info("Segmenting stray light with method=%r, params=%s", method, p)

    # fill the masked PSF with the background and denoise
    _, tv, psf_mask = _prepare_segmentation_image(result_obj)

    foreground, label_map = _segment_single_method(tv, psf_mask, method, p)

    segmentation_plot_file_name = os.path.join(
        results_dir_of(result_obj),
        f"stray_light_segmentation_{method}_{fits_stem_of(result_obj)}.png",
    )
    _plot_segmentation(tv, foreground, label_map, method, segmentation_plot_file_name)

    # update the labels for each stray light region
    result_obj.segment_map = label_map
    result_obj.segmentation_method = method

    return result_obj


def _resolve_segmentation_weights(weights):
    """
    Weight per method: default 1.0 for every method in SEGMENTATION_METHODS, overridden
    by 'weights'.
    Raises if a method is unknown, a weight is outside [0, 1], or all weights are 0.
    """
    weights = weights or {}
    unknown = set(weights) - set(SEGMENTATION_METHODS)
    if unknown:
        raise ValueError(
            f"Unknown segmentation methods in weights: {sorted(unknown)}; choose from "
            f"{SEGMENTATION_METHODS}"
        )
    resolved = {m: float(weights.get(m, 1.0)) for m in SEGMENTATION_METHODS}
    bad = {m: w for m, w in resolved.items() if not 0.0 <= w <= 1.0}
    if bad:
        raise ValueError(f"Segmentation weights must be in [0, 1], got {bad}")
    if sum(resolved.values()) <= 0:
        raise ValueError("At least one segmentation weight must be > 0")
    return resolved


def _resolve_vote_fraction(vote_fraction):
    """
    N in 'summed weight > N * sum(all weights)'; default
    SEGMENTATION_DEFAULT_VOTE_FRACTION, must be in [0, 1].
    """
    if vote_fraction is None:
        vote_fraction = SEGMENTATION_DEFAULT_VOTE_FRACTION
    vote_fraction = float(vote_fraction)
    if not 0.0 <= vote_fraction <= 1.0:
        raise ValueError(
            f"Segmentation vote fraction must be in [0, 1], got {vote_fraction}"
        )
    return vote_fraction


def _combine_method_regions(method_region_maps, weights, vote_fraction, psf_mask):
    """
    Combine the regions found by several segmentation methods.

    Bookkeeping: the regions of all methods are OR-ed together, and each connected
    component
    (8-connectivity) of that union is one 'group' with one label. So regions from
    different methods
    which overlap (or touch) share a label; a group a method did not detect is absent
    from that
    method's label map.

    Final regions: a pixel is in if the summed weight of the methods detecting it is
    > vote_fraction * (sum of all weights), and is not in the real PSF. Final regions
    keep their group label.

    INPUTS:
    - method_region_maps (dict[str, ndarray]): per method, its label map (>0 = in a
      region)
    - weights (dict[str, float]): weight per method (all methods, incl. those with
      weight 0)
    - vote_fraction (float): N, in [0, 1]
    - psf_mask (ndarray[bool]): mask of the real PSF

    OUTPUTS:
    - dict with:
        - group_labels (ndarray[int32]): label of each group (0 = no method detected
          anything)
        - method_label_maps (dict[str, ndarray[int32]]): per method, group labels where
          it detected something
        - score_map (ndarray[float]): summed weight of the methods detecting each pixel
        - sigma (float): sum of all weights
        - threshold (float): vote_fraction * sigma
        - final_label_map (ndarray[int32]): group labels of the pixels passing the
          weighted vote
        - final_labels (list[int]): group labels with at least one final pixel
    """
    psf_mask = psf_mask.astype(bool)
    shape = psf_mask.shape

    # per method: its regions, never inside the PSF
    method_detected = {
        method: (np.asarray(label_map) > 0) & ~psf_mask
        for method, label_map in method_region_maps.items()
    }

    # group regions across methods: connected components of the union
    union = np.zeros(shape, dtype=bool)
    for detected in method_detected.values():
        union |= detected
    _, group_labels = cv2.connectedComponents(
        union.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
    )

    method_label_maps = {
        method: np.where(detected, group_labels, 0).astype(np.int32, copy=False)
        for method, detected in method_detected.items()
    }

    # weighted vote
    sigma = float(sum(weights.values()))
    threshold = vote_fraction * sigma
    score_map = np.zeros(shape, dtype=float)
    for method, detected in method_detected.items():
        score_map += weights[method] * detected
    final = (score_map > threshold) & ~psf_mask
    final_label_map = np.where(final, group_labels, 0).astype(np.int32, copy=False)

    final_labels = [int(lab) for lab in np.unique(final_label_map) if lab > 0]

    return {
        "group_labels": group_labels,
        "method_label_maps": method_label_maps,
        "score_map": score_map,
        "sigma": sigma,
        "threshold": threshold,
        "final_label_map": final_label_map,
        "final_labels": final_labels,
    }


def _labels_str(labels, max_shown=20):
    """Comma-separated labels, truncated after max_shown."""
    labels = [str(lab) for lab in labels]
    if len(labels) > max_shown:
        return ", ".join(labels[:max_shown]) + f", ... ({len(labels)} total)"
    return ", ".join(labels)


def _plot_segmentation_combined(
    image, psf_mask, combo, weights, vote_fraction, fits_stem, plot_file_name
):
    """
    FYI plot: image with each method's regions outlined in its own colour, the final
    (weighted-vote)
    regions with a bold border, and each group's label; saved to plot_file_name.
    """
    image = image.astype(float)
    vmin, vmax = np.nanpercentile(image, [1.0, 99.5])
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Counts")
    ax.contour(
        psf_mask.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=1.0,
        linestyles="--",
    )
    ax.plot([], [], color="red", linestyle="--", label="PSF mask")

    # each method's regions (one outline per method; regions of a method never touch
    # each other's groups, so the outlines are the same as one per group)
    for method, label_map in combo["method_label_maps"].items():
        color = SEGMENTATION_METHOD_COLORS[method]
        labels_this = [int(lab) for lab in np.unique(label_map) if lab > 0]
        if labels_this:
            ax.contour(
                (label_map > 0).astype(np.uint8),
                levels=[0.5],
                colors=color,
                linewidths=1.2,
            )
        ax.plot(
            [],
            [],
            color=color,
            label=f"{method} (w={weights[method]:g}; labels "
            f"{_labels_str(labels_this) or '-'})",
        )

    # final regions: bold white border over a black underlay
    if combo["final_labels"]:
        final_any = (combo["final_label_map"] > 0).astype(np.uint8)
        ax.contour(final_any, levels=[0.5], colors="black", linewidths=1.5)
        ax.contour(final_any, levels=[0.5], colors="white", linewidths=1.2)
    ax.plot([], [], color="white", linewidth=2.5, label="Final (weighted vote)")

    # label each group just above-right of its bbox; white if it survived the vote
    final_labels = set(combo["final_labels"])
    for i, slices in enumerate(ndimage.find_objects(combo["group_labels"])):
        if slices is None:
            continue
        lab = i + 1
        is_final = lab in final_labels
        ax.text(
            slices[1].stop + 9,
            slices[0].stop + 9,
            str(lab),
            color="white" if is_final else "khaki",
            fontsize=12,
            fontweight="bold" if is_final else "normal",
            ha="left",
            va="bottom",
        )

    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title(
        f"Combined segmentation: Σ = {combo['sigma']:g}, N = {vote_fraction:g}, "
        f"in if Σ' > N·Σ = {combo['threshold']:g}\n{fits_stem}",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(plot_file_name)
    logging.info(
        "Saved plot of combined stray light segmentation to %s", plot_file_name
    )
    plt.close(fig)


def _write_segmentation_combined_fits(
    result_obj, psf_mask, combo, weights, vote_fraction, fits_file_name
):
    """
    Write the combined segmentation to a FITS file of boolean maps (uint8: 1 = stray
    light region, 0 = not):
    - PRIMARY: header only (input file, weights, vote fraction, threshold)
    - one extension per method that was run (EXTNAME = method): regions it detected
    - FINAL: regions from the weighted vote
    The real PSF is always 0. Cross-method group labels are listed in each extension's
    LABELS keyword
    (and annotated in the combined FYI plot).
    """
    psf_mask = psf_mask.astype(bool)

    def _bool_map(label_map):
        return ((label_map > 0) & ~psf_mask).astype(np.uint8)

    def _labels_str_fits(label_map):
        return ",".join(str(lab) for lab in np.unique(label_map) if lab > 0)

    primary = fits.PrimaryHDU()
    hdr = primary.header
    hdr["INFILE"] = (os.path.basename(result_obj.file_absname), "input frame")
    hdr["FILTER"] = (str(result_obj.filter_name), "filter")
    hdr["DETECTOR"] = (str(result_obj.detector), "detector")
    for method, weight in weights.items():
        hdr[f"HIERARCH WEIGHT {method.upper()}"] = (weight, f"weight of {method}")
    hdr["VOTEFRAC"] = (vote_fraction, "N: in if summed weight > N * SUMWGHT")
    hdr["SUMWGHT"] = (combo["sigma"], "sum of all weights")
    hdr["THRESH"] = (combo["threshold"], "VOTEFRAC * SUMWGHT")
    hdr["COMMENT"] = (
        "Extensions: one boolean map per segmentation method, then FINAL (weighted "
        "vote)."
    )
    hdr["COMMENT"] = "1 = stray light region, 0 = not (real PSF always 0)."
    hdus = [primary]

    for method, label_map in combo["method_label_maps"].items():
        hdu = fits.ImageHDU(_bool_map(label_map), name=method.upper())
        hdu.header["METHOD"] = (method, "segmentation method")
        hdu.header["WEIGHT"] = (weights[method], "weight of this method")
        hdu.header["NREGION"] = (
            len(np.unique(label_map[label_map > 0])),
            "number of regions",
        )
        hdu.header["LABELS"] = (
            _labels_str_fits(label_map),
            "group labels present (see FYI plot)",
        )
        hdus.append(hdu)

    final_hdu = fits.ImageHDU(_bool_map(combo["final_label_map"]), name="FINAL")
    final_hdu.header["VOTEFRAC"] = (
        vote_fraction,
        "N: in if summed weight > N * SUMWGHT",
    )
    final_hdu.header["SUMWGHT"] = (combo["sigma"], "sum of all weights")
    final_hdu.header["THRESH"] = (combo["threshold"], "VOTEFRAC * SUMWGHT")
    final_hdu.header["NREGION"] = (
        len(combo["final_labels"]),
        "number of final regions",
    )
    final_hdu.header["LABELS"] = (
        _labels_str_fits(combo["final_label_map"]),
        "group labels present (see FYI plot)",
    )
    hdus.append(final_hdu)

    fits.HDUList(hdus).writeto(fits_file_name, overwrite=True)
    logging.info(
        "Saved FITS file of combined stray light segmentation to %s", fits_file_name
    )


@pipeline_stage(
    name="stray_light_segmentation_combined",
    depends_on=("stray_light_mask_real",),
    cluster=CLUSTER_STRAY_SHARED,
    cluster_color=COLOR_STRAY_SHARED,
    label="Segment stray light (weighted combination of methods)",
)
def stray_light_segmentation_combined(
    result_obj,
    weights: dict | None = None,
    vote_fraction: float | None = None,
    params: dict | None = None,
):
    """
    Segment the stray light with a weighted combination of the methods in
    SEGMENTATION_METHODS.

    Each method with weight > 0 is run on the same denoised frame. Regions that overlap
    across methods
    get the same label (see _combine_method_regions). A pixel is finally counted as
    stray light if
    Σ' > N * Σ, where Σ' is the summed weight of the methods detecting that pixel, Σ is
    the sum of all
    weights, and N = vote_fraction.

    INPUTS:
    - result_obj: result object (needs image and real_psf_mask)
    - weights: {method: weight in [0, 1]}; methods left out get weight 1, weight 0 skips
      a method.
        If None, uses result_obj.segmentation_weights (e.g. from the data state), else
        all 1.
    - vote_fraction: N in [0, 1]. If None, uses result_obj.segmentation_vote_fraction,
        else SEGMENTATION_DEFAULT_VOTE_FRACTION
    - params: {method: {knob: value}} overriding SEGMENTATION_DEFAULT_PARAMS[method]; if
      None,
        uses result_obj.segmentation_params if set

    OUTPUTS:
    - result_obj (StrayLightResult), updated attributes:
        - segment_map (np.ndarray[int32]): group labels of the final regions (0 = not
          stray light)
        - segmentation_method (str): 'combined'
        - segmentation_weights, segmentation_vote_fraction: as used
        - method_label_maps (dict): per method, group labels of its detections
        - final_label_map (np.ndarray): group labels of the final regions (same as
          segment_map)
    - FYI plots and FITS file in the result object's results_dir
      (stray_light_segmentation_*_<stem>.png, stray_light_segmentation_combined_<stem>
      .png/.fits)
    """

    if weights is None:
        weights = getattr(result_obj, "segmentation_weights", None)
    if vote_fraction is None:
        vote_fraction = getattr(result_obj, "segmentation_vote_fraction", None)
    if params is None:
        params = getattr(result_obj, "segmentation_params", None) or {}
    weights = _resolve_segmentation_weights(weights)
    vote_fraction = _resolve_vote_fraction(vote_fraction)
    unknown = set(params) - set(SEGMENTATION_METHODS)
    if unknown:
        raise ValueError(
            "Combined segmentation params must be nested per method "
            f"{SEGMENTATION_METHODS}; "
            f"got unknown keys {sorted(unknown)}"
        )

    active_methods = [m for m in SEGMENTATION_METHODS if weights[m] > 0]
    logging.info(
        "Combined segmentation: weights=%s, vote fraction N=%s, active methods=%s",
        weights,
        vote_fraction,
        active_methods,
    )

    # fill the masked PSF with the background and denoise (once, shared by all methods)
    _, tv, psf_mask = _prepare_segmentation_image(result_obj)

    results_dir = results_dir_of(result_obj)
    fits_stem = fits_stem_of(result_obj)

    method_region_maps = {}
    for method in active_methods:
        p = _merge_method_params(method, params.get(method) or {})
        logging.info("Segmenting stray light with method=%r, params=%s", method, p)
        foreground, label_map = _segment_single_method(tv, psf_mask, method, p)
        segmentation_plot_file_name = os.path.join(
            results_dir, f"stray_light_segmentation_{method}_{fits_stem}.png"
        )
        _plot_segmentation(
            tv, foreground, label_map, method, segmentation_plot_file_name
        )
        method_region_maps[method] = label_map

    combo = _combine_method_regions(method_region_maps, weights, vote_fraction, psf_mask)
    del method_region_maps

    # which methods found which group, and whether it survived the vote
    method_labels = {
        m: set(np.unique(label_map[label_map > 0]).tolist())
        for m, label_map in combo["method_label_maps"].items()
    }
    final_labels = set(combo["final_labels"])
    for lab in range(1, int(combo["group_labels"].max()) + 1):
        detected_by = [m for m, labels in method_labels.items() if lab in labels]
        logging.info(
            "Region %s: detected by %s (summed weight %g vs. threshold %g); %s",
            lab,
            detected_by,
            sum((weights[m] for m in detected_by)),
            combo["threshold"],
            "FINAL" if lab in final_labels else "rejected",
        )
    if not combo["final_labels"]:
        logging.warning(
            "Combined segmentation found no stray light regions passing the weighted "
            "vote"
        )
    else:
        logging.info(
            "Combined segmentation: %s final regions, labels %s",
            len(combo["final_labels"]),
            combo["final_labels"],
        )

    _plot_segmentation_combined(
        result_obj.image,
        psf_mask,
        combo,
        weights,
        vote_fraction,
        fits_stem,
        os.path.join(results_dir, f"stray_light_segmentation_combined_{fits_stem}.png"),
    )
    _write_segmentation_combined_fits(
        result_obj,
        psf_mask,
        combo,
        weights,
        vote_fraction,
        os.path.join(
            results_dir, f"stray_light_segmentation_combined_{fits_stem}.fits"
        ),
    )

    result_obj.segment_map = combo["final_label_map"]
    result_obj.segmentation_method = "combined"
    result_obj.segmentation_weights = weights
    result_obj.segmentation_vote_fraction = vote_fraction
    result_obj.method_label_maps = combo["method_label_maps"]
    result_obj.final_label_map = combo["final_label_map"]

    return result_obj
