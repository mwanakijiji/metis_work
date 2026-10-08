import os
from pathlib import Path
import numpy as np
import numpy.ma as ma
from . import psf_grid_prep, helpers
from photutils.centroids import centroid_2dg, centroid_sources
import ipdb
from dataclasses import dataclass, field
from typing import Any
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import cv2
import skimage as ski
from skimage.segmentation import active_contour
from skimage.feature import peak_local_max
from typing import Literal
import logging
import pandas as pd
from pipeline_registry import CLUSTER_STRAY, COLOR_STRAY, pipeline_stage

# where IMG-OPT-04 analysis plots are written (METIS_MAIT_code_eckhart/results/IMG_04_analysis_results/)
RESULTS_DIR_IMG_04 = str(Path(__file__).resolve().parents[2] / "results" / "IMG_04_analysis_results")


# class for containing information about a stray light region
@dataclass
class StrayLightRegion:
    label: int
    spatial_scale: str  # e.g. "point", "extended", "large"
    peak_irradiance: float
    total_flux: float
    area_pix: int
    surface_brightness: float
    # optional: bbox, centroid, ...


# class for containing information about a stray light result from a single FITS file
@dataclass
class StrayLightResult:
    # identity
    file_absname: str
    filter_name: str
    detector: str
    image: np.ndarray

    # derived from observing config
    wavel_central: float | None = None  # meters
    pixel_scale: float | None = None  # mas/pixel

    # images
    # mask of the real PSF
    real_psf_mask: np.ndarray | None = None
    # integer labels for each stray light region
    segment_map: np.ndarray | None = None
    # list of masks of the stray light regions (could be larger than the regions themselves in segment_map)
    stray_light_masks: list[np.ndarray] | None = None

    # segmentation choice (see stray_light_segmentation); can be set per data state
    segmentation_method: str | None = None
    segmentation_params: dict | None = None


    # bookkeeping
    centroids: Any | None = None  # from centroid_2passes_oversample
    regions: list[StrayLightRegion] = field(default_factory=list)

    # global quantities
    background_level: float | None = None
    background_rms: float | None = None


@pipeline_stage(
    name="populate_result_obj_info",
    depends_on=("generate_stray_light_sim",),
    cluster=CLUSTER_STRAY,
    cluster_color=COLOR_STRAY,
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

    # add the central wavelength for the filter, from the table for this band (img_lm → _lm, img_n → _n)
    filters_key = "monochromatic_observing_filters_" + scale_key.removeprefix("img_")
    try:
        result_obj.wavel_central = float(observing_config[filters_key][result_obj.filter_name])  # in m
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
    name="stray_light_brightness_spectrum",
    depends_on=("stray_light_segmentation",),
    cluster=CLUSTER_STRAY,
    cluster_color=COLOR_STRAY,
    label="Arrange segments into brightnesses as function of spatial scales",
)
def stray_light_brightness_spectrum(result_obj, observing_config):
    """
    Sort out the segments into brightnesses as function of spatial scales.

    INPUTS:
    - result_obj: result object, which should contain 
        - image (ndarray): science image
        - segment_map (list[ndarray]): integer region labels (0 = not stray light)
        - stray_light_masks (list[ndarray]): one mask per region (n.b. each mask can be larger than the region itself in segment_map)
    - observing_config: observing config


    OUTPUTS:
    - result_obj (StrayLightResult), updated attributes:
        - TBD
    """

    # required attributes are there?
    if (result_obj.image is None) or (result_obj.segment_map is None) or (result_obj.stray_light_masks is None):
        msg = "stray_light_brightness_spectrum missing required attributes"
        logging.error(msg)
        raise ValueError(msg)

    # extract the stray light fluxes from inside the region masks
    result_obj.regions = []

    # segmentation found no stray light regions: nothing to measure, so return an empty spectrum
    if len(result_obj.stray_light_masks) == 0:
        logging.warning(
            "No stray light regions from segmentation; returning an empty brightness spectrum"
        )
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_xlabel("Radius of equiv. circle (pixels)")
        ax.set_ylabel("Avg. surf. brightness (counts)")
        ax.set_title("Stray Light Fluxes vs. Radius (no stray light regions found)")
        plt.show()
        plt.close(fig)

        # no stray light detected, so the requirements are trivially met
        logging.info("--------------------------------------------------")
        logging.info("--------------------------------------------------")
        logging.info("METIS-1189 and -1429: Pass (no stray light regions detected)")
        logging.info("METIS-9522: Pass (no stray light regions detected)")
        logging.info("--------------------------------------------------")
        logging.info("IMG 04 STRAY LIGHT TEST: PASS")
        logging.info("--------------------------------------------------")
        return result_obj

    # loop over each stray light region and get the flux statistics
    for region_num in range(0,len(result_obj.stray_light_masks)):
        mask_this = result_obj.stray_light_masks[region_num]

        # inside the stray light region
        #inside = np.multiply(result_obj.image,mask_this)

        #inside = region_mask_this.astype(bool)
        inside_flag = np.where(mask_this > 0, 1.0, np.nan)  # 255 → True, 0 → nan
        # the stray light in 2D, with all else masked
        stray_light = np.multiply(inside_flag, result_obj.image.astype(float))

        #segment_map[inside] = i # label the region

        # append StrayLightRegions to the result_obj
        result_obj.regions.append(
            StrayLightRegion(
                label=region_num+1, # >0 to avoid confusion with non-stray-light regions
                spatial_scale=np.sqrt(float(np.nansum(inside_flag))/np.pi), # radius of an area-equivalent circle
                peak_irradiance=float(np.nanmax(stray_light)), # max irradiance in the region
                total_flux=float(np.nansum(stray_light)), # total flux in the region
                area_pix=int(np.nansum(inside_flag)), # area in pixels
                surface_brightness=float(np.nansum(stray_light))/float(np.nansum(inside_flag)), # average surface brightness in the region
            )
        )

    # mask the PSF and all stray light regions to get the background level
    mask_all_stray_light_regions = np.asarray(result_obj.stray_light_masks) # convert list to 3D array
    mask_all_stray_light_regions = np.sum(mask_all_stray_light_regions, axis = 0)
    # add in the psf mask
    mask_all_stray_light_regions = np.sum((mask_all_stray_light_regions, result_obj.real_psf_mask), axis=0)
    mask_all_stray_light_regions = np.where(mask_all_stray_light_regions == 0, 1.0, np.nan) # 1: background; 0: psf or stray light

    # the background-only image
    background_only = np.multiply(result_obj.image, mask_all_stray_light_regions.astype(float))

    # the psf-only image
    psf_only = np.where(
        result_obj.real_psf_mask.astype(bool),
        result_obj.image.astype(float),
        np.nan,
    )

    # get the stdev of the background
    stdev_background_only = np.nanstd(background_only)

    # get the max of the psf
    psf_max = np.nanmax(psf_only)

    logging.info("Background rms (outside PSF and stray regions): %.4g", stdev_background_only)
    logging.info("PSF peak: %.4g", psf_max)

    # threshold for significant stray light (req. METIS-9522)
    noise_3sig = 3.0 * stdev_background_only

    # dividing line between point-like (i.e., size of PSF) and extended stray light
    # wavel = float(observing_config['filter_name']['wavelength'])
    lambda_over_D_asec = (
        206265.0
        * float(result_obj.wavel_central)
        / float(observing_config["D_aperture"]["full"])
    )  # in arcsec
    lambda_over_D_pix = lambda_over_D_asec / (result_obj.pixel_scale / 1000)


    # display the brightness spectrum
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for region in result_obj.regions:
        # flux vs. radius
        plt.scatter(region.spatial_scale, region.surface_brightness, label=f'Region {region.label}')
    #plt.yscale('log')
    plt.axvline(x=lambda_over_D_pix, color='red', linestyle='--', label='Size scale: lambda/D')
    plt.axhline(y=noise_3sig, color='green', linestyle='--', label='Counts: 3-sigma of background noise')
    plt.axhline(y=0.0004 * psf_max, color='orange', linestyle='--', label='Counts: 0.04% of PSF peak')
    plt.axhline(y=psf_max, color='blue', linestyle='--', label='Counts: PSF peak')
    plt.xlabel("Radius of equiv. circle (pixels)")
    plt.ylabel("Avg. surf. brightness (counts)")
    plt.legend()
    plt.title("Stray Light Fluxes vs. Radius")
    # save before plt.show(), which discards the figure in interactive sessions
    os.makedirs(RESULTS_DIR_IMG_04, exist_ok=True)
    fits_stem = os.path.splitext(os.path.basename(result_obj.file_absname))[0]
    plot_file_name = os.path.join(
        RESULTS_DIR_IMG_04, f"stray_light_brightness_spectrum_{fits_stem}.png"
    )
    fig.savefig(plot_file_name)
    logging.info(f"Saved plot of brightness spectrum to {plot_file_name}")
    plt.show()
    plt.close(fig)

    # display where each region is on the 2D readout (colors match the brightness spectrum)
    image = result_obj.image.astype(float)
    vmin, vmax = np.nanpercentile(image, [1.0, 99.5])
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Counts")
    ax.contour(
        result_obj.real_psf_mask.astype(float), levels=[0.5], colors="red",
        linewidths=1.0, linestyles="--",
    )
    ax.plot([], [], color="red", linestyle="--", label="PSF mask")
    for i, region in enumerate(result_obj.regions):
        mask_this = result_obj.stray_light_masks[i] > 0
        color = f"C{i % 10}"
        ax.contour(mask_this.astype(float), levels=[0.5], colors=color, linewidths=1.5)
        # label just above-right of the region, so it doesn't hide the outline
        ys, xs = np.nonzero(mask_this)
        ax.text(
            xs.max() + 10, ys.max() + 10, str(region.label), color=color,
            fontsize=12, fontweight="bold", ha="left", va="bottom",
        )
        ax.plot([], [], color=color, label=f"Region {region.label}")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title(f"Stray light regions ({result_obj.segmentation_method})\n{fits_stem}", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    regions_plot_file_name = os.path.join(
        RESULTS_DIR_IMG_04, f"stray_light_regions_{fits_stem}.png"
    )
    fig.savefig(regions_plot_file_name)
    logging.info(f"Saved plot of stray light region locations to {regions_plot_file_name}")
    plt.show()
    plt.close(fig)

    ipdb.set_trace()

    # displ

    # Req. METIS-1189 and -1429: is the stray light irradiance <0.04% of the peak irradiance in the focal plane?
    check_metis_1189_1429_array = [] # each stray light region: pass or no
    for region_this in result_obj.regions:
        if region_this.peak_irradiance < 0.0004 * psf_max:
            logging.info(f"Stray light region {region_this.label} fulfills <0.04% of the peak irradiance")
            check_metis_1189_1429_array.append(True)
        else:
            logging.info(f"Stray light region {region_this.label} >0.04% of the peak irradiance!")
            check_metis_1189_1429_array.append(False)
       
    # Req. METIS-9522: is the flux in optical artefacts and ghosts shall be
    # less than the 3-sigma thermal background noise for one hour of observations and for
    # the respective spatial scale of the ghost?
    check_metis_9522_array = []
    for region_this in result_obj.regions:
        if region_this.peak_irradiance < noise_3sig:
            logging.info(f"Stray light region {region_this.label} fulfills <3-sigma of thermal background")
            check_metis_9522_array.append(True)
        else:
            logging.info(f"Stray light region {region_this.label} >3-sigma of thermal background!")
            check_metis_9522_array.append(False)

    # check if all the regions passed the checks
    logging.info("--------------------------------------------------")
    logging.info("--------------------------------------------------")
    if all(check_metis_1189_1429_array):
        logging.info("METIS-1189 and -1429: Pass")
    else:
        logging.info("METIS-1189 and -1429: Fail")
    if all(check_metis_9522_array):
        logging.info("METIS-9522: Pass")
    else:
        logging.info("METIS-9522: Fail")
    logging.info("--------------------------------------------------")
    if all(check_metis_1189_1429_array) and all(check_metis_9522_array):
        logging.info("IMG 04 STRAY LIGHT TEST: PASS")
    else:
        logging.info("IMG 04 STRAY LIGHT TEST: FAIL")
    logging.info("--------------------------------------------------")


    return result_obj


@pipeline_stage(
    name="centroid_2passes_oversample",
    depends_on=("populate_result_obj_info",),
    cluster=CLUSTER_STRAY,
    cluster_color=COLOR_STRAY,
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
    Take 1 FITS file and centroid the PSFs, starting with first guesses of the positions.
    This uses 2 passes for accuracy.

    INPUTS:
    - result_obj: result object
    - config_coords_guesses_file_name: path to the config file with the coordinates guesses
    - psfs_subset: subset of PSFs to use, "all" or "subset"
    - oversample_factor: oversample factor
    - grid_header: header of the grid
    - centroid_box_size: size of the centroid box
    - zoom_order: order of the zoom
    - centroid_func: function to use for centroiding
    - centroid_sources_impl: implementation of the centroid_sources function

    OUTPUTS:
    - CentroidResult object
    - prep: preparation object
    - results: results object
    - _: _
    - _: _
    - _: _
    - _: _
    - _: _
    """

    # load the empirical readout
    # data_original, header = psf_grid_prep.load_fits_data(file_name=image_array, hdu_index=1)

    data_original = result_obj.image

    # load the config file with the coordinates guesses
    coords_guesses = helpers.load_config_and_pipe(
        config_file_choice=config_coords_guesses_file_name,
    )

    # 1st pass: centroid with photutils
    prep = psf_grid_prep.oversample_1st_pass_centroid(data_original, coords_guesses)

    # 2nd pass: centroid with Gaussian fit
    centroid_post_2nd_pass = psf_grid_prep.refine_2nd_pass_centroids(
        data_original, prep
    )

    # return CentroidResult(prep=prep, refined=results)

    # now attach this to the result object
    result_obj.centroids = centroid_post_2nd_pass

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


# options for how to segment the stray light (methods from dev_notebooks/TP_image_segmentation.ipynb)
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
        "erode_kernel": 7,  # pixels; ellipse used to erode the foreground before the distance transform
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
    Global threshold of the denoised image (statistics from pixels outside the PSF only).
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
    logging.info(f"Segmentation threshold ({threshold_kind} x {threshold_factor}): {thr:.4g}")

    binary = ((tv > thr) & ~psf_mask).astype(np.uint8)
    if open_kernel and open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
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

    # Remove speckles ('erosion followed by dilation'), then fill small gaps in real circles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = (binary > 0).astype(np.uint8)

    # Fit a circle to each connected component
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    circles_xyr = []
    for lab in range(1, n_lab):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == lab).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    logging.info(f"Detected {len(circles_xyr)} circular regions (edge thr={thr:.4g})")
    for cx, cy, radius, area, circ in circles_xyr:
        logging.info(f"  center=({cx:.1f}, {cy:.1f}) r={radius:.1f} area={area} circ={circ:.2f}")

    return circles_xyr, binary


def _binary_to_region_masks(binary, psf_mask, min_area):
    """One 255-valued uint8 mask per connected component of 'binary' (PSF excluded)."""
    binary = ((binary > 0) & ~psf_mask).astype(np.uint8)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return _labels_to_region_masks(labels, psf_mask, min_area)


def _labels_to_region_masks(labels, psf_mask, min_area):
    """One 255-valued uint8 mask per positive label in 'labels' (PSF excluded)."""
    region_masks = []
    for lab in np.unique(labels):
        if lab <= 0:
            continue
        region = (labels == lab) & ~psf_mask
        if int(region.sum()) < min_area:
            continue
        region_masks.append(region.astype(np.uint8) * 255)
    return region_masks


def _plot_segmentation(tv, foreground, region_masks, method):
    """FYI plot: denoised image, foreground used by the method, and the final region outlines."""
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axs[0].imshow(tv, origin="lower", cmap="gray")
    axs[0].set_title("Denoised")
    axs[1].imshow(foreground, origin="lower", cmap="gray")
    axs[1].set_title("Foreground")
    axs[2].imshow(tv, origin="lower", cmap="gray")
    for region_mask in region_masks:
        axs[2].contour(region_mask > 0, levels=[0.5], colors="lime", linewidths=1.5)
    axs[2].set_title(f"Regions ({len(region_masks)})")
    for ax in axs:
        ax.set_aspect("equal")
    fig.suptitle(f"Stray light segmentation: {method}")
    plt.show()
    plt.close(fig)


@pipeline_stage(
    name="stray_light_segmentation",
    depends_on=("stray_light_mask_real",),
    cluster=CLUSTER_STRAY,
    cluster_color=COLOR_STRAY,
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
    (dev_notebooks/TP_image_segmentation.ipynb). Many knobs remain to be tuned with real data.

    INPUTS:
    - result_obj: result object (needs image and real_psf_mask)
    - method: segmentation method; one of SEGMENTATION_METHODS:
        - 'threshold': global threshold (otsu/mean/median) of the denoised image, then connected components
        - 'hough_circle': circles fitted to bright blobs; each region is a filled circle
        - 'active_contour': as 'hough_circle', but each circle is refined with an active contour (snake)
            and dilated by expand_px
        - 'watershed': thresholded foreground split into touching blobs by distance-transform seeds + watershed
        - 'kmeans': k-means clustering of pixel intensities; the brightest cluster(s) are stray light
        - 'meanshift': mean-shift filtering, then an Otsu threshold of the filtered image
      If None, uses result_obj.segmentation_method (e.g. from the data state), else 'active_contour'.
    - params: dict overriding SEGMENTATION_DEFAULT_PARAMS[method]; if None, uses
        result_obj.segmentation_params if set

    OUTPUTS:
    - result_obj (StrayLightResult), updated attributes:
        - segment_map list(np.ndarray): integer region labels (0 = not stray light), one map per region
        - stray_light_masks list(np.ndarray): one 255-valued uint8 mask per region
        - segmentation_method (str): the method used
    """

    if method is None:
        method = getattr(result_obj, "segmentation_method", None) or "active_contour"
    if method not in SEGMENTATION_METHODS:
        raise ValueError(
            f"Unknown segmentation method {method!r}; choose one of {SEGMENTATION_METHODS}"
        )
    if params is None:
        params = getattr(result_obj, "segmentation_params", None) or {}
    unknown = set(params) - set(SEGMENTATION_DEFAULT_PARAMS[method])
    if unknown:
        logging.warning(f"Ignoring segmentation params not used by {method!r}: {sorted(unknown)}")
    p = {**SEGMENTATION_DEFAULT_PARAMS[method], **params}
    logging.info(f"Segmenting stray light with method={method!r}, params={p}")

    # fill the masked PSF with the background and denoise
    _, tv, psf_mask = _prepare_segmentation_image(result_obj)

    if method == "threshold":

        foreground = _threshold_binary(
            tv, psf_mask, p["threshold_kind"], p["threshold_factor"], p["open_kernel"]
        )
        region_masks = _binary_to_region_masks(foreground, psf_mask, p["min_area"])

    elif method == "hough_circle":

        circles_xyr, foreground = _find_circles(
            tv, p["edge_percentile"], p["min_area"], p["min_circularity"]
        )
        # make a mask for each circle
        region_masks = []
        for cx, cy, radius, *_ in circles_xyr:
            region_mask = np.zeros(tv.shape, dtype=np.uint8)
            cv2.circle(region_mask, (int(cx), int(cy)), int(radius), 255, -1)
            region_masks.append(region_mask)

    elif method == "active_contour":

        circles_xyr, foreground = _find_circles(
            tv, p["edge_percentile"], p["min_area"], p["min_circularity"]
        )
        expand_px = p["expand_px"]
        expand_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * expand_px + 1, 2 * expand_px + 1)
        )
        tv_blurred = ski.filters.gaussian(tv, sigma=0.5)
        expanded_contours = []

        for cx, cy, radius, *_ in circles_xyr:
            # Generate an initial circular contour
            theta = np.linspace(0, 2 * np.pi, 100)
            init = np.array([cy + radius * np.sin(theta), cx + radius * np.cos(theta)]).T

            # Apply active contour model (snake)
            snake = active_contour(
                tv_blurred, init, alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"]
            )

            # Convert the snake to integer (x, y) for OpenCV and fill it
            snake_int = np.array(
                [(int(point[1]), int(point[0])) for point in snake], dtype=np.int32
            )
            comp = np.zeros(tv.shape, dtype=np.uint8)
            cv2.fillPoly(comp, [snake_int], 255)
            # expand the contour, to make sure we capture all of the stray light
            comp_exp = cv2.dilate(comp, expand_kernel, iterations=1)

            cnts, _ = cv2.findContours(comp_exp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            expanded_contours.extend(cnts)

        # make a mask for each expanded contour
        region_masks = []
        for contour in expanded_contours:
            region_mask_this = np.zeros(tv.shape, dtype=np.uint8)
            cv2.fillPoly(region_mask_this, [contour], 255)
            region_masks.append(region_mask_this)

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
            dist_transform, min_distance=p["min_distance"], labels=(eroded > 0).astype(int)
        )
        logging.info(f"Watershed seeds: {len(peaks)}")

        markers = np.zeros(tv.shape, dtype=np.int32)
        yy, xx = np.indices(tv.shape)
        for i, (py, px) in enumerate(peaks):
            r = p["marker_radius"]
            y0, y1 = max(0, py - r), min(tv.shape[0], py + r + 1)
            x0, x1 = max(0, px - r), min(tv.shape[1], px + r + 1)
            disk = (yy[y0:y1, x0:x1] - py) ** 2 + (xx[y0:y1, x0:x1] - px) ** 2 <= r**2
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
        region_masks = _labels_to_region_masks(labels, psf_mask, p["min_area"])

    elif method == "kmeans":

        # cluster the intensities of the pixels outside the PSF
        vals = tv[~psf_mask].reshape(-1, 1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels_1d, centers = cv2.kmeans(
            vals, p["k"], None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        bright = np.argsort(centers.ravel())[::-1][: p["n_bright"]]
        logging.info(
            f"k-means centers: {np.sort(centers.ravel())}; stray light = brightest {p['n_bright']}"
        )
        foreground = np.zeros(tv.shape, dtype=np.uint8)
        foreground[~psf_mask] = np.isin(labels_1d.ravel(), bright).astype(np.uint8)
        region_masks = _binary_to_region_masks(foreground, psf_mask, p["min_area"])

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
        region_masks = _binary_to_region_masks(foreground, psf_mask, p["min_area"])

    # never count the real PSF as stray light; drop regions that were entirely inside it
    region_masks = [
        np.where(psf_mask, 0, m).astype(np.uint8) for m in region_masks
    ]
    region_masks = [m for m in region_masks if np.any(m)]
    if not region_masks:
        logging.warning(f"Segmentation method {method!r} found no stray light regions")
    else:
        logging.info(f"Segmentation method {method!r} found {len(region_masks)} regions")

    _plot_segmentation(tv, foreground, region_masks, method)

    # turn the masks into a segment map: one integer map per region, region i labelled i+1
    segment_map = []
    for i, mask_this in enumerate(region_masks):
        segment_this = np.zeros(mask_this.shape, dtype=int)
        segment_this[mask_this > 0] = i + 1
        segment_map.append(segment_this)

    # update the masks and labels for each stray light region
    result_obj.segment_map = segment_map
    result_obj.stray_light_masks = region_masks
    result_obj.segmentation_method = method

    return result_obj


@pipeline_stage(
    name="stray_light_mask_real",
    depends_on=("centroid_2passes_oversample",),
    cluster=CLUSTER_STRAY,
    cluster_color=COLOR_STRAY,
    label="Mask the real PSF",
)
def stray_light_mask_real(result_obj, observing_config):
    """
    Mask the real PSF, so we can find the stray light.

    INPUTS:
    - result_obj (StrayLightResult): result object
    - observing_config (ObservingConfig): observing config

    OUTPUTS:
    - None; updates result_obj
    """

    # just make a mask over N*lambda/D for now''
    N_val = 5.0
    # wavel = float(observing_config['filter_name']['wavelength'])
    lambda_over_D = (
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
    mask = angular_distances < N_val * lambda_over_D

    # add the mask to the result object
    result_obj.real_psf_mask = mask

    return result_obj


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
