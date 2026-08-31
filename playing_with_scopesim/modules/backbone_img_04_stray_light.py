import numpy as np
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


# class for containing information about a stray light region
@dataclass
class StrayLightRegion:
    label: int
    spatial_scale: str          # e.g. "point", "extended", "large"
    peak_irradiance: float
    total_flux: float
    area_pix: int
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
    real_psf_mask: np.ndarray | None = None
    segment_map: np.ndarray | None = None   # integer labels

    # bookkeeping
    centroids: Any | None = None      # from centroid_2passes_oversample
    regions: list[StrayLightRegion] = field(default_factory=list)

    # global quantities
    background_level: float | None = None
    background_rms: float | None = None


def populate_result_obj_info(result_obj, data_state, observing_config):
    '''
    Populate the result object with information from the data state.

    INPUTS:
    - result_obj (StrayLightResult): result object
    - data_state (dict): data state
    - observing_config (ObservingConfig): observing config

    OUTPUTS:
    - result_obj (StrayLightResult): result object
    '''

    # add the things from the data state
    for key, value in data_state.items():
        setattr(result_obj, key, value)

    # add the central wavelength for the filter
    filters = observing_config["monochromatic_observing_filters_lm"]
    try:
        result_obj.wavel_central = float(filters[result_obj.filter_name])  # in m
    except KeyError as exc:
        raise KeyError(
            f"No central wavelength for filter {result_obj.filter_name!r} "
            f"in monochromatic_observing_filters_lm"
        ) from exc

    # pixel scale: detector label (LM/N) → analysis key (img_lm/img_n) → mas/pixel
    detector_to_scale_key = observing_config["scope_sim_to_analysis"]
    try:
        scale_key = detector_to_scale_key[result_obj.detector]
    except KeyError as exc:
        raise KeyError(
            f"No ScopeSim→analysis mapping for detector {result_obj.detector!r} "
            f"in scope_sim_to_analysis"
        ) from exc
    try:
        result_obj.pixel_scale = float(observing_config["pixel_scales"][scale_key])
    except KeyError as exc:
        raise KeyError(
            f"No pixel scale for key {scale_key!r} in pixel_scales"
        ) from exc

    return result_obj


def centroid_2passes_oversample(
    result_obj, 
    config_coords_guesses_file_name, 
    psfs_subset="all", 
    oversample_factor=3, 
    grid_header=None, 
    centroid_box_size=41, 
    zoom_order=3, 
    centroid_func=centroid_2dg, 
    centroid_sources_impl=centroid_sources):
    '''
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
    '''

    # load the empirical readout
    #data_original, header = psf_grid_prep.load_fits_data(file_name=image_array, hdu_index=1)

    data_original = result_obj.image

    # load the config file with the coordinates guesses
    coords_guesses = helpers.load_config_and_pipe(config_file_choice=config_coords_guesses_file_name, print_one_line=False)

    # 1st pass: centroid with photutils
    prep = psf_grid_prep.oversample_1st_pass_centroid(data_original, coords_guesses)

    # 2nd pass: centroid with Gaussian fit
    centroid_post_2nd_pass = psf_grid_prep.refine_2nd_pass_centroids(data_original, prep)

    #return CentroidResult(prep=prep, refined=results)

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
    nbrs = [(-1, -1), (-1, 0), (-1, 1),
            ( 0, -1),          ( 0, 1),
            ( 1, -1), ( 1, 0), ( 1, 1)]

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
                if 0 <= yy < ny and 0 <= xx < nx and not occupied[yy, xx] and (yy, xx) not in pixels:
                    cands.append((yy, xx))

            if not cands:
                frontier.discard((cy, cx))
                continue

            yy, xx = cands[rng.integers(len(cands))]
            pixels.add((yy, xx))
            frontier.add((yy, xx))

            # if source is boxed in now, drop it
            has_free_nbr = any(
                0 <= cy + dy < ny and 0 <= cx + dx < nx and
                (not occupied[cy + dy, cx + dx]) and ((cy + dy, cx + dx) not in pixels)
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
                out[j, i] = pad[j:j+3, i:i+3].mean()
        stray = out

    return stray, label_map


# options for how to segment the stray light
HoughVariant = Literal["circle", "active"]

def stray_light_segmentation(
    result_obj,
    option: str = "hough_transform",
    hough_variant: HoughVariant = "circle",
):
    '''
    Segment the stray light.

    Ideas and methods herein are from Eric Pantin's notebook for instructional purposes. Many knobs remain
    to be tuned with real data.
    '''

    # apply the PSF mask to the image
    result_obj_masked = result_obj.image.astype(float, copy=True)
    result_obj_masked[result_obj.real_psf_mask.astype(bool)] = np.nan

    # fill the nans with zeros to avoid choking stats; will reapply mask later
    result_obj_masked[np.isnan(result_obj_masked)] = 0.0
    # to avoid choking stats (redundant?)
    finite = np.isfinite(result_obj_masked)
    finite_vals = result_obj_masked[finite]

    '''
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(result_obj_masked, cmap='gray', origin='lower')
    ax.set_title('Masked Image')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(result_obj_masked.ravel(), bins=100)
    plt.show()
    '''

    if option == 'threshold':
        #logging.info('Using thresholding method')
        # thresholds from finite pixels only (Otsu/cv2 do not accept NaNs)
        mean = float(np.mean(finite_vals))
        median = float(np.median(finite_vals))
        otsu_threshold = float(ski.filters.threshold_otsu(finite_vals))

        # keep full 2D shape; masked/NaN pixels stay False (0)
        #ret, thresh1 = cv2.threshold(result_obj_masked, mean, 255, cv2.THRESH_BINARY)
        ret, image_passed = cv2.threshold(result_obj_masked, 1.e3*median, 255, cv2.THRESH_BINARY) ## ## TODO: make coeff in front of median a parameter
        #ret, thresh3 = cv2.threshold(result_obj_masked, otsu_threshold, 255, cv2.THRESH_BINARY)
        #thresh1 = (finite & (result_obj_masked > mean)).astype(np.uint8) * 10
        #thresh2 = (finite & (result_obj_masked > median)).astype(np.uint8) * 10
        #thresh3 = (finite & (result_obj_masked > otsu_threshold)).astype(np.uint8) * 10

    elif option == 'hough_transform':

        # fit initial circles to blobs

        # Bright circular stray-light blobs: intensity → clean binary → fit circles.
        # (Edge-Sobel + Hough on a zero-filled PSF hole finds noise and hole-boundary rings.)
        psf_mask = result_obj.real_psf_mask.astype(bool)
        filled = result_obj.image.astype(float, copy=True)
        bg = float(np.nanmedian(filled[~psf_mask])) if np.any(~psf_mask) else 0.0
        filled[psf_mask] = bg

        # TV denoise on normalized data (smaller weight => more denoise)
        vmin = float(np.min(filled))
        vmax = float(np.max(filled))
        scale = vmax - vmin if vmax > vmin else 1.0
        tv = ski.restoration.denoise_tv_bregman((filled - vmin) / scale, weight=1.0)
        tv = tv * scale + vmin
        ipdb.set_trace()

        # Sobel edge detection
        gx = cv2.Sobel(tv, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(tv, cv2.CV_64F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        ipdb.set_trace()

        # Keep only strong edges so Hough isn't swamped by noise
        thr = float(np.percentile(mag, 99.8))
        edges = (mag > thr).astype(np.uint8) * 255
        image_passed = edges.astype(float)
        ipdb.set_trace()

        # Remove speckles, fill small gaps in real circles
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ipdb.set_trace()
        # remove a bit more noise; 'erosion followed by dilation'
        binary = cv2.morphologyEx(image_passed, cv2.MORPH_OPEN, kernel)
        ipdb.set_trace()
        # docs.opencv.org: 'reverse of Opening, Dilation followed by Erosion. It is useful in closing small holes inside the foreground objects, or small black points on the object'
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        ipdb.set_trace()

        # Fit a circle to each connected component (more stable than Hough on noisy edges)
        binary = (binary > 0).astype(np.uint8) # convert to binary image to prevent problems with connectedComponentsWithStats
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_area = 50  # pixels; drop noise crumbs
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
            if circularity < 0.4:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            circles_xyr.append((float(cx), float(cy), float(radius), area, circularity))

        print(f"Detected {len(circles_xyr)} circular regions (thr={thr:.4g})")
        for cx, cy, radius, area, circ in circles_xyr:
            print(f"  center=({cx:.1f}, {cy:.1f}) r={radius:.1f} area={area} circ={circ:.2f}")

        # fyi plot
        fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        axs[0].imshow(tv, origin="lower", cmap="gray")
        axs[0].set_title("Denoised")
        axs[1].imshow(binary, origin="lower", cmap="gray")
        axs[1].set_title("Bright mask")
        axs[2].imshow(tv, origin="lower", cmap="gray")
        for cx, cy, radius, *_ in circles_xyr:
            circ_patch = plt.Circle((cx, cy), radius, fill=False, color="lime", lw=2)
            axs[2].add_patch(circ_patch)
            axs[2].plot(cx, cy, "r+", ms=8)
        axs[2].set_title("Fitted circles")
        for ax in axs:
            ax.set_aspect("equal")
        plt.show()

        # simple circle fitting
        if hough_variant == "circle":

            # make a mask for each circle
            region_masks = []
            for cx, cy, radius, *_ in circles_xyr:
                region_mask = np.zeros(tv.shape, dtype=np.uint8)
                cv2.circle(region_mask, (int(cx), int(cy)), int(radius), 255, -1)
                region_masks.append(region_mask)
            logging.info('Using simple circle fitting for segmentation')

        elif hough_variant == "active":

            logging.info('Using active contour fitting for segmentation')

            mask = np.zeros(tv.shape, dtype=np.uint8)
            expand_px = 5
            expand_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * expand_px + 1, 2 * expand_px + 1)
            )
            snakes = []
            expanded_contours = []

            for circle in circles_xyr:
                center = (circle[0], circle[1])  # Circle center
                radius = circle[2]              # Circle radius

                # Generate an initial circular contour
                theta = np.linspace(0, 2 * np.pi, 100)
                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta)
                init = np.array([y, x]).T # initial snake coords

                # Apply active contour model
                tv_blurred = ski.filters.gaussian(tv, sigma=0.5)
                # make a snake
                # alpha: length shape; beta: smoothness; gamma: stepping parameter
                snake = active_contour(tv_blurred, init, alpha=0.05, beta=3, gamma=0.1)

                # Convert the snake (refined contour) to an integer format and swap coordinates for OpenCV
                snake_int = np.array([(int(point[1]), int(point[0])) for point in snake], dtype=np.int32)

                # init the component
                comp = np.zeros(tv.shape, dtype=np.uint8)
                # fill the contour on the mask
                cv2.fillPoly(comp, [snake_int], 255)
                # expand the contour, to make sure we capture all of the stray light
                comp_exp = cv2.dilate(comp, expand_kernel, iterations=1)
                mask = np.maximum(mask, comp_exp)

                snakes.append(snake)
                cnts, _ = cv2.findContours(comp_exp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                expanded_contours.extend(cnts) # add all elements as separate iterables into the list

            ipdb.set_trace()
            # make a mask for each expanded contour
            region_masks = []
            for contour in expanded_contours:
                region_mask_this = np.zeros(tv.shape, dtype=np.uint8)
                cv2.fillPoly(region_mask_this, [contour], 255)
                region_masks.append(region_mask_this)
            ipdb.set_trace()

            '''
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(tv, origin="lower", cmap="gray")
            for snake in snakes:
                ax.plot(snake[:, 1], snake[:, 0], "r-", lw=1.5)
            for cnt in expanded_contours:
                ax.plot(cnt[:, 0, 0], cnt[:, 0, 1], "lime", lw=2)
            ax.set_title("Active contours (red) and expanded boundaries (lime)")
            ax.set_aspect("equal")
            plt.show()
            '''

    #image_passed = mask.astype(float)

    # mask the science image with the stray light region masks
    result_obj_stray_light_masked = result_obj_masked.astype(float)

    # loop over stray light regions and apply each as a patch of nans to the science image
    for region_mask_this in region_masks:
        region_mask_this_nan = region_mask_this.copy().astype(float) # float to allow nans
        idx = region_mask_this_nan == 255 # indices of stray light pixels
        result_obj_stray_light_masked[idx] = np.nan # make stray light nans
   
        #result_obj_stray_light_masked *= np.nan_to_num(region_mask_this)
    # extract the stray light fluxes from inside the region masks
    ipdb.set_trace()

    # reapply psf mask
    #image_passed = binary.astype(float) * 255.0
    #image_passed[psf_mask] = np.nan



    

    ipdb.set_trace()

    # reapply the psf mask to the thresholded image
    image_passed[result_obj.real_psf_mask.astype(bool)] = np.nan

    cmap = plt.cm.gray.copy()
    cmap.set_bad("purple") # these are the psf masked pixels

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image_passed, cmap=cmap, origin='lower')
    ax.set_title(f'Thresholded Image')
    plt.colorbar(ax.images[0], ax=ax)
    plt.savefig(f'junk.png')

    return result_obj


def stray_light_mask_real(result_obj, observing_config):
    '''
    Mask the real PSF, so we can find the stray light.

    INPUTS:
    - result_obj (StrayLightResult): result object
    - observing_config (ObservingConfig): observing config

    OUTPUTS:
    - None; updates result_obj
    '''

    # just make a mask over N*lambda/D for now''
    N_val = 5.
    #wavel = float(observing_config['filter_name']['wavelength'])
    lambda_over_D = 206265. * float(result_obj.wavel_central) / float(observing_config['D_aperture']['full']) # in arcsec

    # make a 2D array of the distances from the centroid, units pixels
    del_x_pix = np.arange(result_obj.image.shape[1]) - result_obj.centroids['x_center_pix_fullarray_normsamp']
    del_y_pix = np.arange(result_obj.image.shape[0]) - result_obj.centroids['y_center_pix_fullarray_normsamp']
    xx_pix, yy_pix = np.meshgrid(del_x_pix, del_y_pix)

    xx_arcsec = xx_pix * result_obj.pixel_scale / 1000 # /1000 because pixel scale is in mas/pixel
    yy_arcsec = yy_pix * result_obj.pixel_scale / 1000

    angular_distances = np.sqrt(xx_arcsec**2 + yy_arcsec**2)
    mask = angular_distances < N_val*lambda_over_D

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

    in_outer = (xx - cx) ** 2 + (yy - cy) ** 2 <= r_outer ** 2
    in_cut = (xx - cut_cx) ** 2 + (yy - cut_cy) ** 2 <= r_outer ** 2
    mask = in_outer & ~in_cut

    out[mask] += amplitude

    return out