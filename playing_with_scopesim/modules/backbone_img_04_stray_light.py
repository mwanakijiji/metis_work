import numpy as np
from . import psf_grid_prep, helpers
from photutils.centroids import centroid_2dg, centroid_sources
import ipdb
from dataclasses import dataclass, field
from typing import Any
import astropy.io.fits as fits


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


def stray_light_mask_real(result_obj, observing_config):
    '''
    Mask the real PSF, so we can find the stray light.

    INPUTS:
    - result_obj (StrayLightResult): result object
    - observing_config (ObservingConfig): observing config

    OUTPUTS:
    - None; updates result_obj
    '''

    # just make a mask over 3*lambda/D for now''
    #wavel = float(observing_config['filter_name']['wavelength'])
    lambda_over_D = 206265. * float(result_obj.wavel_central) / float(observing_config['D_aperture']['full']) # in arcsec

    # make a 2D array of the distances from the centroid, units pixels
    del_x_pix = np.arange(result_obj.image.shape[1]) - result_obj.centroids['x_center_pix_fullarray_normsamp']
    del_y_pix = np.arange(result_obj.image.shape[0]) - result_obj.centroids['y_center_pix_fullarray_normsamp']
    xx_pix, yy_pix = np.meshgrid(del_x_pix, del_y_pix)

    xx_arcsec = xx_pix * result_obj.pixel_scale / 1000 # /1000 because pixel scale is in mas/pixel
    yy_arcsec = yy_pix * result_obj.pixel_scale / 1000

    angular_distances = np.sqrt(xx_arcsec**2 + yy_arcsec**2)
    mask = angular_distances < 3.*lambda_over_D

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