from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.signal import convolve2d
from matplotlib.colors import LogNorm
from astropy.visualization import ZScaleInterval
from astropy.modeling.models import GeneralSersic2D, Gaussian2D, Ring2D
import ipdb
import photutils
from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel, detect_sources, SourceCatalog

# generate a detector image with a Gaussian PSF in the middle
detector_array = np.zeros((2048, 2048))
# Define the PSF parameters
fwhm_pix = 20.0  # Full Width at Half Maximum in pixels
sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # Convert FWHM to sigma
#mu = 20
detector_array_xx, detector_array_yy = np.meshgrid(np.arange(2048), np.arange(2048))
# Create a 2D Gaussian PSF
center_x, center_y = 1024, 1024  # Center of the array
dst = np.sqrt((detector_array_xx - center_x) ** 2 + (detector_array_yy - center_y) ** 2)
# Normalization for a 2D Gaussian
normal = 1 / (2 * np.pi * sigma_pix**2)
# Calculate Gaussian filter centered in the array
exp_part = np.exp(-((dst) ** 2) / (2.0 * sigma_pix**2))
detector_array = 1 * exp_part/np.max(exp_part) + 0.01 * np.random.randn(2048, 2048)

#plt.imshow(detector_array, origin='lower')
#plt.title('PSF max: ' + str(np.max(detector_array)))
#plt.colorbar()
#plt.show()

def random_contiguous_stray_light(
    shape,
    n_shapes=(3, 8),
    seed=None,
    pixels_per_shape=(80, 600),
    shape_types=("circle",),
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
    shape_types : str or sequence[str]
        Which morphology(ies) to use for each random object. Allowed values are:
        "circle", "GeneralSersic2D", "Gaussian2D", "Ring2D".
        If a sequence is passed, each object randomly picks one from this list.
    growth_p : float
        Kept for backwards compatibility. Not used in the analytic-shape mode.
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

    allowed = {"circle", "GeneralSersic2D", "Gaussian2D", "Ring2D"}
    if isinstance(shape_types, str):
        shape_types = (shape_types,)
    shape_choices = tuple(shape_types)
    bad = [s for s in shape_choices if s not in allowed]
    if bad:
        raise ValueError(f"Unsupported shape_types: {bad}. Allowed: {sorted(allowed)}")
    if len(shape_choices) == 0:
        raise ValueError("shape_types must contain at least one allowed shape.")

    if isinstance(n_shapes, int):
        n_obj = n_shapes
    else:
        n_obj = int(rng.integers(n_shapes[0], n_shapes[1] + 1))

    occupied = np.zeros((ny, nx), dtype=bool)

    for obj_id in range(1, n_obj + 1):
        target = int(rng.integers(pixels_per_shape[0], pixels_per_shape[1] + 1))
        intensity = float(rng.uniform(*intensity_range))
        kind = shape_choices[rng.integers(len(shape_choices))]

        free = np.argwhere(~occupied)
        if free.size == 0:
            break
        y0, x0 = free[rng.integers(len(free))]
        y0 = int(y0)
        x0 = int(x0)

        # Approximate characteristic scale from target area
        r_char = max(2.0, np.sqrt(target / np.pi))
        half = int(np.clip(4.0 * r_char, 12, min(ny, nx) // 2))
        y1 = max(0, y0 - half)
        y2 = min(ny, y0 + half + 1)
        x1 = max(0, x0 - half)
        x2 = min(nx, x0 + half + 1)

        yy_loc, xx_loc = np.indices((y2 - y1, x2 - x1))
        yy_abs = yy_loc + y1
        xx_abs = xx_loc + x1

        if kind == "circle":
            rr = np.sqrt((xx_abs - x0) ** 2 + (yy_abs - y0) ** 2)
            mask_local = rr <= r_char

        elif kind == "Gaussian2D":
            sigma_x = max(1.5, r_char / rng.uniform(1.8, 2.8))
            sigma_y = max(1.5, r_char / rng.uniform(1.8, 2.8))
            theta = rng.uniform(0.0, np.pi)
            g = Gaussian2D(
                amplitude=1.0,
                x_mean=x0,
                y_mean=y0,
                x_stddev=sigma_x,
                y_stddev=sigma_y,
                theta=theta,
            )
            vals = g(xx_abs, yy_abs)
            mask_local = vals >= 0.2 * np.nanmax(vals)

        elif kind == "GeneralSersic2D":
            ellip = float(rng.uniform(0.0, 0.6))
            theta = float(rng.uniform(0.0, np.pi))
            n_ser = float(rng.uniform(0.8, 4.0))
            r_eff = max(1.5, r_char / rng.uniform(1.2, 2.2))
            s = GeneralSersic2D(
                amplitude=1.0,
                r_eff=r_eff,
                n=n_ser,
                x_0=x0,
                y_0=y0,
                ellip=ellip,
                theta=theta,
            )
            vals = s(xx_abs, yy_abs)
            mask_local = vals >= 0.2 * np.nanmax(vals)

        elif kind == "Ring2D":
            width = max(1.0, r_char * rng.uniform(0.12, 0.35))
            ring_r = max(width + 1.0, r_char)
            rmodel = Ring2D(
                amplitude=1.0,
                x_0=x0,
                y_0=y0,
                r_in=ring_r - width,
                width=width,
            )
            vals = rmodel(xx_abs, yy_abs)
            mask_local = vals >= 0.25 * np.nanmax(vals)

        else:
            raise RuntimeError(f"Unhandled shape kind: {kind}")

        if not np.any(mask_local):
            continue

        yy_hits = yy_abs[mask_local]
        xx_hits = xx_abs[mask_local]
        free_hits = ~occupied[yy_hits, xx_hits]
        yy_hits = yy_hits[free_hits]
        xx_hits = xx_hits[free_hits]
        if yy_hits.size == 0:
            continue

        stray[yy_hits, xx_hits] += intensity
        label_map[yy_hits, xx_hits] = obj_id
        occupied[yy_hits, xx_hits] = True

    if smooth_edges:
        # small 3x3 mean filter without scipy dependency
        pad = np.pad(stray, 1, mode="edge")
        out = np.zeros_like(stray)
        for j in range(ny):
            for i in range(nx):
                out[j, i] = pad[j:j+3, i:i+3].mean()
        stray = out

    return stray, label_map

def _colval(row, name, default=None):
    if name not in row.colnames:
        return default
    val = row[name]
    if hasattr(val, "value"):
        return float(val.value)
    return float(val)


_ALLOWED_SHAPE_TYPES = ("circle", "Gaussian2D", "GeneralSersic2D", "Ring2D")


def _shape_model_from_row(row, kind, xx, yy):
    """Build a 2D model patch for one catalog row and morphology kind."""
    x0 = _colval(row, "xcentroid")
    y0 = _colval(row, "ycentroid")
    amp = _colval(row, "max_value", default=None)
    if amp is None:
        amp = _colval(row, "segment_flux", default=0.0)
    amp = max(0.0, amp)

    a_sig = _colval(row, "semimajor_sigma", default=None)
    b_sig = _colval(row, "semiminor_sigma", default=None)
    theta = _colval(row, "orientation", default=0.0)
    area = _colval(row, "area", default=None)
    if area is None or area <= 0:
        area = 9.0
    r_eq = np.sqrt(area / np.pi)

    if kind == "circle":
        rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
        return (rr <= r_eq) * amp

    if kind == "Gaussian2D":
        sigma_x = max(1e-3, a_sig if a_sig and a_sig > 0 else r_eq / 2.0)
        sigma_y = max(1e-3, b_sig if b_sig and b_sig > 0 else r_eq / 2.0)
        g = Gaussian2D(
            amplitude=amp,
            x_mean=x0,
            y_mean=y0,
            x_stddev=sigma_x,
            y_stddev=sigma_y,
            theta=theta,
        )
        return g(xx, yy)

    if kind == "GeneralSersic2D":
        r_eff = max(1.0, a_sig if a_sig and a_sig > 0 else r_eq)
        if a_sig and b_sig and a_sig > 0:
            ellip = float(np.clip(1.0 - b_sig / a_sig, 0.0, 0.99))
        else:
            ellip = 0.0
        sersic_n = _colval(row, "sersic_n", default=2.0)
        s = GeneralSersic2D(
            amplitude=amp,
            r_eff=r_eff,
            n=sersic_n,
            x_0=x0,
            y_0=y0,
            ellip=ellip,
            theta=theta,
        )
        return s(xx, yy)

    if kind == "Ring2D":
        width = max(1.0, 0.2 * r_eq)
        r_in = max(0.5, r_eq - width)
        ring = Ring2D(
            amplitude=amp,
            x_0=x0,
            y_0=y0,
            r_in=r_in,
            width=width,
        )
        return ring(xx, yy)

    raise ValueError(f"Unsupported shape kind: {kind!r}")


def subtract_shapes_from_table(image_2d, tbl, shape_type="Gaussian2D"):
    """
    Build a model image from SourceCatalog table parameters and subtract it.

    Parameters
    ----------
    image_2d : ndarray
        Input image (e.g. background-subtracted readout).
    tbl : astropy Table
        SourceCatalog table from photutils.
    shape_type : str
        Morphology used for every detection: circle, Gaussian2D,
        GeneralSersic2D, or Ring2D. If tbl has a per-row ``shape_type``
        column, that value overrides this default for that row.

    Returns
    -------
    model_sum, cleaned : ndarray
        Combined model and image minus model.
    """
    if shape_type not in _ALLOWED_SHAPE_TYPES:
        raise ValueError(
            f"shape_type must be one of {_ALLOWED_SHAPE_TYPES}, got {shape_type!r}"
        )

    ny, nx = image_2d.shape
    yy, xx = np.indices((ny, nx))
    model_sum = np.zeros_like(image_2d, dtype=float)

    for row in tbl:
        kind = shape_type
        if "shape_type" in row.colnames:
            kind = str(row["shape_type"])
        if kind not in _ALLOWED_SHAPE_TYPES:
            raise ValueError(f"Unsupported shape_type in table row: {kind!r}")

        model_sum += _shape_model_from_row(row, kind, xx, yy)

    cleaned = image_2d - model_sum

    # use plot_subtraction_triple
    plot_subtraction_triple(image_2d, model_sum, cleaned)

    return model_sum, cleaned


def plot_subtraction_triple(
    input_image,
    model,
    residuals,
    *,
    input_title="Input image",
    model_title="Model from table",
    residual_title="Residuals (input - model)",
    figsize=(14, 4),
):
    """Plot input, model, and residuals in a 1x3 panel."""
    z = ZScaleInterval()
    vmin_i, vmax_i = z.get_limits(input_image)
    vmin_m, vmax_m = z.get_limits(model)
    vmin_r, vmax_r = z.get_limits(residuals)
    vlim_r = max(abs(vmin_r), abs(vmax_r))
    if not np.isfinite(vlim_r) or vlim_r == 0:
        vlim_r = 1e-15

    fig, axs = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    #im0 = axs[0].imshow(input_image, origin="lower", cmap="gray_r", vmin=vmin_i, vmax=vmax_i)
    im0 = axs[0].imshow(input_image, origin="lower", cmap="gray_r")
    axs[0].set_title(input_title)
    fig.colorbar(im0, ax=axs[0], fraction=0.046)

    im1 = axs[1].imshow(model, origin="lower", cmap="inferno", vmin=vmin_m, vmax=vmax_m)
    axs[1].set_title(model_title)
    fig.colorbar(im1, ax=axs[1], fraction=0.046)

    #im2 = axs[2].imshow(residuals, origin="lower", cmap="gray_r", vmin=vmin_i, vmax=vmax_i)
    im2 = axs[2].imshow(residuals, origin="lower", cmap="gray_r")
    axs[2].set_title(residual_title)
    fig.colorbar(im2, ax=axs[2], fraction=0.046)

    plt.show()
    return fig, axs


def detect_and_catalog_sources(
    data,
    detection_threshold_coeff=1.5,
    segment_map=None,
    *,
    subtract_background=True,
    bkg_box_size=(50, 50),
    bkg_filter_size=(3, 3),
    det_kernel_fwhm=3.0,
    det_kernel_size=5,
    npixels=10,
    print_catalog=True,
):
    """
    Background-subtract, detect sources (unless segment_map given), and build catalog.

    Parameters
    ----------
    data_array : ndarray
        Input 2D detector image.
    detection_threshold : float or None
        Detection threshold for ``detect_sources``. If None, uses
        ``1.5 * background_rms`` after background estimation.
    segment_map : SegmentationImage or None
        Precomputed segmentation map. If None, sources are detected using
        ``detection_threshold``.

    Returns
    -------
    data_bkg_subtracted : ndarray
        Background-subtracted image used for detection/cataloging.
    segment_map : SegmentationImage
        Segmentation map (detected or passed through).
    convolved_data : ndarray
        Smoothed image used for detection.
    catalog : SourceCatalog
        photutils source catalog object.
    table : astropy Table
        Tabular catalog from ``catalog.to_table()``.
    """

    if subtract_background:
        bkg_estimator = MedianBackground()
        bkg = Background2D(
            data,
            bkg_box_size,
            filter_size=bkg_filter_size,
            bkg_estimator=bkg_estimator,
        )
        data -= bkg.background
        detection_threshold = detection_threshold_coeff * bkg.background_rms

    det_kernel = make_2dgaussian_kernel(det_kernel_fwhm, size=det_kernel_size)
    convolved_data = convolve(data, det_kernel)

    if segment_map is None:
        segment_map = detect_sources(convolved_data, detection_threshold, npixels=npixels)
        print(segment_map)

    catalog = SourceCatalog(data, segment_map, convolved_data=convolved_data)
    table = catalog.to_table()
    if print_catalog:
        with np.printoptions(threshold=np.inf):
            print(table[:][table.colnames])

    return data, segment_map, convolved_data, catalog, table


# generate some stray light shapes
# circles
stray_rand_circles, labels = random_contiguous_stray_light(
    shape=detector_array.shape,
    n_shapes=(3, 4),
    seed=123,
    pixels_per_shape=(100, 1200),
    shape_types=("circle"),
    intensity_range=(0.1, 0.7),
)
net_readout_rand_circles = detector_array + stray_rand_circles

# Gaussian2D
stray_rand_gaussian2d, labels = random_contiguous_stray_light(
    shape=detector_array.shape,
    n_shapes=(3, 4),
    seed=123,
    pixels_per_shape=(100, 1200),
    shape_types=("Gaussian2D"),
    intensity_range=(0.1, 0.7),
)
net_readout_rand_gaussian2d = detector_array + stray_rand_gaussian2d

# GeneralSersic2D
stray_rand_sersic2d, labels = random_contiguous_stray_light(
    shape=detector_array.shape,
    n_shapes=(3, 4),
    seed=123,
    pixels_per_shape=(1200, 4000),
    shape_types=("GeneralSersic2D"),
    intensity_range=(1, 3),
)
net_readout_rand_sersic2d = detector_array + stray_rand_sersic2d

# Ring2D
stray_rand_ring2d, labels = random_contiguous_stray_light(
    shape=detector_array.shape,
    n_shapes=(3, 4),
    seed=123,
    pixels_per_shape=(100, 1200),
    shape_types=("Ring2D"),
    intensity_range=(0.1, 0.7),
)
net_readout_rand_ring2d = detector_array + stray_rand_ring2d

# plot all fake data
z = ZScaleInterval()
v1, v2 = z.get_limits(net_readout_rand_circles)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(stray_rand_circles, origin="lower", cmap="inferno", vmin=v1, vmax=v2)
ax[0].set_title("Random contiguous stray light")
ax[1].imshow(net_readout_rand_circles, origin="lower", cmap="gray_r", vmin=v1, vmax=v2)
ax[1].set_title("Readout + random stray light")
plt.tight_layout()
plt.show()

# gaussian2d
z = ZScaleInterval()
v1, v2 = z.get_limits(net_readout_rand_gaussian2d)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(stray_rand_gaussian2d, origin="lower", cmap="inferno", vmin=v1, vmax=v2)
ax[0].set_title("Random contiguous stray light")
ax[1].imshow(net_readout_rand_gaussian2d, origin="lower", cmap="gray_r", vmin=v1, vmax=v2)
ax[1].set_title("Readout + random stray light")
plt.tight_layout()
plt.show()

# GeneralSersic2D
z = ZScaleInterval()
v1, v2 = z.get_limits(net_readout_rand_sersic2d)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(stray_rand_sersic2d, origin="lower", cmap="inferno", vmin=v1, vmax=v2)
ax[0].set_title("Random contiguous stray light")
ax[1].imshow(net_readout_rand_sersic2d, origin="lower", cmap="gray_r", vmin=v1, vmax=v2)
ax[1].set_title("Readout + random stray light")
plt.tight_layout()
plt.show()

# Ring2D
z = ZScaleInterval()
v1, v2 = z.get_limits(net_readout_rand_ring2d)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(stray_rand_ring2d, origin="lower", cmap="inferno", vmin=v1, vmax=v2)
ax[0].set_title("Random contiguous stray light")
ax[1].imshow(net_readout_rand_ring2d, origin="lower", cmap="gray_r", vmin=v1, vmax=v2)
ax[1].set_title("Readout + random stray light")
plt.tight_layout()
plt.show()

########################################################
# find shapes and characterize them

########################################################
# circles test: detect sources
net_readout_rand_circles, segment_map_circles, convolved_data_circles, cat_circles, tbl_circles = detect_and_catalog_sources(
    net_readout_rand_circles,
    detection_threshold_coeff=1.5,
    segment_map=None,
)
# circles test: subtract sources
model_from_tbl_circles, net_readout_subtracted = subtract_shapes_from_table(
    net_readout_rand_circles, tbl_circles, shape_type="circle"
)
residuals_circles = net_readout_rand_circles - model_from_tbl_circles

########################################################
# Gaussian2D test: detect sources
net_readout_rand_gaussian2d, segment_map_gaussian2d, convolved_data_gaussian2d, cat_gaussian2d, tbl_gaussian2d = detect_and_catalog_sources(
    net_readout_rand_gaussian2d,
    detection_threshold_coeff=1.5,
    segment_map=None,
)
# Gaussian2D test: subtract sources
model_from_tbl_gaussian2d, net_readout_subtracted = subtract_shapes_from_table(
    net_readout_rand_gaussian2d, tbl_gaussian2d, shape_type="Gaussian2D"
)
residuals_gaussian2d = net_readout_rand_gaussian2d - model_from_tbl_gaussian2d

########################################################
# GeneralSersic2D test: detect sources
net_readout_rand_sersic2d, segment_map_sersic2d, convolved_data_sersic2d, cat_sersic2d, tbl_sersic2d = detect_and_catalog_sources(
    net_readout_rand_sersic2d,
    detection_threshold_coeff=1.5,
    segment_map=None,
)
# GeneralSersic2D test: subtract sources
model_from_tbl_sersic2d, net_readout_subtracted = subtract_shapes_from_table(
    net_readout_rand_sersic2d, tbl_sersic2d, shape_type="GeneralSersic2D"
)
residuals_sersic2d = net_readout_rand_sersic2d - model_from_tbl_sersic2d

########################################################
# Ring2D test: detect sources
net_readout_rand_ring2d, segment_map_ring2d, convolved_data_ring2d, cat_ring2d, tbl_ring2d = detect_and_catalog_sources(
    net_readout_rand_ring2d,
    detection_threshold_coeff=1.5,
    segment_map=None,
)
# Ring2D test: subtract sources
model_from_tbl_ring2d, net_readout_subtracted = subtract_shapes_from_table(
    net_readout_rand_ring2d, tbl_ring2d, shape_type="Ring2D"
)
residuals_ring2d = net_readout_rand_ring2d - model_from_tbl_ring2d