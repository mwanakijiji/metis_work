from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.signal import convolve2d
from matplotlib.colors import LogNorm
from astropy.visualization import ZScaleInterval
from astropy.modeling.models import GeneralSersic2D, Gaussian2D, Ring2D


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
# dots
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
# dots
net_readout_rand_gaussian2d = detector_array + stray_rand_gaussian2d


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
stray_rand_sersic2d, labels = random_contiguous_stray_light(
    shape=detector_array.shape,
    n_shapes=(3, 4),
    seed=123,
    pixels_per_shape=(1200, 4000),
    shape_types=("GeneralSersic2D"),
    intensity_range=(1, 3),
)
# dots
net_readout_rand_sersic2d = detector_array + stray_rand_sersic2d

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
stray_rand_ring2d, labels = random_contiguous_stray_light(
    shape=detector_array.shape,
    n_shapes=(3, 4),
    seed=123,
    pixels_per_shape=(100, 1200),
    shape_types=("Ring2D"),
    intensity_range=(0.1, 0.7),
)
# dots
net_readout_rand_ring2d = detector_array + stray_rand_ring2d

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