"""WCU readout playground (LM / N), IMG-RAD-07 METIS-3291 RSRF.

Set the filter, masks and integration below, then run the script to get a ScopeSim
readout of the WCU and display it. Uses the same set-up as the SIM scripts
(`modules.sim_common`).

Settings:
- `obs_mode`: `wcu_img_lm`, `wcu_img_n`, or an LSS mode such as `wcu_lss_l`
- `fp_mask`: `open` (flat), `pinhole_lm`, `pinhole_n`, `grid_lm`
- `pp_mask`: e.g. `PPS-CFO2`, `SPM-LM`, `CLS-LM`, `SPM-N`, `CLS-N`;
  `psf_kernel`: `None` = `<pp_mask>_WCU`
- `nd_filter`: `None`, `ND_OD1` ... `ND_OD4`
- `dit` (s), `ndit`; `bb_aperture` (0 = closed, 1 = open); `bb_temp_k`

Wavelength solution (LSS modes only): ScopeSim's `spectral_traces` effect maps
focal-plane (x, y) [mm] to wavelength [um] (`trace.xy2lam`) and slit position xi
[arcsec] (`trace.xy2xi`); detector pixels are converted to mm with the IRDB detector
layout file. Dispersion is along the rows. The spectrum is summed over the illuminated
slit columns, and only rows inside the trace table's wavelength range are kept.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import ascii, fits
from astropy.modeling.models import BlackBody
from astropy.visualization import ZScaleInterval
from scipy.ndimage import gaussian_filter1d

import ipdb

# put main_scripts/ (for `modules`) on the path
ROOT = Path(__file__).resolve().parent
while not (ROOT / "main_scripts").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "main_scripts"))

from modules import sim_common  # noqa: E402
from modules.config_common import load_observing_config  # noqa: E402
from modules.paths import project_root  # noqa: E402

sim_common.setup_scopesim()
observing = load_observing_config(project_root())

# ---- settings ----
instrument = {
    "obs_mode": "wcu_lss_l",
    "obs_filter": "L_spec",
    "fp_mask": "pinhole_lm",
    "pp_mask": "PPS-CFO2",
    "psf_kernel": None,
    "nd_filter": None,
    "dit": 0.04,
    "ndit": 1,
    "intrapixel_capacitance": True,
}
bb_aperture = 1.0
bb_temp_k = 1000
subtract_dark = True  # also read out a dark (BB aperture closed) and subtract it

# ---- readout ----
metis = sim_common.build_optical_train(instrument)
metis["wcu_source"].set_temperature(bb_temp=bb_temp_k * u.K)

frame, header = sim_common.readout_frame(metis, bb_aperture, instrument)
dark = sim_common.readout_frame(metis, 0.0, instrument)[0] if subtract_dark else None

saturation = sim_common.saturation_of(
    observing, sim_common.detector_key(instrument["obs_mode"])
)
print(f"raw: median {np.median(frame):.4g}, max {np.max(frame):.4g} ADU "
      f"= {np.max(frame) / saturation:.3f} of saturation ({saturation:.0f} ADU)")

# ---- display ----
image = frame - dark if subtract_dark else frame
title = (f"{instrument['obs_filter']}, {instrument['fp_mask']}, {instrument['pp_mask']}, "
         f"ND {instrument['nd_filter']}, DIT {instrument['dit']} s"
         + (" (dark-subtracted)" if subtract_dark else ""))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
zscale = ZScaleInterval()
lo, hi = zscale.get_limits(image)
im = axes[0].imshow(image, origin="lower", vmin=lo, vmax=hi)
axes[0].set_title(title, fontsize=9)
fig.colorbar(im, ax=axes[0], label="ADU")
# zoom around the brightest pixel, e.g. a pinhole PSF (zscale of the cutout)
half = 40  # half-width of the zoom (pixels)
y0, x0 = np.unravel_index(np.argmax(image), image.shape)
ys, xs = slice(max(y0 - half, 0), y0 + half + 1), slice(max(x0 - half, 0), x0 + half + 1)
cut = image[ys, xs]
lo, hi = zscale.get_limits(cut)
im = axes[1].imshow(cut, origin="lower", vmin=lo, vmax=hi,
                    extent=(xs.start - 0.5, xs.stop - 0.5, ys.start - 0.5, ys.stop - 0.5))
axes[1].set_title(f"zoom on max at (x, y) = ({x0}, {y0}), zscale")
fig.colorbar(im, ax=axes[1], label="ADU")
plt.show()

# ---- save a FITS file like the SIM scripts ----
# header-only primary, then BCKGD_SUBTED (raw - background), RAW_READOUT and
# BACKGROUND (the dark)
out_file = ROOT / "data" / "frame.fits"
out_file.parent.mkdir(parents=True, exist_ok=True)

fits.HDUList([
    fits.PrimaryHDU(header=header),
    fits.ImageHDU(data=(frame - dark).astype(np.float32), name="BCKGD_SUBTED"),
    fits.ImageHDU(data=frame.astype(np.float32), name="RAW_READOUT"),
    fits.ImageHDU(data=dark.astype(np.float32), name="BACKGROUND"),
]).writeto(out_file, overwrite=True)
print(out_file)

# ---- wavelength solution (LSS modes) and the slit-collapsed spectrum ----
# pixel -> focal-plane mm, from the detector layout file in the IRDB
layout_file = {
    "img_lm": "FPA_metis_img_lm_layout.dat",
    "img_n": "FPA_metis_img_n_geosnap_layout.dat",
}[sim_common.detector_key(instrument["obs_mode"])]
layout = ascii.read(Path(sim_common.IRDB_DIR) / "METIS" / layout_file)
pix_mm = float(layout["pixel_size" if "pixel_size" in layout.colnames else "pixsize"][0])

# ScopeSim's fitted trace transforms: (x, y) [mm] -> wavelength [um] and slit
# position [arcsec]
trace = next(iter(metis["spectral_traces"].spectral_traces.values()))
ny, nx = frame.shape
cols, rows = np.meshgrid(np.arange(nx), np.arange(ny))
x_mm = (cols - (nx - 1) / 2) * pix_mm + float(layout["x_cen"][0])
y_mm = (rows - (ny - 1) / 2) * pix_mm + float(layout["y_cen"][0])
lam_map = trace.xy2lam(x_mm, y_mm, grid=False)  # um, per pixel
xi_map = trace.xy2xi(x_mm, y_mm, grid=False)  # arcsec along the slit, per pixel

# slit-collapsed spectrum: sum over the illuminated columns (dispersion is along the
# rows), keeping only rows inside the trace table's wavelength range (no extrapolation)
profile = np.median(image, axis=0)
slit_cols = profile > 0.5 * np.max(profile)
spectrum = image[:, slit_cols].sum(axis=1)
lam_rows = np.median(lam_map[:, slit_cols], axis=1)
in_range = (lam_rows >= trace.wave_min) & (lam_rows <= trace.wave_max)
print(f"trace {trace.trace_id}: {trace.wave_min:.3f}-{trace.wave_max:.3f} um; "
      f"{slit_cols.sum()} slit columns; "
      f"dispersion {np.median(np.diff(lam_rows[in_range])) * 1e3:.3f} nm/pix")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im = axes[0].imshow(lam_map, origin="lower")
axes[0].contour(slit_cols[None, :] & in_range[:, None], levels=[0.5], colors="w")
axes[0].set_title("wavelength per pixel (white: slit, in trace range)")
fig.colorbar(im, ax=axes[0], label="um")
axes[1].plot(lam_rows[in_range], spectrum[in_range])
axes[1].set_xlabel("wavelength [um]")
axes[1].set_ylabel("sum over slit [ADU]")
axes[1].set_title(
    f"{instrument['obs_mode']}, {instrument['obs_filter']}, {instrument['fp_mask']}"
)
plt.show()

# ---- a perfect source spectrum at the BB temperature, in ADU ----
# Planck photon intensity (the same form WCUSource uses, ph / s / m2 / arcsec2 / um),
# with unit throughput and QE, per row of the slit-summed spectrum above:
#   ADU = B_ph(lam, T) * TEL.area * pixel solid angle * n_slit_cols * dlam/pix
#         * DIT * NDIT / gain
# The ratio spectrum / src_adu is then the system's relative spectral response (RSRF).
# Not included: BB emissivity (WCU emiss_bb), BB -> integrating-sphere coupling and IS
# gain, bb_aperture, and the IS's own thermal emission (removed by the dark subtraction).
bb_scale = 1 * u.ph / (u.s * u.m**2 * u.arcsec**2 * u.um)
bb = BlackBody(bb_temp_k * u.K, scale=bb_scale)

# cmds values may be plain floats or already Quantities: u.Quantity handles both
area = u.Quantity(metis.cmds["!TEL.area"], u.m**2)
pix_scale = u.Quantity(metis.cmds["!INST.pixel_scale"], u.arcsec)
gain = float(metis.cmds["!DET.gain"])  # e- / ADU (depends on the readout mode)
n_cols = slit_cols.sum()
exptime = instrument["dit"] * instrument["ndit"] * u.s

lam_src = lam_rows[in_range] * u.um
dlam_pix = np.abs(np.gradient(lam_rows))[in_range] * u.um  # wavelength width per row

src_ph = (bb(lam_src) * area * pix_scale**2 * n_cols * dlam_pix * exptime).to(u.ph)
src_adu = src_ph.value / gain  # 1 photon = 1 e- (QE = 1)

# normalize the integral of the perfect spectrum to a scale factor of the measured spectrum (to pretend that there is a net throughput loss)
# Each row already holds counts integrated over its own dlam, so the integral over
# wavelength is the plain sum over rows.
src_scale_factor = 2.
src_adu_unit_tp = src_adu  # keep the unit-throughput version
src_norm = src_scale_factor * np.sum(spectrum[in_range]) / np.sum(src_adu_unit_tp)
src_adu = src_adu_unit_tp * src_norm
print(f"perfect spectrum scaled by {src_norm:.4g} so its integral is "
      f"{src_scale_factor} x the measured one")

# add a made-up, normalized placeholder function alpha_fcn to describe the variations to be expected from the instrument
# ---> RSRF propto S_empirical(lambda) = ( S_perfect(lambda) * alpha_fcn(lambda) )
# (this placeholder function makes use of the general, smoothed ripple shape of the filter being used)
# alpha_fcn = (filter curve, lightly smoothed) / (its broad in-band trend), i.e. only the
# ripple, taken in the passband (T > ALPHA_IN_BAND_FRAC * max(T)) and set to 1 outside
# it, then scaled so that integral(S_perfect * alpha_fcn) = 0.7 * integral(S_perfect).
ALPHA_RIPPLE_SIGMA_UM = 0.005  # light smoothing: keeps the ripple, removes noise
ALPHA_TREND_SIGMA_UM = 0.1  # broad smoothing: the in-band trend divided out
ALPHA_IN_BAND_FRAC = 0.9  # lower values let the steep band edges leak into the ripple

band = sim_common.detector_key(instrument["obs_mode"]).split("_")[-1]  # "lm" / "n"
filter_file = Path(sim_common.IRDB_DIR) / observing[
    f"polychromatic_observing_filters_{band}_leaf_name"
][instrument["obs_filter"]]
filter_tbl = ascii.read(filter_file)
filt_lam = np.asarray(filter_tbl["wavelength"], dtype=float)  # um
filt_t = np.asarray(filter_tbl["transmission"], dtype=float)
filt_step = np.median(np.diff(filt_lam))

filt_in_band = filt_t > ALPHA_IN_BAND_FRAC * np.max(filt_t)
filt_smooth = gaussian_filter1d(filt_t, ALPHA_RIPPLE_SIGMA_UM / filt_step)
# trend from in-band samples only (normalized convolution), so the band edges don't
# pull it down
in_band_weight = gaussian_filter1d(filt_in_band.astype(float), ALPHA_TREND_SIGMA_UM / filt_step)
filt_trend = gaussian_filter1d(filt_t * filt_in_band, ALPHA_TREND_SIGMA_UM / filt_step)
filt_trend = np.divide(filt_trend, in_band_weight, out=np.ones_like(filt_t),
                       where=in_band_weight > 0)
filt_ripple = np.where(filt_in_band, filt_smooth / filt_trend, 1.0)


ALPHA_INTEGRAL_FRAC = 0.7  # integral of S_perfect * alpha_fcn / integral of S_perfect
lam_um = lam_src.to_value(u.um)
alpha_in_band = np.interp(lam_um, filt_lam, filt_in_band.astype(float)) > 0.5


def _ripple(lam):
    return np.interp(lam, filt_lam, filt_ripple, left=1.0, right=1.0)


# scale so that sum(S_perfect * alpha_fcn) = ALPHA_INTEGRAL_FRAC * sum(S_perfect) over the
# spectrum's rows (each row holds counts integrated over its own dlam, so the sum is
# the integral over wavelength)
alpha_scale = ALPHA_INTEGRAL_FRAC * np.sum(src_adu) / np.sum(src_adu * _ripple(lam_um))


def alpha_fcn(lam_um):
    """Placeholder instrument variation: the filter's smoothed ripple, scaled so that
    the integral of S_perfect * alpha_fcn is ALPHA_INTEGRAL_FRAC of that of S_perfect."""
    return alpha_scale * _ripple(lam_um)


# alpha_fcn on the spectrum's wavelengths
alpha_vals = alpha_fcn(lam_um)
print(f"alpha_fcn scale {alpha_scale:.4g}: integral(S_perfect * alpha_fcn) / "
      f"integral(S_perfect) = {np.sum(src_adu * alpha_vals) / np.sum(src_adu):.3f}")

print(f"alpha_fcn from {filter_file.name}: in-band range "
      f"{alpha_vals[alpha_in_band].min():.4f}-{alpha_vals[alpha_in_band].max():.4f}")

# ---- RSRF ----
#   rsrf(lambda) = S_empirical(lambda) / ( S_perfect(lambda) * alpha_fcn(lambda) )
s_empirical = spectrum[in_range]  # S_empirical: measured, slit-summed spectrum [ADU/row]
s_perfect = src_adu  # S_perfect: BB at bb_temp_k, with the scale factor [ADU/row]
s_perfect_alpha = s_perfect * alpha_vals  # S_perfect * alpha_fcn [ADU/row]
rsrf = s_empirical / s_perfect_alpha

print(f"TEL.area {area:.1f}, pixel scale {pix_scale:.5f}, gain {gain} e-/ADU, "
      f"{n_cols} columns, exptime {exptime}")
print(f"S_perfect: {s_perfect.min():.4g}-{s_perfect.max():.4g} ADU/row; "
      f"rsrf median {np.median(rsrf):.3g}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(lam_src, s_empirical, label="S_empirical (measured, sum over slit)")
axes[0].plot(lam_src, s_perfect,
             label=f"S_perfect: BB {bb_temp_k} K, integral = {src_scale_factor} x S_empirical")
axes[0].plot(lam_src, s_perfect_alpha, label="S_perfect * alpha_fcn")
axes[0].set_yscale("log")
axes[0].set_xlabel("wavelength [um]")
axes[0].set_ylabel("ADU per row")
axes[0].legend()
axes[1].plot(lam_src, rsrf, label="rsrf = S_empirical / (S_perfect * alpha_fcn)")
axes[1].legend()
axes[1].set_xlabel("wavelength [um]")
axes[1].set_ylabel("ratio")
axes[1].set_title("rsrf = S_empirical / (S_perfect * alpha_fcn)")
plt.show()
