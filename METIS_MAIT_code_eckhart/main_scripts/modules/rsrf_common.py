"""
Relative spectral response function (RSRF) machinery for IMG-RAD-07 METIS-3291, shared
by the SIM and the ANALYSIS (from dev_notebooks/dev_IMG_RAD_07_rsrf.py).

Naming convention (all spectra in ADU per detector row; dispersion is along the rows):
- S_empirical(lambda): the measured spectrum, summed over the illuminated slit columns
- S_perfect(lambda): a black body at the WCU BB temperature, with unit throughput and
  QE, scaled so that its integral is src_scale_factor x that of S_empirical (to
  pretend there is a net throughput loss)
- alpha_fcn(lambda): placeholder for the spectral variation expected from the
  instrument: the observing filter's smoothed ripple, scaled so that
  integral(S_perfect * alpha_fcn) = alpha_integral_frac x integral(S_perfect)
- rsrf(lambda) = S_empirical(lambda) / ( S_perfect(lambda) * alpha_fcn(lambda) )

Each row holds counts integrated over its own wavelength width, so an integral over
wavelength is the plain sum over rows.

Wavelength solution: ScopeSim's spectral_traces effect maps focal-plane (x, y) [mm] to
wavelength [um] (trace.xy2lam) and slit position xi [arcsec] (trace.xy2xi); detector
pixels are converted to mm with the IRDB detector layout file.
"""

# astropy.units members (u.um, u.ph, ...) are created at import time
# pylint: disable=no-member

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import ascii as ascii_io
from astropy.modeling.models import BlackBody
from scipy.ndimage import gaussian_filter1d

# detector key -> IRDB detector layout file (pixel size and centre in mm)
LAYOUT_FILES = {
    "img_lm": "FPA_metis_img_lm_layout.dat",
    "img_n": "FPA_metis_img_n_geosnap_layout.dat",
}

# black-body photon intensity unit, as in ScopeSim's WCUSource
BB_SCALE = 1 * u.ph / (u.s * u.m**2 * u.arcsec**2 * u.um)


@dataclass
class SlitSpectrum:
    """Slit-collapsed spectrum of a frame (one entry per detector row)."""

    slit_cols: np.ndarray  # bool (nx,): illuminated slit columns
    spectrum: np.ndarray  # (ny,): sum over the slit columns [ADU]
    lam_rows: np.ndarray  # (ny,): median wavelength of the slit pixels per row [um]
    in_range: np.ndarray  # bool (ny,): row inside the trace table's wavelength range


def lss_wavelength_map(metis, shape, det_key, irdb_dir):
    """
    Wavelength and slit position of every detector pixel, from the first spectral
    trace of the OpticalTrain.

    INPUTS:
    - metis (OpticalTrain): in an LSS mode (with a 'spectral_traces' effect)
    - shape (tuple): (ny, nx) of the detector frame
    - det_key (str): 'img_lm' or 'img_n' (selects the detector layout file)
    - irdb_dir (str): IRDB location

    OUTPUTS:
    - ndarray: wavelength per pixel [um]
    - ndarray: slit position xi per pixel [arcsec]
    - trace: the ScopeSim spectral trace used (wave_min, wave_max, trace_id)
    """
    layout = ascii_io.read(Path(irdb_dir) / "METIS" / LAYOUT_FILES[det_key])
    pix_mm = float(
        layout["pixel_size" if "pixel_size" in layout.colnames else "pixsize"][0]
    )
    trace = next(iter(metis["spectral_traces"].spectral_traces.values()))
    ny, nx = shape
    cols, rows = np.meshgrid(np.arange(nx), np.arange(ny))
    x_mm = (cols - (nx - 1) / 2) * pix_mm + float(layout["x_cen"][0])
    y_mm = (rows - (ny - 1) / 2) * pix_mm + float(layout["y_cen"][0])
    lam_map = np.asarray(trace.xy2lam(x_mm, y_mm, grid=False), dtype=float)
    xi_map = np.asarray(trace.xy2xi(x_mm, y_mm, grid=False), dtype=float)
    return lam_map, xi_map, trace


def slit_columns(image, slit_frac=0.5):
    """Illuminated slit columns: the median over the rows > slit_frac x its maximum."""
    profile = np.median(image, axis=0)
    return profile > slit_frac * np.max(profile)


def collapse_slit(image, slit_cols):
    """S_empirical: the sum over the slit columns, per row."""
    return np.asarray(image)[:, slit_cols].sum(axis=1)


def slit_spectrum(image, lam_map, wave_min, wave_max, slit_frac=0.5):
    """
    Slit-collapsed spectrum: the sum over the illuminated columns (slit_columns), with
    each row's median slit wavelength; rows are flagged in_range if inside
    [wave_min, wave_max] (no extrapolation of the trace).

    OUTPUTS:
    - SlitSpectrum
    """
    slit_cols = slit_columns(image, slit_frac)
    spectrum = collapse_slit(image, slit_cols)
    lam_rows = np.median(lam_map[:, slit_cols], axis=1)
    in_range = (lam_rows >= wave_min) & (lam_rows <= wave_max)
    return SlitSpectrum(slit_cols, spectrum, lam_rows, in_range)


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def perfect_bb_spectrum(
    lam_rows_um, bb_temp, area, pixel_scale, gain, n_cols, exptime_s
):
    """
    Black-body spectrum through a system of unit throughput and QE, in ADU per row:
      B_ph(lam, T) * area * pixel solid angle * n_cols * dlam per row * exptime / gain
    (1 photon = 1 e-). Not included: BB emissivity, BB -> integrating-sphere coupling
    and gain, BB aperture, the sphere's own thermal emission.

    INPUTS:
    - lam_rows_um (ndarray): wavelength per row [um] (all rows, so the row widths
      np.gradient(lam_rows_um) are defined everywhere)
    - bb_temp (Quantity or float): BB temperature [K]
    - area, pixel_scale: Quantities (or floats in m2, arcsec), e.g. from
      metis.cmds['!TEL.area'], metis.cmds['!INST.pixel_scale']
    - gain (float): e- / ADU
    - n_cols (int): number of slit columns summed
    - exptime_s (float): DIT x NDIT [s]

    OUTPUTS:
    - ndarray: ADU per row
    """
    black_body = BlackBody(u.Quantity(bb_temp, u.K), scale=BB_SCALE)
    lam = np.asarray(lam_rows_um, dtype=float) * u.um
    dlam = np.abs(np.gradient(np.asarray(lam_rows_um, dtype=float))) * u.um
    photons = (
        black_body(lam)
        * u.Quantity(area, u.m**2)
        * u.Quantity(pixel_scale, u.arcsec) ** 2
        * int(n_cols)
        * dlam
        * (float(exptime_s) * u.s)
    ).to(u.ph)
    return photons.value / float(gain)


def scale_to_integral(s_unit, s_empirical, factor):
    """
    S_perfect: s_unit scaled so that sum(S_perfect) = factor x sum(s_empirical) (sums
    over the rows given).

    OUTPUTS:
    - ndarray: S_perfect
    - float: the scale applied
    """
    norm = float(factor) * float(np.sum(s_empirical)) / float(np.sum(s_unit))
    return s_unit * norm, norm


def filter_file_of(observing, det_key, obs_filter, irdb_dir):
    """Filter curve file of obs_filter (polychromatic_observing_filters_<band>_...)."""
    band = det_key.split("_")[-1]  # 'lm' / 'n'
    leaf = observing[f"polychromatic_observing_filters_{band}_leaf_name"][obs_filter]
    return Path(irdb_dir) / leaf


def filter_ripple(filter_file, ripple_sigma_um, trend_sigma_um, in_band_frac):
    """
    Ripple of a filter curve: (curve, lightly smoothed) / (its broad in-band trend),
    taken in the passband (T > in_band_frac x max(T)) and 1 outside it. The trend is
    a normalized convolution of the in-band samples only, so the band edges don't pull
    it down; in_band_frac ~0.9 keeps the steep edges out of the ripple.

    INPUTS:
    - filter_file (Path): table with columns wavelength [um], transmission
    - ripple_sigma_um (float): light smoothing (keeps the ripple, removes noise)
    - trend_sigma_um (float): broad smoothing (the trend divided out)
    - in_band_frac (float): passband threshold, as a fraction of max(T)

    OUTPUTS:
    - ndarray: filter wavelengths [um]
    - ndarray: ripple (~1 in band, 1 outside)
    - ndarray: bool, in the passband
    """
    table = ascii_io.read(filter_file)
    filt_lam = np.asarray(table["wavelength"], dtype=float)
    filt_t = np.asarray(table["transmission"], dtype=float)
    step = float(np.median(np.diff(filt_lam)))

    in_band = filt_t > float(in_band_frac) * np.max(filt_t)
    smooth = gaussian_filter1d(filt_t, float(ripple_sigma_um) / step)
    weight = gaussian_filter1d(in_band.astype(float), float(trend_sigma_um) / step)
    trend = gaussian_filter1d(filt_t * in_band, float(trend_sigma_um) / step)
    trend = np.divide(trend, weight, out=np.ones_like(filt_t), where=weight > 0)
    ripple = np.where(in_band, smooth / trend, 1.0)
    return filt_lam, ripple, in_band


def make_alpha_fcn(filt_lam, ripple, lam_um, s_perfect, integral_frac):
    """
    alpha_fcn(lambda) = alpha_scale x ripple(lambda), with alpha_scale such that
    sum(S_perfect * alpha_fcn) = integral_frac x sum(S_perfect) over the rows lam_um.

    OUTPUTS:
    - callable: alpha_fcn(lam_um)
    - float: alpha_scale
    """

    def ripple_at(lam):
        return np.interp(lam, filt_lam, ripple, left=1.0, right=1.0)

    alpha_scale = (
        float(integral_frac)
        * float(np.sum(s_perfect))
        / float(np.sum(s_perfect * ripple_at(lam_um)))
    )

    def alpha_fcn(lam):
        """Placeholder instrument variation: the filter's scaled, smoothed ripple."""
        return alpha_scale * ripple_at(lam)

    return alpha_fcn, alpha_scale


def eval_rows(lam_rows, in_range, filt_lam, filt_in_band):
    """
    Rows on which the RSRF is evaluated: inside the trace's wavelength range and in
    the filter passband (stands in for the METIS-1090 wavelength range).
    """
    in_band = np.interp(lam_rows, filt_lam, filt_in_band.astype(float)) > 0.5
    return np.asarray(in_range, bool) & in_band


def rsrf(s_empirical, s_perfect, alpha):
    """rsrf = S_empirical / (S_perfect * alpha_fcn)."""
    return np.asarray(s_empirical, float) / (np.asarray(s_perfect) * np.asarray(alpha))
