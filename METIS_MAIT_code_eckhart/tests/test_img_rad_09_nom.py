"""
Tests for the IMG-RAD-09 analysis (modules/backbone_img_rad_09_nom.py) on a synthetic
SIM-like file: populate_run -> measure_noise -> sens_10000s_check -> log_test_outcome.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "main_scripts", _ROOT / "misc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from modules import backbone_img_rad_09_nom as nom  # noqa: E402

OBSERVING = {
    "monochromatic_observing_filters_lm": {"Lp": 3.82e-6},
    "pixel_scales": {"img_lm": 5.47},
    "img_rad_09": {
        "flux_per_adu_per_s": {"img_lm": 2.0},
        "flux_unit": "test units",
        "aperture_radius_lambda_over_d": 1.0,
        "aperture_spacing_radii": 3.0,
        "telescope_diameter_m": 39.146,
        "mask_frac_of_median": 0.5,
        "mask_erosion_px": 4,
        "n_sigma": 3,
        "limits": {"SENS-10000S": {"noise_flux": 12.0, "exptime_s": 10000}},
    },
}


def write_sim_like_file(path, noise_flux, expected, dit=0.3, border=16, seed=0):
    """
    SIM-like file: RAW = B + D, BACKGROUND = B, BCKGD_SUBTED = D, with white noise D
    scaled so the aperture noise in flux units is noise_flux.
    """
    rng = np.random.default_rng(seed)
    shape = (512, 512)
    background = np.zeros(shape)
    background[border:-border, border:-border] = 10000.0
    radius = nom.aperture_radius_px(3.82e-6, 39.146, 5.47, 1.0)
    _, n_pix = nom.aperture_sums(
        np.zeros(shape), np.ones(shape, bool), radius, 3 * radius
    )
    # white noise: aperture noise (ADU/DIT) = sigma_px x sqrt(n_pix)
    sigma_px = noise_flux / 2.0 * dit / np.sqrt(n_pix)
    diff = rng.normal(0.0, sigma_px, shape)
    header = fits.Header(
        {
            "REQ_ID": "IMG-RAD-09 sensitivity",
            "CASE": expected.lower(),
            "EXPECTED": expected,
            "REQ_QTY": "noise_flux",
            "REQ_LIM": 12.0,
            "INJ_VAL": noise_flux,
            "FILTER": "Lp",
            "DIT": dit,
            "EXPTIME": 10000.0,
        }
    )
    fits.HDUList(
        [
            fits.PrimaryHDU(header=header),
            fits.ImageHDU(diff, name="BCKGD_SUBTED"),
            fits.ImageHDU(background + diff, name="RAW_READOUT"),
            fits.ImageHDU(background, name="BACKGROUND"),
        ]
    ).writeto(path)
    return str(path)


@pytest.mark.parametrize("noise_flux, expected", [(8.0, "PASS"), (16.0, "FAIL")])
def test_analysis_recovers_noise_and_verdict(tmp_path, noise_flux, expected):
    file_name = write_sim_like_file(tmp_path / "f.fits", noise_flux, expected)
    state = {
        "file_name_abs": file_name,
        "req_id": "SENS-10000S",
        "instrument": "Lp",
        "det_key": "img_lm",
        "case": expected.lower(),
        "results_write_dir": str(tmp_path),
    }
    result = nom.populate_run(state, OBSERVING)
    result = nom.measure_noise(result, OBSERVING)
    assert result.measured["noise_flux"] == pytest.approx(noise_flux, rel=0.05)
    result = nom.sens_10000s_check(result)
    result = nom.log_test_outcome(result)
    assert result.verdict_overall == expected and result.consistent is True
    assert nom.log_summary([result]) == 0


def test_noise_uncertainty_scales_with_apertures(tmp_path):
    file_name = write_sim_like_file(tmp_path / "f.fits", 8.0, "PASS")
    state = {
        "file_name_abs": file_name,
        "req_id": "SENS-10000S",
        "instrument": "Lp",
        "det_key": "img_lm",
        "case": "pass",
        "results_write_dir": str(tmp_path),
    }
    result = nom.measure_noise(nom.populate_run(state, OBSERVING), OBSERVING)
    n_ap = result.measured["n_apertures"]
    assert result.sigma["noise_flux"] == pytest.approx(
        result.measured["noise_flux"] * nom.MAD_SIGMA_REL_ERR / np.sqrt(n_ap)
    )
