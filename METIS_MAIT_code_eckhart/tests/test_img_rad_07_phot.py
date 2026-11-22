"""
Unit tests for the IMG-RAD-07 analysis (modules/backbone_img_rad_07_phot.py) and the
shared ANALYSIS machinery (modules/analysis_common.py), on synthetic flats.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "main_scripts", _ROOT / "misc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from modules import analysis_common  # noqa: E402
from modules import backbone_img_rad_07_phot as phot  # noqa: E402
from modules import rsrf_common  # noqa: E402
from modules.sim_common import smooth_pattern  # noqa: E402

SHAPE = (512, 512)
BORDER = 32
LEVEL = 10000.0


def synthetic_flat(amplitude=0.0, kind="tilt_and_radial", noise=0.01, seed=0):
    """Flat of LEVEL x (1 + amplitude x pattern) in the field, zero border, noise."""
    rng = np.random.default_rng(seed)
    inner = (SHAPE[0] - 2 * BORDER, SHAPE[1] - 2 * BORDER)
    flat = np.zeros(SHAPE)
    flat[BORDER:-BORDER, BORDER:-BORDER] = LEVEL * (
        1.0 + amplitude * smooth_pattern(inner, kind)
    )
    return flat + rng.normal(0.0, noise * LEVEL, SHAPE)


def test_field_mask_drops_border():
    mask = phot.field_mask(synthetic_flat(), frac=0.5, erosion_px=4)
    assert not mask[: BORDER + 4].any() and not mask[:, -(BORDER + 4) :].any()
    assert mask[SHAPE[0] // 2, SHAPE[1] // 2]


def test_half_ptv_recovers_injected_amplitude():
    flat = synthetic_flat(amplitude=0.04)
    mask = phot.field_mask(flat, erosion_px=0)
    mean, sem = phot.block_stats(flat, mask, block_px=16)
    value, sigma, _ = phot.half_ptv(mean, sem)
    # block averaging smooths the corners a little
    assert value == pytest.approx(0.04, rel=0.15)
    assert 0 < sigma < 0.002


def test_flux_errors_alternating():
    exposure = np.arange(1, 16) / 15.0
    signs = np.where(np.arange(15) % 2 == 0, 1.0, -1.0)
    flux = 1000.0 * exposure * (1.0 + 0.01 * signs)
    errors, sigmas = phot.flux_errors(flux, np.zeros(15), exposure)
    # scale from the mean rate (8 high, 7 low steps)
    assert np.max(np.abs(errors)) == pytest.approx(0.01 * 16 / 15, rel=0.01)
    assert np.all(sigmas == 0)


def test_pattern_changes_recovers_ramp():
    ramp = np.linspace(-1.0, 1.0, 5)
    means, sems = [], []
    for k, r in enumerate(ramp):
        flat = synthetic_flat(amplitude=0.01 * r, kind="saddle", noise=0.001, seed=k)
        mask = phot.field_mask(flat, erosion_px=0)
        mean, sem = phot.block_stats(flat, mask, block_px=16)
        means.append(mean)
        sems.append(sem)
    value, sigma, step, change = phot.pattern_changes(means, sems)
    assert value == pytest.approx(0.01, rel=0.15)
    assert step in (0, 4) and change.shape == (5,) + means[0].shape
    assert sigma < 0.001


@pytest.mark.parametrize(
    "measured, sigma, expected",
    [(0.01, 0.001, "PASS"), (0.05, 0.001, "FAIL"), (0.029, 0.001, "INCONCLUSIVE")],
)
def test_verdict_with_noise(measured, sigma, expected):
    assert analysis_common.verdict_with_noise(measured, sigma, 0.03, 3) == expected


def test_combine_verdicts():
    assert analysis_common.combine_verdicts(["PASS", "PASS"]) == "PASS"
    assert analysis_common.combine_verdicts(["PASS", "INCONCLUSIVE"]) == "INCONCLUSIVE"
    assert analysis_common.combine_verdicts(["INCONCLUSIVE", "FAIL"]) == "FAIL"


def test_load_data_states(tmp_path):
    config = tmp_path / "states.yaml"
    config.write_text(
        "defaults: {results_write_dir: results/x/, block: 1}\n"
        "runs:\n"
        "  - {file_name_abs: data/a.fits}\n"
        "  - {file_name_abs: /abs/b.fits, block: 2}\n"
    )
    stem = str(tmp_path) + "/"
    states = analysis_common.load_data_states(
        stem, str(config), path_keys=("file_name_abs", "results_write_dir")
    )
    assert states[0]["file_name_abs"] == stem + "data/a.fits"
    assert states[0]["results_write_dir"] == stem + "results/x/"
    assert states[1]["file_name_abs"] == "/abs/b.fits" and states[1]["block"] == 2


def test_compare_with_expected():
    header = fits.Header({"EXPECTED": "PASS"})
    assert analysis_common.compare_with_expected("R", "PASS", header) is True
    assert analysis_common.compare_with_expected("R", "FAIL", header) is False
    assert analysis_common.compare_with_expected("R", "PASS", fits.Header()) is None


# ---------------------------------------------------------------- METIS-3291 (RSRF)

RSRF_SHAPE = (200, 64)
SLIT = slice(30, 34)  # illuminated slit columns


def synthetic_slit_frame(spectrum, background=1.0):
    """LSS frame: spectrum / n_cols in each SLIT column per row, faint elsewhere."""
    frame = np.full(RSRF_SHAPE, background)
    n_cols = SLIT.stop - SLIT.start
    frame[:, SLIT] = np.asarray(spectrum)[:, None] / n_cols
    return frame


def test_slit_columns_and_collapse():
    spectrum = np.linspace(1000.0, 2000.0, RSRF_SHAPE[0])
    frame = synthetic_slit_frame(spectrum, background=0.0)
    cols = rsrf_common.slit_columns(frame, 0.5)
    assert np.flatnonzero(cols).tolist() == list(range(SLIT.start, SLIT.stop))
    np.testing.assert_allclose(rsrf_common.collapse_slit(frame, cols), spectrum)


def test_rsrf_stats():
    values = np.array([0.9, 1.0, 1.1, 1.0, 50.0])
    evaluate = np.array([True, True, True, True, False])
    stats = phot.rsrf_stats(values, evaluate)
    assert stats["median"] == pytest.approx(1.0)
    assert stats["half_ptv"] == pytest.approx(0.1)
    assert stats["rel_rms"] == pytest.approx(np.std([0.9, 1.0, 1.1, 1.0]))
    assert stats["max"] == pytest.approx(1.1)


def test_metis_3291_characterize_recovers_true_rsrf(tmp_path):
    ny = RSRF_SHAPE[0]
    lam = np.linspace(2.9, 4.2, ny)
    in_range = np.ones(ny, bool)
    evaluate = in_range.copy()
    evaluate[:10] = False
    s_perfect = 1e4 * np.exp(-((lam - 3.5) ** 2))
    alpha = 0.7 * (1.0 + 0.05 * np.sin(20 * lam))
    rsrf_true = 0.8 * (1.0 + 0.02 * np.cos(5 * lam))
    frame = synthetic_slit_frame(rsrf_true * s_perfect * alpha, background=0.0)

    table = Table(
        {
            "ROW": np.arange(ny),
            "WAVELENGTH": lam,
            "IN_RANGE": in_range,
            "EVAL": evaluate,
            "S_PERFECT": s_perfect,
            "ALPHA": alpha,
            "RSRF_TRUE": rsrf_true,
        }
    )
    rsrf_hdu = fits.table_to_hdu(table)
    rsrf_hdu.name = "RSRF"
    primary = fits.PrimaryHDU()
    primary.header["NSLITCOL"] = SLIT.stop - SLIT.start
    file_name = tmp_path / "IMG_RAD_07_METIS-3291_L_spec_pass.fits"
    fits.HDUList(
        [
            primary,
            fits.ImageHDU(frame.astype(np.float32), name="BCKGD_SUBTED"),
            rsrf_hdu,
        ]
    ).writeto(file_name)

    result = phot.PhotRunResult(
        run_name="METIS-3291_L_spec_pass",
        req_id="METIS-3291",
        instrument="L_spec",
        case="pass",
        file_names=[str(file_name)],
        headers=[fits.getheader(file_name, 0)],
        results_dir=str(tmp_path / "results"),
        limits={"rsrf_error": 0.03},
        analysis_cfg={"rsrf": {"slit_frac": 0.5}},
    )
    result = phot.metis_3291_characterize(result)

    assert result.verdicts["METIS-3291"] == "NOT EVALUATED"
    assert result.measured["rsrf_vs_true"] < 1e-5
    np.testing.assert_allclose(result.rsrf["rsrf"], rsrf_true, rtol=1e-5)
    assert result.measured["rsrf_median"] == pytest.approx(
        np.median(rsrf_true[evaluate]), rel=1e-5
    )
    assert (tmp_path / "results" / "METIS-3291_L_spec_pass_rsrf.fits").exists()
    assert Path(phot.plot_run(result)[0]).exists()
    assert phot.log_test_outcome(result).consistent is None
