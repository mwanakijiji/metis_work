"""
Unit tests for the IMG-RAD-07 analysis (modules/backbone_img_rad_07_phot.py) and the
shared ANALYSIS machinery (modules/analysis_common.py), on synthetic flats.
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

from modules import analysis_common  # noqa: E402
from modules import backbone_img_rad_07_phot as phot  # noqa: E402
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
