"""Unit tests for the noise of the summed flux per spatial scale (METIS-9522 limit)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

# backbone_img_04_stray_light imports pipeline_registry from misc/ and lives in main_scripts/modules
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "main_scripts", _ROOT / "misc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from modules.backbone_img_04_stray_light import (  # noqa: E402
    area_noise_sigma,
    nan_box_mean,
)

SIGMA = 2.0


@pytest.fixture
def white_noise():
    return np.random.default_rng(1).normal(0.0, SIGMA, size=(512, 512))


def test_white_noise_scales_as_sqrt_area(white_noise):
    valid = np.ones(white_noise.shape, dtype=bool)
    for area in (1, 100, 400):
        assert area_noise_sigma(white_noise, valid, area) == pytest.approx(
            SIGMA * np.sqrt(area), rel=0.1
        )


def test_correlated_noise_exceeds_white_estimate(white_noise):
    # 3x3-smoothed noise (like inter-pixel capacitance, but stronger): the per-pixel
    # rms underestimates the noise of a sum over many pixels
    correlated = ndimage.uniform_filter(white_noise, size=3)
    valid = np.ones(correlated.shape, dtype=bool)
    area = 100
    sigma_pix = np.std(correlated)
    assert area_noise_sigma(correlated, valid, area) > 2.0 * sigma_pix * np.sqrt(area)


def test_masked_pixels_are_ignored(white_noise):
    image = white_noise.copy()
    valid = np.ones(image.shape, dtype=bool)
    # a bright 'region' that is masked out must not inflate the noise
    image[100:200, 100:200] += 1e4
    valid[100:200, 100:200] = False
    assert area_noise_sigma(image, valid, 100) == pytest.approx(
        SIGMA * 10.0, rel=0.1
    )


def test_cache_is_used(white_noise):
    valid = np.ones(white_noise.shape, dtype=bool)
    cache = {}
    area_noise_sigma(white_noise, valid, 100, cache)
    assert set(cache) == {10}
    cache[10] = 1.0  # a different cached value per box mean is returned as such
    assert area_noise_sigma(white_noise, valid, 100, cache) == pytest.approx(100.0)


def test_falls_back_to_white_noise_with_few_boxes():
    image = np.random.default_rng(2).normal(0.0, SIGMA, size=(30, 30))
    valid = np.ones(image.shape, dtype=bool)
    # boxes of side 20 on a 30x30 frame: far fewer than 20 samples
    assert area_noise_sigma(image, valid, 400) == pytest.approx(
        SIGMA * 20.0, rel=0.3
    )


def test_nan_box_mean_ignores_invalid():
    image = np.ones((20, 20))
    valid = np.ones(image.shape, dtype=bool)
    image[:, :10] = 100.0
    valid[:, :10] = False
    mean = nan_box_mean(image, valid, 5)
    assert np.allclose(mean[:, 12:], 1.0)
    assert np.all(np.isnan(mean[:, :7]))  # boxes with < half valid pixels
