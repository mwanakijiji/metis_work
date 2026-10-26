"""Unit tests for the IMG-OPT-06 METIS-5838 verdict (detection vs. lower limit)."""

import sys
from pathlib import Path

import numpy as np
import pytest

# the backbones import pipeline_registry from misc/ and live in main_scripts/modules
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "main_scripts", _ROOT / "misc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from modules.backbone_img_opt_06_out_of_field import (  # noqa: E402
    _attenuation_verdict,
    make_out_of_field_scatter,
)

PEAK = 1e5
REQ = 1e4


def test_detected_and_attenuated_enough():
    # detected (above threshold) at 2e-5 of the peak: attenuation 5e4
    att, lower, verdict = _attenuation_verdict(PEAK, 2.0, threshold=1.0, attenuation_req=REQ)
    assert att == pytest.approx(5e4) and not lower and verdict == "PASS"


def test_detected_and_too_bright():
    att, lower, verdict = _attenuation_verdict(PEAK, 100.0, threshold=1.0, attenuation_req=REQ)
    assert att == pytest.approx(1e3) and not lower and verdict == "FAIL"


def test_not_detected_deep_enough():
    # nothing above a threshold of 5 counts: attenuation > 2e4 >= 1e4
    att, lower, verdict = _attenuation_verdict(PEAK, 4.0, threshold=5.0, attenuation_req=REQ)
    assert att == pytest.approx(2e4) and lower and verdict == "PASS"


def test_not_detected_too_shallow():
    # nothing above a threshold of 50 counts: only attenuation > 2e3 can be shown
    att, lower, verdict = _attenuation_verdict(PEAK, 40.0, threshold=50.0, attenuation_req=REQ)
    assert att == pytest.approx(2e3) and lower and verdict == "INCONCLUSIVE"


@pytest.mark.parametrize("direction", ["+x", "-x", "+y", "-y"])
def test_scatter_mean_is_peak_over_attenuation(direction):
    scatter = make_out_of_field_scatter((64, 80), PEAK, 1e4, direction=direction)
    assert np.mean(scatter) == pytest.approx(PEAK / 1e4)
