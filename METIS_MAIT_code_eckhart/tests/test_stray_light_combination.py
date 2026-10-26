"""Unit tests for the weighted combination of stray light segmentation methods (synthetic masks only)."""

import sys
from pathlib import Path

import numpy as np
import pytest

# backbone_img_04_stray_light imports pipeline_registry from misc/ and lives in main_scripts/modules
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "main_scripts", _ROOT / "misc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from modules.backbone_img_04_stray_light import (  # noqa: E402
    SEGMENTATION_METHODS,
    _combine_method_regions,
    _relabel_min_area,
    _resolve_segmentation_weights,
    _resolve_vote_fraction,
    iter_regions,
)

SHAPE = (60, 60)


def _box(y0, y1, x0, x1, label=1):
    m = np.zeros(SHAPE, dtype=np.int32)
    m[y0:y1, x0:x1] = label
    return m


@pytest.fixture
def psf_mask():
    m = np.zeros(SHAPE, dtype=bool)
    m[0:5, 0:5] = True
    return m


@pytest.fixture
def combo(psf_mask):
    # blob A: seen by a (large) and b (offset, overlapping); blob B: seen by c only;
    # a also covers part of the PSF
    method_region_maps = {
        "a": _box(0, 20, 0, 20),
        "b": _box(10, 25, 10, 25),
        "c": _box(40, 50, 40, 50),
    }
    weights = {"a": 1.0, "b": 1.0, "c": 0.2}
    return _combine_method_regions(method_region_maps, weights, 0.5, psf_mask)


def test_overlapping_regions_share_label(combo):
    a, b = combo["method_label_maps"]["a"], combo["method_label_maps"]["b"]
    labels_a = set(np.unique(a[a > 0]))
    labels_b = set(np.unique(b[b > 0]))
    assert labels_a == labels_b and len(labels_a) == 1


def test_region_missing_from_method_that_did_not_detect_it(combo):
    c = combo["method_label_maps"]["c"]
    (label_c,) = set(np.unique(c[c > 0]))
    for method in ("a", "b"):
        assert label_c not in combo["method_label_maps"][method]


def test_weighted_vote(combo):
    # sigma = 2.2, threshold = 1.1: only pixels seen by both a and b pass
    assert combo["sigma"] == pytest.approx(2.2)
    assert combo["threshold"] == pytest.approx(1.1)
    final = combo["final_label_map"] > 0
    expected = np.zeros(SHAPE, dtype=bool)
    expected[10:20, 10:20] = True
    assert np.array_equal(final, expected)
    # final region keeps the shared group label; the c-only blob is rejected
    a = combo["method_label_maps"]["a"]
    assert combo["final_labels"] == list(np.unique(a[a > 0]))
    assert len(combo["final_labels"]) == 1


def test_psf_never_in_regions(combo, psf_mask):
    assert not np.any(combo["final_label_map"][psf_mask])
    for label_map in combo["method_label_maps"].values():
        assert not np.any(label_map[psf_mask])


def test_label_maps_are_int32(combo):
    # one compact label map per method (not one full-frame mask per region)
    assert combo["final_label_map"].dtype == np.int32
    for label_map in combo["method_label_maps"].values():
        assert label_map.dtype == np.int32


def test_relabel_min_area(psf_mask):
    labels = _box(0, 3, 10, 13, label=7) + _box(30, 40, 30, 40, label=3)
    labels += _box(0, 5, 0, 5, label=9)  # entirely inside the PSF
    out = _relabel_min_area(labels, psf_mask, min_area=10)
    # the 9-pixel region and the PSF-only region are dropped; the rest is renumbered 1..n
    assert set(np.unique(out)) == {0, 1}
    assert np.array_equal(out > 0, _box(30, 40, 30, 40) > 0)


def test_iter_regions():
    labels = _box(2, 4, 2, 6, label=1) + _box(20, 30, 40, 45, label=3)
    regions = {lab: (slices, inside) for lab, slices, inside in iter_regions(labels)}
    assert set(regions) == {1, 3}  # absent label 2 is skipped
    slices, inside = regions[3]
    assert slices == (slice(20, 30), slice(40, 45))
    assert inside.all() and inside.shape == (10, 5)


def test_resolve_weights_defaults_and_overrides():
    w = _resolve_segmentation_weights(None)
    assert w == {m: 1.0 for m in SEGMENTATION_METHODS}
    w = _resolve_segmentation_weights({"kmeans": 0.0, "watershed": 0.3})
    assert w["kmeans"] == 0.0 and w["watershed"] == 0.3 and w["threshold"] == 1.0


@pytest.mark.parametrize(
    "weights",
    [
        {m: 0.0 for m in SEGMENTATION_METHODS},  # sigma == 0
        {"threshold": 1.5},  # out of range
        {"threshold": -0.1},
        {"not_a_method": 1.0},
    ],
)
def test_resolve_weights_raises(weights):
    with pytest.raises(ValueError):
        _resolve_segmentation_weights(weights)


def test_resolve_vote_fraction():
    assert _resolve_vote_fraction(None) == 0.5
    with pytest.raises(ValueError):
        _resolve_vote_fraction(1.2)
