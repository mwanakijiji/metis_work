"""Unit tests for modules.psf_grid_prep (no FITS on disk required)."""

import numpy as np
import pytest

from modules.psf_grid_prep import PsfGridPrep, prepare_psf_grid, resolve_psfs_subset


def _fake_centroid_sources(_img, xpos, ypos, box_size, centroid_func):
    """Return guesses unchanged (deterministic; avoids photutils in unit tests)."""
    return np.asarray(xpos, dtype=float), np.asarray(ypos, dtype=float)


def test_resolve_psfs_subset_all():
    assert resolve_psfs_subset("all", 5) == 5


def test_resolve_psfs_subset_int_capped():
    assert resolve_psfs_subset(2, 5) == 2
    assert resolve_psfs_subset(10, 5) == 5


def test_resolve_psfs_subset_invalid():
    with pytest.raises(ValueError, match="psfs_subset"):
        resolve_psfs_subset("first", 3)


def test_prepare_psf_grid_requires_2d():
    with pytest.raises(ValueError, match="2D"):
        prepare_psf_grid(np.zeros(10), {"psf_coordinate_guesses": []}, centroid_sources_impl=_fake_centroid_sources)


def test_prepare_psf_grid_invalid_subset():
    grid = np.zeros((20, 20))
    cfg = {"psf_coordinate_guesses": [{"y": 10.0, "x": 10.0}]}
    with pytest.raises(ValueError, match="psfs_subset"):
        prepare_psf_grid(grid, cfg, psfs_subset="first", centroid_sources_impl=_fake_centroid_sources)


def test_prepare_psf_grid_oversample_and_centroids():
    grid = np.zeros((10, 12))
    cfg = {"psf_coordinate_guesses": [{"y": 4.0, "x": 5.0}, {"y": 8.0, "x": 9.0}]}
    prep = prepare_psf_grid(
        grid,
        cfg,
        psfs_subset="all",
        oversample_factor=3,
        centroid_sources_impl=_fake_centroid_sources,
    )
    assert isinstance(prep, PsfGridPrep)
    assert prep.grid_data_oversamp.shape == (30, 36)
    assert prep.total_psfs == 2
    assert prep.num_psfs_to_process == 2
    assert np.allclose(prep.x_pos_pix_oversamp, [15.0, 27.0])
    assert np.allclose(prep.y_pos_pix_oversamp, [12.0, 24.0])
    assert prep.coords_centroided_1st_pass_all_oversamp.shape == (2, 2)
    assert prep.raw_cutout_size_oversampled == 20 * 3
    assert prep.canvas_grid_data.shape == grid.shape
    assert not np.shares_memory(prep.grid_data_original, prep.grid_data)


def test_prepare_psf_grid_subset_limits_loop_count_field():
    grid = np.zeros((20, 20))
    cfg = {
        "psf_coordinate_guesses": [
            {"y": 10.0, "x": 10.0},
            {"y": 12.0, "x": 12.0},
        ]
    }
    prep = prepare_psf_grid(
        grid,
        cfg,
        psfs_subset=1,
        oversample_factor=2,
        centroid_sources_impl=_fake_centroid_sources,
    )
    assert prep.total_psfs == 2
    assert prep.num_psfs_to_process == 1
