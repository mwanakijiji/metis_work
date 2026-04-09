'''
Prepare oversampled PSF grid data and first-pass centroids for strehl_psfs.

The core entry point is prepare_psf_grid(): pass in-memory arrays and a coords
config dict so unit tests do not need FITS or YAML on disk.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Union

import copy
import numpy as np
from photutils.centroids import centroid_2dg, centroid_sources
from scipy.ndimage import zoom
from astropy.io import fits


@dataclass(frozen=True)
class PsfGridPrep:
    '''
    Everything strehl_psfs needs before the per-PSF loop (except observing config).
    '''

    grid_data: np.ndarray  # original data (native pixel scale)
    grid_data_original: np.ndarray  # original data (native pixel scale)
    grid_data_oversamp: np.ndarray  # oversampled data
    oversample_factor: int  # factor by which the data is oversampled
    raw_cutout_size_oversampled: int
    x_pos_pix_oversamp: np.ndarray  # x centroids, oversampled pixel coords (same frame as grid_data_oversamp)
    y_pos_pix_oversamp: np.ndarray  # y centroids, oversampled pixel coords
    coords_centroided_1st_pass_all_oversamp: np.ndarray  # shape (N, 2), columns [y, x] oversampled
    x_pos_pix_native: np.ndarray  # x centroids in original/native pixel coords (÷ oversample_factor)
    y_pos_pix_native: np.ndarray  # y centroids in original/native pixel coords
    coords_centroided_1st_pass_all_native: np.ndarray  # shape (N, 2), columns [y, x] native
    total_psfs: int  # total number of PSFs in the grid
    num_psfs_to_process: int  # number of PSFs to process
    canvas_grid_data: np.ndarray  # canvas data (original data)
    grid_header: Any | None = None  # header of the grid data


def resolve_psfs_subset(psfs_subset: Union[str, int], total_psfs: int) -> int:
    '''
    Resolve the user's PSF-subset request into the number of PSFs that should be
    processed from the available grid.

    INPUTS
    ----------
    psfs_subset : str or int
        Either ``"all"`` to process the full grid or an integer requesting only the
        first N PSFs.
    total_psfs : int
        Total number of PSFs available in the centroided grid.

    OUTPUTS
    -------
    int
        Number of PSFs to process, capped at ``total_psfs`` when an integer request
        exceeds the number available.
    '''
    if psfs_subset == "all":
        return total_psfs
    if isinstance(psfs_subset, int):
        return min(psfs_subset, total_psfs)
    raise ValueError(f"psfs_subset must be 'all' or an integer, got {psfs_subset!r}")


def _guesses_yx_from_config(coords_config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    '''
    Extract arrays of y and x coordinate guesses from the coordinate-configuration
    dictionary used to seed first-pass centroiding.

    INPUTS
    ----------
    coords_config : Mapping[str, Any]
        Configuration mapping expected to contain a ``psf_coordinate_guesses`` entry
        with a list of dictionaries holding ``y`` and ``x`` values.

    OUTPUTS
    -------
    tuple[np.ndarray, np.ndarray]
        Two float arrays containing the y guesses and x guesses, respectively. If no
        coordinate guesses are present, returns two empty float arrays.
    '''
    entries = coords_config.get("psf_coordinate_guesses", [])
    if not entries:
        return np.array([], dtype=float), np.array([], dtype=float)
    coords = np.array([(entry["y"], entry["x"]) for entry in entries], dtype=float)
    return coords[:, 0], coords[:, 1]


def prepare_psf_grid(
    grid_data: np.ndarray,
    coords_config: Mapping[str, Any],
    psfs_subset: Union[str, int] = "all",
    oversample_factor: int = 3,
    *,
    centroid_box_size: int = 41,
    zoom_order: int = 3,
    centroid_func: Callable = centroid_2dg,
    centroid_sources_impl: Callable = centroid_sources,
    grid_header: Any | None = None,
) -> PsfGridPrep:
    '''
    Oversample a PSF grid, run first-pass centroiding on the oversampled image, and
    package the derived arrays and bookkeeping values needed before the per-PSF loop.

    INPUTS
    ----------
    grid_data : np.ndarray
        Two-dimensional science image containing the PSF grid at native pixel sampling.
    coords_config : Mapping[str, Any]
        Configuration mapping expected to contain ``psf_coordinate_guesses`` with a list
        of initial ``{"y", "x"}`` coordinate guesses.
    psfs_subset : str or int, optional
        Either ``"all"`` to process every PSF or an integer requesting only the first N
        PSFs after centroiding.
    oversample_factor : int, optional
        Factor by which to oversample the grid before the first-pass centroid step.
    centroid_box_size : int, optional
        Side length, in oversampled pixels, of the box passed to the centroid routine.
    zoom_order : int, optional
        Interpolation order used when oversampling the grid with ``scipy.ndimage.zoom``.
    centroid_func : Callable, optional
        Centroid function passed through to ``centroid_sources``.
    centroid_sources_impl : Callable, optional
        Injectable centroid driver used mainly to simplify unit testing.
    grid_header : Any or None, optional
        FITS header or other metadata object associated with ``grid_data``.

    OUTPUTS
    -------
    PsfGridPrep
        Dataclass containing the original and oversampled grids, first-pass centroid
        coordinates in both oversampled and native pixel units, and related bookkeeping
        values needed by downstream Strehl-analysis code.
    '''
    grid_data = np.asarray(grid_data)
    if grid_data.ndim != 2:
        raise ValueError(f"grid_data must be 2D, got shape {grid_data.shape}")

    grid_data_original = copy.deepcopy(grid_data)
    coords_guesses_y_all, coords_guesses_x_all = _guesses_yx_from_config(coords_config)

    grid_data_oversamp = zoom(grid_data, oversample_factor, order=zoom_order)
    coords_guesses_x_all_oversamp = coords_guesses_x_all * oversample_factor
    coords_guesses_y_all_oversamp = coords_guesses_y_all * oversample_factor

    x_pos_pix_oversamp, y_pos_pix_oversamp = centroid_sources_impl(
        grid_data_oversamp,
        xpos=coords_guesses_x_all_oversamp,
        ypos=coords_guesses_y_all_oversamp,
        box_size=centroid_box_size,
        centroid_func=centroid_func,
    )

    coords_centroided_1st_pass_all_oversamp = np.vstack(
        (y_pos_pix_oversamp, x_pos_pix_oversamp)
    ).T

    fac = float(oversample_factor)
    x_pos_pix_native = x_pos_pix_oversamp / fac
    y_pos_pix_native = y_pos_pix_oversamp / fac
    coords_centroided_1st_pass_all_native = np.vstack(
        (y_pos_pix_native, x_pos_pix_native)
    ).T

    total_psfs = len(y_pos_pix_oversamp)
    num_psfs_to_process = resolve_psfs_subset(psfs_subset, total_psfs)
    raw_cutout_size_oversampled = 20 * oversample_factor
    canvas_grid_data = np.copy(grid_data)

    return PsfGridPrep(
        grid_data=grid_data,
        grid_data_original=grid_data_original,
        grid_data_oversamp=grid_data_oversamp,
        oversample_factor=oversample_factor,
        raw_cutout_size_oversampled=raw_cutout_size_oversampled,
        x_pos_pix_oversamp=x_pos_pix_oversamp,
        y_pos_pix_oversamp=y_pos_pix_oversamp,
        coords_centroided_1st_pass_all_oversamp=coords_centroided_1st_pass_all_oversamp,
        x_pos_pix_native=x_pos_pix_native,
        y_pos_pix_native=y_pos_pix_native,
        coords_centroided_1st_pass_all_native=coords_centroided_1st_pass_all_native,
        total_psfs=total_psfs,
        num_psfs_to_process=num_psfs_to_process,
        canvas_grid_data=canvas_grid_data,
        grid_header=grid_header,
    )


def load_grid_data_from_fits(file_name: str, hdu_index: int = 1) -> tuple[np.ndarray, Any]:
    '''
    Load image data and header metadata from a selected FITS HDU.

    INPUTS
    ----------
    file_name : str
        Path to the FITS file containing the PSF-grid data.
    hdu_index : int, optional
        Index of the HDU to read from the FITS file.

    OUTPUTS
    -------
    tuple[np.ndarray, Any]
        Tuple containing a copy of the HDU data as a NumPy array and a copy of the HDU
        header.
    '''

    with fits.open(file_name) as hdul:
        hdu = hdul[hdu_index]
        data = np.array(np.asarray(hdu.data), copy=True)
        header = hdu.header.copy()
    return data, header
