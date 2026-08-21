'''
Prepare oversampled PSF grid data and first-pass centroids for strehl_psfs.

The core entry point is oversample_1st_pass_centroid(): pass in-memory arrays and a coords
config dict so unit tests do not need FITS or YAML on disk.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Union

import copy
import os
import numpy as np
from photutils.centroids import centroid_2dg, centroid_sources
from . import helpers
from scipy.ndimage import zoom
from astropy.io import fits
import ipdb
import matplotlib.pyplot as plt


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


def oversample_1st_pass_centroid(
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


def refine_2nd_pass_centroids(
    data_original: np.ndarray, 
    prep: PsfGridPrep, 
    oversample_factor: int = 3) -> PsfGridPrep:

    '''
    Refine the centroids of the PSFs using a Gaussian fit.

    INPUTS:
    ----------
    prep : PsfGridPrep
        Dataclass containing the original and oversampled grids, first-pass centroid
        coordinates in both oversampled and native pixel units, and some other (legacy) stuff

    OUTPUTS:
    -------
    PsfGridPrep
    '''

    # loop over all the PSFs on this FITS file (might just be one)
    for num_coord in range(prep.num_psfs_to_process):

        # coordinates from first-pass centroiding
        x_cen_1st_pass_native = prep.coords_centroided_1st_pass_all_native[num_coord][1]
        y_cen_1st_pass_native = prep.coords_centroided_1st_pass_all_native[num_coord][0]
        coords_xy_1st_pass_normsamp = (x_cen_1st_pass_native, y_cen_1st_pass_native)

        # make the cookie cutout around this PSF
        edge_size_native_sampling = 20 # length of one side of a box around the PSF, native sampling (KEEP EVEN)
        grid_data_original_cutout_this_psf = data_original[
            int(x_cen_1st_pass_native - 0.5*edge_size_native_sampling):int(x_cen_1st_pass_native + 0.5*edge_size_native_sampling),
            int(y_cen_1st_pass_native - 0.5*edge_size_native_sampling):int(y_cen_1st_pass_native + 0.5*edge_size_native_sampling)
        ]

        # fit a Gaussian to the PSF
        xy_coords_2nd_pass, _ = helpers.fit_psf_gaussian_from_native_array(
            original_array=data_original,
            oversample_factor=oversample_factor,
            coords_xy_1st_pass_normsamp=coords_xy_1st_pass_normsamp,
            edge_size_oversamp=edge_size_native_sampling,
        )

    return xy_coords_2nd_pass


def _is_resource_deadlock(exc: BaseException) -> bool:
    errno = getattr(exc, "errno", None)
    return errno == 35 or "Resource deadlock avoided" in str(exc)


def _local_copy_of_fits(file_name: str) -> str:
    """
    Copy a FITS file to container-local /tmp.

    Needed when Podman virtiofs on macOS raises errno 35 on direct reads.
    Tries several copy methods because ``cp``/``shutil`` can hit the same deadlock.
    """
    import subprocess
    import tempfile

    dest = os.path.join(
        tempfile.gettempdir(),
        f"metis_fits_cache_{os.path.basename(file_name)}",
    )
    errors: list[str] = []

    # 1) shell cp
    try:
        subprocess.run(["cp", "-f", file_name, dest], check=True, capture_output=True)
        return dest
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"cp: {exc}")

    # 2) dd (sometimes succeeds when cp fails on virtiofs)
    try:
        subprocess.run(
            ["dd", f"if={file_name}", f"of={dest}", "bs=1M", "status=none"],
            check=True,
            capture_output=True,
        )
        return dest
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"dd: {exc}")

    # 3) cat redirect via shell
    try:
        subprocess.run(
            f'cat "{file_name}" > "{dest}"',
            shell=True,
            check=True,
            capture_output=True,
        )
        return dest
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"cat: {exc}")

    raise OSError(
        35,
        "Resource deadlock avoided while reading FITS on a shared mount. "
        "Tried cp/dd/cat to /tmp and all failed. "
        "On the Mac host run: xattr -c <file>; or restart Podman "
        "(`podman machine stop && podman machine start`). "
        f"Details: {errors}",
        file_name,
    )


def _open_fits_hdul(file_name: str):
    """Open a FITS file with memmap off, falling back to a /tmp copy on errno 35."""
    try:
        return fits.open(file_name, memmap=False)
    except OSError as exc:
        if not _is_resource_deadlock(exc):
            raise
        local_path = _local_copy_of_fits(file_name)
        return fits.open(local_path, memmap=False)


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
    return load_fits_data(file_name, hdu_index=hdu_index)


def load_fits_data(file_name: str, hdu_index: int = 1) -> tuple[np.ndarray, Any]:
    '''
    Load image data and header metadata from a selected FITS HDU.

    Uses ``memmap=False``. If the path is on a flaky Podman/macOS share and open
    raises errno 35 (Resource deadlock avoided), copies the file to ``/tmp`` via
    shell ``cp`` and opens that instead.

    INPUTS
    ----------
    file_name : str
        Abs path to the FITS file containing the PSF-grid data.
    hdu_index : int, optional
        Index of the HDU to read from the FITS file. Science arrays from the
        IMG-OPT-04 SIM writers live in extension 1 (``BCKGD_SUBTED``).

    OUTPUTS
    -------
    tuple[np.ndarray, Any]
        Tuple containing a copy of the HDU data as a NumPy array and a copy of the HDU
        header.
    '''
    with _open_fits_hdul(file_name) as hdul:
        hdu = hdul[hdu_index]
        data = np.array(np.asarray(hdu.data), copy=True)
        header = hdu.header.copy()
    return data, header
