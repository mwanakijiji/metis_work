"""Analysis backbone for the IMG-OPT-01 field-of-view test."""

import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import LogNorm
from pipeline_registry import (
    CLUSTER_FOV,
    COLOR_FOV,
    pipeline_stage,
)
from skimage.measure import block_reduce

from .helpers import fit_psf_gaussian_from_native_array
from .psf_grid_prep import load_grid_data_from_fits, oversample_1st_pass_centroid
from .tbd_warnings import add_tbd_banner

# FYI plots: binning of the co-added frame, and side of the pinhole cutouts (pixels)
FYI_BIN = 8
FYI_CUTOUT = 41

# 1st-pass centroiding: half side (native pixels) of the box around the guess that is
# oversampled (not the whole frame: a 3x-oversampled 2048x2048 frame is ~300 MB)
FIRST_PASS_HALF_BOX = 50

# FYI plot markers: guess, 1st-pass centroid, final (2-pass) centroid
MARKER_GUESS = {"marker": "o", "color": "cyan", "fillstyle": "none", "linestyle": ""}
MARKER_1ST_PASS = {"marker": "+", "color": "orange", "linestyle": ""}
MARKER_CENTROID = {"marker": "x", "color": "red", "linestyle": "", "mew": 2}


def _resolve_path(path: str, stem: str = "") -> str:
    if stem and not path.startswith("/"):
        return stem + path
    return path


def _cutout(image, center_yx, size):
    """size x size cutout centred on center_yx (rounded), zero-padded at the edges."""
    half = size // 2
    padded = np.pad(np.asarray(image, dtype=float), half, mode="constant")
    y, x = int(round(center_yx[0])) + half, int(round(center_yx[1])) + half
    return padded[y - half : y + half + 1, x - half : x + half + 1]


def first_pass_centroid(
    image, guess_x, guess_y, oversample_factor=3, half_box=FIRST_PASS_HALF_BOX
):
    """
    1st-pass centroid (x, y) in native pixels of the full frame, as in the other tests
    (psf_grid_prep.oversample_1st_pass_centroid: photutils centroid_2dg on the
    oversampled image, seeded by the guess), on a box of +/- half_box pixels around the
    guess. Falls back to the guess if the centroiding fails.
    """
    ny, nx = image.shape
    y0 = max(0, int(round(guess_y)) - half_box)
    x0 = max(0, int(round(guess_x)) - half_box)
    box = np.asarray(image[y0 : y0 + 2 * half_box + 1, x0 : x0 + 2 * half_box + 1])
    if box.shape[0] < 3 or box.shape[1] < 3 or y0 >= ny or x0 >= nx:
        logging.warning("1st pass: guess (%s, %s) is off the frame", guess_x, guess_y)
        return float(guess_x), float(guess_y)
    prep = oversample_1st_pass_centroid(
        np.nan_to_num(box.astype(float)),
        {"psf_coordinate_guesses": [{"x": guess_x - x0, "y": guess_y - y0}]},
        psfs_subset="all",
        oversample_factor=oversample_factor,
    )
    y_box, x_box = prep.coords_centroided_1st_pass_all_native[0]
    if not (np.isfinite(x_box) and np.isfinite(y_box)):
        logging.warning(
            "1st pass: centroiding failed near (%s, %s); using the guess",
            guess_x,
            guess_y,
        )
        return float(guess_x), float(guess_y)
    return float(x_box + x0), float(y_box + y0)


def fyi_plot_dithers(  # pylint: disable=too-many-arguments,too-many-locals
    results_dir,
    plot_stem,
    centroid_results_all,
    coadded_binned,
    cutouts,
    frame_shape,
):
    """
    FYI plot of one run: the four dithers co-added (binned), with the guesses, the
    1st-pass centroids, the final (2-pass) centroids and the square they make, and a
    cutout around each pinhole with the same markers.

    OUTPUTS:
    - str: path of the written plot
    """
    ny, nx = frame_shape
    keys = sorted(centroid_results_all)

    def xy(key_x, key_y):
        return np.array(
            [
                [centroid_results_all[k][key_x], centroid_results_all[k][key_y]]
                for k in keys
            ]
        )

    cen = xy("x_center_pix_native", "y_center_pix_native")
    first = xy("x_1st_pass_pix_native", "y_1st_pass_pix_native")
    guess = xy("guess_x", "guess_y")

    fig = plt.figure(figsize=(16, 8.5))
    grid = fig.add_gridspec(2, 4, width_ratios=[2, 2, 1, 1])
    ax_main = fig.add_subplot(grid[:, :2])
    positive = np.clip(coadded_binned, 0.0, None)
    vmax = float(np.max(positive)) if np.max(positive) > 0 else 1.0
    ax_main.imshow(
        positive + vmax * 1e-5,
        origin="lower",
        cmap="magma",
        norm=LogNorm(vmin=vmax * 1e-4, vmax=vmax),
        extent=(-0.5, nx - 0.5, -0.5, ny - 0.5),
    )
    closed = np.vstack([cen, cen[:1]])
    ax_main.plot(closed[:, 0], closed[:, 1], "w-", linewidth=0.8, label="Dither square")
    ax_main.plot(guess[:, 0], guess[:, 1], markersize=12, label="Guess", **MARKER_GUESS)
    ax_main.plot(
        first[:, 0],
        first[:, 1],
        markersize=12,
        label="1st-pass centroid",
        **MARKER_1ST_PASS,
    )
    ax_main.plot(
        cen[:, 0],
        cen[:, 1],
        markersize=12,
        label="Centroid (2 passes)",
        **MARKER_CENTROID,
    )
    for i, (x, y) in enumerate(cen):
        ax_main.annotate(
            f"{i}: ({x:.1f}, {y:.1f})",
            (x, y),
            textcoords="offset points",
            xytext=(8, 8),
            color="white",
            fontsize=8,
        )
    ax_main.set_xlabel("x (pixels)")
    ax_main.set_ylabel("y (pixels)")
    ax_main.legend(loc="lower right", fontsize=8)
    ax_main.set_title(
        f"Dithers co-added (binned {FYI_BIN}x{FYI_BIN}, log stretch)", fontsize=10
    )

    half = FYI_CUTOUT // 2
    for i, image in enumerate(cutouts):
        ax = fig.add_subplot(grid[i // 2, 2 + i % 2])
        ax.imshow(image, origin="lower", cmap="magma")
        # cutout centred on the rounded final centroid
        x0, y0 = round(cen[i][0]) - half, round(cen[i][1]) - half
        ax.plot(guess[i][0] - x0, guess[i][1] - y0, markersize=9, **MARKER_GUESS)
        ax.plot(first[i][0] - x0, first[i][1] - y0, markersize=9, **MARKER_1ST_PASS)
        ax.plot(cen[i][0] - x0, cen[i][1] - y0, markersize=9, **MARKER_CENTROID)
        offset = np.hypot(*(cen[i] - guess[i]))
        ax.set_title(
            f"Dither {i}: centroid - guess = {offset:.1f} pix",
            fontsize=8,
            color="red" if offset > half else "black",
        )
        ax.set_xlim(-0.5, FYI_CUTOUT - 0.5)
        ax.set_ylim(-0.5, FYI_CUTOUT - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"IMG-OPT-01 ANALYSIS {plot_stem}: pinhole centroids "
        "(o guess, + 1st pass, x final after 2 passes)",
        y=0.95,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    add_tbd_banner(fig)
    file_name = os.path.join(results_dir, f"fyi_fov_dithers_{plot_stem}.png")
    fig.savefig(file_name)
    plt.close(fig)
    logging.info("Saved FYI plot %s", file_name)
    return file_name


def fyi_plot_fov(  # pylint: disable=too-many-arguments
    results_dir,
    plot_stem,
    fov_per_side_x,
    fov_per_side_y,
    fov_limits,
    verdict,
    fov_injected=None,
):
    """
    FYI plot of one run: the FoV from each side of the dither square (pixel scale from
    that side times the illuminated pixels, x and y), their mean, the requirement range
    and, for simulated data, the injected FoV.

    OUTPUTS:
    - str: path of the written plot
    """
    fov_min, fov_max = fov_limits
    sides = np.arange(len(fov_per_side_x))
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.6))
    ax.axhspan(fov_min, fov_max, color="green", alpha=0.15, label="METIS-1095 range")
    ax.plot(sides - 0.05, fov_per_side_x, "o", label="From side, x illuminated pixels")
    ax.plot(sides + 0.05, fov_per_side_y, "s", label="From side, y illuminated pixels")
    ax.axhline(
        np.mean(fov_per_side_x),
        color="C0",
        linestyle="--",
        label=f"Mean x: {np.mean(fov_per_side_x):.3f}\"",
    )
    ax.axhline(
        np.mean(fov_per_side_y),
        color="C1",
        linestyle=":",
        label=f"Mean y: {np.mean(fov_per_side_y):.3f}\"",
    )
    if fov_injected is not None:
        ax.axhline(
            fov_injected,
            color="red",
            linestyle="-.",
            label=f"Injected (SIM): {fov_injected:.3f}\"",
        )
    ax.set_xticks(sides)
    ax.set_xticklabels([f"{i}-{(i + 1) % len(sides)}" for i in sides])
    ax.set_xlabel("Side of the dither square (dithers)")
    ax.set_ylabel("FoV (arcsec)")
    ax.legend(fontsize=8)
    ax.set_title(f"IMG-OPT-01 ANALYSIS {plot_stem}\nMETIS-1095: {verdict}", fontsize=10)
    fig.subplots_adjust(top=0.85)
    add_tbd_banner(fig)
    file_name = os.path.join(results_dir, f"fyi_fov_per_side_{plot_stem}.png")
    fig.savefig(file_name)
    plt.close(fig)
    logging.info("Saved FYI plot %s", file_name)
    return file_name


@pipeline_stage(
    name="fov_calc",
    depends_on=("generate_fov_sim",),
    cluster=CLUSTER_FOV,
    cluster_color=COLOR_FOV,
    label="Compute FOV",
)
def fov_calc(  # pylint: disable=too-many-locals,too-many-statements
    run_state: dict,
    stem: str = "",
    oversample_factor: int = 3,
    edge_size_oversamp: int = 30,
):
    """
    For one filter configuration, centroid the PSF in each dithered FITS file in two
    passes, as in the other tests: a 1st-pass centroid seeded by guess_x / guess_y from
    the merged analysis run state (first_pass_centroid), then a Gaussian fit seeded by
    the 1st-pass centroid. Also writes FYI plots (fyi_fov_dithers_*.png,
    fyi_fov_per_side_*.png) to results_write_dir.

    INPUTS
    ----------
    run_state : dict
        Merged entry from config_file_IMG_01_METIS_AIT_img_cal_fov_ANALYSIS.yaml
        (defaults + one ``runs`` item). Must include ``filter_name`` and a
        ``dithers`` list with ``guess_x``, ``guess_y``, and ``file_name`` per dither;
        an optional ``run_label`` distinguishes runs with the same filter and masks.
    stem : str, optional
        Project root prepended to relative paths in ``file_name`` and
        ``results_write_dir``.
    oversample_factor : int, optional
        Oversampling factor for both passes.
    edge_size_oversamp : int, optional
        Cutout size in oversampled pixels for the Gaussian fit (2nd pass).

    OUTPUTS
    -------
    dict
        ``centroid_results_all`` keyed by dither index. Also writes a pickle
        under ``results_write_dir``.
    """
    filter_name = run_state["filter_name"]
    fp_mask = run_state.get("fp_mask", "unknown_fp")
    pp_mask = run_state.get("pp_mask", "unknown_pp")
    run_label = run_state.get("run_label")
    results_write_dir = _resolve_path(
        run_state.get("results_write_dir", "figs_dump"), stem
    )
    label_part = f"_{run_label}" if run_label else ""
    plot_stem = f"{fp_mask}_{pp_mask}_{filter_name}{label_part}"

    dithers = run_state.get("dithers", [])
    if not dithers:
        raise ValueError(f"No dithers listed for filter {filter_name}")

    os.makedirs(results_write_dir, exist_ok=True)
    logging.info(
        "FOV analysis: filter=%s, fp_mask=%s, pp_mask=%s, run=%s, %s dither file(s)",
        filter_name,
        fp_mask,
        pp_mask,
        run_label,
        len(dithers),
    )
    logging.info("PSF oversampling factor: %s", oversample_factor)

    # injected truth from the SIM primary header, if there (absent for real data)
    header_primary = fits.getheader(_resolve_path(dithers[0]["file_name"], stem), 0)
    if "REQ_ID" in header_primary:
        logging.info(
            "Simulated data: %s case %s, injected %s = %s -> expected %s",
            header_primary["REQ_ID"],
            header_primary.get("CASE"),
            header_primary.get("REQ_QTY"),
            header_primary.get("INJ_VAL"),
            header_primary.get("EXPECTED"),
        )

    centroid_results_all = {}
    # for the FYI plots (binned frames and cutouts only, not the full frames)
    coadded_binned, cutouts, frame_shape = None, [], None

    for i, dither in enumerate(dithers):
        file_name = _resolve_path(dither["file_name"], stem)
        guess_x = float(dither["guess_x"])
        guess_y = float(dither["guess_y"])

        logging.info(
            "Dither %s: file=%s, guess_x=%s, guess_y=%s", i, file_name, guess_x, guess_y
        )

        # the background-subtracted frame (BCKGD_SUBTED, or HDU 1 in older files)
        grid_data, _grid_header = load_grid_data_from_fits(file_name, hdu_index=1)

        # 1st pass: centroid on the oversampled image around the guess
        x_1st, y_1st = first_pass_centroid(
            grid_data, guess_x, guess_y, oversample_factor=oversample_factor
        )
        logging.info("  1st-pass centroid (native): x=%.3f, y=%.3f", x_1st, y_1st)

        # 2nd pass: Gaussian fit seeded by the 1st-pass centroid
        _, gaussian_fit_outputs = fit_psf_gaussian_from_native_array(
            original_array=grid_data,
            oversample_factor=oversample_factor,
            coords_xy_1st_pass_normsamp=[x_1st, y_1st],
            edge_size_oversamp=edge_size_oversamp,
        )

        x_cen = gaussian_fit_outputs["x_center_pix_fullarray_normsamp"]
        y_cen = gaussian_fit_outputs["y_center_pix_fullarray_normsamp"]
        logging.info("  2nd-pass centroid (native): x=%.3f, y=%.3f", x_cen, y_cen)

        frame_shape = grid_data.shape
        cutouts.append(_cutout(grid_data, (y_cen, x_cen), FYI_CUTOUT))
        binned = block_reduce(
            np.nan_to_num(grid_data.astype(float)), (FYI_BIN, FYI_BIN), func=np.mean
        )
        coadded_binned = binned if coadded_binned is None else coadded_binned + binned

        centroid_results_all[f"dither_{i:02d}"] = {
            "file_name": file_name,
            "guess_x": guess_x,
            "guess_y": guess_y,
            "x_1st_pass_pix_native": x_1st,
            "y_1st_pass_pix_native": y_1st,
            "x_center_pix_native": x_cen,
            "y_center_pix_native": y_cen,
            "fwhm_x_pix_native": gaussian_fit_outputs["fwhm_x_pix_fullarray_normsamp"],
            "fwhm_y_pix_native": gaussian_fit_outputs["fwhm_y_pix_fullarray_normsamp"],
            "amplitude_counts": gaussian_fit_outputs[
                "amplitude_counts_cookie_oversamp"
            ],
            "angle_theta_deg": gaussian_fit_outputs["angle_theta_deg"],
        }

    # measure distances in pixel space between the corners of the grid (i.e., PSFs in
    # frame 0 and 1; 1 and 2; 2 and 3; 3 and 0)
    distances_corner_psfs_pixel = []

    for i in range(len(centroid_results_all)):
        key_0 = f"dither_{i:02d}"
        key_1 = f"dither_{(i + 1) % 4:02d}"
        distances_corner_psfs_pixel.append(
            np.sqrt(
                (
                    centroid_results_all[key_0]["x_center_pix_native"]
                    - centroid_results_all[key_1]["x_center_pix_native"]
                )
                ** 2
                + (
                    centroid_results_all[key_0]["y_center_pix_native"]
                    - centroid_results_all[key_1]["y_center_pix_native"]
                )
                ** 2
            )
        )
    distances_corner_psfs_pixel = np.array(distances_corner_psfs_pixel)

    # given a known offset in arcseconds between the PSFs, what is the effective pixel
    # scale?
    distances_corner_psfs_arcsec = run_state.get("distances_corner_psfs_arcsec", 0)
    effective_pixel_scale = distances_corner_psfs_arcsec / distances_corner_psfs_pixel

    illuminated_pixels_x = run_state.get("illuminated_pixels_x", 0)
    illuminated_pixels_y = run_state.get("illuminated_pixels_y", 0)
    fov_per_side_x = effective_pixel_scale * illuminated_pixels_x
    fov_per_side_y = effective_pixel_scale * illuminated_pixels_y
    effective_fov_x_arcsec = np.mean(fov_per_side_x)
    effective_fov_y_arcsec = np.mean(fov_per_side_y)
    effective_fov_x_sigma_arcsec = np.std(fov_per_side_x)
    effective_fov_y_sigma_arcsec = np.std(fov_per_side_y)

    # convert distances in pixel space to arcseconds
    # distances_corner_psfs_arcsec = distances_corner_psfs_pixel * pixel_scale

    detector = run_state.get("detector", "unknown")

    if detector == "LM":
        fov_min, fov_max = 10.0, 11.0
    elif detector == "N":
        fov_min, fov_max = 12.48, 13.97
    else:
        raise ValueError(f"Detector {detector} not supported")
    pass_fail_all = bool(
        fov_min < effective_fov_x_arcsec < fov_max
        and fov_min < effective_fov_y_arcsec < fov_max
    )
    verdict = "PASS" if pass_fail_all else "FAIL"

    logging.info("--------------------------------")
    logging.info("--------------------------------")
    logging.info(
        "Reqs (Ref. Overleaf doc "
        "IMG_OPT_01_Test_Description_Field_of_View_and_Ghost_Measurement):\n"
        "1) METIS-1095: The FoV shall be 10.0 +1.0/-0.0 arcsec for the LM-arm and "
        "13.47 +0.50/-0.99 arcsec for the N-arm of the IMG."
    )
    logging.info("--------------------------------")
    logging.info("--------------------------------")
    logging.info(
        "EFFECTIVE FOV, DETECTOR %s: x: %.2f +/- %.2f; arcsec; y: %.2f +/- %.2f arcsec",
        detector,
        effective_fov_x_arcsec,
        effective_fov_x_sigma_arcsec,
        effective_fov_y_arcsec,
        effective_fov_y_sigma_arcsec,
    )
    logging.info("--------------------------------")
    logging.info("--------------------------------")
    logging.info("METIS-1095 (%.2f < FoV < %.2f arcsec): %s", fov_min, fov_max, verdict)
    if "EXPECTED" in header_primary:
        expected = header_primary["EXPECTED"]
        logging.info(
            "Simulated data: expected %s; measured %s -> %s",
            expected,
            verdict,
            "consistent" if expected == verdict else "INCONSISTENT",
        )
    logging.info("--------------------------------")
    logging.info("--------------------------------")

    # FYI plots
    fyi_plot_dithers(
        results_write_dir,
        plot_stem,
        centroid_results_all,
        coadded_binned,
        cutouts,
        frame_shape,
    )
    fov_injected = header_primary.get("INJ_VAL")
    fyi_plot_fov(
        results_write_dir,
        plot_stem,
        fov_per_side_x,
        fov_per_side_y,
        (fov_min, fov_max),
        verdict,
        fov_injected=float(fov_injected) if fov_injected is not None else None,
    )

    basename_file_name_pickle = f"fov_centroid_results_{plot_stem}.pkl"
    abs_file_name_pickle = os.path.join(results_write_dir, basename_file_name_pickle)
    with open(abs_file_name_pickle, "wb") as f:
        pickle.dump(
            {
                "run_state": run_state,
                "centroid_results_all": centroid_results_all,
                "pass_fail_all": pass_fail_all,
            },
            f,
        )
    logging.info("Saved FOV analysis results to %s", abs_file_name_pickle)

    return centroid_results_all
