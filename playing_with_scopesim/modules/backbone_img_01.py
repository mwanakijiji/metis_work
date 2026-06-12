import logging
import os
import pickle
import numpy as np
import ipdb

from .helpers import fit_psf_gaussian_from_native_array
from .psf_grid_prep import load_grid_data_from_fits


def _resolve_path(path: str, stem: str = "") -> str:
    if stem and not path.startswith("/"):
        return stem + path
    return path


def fov_calc(
    run_state: dict,
    stem: str = "",
    oversample_factor: int = 3,
    edge_size_oversamp: int = 30,
):
    '''
    For one filter configuration, centroid the PSF in each dithered FITS file
    using guess_x / guess_y from the merged analysis run state.

    INPUTS
    ----------
    run_state : dict
        Merged entry from config_file_IMG_01_METIS_AIT_img_cal_fov_ANALYSIS.yaml
        (defaults + one ``runs`` item). Must include ``filter_name`` and a
        ``dithers`` list with ``guess_x``, ``guess_y``, and ``file_name`` per dither.
    stem : str, optional
        Project root prepended to relative paths in ``file_name`` and
        ``results_write_dir``.
    oversample_factor : int, optional
        Oversampling factor passed to ``fit_psf_gaussian_from_native_array``.
    edge_size_oversamp : int, optional
        Cutout size in oversampled pixels for the Gaussian fit.

    OUTPUTS
    -------
    dict
        ``centroid_results_all`` keyed by dither index. Also writes a pickle
        under ``results_write_dir``.
    '''
    filter_name = run_state["filter_name"]
    fp_mask = run_state.get("fp_mask", "unknown_fp")
    pp_mask = run_state.get("pp_mask", "unknown_pp")
    results_write_dir = _resolve_path(
        run_state.get("results_write_dir", "figs_dump"), stem
    )

    dithers = run_state.get("dithers", [])
    if not dithers:
        raise ValueError(f"No dithers listed for filter {filter_name}")

    os.makedirs(results_write_dir, exist_ok=True)
    logging.info(
        f"FOV analysis: filter={filter_name}, fp_mask={fp_mask}, "
        f"pp_mask={pp_mask}, {len(dithers)} dither file(s)"
    )
    logging.info(f"PSF oversampling factor: {oversample_factor}")

    centroid_results_all = {}

    for i, dither in enumerate(dithers):
        file_name = _resolve_path(dither["file_name"], stem)
        guess_x = float(dither["guess_x"])
        guess_y = float(dither["guess_y"])

        logging.info(
            f"Dither {i}: file={file_name}, guess_x={guess_x}, guess_y={guess_y}"
        )

        grid_data, _grid_header = load_grid_data_from_fits(file_name, hdu_index=1)

        gaussian_fit_outputs = fit_psf_gaussian_from_native_array(
            original_array=grid_data,
            oversample_factor=oversample_factor,
            coords_xy_1st_pass_normsamp=[guess_x, guess_y],
            edge_size_oversamp=edge_size_oversamp,
        )

        x_cen = gaussian_fit_outputs["x_center_pix_fullarray_normsamp"]
        y_cen = gaussian_fit_outputs["y_center_pix_fullarray_normsamp"]
        logging.info(f"  centroid (native): x={x_cen:.3f}, y={y_cen:.3f}")

        centroid_results_all[f"dither_{i:02d}"] = {
            "file_name": file_name,
            "guess_x": guess_x,
            "guess_y": guess_y,
            "x_center_pix_native": x_cen,
            "y_center_pix_native": y_cen,
            "fwhm_x_pix_native": gaussian_fit_outputs["fwhm_x_pix_fullarray_normsamp"],
            "fwhm_y_pix_native": gaussian_fit_outputs["fwhm_y_pix_fullarray_normsamp"],
            "amplitude_counts": gaussian_fit_outputs["amplitude_counts_cookie_oversamp"],
            "angle_theta_deg": gaussian_fit_outputs["angle_theta_deg"],
        }

    # measure distances in pixel space between the corners of the grid (i.e., PSFs in frame 0 and 1; 1 and 2; 2 and 3; 3 and 0)
    distances_corner_psfs_pixel = []

    for i in range(len(centroid_results_all)):
        key_0 = 'dither_{:02d}'.format(i)
        key_1 = 'dither_{:02d}'.format((i+1)%4)
        distances_corner_psfs_pixel.append(np.sqrt((centroid_results_all[key_0]["x_center_pix_native"] - centroid_results_all[key_1]["x_center_pix_native"])**2 + (centroid_results_all[key_0]["y_center_pix_native"] - centroid_results_all[key_1]["y_center_pix_native"])**2))
    distances_corner_psfs_pixel = np.array(distances_corner_psfs_pixel)


    # given a known offset in arcseconds between the PSFs, what is the effective pixel scale?
    distances_corner_psfs_arcsec = run_state.get("distances_corner_psfs_arcsec", 0)
    effective_pixel_scale = distances_corner_psfs_arcsec / distances_corner_psfs_pixel

    illuminated_pixels_x = run_state.get("illuminated_pixels_x", 0)
    illuminated_pixels_y = run_state.get("illuminated_pixels_y", 0)
    effective_fov_x_arcsec = np.mean(effective_pixel_scale * illuminated_pixels_x)
    effective_fov_y_arcsec = np.mean(effective_pixel_scale * illuminated_pixels_y)
    effective_fov_x_sigma_arcsec = np.std(effective_pixel_scale * illuminated_pixels_x)
    effective_fov_y_sigma_arcsec = np.std(effective_pixel_scale * illuminated_pixels_y)
    


    # convert distances in pixel space to arcseconds
    #distances_corner_psfs_arcsec = distances_corner_psfs_pixel * pixel_scale

    distances_corner_psfs = run_state.get("distances_corner_psfs", 0)

    pass_fail_list = [False for _ in centroid_results_all]
    pass_fail_all = all(pass_fail_list)

    detector = run_state.get("detector", "unknown")

    if detector == "LM":
        pass_fail_list = [True if effective_fov_x_arcsec > 10.0 and effective_fov_x_arcsec < 11.0 and effective_fov_y_arcsec > 10.0 and effective_fov_y_arcsec < 11.0 else False for _ in centroid_results_all]
    elif detector == "N":
        pass_fail_list = [True if effective_fov_x_arcsec > 12.48 and effective_fov_x_arcsec < 13.97 and effective_fov_y_arcsec > 12.48 and effective_fov_y_arcsec < 13.97 else False for _ in centroid_results_all]
    else:
        raise ValueError(f"Detector {detector} not supported")

    logging.info("--------------------------------")
    logging.info("--------------------------------")
    logging.info(
        "Reqs (Ref. Overleaf doc IMG_OPT_01_Test_Description_Field_of_View_and_Ghost_Measurement):\n"
        "1) METIS-1095: The FoV shall be 10.0 +1.0/-0.0 arcsec for the LM-arm and "
        "13.47 +0.50/-0.99 arcsec for the N-arm of the IMG."
    )
    logging.info("--------------------------------")
    logging.info("--------------------------------")
    logging.info(f"EFFECTIVE FOV, DETECTOR {detector}: x: {effective_fov_x_arcsec:.2f} +/- {effective_fov_x_sigma_arcsec:.2f}; arcsec; y: {effective_fov_y_arcsec:.2f} +/- {effective_fov_y_sigma_arcsec:.2f} arcsec")
    logging.info("--------------------------------")
    logging.info("--------------------------------")
    logging.info(f"PASS/FAIL: {all(pass_fail_list)}")
    logging.info("--------------------------------")
    logging.info("--------------------------------")

    basename_file_name_pickle = f"fov_centroid_results_{fp_mask}_{pp_mask}_{filter_name}.pkl"
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
    logging.info(f"Saved FOV analysis results to {abs_file_name_pickle}")

    return centroid_results_all
