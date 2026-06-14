#!/usr/bin/env python
# coding: utf-8
'''
MWE: run fit_annular_aperture_free_parameters on one PSF from one FITS file.

Mirrors the call in backbone_img_03.process_one_psf (lines ~191-204):
  1. Load run settings from strehl_runs yaml (same as IMG_OPT_03 analysis)
  2. Load coordinate guesses from the coords yaml named in that config
  3. Centroid one PSF via prepare_psf_grid + native cookie cutout
  4. Gaussian-fit the cutout for a refined center (oversampled)
  5. Fit the free annular-aperture model and write diagnostic plots

Run from playing_with_scopesim/:
  python mwe_fit_annular_aperture_free.py
'''

import datetime
import logging
import os

from modules.helpers import fit_psf_gaussian_from_native_array, load_config_and_pipe, setup_logging
from modules.psf_grid_prep import load_grid_data_from_fits, prepare_psf_grid
from modules.strehl_fcns import fit_annular_aperture_free_parameters


def _stem_path(stem, path):
    if path.startswith('/'):
        return path
    return stem + path


def main():
    stem = '/podman-share/metis_work/playing_with_scopesim/'

    # Same config chain as IMG_OPT_03_METIS_AIT_img_cal_psf_ANALYSIS.py
    observing_config_file = stem + 'config/config_file_IMG_03_observing.yaml'
    data_states_config_file = stem + 'config/config_file_IMG_03_strehl_runs.yaml'

    psf_index = 0  # index into the centroided PSF list
    run_index = 0  # which entry in strehl_runs "runs" to use
    edge_size_original = 21  # matches backbone_img_03.strehl_psfs
    oversample_factor = 3
    results_write_dir = stem + 'mwe_fit_annular_aperture_free_results/'

    os.makedirs(results_write_dir, exist_ok=True)
    now = datetime.datetime.now()
    log_file = (
        results_write_dir
        + 'log_mwe_fit_annular_aperture_free_'
        + now.strftime('%Y-%m-%d_%H-%M-%S')
        + '.txt'
    )
    setup_logging(log_dir=results_write_dir, log_file_name=log_file, now=now)

    config_observing = load_config_and_pipe(
        config_file_choice=observing_config_file, print_one_line=False
    )
    data_states_config = load_config_and_pipe(
        config_file_choice=data_states_config_file, print_one_line=False
    )
    defaults = data_states_config.get('defaults', {})
    runs = data_states_config.get('runs', [])
    if not runs:
        raise ValueError(f'No runs found in {data_states_config_file}')

    run = {**defaults, **runs[run_index]}
    fits_file = _stem_path(stem, run['file_name'])
    coords_config_file = _stem_path(stem, run['config_coords_guesses_file_name'])
    filter_name = run['filter_name']
    fp_mask = run['fp_mask']
    pp_mask = run['pp_mask']
    fit_method = run.get('fit_method', 'curve_fit')

    logging.info('MWE: fit_annular_aperture_free_parameters')
    logging.info(f'FITS file: {fits_file}')
    logging.info(f'Coords config: {coords_config_file}')
    logging.info(f'PSF index: {psf_index}')

    config_coords = load_config_and_pipe(
        config_file_choice=coords_config_file, print_one_line=False
    )

    grid_data, grid_header = load_grid_data_from_fits(fits_file, hdu_index=1)
    prep = prepare_psf_grid(
        grid_data,
        config_coords,
        psfs_subset=psf_index + 1,
        oversample_factor=oversample_factor,
        grid_header=grid_header,
    )

    if psf_index >= prep.total_psfs:
        raise IndexError(
            f'psf_index={psf_index} out of range; only {prep.total_psfs} PSF(s) centroided'
        )

    x_cen_native = prep.coords_centroided_1st_pass_all_native[psf_index][1]
    y_cen_native = prep.coords_centroided_1st_pass_all_native[psf_index][0]
    half = int(0.5 * edge_size_original)

    # Same cookie slicing as backbone_img_03.strehl_psfs
    cookie_cutout_original = prep.grid_data_original[
        int(x_cen_native - half):int(x_cen_native + half),
        int(y_cen_native - half):int(y_cen_native + half),
    ]
    logging.info(
        f'Cookie cutout shape: {cookie_cutout_original.shape}, '
        f'native center (x, y)=({x_cen_native:.2f}, {y_cen_native:.2f})'
    )

    gaussian_fit_outputs = fit_psf_gaussian_from_native_array(
        original_array=cookie_cutout_original,
        oversample_factor=oversample_factor,
        coords_xy_1st_pass_normsamp=None,
        edge_size_oversamp=None,
    )
    cookie_cutout_oversamp = gaussian_fit_outputs['cookie_cut_out_sci_oversamp']
    x_center_oversamp = gaussian_fit_outputs['x_center_pix_fullarray_oversamp']
    y_center_oversamp = gaussian_fit_outputs['y_center_pix_fullarray_oversamp']
    logging.info(
        f'Gaussian centroid (oversampled x, y)=({x_center_oversamp:.2f}, {y_center_oversamp:.2f})'
    )

    plot_string = (
        f'mwe_psf_{psf_index}_fpmask_{fp_mask}_ppmask_{pp_mask}_filter_{filter_name}'
    )
    strehl_results = fit_annular_aperture_free_parameters(
        cookie_cut_out_sci_oversamp=cookie_cutout_oversamp,
        cookie_cut_out_sci_original=cookie_cutout_original,
        filter_name=filter_name,
        plot_string=plot_string,
        x_center_final_cookie_oversamp=x_center_oversamp,
        y_center_final_cookie_oversamp=y_center_oversamp,
        config_observing=config_observing,
        fac_oversamp=oversample_factor,
        fit_method=fit_method,
        results_write_dir=results_write_dir,
    )

    logging.info('Done.')
    logging.info(f'Strehl / debug results: {strehl_results}')
    logging.info(f'Plots and log written under: {results_write_dir}')


if __name__ == '__main__':
    main()
