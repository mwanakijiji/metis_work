# Does some simple analysis of simulated images written out by the sim notebook.

import os
import datetime
import logging
import yaml

from modules.helpers import load_config_and_pipe
from modules.backbone import strehl_psfs


def main():

    stem = '/podman-share/metis_work/playing_with_scopesim/'
    # config file containing the observing parameters
    observing_config_file = stem + 'config/config_file_IMG_04_observing.yaml'
    # config file containing guesses for the PSF coordinates
    #config_coords_guesses_file_name = stem + 'config/config_file_IMG_04_coords.yaml' # for ScopeSim 10 coords
    config_coords_guesses_file_name = stem + 'config/config_file_IMG_04_coords_ver11.yaml' # coords for LM grid in ScopeSim 11

    now = datetime.datetime.now()

    # initialize logging
    log_dir = stem + 'IMG_04_logs/'
    log_file_name = log_dir + 'log_IMG_04_analysis_psf_image_quality_' + now.strftime('%Y-%m-%d_%H-%M-%S') + '.txt'    
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_name),
            logging.StreamHandler()
        ],
        force=True
    )
    logging.info(f'Log file created at {now.strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info(f'Log file name: {log_file_name}')
    logging.info(f'Log file directory: {stem + "IMG_04_logs/"}')

    # read in config files
    observing_config = load_config_and_pipe(config_file_choice=observing_config_file, print_one_line=False)
    config_coords_guesses = load_config_and_pipe(config_file_choice=config_coords_guesses_file_name, print_one_line=False)

    # pp mask choices
    # 'APP-LMS', 'APP-LM', 'CLS-LMS', 'CLS-LM', 'CLS-N', 'PPS-LMS', 'PPS-LM', 'PPS-N', 'PPS-CFO2', 'RLS-LMS', 'RLS-LM', 'SPM-LMS', 'SPM-LM', 'SPM-N', 'open'

    # Strehl analysis configurations
    data_states = [
        {"filter_name": "Br_alpha",     "fit_simmed_psf": True,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_PPS-CFO2_filter_Br_alpha_noiseless.fits"},
        {"filter_name": "Br_alpha",     "fit_simmed_psf": False,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_PPS-CFO2_filter_Br_alpha.fits"},
        {"filter_name": "Br_alpha_ref", "fit_simmed_psf": False,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_Open_filter_Br_alpha_ref.fits"},
        {"filter_name": "Lp",           "fit_simmed_psf": False,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_Open_filter_Lp.fits"},
        {"filter_name": "H2O-ice",      "fit_simmed_psf": False,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_Open_filter_H2O-ice.fits"},
        {"filter_name": "PAH_3.3",      "fit_simmed_psf": False,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_Open_filter_PAH_3.3.fits"},
        {"filter_name": "PAH_3.3_ref",  "fit_simmed_psf": False,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_Open_filter_PAH_3.3_ref.fits"},
        {"filter_name": "short-L",      "fit_simmed_psf": False,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_Open_filter_short-L.fits"},
        {"filter_name": "IB_4.05",      "fit_simmed_psf": False,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_Open_filter_IB_4.05.fits"},
        {"filter_name": "HCI_L_short",  "fit_simmed_psf": False,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_Open_filter_HCI_L_short.fits"},
        {"filter_name": "HCI_L_long",   "fit_simmed_psf": False,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_Open_filter_HCI_L_long.fits"},
        {"filter_name": "Mp",           "fit_simmed_psf": False,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_Open_filter_Mp.fits"},
        {"filter_name": "CO_1-0_ice",   "fit_simmed_psf": False,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_Open_filter_CO_1-0_ice.fits"},
        {"filter_name": "CO_ref",       "fit_simmed_psf": False,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_Open_filter_CO_ref.fits"},
        {"filter_name": "HCI_M",        "fit_simmed_psf": False,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_Open_filter_HCI_M.fits"},
        {"filter_name": "L_spec",       "fit_simmed_psf": False,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_Open_filter_L_spec.fits"},
        {"filter_name": "M_spec",       "fit_simmed_psf": False,  "fit_annular_aperture_free": True, "fit_annular_aperture_fixed": True, "file_name": stem + "IMG_04_sample_input_data/strehl/IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_grid_lm_pupil_mask_Open_filter_M_spec.fits"},

    ]

    fp_mask = 'grid_lm'
    pp_mask = 'open'
    clocking_angle = 0

    for _state in data_states[0:1]:

        strehl_psfs(_state["file_name"],
                    fp_mask=fp_mask,
                    pp_mask=pp_mask,
                    filter_name=_state["filter_name"],
                    fit_simmed_psf=_state["fit_simmed_psf"],
                    fit_annular_aperture_free=_state["fit_annular_aperture_free"],
                    fit_annular_aperture_fixed=_state["fit_annular_aperture_fixed"],
                    psfs_subset=1,
                    config_coords_guesses_file_name=config_coords_guesses_file_name,
                    config_observing=observing_config)



if __name__ == "__main__":
    main()