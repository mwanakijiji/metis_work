# Does some simple analysis of simulated images written out by the sim notebook.

import os
import datetime
import logging
import yaml
import ipdb

from modules.helpers import load_config_and_pipe
from modules.backbone import strehl_psfs

# Reqs.:
# - Ref. Overleaf doc IMG_OPT_04_Test_Description_PSF_Image_Quality
# 
# 1. METIS-1408: Quality and alignment of the optical components within Mid-infrared ELT Imager and
# Spectrograph (METIS) shall provide diffraction limited performance (Strehl ≥ 80 %)
# at λ > 3μm in all modes over the entire FOV.
# 2. METIS-1409: The Instrument Wavefront Error (WFE) shall satisfy the diffraction limit requirement
# (Strehl>0.8) at lambda = 3 μm for IMG (both LM and NQ) and IMG. The minimum
# RMS WFE below shall be satisfied over the full Field Of View (FOV) relevant to the
# given optical path.
# 3. METIS-2864: The minimum Strehl ratio of the WCU+CFO+IMG-LM optical path shall be >80% at
# 3.3μm over the entire field of view.
# 4. METIS-3503: METIS shall be able to characterise the shape of the instrument PSF across the entire
# FoV using the WCU.

def main():

    stem = '/podman-share/metis_work/playing_with_scopesim/'
    # config file with the observing parameters
    observing_config_file = stem + 'config/config_file_IMG_04_observing.yaml'
    # config file with the data states (i.e., how to analyze each PSF)
    data_states_config_file = stem + 'config/config_file_IMG_04_strehl_runs.yaml'

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

    # read in main config file
    observing_config = load_config_and_pipe(config_file_choice=observing_config_file, print_one_line=False)

    # load data states (defaults + per-run overrides)
    data_states_config = load_config_and_pipe(config_file_choice=data_states_config_file, print_one_line=False)
    defaults = data_states_config.get("defaults", {})
    runs = data_states_config.get("runs", [])

    # to set up the data states, merge config data state defaults with overrides that are specific for each run
    data_states = []
    for entry in runs:
        merged = {**defaults, **entry}
        # Prepend stem to relative file paths
        fname = merged["file_name"]
        if not fname.startswith("/"):
            merged["file_name"] = stem + fname
        results_write_dir = merged["results_write_dir"]
        if not results_write_dir.startswith("/"):
            merged["results_write_dir"] = stem + results_write_dir
        data_states.append(merged)

    # loop over each data state
    #ipdb.set_trace()
    for state in data_states: # data_states[0:1]: # if just for a small test
        strehl_psfs(
            state["file_name"],
            fp_mask=state["fp_mask"],
            pp_mask=state["pp_mask"],
            filter_name=state["filter_name"],
            fit_simmed_psf=state["fit_simmed_psf"],
            fit_annular_aperture_free=state["fit_annular_aperture_free"],
            fit_annular_aperture_fixed=state["fit_annular_aperture_fixed"],
            psfs_subset=state["psfs_subset"],
            config_coords_guesses_file_name=state["config_coords_guesses_file_name"],
            config_observing=observing_config,
            results_write_dir=state["results_write_dir"],
            fit_method="curve_fit"
        )



if __name__ == "__main__":
    main()