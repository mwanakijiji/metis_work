# Does some simple analysis of simulated images written out by the sim notebook.

import os
import datetime
import logging
import yaml
import ipdb

from modules.helpers import load_config_and_pipe
from modules.backbone_img_02 import image_distortion


# Make simulated data for the IMG-OPT-02 image distorition test

# Reqs.:
# - Ref. Overleaf doc IMG_OPT_02_Test_Geometric_Distortion
# 
# 1. METIS-1097: The Imager shall provide a pixel scale of 5.47 +0.26/-0.26 mas/pix for the LM-band and 
#               6.79+0.25/-0.50 mas/pix for the N-band.
# METIS-3502: After calibration, the distortions introduced by METIS shall be removed to better than
#               0.5 mas (ca. 1/10 px for the L band imager) over the full field of view.
# METIS-8222: The center of the H2RG chip within IMG-LM-DET shall be offset from the METIS optical axis 
#               by 175 mas ± 25 mas on-sky PtV in the "across H2RG stripe" direction (i.e., perpendicularly 
#               to the orientation of the 32 stripes in the H2RG detector).
# METIS-9920: Image scale and distortion of METIS shall be constant (for each optical configuration, even after 
#               change of observing modes) to an accuracy of 10-3 (goal: 10-4) at L/M-band and 2×10-3 (goal: 2×10-4) 
#               at N-band with respect to the full field of view. 

def main():

    stem = '/podman-share/metis_work/playing_with_scopesim/'
    # config file with the observing parameters
    observing_config_file = stem + 'config/config_file_IMG_02_observing.yaml' # needed? TBD
    # config file with the data states (i.e., how to analyze each PSF)
    data_states_config_file = stem + 'config/config_file_IMG_02_strehl_runs.yaml' # needed? TBD

    now = datetime.datetime.now()

    # initialize logging
    log_dir = stem + 'IMG_02_logs/'
    log_file_name = log_dir + 'log_IMG_02_analysis_psf_image_quality_' + now.strftime('%Y-%m-%d_%H-%M-%S') + '.txt'    
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
    logging.info(f'Log file directory: {stem + "IMG_02_logs/"}')

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
    for state in data_states: # [0:1]: # if just for a small test
        image_distortion(
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