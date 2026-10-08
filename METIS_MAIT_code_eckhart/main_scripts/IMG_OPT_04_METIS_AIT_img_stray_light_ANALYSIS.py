# Does some simple analysis of simulated images written out by the sim
# notebook.

import os
import datetime
import logging
import ipdb

from modules.helpers import load_config_and_pipe, setup_logging
from modules.psf_grid_prep import load_fits_data
import modules.backbone_img_04_stray_light as b04

# Analyze data for the IMG-OPT-04 stray light test

# Reqs.:
# - Ref. Overleaf doc
#   IMG_OPT_04_Test_Description_In_Field_Straylight_and_Ghosts
#
# 1. METIS-1189: The maximum allowed stray light irradiance from an
# in-field source shall be less than 0.1 % of the peak irradiance in the
# focal planes of the IMG. Hereby, stray light contains scattering from
# opto-mechanical surfaces in Mid-infrared ELT Imager and Spectrograph
# (METIS).

# 2. METIS-1429: The maximum allowed stray light irradiance in the
# CFO-FP2 plane from an in-field source positioned in the METIS input
# focal plane shall be less than 0.06 % of the peak irradiance. The
# maximum allowed stray light irradiance in the IMG-LM and IMG-N
# detector planes from an in-field source positioned in the CFO-FP2
# plane shall be less than 0.04% of the peak irradiance.

# 3. METIS-9522: After data reduction and calibration, the flux in
# optical artefacts and ghosts shall be less than the 3-sigma thermal
# background noise for one hour of observations and for the respective
# spatial scale of the ghost, i.e. point-source-like ghosts shall
# contain less flux than the point-source sensitivity limit; extended
# ghosts shall contain less flux than the surface brightness limit for
# that extension. This shall hold when the brightness of the celestial
# source causing the artefact(s) corresponds to the saturation limit in
# the fastest full-frame operation.


def main():

    stem = "/podman-share/metis_work/METIS_MAIT_code_eckhart/"
    # config file with the observing parameters
    observing_config_file = (
        stem + "config/config_file_IMG_04_stray_light_observing.yaml"
    )  # needed? TBD
    # config file with the data states (i.e., how to analyze each PSF),
    # incl. file names
    data_states_config_file = (
        stem + "config/config_file_IMG_04_stray_light_data_states.yaml"
    )  # needed? TBD
    # config file with the coordinates guesses for the PSFs
    coords_guesses_config_file = (
        stem + "config/config_file_IMG_04_stray_light_coords_guesses.yaml"
    )

    now = datetime.datetime.now()

    # initialize logging
    log_dir = stem + "IMG_04_stray_light_analysis_logs/"
    log_file_name = (
        log_dir
        + "log_IMG_04_stray_light_analysis_"
        + now.strftime("%Y-%m-%d_%H-%M-%S")
        + ".txt"
    )
    setup_logging(log_dir=log_dir, log_file_name=log_file_name, now=now)

    # config file with generic observing parameters
    observing_config = load_config_and_pipe(
        config_file_choice=observing_config_file, print_one_line=False
    )

    # config file with data states incl. data file names
    data_states_config = load_config_and_pipe(
        config_file_choice=data_states_config_file, print_one_line=False
    )

    # loop over each data state (corresponding to one FITS image)
    for data_state in data_states_config["data_states"]:  # [0:1]: # if just for a small test

        # Resolve relative paths against the project stem
        file_name = data_state["file_name"]
        if not os.path.isabs(file_name):
            file_name = os.path.join(stem, file_name)
        data_state["file_absname"] = file_name
        logging.info(f"Processing file: {file_name}")

        # HDU 1 = BCKGD_SUBTED from the SIM writer
        data_readout, _ = load_fits_data(
            file_name, 
            hdu_index=1
        )

        print('CHECK ALL THE CENTRAL WAVELENGTHS OF THE N-BAND FILTERS ARE RIGHT!')
        ipdb.set_trace()
        

        # init the result object 
        result = b04.StrayLightResult(
            file_absname=file_name,
            filter_name=data_state["filter_name"],
            detector=data_state["detector"],
            image=data_readout,
        )

        # populate object with data state info
        result = b04.populate_result_obj_info(
            result, data_state, observing_config=observing_config
        )

        # center on the real PSF
        result = b04.centroid_2passes_oversample(
            result_obj=result,
            config_coords_guesses_file_name=coords_guesses_config_file,
        )

        # make the mask for the real PSF
        result = b04.stray_light_mask_real(
            result_obj=result, observing_config=observing_config
        )

        # segment the remaining light; method/params can be set per data state
        # (see b04.SEGMENTATION_METHODS and b04.SEGMENTATION_DEFAULT_PARAMS)
        result = b04.stray_light_segmentation(
            result,
            method=data_state.get("segmentation_method", "active_contour"),
            params=data_state.get("segmentation_params"),
        )

        # sort out the segments into different spatial scales, keeping track of illumination
        result = b04.stray_light_brightness_spectrum(
            result_obj=result, 
            observing_config=observing_config)


if __name__ == "__main__":
    main()
