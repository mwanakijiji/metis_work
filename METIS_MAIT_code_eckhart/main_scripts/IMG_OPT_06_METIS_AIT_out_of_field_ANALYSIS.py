"""
Analyze the simulated (or real) data for the IMG-OPT-06 out-of-field stray
light test.
"""

import datetime
import logging
import os

from astropy.io import fits

import modules.backbone_img_06_out_of_field as oof
import modules.stray_light_common as slc
from modules.helpers import load_config_and_pipe, setup_logging
from modules.psf_grid_prep import load_fits_data

# Analyze data for the IMG-OPT-06 out-of-field stray light test

# Reqs. (Table 1-1 of the IMG-OPT-06 test case description):
#
# 1. METIS-3114 (verified by test at System level): METIS shall be
# designed and built to be insensitive to outside illumination so that
# all calibration procedures needed by the instrument can be performed
# without being affected by routine observatory operations (including
# also the operation of other instruments).
# -> NOT tested here; done at observatory level.
#
# 2. METIS-5838 (verified by analysis at System level, applicable to
# IMG): Irradiance from sources outside the input beam envelope (defined
# by the 'FOV' and the max ELT pupil area), propagating between the
# specified 'input plane' and 'output plane' through scattering shall be
# attenuated by a factor indicated below for the indicated 'wavelength
# ranges'.
#   Subsystem | Input plane | Output plane    | FOV            | Band | Attenuation
#   IMG       | CFO FP2     | IMG-N detector  | 14"x14" square | N    | 1e4
#   IMG       | CFO FP2     | IMG-LM detector | 11"x11" square | L, M | 1e4
#
# 3. METIS-9522 (verified by test at System level, applicable to IMG):
# After data reduction and calibration, the flux in optical artefacts
# and ghosts shall be less than the 3-sigma thermal background noise for
# one hour of observations and for the respective spatial scale of the
# ghost, i.e. point-source-like ghosts shall contain less flux than the
# point-source sensitivity limit; extended ghosts shall contain less flux
# than the surface brightness limit for that extension. This shall hold
# when the brightness of the celestial source causing the artefact(s)
# corresponds to the saturation limit in the fastest full-frame
# operation.
# -> tested the same way as in IMG-OPT-04
#
# Pass/fail of each requirement is decided in one place:
# backbone_img_06_out_of_field.log_test_outcome


def main():
    """Run the IMG-OPT-06 out-of-field stray light analysis for each data state."""

    stem = "/podman-share/metis_work/METIS_MAIT_code_eckhart/"
    # generic observing parameters (filters, pixel scales, sensitivities, ...),
    # shared with IMG-OPT-04
    observing_config_file_img_04 = (
        stem + "config/config_file_IMG_04_stray_light_observing.yaml"
    )
    # IMG-OPT-06 specifics (METIS-5838 parameters)
    observing_config_file_img_06 = (
        stem + "config/config_file_IMG_06_out_of_field_observing.yaml"
    )
    # config file with the data states, incl. file names
    data_states_config_file = (
        stem + "config/config_file_IMG_06_out_of_field_data_states.yaml"
    )
    # config file with the coordinates guesses for the PSF
    coords_guesses_config_file = (
        stem + "config/config_file_IMG_06_out_of_field_coords_guesses.yaml"
    )

    now = datetime.datetime.now()

    # initialize logging
    log_dir = stem + "IMG_06_out_of_field_analysis_logs/"
    log_file_name = (
        log_dir
        + "log_IMG_06_out_of_field_analysis_"
        + now.strftime("%Y-%m-%d_%H-%M-%S")
        + ".txt"
    )
    setup_logging(log_dir=log_dir, log_file_name=log_file_name, now=now)

    # IMG-OPT-06 values override the IMG-OPT-04 ones where both are set
    observing_config = {
        **load_config_and_pipe(
            config_file_choice=observing_config_file_img_04, print_one_line=False
        ),
        **load_config_and_pipe(
            config_file_choice=observing_config_file_img_06, print_one_line=False
        ),
    }

    data_states_config = load_config_and_pipe(
        config_file_choice=data_states_config_file, print_one_line=False
    )

    # loop over each data state (corresponding to one FITS image)
    for data_state in data_states_config["data_states"]:

        # Resolve relative paths against the project stem
        file_name = data_state["file_name"]
        if not os.path.isabs(file_name):
            file_name = os.path.join(stem, file_name)
        data_state["file_absname"] = file_name
        logging.info("==================================================")
        logging.info("Processing file: %s", file_name)

        # HDU 2 = BCKGD_SUBTED from the SIM writer (1: RAW_READOUT, 3: BACKGROUND,
        # 4: BACKGROUND_2)
        data_readout, header_readout = load_fits_data(file_name, hdu_index=2)
        if header_readout.get("EXTNAME") != "BCKGD_SUBTED":
            logging.warning(
                "HDU 2 has EXTNAME %s, not BCKGD_SUBTED",
                header_readout.get("EXTNAME"),
            )

        with fits.open(file_name, memmap=False) as hdul:
            # injected truth from the SIM primary header, if there (absent for real
            # data)
            header_primary = hdul[0].header.copy()
            # null frame from two independent backgrounds, if there
            ext_names = [hdu.name for hdu in hdul]
            if "BACKGROUND" in ext_names and "BACKGROUND_2" in ext_names:
                null_image = hdul["BACKGROUND_2"].data.astype(float) - hdul[
                    "BACKGROUND"
                ].data.astype(float)
            else:
                null_image = None
                logging.warning(
                    "No BACKGROUND and BACKGROUND_2 extensions; no null frame for the "
                    "METIS-5838 noise"
                )

        # init the result object
        result = oof.OutOfFieldResult(
            file_absname=file_name,
            filter_name=data_state["filter_name"],
            detector=data_state["detector"],
            image=data_readout,
            results_dir=oof.RESULTS_DIR_IMG_06,
            expected=header_primary.get("EXPECTED"),
            attenuation_injected=header_primary.get("ATT_INJ"),
            ref_peak_header=header_primary.get("REF_PEAK"),
            null_image=null_image,
        )

        # shared stray-light stages (stray_light_common)

        # populate object with data state info
        result = slc.populate_result_obj_info(
            result, data_state, observing_config=observing_config
        )

        # center on the real PSF
        result = slc.centroid_2passes_oversample(
            result_obj=result,
            config_coords_guesses_file_name=coords_guesses_config_file,
        )

        # make the mask for the real PSF
        result = slc.stray_light_mask_real(
            result_obj=result, observing_config=observing_config
        )

        # IMG-OPT-06 stages

        # METIS-5838: attenuation of the out-of-field irradiance
        result = oof.metis_5838_attenuation(
            result_obj=result, observing_config=observing_config
        )

        # segment the remaining light (for METIS-9522), as in IMG-OPT-04 (shared stage)
        segmentation_method = data_state.get("segmentation_method", "combined")
        if segmentation_method == "combined":
            result = slc.stray_light_segmentation_combined(
                result,
                weights=data_state.get("segmentation_weights"),
                vote_fraction=data_state.get("segmentation_vote_fraction"),
                params=data_state.get("segmentation_params"),
            )
        else:
            result = slc.stray_light_segmentation(
                result,
                method=segmentation_method,
                params=data_state.get("segmentation_params"),
            )

        # METIS-9522: artefacts and ghosts vs. 3-sigma of the background, for their
        # spatial scale
        result = oof.metis_9522_check(
            result_obj=result, observing_config=observing_config
        )

        # pass/fail of each requirement
        oof.log_test_outcome(result)


if __name__ == "__main__":
    main()
