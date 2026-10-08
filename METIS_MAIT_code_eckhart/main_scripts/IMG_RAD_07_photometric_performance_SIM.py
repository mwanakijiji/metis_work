# Make simulated data for the IMG-RAD-07 photometric performance test

# Reqs.:
# - Ref. IMG-RAD-07 Photometric performance, E-XXX-NOVA-MET-NNNN 0-1,
#   Table 1-1
#
# 1. METIS-2594 (verified by test at System level): METIS internal
# calibration shall provide a range of flux levels matching that
# encountered in scientific operation as defined in METIS-3137 -
# E-TNT-MPIA-MET-1004.

# 2. METIS-2595 (verified by test at Observatory level, applicable to
# IMG): Relative photometric accuracy shall be better than 0.97% rms.

# 3. METIS-2595 (verified by test at Observatory level, applicable to
# IMG): Uncertainty in color correction should be less than 1% rms.

# 4. METIS-2883 (verified by test at subsystem level): The source used
# for detector linearity calibration METIS-3289 shall provide an
# illumination that is spatially homogeneous within +/- 3%
# peak-to-valley over the science field of view of METIS.
# ------> Procedure: see Calibration Plan E-PLA-NOVA-MET-1066 4-0 24-08-2022, Sec. 2.3

# 5. METIS-3291 (verified by test at System level, applicable to WCU and
# PIP): METIS shall allow the characterisation and correction of the
# relative spectral response function (RSRF) with 3% precision over the
# wavelength range specified in METIS-1090.
# ------> Procedure: see Calibration Plan E-PLA-NOVA-MET-1066 4-0 24-08-2022, p. 23-4

# 6. METIS-3300 (verified by test at System level, applicable to WCU):
# The level of flux of the flat field shall be tunable to 15 discrete
# values and shall be known at each step to 0.3%. The resulting spatial
# illumination pattern shall remain constant across the field within
# 0.5% (0.3% as goal) after tuning.

# 7. METIS-3301 (verified by test at subsystem level, applicable to
# WCU): After calibration, the distortions introduced by METIS shall be
# removed to better than 0.5 mas (ca. 1/10 px for the L band imager) over
# the full field of view.

# 8. Stretch goal 1 (not an FDR req.): Aperture correction factor shall
# be known.


import datetime
import logging
import os
import sys

import numpy as np
from astropy import units as u
from astropy.io import fits
import scopesim as sim

import ipdb

import modules.backbone_img_rad_07_phot as backbone
from modules.helpers import pipe_2_log
from pipeline_registry import pipeline_stage

# no IMG-RAD-07 cluster in pipeline_registry yet; move these there when there is one
CLUSTER_PHOT = "IMG-RAD-07 Photometric performance"
COLOR_PHOT = "#9467bd"

# Edit this path if you have a custom install directory, otherwise comment it out. [For ReadTheDocs only]
sim.link_irdb("../../../../")

# simulate observations with METIS (comment this out if packages already exist)
# sim.download_packages(["METIS"])   # or git pull your IRDB clone

# print versions of things
sim.bug_report()


def sanity_check_user_inputs(metis, obs_filter, fp_mask, pp_mask):
    """Exit if configured filters/masks do not match the OpticalTrain."""
    pass


@pipeline_stage(
    name="make_flat_frames",
    cluster=CLUSTER_PHOT,
    cluster_color=COLOR_PHOT,
    label="Generate flats",
)
def make_flat_frames(
    t_integration,
    fp_mask,
    pp_mask,
    psf_kernel,
    nd_filter,
    obs_filter,
    obs_mode,
    use_exp_time_only=False,
    out_dir=None,
    intrapixel_capacitance=True,
):
    """
    Generate simulated data for the IMG-RAD-07 photometric performance test

    INPUTS:
    - fp_mask: focal plane mask
    - pp_mask: pupil plane mask
    - nd_filter: ND filter
    - obs_filter: observing filter
    - obs_mode: observing mode
    - dit: dit time
    - ndit: number of DITs
    - exptime: exposure time
    - use_exp_time_only: if True, only use the exposure time to set the exposure time; but this will be broken
        down into dit and ndit to avoid saturation, so integration parameters may change accordingly
    - out_dir: directory to write the simulated data to
    - intrapixel_capacitance: if False, turn off inter-pixel capacitance

    OUTPUTS:
    - None; writes out files
    """

    # set up instrument
    cmd = None  # reset

    if nd_filter is not None:
        cmd = sim.UserCommands(
            use_instrument="METIS",
            set_modes=[obs_mode],
            properties={
                "!OBS.filter_name": obs_filter,
                "!WCU.current_fpmask": fp_mask,
                "!OBS.pupil_mask": pp_mask,
                "!OBS.nd_filter_name": nd_filter,
                "!SIM.psf.interp_order": 1
            },
            # ignore_effects=["shot_noise", "readout_noise", "dark_current", "ipc"]
        )
    else:
        cmd = sim.UserCommands(
            use_instrument="METIS",
            set_modes=[obs_mode],
            properties={
                "!OBS.filter_name": obs_filter,
                "!WCU.current_fpmask": fp_mask,
                "!OBS.pupil_mask": pp_mask,
                "!SIM.psf.interp_order": 1
            },
            # ignore_effects=["shot_noise", "readout_noise", "dark_current", "ipc"]
        )

    metis = sim.OpticalTrain(cmd)
    # metis['ipc'].included = False # turn off inter-pixel capacitance for now

    if not intrapixel_capacitance:
        # metis['ipc'].update(alpha_edge=0.0, alpha_corner=0.0, alpha_aniso=0.0) # turn off inter-pixel capacitance for now
        metis["ipc"].include = False
        logging.info("Turning off inter-pixel capacitance for now")
    else:
        logging.info("Using default inter-pixel capacitance")

    # Set the pupil mask (transmission) and the matching PSF kernel (ScopeSim doesn't link the two)
    if psf_kernel is None:
        psf_kernel = pp_mask + "_WCU"
    metis["pupil_masks"].change_mask(pp_mask)
    metis["psf"].update(pupil_mask=psf_kernel)
    logging.info("Setting WCU PP mask to be: " + str(pp_mask))
    logging.info("Setting PSF kernel to be: " + str(psf_kernel))

    wcu = metis["wcu_source"]

    bb_temp = 1000 * u.K

    pipe_2_log(
        lambda m=metis: m.effects.pprint_all(),
        msg="Optical train effects (initial)",
    )

    #########################################################
    # BACKGROUND ONLY (WE WANT A DARK)

    # loop over the integration times; each one gives one frame
    for t_int in t_integration:

        # integration parameters for this frame: one DIT of length t_int (or exptime=t_int)
        dit = float(t_int)
        ndit = 1
        exptime = np.nan # dont modify for now
        logging.info(f"Integration time: {t_int} s")

        logging.info("Closing WCU BB to get a background ...")
        wcu.set_bb_aperture(value=0.0)

        metis.observe()

        # save a perfect background (for debugging)
        # hdul_perfect_background = metis.image_planes[0].hdu
        # hdul_perfect_background.writeto(out_dir + 'junk_perfect_background.fits', overwrite=True)

        pipe_2_log(
            lambda m=metis: m.effects.pprint_all(),
            msg="Optical train effects (background)",
        )

        if use_exp_time_only:
            # Method 1 for setting exposure times: exptime alone
            outhdul_off = metis.readout(exptime=exptime, reset=False)[0]
        else:
            # Method 2 for setting exposure times: use ndit and dit together
            outhdul_off = metis.readout(ndit=ndit, dit=dit, reset=False)[0]

        logging.info("--------------------------------")
        logging.info("Background readout:")
        logging.info("OBS filter: " + str(metis.cmds.get("!OBS.filter_name")))
        logging.info("WCU FP mask: " + str(metis.cmds.get("!WCU.current_fpmask")))
        logging.info("OBS PP mask: " + str(metis.cmds.get("!OBS.pupil_mask")))
        logging.info("OBS ND filter: " + str(metis.cmds.get("!OBS.nd_filter_name")))
        logging.info("NDIT: " + str(metis.cmds["!OBS.ndit"]))
        logging.info("DIT: " + str(metis.cmds["!OBS.dit"]))
        logging.info("WCU source state:")
        pipe_2_log(
            lambda m=metis: metis["wcu_source"].info(),
            msg="Optical train effects (background)",
        )

        # sanity check that user inputs really are the same as what the instrument is using
        def sanity_check_user_inputs(metis, obs_filter, fp_mask, pp_mask):
            if metis.cmds.get("!OBS.filter_name") != obs_filter:
                logging.error(
                    "! ------- OBS filter: "
                    + str(metis.cmds.get("!OBS.filter_name"))
                    + " does not match user input: "
                    + str(obs_filter)
                )
                exit()
            if metis.cmds.get("!WCU.current_fpmask") != fp_mask:
                logging.error(
                    "! ------- WCU FP mask: "
                    + str(metis.cmds.get("!WCU.current_fpmask"))
                    + " does not match user input: "
                    + str(fp_mask)
                )
                exit()
            if metis.cmds.get("!OBS.pupil_mask") != pp_mask:
                logging.error(
                    "! ------- OBS PP mask: "
                    + str(metis.cmds.get("!OBS.pupil_mask"))
                    + " does not match user input: "
                    + str(pp_mask)
                )
                exit()
            else:
                logging.info("User filter inputs match instrument inputs")
            return

        # check for background-taking
        # sanity_check_user_inputs(metis, obs_filter=obs_filter, fp_mask=fp_mask, pp_mask=pp_mask)

    
        if use_exp_time_only:
            # Method 1 for setting exposure times: exptime alone
            outhdul_off = metis.readout(exptime=exptime, reset=False)[0]
        else:
            # Method 2 for setting exposure times: use ndit and dit together
            outhdul_off = metis.readout(ndit=ndit, dit=dit, reset=False)[0]

        # background-subtract (there is none, actually, since we are taking a dark)
        raw_sci_readout = outhdul_off[1].data
        bckgd_subted = raw_sci_readout

        basename_file_name_write = (
            "IMG_OPT_03_wcu_focal_mask_bckgrnd_subted_"
            + str(fp_mask)
            + "_pupil_mask_"
            + str(pp_mask)
            + "_filter_"
            + str(obs_filter)
            + f"_tint_{t_int:.3f}s"
            + ".fits"
        )
        abs_file_name_write = out_dir + basename_file_name_write

        # Copy the primary header (just copy the same frame into the ImageHDUs)
        primary_hdu = fits.PrimaryHDU(header=outhdul_off[0].header)
        # Add background-subtracted readout as first extension
        hdu_bckgd_subted = fits.ImageHDU(data=bckgd_subted, name="BCKGD_SUBTED")
        # Add raw science readout as second extension
        hdu_raw_readout = fits.ImageHDU(data=bckgd_subted, name="RAW_READOUT")
        # Add background as third extension
        hdu_background = fits.ImageHDU(data=bckgd_subted, name="BACKGROUND")
        hdul_new = fits.HDUList(
            [primary_hdu, hdu_bckgd_subted, hdu_raw_readout, hdu_background]
        )

        # add some stuff to the header, some of which may be redundant
        hdul_new[0].header["FILTER"] = (obs_filter, "Observing filter")
        hdul_new[0].header["WCU_FP"] = (fp_mask, "WCU focal plane mask")
        hdul_new[0].header["WCU_PP"] = (pp_mask, "WCU pupil plane mask")
        hdul_new[0].header["PSF_KERN"] = (psf_kernel, "ScopeSim PSF kernel")
        hdul_new[0].header["BB_TEMP"] = (bb_temp.value, "BB temperature")
        if ndit is not None:
            hdul_new[0].header["NDIT"] = (ndit, "Number of dithered exposures")
            hdul_new[0].header["DIT"] = (dit, "Det integration time")
        else:
            hdul_new[0].header["EXPTIME"] = (exptime, "Exposure time")

        hdul_new.writeto(abs_file_name_write, overwrite=True)
        logging.info(
            "Saved background-subtracted readout without aberrations to "
            + abs_file_name_write
        )

        logging.info("--------------------------------")
        logging.info(f"Median of dark frame: {np.median(bckgd_subted):.4f}")

    pass


def main():
    """
    Generate IMG-RAD-07 photometric performance simulations for the configured filters.
    """

    # set up paths, logging and output directory
    stem = "/podman-share/metis_work/METIS_MAIT_code_eckhart/"

    now = datetime.datetime.now()
    log_dir = stem + "logs/IMG_07_phot_logs/"
    log_file_name = (
        log_dir
        + "log_IMG_07_photometric_performance_"
        + now.strftime("%Y-%m-%d_%H-%M-%S")
        + ".txt"
    )
    out_dir = (
        stem + "data/IMG_OPT_07_photometric_performance_simmed_data/"
    )  # directory to write the simulated data to

    # Ensure log directory exists and force config in case handlers already set
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_name), logging.StreamHandler()],
        force=True,
    )
    os.makedirs(out_dir, exist_ok=True)

    logging.info(f'Log file created at {now.strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info(f"Log file name: {log_file_name}")
    logging.info(f'Log file directory: {stem + log_dir}')
    logging.info(f"Simmed file output directory: {out_dir}")

    # observing configurations (filters, masks, ND filters, integration times)
    lm_dark_configs = [
        {
            "fp_mask": "grid_lm", "pp_mask": "SPM-LM", "psf_kernel": "SPM-LM",
            "obs_filter": "HCI_L_short", "nd_filter": None,
            "obs_mode": "wcu_img_lm", "use_exp_time_only": False
            }
    ]

    # use different integration times (need at least 15 background levels)
    t_integration = np.linspace(0.01, 0.04, num=15)
    make_flat_frames(**lm_dark_configs[0], t_integration=t_integration, out_dir=out_dir)

    # loop over configurations and generate the simulated data

    pass


if __name__ == "__main__":
    main()
