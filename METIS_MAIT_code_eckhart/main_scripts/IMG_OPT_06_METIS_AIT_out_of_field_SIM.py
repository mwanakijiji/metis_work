"""
Make simulated data for the IMG-OPT-06 out-of-field stray light test

Req.:
Irradiance from sources outside the input beam envelope (defined by the
'FOV' and the max ELT pupil area), propagating between the specified
'input plane' and 'output plane' through scattering shall be attenuated
by a factor indicated below for the indicated 'wavelength ranges'.

Subsystem | Input plane | Output plane    | FOV            | Band | Attenuation
IMG       | CFO FP2     | IMG-N detector  | 14"x14" square | N    | 1e4
IMG       | CFO FP2     | IMG-LM detector | 11"x11" square | L, M | 1e4

The WCU pinhole in ScopeSim is centred, so a truly out-of-field source
cannot be simulated. Instead, as for IMG-OPT-04, a single PSF is simulated
in the middle of the detector (its peak is the unattenuated irradiance),
and synthetic scattered light from a virtual out-of-field source is added
at a level of peak / attenuation. For each filter, two files are written:
'_with_stray_light' (violates the attenuation requirement) and
'_no_stray_light' (meets it).

The exposures are deep (many NDIT at an unsaturated DIT), as in the test
description, so that an attenuation of 1e4 is measurable above the noise.
Each file also contains a second, independent background frame, so the
analysis can measure the noise from a null frame (BACKGROUND_2 - BACKGROUND).
"""

import datetime
import logging
import os
import sys

import numpy as np
import scopesim as sim
from astropy import units as u
from astropy.io import fits

import modules.backbone_img_06_out_of_field as backbone
from modules.helpers import pipe_2_log
from pipeline_registry import pipeline_stage  # pylint: disable=import-error

# no IMG-OPT-06 cluster in pipeline_registry yet; move these there when there is one
CLUSTER_OOF = "IMG-OPT-06 Out-of-field stray light"
COLOR_OOF = "#8c564b"

# Edit this path if you have a custom install directory, otherwise comment
# it out. [For ReadTheDocs only]
sim.link_irdb("../../../")

# simulate observations with METIS (comment this out if packages already exist)
#sim.download_packages(["METIS", "ELT", "Armazones"])

# print versions of things
sim.bug_report()


def sanity_check_user_inputs(metis, obs_filter, fp_mask, pp_mask):
    """Exit if configured filters/masks do not match the OpticalTrain."""
    if metis.cmds.get("!OBS.filter_name") != obs_filter:
        logging.error(
            "! ------- OBS filter: %s does not match user input: %s",
            metis.cmds.get("!OBS.filter_name"),
            obs_filter,
        )
        sys.exit(1)
    if metis.cmds.get("!WCU.current_fpmask") != fp_mask:
        logging.error(
            "! ------- WCU FP mask: %s does not match user input: %s",
            metis.cmds.get("!WCU.current_fpmask"),
            fp_mask,
        )
        sys.exit(1)
    if metis.cmds.get("!OBS.pupil_mask") != pp_mask:
        logging.error(
            "! ------- OBS PP mask: %s does not match user input: %s",
            metis.cmds.get("!OBS.pupil_mask"),
            pp_mask,
        )
        sys.exit(1)
    logging.info("User filter inputs match instrument inputs")


def log_readout_state(metis, label):
    """Log the instrument configuration for a readout."""
    logging.info("--------------------------------")
    logging.info("%s readout:", label)
    logging.info("OBS filter: %s", metis.cmds.get("!OBS.filter_name"))
    logging.info("WCU FP mask: %s", metis.cmds.get("!WCU.current_fpmask"))
    logging.info("OBS PP mask: %s", metis.cmds.get("!OBS.pupil_mask"))
    logging.info("OBS ND filter: %s", metis.cmds.get("!OBS.nd_filter_name"))
    logging.info("NDIT: %s", metis.cmds["!OBS.ndit"])
    logging.info("DIT: %s", metis.cmds["!OBS.dit"])
    logging.info("WCU source state:")
    pipe_2_log(
        lambda m=metis: m["wcu_source"].info(),
        msg=f"WCU source state ({label})",
    )


@pipeline_stage(
    name="generate_out_of_field_sim",
    cluster=CLUSTER_OOF,
    cluster_color=COLOR_OOF,
    label="Generate out-of-field sims",
)
def generate_out_of_field_data(
    fp_mask,
    pp_mask,
    nd_filter,
    obs_filter,
    obs_mode,
    fov_req,
    dit=1,
    ndit=1,
    exptime=0.01,
    use_exp_time_only=False,
    attenuation_req=1e4,
    attenuation_fail=1e3,
    attenuation_pass=1e5,
    oof_direction="+x",
    out_dir=None,
    intrapixel_capacitance=True,
):
    """
    Generate simulated data for the IMG-OPT-06 out-of-field stray light test

    INPUTS:
    - fp_mask: focal plane mask
    - pp_mask: pupil plane mask
    - nd_filter: ND filter
    - obs_filter: observing filter
    - obs_mode: observing mode
    - fov_req: side length of the required square FOV (arcsec)
    - dit: dit time
    - ndit: number of dithered exposures
    - exptime: exposure time
    - use_exp_time_only: if True, only use the exposure time to set the
      exposure time; but this will be broken down into dit and ndit to
      avoid saturation, so integration parameters may change accordingly
    - attenuation_req: required attenuation factor
    - attenuation_fail: injected attenuation for the '_with_stray_light'
      file, which violates the requirement (< attenuation_req)
    - attenuation_pass: injected attenuation for the '_no_stray_light'
      file, which meets the requirement (> attenuation_req)
    - oof_direction: side of the detector the virtual out-of-field source
      sits beyond ('+x', '-x', '+y', '-y')
    - out_dir: directory to write to
    - intrapixel_capacitance: if False, turn off IPC

    OUTPUTS:
    - None; writes out files
    """
    if not attenuation_fail < attenuation_req < attenuation_pass:
        logging.error(
            "Need attenuation_fail < attenuation_req < attenuation_pass; got "
            "%s, %s, %s",
            attenuation_fail,
            attenuation_req,
            attenuation_pass,
        )
        sys.exit(1)

    # set up instrument

    properties = {
        "!OBS.filter_name": obs_filter,
        "!WCU.current_fpmask": fp_mask,
        "!OBS.pupil_mask": pp_mask,
    }
    if nd_filter is not None:
        properties["!OBS.nd_filter_name"] = nd_filter

    cmd = sim.UserCommands(
        use_instrument="METIS",
        set_modes=[obs_mode],
        properties=properties,
    )

    metis = sim.OpticalTrain(cmd)

    if not intrapixel_capacitance:
        metis["ipc"].include = False
        logging.info("Turning off inter-pixel capacitance")
    else:
        logging.info("Using default inter-pixel capacitance")

    # Generate a circularly-symmetric PSF from an annular aperture
    metis["pupil_masks"].change_mask(pp_mask)
    metis["psf"].update(pupil_mask=pp_mask + "_WCU")
    logging.info("Setting WCU PP mask to be: %s", pp_mask + "_WCU")

    wcu = metis["wcu_source"]

    bb_temp = 1000 * u.K

    pipe_2_log(
        lambda m=metis: m.effects.pprint_all(),
        msg="Optical train effects (initial)",
    )

    def readout():
        if use_exp_time_only:
            # Method 1 for setting exposure times: exptime alone
            return metis.readout(exptime=exptime, reset=False)[0]
        # Method 2 for setting exposure times: use ndit and dit together
        return metis.readout(ndit=ndit, dit=dit, reset=False)[0]

    #########################################################
    # BACKGROUND (two independent readouts: the second one is for the null frame
    # BACKGROUND_2 - BACKGROUND, which has the same noise as the
    # background-subtracted science frame but no signal)

    logging.info("Closing WCU BB aperture first to get a background ...")
    wcu.set_bb_aperture(value=0.0)
    metis.observe()
    outhdul_off = readout()
    log_readout_state(metis, "Background")
    sanity_check_user_inputs(metis, obs_filter, fp_mask, pp_mask)
    background = outhdul_off[1].data

    outhdul_off_2 = readout()
    log_readout_state(metis, "Background 2 (independent, for the null frame)")
    background_2 = outhdul_off_2[1].data
    null_std = float(np.std(background_2 - background))
    logging.info(
        "Std of null frame (BACKGROUND_2 - BACKGROUND): %.4g counts%s",
        null_std,
        "" if null_std > 0 else " -- NOT INDEPENDENT READOUTS!",
    )

    #########################################################
    # SCIENCE FRAME: single PSF in the middle

    logging.info("Re-opening WCU BB aperture to get a PSF ...")
    wcu.set_bb_aperture(value=1.0)
    metis.observe()
    logging.info("PSF model wavel range: %s", vars(metis["psf"])["_waveset"])
    logging.info("PSF model kernel shape: %s", vars(metis["psf"])["kernel"].shape)
    logging.info(
        "PSF model kernel file name: %s",
        vars(metis["psf"])["meta"]["filename"],
    )
    outhdul_on = readout()
    log_readout_state(metis, "Science")
    sanity_check_user_inputs(metis, obs_filter, fp_mask, pp_mask)

    psf_readout = outhdul_on[1].data
    ref_peak = float(np.max(psf_readout - background))
    logging.info("PSF peak (bckgd-subtracted): %.4f", ref_peak)

    #########################################################
    # ADD SCATTERED LIGHT from a virtual source outside the FOV;
    # one file per case

    cases = (
        ("FAIL", attenuation_fail, "_with_stray_light"),
        ("PASS", attenuation_pass, "_no_stray_light"),
    )
    for case, attenuation, file_suffix in cases:
        scatter = backbone.make_out_of_field_scatter(
            shape=background.shape,
            peak=ref_peak,
            attenuation=attenuation,
            direction=oof_direction,
        )
        raw_sci_readout = psf_readout + scatter
        bckgd_subted = raw_sci_readout - background

        logging.info(
            "%s case: injected attenuation %.1e (req. %.1e); "
            "mean scattered irradiance %.4f",
            case,
            attenuation,
            attenuation_req,
            np.mean(scatter),
        )

        primary_hdu = fits.PrimaryHDU(header=outhdul_on[0].header)
        hdul_new = fits.HDUList(
            [
                primary_hdu,
                fits.ImageHDU(data=raw_sci_readout, name="RAW_READOUT"),
                fits.ImageHDU(data=bckgd_subted, name="BCKGD_SUBTED"),
                fits.ImageHDU(data=background, name="BACKGROUND"),
                fits.ImageHDU(data=background_2, name="BACKGROUND_2"),
            ]
        )

        # add some stuff to the header, some of which may be redundant
        hdr = hdul_new[0].header
        hdr["FILTER"] = (obs_filter, "Observing filter")
        hdr["WCU_FP"] = (fp_mask, "WCU focal plane mask")
        hdr["WCU_PP"] = (pp_mask, "WCU pupil plane mask")
        hdr["BB_TEMP"] = (bb_temp.value, "BB temperature")
        if use_exp_time_only:
            hdr["EXPTIME"] = (exptime, "Exposure time")
        else:
            hdr["NDIT"] = (ndit, "Number of dithered exposures")
            hdr["DIT"] = (dit, "Det integration time")
        hdr["ATT_REQ"] = (attenuation_req, "Required out-of-field attenuation")
        hdr["ATT_INJ"] = (attenuation, "Injected attenuation factor")
        hdr["EXPECTED"] = (case, "Expected outcome vs. requirement")
        hdr["FOV_REQ"] = (fov_req, "Required square FOV side (arcsec)")
        hdr["OOF_DIR"] = (oof_direction, "Side of virtual out-of-field source")
        hdr["REF_PEAK"] = (ref_peak, "Peak of bckgd-subted PSF")

        basename_file_name_write = (
            "IMG_OPT_06_out_of_field_"
            + str(fp_mask)
            + "_pupil_mask_"
            + str(pp_mask)
            + "_filter_"
            + str(obs_filter)
            + file_suffix
            + ".fits"
        )
        abs_file_name_write = out_dir + basename_file_name_write

        hdul_new.writeto(abs_file_name_write, overwrite=True)
        logging.info("Saved out-of-field simulated data to %s", abs_file_name_write)


def main():
    """
    Generate IMG-OPT-06 out-of-field simulations for the configured filters.
    """

    stem = "/podman-share/metis_work/playing_with_scopesim/"

    now = datetime.datetime.now()
    log_dir = stem + "IMG_06_simmed_out_of_field_logs/"
    log_file_name = (
        log_dir
        + "log_IMG_06_simmed_out_of_field_"
        + now.strftime("%Y-%m-%d_%H-%M-%S")
        + ".txt"
    )
    # write simulated data here, where the ANALYSIS script reads it
    # (data_states config: data/IMG_06_simmed_out_of_field_data/...)
    out_dir = (
        "/podman-share/metis_work/METIS_MAIT_code_eckhart/"
        "data/IMG_06_simmed_out_of_field_data/"
    )

    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_name), logging.StreamHandler()],
        force=True,
    )
    os.makedirs(out_dir, exist_ok=True)

    logging.info("Log file created at %s", now.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("Log file name: %s", log_file_name)
    logging.info("Simmed file output directory: %s", out_dir)

    # Deep exposures: DIT as for IMG-OPT-04 (so no single DIT saturates on the PSF), but
    # many NDIT. Signal grows as NDIT and noise as sqrt(NDIT); with the shallow
    # IMG-OPT-04 settings (LM NDIT 2, N NDIT 1) the noise of the 31-pix-smoothed map
    # was ~8e-5 (LM) and ~7e-4 (N) of the PSF peak, so 1e-4 was not measurable. These
    # NDITs bring it to ~1e-5 of the PSF peak, i.e. 5-sigma ~ 0.5e-4 < 1e-4.
    obs_configs = [
        {
            "fp_mask": "pinhole_lm",
            "pp_mask": "PPS-CFO2",
            "obs_filter": "Br_alpha",
            "nd_filter": None,
            "dit": 0.065,
            "ndit": 128,
            "exptime": np.nan,
            "obs_mode": "wcu_img_lm",
            "use_exp_time_only": False,
            "fov_req": 11.0,
        },
        {
            "fp_mask": "pinhole_n",
            "pp_mask": "PPS-CFO2",
            "obs_filter": "N2",
            "nd_filter": None,
            "dit": 0.0025,
            "ndit": 5000,
            "exptime": np.nan,
            "obs_mode": "wcu_img_n",
            "use_exp_time_only": False,
            "fov_req": 14.0,
        },
    ]

    for config in obs_configs:

        logging.info(
            "! -- ScopeSim assumes blackbody injection efficiency of 1., but true "
            "value is expected to be between ~1/6 and 1."
        )

        generate_out_of_field_data(
            fp_mask=config["fp_mask"],
            pp_mask=config["pp_mask"],
            nd_filter=config["nd_filter"],
            obs_filter=config["obs_filter"],
            obs_mode=config["obs_mode"],
            fov_req=config["fov_req"],
            dit=config["dit"],
            ndit=config["ndit"],
            exptime=config["exptime"],
            use_exp_time_only=config["use_exp_time_only"],
            attenuation_req=1e4,
            attenuation_fail=1e3,  # '_with_stray_light': violates the requirement
            attenuation_pass=1e5,  # '_no_stray_light': meets the requirement
            oof_direction="+x",
            out_dir=out_dir,
            intrapixel_capacitance=True,
        )


if __name__ == "__main__":
    main()
