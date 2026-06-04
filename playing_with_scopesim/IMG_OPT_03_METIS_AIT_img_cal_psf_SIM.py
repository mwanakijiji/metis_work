# Maked simulated data for the IMG-OPT-04 PSF image quality test

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

# The procedure for measuring the image quality follows that of IMG-OPT-01, but with additional exposures
# made at positions designed to obtain fully spatially sampled imaging. A basic offset of half a pixel is used
# for the PSF image quality measurements in this test in horizontal, vertical, and diagonal directions. We also
# need to achieve a significantly higher SNR for accurate measurement of the FWHM, and better control of
# calibration and flat fielding errors for measurement of the Encircled Energy. The flat field will be derived
# from IMG-RAD-04.

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS

import scipy

from matplotlib import pyplot as plt
from matplotlib import colors
from astropy.visualization import ZScaleInterval

from scipy import ndimage

import time
import ipdb
import datetime
import io
import os
import sys
import logging

import scopesim as sim
from modules.helpers import pipe_2_log

# Edit this path if you have a custom install directory, otherwise comment it out. [For ReadTheDocs only]
sim.link_irdb("../../../")

# simulate observations with METIS (comment this out if packages already exist)
#sim.download_packages(["METIS", "ELT", "Armazones"])

# print versions of things
sim.bug_report()


def generate_psf_image_quality_data(fp_mask, pp_mask, nd_filter, obs_filter, obs_mode, angle_array, dit=1, ndit=1, exptime=0.01, use_exp_time_only=False, out_dir=None):
    '''
    Generate simulated data for the IMG-OPT-04 PSF image quality test
    
    INPUTS:
    - fp_mask: focal plane mask
    - pp_mask: pupil plane mask
    - nd_filter: ND filter
    - obs_filter: observing filter
    - obs_mode: observing mode
    - angle_array: array of clocking angles
    - dit: dit time
    - ndit: number of dithered exposures
    - exptime: exposure time
    - use_exp_time_only: if True, only use the exposure time to set the exposure time; but this will be broken down into dit and ndit to avoid saturation, so integration parameters may change accordingly

    OUTPUTS:
    - None; writes out files
    '''

    # set up instrument

    cmd = None # reset
    if nd_filter is not None:
        cmd = sim.UserCommands(use_instrument='METIS', set_modes=[obs_mode], properties={"!OBS.filter_name": obs_filter, "!WCU.current_fpmask": fp_mask, "!OBS.pupil_mask": pp_mask, "!OBS.nd_filter_name": nd_filter})
    else:
        cmd = sim.UserCommands(use_instrument='METIS', set_modes=[obs_mode], properties={"!OBS.filter_name": obs_filter, "!WCU.current_fpmask": fp_mask})

    metis = sim.OpticalTrain(cmd)

    # Generate a circularly-symmetric PSF from an annular aperture
    metis['pupil_masks'].change_mask(pp_mask)
    metis['psf'].update(pupil_mask=pp_mask+"_WCU")
    logging.info('Setting WCU PP mask to be: ' + str(pp_mask+"_WCU"))

    wcu = metis['wcu_source']


    bb_temp = 1000 * u.K

    pipe_2_log(lambda m=metis: m.effects.pprint_all(), msg="Optical train effects (initial)")

    #########################################################
    # BACKGROUND

    logging.info('Closing WCU BB aperture first to get a background ...')
    wcu.set_bb_aperture(value = 0.0)
    
    metis.observe()

    pipe_2_log(lambda m=metis: m.effects.pprint_all(), msg="Optical train effects (background)")

    if use_exp_time_only:
        # Method 1 for setting exposure times: exptime alone
        outhdul_off = metis.readout(exptime = exptime, reset=False)[0]
    else:
        # Method 2 for setting exposure times: use ndit and dit together
        outhdul_off = metis.readout(ndit = ndit, dit = dit, reset=False)[0]

    logging.info('--------------------------------')
    logging.info('Background readout:')
    logging.info('OBS filter: ' + str(metis.cmds.get("!OBS.filter_name")))
    logging.info('WCU FP mask: ' + str(metis.cmds.get("!WCU.current_fpmask")))
    logging.info('OBS PP mask: ' + str(metis.cmds.get("!OBS.pupil_mask")))
    logging.info('OBS ND filter: ' + str(metis.cmds.get("!OBS.nd_filter_name")))
    logging.info('NDIT: ' + str(metis.cmds["!OBS.ndit"]))
    logging.info('DIT: ' + str(metis.cmds["!OBS.dit"]))
    logging.info('WCU source state:')
    pipe_2_log(lambda m=metis: metis["wcu_source"].info(), msg="Optical train effects (background)")

    # sanity check that user inputs really are the same as what the instrument is using
    def sanity_check_user_inputs(metis, obs_filter, fp_mask, pp_mask):
        if metis.cmds.get("!OBS.filter_name") != obs_filter:
            logging.error('! ------- OBS filter: ' + str(metis.cmds.get("!OBS.filter_name")) + ' does not match user input: ' + str(obs_filter))
            exit()
        if metis.cmds.get("!WCU.current_fpmask") != fp_mask:
            logging.error('! ------- WCU FP mask: ' + str(metis.cmds.get("!WCU.current_fpmask")) + ' does not match user input: ' + str(fp_mask))
            exit()
        if metis.cmds.get("!OBS.pupil_mask") != pp_mask:
            logging.error('! ------- OBS PP mask: ' + str(metis.cmds.get("!OBS.pupil_mask")) + ' does not match user input: ' + str(pp_mask))
            exit()
        else:
            logging.info('User filter inputs match instrument inputs')
        return

    # check for background-taking
    #sanity_check_user_inputs(metis, obs_filter=obs_filter, fp_mask=fp_mask, pp_mask=pp_mask)
    background = outhdul_off[1].data

    #########################################################
    # SCIENCE FRAME

    logging.info('Re-opening WCU BB aperture to get a PSF ...')
    wcu.set_bb_aperture(value = 1.0) # open BB source

    metis.observe()
    # print the ingredients of the PSF generation
    # pipe_2_log(lambda m=metis: [print(f"{k}: {v}") for k, v in vars(m["psf"]).items()], msg="PSF ingredients") # this prints EVERYTHING
    logging.info('PSF model wavel range: ' + str(vars(metis['psf'])['_waveset']))
    logging.info('PSF model kernel shape: ' + str(vars(metis['psf'])['kernel'].shape))
    logging.info('PSF model kernel file name: ' + str(vars(metis['psf'])['meta']['filename']))
    pipe_2_log(lambda m=metis: str(vars(m['psf'])['_waveset']), msg="PSF model wavel range")

    # Get perfect PSF - no detector noise
    #hdul_perfect = metis.image_planes[0].hdu

    pipe_2_log(lambda m=metis: m.effects.pprint_all(), msg="Optical train effects (science)")

    if use_exp_time_only:
        # Method 1 for setting exposure times: exptime alone
        outhdul_on = metis.readout(exptime = exptime, reset=False)[0]
    else:
        # Method 2 for setting exposure times: use ndit and dit together
        outhdul_on = metis.readout(ndit = ndit, dit = dit, reset=False)[0]
    logging.info('--------------------------------')
    logging.info('Science readout:')
    logging.info('OBS filter: ' + str(metis.cmds.get("!OBS.filter_name")))
    logging.info('WCU FP mask: ' + str(metis.cmds.get("!WCU.current_fpmask")))
    logging.info('OBS PP mask: ' + str(metis.cmds.get("!OBS.pupil_mask")))
    logging.info('OBS ND filter: ' + str(metis.cmds.get("!OBS.nd_filter_name")))
    logging.info('NDIT:' + str(metis.cmds["!OBS.ndit"]))
    logging.info('DIT:' + str(metis.cmds["!OBS.dit"]))
    logging.info('WCU source state:')
    pipe_2_log(lambda m=metis: metis["wcu_source"].info(), msg="Optical train effects (background)")

    # check for science-taking
    #sanity_check_user_inputs(metis, obs_filter=obs_filter, fp_mask=fp_mask, pp_mask=pp_mask)
    #hdul_perfect = metis.image_planes[0].hdu

    # background-subtract
    raw_sci_readout = outhdul_on[1].data
    bckgd_subted = raw_sci_readout - background
    
    basename_file_name_write = 'IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_' + str(fp_mask) + '_pupil_mask_' + str(pp_mask) + '_filter_' + str(obs_filter) + '.fits'
    abs_file_name_write = out_dir + basename_file_name_write


    # Copy the primary header
    primary_hdu = fits.PrimaryHDU(header=outhdul_on[0].header)
    # Add background-subtracted readout as first extension
    hdu_bckgd_subted = fits.ImageHDU(data=bckgd_subted, name='BCKGD_SUBTED')
    # Add raw science readout as second extension
    hdu_raw_readout = fits.ImageHDU(data=raw_sci_readout, name='RAW_READOUT')
    # Add background as third extension
    hdu_background = fits.ImageHDU(data=background, name='BACKGROUND')
    hdul_new = fits.HDUList([primary_hdu, hdu_bckgd_subted, hdu_raw_readout, hdu_background])

    # add some stuff to the header, some of which may be redundant
    hdul_new[0].header['FILTER'] = (obs_filter, 'Observing filter')
    hdul_new[0].header['WCU_FP'] = (fp_mask, 'WCU focal plane mask')
    hdul_new[0].header['WCU_PP'] = (pp_mask, 'WCU pupil plane mask')
    hdul_new[0].header['BB_TEMP'] = (bb_temp.value, 'BB temperature')
    if ndit is not None:
        hdul_new[0].header['NDIT'] = (ndit, 'Number of dithered exposures')
        hdul_new[0].header['DIT'] = (dit, 'Det integration time')
    else:
        hdul_new[0].header['EXPTIME'] = (exptime, 'Exposure time')
    
    hdul_new.writeto(abs_file_name_write, overwrite=True)
    logging.info('Saved background-subtracted readout without aberrations to ' + abs_file_name_write)

    logging.info('--------------------------------')
    logging.info(f'Median of raw science readout: {np.median(raw_sci_readout):.4f}')
    logging.info(f'Median of background: {np.median(background):.4f}')
    logging.info(f'Median of background-subtracted readout: {np.median(bckgd_subted):.4f}')



def main():

    stem = '/podman-share/metis_work/playing_with_scopesim/'

    now = datetime.datetime.now()
    log_dir = stem + 'IMG_04_logs/'
    log_file_name = log_dir + 'log_IMG_04_simulation_psf_image_quality_' + now.strftime('%Y-%m-%d_%H-%M-%S') + '.txt'
    out_dir = stem + 'IMG_04_simmed_data/' # directory to write the simulated data to
    

    # Ensure log directory exists and force config in case handlers already set
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
    os.makedirs(out_dir, exist_ok=True)

    logging.info(f'Log file created at {now.strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info(f'Log file name: {log_file_name}')
    logging.info(f'Log file directory: {stem + "IMG_04_logs/"}')
    logging.info(f'Log file directory: {stem + "IMG_04_logs/"}')
    logging.info(f'Simmed file output directory: {out_dir}')

    # clocking angles for the PSF
    #angle_array = [0, 45, 60]
    angle_array = [0] # can be implemented later

    # LM filters
    # dict_keys(['open', 'Lp', 'short-L', 'L_spec', 'Mp', 'M_spec', 'Br_alpha', 'Br_alpha_ref', 'PAH_3.3', 'PAH_3.3_ref', 'CO_1-0_ice', 'CO_ref', 'H2O-ice', 'IB_4.05', 'HCI_L_short', 'HCI_L_long', 'HCI_M'])
    lm_obs_configs = [
        {"fp_mask": "grid_lm", "pp_mask": "PPS-CFO2", "obs_filter": "Br_alpha",     "nd_filter": None,      "dit": 0.065, "ndit": 2, "exptime": np.nan, "obs_mode": "wcu_img_lm", "use_exp_time_only": False},
        {"fp_mask": "grid_lm", "pp_mask": "PPS-CFO2", "obs_filter": "Br_alpha_ref", "nd_filter": "ND_OD1",  "dit": 0.4, "ndit": 5, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False},
        {"fp_mask": "grid_lm", "pp_mask": "PPS-CFO2", "obs_filter": "Lp",           "nd_filter": "ND_OD2",  "dit": float(3/8), "ndit": 3, "exptime": np.nan, "obs_mode": "wcu_img_lm", "use_exp_time_only": False},
        {"fp_mask": "grid_lm", "pp_mask": "PPS-CFO2", "obs_filter": "H2O-ice",      "nd_filter": "ND_OD1",      "dit": 0.06, "ndit": 1, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False}, 
        {"fp_mask": "grid_lm", "pp_mask": "PPS-CFO2", "obs_filter": "short-L",      "nd_filter": "ND_OD2",  "dit": 0.375, "ndit": 4, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False},
        {"fp_mask": "grid_lm", "pp_mask": "PPS-CFO2", "obs_filter": "PAH_3.3",      "nd_filter": "ND_OD1",  "dit": 0.1875, "ndit": 7, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False},
        {"fp_mask": "grid_lm", "pp_mask": "PPS-CFO2", "obs_filter": "PAH_3.3_ref",  "nd_filter": "ND_OD1",  "dit": 0.1875, "ndit": 8, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False},
        {"fp_mask": "grid_lm", "pp_mask": "PPS-CFO2", "obs_filter": "IB_4.05",      "nd_filter": "ND_OD1",  "dit": float(1/6), "ndit": 12, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False},
        {"fp_mask": "grid_lm", "pp_mask": "PPS-CFO2", "obs_filter": "HCI_L_short",  "nd_filter": "ND_OD2",  "dit": float(5/6), "ndit": 3, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False},
        {"fp_mask": "grid_lm", "pp_mask": "PPS-CFO2", "obs_filter": "HCI_L_long",   "nd_filter": "ND_OD1",  "dit": 0.06, "ndit": 25, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False},
        {"fp_mask": "grid_lm", "pp_mask": "PPS-CFO2", "obs_filter": "Mp",           "nd_filter": "ND_OD2",  "dit": 0.5, "ndit": 5, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False},
        {"fp_mask": "grid_lm", "pp_mask": "PPS-CFO2", "obs_filter": "CO_1-0_ice",   "nd_filter": "ND_OD1",  "dit": float(2/21), "ndit": 21, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False},
        {"fp_mask": "grid_lm", "pp_mask": "PPS-CFO2", "obs_filter": "CO_ref",       "nd_filter": "ND_OD1",  "dit": float(2/21), "ndit": 21, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False},
        {"fp_mask": "grid_lm", "pp_mask": "PPS-CFO2", "obs_filter": "HCI_M",        "nd_filter": "ND_OD1",  "dit": float(1/11), "ndit": 22, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False},
        {"fp_mask": "grid_lm", "pp_mask": "PPS-CFO2", "obs_filter": "L_spec",       "nd_filter": "ND_OD2",  "dit": 0.12, "ndit": 11, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False},
        {"fp_mask": "grid_lm", "pp_mask": "PPS-CFO2", "obs_filter": "M_spec",       "nd_filter": "ND_OD2",  "dit": float(2/7), "ndit": 7, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False},
    ]

    # for debug
    '''
    ipdb.set_trace()
    for config in lm_obs_configs:
        #_ = debug_mwe(fp_mask = "grid_lm", pp_mask = "Open", nd_filter = None, obs_filter = "Br_alpha", obs_mode = "wcu_img_lm", exptime = 0.1)
        #ipdb.set_trace()
        _ = debug_mwe(fp_mask = 'open', pp_mask= config["pp_mask"], nd_filter = "ND_OD3", obs_filter= config["obs_filter"], obs_mode= config["obs_mode"], exptime=config["exptime"])
        _ = debug_mwe(fp_mask = config["fp_mask"], pp_mask= config["pp_mask"], nd_filter = "ND_OD3", obs_filter= config["obs_filter"], obs_mode= config["obs_mode"], exptime=config["exptime"])
        exit()
        #ipdb.set_trace()
    ipdb.set_trace()
    '''

    for config in lm_obs_configs:

        # below line is kludge for testing just one combo
        #config = {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "Mp",           "nd_filter": "ND_OD2",  "dit": 1, "ndit": 10, "exptime": 1,   "obs_mode": "wcu_img_lm", "use_exp_time_only": True}

        generate_psf_image_quality_data(
            fp_mask=config["fp_mask"],
            pp_mask=config["pp_mask"],
            nd_filter=config["nd_filter"],
            obs_filter=config["obs_filter"],
            obs_mode=config["obs_mode"],
            angle_array=angle_array,
            dit=config["dit"],
            ndit=config["ndit"],
            exptime=config["exptime"],
            out_dir=out_dir
        )


if __name__ == "__main__":
    main()