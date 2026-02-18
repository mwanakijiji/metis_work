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
sim.bug_report()

# Edit this path if you have a custom install directory, otherwise comment it out. [For ReadTheDocs only]
sim.link_irdb("../../../")

# simulate observations with METIS (comment this out if packages already exist)
#sim.download_packages(["METIS", "ELT", "Armazones"])


def pipe_2_log(callable_func, msg="Output"):
    '''
    Capture stdout from any ScopeSim callable and write each line to the log.

    INPUTS:
    - callable_func: callable (no args) that prints to stdout when invoked
    - msg: string header to add to the log

    OUTPUTS:
    - None; writes out to log

    Example:
        pipe_2_log(lambda m=metis: m.effects.pprint_all(), msg="Optical train effects")
    '''
    buffer = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = buffer
        callable_func()
        output = buffer.getvalue()
    finally:
        sys.stdout = old_stdout
    logging.info('--------------------------------')
    logging.info(msg)
    for line in output.rstrip().splitlines():
        logging.info(line)


def generate_psf_image_quality_data(fp_mask, pp_mask, nd_filter, obs_filter, obs_mode, angle_array, dit=1, ndit=1, exptime=0.01, use_exp_time_only=False):
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
    cmd = sim.UserCommands(use_instrument='METIS', set_modes=[obs_mode], properties={"!OBS.filter_name": obs_filter, "!WCU.current_fpmask": fp_mask, "!OBS.pupil_mask": pp_mask, "!OBS.nd_filter_name": nd_filter})

    metis = sim.OpticalTrain(cmd)

    wcu = metis['wcu_source']

    bb_temp = 1000 * u.K

    pipe_2_log(lambda m=metis: m.effects.pprint_all(), msg="Optical train effects (initial)")

    #########################################################
    # BACKGROUND

    logging.info('Closing WCU BB aperture first to get a background ...')
    wcu.set_bb_aperture(value = 0.0)
    
    metis.observe()

    pipe_2_log(lambda m=metis: m.effects.pprint_all(), msg="Optical train effects (background)")

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


    background = outhdul_off[1].data

    #########################################################
    # SCIENCE FRAME

    logging.info('Re-opening WCU BB aperture to get a PSF ...')
    wcu.set_bb_aperture(value = 1.0) # open BB source
    #ipdb.set_trace()
    metis.observe()
    # print the ingredients of the PSF generation
    # pipe_2_log(lambda m=metis: [print(f"{k}: {v}") for k, v in vars(m["psf"]).items()], msg="PSF ingredients") # this prints EVERYTHING
    logging.info('PSF model wavel range: ' + str(vars(metis['psf'])['_waveset']))
    logging.info('PSF model kernel shape: ' + str(vars(metis['psf'])['kernel'].shape))
    logging.info('PSF model kernel file name: ' + str(vars(metis['psf'])['meta']['filename']))
    pipe_2_log(lambda m=metis: str(vars(m['psf'])['_waveset']), msg="PSF model wavel range")

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

    # background-subtract
    raw_sci_readout = outhdul_on[1].data
    bckgd_subted = raw_sci_readout - background
    
    ## BEGIN CHECK
    file_name_write = 'IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_' + str(fp_mask) + '_pupil_mask_' + str(pp_mask) + '_filter_' + str(obs_filter) + '.fits'

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
    
    hdul_new.writeto(file_name_write, overwrite=True)
    logging.info('Saved background-subtracted readout without aberrations to ' + file_name_write)

    logging.info('--------------------------------')
    logging.info(f'Median of raw science readout: {np.median(raw_sci_readout):.4f}')
    logging.info(f'Median of background: {np.median(background):.4f}')
    logging.info(f'Median of background-subtracted readout: {np.median(bckgd_subted):.4f}')


def main():

    stem = './'

    now = datetime.datetime.now()
    log_dir = stem + 'IMG_04_logs/'
    log_file_name = log_dir + 'log_IMG_04_simulation_psf_image_quality_' + now.strftime('%Y-%m-%d_%H-%M-%S') + '.txt'

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

    logging.info(f'Log file created at {now.strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info(f'Log file name: {log_file_name}')
    logging.info(f'Log file directory: {stem + "IMG_04_logs/"}')
    logging.info(f'Log file directory: {stem + "IMG_04_logs/"}')

    # clocking angles for the PSF
    #angle_array = [0, 45, 60]
    angle_array = [0]

    # LM filters
    # dict_keys(['open', 'Lp', 'short-L', 'L_spec', 'Mp', 'M_spec', 'Br_alpha', 'Br_alpha_ref', 'PAH_3.3', 'PAH_3.3_ref', 'CO_1-0_ice', 'CO_ref', 'H2O-ice', 'IB_4.05', 'HCI_L_short', 'HCI_L_long', 'HCI_M'])
    lm_obs_configs = [
        {"fp_mask": "pinhole_lm", "pp_mask": "Open", "obs_filter": "Br_alpha_ref", "nd_filter": "ND_OD1",  "dit": 0.2, "ndit": 5, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False},
        {"fp_mask": "pinhole_lm", "pp_mask": "Open", "obs_filter": "H2O-ice",      "nd_filter": "ND_OD1",      "dit": 0.04, "ndit": 1, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False}, 
        {"fp_mask": "pinhole_lm", "pp_mask": "Open", "obs_filter": "L_spec",       "nd_filter": "ND_OD2",  "dit": float(1/11), "ndit": 11, "exptime": np.nan,   "obs_mode": "wcu_img_lm", "use_exp_time_only": False}
    ]


    for config in lm_obs_configs:

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
        )


if __name__ == "__main__":
    main()