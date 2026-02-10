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

import scopesim as sim
sim.bug_report()

# Edit this path if you have a custom install directory, otherwise comment it out. [For ReadTheDocs only]
sim.link_irdb("../../../")

# simulate observations with METIS (comment this out if packages already exist)
#sim.download_packages(["METIS", "ELT", "Armazones"])

def debug_mwe(fp_mask, pp_mask, nd_filter, obs_filter, obs_mode, exptime=0.01):
    '''
    Generate simulated data for the IMG-OPT-04 PSF image quality test
    
    INPUTS:
    - fp_mask: focal plane mask
    - pp_mask: pupil plane mask
    - nd_filter: ND filter
    - obs_filter: observing filter
    - obs_mode: observing mode
    - exptime: exposure time
    - use_exp_time_only: if True, only use the exposure time to set the exposure time; if False, use the exposure time and ndit to set the exposure time

    OUTPUTS:
    - None
    '''

    # set up instrument
    cmd = None # reset
    if nd_filter is not None:
        cmd = sim.UserCommands(use_instrument='METIS', set_modes=[obs_mode], properties={"!OBS.filter_name": obs_filter, "!WCU.current_fpmask": fp_mask, "!OBS.pupil_mask": pp_mask, "!OBS.nd_filter_name": nd_filter})
    else:
        cmd = sim.UserCommands(use_instrument='METIS', set_modes=[obs_mode], properties={"!OBS.filter_name": obs_filter, "!WCU.current_fpmask": fp_mask, "!OBS.pupil_mask": pp_mask})

    metis = sim.OpticalTrain(cmd)

    wcu = metis['wcu_source']

    bb_temp = 1000 * u.K

    #########################################################
    # BACKGROUND

    print('Closing WCU BB aperture first to get a background ...')
    wcu.set_bb_aperture(value = 0.0)
    metis.observe()

    print('Before background readout:', metis.cmds.get("!OBS.filter_name"), metis.cmds.get("!WCU.current_fpmask"), metis.cmds.get("!OBS.pupil_mask"), metis.cmds.get("!OBS.nd_filter_name"))
    metis.effects.pprint_all()

    outhdul_off = metis.readout(exptime = exptime, reset=False)[0]

    background = outhdul_off[1].data

    #########################################################
    # SCIENCE FRAME

    print('Re-opening WCU BB aperture to get a PSF ...')
    wcu.set_bb_aperture(value = 1.0) # open BB source
    metis.observe()

    #cmd = sim.UserCommands(use_instrument='METIS', set_modes=[obs_mode], properties={"!OBS.filter_name": obs_filter, "!WCU.current_fpmask": fp_mask, "!OBS.pupil_mask": pp_mask, "!OBS.nd_filter_name": nd_filter})
    #metis = sim.OpticalTrain(cmd)

    print('--------------------------------')
    print('Current Observing filter:', obs_filter)
    print('Current WCU FP mask:', wcu.fpmask)
    print('Current WCU PP mask:', pp_mask)
    print('Opening WCU BB aperture...')

    # Get perfect PSF - no detector noise
    #hdul_perfect = metis.image_planes[0].hdu

    print('--------------------------------')
    print('Before science readout:', metis.cmds.get("!OBS.filter_name"), metis.cmds.get("!WCU.current_fpmask"), metis.cmds.get("!OBS.pupil_mask"), metis.cmds.get("!OBS.nd_filter_name"))
    metis.effects.pprint_all()

    outhdul_on = metis.readout(exptime = exptime, reset=False)[0]

    # background-subtract
    raw_sci_readout = outhdul_on[1].data
    bckgd_subted = raw_sci_readout - background

    print('--------------------------------')
    print('Medians:')
    print('--------------------------------')
    print('Raw science readout:', np.median(raw_sci_readout))
    print('Background:', np.median(background))
    print('Bckgd-subtracted readout:', np.median(bckgd_subted))

    plt.clf()
    plt.imshow(raw_sci_readout, origin='lower')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('raw_sci_readout.png')
    plt.close()

    plt.clf()
    plt.imshow(background, origin='lower')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('background.png')
    plt.close()

    plt.clf()
    plt.imshow(bckgd_subted, origin='lower')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('bckgd_subted.png')
    plt.close()



def generate_psf_image_quality_data(fp_mask, pp_mask, nd_filter, obs_filter, obs_mode, angle_array, dit=1, ndit=1, exptime=0.01, use_exp_time_only=True):
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
    - use_exp_time_only: if True, only use the exposure time to set the exposure time; if False, use the exposure time and ndit to set the exposure time

    OUTPUTS:
    - None; writes out files
    '''

    # set up instrument
    cmd = None # reset
    if nd_filter is not None:
        cmd = sim.UserCommands(use_instrument='METIS', set_modes=[obs_mode], properties={"!OBS.filter_name": obs_filter, "!WCU.current_fpmask": fp_mask, "!OBS.pupil_mask": pp_mask, "!OBS.nd_filter_name": nd_filter})
    else:
        cmd = sim.UserCommands(use_instrument='METIS', set_modes=[obs_mode], properties={"!OBS.filter_name": obs_filter, "!WCU.current_fpmask": fp_mask, "!OBS.pupil_mask": pp_mask})

    metis = sim.OpticalTrain(cmd)

    wcu = metis['wcu_source']

    bb_temp = 1000 * u.K

    metis.effects.pprint_all()

    #########################################################
    # BACKGROUND

    print('Closing WCU BB aperture first to get a background ...')
    wcu.set_bb_aperture(value = 0.0)
    metis.observe()

    print('Before background readout:', metis.cmds.get("!OBS.filter_name"), metis.cmds.get("!WCU.current_fpmask"), metis.cmds.get("!OBS.pupil_mask"), metis.cmds.get("!OBS.nd_filter_name"))
    metis.effects.pprint_all()

    if use_exp_time_only:
        # Method 1 for setting exposure times: exptime alone
        outhdul_off = metis.readout(exptime = exptime, reset=False)[0]
    else:
        # Method 2 for setting exposure times: use ndit and dit together
        outhdul_off = metis.readout(ndit = ndit, dit = dit, reset=False)[0]

    background = outhdul_off[1].data

    #########################################################
    # SCIENCE FRAME

    print('Re-opening WCU BB aperture to get a PSF ...')
    wcu.set_bb_aperture(value = 1.0) # open BB source
    metis.observe()

    #cmd = sim.UserCommands(use_instrument='METIS', set_modes=[obs_mode], properties={"!OBS.filter_name": obs_filter, "!WCU.current_fpmask": fp_mask, "!OBS.pupil_mask": pp_mask, "!OBS.nd_filter_name": nd_filter})
    #metis = sim.OpticalTrain(cmd)

    print('--------------------------------')
    print('Current Observing filter:', obs_filter)
    print('Current WCU FP mask:', wcu.fpmask)
    print('Current WCU PP mask:', pp_mask)
    print('Opening WCU BB aperture...')

    # Get perfect PSF - no detector noise
    #hdul_perfect = metis.image_planes[0].hdu

    print('Before science readout:', metis.cmds.get("!OBS.filter_name"), metis.cmds.get("!WCU.current_fpmask"), metis.cmds.get("!OBS.pupil_mask"), metis.cmds.get("!OBS.nd_filter_name"))
    metis.effects.pprint_all()

    if use_exp_time_only:
        # Method 1 for setting exposure times: exptime alone
        outhdul_on = metis.readout(exptime = exptime, reset=False)[0]
    else:
        # Method 2 for setting exposure times: use ndit and dit together
        outhdul_on = metis.readout(ndit = ndit, dit = dit, reset=False)[0]

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
    print('Saved background-subtracted readout without aberrations to ' + file_name_write)

    print('--------------------------------')
    print('Medians:')
    print('--------------------------------')
    print('Raw science readout:', np.median(raw_sci_readout))
    print('Background:', np.median(background))
    print('Bckgd-subtracted readout:', np.median(bckgd_subted))

    ## END CHECK
    #exit()

    '''
    # loop over rotation angles
    for angle in angle_array:


        ipdb.set_trace()

        print('Rotating by ' + str(angle) + ' degrees')
        sci_rotated = ndimage.rotate(outhdul_on[1].data, angle, order=3, reshape=False)
        background_rotated = ndimage.rotate(background, angle, order=3, reshape=False)
        bckgd_subted_rotated = ndimage.rotate(bckgd_subted, angle, order=3, reshape=False)

        outhdul_on[1].data = bckgd_subted_rotated # reassign; note that this step will have to be done later outside the ScopeSim context
        ipdb.set_trace()
        
        plt.clf()
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(bckgd_subted)
        plt.imshow(sci_rotated, origin='lower', vmin=vmin, vmax=vmax)
        plt.title(f'Raw readout\nWCU FP mask: ' + str(fp_mask) + '\n' + 'WCU PP mask: ' + str(pp_mask) + '\n' + 'Observing filter: ' + str(obs_filter) + '\n' + 'BB temp: ' + str(bb_temp) + '\n' + 'Clocking angle: ' + str(angle))
        plt.colorbar()
        plt.tight_layout()
        file_name_plot_raw_readout = 'IMG_OPT_04_wcu_focal_mask_' + str(fp_mask) + '_pupil_mask_' + str(pp_mask) + '_filter_' + str(obs_filter) + '_clocking_angle_' + str(angle) + '_raw_readout.png'
        plt.savefig(file_name_plot_raw_readout)
        #plt.show()
        plt.close()
        print('Saved PNG of raw readout to ' + file_name_plot_raw_readout)

        plt.clf()
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(bckgd_subted)
        plt.imshow(background_rotated, origin='lower', vmin=vmin, vmax=vmax)
        plt.title(f'Background\nWCU FP mask: ' + str(fp_mask) + '\n' + 'WCU PP mask: ' + str(pp_mask) + '\n' + 'Observing filter: ' + str(obs_filter) + '\n' + 'BB temp: ' + str(bb_temp) + '\n' + 'Clocking angle: ' + str(angle))
        plt.colorbar()
        plt.tight_layout()
        file_name_plot_background = 'IMG_OPT_04_wcu_focal_mask_' + str(fp_mask) + '_pupil_mask_' + str(pp_mask) + '_filter_' + str(obs_filter) + '_clocking_angle_' + str(angle) + '_background.png'
        plt.savefig(file_name_plot_background)
        #plt.show()
        plt.close()
        print('Saved PNG of background to ' + file_name_plot_background)

        # detector
        plt.clf()
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(bckgd_subted)
        plt.imshow(bckgd_subted_rotated, origin='lower', vmin=vmin, vmax=vmax)
        plt.title(f'Bckgd-subtracted readout\nWCU FP mask: ' + str(fp_mask) + '\n' + 'WCU PP mask: ' + str(pp_mask) + '\n' + 'Observing filter: ' + str(obs_filter) + '\n' + 'BB temp: ' + str(bb_temp) + '\n' + 'Clocking angle: ' + str(angle))
        plt.colorbar()
        plt.tight_layout()
        file_name_plot_bckgd_subtracted_readout = 'IMG_OPT_04_wcu_focal_mask_' + str(fp_mask) + '_pupil_mask_' + str(pp_mask) + '_filter_' + str(obs_filter) + '_clocking_angle_' + str(angle) + '_bckgd_subtracted_readout.png'
        plt.savefig(file_name_plot_bckgd_subtracted_readout)
        #plt.show()
        plt.close()
        print('Saved PNG of bckgd-subtracted readout to ' + file_name_plot_bckgd_subtracted_readout)

        # histogram
        plt.clf()
        plt.hist(bckgd_subted.ravel(), bins=200)
        plt.title('Bckgd-subtracted histogram; WCU FP mask: ' + str(fp_mask) + '\n' + 'WCU PP mask: ' + str(pp_mask) + '\n' + 'Observing filter: ' + str(obs_filter) + '\n' + 'BB temp: ' + str(bb_temp) + '\n' + 'Clocking angle: ' + str(angle))
        plt.tight_layout()
        file_name_plot_bckgd_subtracted_histogram = 'IMG_OPT_04_wcu_focal_mask_' + str(fp_mask) + '_pupil_mask_' + str(pp_mask) + '_filter_' + str(obs_filter) + '_clocking_angle_' + str(angle) + '_bckgd_subtracted_histogram.png'
        plt.savefig(file_name_plot_bckgd_subtracted_histogram) 
        #plt.show()
        plt.close()
        print('Saved PNG of bckgd-subtracted histogram to ' + file_name_plot_bckgd_subtracted_histogram)


        # save background-subtracted to FITS file, with filter and other info in the header
        # Make a new FITS file (let's call it file_name_write_all_layers) with the same primary header as outhdul,
        # but with: 
        #  [0] header, 
        #  [1] background-subtracted image (bckgd_subted_rotated), 
        #  [2] raw_sci_readout, 
        #  [3] background_rotated

 
        #hdul_new.writeto(file_name_write_all_layers, overwrite=True)

        file_name_write = 'IMG_OPT_04_wcu_focal_mask_bckgrnd_subted_' + str(fp_mask) + '_pupil_mask_' + str(pp_mask) + '_filter_' + str(obs_filter) + '_clocking_angle_' + str(angle) + '.fits'

        # Copy the primary header
        primary_hdu = fits.PrimaryHDU(header=outhdul_on[0].header)
        # Add background-subtracted readout as first extension
        hdu_bckgd_subted = fits.ImageHDU(data=bckgd_subted_rotated, name='BCKGD_SUBTED')
        # Add raw science readout as second extension
        hdu_raw_readout = fits.ImageHDU(data=raw_sci_readout, name='RAW_READOUT')
        # Add background as third extension
        hdu_background = fits.ImageHDU(data=background_rotated, name='BACKGROUND')
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
        print('Saved background-subtracted readout without aberrations to ' + file_name_write)
    '''


def main():

    # initialize instrument here just to obtain filter lists: LM band
    '''
    cmd = sim.UserCommands(use_instrument='METIS', set_modes=['wcu_img_lm'])
    metis = sim.OpticalTrain(cmd)
    lm_filters_list = metis["filter_wheel"].filters.keys() # filters
    #lm_fpmasks_list = ["pinhole_lm", "grid_lm"] # FP masks
    lm_fpmasks_list = ["grid_lm"] # FP masks

    # same for N band
    cmd = sim.UserCommands(use_instrument='METIS', set_modes=['wcu_img_n'])
    metis = sim.OpticalTrain(cmd)
    n_filters_list = metis["filter_wheel"].filters.keys() # filters
    n_fpmasks_list = ["pinhole_n"] # FP masks
    '''

    # clocking angles for the PSF
    #angle_array = [0, 45, 60]
    angle_array = [0]

    # LM filters
    # dict_keys(['open', 'Lp', 'short-L', 'L_spec', 'Mp', 'M_spec', 'Br_alpha', 'Br_alpha_ref', 'PAH_3.3', 'PAH_3.3_ref', 'CO_1-0_ice', 'CO_ref', 'H2O-ice', 'IB_4.05', 'HCI_L_short', 'HCI_L_long', 'HCI_M'])
    lm_obs_configs = [
        {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "Br_alpha",     "nd_filter": None,      "dit": 1, "ndit": 10, "exptime": 0.1, "obs_mode": "wcu_img_lm", "use_exp_time_only": True},
        {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "Br_alpha_ref", "nd_filter": "ND_OD1",  "dit": 1, "ndit": 10, "exptime": 1,   "obs_mode": "wcu_img_lm", "use_exp_time_only": True},
        {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "Lp",           "nd_filter": "ND_OD2",  "dit": 1, "ndit": 10, "exptime": 0.5, "obs_mode": "wcu_img_lm", "use_exp_time_only": True},
        {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "H2O-ice",      "nd_filter": None,      "dit": 1, "ndit": 10, "exptime": 1,   "obs_mode": "wcu_img_lm", "use_exp_time_only": True},
        {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "short-L",      "nd_filter": "ND_OD2",  "dit": 1, "ndit": 10, "exptime": 1,   "obs_mode": "wcu_img_lm", "use_exp_time_only": True},
        {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "PAH_3.3",      "nd_filter": "ND_OD1",  "dit": 1, "ndit": 10, "exptime": 1,   "obs_mode": "wcu_img_lm", "use_exp_time_only": True},
        {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "PAH_3.3_ref",  "nd_filter": "ND_OD1",  "dit": 1, "ndit": 10, "exptime": 1,   "obs_mode": "wcu_img_lm", "use_exp_time_only": True},
        {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "IB_4.05",      "nd_filter": "ND_OD1",  "dit": 1, "ndit": 10, "exptime": 1,   "obs_mode": "wcu_img_lm", "use_exp_time_only": True},
        {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "HCI_L_short",  "nd_filter": "ND_OD2",  "dit": 1, "ndit": 10, "exptime": 1,   "obs_mode": "wcu_img_lm", "use_exp_time_only": True},
        {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "HCI_L_long",   "nd_filter": "ND_OD1",  "dit": 1, "ndit": 10, "exptime": 1,   "obs_mode": "wcu_img_lm", "use_exp_time_only": True},
        {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "Mp",           "nd_filter": "ND_OD2",  "dit": 1, "ndit": 10, "exptime": 1,   "obs_mode": "wcu_img_lm", "use_exp_time_only": True},
        {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "CO_1-0_ice",   "nd_filter": "ND_OD1",  "dit": 1, "ndit": 10, "exptime": 1,   "obs_mode": "wcu_img_lm", "use_exp_time_only": True},
        {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "CO_ref",       "nd_filter": "ND_OD1",  "dit": 1, "ndit": 10, "exptime": 1,   "obs_mode": "wcu_img_lm", "use_exp_time_only": True},
        {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "HCI_M",        "nd_filter": "ND_OD1",  "dit": 1, "ndit": 10, "exptime": 1,   "obs_mode": "wcu_img_lm", "use_exp_time_only": True},
        {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "L_spec",       "nd_filter": "ND_OD2",  "dit": 1, "ndit": 10, "exptime": 1,   "obs_mode": "wcu_img_lm", "use_exp_time_only": True},
        {"fp_mask": "grid_lm", "pp_mask": "Open", "obs_filter": "M_spec",       "nd_filter": "ND_OD2",  "dit": 1, "ndit": 10, "exptime": 1,   "obs_mode": "wcu_img_lm", "use_exp_time_only": True},
    ]

    # debug
    ipdb.set_trace()
    for config in lm_obs_configs:
        #_ = debug_mwe(fp_mask = "grid_lm", pp_mask = "Open", nd_filter = None, obs_filter = "Br_alpha", obs_mode = "wcu_img_lm", exptime = 0.1)
        #ipdb.set_trace()
        _ = debug_mwe(fp_mask = 'open', pp_mask= config["pp_mask"], nd_filter = "ND_OD3", obs_filter= config["obs_filter"], obs_mode= config["obs_mode"], exptime=config["exptime"])
        _ = debug_mwe(fp_mask = config["fp_mask"], pp_mask= config["pp_mask"], nd_filter = "ND_OD3", obs_filter= config["obs_filter"], obs_mode= config["obs_mode"], exptime=config["exptime"])
        exit()
        #ipdb.set_trace()
    ipdb.set_trace()

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
        )
        ipdb.set_trace()
        #exit()


if __name__ == "__main__":
    main()