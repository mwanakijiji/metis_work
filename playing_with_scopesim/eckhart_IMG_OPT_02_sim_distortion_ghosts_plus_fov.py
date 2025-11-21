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

from typing import Any
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
import pickle

import scipy

from matplotlib import pyplot as plt
from matplotlib import colors
from astropy.visualization import ZScaleInterval

import time
import ipdb

import scopesim as sim
from scopesim.effects.electronic.exposure import AutoExposure

sim.bug_report()

# Edit this path if you have a custom install directory, otherwise comment it out. [For ReadTheDocs only]
sim.link_irdb("../../../")

write_dir = '/podman-share/metis_work/playing_with_scopesim/IMG_02_output/'

# simulate observations with METIS (comment this out if packages already exist)
#sim.download_packages(["METIS", "ELT", "Armazones"])

def four_corner_psf(fp_mask, pp_mask, obs_filter, dither_position_array, fov, ps, obs_mode, source='bb_source'):
    '''
    Image the source at each corner of the FOV.
    
    INPUTS:
    - fp_mask: focal plane mask
    - pp_mask: pupil plane mask
    - obs_filter: observing filter
    - dither_position_array: array of dither positions
    - fov: field of view
    - ps: pixel scale
    - obs_mode: observing mode
    - source: source of light ('bb_source' or 'laser')

    OUTPUT:
    - writes pngs and FITS to disk
    '''

    # set up instrument
    cmd = sim.UserCommands(use_instrument='METIS', set_modes=[obs_mode])
    metis = sim.OpticalTrain(cmd)

    metis.effects.pprint_all()
    wcu = metis['wcu_source']

    bb_temp = 10 * u.K
    NDIT, EXPTIME = 10, 0.5

    print('Generating ' + str(fp_mask)) 
    wcu.set_fpmask(fp_mask)

    print('Closing WCU BB aperture first for background ...')
    # background

    '''
    ## BEGIN DEBUGGING
    DIT, NDIT = 30, 120
    wcu.set_bb_aperture(0)
    metis.observe()
    readout_off = metis.readout(dit=DIT, ndit=NDIT)[0]
    wcu.set_bb_aperture(1.)
    metis.observe()
    readout_nom = metis.readout(dit=DIT, ndit=NDIT)[0]
    ## END DEBUGGING
    '''

    wcu.set_bb_aperture(value = 0.0)
    metis.observe()
    #outhdul_off = metis.readout(ndit = NDIT, exptime = EXPTIME)[0]
    #outhdul_off = metis.readout(exptime = 0.001*EXPTIME)[0]
    #outhdul_off = metis.readout(ndit=40, dit=0.25)[0]
    #outhdul_off = metis.readout(ndetector_readout_mode='auto', dit=None, ndit=None)
    outhdul_off = metis.readout(exptime = 25)[0]

    background = outhdul_off[1].data

    print('Opening WCU BB aperture...')
    wcu.set_bb_aperture(value = 1.0) # open BB source

    metis["filter_wheel"].change_filter(obs_filter)

    print('Dithering PSF to each corner of the FOV. This is agnostic to whether the source is being translated, ' + 
        'the telescope is being nodded, or an optical element is moving')

    metis.observe()
    outhdul = metis.readout(ndit = NDIT, exptime = EXPTIME)[0]

    # dither psf to each corner of the FOV
    for dither_pos in dither_position_array:

        # rescale dither position to the designed pixel scale, FOV

        fov_half = int(np.floor(0.5 * fov)) # round down to nearest integer
        dither_pos = tuple(np.array(dither_pos) * fov_half) # arcsec
        dither_pos_pix = tuple(int(x) for x in (np.array(dither_pos) * 0.45 * 2048 * (1/5))) # pixels; 1/5 factor to account for 5 arcsec across one half of one side

        print('--------------------------------')
        print('Current Observing filter:', obs_filter)
        print('Current WCU FP mask:', wcu.fpmask)
        print('Current WCU PP mask:', pp_mask)
        print('Current dither position:', dither_pos)

        # dither by shifting the FP mask
        # (note these shifts are absolute, not relative)
        # ! ------ this functionality is messed up! just roll the image ------ !
        #wcu.set_fpmask(fp_mask, angle=0, shift=dither_pos)

        # ersatz dithering: just roll the image with no dither
        # This is correct: np.roll for 2D, axis=(0,1) for (y,x)
        ersatz_dither = np.roll(
            outhdul[1].data, shift=(dither_pos_pix[1], dither_pos_pix[0]), axis=(0, 1)
        )

        # background-subtract
        #bckgd_subted = outhdul[1].data - background
        bckgd_subted = ersatz_dither - background

        # detector
        plt.clf()
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(bckgd_subted)
        plt.imshow(bckgd_subted, origin='lower', vmin=vmin, vmax=vmax)
        plt.title(f'Readout\nWCU FP mask: ' + str(fp_mask) + '\n' + 'WCU PP mask: ' + str(pp_mask) + \
            '\n' + 'Observing filter: ' + str(obs_filter) + '\n' + 'BB temp: ' + str(bb_temp) + \
                '\n' + 'dither position: ' + str(dither_pos))
        plt.tight_layout()
        plt.show()
        plt.close()

        # histogram
        plt.clf()
        plt.hist(bckgd_subted.ravel(), bins=200)
        plt.title('Bckgd-subtracted histogram; WCU FP mask: ' + str(fp_mask) + '\n' + 'WCU PP mask: ' + str(pp_mask) + '\n' + 'Observing filter: ' + str(obs_filter) + '\n' + 'BB temp: ' + str(bb_temp))
        plt.tight_layout()
        plt.show()
        plt.close()

        # save to FITS file, with filter and other info in the header
        file_name = write_dir + 'IMG_OPT_02_wcu_focal_plane_' + str(fp_mask) + '.fits'
        outhdul[0].data = bckgd_subted # background-subtracted image (note [1] still contains the raw readout)
        outhdul[0].header['FILTER'] = (obs_filter, 'Observing filter')
        outhdul[0].header['WCU_FP'] = (fp_mask, 'WCU focal plane mask')
        #outhdul[0].header['DITH_POS'] = (dither_pos, 'WCU dither position')
        outhdul[0].header['WCU_PP'] = (pp_mask, 'WCU pupil plane mask')
        outhdul[0].header['BB_TEMP'] = (bb_temp.value, 'BB temperature')
        outhdul[0].header['NDIT'] = (NDIT, 'Number of dithered exposures')
        outhdul[0].header['EXPTIME'] = (EXPTIME, 'Exposure time')
        
        outhdul.writeto(file_name, overwrite=True)

        print('Saved readout without aberrations to ' + file_name)

        # do a hackneyed aberration: blurring made to look like defocus 
        file_name = write_dir + 'IMG_OPT_02_wcu_focal_plane_' + str(fp_mask) + '_blur.fits'
        outhdul[1].data = scipy.ndimage.gaussian_filter(outhdul[1].data, sigma=3)
        outhdul.writeto(file_name, overwrite=True)
        print('Saved readout with aberrations to ' + file_name)

        # gather local variables
        local_vars = {}
        # Only include scalars, arrays, and strings -- exclude objects like metis, wcu, plt, etc.
        for var_name, var_value in locals().items():
            # Exclude callables and modules
            if callable(var_value):
                continue
            if hasattr(var_value, "__module__") and var_value.__module__ == 'builtins':
                local_vars[var_name] = var_value
            # Accept numpy arrays and astropy quantities as "arrays"
            elif "numpy" in str(type(var_value)):
                local_vars[var_name] = var_value
            elif "astropy" in str(type(var_value)):
                # Accept astropy.units.Quantity as array/scalar
                if "Quantity" in str(type(var_value)):
                    local_vars[var_name] = var_value
            elif isinstance(var_value, str):
                local_vars[var_name] = var_value

        # Write variables to a pickle file
        file_name_pickle = write_dir + 'all_variables_' + str(fp_mask) + '_' + str(obs_filter) + '.pkl'
        with open(file_name_pickle, 'wb') as f:
            pickle.dump(local_vars, f)
        print('Pickled variables to ', file_name_pickle)


def image_fp_masks(fp_mask, obs_filter, pp_mask, obs_mode, source='bb_source'):
    '''
    Get a grid image of PSFs.

    INPUTS:
    - fp_mask: focal plane mask
    - obs_filter: observing filter
    - pp_mask: pupil plane mask
    - obs_mode: observing mode
    - source: source of light ('bb_source' or 'laser')

    OUTPUT:
    - writes pngs and FITS to disk
    '''

    # set up instrument
    cmd = sim.UserCommands(use_instrument='METIS', set_modes=[obs_mode])
    metis = sim.OpticalTrain(cmd)



    metis.effects.pprint_all()

    NDIT, EXPTIME = 1, 0.2

    # initialize stuff
    background = None
    bb_temp = 0 * u.K
    if source == 'bb_source':
        # default source is BB
        wcu = metis['wcu_source']
        print('Generating ' + str(fp_mask)) 
        wcu.set_fpmask(fp_mask)
        print('Closing WCU BB aperture first for background ...')
        # background
        bb_temp = 1000 * u.K
        wcu.set_bb_aperture(value = 0.0)
        # take background
        metis.observe()
        outhdul_off = metis.readout(ndit = NDIT, exptime = EXPTIME)[0]
        background = outhdul_off[1].data
        wcu.set_bb_aperture(value = 1.0) # open BB source
    elif source == 'laser':
        metis["wcu_source"].set_lamp("laser")
        wcu = metis['wcu_source']
    else:
        raise ValueError('Invalid source: ' + str(source))

    metis["filter_wheel"].change_filter(obs_filter)

    print('--------------------------------')
    print('Current Observing filter:', obs_filter)
    print('Current WCU FP mask:', wcu.fpmask)
    print('Current WCU PP mask:', pp_mask)
    #print('Next absolute dither position:', dither_pos)

    # dither by shifting the FP mask
    # (note these shifts are absolute, not relative)
    wcu.set_fpmask(fp_mask)

    print('Opening WCU BB aperture...')

    metis.observe()
    outhdul = metis.readout(ndit = NDIT, exptime = EXPTIME)[0]

    # background-subtract
    # if there is a background

    if background is not None:
        bckgd_subted = outhdul[1].data - background
    else: 
        bckgd_subted = outhdul[1].data

    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(bckgd_subted)
    plt.clf()
    plt.imshow(bckgd_subted, origin='lower', vmin=vmin, vmax=vmax)
    plt.title('Bckgd-subtracted; WCU FP mask: ' + str(fp_mask) + '\n' + 'WCU PP mask: ' + str(pp_mask) + '\n' + \
        'Observing filter: ' + str(obs_filter) + '\n' + 'Source: ' + str(source) + '\n' + \
            'BB temp (if BB): ' + str(bb_temp))
    plt.tight_layout()
    plt.show()
    plt.close()

    # histogram
    plt.clf()
    plt.hist(bckgd_subted.ravel(), bins=200)
    plt.title('Bckgd-subtracted histogram; WCU FP mask: ' + str(fp_mask) + '\n' + 'WCU PP mask: ' + str(pp_mask) + '\n' + \
        'Observing filter: ' + str(obs_filter) + '\n' + 'Source: ' + str(source) + '\n' + \
            'BB temp (if BB): ' + str(bb_temp))
    plt.tight_layout()
    plt.show()
    plt.close()

    # save to FITS file, with filter and other info in the header
    file_name = write_dir + 'IMG_OPT_02_image_' + str(fp_mask) + '_' + str(obs_filter) + '.fits'
    outhdul[0].data = bckgd_subted # background-subtracted image (note [1] still contains the raw readout)
    outhdul[0].header['FILTER'] = (obs_filter, 'Observing filter')
    outhdul[0].header['WCU_FP'] = (fp_mask, 'WCU focal plane mask')
    outhdul[0].header['WCU_PP'] = (pp_mask, 'WCU pupil plane mask')
    outhdul[0].header['BB_TEMP'] = (bb_temp.value, 'BB temperature')
    outhdul[0].header['SOURCE'] = (source, 'Source')
    outhdul[0].header['NDIT'] = (NDIT, 'Number of dithered exposures')
    outhdul[0].header['EXPTIME'] = (EXPTIME, 'Exposure time')
    
    outhdul.writeto(file_name, overwrite=True)
    print('Saved readout without aberrations to ' + file_name)


def main():

    # initialize instrument here just to obtain filter lists: LM band
    '''
    cmd = sim.UserCommands(use_instrument='METIS', set_modes=['wcu_img_lm'])
    metis = sim.OpticalTrain(cmd)
    lm_filters_list = metis["filter_wheel"].filters.keys()
    '''
    lm_filters_list = ['Lp', 'short-L', 'L_spec', 'Mp', 'M_spec', \
        'Br_alpha', 'Br_alpha_ref', 'PAH_3.3', 'PAH_3.3_ref', 'CO_1-0_ice', \
            'CO_ref', 'H2O-ice'] # make a smaller selection of filters
    lm_fpmasks_list = ["pinhole_lm", "grid_lm"]

    '''
    ## BEGIN DEBUGGING
    DIT, NDIT = 30, 120
    wcu = metis['wcu_source']
    wcu.set_bb_aperture(0)
    metis.observe()
    readout_off = metis.readout(dit=DIT, ndit=NDIT)[0]
    wcu.set_bb_aperture(1.)
    metis.observe()
    readout_nom = metis.readout(dit=DIT, ndit=NDIT)[0]
    ## END DEBUGGING
    '''


    # same for N band
    cmd = sim.UserCommands(use_instrument='METIS', set_modes=['wcu_img_n'])
    metis = sim.OpticalTrain(cmd)
    n_filters_list = metis["filter_wheel"].filters.keys()
    n_fpmasks_list = ["pinhole_n"]

    # relative dither positions to place PSF at center and each corner of the FOV (y, x)
    # note these positions are scaled corresponding to the designed FOV
    rel_dither_position_array = [(0, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]
    designed_fov_img_lm = 10.5 # arcsec on one side
    designed_fov_img_n = 13.5 # arcsec on one side
    designed_pixel_scale_img_lm = 5.47 # mas/pix
    designed_pixel_scale_img_n = 6.79 # mas/pix

    # just one mask for now (Open)
    pp_mask = metis['pupil_masks'].meta['current_mask']

    # use nested for-loops, or generate permutations;
    # example of permutations for LM filters and masks:
    '''
    obs_counter = 1
    for filt, mask in itertools.product(filter_list_img_lm, wcu_fp2_masks_lm):
        obs_name = f"obs{obs_counter}"
        dict_config[obs_name] = {
            'mode': 'wcu_img_lm',
            'filter': filt,
            'fpmask': mask,
            'ndit': 10,
            'dit': 1.
        }
        obs_counter += 1
    '''

    #########################################################################################################################
    ## FOV SIMULATION: 
    
    # option 1: image PSF at each corner of the array (this doesn't produce good simmed data)

    '''
    # LM band    
    for fp_mask in lm_fpmasks_list:
        for obs_filter in lm_filters_list:
            print('fp_mask: ' + str(fp_mask) + ' obs_filter: ' + str(obs_filter))
            four_corner_psf(fp_mask, pp_mask, obs_filter, dither_position_array=rel_dither_position_array, fov=designed_fov_img_lm, ps=designed_pixel_scale_img_lm, obs_mode='wcu_img_lm', source='bb')

    # N band
    for fp_mask in n_fpmasks_list:
        for obs_filter in n_filters_list:
            four_corner_psf(fp_mask, pp_mask, obs_filter, dither_position_array=rel_dither_position_array, fov=designed_fov_img_n, ps=designed_pixel_scale_img_n, obs_mode='wcu_img_n', source='bb')
    '''

    # option 2: just image a grid

    '''
    # LM band
    lm_fpmasks_list = ['grid_lm']
    for fp_mask in lm_fpmasks_list:
        for obs_filter in lm_filters_list:
            image_fp_masks(fp_mask, obs_filter, pp_mask, obs_mode='wcu_img_lm', source='bb_source')
    '''

    #########################################################################################################################
    ## PLATE SCALE SIMULATION: image a grid of PSFs

    '''
    lm_fpmasks_list = ["grid_lm"] # only want a grid image
    #n_fpmasks_list = ["pinhole_n"]

    # LM band
    for fp_mask in lm_fpmasks_list:
        for obs_filter in lm_filters_list:
            image_fp_masks(fp_mask, obs_filter, pp_mask, obs_mode='wcu_img_lm', source='bb_source')

    # N band TBD; will require a grid mask
    '''

    #########################################################################################################################
    ## STRAY LIGHT: image a single PSF


    lm_fpmasks_list = ["pinhole_lm"]
    
    for fp_mask in lm_fpmasks_list:
        for obs_filter in lm_filters_list:
            image_fp_masks(fp_mask, obs_filter, pp_mask, obs_mode='wcu_img_lm', source='bb_source')



if __name__ == "__main__":
    main()