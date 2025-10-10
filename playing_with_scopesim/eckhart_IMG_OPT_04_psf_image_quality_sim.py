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

import time
import ipdb

import scopesim as sim
sim.bug_report()

# Edit this path if you have a custom install directory, otherwise comment it out. [For ReadTheDocs only]
sim.link_irdb("../../../")

# simulate observations with METIS (comment this out if packages already exist)
#sim.download_packages(["METIS", "ELT", "Armazones"])

def generate_psf_image_quality_data(fp_mask, obs_filter, dither_position_array, obs_mode):

    # set up instrument
    cmd = sim.UserCommands(use_instrument='METIS', set_modes=[obs_mode])
    metis = sim.OpticalTrain(cmd)

    metis.effects.pprint_all()
    wcu = metis['wcu_source']

    bb_temp = 1000 * u.K
    NDIT, EXPTIME = 1, 0.2

    print('Generating ' + str(fp_mask)) 
    wcu.set_fpmask(fp_mask)

    print('Closing WCU BB aperture first for background ...')
    # background
    wcu.set_bb_aperture(value = 0.0)
    metis.observe()
    outhdul_off = metis.readout(ndit = NDIT, exptime = EXPTIME)[0]
    background = outhdul_off[1].data

    dither_num_array = [0, 1] # 0: no dither, 1: dither
    wcu.set_bb_aperture(value = 1.0) # open BB source

    metis["filter_wheel"].change_filter(obs_filter)

    for dither_pos in dither_position_array:

        print('--------------------------------')
        print('Current WCU FP mask:', wcu.fpmask)
        print('Current Observing filter:', metis["filter_wheel"].current_filter)
        print('Current WCU PP mask:', metis['pupil_masks'].current_mask)

        # dither by shifting the FP mask
        # (note these shifts are absolute, not relative)
        wcu.set_fpmask(fp_mask, angle=0, shift=dither_pos)

        print('Opening WCU BB aperture...')

        metis.observe()
        outhdul = metis.readout(ndit = NDIT, exptime = EXPTIME)[0]
        #outhdul[1].data
        #outhdul.writeto(f"IMG_OPT_02_wcu_focal_plane_{mask}.fits", overwrite=True)

        ipdb.set_trace()

        # background-subtract
        bckgd_subted = outhdul[1].data - background

        ipdb.set_trace()

        # detector
        plt.clf()
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(bckgd_subted)
        plt.imshow(bckgd_subted, origin='lower', vmin=vmin, vmax=vmax)
        plt.title(f'Readout\nWCU FP mask: ' + str(fp_mask) + '\n' + 'Observing filter: ' + str(obs_filter) + '\n' + 'BB temp: ' + str(bb_temp))
        plt.tight_layout()
        plt.show()
        plt.close()

        # histogram
        plt.clf()
        plt.hist(bckgd_subted.ravel(), bins=200)
        plt.title('Bckgd-subtracted histogram; WCU FP mask: ' + str(fp_mask))
        plt.tight_layout()
        plt.show()
        plt.close()

        # save to FITS file
        file_name = 'IMG_OPT_04_wcu_focal_plane_' + str(fp_mask) + '.fits'
        outhdul.writeto(file_name, overwrite=True)
        print('Saved readout without aberrations to ' + file_name)

        # do a hackneyed aberration: blurring made to look like defocus 
        file_name = 'IMG_OPT_04_wcu_focal_plane_' + str(fp_mask) + '_blur.fits'
        outhdul[1].data = scipy.ndimage.gaussian_filter(outhdul[1].data, sigma=3)
        outhdul.writeto(file_name, overwrite=True)
        print('Saved readout with aberrations to ' + file_name)


def main():

    # initialize instrument here just to obtain filter lists: LM band
    cmd = sim.UserCommands(use_instrument='METIS', set_modes=['wcu_img_lm'])
    metis = sim.OpticalTrain(cmd)
    lm_filters_list = metis["filter_wheel"].filters.keys()
    lm_fpmasks_list = ["pinhole_lm", "grid_lm"]

    # same for N band
    cmd = sim.UserCommands(use_instrument='METIS', set_modes=['wcu_img_n'])
    metis = sim.OpticalTrain(cmd)
    n_filters_list = metis["filter_wheel"].filters.keys()
    n_fpmasks_list = ["pinhole_n"]

    # make as many dither positions as desired
    dither_position_array = [(0, 0), (1, 0), (0, 1), (1, 1)]

    # LM band
    for fp_mask in lm_fpmasks_list:
        for obs_filter in lm_filters_list:
            generate_psf_image_quality_data(fp_mask, obs_filter, dither_position_array=dither_position_array, obs_mode='wcu_img_lm')

    # N band
    for fp_mask in n_fpmasks_list:
        for obs_filter in n_filters_list:
            generate_psf_image_quality_data(fp_mask, obs_filter, dither_position_array=dither_position_array, obs_mode='wcu_img_n')


if __name__ == "__main__":
    main()