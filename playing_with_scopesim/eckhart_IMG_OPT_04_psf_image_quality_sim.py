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

# set up instrument for LM imaging
cmd = sim.UserCommands(use_instrument='METIS', set_modes=['wcu_img_lm'], properties={"!OBS.filter_name": "Lp"})

# alternative: Mp imaging and different filter

#cmd_2 = sim.UserCommands(use_instrument="METIS", set_modes=["img_lm"],
#                         properties={"!OBS.filter_name": "Mp", "!OBS.exptime": 100., "!DET.dit": 200})
#cmd = sim.UserCommands(use_instrument='METIS', set_modes=['img_lm'])
metis = sim.OpticalTrain(cmd)

#metis["chop_nod"].include = True # allow chopping
metis.effects.pprint_all()
wcu = metis['wcu_source']
#cfo = metis['cfo_source']
mask = "grid_lm"
bb_temp = 1000 * u.K
DIT, NDIT = 30, 120

fpmasks_list = ["open", "pinhole_lm", "pinhole_n", "grid_lm"]
lm_filters_list = ["Lp", "H2Oice", "shortL", "IB4.05", "PAH3.3", "PAH3.3ref", "Br-alpha", "Br-alpharef", "M'", "CO(1-0)/ice", 
                    "COref", "HCIL-short", "HCIL-long", "full_L", "full_M"]
#n_filters_list = ["N1", "N2", "PAH8.6", "PAH8.6_ref", "PAH11.25", "PAH11.25_ref", "[NeII]", "[NeII]_ref", "[SIV]", "[SIV]_ref", "N3", "full_N"]

# make as many dither positions as desired
dither_position_array = [(0, 0), (1, 0), (0, 1), (1, 1)]

#for mask in fpmasks_list:



def generate_psf_image_quality_data(fp_mask, obs_filter):

    # DO NOT REMOVE THIS 'for'! this option has to be present to avoid a bug that gets triggered in the 'else' case; don't know why
    for mode in ["close BB aperture first for background", "don't close BB aperture first for background"]:

        print('Generating ' + str(fp_mask)) 
        wcu.set_fpmask(fp_mask)

        if mode == "close BB aperture first for background":
            # DO NOT REMOVE THIS! this option has to be present to avoid a bug that gets triggered in the 'else' case; don't know why
            print('Closing WCU BB aperture first...')
            # background
            wcu.set_bb_aperture(value = 0.0)
            metis.observe()
            outhdul_off = metis.readout(ndit = 1, exptime = 0.2)[0]
            background = outhdul_off[1].data
        else:

            dither_num_array = [0, 1] # 0: no dither, 1: dither
            wcu.set_bb_aperture(value = 1.0) # open BB source

            metis["filter_wheel"].change_filter(obs_filter)

            for dither_pos in dither_position_array:

                # just make background a bunch of zeros for now to get around aforementioned bug
                outhdul_off = metis.readout(ndit = 1, exptime = 0.2)[0]
                background = np.zeros_like(outhdul_off[1].data)

                print('--------------------------------')
                print('Current WCU FP mask:', wcu.fpmask)
                print('Current Observing filter:', metis["filter_wheel"].current_filter)
                print('Current WCU PP mask:', metis['pupil_masks'].current_mask)

                # dither by shifting the FP mask
                # (note these shifts are absolute, not relative)
                wcu.set_fpmask(fp_mask, angle=0, shift=dither_pos)

                print('Opening WCU BB aperture...')

                metis.observe()
                outhdul = metis.readout(ndit = 1, exptime = 0.2)[0]
                #outhdul[1].data
                #outhdul.writeto(f"IMG_OPT_02_wcu_focal_plane_{mask}.fits", overwrite=True)

                ipdb.set_trace()

                # detector
                plt.clf()
                zscale = ZScaleInterval()
                vmin, vmax = zscale.get_limits(outhdul[1].data)
                plt.imshow(outhdul[1].data - background, origin='lower', vmin=vmin, vmax=vmax)
                plt.title(f'Readout\nWCU FP mask: ' + str(fp_mask) + '\n' + 'Observing filter: ' + str(obs_filter) + '\n' + 'BB temp: ' + str(bb_temp))
                plt.tight_layout()
                plt.show()
                plt.close()

                # histogram
                plt.clf()
                plt.hist(outhdul[1].data.ravel(), bins=200)
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
    for fp_mask in fpmasks_list:
        for obs_filter in lm_filters_list:
            generate_psf_image_quality_data(fp_mask, obs_filter)

if __name__ == "__main__":
    main()


'''
CONTINUE HERE: 
- how to treat N-band correctly, too? (note this will require chopping; see Overleaf)
- x how to dither?
- label all plots and write them out when they are seen to be well-behaved
'''