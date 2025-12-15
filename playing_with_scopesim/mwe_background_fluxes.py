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

def generate_psf_image_quality_data(fp_mask, pp_mask, obs_filter, obs_mode):

    # set up instrument
    cmd = None # reset
    cmd = sim.UserCommands(use_instrument='METIS', set_modes=[obs_mode])
    metis = sim.OpticalTrain(cmd)

    metis.effects.pprint_all()
    wcu = metis['wcu_source']

    bb_temp = 1000 * u.K
    NDIT, EXPTIME = 1, 0.01

    print('Setting FP mask: ' + str(fp_mask))
    wcu.set_fpmask(fp_mask)

    print('Setting PP mask: ' + str(pp_mask))
    metis['pupil_masks'].change_mask(pp_mask)
    #metis['psf'].update(pupil_mask=pp_mask) # needed due to shortcoming in ScopeSim: https://irdb.readthedocs.io/en/latest/METIS/docs/example_notebooks/demos/demo_metis_wcu_psfs.html
    #wcu.set_ppmask(pp_mask)

    print('Setting observing filter: ' + str(obs_filter))
    metis["filter_wheel"].change_filter(obs_filter)

    print('Closing WCU BB aperture first to get a background ...')
    # background
    wcu.set_bb_aperture(value = 0.0)
    metis.observe()
    outhdul_off = metis.readout(ndit = NDIT, exptime = EXPTIME)[0]
    background = outhdul_off[1].data

    print('Re-opening WCU BB aperture to get a PSF ...')
    wcu.set_bb_aperture(value = 1.0) # open BB source

    #metis["filter_wheel"].change_filter(obs_filter)

    print('--------------------------------')
    print('Current Observing filter:', obs_filter)
    print('Current WCU FP mask:', wcu.fpmask)
    print('Current WCU PP mask:', pp_mask)
    print('Opening WCU BB aperture...')

    metis.observe()
    # Get perfect PSF - no detector noise
    #hdul_perfect = metis.image_planes[0].hdu
    outhdul = metis.readout(ndit = NDIT, exptime = EXPTIME)[0]

    # background-subtract
    bckgd_subted = outhdul[1].data - background
    outhdul[1].data = bckgd_subted # reassign; note that this step will have to be done later outside the ScopeSim context
    #hdu_bckgd_subted = fits.ImageHDU(data=bckgd_subted)
    #outhdul.append(hdu_bckgd_subted) # outhdul[2].data is the background-subtracted image

    
    plt.clf()
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(bckgd_subted)
    plt.imshow(outhdul[1].data, origin='lower', vmin=vmin, vmax=vmax)
    plt.title(f'Raw readout\nWCU FP mask: ' + str(fp_mask) + '\n' + 'WCU PP mask: ' + str(pp_mask) + '\n' + 'Observing filter: ' + str(obs_filter) + '\n' + 'BB temp: ' + str(bb_temp))
    plt.colorbar()
    plt.tight_layout()
    file_name_plot_raw_readout = 'IMG_OPT_04_wcu_focal_mask_' + str(fp_mask) + '_pupil_mask_' + str(pp_mask) + '_filter_' + str(obs_filter) + '_raw_readout.png'
    plt.savefig(file_name_plot_raw_readout)
    #plt.show()
    plt.close()
    print('Saved PNG of raw readout to ' + file_name_plot_raw_readout)

    plt.clf()
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(bckgd_subted)
    plt.imshow(background, origin='lower', vmin=vmin, vmax=vmax)
    plt.title(f'Background\nWCU FP mask: ' + str(fp_mask) + '\n' + 'WCU PP mask: ' + str(pp_mask) + '\n' + 'Observing filter: ' + str(obs_filter) + '\n' + 'BB temp: ' + str(bb_temp))
    plt.colorbar()
    plt.tight_layout()
    file_name_plot_background = 'IMG_OPT_04_wcu_focal_mask_' + str(fp_mask) + '_pupil_mask_' + str(pp_mask) + '_filter_' + str(obs_filter) + '_background.png'
    plt.savefig(file_name_plot_background)
    #plt.show()
    plt.close()
    print('Saved PNG of background to ' + file_name_plot_background)

    # detector
    plt.clf()
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(bckgd_subted)
    plt.imshow(bckgd_subted, origin='lower', vmin=vmin, vmax=vmax)
    plt.title(f'Bckgd-subtracted readout\nWCU FP mask: ' + str(fp_mask) + '\n' + 'WCU PP mask: ' + str(pp_mask) + '\n' + 'Observing filter: ' + str(obs_filter) + '\n' + 'BB temp: ' + str(bb_temp))
    plt.colorbar()
    plt.tight_layout()
    file_name_plot_bckgd_subtracted_readout = 'IMG_OPT_04_wcu_focal_mask_' + str(fp_mask) + '_pupil_mask_' + str(pp_mask) + '_filter_' + str(obs_filter) + '_bckgd_subtracted_readout.png'
    plt.savefig(file_name_plot_bckgd_subtracted_readout)
    #plt.show()
    plt.close()
    print('Saved PNG of bckgd-subtracted readout to ' + file_name_plot_bckgd_subtracted_readout)

    # histogram
    plt.clf()
    plt.hist(bckgd_subted.ravel(), bins=200)
    plt.title('Bckgd-subtracted histogram; WCU FP mask: ' + str(fp_mask) + '\n' + 'WCU PP mask: ' + str(pp_mask) + '\n' + 'Observing filter: ' + str(obs_filter) + '\n' + 'BB temp: ' + str(bb_temp))
    plt.tight_layout()
    file_name_plot_bckgd_subtracted_histogram = 'IMG_OPT_04_wcu_focal_mask_' + str(fp_mask) + '_pupil_mask_' + str(pp_mask) + '_filter_' + str(obs_filter) + '_bckgd_subtracted_histogram.png'
    plt.savefig(file_name_plot_bckgd_subtracted_histogram) 
    #plt.show()
    plt.close()
    print('Saved PNG of bckgd-subtracted histogram to ' + file_name_plot_bckgd_subtracted_histogram)


    # save to FITS file, with filter and other info in the header
    file_name = 'IMG_OPT_04_wcu_focal_mask_' + str(fp_mask) + '_pupil_mask_' + str(pp_mask) + '_filter_' + str(obs_filter) + '.fits'
    outhdul[0].header['FILTER'] = (obs_filter, 'Observing filter')
    outhdul[0].header['WCU_FP'] = (fp_mask, 'WCU focal plane mask')
    outhdul[0].header['WCU_PP'] = (pp_mask, 'WCU pupil plane mask')
    outhdul[0].header['BB_TEMP'] = (bb_temp.value, 'BB temperature')
    outhdul[0].header['NDIT'] = (NDIT, 'Number of dithered exposures')
    outhdul[0].header['EXPTIME'] = (EXPTIME, 'Exposure time')
    
    outhdul.writeto(file_name, overwrite=True)
    print('Saved readout without aberrations to ' + file_name)


def main():

    # initialize instrument here just to obtain filter lists: LM band
    cmd = sim.UserCommands(use_instrument='METIS', set_modes=['wcu_img_lm'])
    metis = sim.OpticalTrain(cmd)
    lm_filters_list = metis["filter_wheel"].filters.keys() # filters
    lm_fpmasks_list = ["pinhole_lm", "grid_lm"] # FP masks

    # same for N band
    cmd = sim.UserCommands(use_instrument='METIS', set_modes=['wcu_img_n'])
    metis = sim.OpticalTrain(cmd)
    n_filters_list = metis["filter_wheel"].filters.keys() # filters
    n_fpmasks_list = ["pinhole_n"] # FP masks

    # just one mask for now (Open)
    pp_mask = metis['pupil_masks'].meta['current_mask'] # PP mask

    # LM band
    for fp_mask in lm_fpmasks_list:
        for obs_filter in lm_filters_list:
            generate_psf_image_quality_data(fp_mask, pp_mask, obs_filter, obs_mode='wcu_img_lm')

    # N band
    for fp_mask in n_fpmasks_list:
        for obs_filter in n_filters_list:
            generate_psf_image_quality_data(fp_mask, pp_mask, obs_filter, obs_mode='wcu_img_n')


if __name__ == "__main__":
    main()