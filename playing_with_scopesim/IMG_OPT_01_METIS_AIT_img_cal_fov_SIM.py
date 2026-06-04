#!/usr/bin/env python
# coding: utf-8

# # DRAFT IMG-OPT-01

# Maked simulated data for the IMG-OPT-1 FOV, distortion, and ghost measurements

# Reqs.:
# - Ref. Overleaf doc IMG_OPT_01_Test_Description_Field_of_View_and_Ghost_Measurement.pdf
# 
# 1. METIS-1095: The FoV shall be 10.0 +1.0/-0.0 arcsec for the LM-arm and 13.47 +0.50/-0.99 arcsec
# for the N-arm of the IMG.

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS

import scipy

from matplotlib import pyplot as plt
from matplotlib import colors

import time
import ipdb
import itertools

import scopesim as sim
sim.bug_report()

# Edit this path if you have a custom install directory, otherwise comment it out. [For ReadTheDocs only]
sim.link_irdb("../../../")

# simulate observations with METIS (comment this out if packages already exist)
# sim.download_packages(["METIS", "ELT", "Armazones"])

# lists of imaging filters
filter_list_img_lm = [
    'Lp',
    'H2Oice',
    'shortL',
    'IB4.05',
    'PAH3.3',
    'PAH3.3ref',
    'Br-alpha',
    'Br-alpharef',
    'Mp',
    'CO(1-0)/ice',
    'COref',
    'HCIL-short',
    'HCIL-long',
    'full_L',
    'full_M'
]

filter_list_img_n = [
    'N1', 
    'N2',
    'PAH8.6',
    'PAH8.6_ref',
    'PAH11.25',
    'PAH11.25_ref',
    '[NeII]',
    '[NeII]_ref',
    '[SIV]',
    '[SIV]_ref',
    'N3',
    'full_N'
]

# WCU FP2 focal planemask wheel
wcu_fp2_masks_lm = ["pinhole_lm", "grid_lm", "grid_lms"]
wcu_fp2_masks_n = ["pinhole_n"]



dict_config = {}

# Generate permutations for LM filters and masks
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

# Generate permutations for N filters and masks
for filt, mask in itertools.product(filter_list_img_n, wcu_fp2_masks_n):
    obs_name = f"obs{obs_counter}"
    dict_config[obs_name] = {
        'mode': 'wcu_img_lm',
        'filter': filt,
        'fpmask': mask,
        'ndit': 10,
        'dit': 1.
    }
    obs_counter += 1

## ## QUESTION: IS wcu_img_lm THE RIGHT MODE FOR N FILTERS?
## ## QUESTION: DO WE REALLY NEED THIS MANY PARAMETERS?

# dictionary of all observing configurations
print('Number of observing configurations: ' + str(len(dict_config)))
for obs_name, config in dict_config.items():
    print(f"{obs_name}: {config}")
ipdb.set_trace()
# fpmasks_list = ["open", "pinhole_lm", "pinhole_n", "grid_lm"]


# TODO: ADD other config.
#cmd_2 = sim.UserCommands(use_instrument="METIS", set_modes=["img_lm"],
#                    properties={"!OBS.filter_name": "Mp", "!OBS.exptime": 100., "!DET.dit": 200})

for config_params in dict_config:
    # take exposures for 
    # each filter in filter_list = ["Mp", "Lp"]
    # each fpmasks_list = ["open", "pinhole_lm", "pinhole_n", "grid_lm"]

    print('--------------------------------')
    print('Running config: ' + str(dict_config[config_params]))
    print('--------------------------------')

    mode = dict_config[config_params]['mode']
    ndit = dict_config[config_params]['ndit']
    dit = dict_config[config_params]['dit']
    filter = dict_config[config_params]['filter']

    # set up instrument for imaging
    cmd = sim.UserCommands(use_instrument='METIS', set_modes=[mode], properties={"!OBS.filter_name": filter})

    #########################################################
    # Configure WCU to LM pinhole mask.
    metis = sim.OpticalTrain(cmd)
    metis.effects.pprint_all()

    wcu = metis['wcu_source']

    print('previous wcu fpmask:', wcu.fpmask)
    wcu.set_fpmask('grid_lm')  # change to LM pinhole mask
    print('new wcu fpmask:', wcu.fpmask)
    print(wcu.fpmask)

    #########################################################
    # Set the WCU Flux Controlling Mask to "CLOSED".
    print('setting the wcu bb aperture to 0.')
    wcu.set_bb_aperture(value = 0.)
    print('wcu.bb_aperture:', wcu.bb_aperture)

    #########################################################
    # Set the WCU BB source to 1000 K.
    print('setting the wcu bb temperature to 1000 K.')
    wcu.set_temperature(bb_temp=1000*u.K)
    print('wcu.bb_temp:', wcu.bb_temp)

    #########################################################
    # Wait for BB source to reach temperature.

    # placeholder in lieu of a thermal model
    print('waiting for the wcu bb source to reach temperature.')
    time.sleep(0.5)

    #########################################################
    # While BB reaches temperature, take background exposure
    print('Taking background exposure.')
    # see current observing params
    print("\nAll OBS parameters:")
    for key, value in cmd['OBS'].items():
        print(f"  {key}: {value}")

    # compile the observation
    print('Compiling the observation.')
    metis.observe()

    # do readout with observation params
    print('Getting readout.')

    # Oliver Cz. recommends just using ndit and dit (not exptime)
    outhdul = metis.readout(ndit = ndit, dit = dit)[0]
    outhdul.info()

    # display
    file_name1 = 'test1.png'
    plt.clf()
    plt.imshow(outhdul[1].data, origin='lower')
    plt.title(config_params)
    plt.show()
    #plt.savefig('/podman-share/' + file_name1)
    #print('Saved ' + file_name1)

    file_name2 = 'test2.png'
    plt.clf()
    plt.hist(outhdul[1].data.ravel(), bins=200)
    plt.title('Counts in science exposure\n' + config_params)
    plt.show()
    #plt.savefig('/podman-share/' + file_name2)
    #print('Saved ' + file_name2)

    #########################################################
    # Set the WCU Flux Controlling Mask to "OPEN".
    print('Setting the wcu bb aperture to OPEN')
    wcu.set_bb_aperture(value = 1.)

    #########################################################
    # Take science exposure with same params as background
    print('Taking science exposure with same params as background.')

    # recompile
    metis.observe()
    # get the readout
    outhdul = metis.readout(ndit = ndit, dit = dit)[0]
    print('Science exposure readout.')

    plt.clf()
    plt.hist(outhdul[1].data.ravel(), bins=200)
    plt.title('Counts in science exposure\n' + config_params)
    plt.show()

    # do a hackneyed aberration: blurring made to look like defocus 
    #outhdul[1].data = scipy.ndimage.gaussian_filter(outhdul[1].data, sigma=3)
    #outhdul.writeto(f"IMG_OPT_02_wcu_focal_plane_{mask}_LM_blur.fits", overwrite=True)