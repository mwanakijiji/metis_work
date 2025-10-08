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
cmd = sim.UserCommands(use_instrument='METIS', set_modes=['wcu_img_lm'])

mask = "grid_lm"

metis = sim.OpticalTrain(cmd)
metis.effects.pprint_all()
wcu = metis['wcu_source']
print('Prior wcu.fpmask:', wcu.fpmask)
bb_temp = 1000 * u.K
wcu.set_temperature(bb_temp=bb_temp)

mask = "pinhole_lm"

for mode in ["close BB aperture first for background", "don't close BB aperture first for background"]:

    print('Generating ' + str(mask)) 
    wcu.set_fpmask(mask)

    if mode == "close BB aperture first for background":
        # DO NOT REMOVE THIS! this option has to be present to avoid a bug that gets triggered in the 'else' case; don't know why
        print('Closing WCU BB aperture first...')
        # background
        wcu.set_bb_aperture(value = 0.0)
        metis.observe()
        outhdul_off = metis.readout(ndit = 1, exptime = 0.2)[0]
        background = outhdul_off[1].data
    else:
        # just make background a bunch of zeros
        outhdul_off = metis.readout(ndit = 1, exptime = 0.2)[0]
        background = np.zeros_like(outhdul_off[1].data)
    
        print('Opening WCU BB aperture...')
        wcu.set_bb_aperture(value = 1.0)
        metis.observe()
        outhdul = metis.readout(ndit = 1, exptime = 0.2)[0]
        #outhdul[1].data
        #outhdul.writeto(f"IMG_OPT_02_wcu_focal_plane_{mask}.fits", overwrite=True)
        plt.clf()
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(outhdul[1].data)
        plt.imshow(outhdul[1].data - background, origin='lower', vmin=vmin, vmax=vmax)
        plt.title(f'Readout; {mode}\nWCU FP mask: ' + str(mask) + '; BB temp: ' + str(bb_temp))
        plt.show()
