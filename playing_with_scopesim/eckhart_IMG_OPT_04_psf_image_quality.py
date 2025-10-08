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

ipdb.set_trace()

# set up instrument for LM imaging
#cmd = sim.UserCommands(use_instrument='METIS', set_modes=['wcu_img_lm'])

# alternative: Mp imaging and different filter

#cmd_2 = sim.UserCommands(use_instrument="METIS", set_modes=["img_lm"],
#                         properties={"!OBS.filter_name": "Mp", "!OBS.exptime": 100., "!DET.dit": 200})
cmd = sim.UserCommands(use_instrument='METIS', set_modes=['wcu_img_lm'])
metis = sim.OpticalTrain(cmd)
metis.effects.pprint_all()
wcu = metis['wcu_source']
mask = "grid_lm"
bb_temp = 1000 * u.K

## START GOOD CODE
fpmasks_list = ["open", "pinhole_lm", "pinhole_n", "grid_lm"]
for mask in fpmasks_list:
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
            # just make background a bunch of zeros for now to get around aforementioned bug
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
            plt.title(f'Readout\nWCU FP mask: ' + str(mask) + '\nBB temp: ' + str(bb_temp))
            plt.tight_layout()
            plt.show()

## END GOOD CODE
ipdb.set_trace()

# ## 1. Configure WCU FP1 to LM pinhole mask.
metis = sim.OpticalTrain(cmd)
metis.effects.pprint_all()
wcu = metis['wcu_source']
print('Prior wcu.fpmask:', wcu.fpmask)

# set the WCU BB source to 1000 K.
print('Setting wcu.bb_temp to 1000 K...')
wcu.set_temperature(bb_temp=1000*u.K)
print('wcu.bb_temp:', wcu.bb_temp)
# wait for BB source to reach temperature.
print('Waiting for BB source to reach temperature (placeholder in lieu of a thermal model)...')
# placeholder in lieu of a thermal model
time.sleep(0.5)
DIT, NDIT = 30, 120

fpmasks_list = ["open", "pinhole_lm", "pinhole_n", "grid_lm"]
wcu = metis['wcu_source']

for mask in fpmasks_list:

    print('--------------------------------')
    
    print('Prior wcu.fpmask:', wcu.fpmask)

    # ## 2. Set the WCU Flux Controlling Mask to "CLOSED" to take a background
    '''
    print('Closing wcu.bb_aperture...')
    wcu.set_bb_aperture(value = 0.)
    print('wcu.bb_aperture:', wcu.bb_aperture)

    print('Generating ' + str(mask)) 
    wcu.set_fpmask(mask)

    # see current observing params
    print("\Current observing parameters:")
    for key, value in cmd['OBS'].items():
        print(f"  {key}: {value}")

    # compile the observation
    print('Compiling the observation...')
    #src = sim.source.source_templates.star()
    #metis.observe(src)
    metis.observe()

    # take background readout
    # do readout with observation params
    print('Taking readout...')
    # Oliver Cz. recommends just using ndit and dit (not exptime)

    outhdul = metis.readout(ndit = NDIT, dit = DIT)[0]
    outhdul.info()

    bckgrnd = outhdul[1].data
    '''

    # ## 6. Set the WCU Flux Controlling Mask to "OPEN".
    print('Setting the wcu bb aperture to OPEN')
    #wcu.set_bb_aperture(value = 1.)

    print('Taking readout with FP mask ' + str(mask) + '...')
    metis.observe()
    # get the readout
    outhdul = metis.readout(ndit = NDIT, dit = DIT)[0]

    sci = outhdul[1].data

    #metis_n = sim.OpticalTrain(cmd)
    print('Current WCU FP mask:', wcu.fpmask)
    print('Current WCU PP mask:', metis['pupil_masks'].current_mask)

    '''
    # plot
    plt.clf()
    plt.title('Bckgd-subtracted readout; WCU FP mask: ' + str(mask))
    plt.imshow(outhdul[1].data, origin='lower')
    plt.show()

    plt.clf()
    plt.title('Bckgd-subtracted histogram; WCU FP mask: ' + str(mask))
    plt.hist(outhdul[1].data.ravel(), bins=200)
    plt.show()
    '''
    

    ipdb.set_trace()

    plt.clf()

    # bckgrnd_subtracted = sci - bckgrnd
    
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(outhdul[1].data)
    plt.title('Bckgd-subtracted readout; WCU FP mask: ' + str(mask))
    #plt.imshow(outhdul[1].data, origin='lower'
    plt.imshow(sci, origin='lower', vmin=vmin, vmax=vmax)
    plt.show()

    plt.clf()
    
    plt.hist(sci.ravel(), bins=200)
    plt.title('Bckgd-subtracted histogram; WCU FP mask: ' + str(mask))
    plt.show()

    ipdb.set_trace()

    # save to FITS file
    file_name = 'IMG_OPT_04_wcu_focal_plane_' + str(mask) + '.fits'
    outhdul.writeto(file_name, overwrite=True)
    print('Saved readout without aberrations to ' + file_name)

    # do a hackneyed aberration: blurring made to look like defocus 
    file_name = 'IMG_OPT_04_wcu_focal_plane_' + str(mask) + '_blur.fits'
    outhdul[1].data = scipy.ndimage.gaussian_filter(outhdul[1].data, sigma=3)
    outhdul.writeto(file_name, overwrite=True)
    print('Saved readout with aberrations to ' + file_name)

'''
CONTINUE HERE: 
- how to treat N-band correctly, too? (note this will require chopping; see Overleaf)
- how to dither?
- label all plots and write them out when they are seen to be well-behaved
'''

'''
# do the same for N band
cmd = sim.UserCommands(use_instrument='METIS', set_modes=['wcu_img_lm'])
wcu = metis['wcu_source']

fpmasks_list = ["open", "pinhole_lm", "pinhole_n", "grid_lm"]
for mask in fpmasks_list:
    print('Generating ' + str(mask)) 
    wcu.set_fpmask(mask)
    wcu.set_temperature(bb_temp=1000*u.K)
    wcu.set_bb_aperture(value = 0.4)
    metis.observe()
    outhdul = metis.readout(ndit = 1, exptime = 0.2)[0]
    #outhdul[1].data
    outhdul.writeto(f"IMG_OPT_02_wcu_focal_plane_{mask}_LM.fits", overwrite=True)
'''

'''
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift

ln_img_plate_scale = 5.5e-3 # 5.5 masec/pix

def add_defocus_to_psf(psf_data, defocus_waves=0.5, wavelength=3.3e-6, pupil_diameter=39.0):
    """
    Add defocus to an existing PSF by applying a defocus phase in the pupil plane.
    
    Parameters:
    -----------
    psf_data : 2D array
        Input PSF data
    defocus_waves : float
        Defocus in waves (RMS)
    wavelength : float
        Wavelength in meters
    pupil_diameter : float
        Pupil diameter in meters
    
    Returns:
    --------
    2D array with defocused PSF
    """
    # Get PSF dimensions
    ny, nx = psf_data.shape
    
    # Create coordinate grids
    x = np.linspace(-nx//2, nx//2-1, nx)
    y = np.linspace(-ny//2, ny//2-1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Convert to angular coordinates (assuming PSF is centered)
    theta_x = X * ln_img_plate_scale
    theta_y = Y * ln_img_plate_scale

    print('---')
    print(ln_img_plate_scale)
    print(theta_x)

    plt.imshow(theta_x)
    plt.colorbar()
    plt.show()
    
    # phase screen (defocus: quadratic phase error)
    defocus_phase = 2 * np.pi * defocus_waves * (theta_x**2 + theta_y**2) / (wavelength / pupil_diameter)

    print('defocus_phase')
    print(defocus_phase)
    plt.imshow(defocus_phase)
    plt.title('phase screen')
    plt.colorbar()
    plt.show()

    print('min, defocus_phase')
    print(np.min(defocus_phase))
    
    # Apply defocus by multiplying PSF with phase factor
    defocused_psf = psf_data * np.exp(1j * defocus_phase)
    
    # Take absolute value to get intensity
    return np.sqrt(np.abs(defocused_psf)**2)

    

# Load your existing PSF
psf_file = '/playing_with_scopesim/METIS_LMS_olivier_notebooks/inst_pkgs/METIS/PSF_LM_9mag_06seeing.fits'
hdul = fits.open(psf_file)
psf_data = hdul[1].data
hdul.close()

# Add defocus
psf_data = np.roll(psf_data, 300)
defocused_psf = add_defocus_to_psf(psf_data, defocus_waves=10)

# Plot comparison
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
ax1.imshow(psf_data, origin='lower', cmap='viridis', norm='log')
ax1.set_title('Original PSF')
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')

ax2.imshow(defocused_psf, origin='lower', cmap='viridis', norm='log')
ax2.set_title('PSF with 0.5λ defocus')
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')

ax3.imshow(defocused_psf-psf_data, origin='lower', cmap='viridis', norm='log')
ax3.set_title('Residuals')
ax3.set_xlabel('X (pixels)')
ax3.set_ylabel('Y (pixels)')

plt.tight_layout()
plt.show()

print(psf_data)
print('----')
print(defocused_psf)
print('----')
print(defocused_psf-psf_data)
'''

# %%
psf_file = '/playing_with_scopesim/METIS_LMS_olivier_notebooks/inst_pkgs/METIS/PSF_LM_9mag_06seeing.fits'
hdul = fits.open(psf_file)

# %%
hdul[1].data


# %% [markdown]
# ## Notes RvB:
# 
# ##    take deep PSF measurements for all pupil stops in the Imager Subsystem (IMG)-PP1. APP would be addressed in HCI test.

# %%
# IMG-LM and IMG-N pupil wheels are not available in ScopeSim yet

# %%
# will also need to apply (x,y) offsets to all PSFs


