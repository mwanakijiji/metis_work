# Does some simple analysis of simulated images written out by the sim notebook.
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS

import scipy
from scipy.spatial import distance_matrix
from scipy.special import j0, j1
from itertools import combinations
import glob
import os
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.ndimage import zoom, shift, center_of_mass

from matplotlib import pyplot as plt
from matplotlib import colors
from astropy.visualization import ZScaleInterval

from photutils.centroids import centroid_sources, centroid_com, centroid_2dg

import pandas as pd

import ipdb

import scopesim as sim
from skimage import measure
from scipy.special import j1


def intensity_annular_aperture(r_rad_array, wavel, D_aperture, D_obscuration):
    '''
    Calculate the intensity through an aperture with a central obscuration
    Ref. 'E-REP-MPIA-1203 0-1 xx-10-2024', Sec. 4.4

    INPUTS:
    - r_rad_array: 2D array of radial distances from the center (units radians)
    - wavel: wavelength (units meters)
    - D_aperture: aperture diameter (units meters)
    - D_obscuration: obscuration diameter (units meters)

    OUTPUTS:
    - I_r_array: 2D array of intensity on the detector
    '''


    nu_ = np.pi * r_rad_array * D_aperture / wavel # unitless

    eps_ = D_obscuration / D_aperture # unitless
    
    # see Eqn. 43 in 'E-REP-MPIA-1203 0-1 xx-10-2024'
    I_r = (1/(1-eps_**2)**2) * ( (2*j1(nu_)/nu_) - eps_**2 * (2*j1(nu_*eps_)/(nu_*eps_)) ) ** 2

    return I_r


def gaussian_2d(xy_mesh, amplitude, xo, yo, sigma_x_pix, sigma_y_pix, theta):
    x, y = xy_mesh
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2 * sigma_x_pix**2) + (np.sin(theta)**2) / (2 * sigma_y_pix**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x_pix**2) + (np.sin(2 * theta)) / (4 * sigma_y_pix**2)
    c = (np.sin(theta)**2) / (2 * sigma_x_pix**2) + (np.cos(theta)**2) / (2 * sigma_y_pix**2)
    g = amplitude * np.exp(-(a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2)))
    return g.ravel()


def fit_empirical_fwhm(frame, plot_string):
    '''
    Take the data as-is, find where the intensity is 50% of the peak intensity, and then calculate the FWHM in x and y.

    INPUTS:
    frame: 2D array of the frame
    plot_string: string to add to the plot file name
    '''

    # find the peak intensity
    #ipdb.set_trace()
    peak_intensity = np.max(frame)
    # find where the intensity is 50% of the peak intensity
    # Find the positions of the maximum value as the initial guess for the center
    y_peak, x_peak = np.unravel_index(np.argmax(frame), frame.shape)
    # Create x and y coordinate arrays
    y, x = np.indices(frame.shape)
    # Define a threshold for 50% of the peak
    half_max = 0.5 * peak_intensity

    # fit an oval to the region above half-max
    mask_half = frame >= half_max
    labeled = measure.label(mask_half)
    props = measure.regionprops(labeled)
    if len(props) > 0:
        # Select the largest region by area
        prop_biggest = [max(props, key=lambda p: p.area)]
    #ipdb.set_trace()
    if len(props) == 0:
        prop_biggest_dims = np.nan, np.nan

    # use bounding box to get the dims in x and y (instead of just major and minor axis lengths)
    min_row, min_col, max_row, max_col = prop_biggest[0].bbox
    height_y = max_row - min_row   # axis-aligned y length
    width_x  = max_col - min_col   # axis-aligned x length

    # Plot the frame
    plt.figure()
    plt.imshow(frame, origin='lower', cmap='gray')
    # Plot the bounding box if prop_biggest was found
    if len(props) > 0:
        rect = plt.Rectangle(
            (min_col, min_row), width_x, height_y,
            edgecolor='red', facecolor='none', linewidth=2, linestyle='--'
        )
        plt.gca().add_patch(rect)
    plt.title(f'Frame with Bounding Box at 50% Peak\nFWHM in x (pix): {width_x:.2f}, FWHM in y (pix): {height_y:.2f}')
    # save the plot to file
    plot_filename = 'empirical_fwhm_' + plot_string + '.png'
    #ipdb.set_trace()
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f'Figure saved as {plot_filename}')
    #plt.show()

    plt.close()

    return height_y, width_x


def fit_gaussian(frame, center_guess):
    """
    Fit a 2D Gaussian function to a given frame.

    Parameters:
    frame (ndarray): 2D array representing the frame.
    center_guess (list): List containing the initial guess for the center coordinates.

    Returns:
    fitted_array (ndarray): 2D array representing the fitted Gaussian function.
    fwhm_x_pix (float): Full Width at Half Maximum (FWHM) in the x-direction.
    fwhm_y_pix (float): Full Width at Half Maximum (FWHM) in the y-direction.
    sigma_x_pix (float): Standard deviation in the x-direction.
    sigma_y_pix (float): Standard deviation in the y-direction.
    amplitude_counts (float): Amplitude of the Gaussian function in counts.
    """
    y, x = np.indices(frame.shape)
    xy_mesh = (x, y)
    p0 = [np.max(frame), center_guess[0], center_guess[1], 1, 1, 0]
    popt, pcov = curve_fit(gaussian_2d, xy_mesh, frame.ravel(), p0=p0)
    fitted_array = gaussian_2d(xy_mesh, *popt).reshape(frame.shape)
    fwhm_x_pix = 2 * np.sqrt(2 * np.log(2)) * np.abs(popt[3])
    fwhm_y_pix = 2 * np.sqrt(2 * np.log(2)) * np.abs(popt[4])
    amplitude_counts = popt[0]
    x_center_pix = popt[1]
    y_center_pix = popt[2]
    sigma_x_pix = popt[3]
    sigma_y_pix = popt[4]
    angle_theta_deg = popt[5]
    
    return fitted_array, x_center_pix, y_center_pix, fwhm_x_pix, fwhm_y_pix, sigma_x_pix, sigma_y_pix, angle_theta_deg, amplitude_counts


def fyi_plot_centroiding(array_to_plot, coords_to_plot, zscale=False):
    # INSERT_YOUR_CODE

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(array_to_plot)
    plt.clf()
    plt.imshow(array_to_plot, origin='lower', vmin=vmin, vmax=vmax, cmap='gray')
    plt.scatter(coords_to_plot[:, 1], coords_to_plot[:, 0], color='red', s=10)
    plt.show()
    plt.close()


def fit_gaussian_fwhm(cookie_cut_out_sci, coords_guess, plot_string, fac_oversamp):
    '''
    Find FWHM of Gaussian-best-fit to empirical; all fit parameters are free

    INPUTS:
    cookie_cut_out_sci: 2D array of the science frame
    coords_guess: 2D array of the centroided coordinates (one coordinate pair)
    plot_string: string to add to the plot file name
    fac_oversamp: oversampling factor

    OUTPUTS:
    fwhm_y_pix: FWHM in y-direction
    fwhm_x_pix: FWHM in x-direction
    '''

    ## ## TO DO: ARE THE INDEXES RIGHT HERE?
    cookie_cut_out_best_fit, x_center_pix_oversamp_cutout, y_center_pix_oversamp_cutout, fwhm_x_pix, fwhm_y_pix, sigma_x_pix, sigma_y_pix, angle_theta_deg, amplitude_counts = fit_gaussian(cookie_cut_out_sci, \
        center_guess = coords_guess)
    residuals = cookie_cut_out_sci - cookie_cut_out_best_fit

    # plot four subplots: 2D science, 2D best-fit, 2D residuals, and 1D overplotting of a cross-section of the science and best-fit
    plt.clf()
    # Determine vmin and vmax for consistent color scaling across all 2D plots
    vmin = min(np.nanmin(cookie_cut_out_sci), np.nanmin(cookie_cut_out_best_fit), np.nanmin(residuals))
    vmax = max(np.nanmax(cookie_cut_out_sci), np.nanmax(cookie_cut_out_best_fit), np.nanmax(residuals))
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # 2D Science image
    im0 = axs[0, 0].imshow(cookie_cut_out_sci, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title('Science')
    plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)
    # 2D Best-fit image
    im1 = axs[0, 1].imshow(cookie_cut_out_best_fit, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title('Best-fit')
    plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)
    # 2D Residuals image
    im2 = axs[1, 0].imshow(residuals, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title('Residuals')
    plt.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)
    # Plot a cross-section through the maximum of the PSF (along the row/col with the peak)
    max_index = np.unravel_index(np.argmax(cookie_cut_out_sci), cookie_cut_out_sci.shape)
    # Extract the row and column through the peak
    sci_row = cookie_cut_out_sci[max_index[0], :]
    best_fit_row = cookie_cut_out_best_fit[max_index[0], :]
    axs[1, 1].plot(sci_row, label='Empirical')
    axs[1, 1].plot(best_fit_row, label='Best-fit')
    # Annotate plot with FWHM in x and y
    fwhm_text = f'FWHM x = {fwhm_x_pix:.2f} pix\nFWHM y = {fwhm_y_pix:.2f} pix'
    axs[1, 1].text(
        0.95, 0.05, fwhm_text,
        transform=axs[1, 1].transAxes,
        fontsize=10, color='black',
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3')
    )
    axs[1, 1].legend()
    axs[1, 1].set_title('1D cross-section, empirical vs best-fit')
    plt.suptitle(f'PSF, oversampling factor: {fac_oversamp:.2f} \n Found coord (y,x): ({y_center_pix_oversamp_cutout:.2f}, {x_center_pix_oversamp_cutout:.2f}) \n Found FWHM x: {fwhm_x_pix:.2f} pix, FWHM y: {fwhm_y_pix:.2f} pix,\nFound amplitude: {amplitude_counts:.2f} counts')
    plt.tight_layout()
    #plt.show()
    # Save the plot to file with num_coord as a 2-digit zero-padded string
    plot_filename = f'psf_gaussian_best_fit_'+plot_string+'.png'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f'Figure saved as {plot_filename}')
    plt.close()
    #ipdb.set_trace()

    return x_center_pix_oversamp_cutout, y_center_pix_oversamp_cutout, fwhm_x_pix, fwhm_y_pix, amplitude_counts


def fit_simmed_psfs(cookie_cut_out_sci, plot_string, fp_mask, x_center_final_oversamp, y_center_final_oversamp, fac_oversamp):
    '''
    Find FWHM of a PSF using a perfect PSF from ScopeSim
    
    INPUTS:
    cookie_cut_out_sci: 2D array of the science frame
    plot_string: string to add to the plot file name
    x_center_final_oversamp: final x-center of the PSF (i.e., no more centroiding will be done here)
    y_center_final_oversamp: final y-center of the PSF
    fac_oversamp: oversampling factor

    OUTPUTS:
    psf_perfect_cutout_best_fit: cutout around the best-fit simulated PSF
    '''

    # set up instrument
    ## ## TO DO: MAKE THIS MORE GENERAL AND FLEXIBLE, FOR MULT OBSERVING MODES
    cmd = sim.UserCommands(use_instrument='METIS', set_modes=['wcu_img_lm'])
    metis = sim.OpticalTrain(cmd)

    wcu = metis['wcu_source']

    # set the filter
    obs_filter = 'Br_alpha'  ## ## TO DO: MAKE THIS MORE GENERAL AND FLEXIBLE, FOR MULT OBSERVING MODES
    metis["filter_wheel"].change_filter(obs_filter)

    wcu.set_fpmask(fp_mask)

    pp_mask = metis['pupil_masks'].meta['current_mask'] # just one mask for now (Open)

    metis.effects.pprint_all()

    bb_temp = 1000 * u.K
    NDIT, EXPTIME = 1, 0.2


    print('--------------------------------')
    print('Current Observing filter:', obs_filter)
    print('Current WCU FP mask:', wcu.fpmask)
    print('Current WCU PP mask:', pp_mask)
    #ipdb.set_trace()
    # background
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
    outhdul_on = metis.readout(ndit = NDIT, exptime = EXPTIME)[0]
    sci = outhdul_on[1].data
    #ipdb.set_trace()
    # Get perfect, background-subtracted PSF - no detector noise
    psf_perfect = sci - background

    print('!!! --- ARTIFICIALLY SUBTRACTING OFF A BACKGROUND RESIDUAL; FIX LATER --- !!')
    psf_perfect -= np.nanmean(psf_perfect)

    # Oversample the background-subtracted PSF to match the cookie_cut_out_sci oversampling
    psf_perfect_oversamp = zoom(psf_perfect, fac_oversamp, order=3)

    # for debugging
    fits.writeto("psf_perfect_oversamp.fits", psf_perfect_oversamp, overwrite=True)
    print("Saved psf_perfect_oversamp.fits for checking.")


    #ipdb.set_trace()

    # take a cutout of the PSF at the exact same coordinates as the cookie cut-out
    psf_perfect_cutout = psf_perfect_oversamp[int(y_center_final_oversamp-0.5*cookie_cut_out_sci.shape[0]):int(y_center_final_oversamp+0.5*cookie_cut_out_sci.shape[0]), \
        int(x_center_final_oversamp-0.5*cookie_cut_out_sci.shape[1]):int(x_center_final_oversamp+0.5*cookie_cut_out_sci.shape[1])]

    # cut out the central region the same size as the cookie cut-out
    #psf_perfect_cutout = psf_perfect[int(psf_perfect.shape[0]/2-0.5*cookie_cut_out_sci.shape[0]):int(psf_perfect.shape[0]/2+0.5*cookie_cut_out_sci.shape[0]), \
    #    int(psf_perfect.shape[1]/2-0.5*cookie_cut_out_sci.shape[1]):int(psf_perfect.shape[1]/2+0.5*cookie_cut_out_sci.shape[1])]

    # multiply psf_perfect_cutout by a coefficient to make it a best-fit to cookie_cut_out_sci
    coefficient = np.sum(cookie_cut_out_sci) / np.sum(psf_perfect_cutout)
    psf_perfect_cutout_best_fit = psf_perfect_cutout * coefficient

    # Make subplots of cookie_cut_out_sci, psf_perfect_cutout_best_fit, and the residuals
    plt.figure(figsize=(12, 4))
    
    # Panel 1: cookie_cut_out_sci
    plt.subplot(1, 3, 1)
    zscale1 = ZScaleInterval()
    vmin1, vmax1 = zscale1.get_limits(cookie_cut_out_sci)
    plt.imshow(cookie_cut_out_sci, origin="lower", cmap="viridis", vmin=vmin1, vmax=vmax1)
    plt.title("cookie_cut_out_sci")
    plt.colorbar(shrink=0.7, label="Counts")
    
    # Panel 2: psf_perfect_cutout_best_fit
    plt.subplot(1, 3, 2)
    zscale2 = ZScaleInterval()
    vmin2, vmax2 = zscale2.get_limits(psf_perfect_cutout_best_fit)
    plt.imshow(psf_perfect_cutout_best_fit, origin="lower", cmap="viridis", vmin=vmin2, vmax=vmax2)
    plt.title("psf_perfect_cutout_best_fit")
    plt.colorbar(shrink=0.7, label="Counts")
    
    # Panel 3: Residuals
    residuals = cookie_cut_out_sci - psf_perfect_cutout_best_fit
    plt.subplot(1, 3, 3)
    zscale3 = ZScaleInterval()
    vmin3, vmax3 = zscale3.get_limits(residuals)
    plt.imshow(residuals, origin="lower", cmap="RdBu", vmin=vmin3, vmax=vmax3)
    plt.title("Residuals (sci - best_fit)")
    plt.colorbar(shrink=0.7, label="Counts")
    
    plt.tight_layout()
    plt.show()

    return psf_perfect_cutout_best_fit


def strehl_grid(file_name, fp_mask):
    '''
    Find the Strehl ratio of a grid of PSFs
    
    INPUTS:
    file_name: name of the file containing the grid of PSFs
    fp_mask: focal plane mask (string)

    OUTPUTS:
    None; writes out plots and data
    '''

    # return the locations and other data for each PSF

    grid_frame = fits.open(file_name)
    grid_data = grid_frame[1].data
    grid_header = grid_frame[1].header

    '''
    (1804, 243), (1804, 633), (1804, 1029), (1804, 1418), (1804, 1810), 
    (1415, 241), (1419, 1808), 
    (1023, 241), (1025, 1810), 
    (632, 240), (630, 1810), 
    (240, 242), (240, 632), (238, 1024), (242, 1418), (240, 1808), (1273, 778), (1273, 1024), (1273, 1273),
    (1024, 778), (1024, 1023), (1024, 1273),
    (776, 776), (776, 1026), (776, 1273)
    '''

    # coordinate starting guesses for the grid
    coords_guesses_all = np.array([(1804, 243), (1804, 633), (1804, 1029), (1804, 1418), (1804, 1810), \
        (1415, 241), (1419, 1808), \
            (1023, 241), (1025, 1810), \
                (632, 240), (630, 1810), \
                    (240, 242), (240, 632), (238, 1024), (242, 1418), (240, 1808), (1273, 778), (1273, 1024), (1273, 1273),\
                        (1027, 780), (1024, 1023), (1024, 1273), \
                            (776, 776), (776, 1026), (776, 1273)])

    coords_guesses_y_all = coords_guesses_all[:, 0]
    coords_guesses_x_all = coords_guesses_all[:, 1]

    #guesses_grid = np.array(())

    # for debugging
    '''
    print('! ---------- debugging --------- !')
    x_grid = np.array([3.0, 3.0, 3.1, 3.3])
    y_grid = np.array([13.0, 13.0, 13.1, 13.3])
    '''

    # oversample the image to find centroids, FWHM
    oversample_factor = 4
    # Step 1: Oversample the PSFs by a factor of 4 using bicubic interpolation
    grid_data_oversamp = zoom(grid_data, oversample_factor, order=3)
    #psf_simmed_oversamp = zoom(psf_simmed, oversample_factor, order=3)
    coords_guesses_x_all_oversamp = coords_guesses_x_all * oversample_factor
    coords_guesses_y_all_oversamp = coords_guesses_y_all * oversample_factor
    coords_guesses_all_oversamp = np.vstack((coords_guesses_y_all_oversamp, coords_guesses_x_all_oversamp)).T

    # find the grid centroids

    # ... using photutils built-in fcn
    x_pos_pix_oversamp, y_pos_pix_oversamp = centroid_sources(grid_data_oversamp, 
                                    xpos=coords_guesses_x_all_oversamp, 
                                    ypos=coords_guesses_y_all_oversamp, 
                                    box_size=41,
                                    centroid_func=centroid_2dg)

    # zip into one array
    coords_centroided_all_oversamp = np.vstack((y_pos_pix_oversamp, x_pos_pix_oversamp)).T

    # FYI
    #fyi_plot_centroiding(grid_data_oversamp, coords_centroided_all_oversamp, zscale=False)

    # make a cut-out of each psf and make a best-fit 2D Gaussian
    raw_cutout_size = 20 * oversample_factor
    num_coord = 0

    cookie_cut_out_best_fit_list = []
    coord_x_array = np.zeros(len(y_pos_pix_oversamp))
    coord_y_array = np.zeros(len(y_pos_pix_oversamp))
    fwhm_x_pix_array = np.zeros(len(y_pos_pix_oversamp))
    fwhm_y_pix_array = np.zeros(len(y_pos_pix_oversamp))
    sigma_x_pix_array = np.zeros(len(y_pos_pix_oversamp))
    sigma_y_pix_array = np.zeros(len(y_pos_pix_oversamp))
    angle_theta_array = np.zeros(len(y_pos_pix_oversamp))
    amplitude_counts_array = np.zeros(len(y_pos_pix_oversamp))

    # make a copy from which we will subtract the PSFs to see the residuals
    canvas_grid_data = np.copy(grid_data)

    # loop over each centroided PSF
    for num_coord in range(len(y_pos_pix_oversamp)):

        ipdb.set_trace()


        # is a cutout even necessary?
        cookie_edge_size = raw_cutout_size
        idx_x_start = int(x_pos_pix_oversamp[num_coord]-0.5*cookie_edge_size)
        idx_x_end = int(x_pos_pix_oversamp[num_coord]+0.5*cookie_edge_size)
        idx_y_start = int(y_pos_pix_oversamp[num_coord]-0.5*cookie_edge_size)
        idx_y_end = int(y_pos_pix_oversamp[num_coord]+0.5*cookie_edge_size)
        cookie_cut_out_sci_oversamp = grid_data_oversamp[idx_y_start:idx_y_end, idx_x_start:idx_x_end]
        #ipdb.set_trace()

        # FYI plot
        plt.clf()
        plt.imshow(cookie_cut_out_sci_oversamp, origin='lower', cmap='gray_r')
        # Convert scatter coordinates to cutout-relative coordinates
        # this point is that found with centroid_sources()
        x_scatter = x_pos_pix_oversamp[num_coord] - idx_x_start
        y_scatter = y_pos_pix_oversamp[num_coord] - idx_y_start
        plt.scatter(x_scatter, y_scatter, color='red', s=10)
        plt.title(f'Cookie cut-out sci at coord (y,x): {y_pos_pix_oversamp[num_coord]}, {x_pos_pix_oversamp[num_coord]}')
        plt.colorbar()
        plt.show()
        plt.close()
        #ipdb.set_trace()

        # Adjust the centroid coordinate for the cut-out: subtract the cutout starting indices to get cutout-relative coordinates
        coords_guess_this_cutout = np.array([
            coords_centroided_all_oversamp[num_coord][0] - idx_y_start,
            coords_centroided_all_oversamp[num_coord][1] - idx_x_start
        ])


        # find FWHM of Gaussian-best-fit to empirical, and get the best centroid on the PSF
        # correct for fact we 
        #ipdb.set_trace()
        x_center_pix_gaussian_best_fit_oversamp, y_center_pix_gaussian_best_fit_oversamp, fwhm_x_pix_gaussian_best_fit_oversamp, fwhm_y_pix_gaussian_best_fit_oversamp, amplitude_counts_gaussian_best_fit_oversamp = fit_gaussian_fwhm(cookie_cut_out_sci_oversamp, 
                                                                                                        coords_guess=coords_guess_this_cutout, 
                                                                                                        plot_string=f'num_coord_{num_coord}', 
                                                                                                        fac_oversamp=oversample_factor)


        # convert the coordinates of the cutout back to those of the entire oversampled image
        x_center_pix_gaussian_best_fit_oversamp_fullarray = x_center_pix_gaussian_best_fit_oversamp + idx_x_start
        y_center_pix_gaussian_best_fit_oversamp_fullarray = y_center_pix_gaussian_best_fit_oversamp + idx_y_start

        # fit a 2D Gaussian

        # find FWHM of empirical 
        '''
        fwhm_y_pix_empirical, fwhm_x_pix_empirical = fit_empirical_fwhm(cookie_cut_out_sci, plot_string=f'num_coord_{num_coord}')
        '''

 
        #ipdb.set_trace()
        # subtract ScopeSim PSFs to see the residals (note we're still using the cookie cut-out)
        # note that the 'final' coords are the 'guessed' ones above; the PSF will in general be off-center
        best_fit_cutout_oversamp = fit_simmed_psfs(cookie_cut_out_sci_oversamp, 
                                        plot_string=f'num_coord_{num_coord}', 
                                        fp_mask=fp_mask,
                                        x_center_final_oversamp=x_pos_pix_oversamp[num_coord], 
                                        y_center_final_oversamp=y_pos_pix_oversamp[num_coord], 
                                        fac_oversamp=oversample_factor)
        #canvas_grid_data[idx_y_start:idx_y_end, idx_x_start:idx_x_end] = resids_cutout_oversamp

        
        # resample back to the original size
        x_center_pix_gaussian_best_fit_normsamp = x_center_pix_gaussian_best_fit_oversamp_fullarray / oversample_factor
        y_center_pix_gaussian_best_fit_normsamp = y_center_pix_gaussian_best_fit_oversamp_fullarray / oversample_factor
        fwhm_x_pix_gaussian_best_fit_normsamp = fwhm_x_pix_gaussian_best_fit_oversamp / oversample_factor
        fwhm_y_pix_gaussian_best_fit_normsamp = fwhm_y_pix_gaussian_best_fit_oversamp / oversample_factor
     

        #ipdb.set_trace()
        # make cutout around the model (for plot)

        # save cookie_cut_out_sci and cookie_cut_out_best_fit as fits files
        '''
        file_name_sci = 'cookie_cut_out_sci.fits'
        file_name_best_fit = 'cookie_cut_out_best_fit.fits'
        fits.writeto(file_name_sci, cookie_cut_out_sci, overwrite=True)
        #fits.writeto(file_name_best_fit, cookie_cut_out_best_fit, overwrite=True)
        print(f'Saved {file_name_sci} and \n{file_name_best_fit}')
        '''

        # update arrays/lists
        cookie_cut_out_best_fit_list.append(best_fit_cutout_oversamp)

        coord_x_array[num_coord] = x_center_pix_gaussian_best_fit_normsamp
        coord_y_array[num_coord] = y_center_pix_gaussian_best_fit_normsamp
        fwhm_x_pix_array[num_coord] = fwhm_x_pix_gaussian_best_fit_normsamp
        fwhm_y_pix_array[num_coord] = fwhm_y_pix_gaussian_best_fit_normsamp
        amplitude_counts_array[num_coord] = amplitude_counts_gaussian_best_fit_oversamp # note the amplitude doesn't need to be resampled
        #sigma_x_pix_array[num_coord] = sigma_x_pix
        #sigma_y_pix_array[num_coord] = sigma_y_pix
        #angle_theta_array[num_coord] = angle_theta


    ipdb.set_trace()
    # plot the grid_data and annotate it with the best-fit fwhm in x and y for each PSF
    plt.clf()
    plt.imshow(grid_data, origin='lower', cmap='gray_r')
    for num_coord in range(len(coord_x_array)):
        # Draw a line from the text location to the PSF's actual (x, y) coordinate
        text_x = coord_x_array[num_coord] - 125
        text_y = coord_y_array[num_coord] + 10
        plt.text(
            text_x,
            text_y,
            f'x: {fwhm_x_pix_array[num_coord]:.2f}, \n y: {fwhm_y_pix_array[num_coord]:.2f}, \n theta: {angle_theta_array[num_coord]:.2f}, \n amp: {amplitude_counts_array[num_coord]:.2f}',
            color='k',
            fontsize=7, rotation=20
        )
    plt.title('FWHM in x and y (pix), amplitude (counts)')
    plt.show()
    plt.close()

    ipdb.set_trace()

    return


def main():

    stem = '/podman-share/metis_work/playing_with_scopesim/IMG_04_sample_input_data/'

    # files for finding the Strehl
    # if grid mask is used
    # the files for finding the plate scale (grid mask)
    file_name = stem + 'strehl/IMG_OPT_04_wcu_focal_mask_grid_lm_pupil_mask_open_filter_Br_alpha_clocking_angle_0.fits'
    #file_name = stem + 'strehl/IMG_OPT_04_wcu_focal_mask_pinhole_lm_pupil_mask_open_filter_Br_alpha_clocking_angle_0.fits'


    # check plate scale 
    strehl_grid(file_name, fp_mask='grid_lm') ## ## TODO: ADD OTHER THIGNS TO ARGUMENT, LIKE OBSERVING FILTER, ETC., TO MAKE EVERYTHING DOWNSTREAM CONSISTENT


if __name__ == "__main__":
    main()