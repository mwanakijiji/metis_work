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

from matplotlib import pyplot as plt
from matplotlib import colors
from astropy.visualization import ZScaleInterval

from photutils.centroids import centroid_sources, centroid_com, centroid_2dg

import pandas as pd

import ipdb

import scopesim as sim
from skimage import measure

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
    ipdb.set_trace()
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
    ipdb.set_trace()
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
    """
    #ipdb.set_trace()
    y, x = np.indices(frame.shape)
    xy_mesh = (x, y)
    p0 = [np.max(frame), center_guess[0], center_guess[1], 1, 1, 0]
    popt, pcov = curve_fit(gaussian_2d, xy_mesh, frame.ravel(), p0=p0)
    fitted_array = gaussian_2d(xy_mesh, *popt).reshape(frame.shape)
    fwhm_x_pix = 2 * np.sqrt(2 * np.log(2)) * np.abs(popt[3])
    fwhm_y_pix = 2 * np.sqrt(2 * np.log(2)) * np.abs(popt[4])
    sigma_x_pix = popt[3]
    sigma_y_pix = popt[4]
    angle_theta = popt[5]
    
    return fitted_array, fwhm_x_pix, fwhm_y_pix, sigma_x_pix, sigma_y_pix, angle_theta


def fyi_plot_centroiding(array_to_plot, coords_to_plot, zscale=False):
    # INSERT_YOUR_CODE

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(array_to_plot)
    plt.clf()
    plt.imshow(array_to_plot, origin='lower', vmin=vmin, vmax=vmax, cmap='gray')
    plt.scatter(coords_to_plot[:, 1], coords_to_plot[:, 0], color='red', s=10)
    plt.show()
    plt.close()


def fit_gaussian_fwhm(cookie_cut_out_sci, coords_centroided, plot_string):
    '''
    Find FWHM of Gaussian-best-fit to empirical; all fit parameters are free

    INPUTS:
    cookie_cut_out_sci: 2D array of the science frame
    coords_centroided: 2D array of the centroided coordinates (one coordinate pair)
    plot_string: string to add to the plot file name
    '''

    ## ## TO DO: ARE THE INDEXES RIGHT HERE?
    cookie_cut_out_best_fit, fwhm_x_pix, fwhm_y_pix, sigma_x_pix, sigma_y_pix, angle_theta = fit_gaussian(cookie_cut_out_sci, \
        [int(cookie_cut_out_sci.shape[1]/2),int(cookie_cut_out_sci.shape[1]/2)])
    residuals = cookie_cut_out_sci - cookie_cut_out_best_fit

    ipdb.set_trace()
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
    axs[1, 1].legend()
    axs[1, 1].set_title('1D cross-section, science vs best-fit')
    plt.suptitle(f'PSF at coord (y,x): {coords_centroided}')
    plt.tight_layout()
    plt.show()
    # Save the plot to file with num_coord as a 2-digit zero-padded string
    plot_filename = f'psf_gaussian_best_fit_'+plot_string+'.png'
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f'Figure saved as {plot_filename}')
    plt.close()

    return fwhm_y_pix, fwhm_x_pix


def find_fwhm_scopesim(cookie_cut_out_sci):
    '''
    Find FWHM of a PSF using a perfect PSF from ScopeSim
    
    INPUTS:
    
    '''

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

    wcu.set_bb_aperture(value = 1.0) # open BB source

    metis["filter_wheel"].change_filter(obs_filter)

    print('--------------------------------')
    print('Current Observing filter:', obs_filter)
    print('Current WCU FP mask:', wcu.fpmask)
    print('Current WCU PP mask:', pp_mask)
    print('Next absolute dither position:', dither_pos)

    # dither by shifting the FP mask
    # (note these shifts are absolute, not relative)
    wcu.set_fpmask(fp_mask, angle=0, shift=dither_pos)

    print('Opening WCU BB aperture...')

    ipdb.set_trace()

    metis.observe()
    # Get perfect PSF - no detector noise
    #hdul_perfect = metis.image_planes[0].hdu

    '''
    FIND FWHM HERE
    '''
    
    return fwhm_y_pix, fwhm_x_pix


def strehl_grid(file_name_grid):
    ##################################################################
    ## TEST 2: pixel scale

    grid_frame = fits.open(file_name_grid)
    grid_data = grid_frame[0].data

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

    # find the grid centroids
    x_pos_pix, y_pos_pix = centroid_sources(grid_data, 
                                    xpos=coords_guesses_x_all, 
                                    ypos=coords_guesses_y_all, 
                                    box_size=21,
                                    centroid_func=centroid_2dg)

    # zip into one array
    coords_centroided_all = np.vstack((y_pos_pix, x_pos_pix)).T

    # FYI
    fyi_plot_centroiding(grid_data, coords_centroided_all, zscale=False)

    # make a cut-out of each psf and make a best-fit 2D Gaussian
    raw_cutout_size = 31
    num_coord = 0

    cookie_cut_out_best_fit_list = []
    fwhm_x_pix_array = np.zeros(len(y_pos_pix))
    fwhm_y_pix_array = np.zeros(len(y_pos_pix))
    sigma_x_pix_array = np.zeros(len(y_pos_pix))
    sigma_y_pix_array = np.zeros(len(y_pos_pix))
    angle_theta_array = np.zeros(len(y_pos_pix))

    # loop over each centroided PSF
    for num_coord in range(len(y_pos_pix)):
        cookie_edge_size = raw_cutout_size
        cookie_cut_out_sci = grid_data[int(y_pos_pix[num_coord]-0.5*cookie_edge_size):int(y_pos_pix[num_coord]+0.5*cookie_edge_size), \
            int(x_pos_pix[num_coord]-0.5*cookie_edge_size):int(x_pos_pix[num_coord]+0.5*cookie_edge_size)]

        print('! ----------- ADDING IN BACKGROUND VALUE TO MAKE THE BACKGROUND ZERO; NEED TO MODIFY LATER ----------- !')
        cookie_cut_out_sci = cookie_cut_out_sci - np.median(grid_data)

        # find FWHM of empirical 
        fwhm_y_pix_empirical, fwhm_x_pix_empirical = fit_empirical_fwhm(cookie_cut_out_sci, plot_string=f'num_coord_{num_coord}')
        # find FWHM of Gaussian-best-fit to empirical
        fwhm_y_pix_empirical, fwhm_x_pix_empirical = fit_gaussian_fwhm(cookie_cut_out_sci, coords_centroided=coords_centroided_all[num_coord], plot_string=f'num_coord_{num_coord}')
        # find FWHM based on a ScopeSim PSF


        ipdb.set_trace()

     


        # make cutout around the model (for plot)

        # save cookie_cut_out_sci and cookie_cut_out_best_fit as fits files
        file_name_sci = 'cookie_cut_out_sci.fits'
        file_name_best_fit = 'cookie_cut_out_best_fit.fits'
        fits.writeto(file_name_sci, cookie_cut_out_sci, overwrite=True)
        fits.writeto(file_name_best_fit, cookie_cut_out_best_fit, overwrite=True)
        print(f'Saved {file_name_sci} and \n{file_name_best_fit}')

        # update arrays/lists
        cookie_cut_out_best_fit_list.append(cookie_cut_out_best_fit)
        fwhm_x_pix_array[num_coord] = fwhm_x_pix
        fwhm_y_pix_array[num_coord] = fwhm_y_pix
        sigma_x_pix_array[num_coord] = sigma_x_pix
        sigma_y_pix_array[num_coord] = sigma_y_pix
        angle_theta_array[num_coord] = angle_theta

    # plot the grid_data and annotate it with the best-fit fwhm in x and y for each PSF
    plt.clf()
    plt.imshow(grid_data, origin='lower', cmap='gray_r')
    for num_coord in range(len(y_pos_pix)):
        # Draw a line from the text location to the PSF's actual (x, y) coordinate
        text_x = x_pos_pix[num_coord] - 125
        text_y = y_pos_pix[num_coord] + 10
        plt.text(
            text_x,
            text_y,
            f'x: {fwhm_x_pix_array[num_coord]:.2f}, \n y: {fwhm_y_pix_array[num_coord]:.2f}, \n theta: {angle_theta_array[num_coord]:.2f}',
            color='k',
            fontsize=7, rotation=20
        )
    plt.title('FWHM in x and y(pix)')
    plt.show()
    plt.close()

    ipdb.set_trace()

    return


def main():

    stem = '/podman-share/metis_work/playing_with_scopesim/IMG_04_sample_input_data/'

    # files for finding the Strehl
    # if grid mask is used
    # the files for finding the plate scale (grid mask)
    file_name_grid = stem + 'strehl/IMG_OPT_02_image_grid_lm_short-L.fits'


    # check plate scale 
    strehl_grid(file_name_grid)


if __name__ == "__main__":
    main()