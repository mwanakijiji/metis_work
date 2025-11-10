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

def gaussian_2d(xy_mesh, amplitude, xo, yo, sigma_x_pix, sigma_y_pix, theta):
    x, y = xy_mesh
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2) / (2 * sigma_x_pix**2) + (np.sin(theta)**2) / (2 * sigma_y_pix**2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x_pix**2) + (np.sin(2 * theta)) / (4 * sigma_y_pix**2)
    c = (np.sin(theta)**2) / (2 * sigma_x_pix**2) + (np.cos(theta)**2) / (2 * sigma_y_pix**2)
    g = amplitude * np.exp(-(a * ((x - xo)**2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo)**2)))
    return g.ravel()


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

        # make best fit Gaussian to empirical; all fit parameters are free
        ## ## TO DO: ARE THE INDEXES RIGHT HERE?
        cookie_cut_out_best_fit, fwhm_x_pix, fwhm_y_pix, sigma_x_pix, sigma_y_pix, angle_theta = fit_gaussian(cookie_cut_out_sci, \
            [
                coords_centroided_all[num_coord][1] - (x_pos_pix[num_coord] - 0.5 * cookie_edge_size),
                coords_centroided_all[num_coord][0] - (y_pos_pix[num_coord] - 0.5 * cookie_edge_size)
            ])

        #fit_result, fwhm_x_pix, fwhm_y_pix, sigma_x_pix, sigma_y_pix = fit_gaussian(grid_data, [coords_centroided_all[num_coord][1], coords_centroided_all[num_coord][0]])

        # make cutout around the model (for plot)
        #cookie_cut_out_best_fit = fit_result[int(y_pos_pix[0]-0.5*cookie_edge_size):int(y_pos_pix[0]+0.5*cookie_edge_size), int(x_pos_pix[0]-0.5*cookie_edge_size):int(x_pos_pix[0]+0.5*cookie_edge_size)]

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