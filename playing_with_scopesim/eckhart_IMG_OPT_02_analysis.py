# Does some simple analysis of simulated images written out by the sim notebook.
import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS

import scipy
from scipy.spatial import distance_matrix
from itertools import permutations
import glob
import os

from matplotlib import pyplot as plt
from matplotlib import colors
from astropy.visualization import ZScaleInterval

from photutils.centroids import centroid_sources, centroid_com, centroid_2dg

import pandas as pd

import ipdb


def fov(file_names_fov, scaling_reln, dither=False):
    '''
    Find the FOV of the telescope by centroiding on PSFs in the field of view, and comparing with what would
    be expected based on a scaling relation.

    INPUTS:
    - file_names_fov: dict containing the file names for the PSFs
        - has keys 'tl', 'tr', 'bl', 'br' if dither=True
        - has key 'grid' if dither=False
    - scaling_reln: scaling relation between arcseconds and pixels
    - dither: whether to dither the PSF (True or False)

    OUTPUTS:
    - FOV: field of view in arcseconds
    '''

    ##################################################################
    ## TEST 1: FOV
    ## read in frames with PSF at each corner and see how far apart they are in pixel space
    ## compare with what is expected based on scaling relation

    # there are two options: read in images with a single PSF in each corner, or a single image with a grid of PSFs

    if dither:
        # read in 4 different FITS files
        file_names_fov_tl = file_names_fov['tl']
        file_names_fov_tr = file_names_fov['tr']
        file_names_fov_bl = file_names_fov['bl']
        file_names_fov_br = file_names_fov['br']

        # read in FITS files
        tl_frame = fits.open(file_names_fov_tl)
        tl_data = tl_frame[0].data

        tr_frame = fits.open(file_names_fov_tr)
        tr_data = tr_frame[0].data

        bl_frame = fits.open(file_names_fov_bl)
        bl_data = bl_frame[0].data

        br_frame = fits.open(file_names_fov_br)
        br_data = br_frame[0].data

        ipdb.set_trace()
        # FYI plot

        image_2_plot = br_data
        zscale = ZScaleInterval()
        vmin, vmax = zscale.get_limits(image_2_plot)
        plt.clf()
        plt.imshow(image_2_plot, origin='lower', vmin=vmin, vmax=vmax)
        plt.tight_layout()
        plt.show()
        plt.close()


        # centroid guesses (y,x)
        guess_tl = (1961, 114)
        guess_tr = (1942, 1946)
        guess_bl = (111, 115)
        guess_br = (99, 1947)

        x_tl, y_tl = centroid_sources(tl_data, xpos=guess_tl[1], ypos=guess_tl[0], box_size=60, centroid_func=centroid_com)
        x_tr, y_tr = centroid_sources(tr_data, xpos=guess_tr[1], ypos=guess_tr[0], box_size=21, centroid_func=centroid_com)
        x_bl, y_bl = centroid_sources(bl_data, xpos=guess_bl[1], ypos=guess_bl[0], box_size=21, centroid_func=centroid_com)
        x_br, y_br = centroid_sources(br_data, xpos=guess_br[1], ypos=guess_br[0], box_size=21, centroid_func=centroid_com)

        x_grid = np.array([x_tl, x_tr, x_bl, x_br])
        y_grid = np.array([y_tl, y_tr, y_bl, y_br])

        # for debugging
        '''
        plt.clf()
        plt.imshow(tl_data, origin='lower', 
        plt.scatter(guess_tl[1], guess_tl[0], color='red', s=10)
        plt.scatter(x_tl, y_tl, color='blue', s=10)
        plt.show()
        ipdb.set_trace()
        '''


    else:
        # if just 1 frame with grid pattern
        file_names_fov_grid = file_names_fov['grid']
        grid_frame = fits.open(file_names_fov_grid)
        grid_data = grid_frame[1].data

        guesses_x = [241, 1808, 240, 1809]
        guesses_y = [1809, 1809, 240, 240]

        ipdb.set_trace()
        # centroid on the 4 corners of the grid
        x_grid, y_grid = centroid_sources(grid_data, 
                                    xpos=guesses_x, 
                                    ypos=guesses_y, 
                                    box_size=21,
                                    centroid_func=centroid_com)

        x_tl, y_tl =  x_grid[0], y_grid[0]
        x_tr, y_tr =  x_grid[1], y_grid[1]
        x_bl, y_bl =  x_grid[2], y_grid[2]
        x_br, y_br =  x_grid[3], y_grid[3]

    # FYI
    plt.clf()
    plt.imshow(grid_data, origin='lower')
    plt.scatter(x_grid, y_grid, color='red', s=10)
    plt.show()

    # find distance across the sides [pix]
    del_x_top = np.abs(x_tl-x_tr)
    del_x_bottom = np.abs(x_bl-x_br)
    del_y_left = np.abs(y_tl-y_bl)
    del_y_right = np.abs(y_tr-y_br)

    fov_calc = scaling_reln * np.mean(np.array([del_x_top,del_x_bottom,del_y_left,del_y_right]))
    print('Mean FOV:', fov_calc, '[arcsec]')

    return fov_calc


def plate_scale(file_name_grid, scaling_reln):
    ##################################################################
    ## TEST 2: pixel scale

    grid_frame = fits.open(file_name_grid)
    grid_data = grid_frame[1].data

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
    x_grid, y_grid = centroid_sources(grid_data, 
                                    xpos=coords_guesses_x_all, 
                                    ypos=coords_guesses_y_all, 
                                    box_size=21,
                                    centroid_func=centroid_com)

    # zip into one array
    coords_centroided_all = np.vstack((y_grid, x_grid)).T

    pairwise_distance_matrix = distance_matrix(coords_centroided_all, coords_centroided_all)

    permutation_distance_lookup = {}
    for (idx1, coord1), (idx2, coord2) in permutations(enumerate(coords_centroided_all), 2):
        distance_pix = pairwise_distance_matrix[idx1, idx2]
        permutation_distance_lookup[(tuple(coord1), tuple(coord2))] = {
            "distance_pix": distance_pix,
            "distance_arcsec_fake_meas": distance_pix * scaling_reln + np.random.normal(loc=0, scale=0.1 * distance_pix * scaling_reln, size=1),
        }

    # Convert permutation_distance_lookup into a pandas dataframe.
    
    ## ## CONTINUE HERE: CONVERT INTO PANDAS DATAFRAME, AND MAKE A CDF WITH CODE FROM DEWARP
    permutation_distance_df = pd.DataFrame([
        {"coord1": k[0], "coord2": k[1], "distance_pix": v["distance_pix"], "distance_arcsec_fake_meas": v["distance_arcsec_fake_meas"][0]}
        for k, v in permutation_distance_lookup.items()
    ])

    # stand-in for true arcsec values

    return 


def scattered_light(file_name_pinhole, ps):
    '''
    ##################################################################
    ## TEST 3: scattered light

    Cut a circle around the PSF and see what residual light there is
    (Note that input file has to be carefully background-subtracted)

    INPUTS:
    - file_name_pinhole: name of the pinhole file
    - ps: pixel scale

    OUTPUTS:
    - scattered_light_fraction: fraction of light in the PSF that is scattered
    '''

    # centroid on the (central) psf
    frame = fits.open(file_name_pinhole)
    data = frame[1].data

    coords_guess = (1024, 1024)

    x_center, y_center = centroid_sources(data, 
                                    xpos=data[1], 
                                    ypos=data[0], 
                                    box_size=21,
                                    centroid_func=centroid_com)

    # based on the wavelength of light, where should the first dark Airy ring be?
    wavelength = 3.3e-6 # [m] # stand-in
    diameter_pupil = 39.0 # [m] # stand-in
    dark_ring_rad = 1.22 * wavelength / diameter_pupil
    dark_ring_pix = dark_ring_rad * 206265 * ps # [pix]

    # make a circular mask where all the pixels within the dark ring are nans
    # Define a circular mask centered at (x_center, y_center) with radius dark_ring_rad [pixels]
    y_indices, x_indices = np.ogrid[:data.shape[0], :data.shape[1]]
    mask_circle = (x_indices - x_center)**2 + (y_indices - y_center)**2 <= dark_ring_pix**2
    data_masked = np.copy(data, deep=True)

    # find the peak irradiance from the PSF 
    peak_irr_inside_psf = np.nanmax(data_masked[mask_circle])
    # find the peak irradiance outside the PSF
    peak_irr_outside_psf = np.nanmax(data_masked[~mask_circle])

    # ratio of peak irradiance
    ratio_irr = peak_irr_inside_psf/peak_irr_outside_psf

    return ratio_irr


def main():

    stem = '/podman-share/metis_work/playing_with_scopesim/IMG_02_sample_input_data/'

    # files for finding the FOV
    '''
    # if dithering is used
    file_names_fov = {
        "tl": stem + 'fov/top_left_IMG_OPT_04_wcu_focal_plane_pinhole_lm.fits',
        "tr": stem + 'fov/top_right_IMG_OPT_04_wcu_focal_plane_pinhole_lm.fits',
        "bl": stem + 'fov/bottom_left_IMG_OPT_04_wcu_focal_plane_pinhole_lm.fits',
        "br": stem + 'fov/bottom_right_IMG_OPT_04_wcu_focal_plane_pinhole_lm.fits',
    }
    '''
    # if grid mask is used
    file_names_fov = {
        "grid": stem + 'fov/IMG_OPT_02_image_grid_lm_short-L.fits',
    }

    # the files for finding the plate scale (grid mask)
    file_name_grid = stem + 'ps/IMG_OPT_02_image_grid_lm_short-L.fits'

    # the files for constraining stray light (pinhole)
    file_name_pinhole = stem + 'stray_light/IMG_OPT_04_plate_scale_grid_image_pinhole_lm_short-L.fits'

    # true scaling relation between arcseconds and pixels (measured, not assumed)
    # value and units may change depending on experimental setup
    print('! ---------- USING A PLACEHOLDER SCALING RELN ---------- !')
    scaling_reln = 0.00679 # [arcsec/pix]

    # check FOV
    fov(file_names_fov, scaling_reln, dither=False)

    # check plate scale 
    #plate_scale(file_name_grid, scaling_reln)

    # check scattered light
    #scattered_light(file_name_pinhole, ps=0.01046) # PS here is just a stand-in


if __name__ == "__main__":
    main()