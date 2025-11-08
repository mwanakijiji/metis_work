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

#from photutils.centroids import centroid_sources, centroid_com

import ipdb


def fov(file_names_fov_tl, file_names_fov_tr, file_names_fov_bl, file_names_fov_br):
    ##################################################################
    ## TEST 1: FOV
    ## read in frames with PSF at each corner and see how far apart they are in pixel space
    ## compare with what is expected based on scaling relation

    # read in FITS files
    tl_frame = fits.open(file_names_fov_tl)
    tl_data = tl_frame[1].data

    tr_frame = fits.open(file_names_fov_tr)
    tr_data = tr_frame[1].data

    bl_frame = fits.open(file_names_fov_bl)
    bl_data = bl_frame[1].data

    br_frame = fits.open(file_names_fov_br)
    br_data = br_frame[1].data

    # FYI plot
    '''
    image_2_plot = br_data
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image_2_plot)
    plt.clf()
    plt.imshow(image_2_plot, origin='lower', vmin=vmin, vmax=vmax)
    plt.tight_layout()
    plt.show()
    plt.close()
    '''

    # centroid guesses (y,x)
    guess_tl = (1937, 119)
    guess_tr = (1942, 1946)
    guess_bl = (111, 115)
    guess_br = (116, 1941)

    x_tl, y_tl = centroid_sources(tl_data, xpos=guess_tl[1], ypos=guess_tl[0])
    x_tr, y_tr = centroid_sources(tr_data, xpos=guess_tr[1], ypos=guess_tr[0])
    x_bl, y_bl = centroid_sources(bl_data, xpos=guess_bl[1], ypos=guess_bl[0])
    x_br, y_br = centroid_sources(br_data, xpos=guess_br[1], ypos=guess_br[0])

    # find distance across the sides [pix]
    del_x_top = np.abs(x_tl-x_tr)
    del_x_bottom = np.abs(x_bl-x_br)
    del_y_left = np.abs(y_tl-y_bl)
    del_y_right = np.abs(y_tr-y_br)

    print('! ---------- USING A PLACEHOLDER SCALING RELN ---------- !')
    scaling_reln = 0.01046 # [arcsec/pix]

    fov_calc = scaling_reln * np.mean(np.array(del_x_top,del_x_bottom,del_y_left,del_y_right))
    print('FOV:', fov_calc, '[arcsec]')


def plate_scale(file_name_grid):

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
                        (1024, 778), (1024, 1023), (1024, 1273), \
                            (776, 776), (776, 1026), (776, 1273)])

    coords_guesses_y_all = coords_guesses_all[:, 0]
    coords_guesses_x_all = coords_guesses_all[:, 1]

    ipdb.set_trace()
    #guesses_grid = np.array(())

    # find the grid centroids
    x_grid, y_grid = centroid_sources(grid_data, xpos=coords_guesses_x_all, ypos=coords_guesses_y_all)

    # zip into one array
    coords_centroided_all = np.vstack((y_grid, x_grid)).T

    pairwise_distance_matrix = distance_matrix(coords_centroided_all, coords_centroided_all)

    permutation_distance_lookup = {}
    for (idx1, coord1), (idx2, coord2) in permutations(enumerate(coords_centroided_all), 2):
        permutation_distance_lookup[(tuple(coord1), tuple(coord2))] = pairwise_distance_matrix[idx1, idx2]

    ipdb.set_trace()

    return x_grid, y_grid


def main():

    stem = '/podman-share/metis_work/playing_with_scopesim/IMG_02_sample_input_data/'
    # the files for finding the FOV (dithered PSFs)
    file_names_fov_tl = stem + 'fov/top_left_IMG_OPT_04_wcu_focal_plane_pinhole_lm.fits' # top left
    file_names_fov_tr = stem + 'fov/top_right_IMG_OPT_04_wcu_focal_plane_pinhole_lm.fits' # top left
    file_names_fov_bl = stem + 'fov/bottom_left_IMG_OPT_04_wcu_focal_plane_pinhole_lm.fits' # top left
    file_names_fov_br = stem + 'fov/bottom_right_IMG_OPT_04_wcu_focal_plane_pinhole_lm.fits' # top left

    # the files for finding the plate scale (grid mask)
    file_name_grid = stem + 'ps/IMG_OPT_04_plate_scale_grid_image_grid_lm_Lp.fits'

    # the files for constraining stray light (pinhole)
    file_name_pinhole = stem + 'stray_light/IMG_OPT_04_plate_scale_grid_image_pinhole_lm_Lp.fits'

    # check FOV
    #fov(file_names_fov_tl, file_names_fov_tr, file_names_fov_bl, file_names_fov_br)

    # check plate scale 
    plate_scale(file_name_grid)


if __name__ == "__main__":
    main()

ipdb.set_trace()