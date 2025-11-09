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
        fyi_plot_centroiding(tl_data, np.array([[x_tl, y_tl]]))
        '''


    else:
        # if just 1 frame with grid pattern
        file_names_fov_grid = file_names_fov['grid']
        grid_frame = fits.open(file_names_fov_grid)
        grid_data = grid_frame[1].data

        guesses_x = [241, 1808, 240, 1809]
        guesses_y = [1809, 1809, 240, 240]

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
    #fyi_plot_centroiding(grid_data, np.array([[x_grid, y_grid]]))

    # find distance across the sides [pix]
    del_x_top = np.abs(x_tl-x_tr)
    del_x_bottom = np.abs(x_bl-x_br)
    del_y_left = np.abs(y_tl-y_bl)
    del_y_right = np.abs(y_tr-y_br)

    fov_calc = scaling_reln * np.mean(np.array([del_x_top,del_x_bottom,del_y_left,del_y_right]))
    print('Mean FOV:', fov_calc, '[arcsec]')

    return fov_calc


def fyi_plot_centroiding(array_to_plot, coords_to_plot, zscale=False):
    # INSERT_YOUR_CODE

    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(array_to_plot)
    plt.clf()
    plt.imshow(array_to_plot, origin='lower', vmin=vmin, vmax=vmax, cmap='gray')
    plt.scatter(coords_to_plot[:, 1], coords_to_plot[:, 0], color='red', s=10)
    plt.show()
    plt.close()


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
                                    centroid_func=centroid_2dg)

    # FYI
    '''
    plt.clf()
    plt.imshow(grid_data, origin='lower')
    plt.scatter(x_grid, y_grid, color='red', s=10)
    # Draw connecting lines between all points
    for i in range(len(x_grid)):
        for j in range(i + 1, len(x_grid)):
            plt.plot([x_grid[i], x_grid[j]], [y_grid[i], y_grid[j]], color='white', linewidth=1, alpha=1)
    plt.show()
    '''

    # zip into one array
    coords_centroided_all = np.vstack((y_grid, x_grid)).T

    pairwise_distance_matrix = distance_matrix(coords_centroided_all, coords_centroided_all)

    permutation_distance_lookup = {}
    for (idx1, coord1), (idx2, coord2) in combinations(enumerate(coords_centroided_all), 2):
        distance_pix = pairwise_distance_matrix[idx1, idx2]
        permutation_distance_lookup[(tuple(coord1), tuple(coord2))] = {
            "distance_pix": distance_pix,
            "distance_arcsec_fake_meas": distance_pix * scaling_reln + np.random.normal(loc=0, scale=0.1 * distance_pix * scaling_reln, size=1),
        }
    print('! -------- Adding in some fake noise to the distance measurements -------- !')

    # Convert permutation_distance_lookup into a pandas dataframe.

    permutation_distance_df = pd.DataFrame([
        {"coord1": k[0], "coord2": k[1], "distance_pix": v["distance_pix"], "distance_arcsec_fake_meas": v["distance_arcsec_fake_meas"][0]}
        for k, v in permutation_distance_lookup.items()
    ])

    # add new col of plate scale values
    permutation_distance_df['plate_scale'] = permutation_distance_df['distance_arcsec_fake_meas'] / permutation_distance_df['distance_pix']

    # FYI: plot arcsec vs pixels
    plt.clf()
    plt.scatter(permutation_distance_df['distance_pix'], permutation_distance_df['distance_arcsec_fake_meas'], alpha=0.5, s=10)
    plt.xlabel('Distance [pix]')
    plt.ylabel('Distance [arcsec]')
    plt.title('Distances between PSFs in grid')
    plt.show()

    # FYI: CDF of plate scale values
    # Compute and plot the CDF of plate scale values
    plate_scales = permutation_distance_df['plate_scale'].values
    sorted_plate_scales = np.sort(plate_scales)
    cdf = np.arange(1, len(sorted_plate_scales)+1) / len(sorted_plate_scales)

    # find median plate scale, and the 16th and 84th percentiles around it
    median_plate_scale = np.median(sorted_plate_scales)
    p16_plate_scale = np.percentile(sorted_plate_scales, 16) # (pre-sorting is redundant)
    p84_plate_scale = np.percentile(sorted_plate_scales, 84)

    sig_1_minus = median_plate_scale - p16_plate_scale
    sig_1_plus = p84_plate_scale - median_plate_scale

    print(f'Plate scale, with 1-sigma uncertainty: {median_plate_scale:.4f} [arcsec/pix] ± {sig_1_minus:.4f} [arcsec/pix] to {sig_1_plus:.4f} [arcsec/pix]')

    fig, (ax_main, ax_resid) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_main.plot(sorted_plate_scales, cdf, marker='.', linestyle='none', label='Empirical CDF')

    # Overplot standard normal CDF (as reference, centered and scaled)
    standardized_x = (sorted_plate_scales - np.mean(sorted_plate_scales)) / np.std(sorted_plate_scales)
    theoretical_cdf = norm.cdf(standardized_x)
    ax_main.plot(sorted_plate_scales, theoretical_cdf, color='k', linestyle='--', label='Std norm CDF', linewidth=1.5, alpha=0.5)

    ax_main.axvline(median_plate_scale, color='k', linestyle='-', label=f'Median: {median_plate_scale:.4f} [arcsec/pix]', alpha=1)
    ax_main.axvline(p16_plate_scale, color='k', linestyle='--', label=f'16th percentile: {p16_plate_scale:.4f} [arcsec/pix]', alpha=0.5)
    ax_main.axvline(p84_plate_scale, color='k', linestyle='--', label=f'84th percentile: {p84_plate_scale:.4f} [arcsec/pix]', alpha=0.5)
    ax_main.set_ylabel('CDF')
    ax_main.set_title('CDF of Plate Scale Values')
    ax_main.legend()
    ax_main.grid(True)

    residuals = cdf - theoretical_cdf
    ax_resid.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax_resid.plot(sorted_plate_scales, residuals, marker='.', linestyle='none', color='C0')
    ax_resid.set_xlabel('Plate Scale [arcsec/pix]')
    ax_resid.set_ylabel('Residual from \nGaussian')
    ax_resid.grid(True)

    fig.tight_layout()
    plt.show()

    ipdb.set_trace()

    return 


def expected_light_exterior(radius_arcsec, wavelength=3.3e-6, diameter=39.0, plate_scale=0.004):
    """
    Calculate the fraction of the total PSF (Airy pattern) energy outside a given radius [arcsec]
    Parameters
    ----------
    radius_arcsec : float or array-like
        Radius or radii (in arcseconds) at which to compute the exterior energy fraction.
    wavelength : float, optional
        Wavelength in meters (default: 3.3e-6 m).
    diameter : float, optional
        Telescope pupil diameter, in meters (default: 39.0 m).
    plate_scale : float, optional
        Arcsec per pixel (default: 0.004 arcsec/pix for e.g. E-ELT MICADO).
        
    Returns
    -------
    energy_outside : float or np.ndarray
        Fraction of the total energy lying outside of specified radius. Value between 0 and 1.
    """    

    # Convert radius in arcsec to radians
    radius_arcsec = np.atleast_1d(radius_arcsec)
    radius_rad = np.deg2rad(radius_arcsec / 3600.)

    lambda_over_d = (wavelength / diameter) * 206265

    # Airy pattern argument
    # alpha = (pi * D / lambda) * sin(theta) ≈ (pi * D / lambda) * theta, for small angles
    alpha = (np.pi * diameter / wavelength) * radius_rad

    # Fractional encircled energy within radius r:
    # E(<r) = 1 - J0^2(alpha) - J1^2(alpha)
    J0 = j0(alpha)
    J1 = j1(alpha)
    encircled = 1 - J0**2 - J1**2

    # The fraction of energy outside radius is 1 - encircled
    energy_outside = 1 - encircled

    # Example usage:
    # At 0.1 arcsec from the center (e.g.)
    # exterior_fraction = expected_light_exterior(0.1)
    # print(f"Fraction of PSF energy outside 0.1 arcsec: {exterior_fraction:.4f}")

    return energy_outside




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
    data = frame[0].data

    coords_guess = (1021, 1019)

    ipdb.set_trace()
    x_center, y_center = centroid_sources(data, 
                                    xpos=coords_guess[1], 
                                    ypos=coords_guess[0], 
                                    box_size=21,
                                    centroid_func=centroid_2dg)

    # for checking
    fyi_plot_centroiding(data, np.array([[y_center, x_center]]), zscale=True)

    ''' First several dark Airy rings in units of lambda/D are
    ## ## CHECK THESE
	1.21967
	2.233131
	3.238315
	4.241063
	5.242764
	6.243922
	7.244760
	8.245395
	9.245893
	10.246293
    '''

    # dark rings in units of lambda/D
    dark_ring_arcsec_array_units_ld = np.array([1.21967, 2.233131, 3.238315, 4.241063, 5.242764, 6.243922, 7.244760, 8.245395, 9.245893, 10.246293])

    # based on the wavelength of light, where should the first dark Airy rings be?
    wavelength = 3.3e-6 # [m] # stand-in
    diameter_pupil = 39.0 # [m] # stand-in
    dark_ring_arcsec_array = (wavelength / diameter_pupil) * 206265 * dark_ring_arcsec_array_units_ld
    dark_ring_pix_array = dark_ring_arcsec_array / ps # [pix]

    exterior_fraction = expected_light_exterior(dark_ring_arcsec_array, \
        wavelength=wavelength, \
            diameter=diameter_pupil, \
                plate_scale=ps)

    # FYI plot to see if function is working right

    plt.clf()
    lambda_over_d = (wavelength / diameter_pupil) * 206265
    steps_subarray = np.linspace(0, 10, 200)
    radius_arcsec_array = lambda_over_d * steps_subarray
    test_exterior_fraction = expected_light_exterior(radius_arcsec_array, \
        wavelength=wavelength, \
            diameter=diameter_pupil, \
                plate_scale=ps)
    plt.plot(steps_subarray, test_exterior_fraction)
    plt.xlabel('Radius [lamdbda/D]')
    plt.ylabel('Fraction of energy')
    plt.yscale('log')
    plt.title('Fraction of energy exterior to radius, circular Airy pattern')
    plt.show()
    plt.close()


    # sum over all the pixels in the array
    sum_pixels_unmasked = np.nansum(data)

    # make a circular mask in the data frame, where all the pixels within the dark ring are nans
    # Define a circular mask centered at (x_center, y_center) with radius dark_ring_rad [pixels]
    y_indices, x_indices = np.ogrid[:data.shape[0], :data.shape[1]]

    ratio_exterior_measured_array = []

    for num_ring in range(0, len(dark_ring_pix_array)):

        mask_circle = (x_indices - x_center)**2 + (y_indices - y_center)**2 <= dark_ring_pix_array[num_ring]**2
        data_copy = np.copy(data)

        # mask the central region of the PSF and add pixels
        data_copy[mask_circle] = np.nan
        sum_pixels_masked = np.nansum(data_copy)

        ratio_exterior_measured = sum_pixels_masked / sum_pixels_unmasked
        ratio_exterior_measured_array.append(ratio_exterior_measured) # append to array

        ratio_exterior_expected = exterior_fraction[num_ring]

        print('Fraction of irradiance measured exterior to radius: {ratio_exterior_measured:.4f}')
        print('Fraction of irradiance expected exterior to radius: {ratio_exterior_expected:.4f}')
        print(f'Ratio of exterior pixels measured to expected: {ratio_exterior_measured:.4f} / {ratio_exterior_expected:.4f}')

        # FYI plot
        '''
        plt.clf()
        plt.imshow(data_copy, origin='lower', cmap='gray')
        circle = plt.Circle((x_center, y_center), dark_ring_pix_array[num_ring], color='red', fill=False, linewidth=2)
        plt.gca().add_patch(circle)
        plt.colorbar()
        plt.show()
        plt.close()
        '''

    ratio_exterior_measured_over_expected = np.divide(ratio_exterior_measured_array, exterior_fraction)
    print(f'Net ratio of exterior pixels measured to expected: {ratio_exterior_measured_over_expected:.4f}')

    return 


    '''
    # find the peak irradiance from the PSF 
    peak_irr_inside_psf = np.nanmax(data_masked[mask_circle])
    # find the peak irradiance outside the PSF
    peak_irr_outside_psf = np.nanmax(data_masked[~mask_circle])

    # ratio of peak irradiance
    ratio_irr = peak_irr_inside_psf/peak_irr_outside_psf
    '''

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
    file_name_pinhole = stem + 'stray_light/IMG_OPT_02_image_pinhole_lm_Lp.fits'

    # true scaling relation between arcseconds and pixels (measured, not assumed)
    # value and units may change depending on experimental setup
    print('! ---------- USING A PLACEHOLDER SCALING RELN ---------- !')
    scaling_reln = 0.00679 # [arcsec/pix]

    # check FOV
    #fov(file_names_fov, scaling_reln, dither=False)

    # check plate scale 
    #plate_scale(file_name_grid, scaling_reln)

    # check scattered light
    scattered_light(file_name_pinhole, ps=0.01046) # PS here is just a stand-in


if __name__ == "__main__":
    main()