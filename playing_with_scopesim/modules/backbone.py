import logging
import yaml
import ipdb
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import zoom
from photutils.centroids import centroid_sources, centroid_2dg

from .helpers import fit_gaussian_psf, fit_simmed_psfs, load_config_and_pipe
from .strehl_fcns import strehl_from_annular_aperture_fixed, fit_annular_aperture_free_parameters


def strehl_psfs(file_name, 
                fp_mask, 
                pp_mask, 
                filter_name=None, 
                fit_simmed_psf=False, 
                fit_annular_aperture_free=False, 
                fit_annular_aperture_fixed=False, 
                psfs_subset='all', 
                config_coords_guesses_file_name=None, 
                config_observing=None,
                fit_method='curve_fit'):
    '''
    Find the Strehl ratio of a grid of PSFs
    
    INPUTS:
    file_name: name of the file containing the grid of PSFs
    fp_mask: focal plane mask (string)
    pp_mask: pupil plane mask (string)
    filter_name: name of the observing filter
    fit_simmed_psf: whether to fit a ScopeSim-simulated PSF
    fit_analytical_psf: whether to fit an analytical PSF
    psfs_subset: 'all' to process all PSFs, or an integer to process only the first N PSFs
    config_coords_guesses_file_name: name of the configuration file containing the coordinates of the PSFs
    config_observing: name of the configuration file containing the observing parameters


    OUTPUTS:
    None; writes out plots and data
    '''

    #logging.info(f'Processing \n\tdata file: {file_name} \n\tfilter: {filter_name} \n\tconfig file: {config_file_name}')

    # return the locations and other data for each PSF

    grid_frame = fits.open(file_name)
    #ipdb.set_trace(context=10)
    grid_data = grid_frame[1].data
    grid_header = grid_frame[1].header


    # read in coordinate guesses
    config_coords_guesses_config = load_config_and_pipe(config_file_choice=config_coords_guesses_file_name, print_one_line=False)
    coords_entries = config_coords_guesses_config.get("psf_coordinate_guesses", [])
    coords_guesses_all = np.array([(entry["y"], entry["x"]) for entry in coords_entries])
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
    logging.info(f'PSF oversampling factor: {oversample_factor}')
    # Step 1: Oversample the PSFs by a factor of 4 using bicubic interpolation
    grid_data_oversamp = zoom(grid_data, oversample_factor, order=3)
    #psf_simmed_oversamp = zoom(psf_simmed, oversample_factor, order=3)
    coords_guesses_x_all_oversamp = coords_guesses_x_all * oversample_factor
    coords_guesses_y_all_oversamp = coords_guesses_y_all * oversample_factor
    coords_guesses_all_oversamp = np.vstack((coords_guesses_y_all_oversamp, coords_guesses_x_all_oversamp)).T

    # find the PSF centroids, first pass
    logging.info('Finding PSF centroids, first pass')
    x_pos_pix_oversamp, y_pos_pix_oversamp = centroid_sources(grid_data_oversamp, 
                                    xpos=coords_guesses_x_all_oversamp, 
                                    ypos=coords_guesses_y_all_oversamp, 
                                    box_size=41,
                                    centroid_func=centroid_2dg)

    ## ## TODO: Make a FYI plot of this

    # zip into one array
    coords_centroided_all_oversamp = np.vstack((y_pos_pix_oversamp, x_pos_pix_oversamp)).T

    # FYI
    #fyi_plot_centroiding(grid_data_oversamp, coords_centroided_all_oversamp, title_string="PSF first guesses", zscale=False)

    # make a cut-out of each psf and make a best-fit 2D Gaussian
    raw_cutout_size = 20 * oversample_factor
    logging.info(f'Raw PSF cutout size: {raw_cutout_size}')
    num_coord = 0

    #cookie_cut_out_best_fit_list = []
    coord_x_array = np.zeros(len(y_pos_pix_oversamp))
    coord_y_array = np.zeros(len(y_pos_pix_oversamp))
    fwhm_x_pix_array = np.zeros(len(y_pos_pix_oversamp))
    fwhm_y_pix_array = np.zeros(len(y_pos_pix_oversamp))
    sigma_x_pix_array = np.zeros(len(y_pos_pix_oversamp))
    sigma_y_pix_array = np.zeros(len(y_pos_pix_oversamp))
    angle_theta_array = np.zeros(len(y_pos_pix_oversamp))
    amplitude_counts_array = np.zeros(len(y_pos_pix_oversamp))
    gaussian_based_strehl_array = np.zeros(len(y_pos_pix_oversamp))

    # make a copy from which we will subtract the PSFs to see the residuals
    canvas_grid_data = np.copy(grid_data)

    # Determine how many PSFs to process based on psfs_subset parameter
    total_psfs = len(y_pos_pix_oversamp)
    logging.info(f'Total PSFs: {total_psfs}')
    if psfs_subset == 'all':
        num_psfs_to_process = total_psfs
        logging.info(f'Processing all {total_psfs} PSFs')
    elif isinstance(psfs_subset, int):
        num_psfs_to_process = min(psfs_subset, total_psfs)  # Don't exceed available PSFs
        logging.info(f'Processing {num_psfs_to_process} out of {total_psfs} PSFs')
    else:
        logging.error(f"psfs_subset must be 'all' or an integer, got {psfs_subset}")
        raise ValueError(f"psfs_subset must be 'all' or an integer, got {psfs_subset}")
    
    logging.info(f"Processing {num_psfs_to_process} out of {total_psfs} PSFs")

    # loop over each centroided PSF
    for num_coord in range(num_psfs_to_process):
        logging.info(f'Processing PSF {num_coord} of {num_psfs_to_process}')

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
        plot_filename = f"junk_cookie_cut_out_sci_oversamp_{num_coord}.png"
        plt.savefig(f"figs_dump/{plot_filename}", bbox_inches="tight")
        plt.close()
        logging.info(f"Saved {plot_filename} to figs_dump/")

        # Adjust the centroid coordinate for the cut-out: subtract the cutout starting indices to get cutout-relative coordinates
        coords_guess_this_cutout = np.array([
            coords_centroided_all_oversamp[num_coord][0] - idx_y_start,
            coords_centroided_all_oversamp[num_coord][1] - idx_x_start
        ])


        # find FWHM, PSF coords (second-pass fit), and Strehl based on Gaussian
        # (note the Strehl here is bogus, because )
        logging.info(f'Fitting Gaussian to PSF {num_coord} of {num_psfs_to_process}')
        x_center_pix_gaussian_best_fit_oversamp, y_center_pix_gaussian_best_fit_oversamp, fwhm_x_pix_gaussian_best_fit_oversamp, fwhm_y_pix_gaussian_best_fit_oversamp, amplitude_counts_gaussian_best_fit_oversamp, gaussian_based_strehl = fit_gaussian_psf(cookie_cut_out_sci_oversamp, 
                                                                                                        obs_filter=filter_name,
                                                                                                        fp_mask=fp_mask,
                                                                                                        pp_mask=pp_mask,
                                                                                                        coords_guess=coords_guess_this_cutout, 
                                                                                                        plot_string=f'num_coord_{num_coord}_fpmask_{fp_mask}_ppmask_{pp_mask}_filter_{filter_name}', 
                                                                                                        fac_oversamp=oversample_factor)
        # convert the coordinates of the cutout back to those of the entire oversampled image
        x_center_pix_gaussian_best_fit_oversamp_fullarray = x_center_pix_gaussian_best_fit_oversamp + idx_x_start
        y_center_pix_gaussian_best_fit_oversamp_fullarray = y_center_pix_gaussian_best_fit_oversamp + idx_y_start

        # make a best fit based on Airy function
        '''
        if fit_airy_psf:
            # return dict of Strehl ratio
            strehl_airy = fit_airy_psf(cookie_cut_out_sci_oversamp, 
                                        obs_filter=filter_name,
                                        x_center_pix_gaussian_best_fit_oversamp=x_center_pix_gaussian_best_fit_oversamp, 
                                        y_center_pix_gaussian_best_fit_oversamp=y_center_pix_gaussian_best_fit_oversamp, 
                                        fac_oversamp=oversample_factor,
                                        config_observing=config_observing,
                                        plot_string=f'num_coord_{num_coord}_fpmask_{fp_mask}_ppmask_{pp_mask}_filter_{filter_name}')
        '''


        # find FWHM of empirical 
        '''
        fwhm_y_pix_empirical, fwhm_x_pix_empirical = fit_empirical_fwhm(cookie_cut_out_sci, plot_string=f'num_coord_{num_coord}')
        '''

 

        # fit a ScopeSim PSF
        if fit_simmed_psf:
            logging.info(f'Fitting ScopeSim PSF {num_coord} of {num_psfs_to_process}')
            # return 2D array of ScopeSim best-fit
            best_fit_cutout_oversamp = fit_simmed_psfs(cookie_cut_out_sci_oversamp, 
                                            plot_string=f'num_coord_{num_coord}_fpmask_{fp_mask}_ppmask_{pp_mask}_filter_{filter_name}', 
                                            obs_filter=filter_name,
                                            fp_mask=fp_mask,
                                            pp_mask=pp_mask,
                                            x_center_final_oversamp=x_pos_pix_oversamp[num_coord], 
                                            y_center_final_oversamp=y_pos_pix_oversamp[num_coord], 
                                            fac_oversamp=oversample_factor)

        # strehl from an analytical PSF with fixed parameters: D_aperture, D_obscuration, and ampl
        if fit_annular_aperture_fixed:
            logging.info(f'Calculating Strehl from annular aperture {num_coord} of {num_psfs_to_process}')
            # return dict of Strehl ratios found with different methods
            strehl_annular_aperture_fixed = strehl_from_annular_aperture_fixed(cookie_cut_out_sci_oversamp, 
                                            filter_name=filter_name,
                                            plot_string=f'num_coord_{num_coord}_fpmask_{fp_mask}_ppmask_{pp_mask}_filter_{filter_name}', 
                                            x_center_final_cookie_oversamp=x_center_pix_gaussian_best_fit_oversamp, 
                                            y_center_final_cookie_oversamp=y_center_pix_gaussian_best_fit_oversamp, 
                                            config_observing=config_observing,
                                            fac_oversamp=oversample_factor, 
                                            polychromatic=True)

        # fit an analytical PSF: free parameters are D_aperture, D_obscuration, and ampl
        if fit_annular_aperture_free:
            logging.info(f'Fitting analytical PSF {num_coord} of {num_psfs_to_process}')
            strehl_annular_aperture_free = fit_annular_aperture_free_parameters(cookie_cut_out_sci_oversamp, 
                                            filter_name=filter_name,
                                            plot_string=f'num_coord_{num_coord}_fpmask_{fp_mask}_ppmask_{pp_mask}_filter_{filter_name}', 
                                            x_center_final_cookie_oversamp=x_center_pix_gaussian_best_fit_oversamp, 
                                            y_center_final_cookie_oversamp=y_center_pix_gaussian_best_fit_oversamp, 
                                            config_observing=config_observing,
                                            fac_oversamp=oversample_factor, 
                                            fit_method=fit_method)


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
        #cookie_cut_out_best_fit_list.append(best_fit_cutout_oversamp)

        coord_x_array[num_coord] = x_center_pix_gaussian_best_fit_normsamp
        coord_y_array[num_coord] = y_center_pix_gaussian_best_fit_normsamp
        fwhm_x_pix_array[num_coord] = fwhm_x_pix_gaussian_best_fit_normsamp
        fwhm_y_pix_array[num_coord] = fwhm_y_pix_gaussian_best_fit_normsamp
        amplitude_counts_array[num_coord] = amplitude_counts_gaussian_best_fit_oversamp # note the amplitude doesn't need to be resampled
        gaussian_based_strehl_array[num_coord] = gaussian_based_strehl
        #sigma_x_pix_array[num_coord] = sigma_x_pix
        #sigma_y_pix_array[num_coord] = sigma_y_pix
        #angle_theta_array[num_coord] = angle_theta

    # merge the Strehl dicts and print
    strehl_results_all = {}
    for d in [locals().get('strehl_airy'), locals().get('strehl_annular_aperture_fixed'), locals().get('strehl_annular_aperture_free')]:
        if isinstance(d, dict):
            strehl_results_all.update(d)
    logging.info(f'Strehl results:')
    for k, v in strehl_results_all.items():
        logging.info(f"\t{k}:\t{v:.3f}")

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
            f'x: {fwhm_x_pix_array[num_coord]:.2f}, \n y: {fwhm_y_pix_array[num_coord]:.2f}, \n theta: {angle_theta_array[num_coord]:.2f}, \n amp: {amplitude_counts_array[num_coord]:.2f}, \n strehl: {gaussian_based_strehl_array[num_coord]:.2f}',
            color='k',
            fontsize=7, rotation=20
        )
    plt.title('FWHM in x and y (pix), amplitude (counts)')
    plot_file_name = f"fyi_plot_fwhm_and_amp.png"
    plt.savefig(f"figs_dump/{plot_file_name}", bbox_inches="tight")
    logging.info(f"Saved {plot_file_name} to figs_dump/")
    plt.close()


    return #strehl_results_all