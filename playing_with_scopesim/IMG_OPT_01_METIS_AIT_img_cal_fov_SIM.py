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
import logging
import datetime
import io
import os
import sys
import yaml

import scopesim as sim

from modules.helpers import pipe_2_log, setup_logging, load_config_and_pipe #, read_observing_configurations


def read_simulation_configurations(simulation_config_file):
    # filters, etc. to set for each simulated data file
    with open(simulation_config_file, 'r') as f:
        simulation_configs = yaml.load(f, Loader=yaml.FullLoader)
    return simulation_configs


def main():

    stem = '/podman-share/metis_work/playing_with_scopesim/'

    # config file with the observing parameters
    observing_config_file = stem + 'config/config_file_IMG_OPT_01_METIS_AIT_img_cal_fov_SIM_observing.yaml'

    now = datetime.datetime.now()
    log_dir = stem + 'IMG_OPT_01_METIS_AIT_img_cal_fov_SIM_logs/'
    log_file_name = log_dir + 'log_IMG_OPT_01_METIS_AIT_img_cal_fov_SIM_' + now.strftime('%Y-%m-%d_%H-%M-%S') + '.txt'
    #out_dir = stem + 'IMG_OPT_01_METIS_AIT_img_cal_fov_SIM_data/' # directory to write the simulated data to
    
    # initialize logging
    #log_dir = stem + 'IMG_03_logs/'
    #log_file_name = log_dir + 'log_IMG_03_analysis_psf_image_quality_' + now.strftime('%Y-%m-%d_%H-%M-%S') + '.txt'    
    setup_logging(log_dir=log_dir, log_file_name=log_file_name, now=now)

    pipe_2_log(lambda: sim.bug_report(), msg="ScopeSim bug report")

    # Edit this path if you have a custom install directory, otherwise comment it out. [For ReadTheDocs only]
    sim.link_irdb("../../../")

    # simulate observations with METIS (comment this out if packages already exist)
    # sim.download_packages(["METIS", "ELT", "Armazones"])

    '''
    # for creating permutations of all observing configurations

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
    '''

    # read in the observing configurations
    #dict_config = read_simulation_configurations(observing_config_file)

    ipdb.set_trace()
    ## ## QUESTION: IS wcu_img_lm THE RIGHT MODE FOR N FILTERS?
    ## ## QUESTION: DO WE REALLY NEED THIS MANY PARAMETERS?

    # read in the simulation configurations
    #instrument_configs = read_simulation_configurations(simulation_config_file = stem + 'config/config_file_IMG_01_METIS_AIT_img_cal_fov_SIM.yaml')

    sim_states_config_file = stem + 'config/config_file_IMG_01_METIS_AIT_img_cal_fov_SIM.yaml'
    sim_states_config = load_config_and_pipe(config_file_choice=sim_states_config_file, print_one_line=False)
    defaults = sim_states_config.get("defaults", {})
    runs = sim_states_config.get("simulation_configs", [])

    # to set up the data states, merge config data state defaults with overrides that are specific for each run
    sim_states = []
    for entry in runs:
        merged = {**defaults, **entry}
        # Prepend stem to relative file paths
        fname = merged["file_name"]
        if not fname.startswith("/"):
            merged["file_name"] = stem + fname
        results_write_dir = merged["results_write_dir"]
        if not results_write_dir.startswith("/"):
            merged["results_write_dir"] = stem + results_write_dir
        sim_states.append(merged)


    ipdb.set_trace()
    # dictionary of all observing configurations
    logging.info('Number of observing configurations: ' + str(len(sim_states)))
    for idx, config in enumerate(sim_states):
        obs_name = f"obs{idx}"
        logging.info(f"\n{obs_name}: {config}")
        ipdb.set_trace()

    # fpmasks_list = ["open", "pinhole_lm", "pinhole_n", "grid_lm"]


    # TODO: ADD other config.
    #cmd_2 = sim.UserCommands(use_instrument="METIS", set_modes=["img_lm"],
    #                    properties={"!OBS.filter_name": "Mp", "!OBS.exptime": 100., "!DET.dit": 200})

    # loop over each configuration
    for config_params in sim_states:
        # take exposures for 
        # each filter in filter_list = ["Mp", "Lp"]
        # each fpmasks_list = ["open", "pinhole_lm", "pinhole_n", "grid_lm"]

        ## ## CONTINUE HERE: ADD DITHERING
        dither_positions = [0]

        for pos_dither in range(len(dither_positions)):

            ipdb.set_trace()

            logging.info('--------------------------------')
            logging.info('Running config: ' + str(config_params))
            logging.info('--------------------------------')

            obs_mode = config_params['obs_mode']
            ndit = config_params['ndit']
            dit = config_params['dit']
            obs_filter = config_params['obs_filter_file_name']
            obs_filter_name = config_params['obs_filter_name']
            fp_mask = config_params['fpmask']
            pp_mask = config_params['pp_mask']
            nd_filter = config_params.get('nd_filter')
            exptime = config_params.get('exptime')
            use_exp_time_only = config_params.get('use_exp_time_only', False)
            out_dir = config_params['results_write_dir']
            file_name = config_params['file_name']

            # set up instrument for imaging (same property keys as IMG_OPT_03)
            ipdb.set_trace()
            if nd_filter is not None:
                cmd = sim.UserCommands(
                    use_instrument='METIS',
                    set_modes=[obs_mode],
                    properties={
                        "!OBS.filter_name": obs_filter_name,
                        "!WCU.current_fpmask": fp_mask,
                        "!OBS.pupil_mask": pp_mask,
                        "!OBS.nd_filter_name": nd_filter,
                    },
                )
            else:
                cmd = sim.UserCommands(
                    use_instrument='METIS',
                    set_modes=[obs_mode],
                    properties={
                        "!OBS.filter_name": obs_filter_name,
                        "!WCU.current_fpmask": fp_mask,
                        "!OBS.pupil_mask": pp_mask,
                    },
                )

            metis = sim.OpticalTrain(cmd)
            metis['pupil_masks'].change_mask(pp_mask)
            logging.info('OBS filter: ' + str(metis.cmds.get("!OBS.filter_name")))
            logging.info('WCU FP mask: ' + str(metis.cmds.get("!WCU.current_fpmask")))
            logging.info('OBS PP mask: ' + str(metis.cmds.get("!OBS.pupil_mask")))
            logging.info('OBS ND filter: ' + str(metis.cmds.get("!OBS.nd_filter_name")))

            pipe_2_log(lambda m=metis: m.effects.pprint_all(), msg="Optical train effects (initial state)")
            wcu = metis['wcu_source']

            #########################################################
            # Set the WCU Flux Controlling Mask to "CLOSED" (redundant?)
            closed_value = 0.
            logging.info(f'Setting the WCU BB aperture to {closed_value}')
            wcu.set_bb_aperture(value = float(closed_value))
            logging.info(f'wcu.bb_aperture: {wcu.bb_aperture}')

            #########################################################
            # Set the WCU BB source to 1000 K.
            bb_temp = 1000 * u.K
            logging.info(f'Setting the WCU BB temperature to {bb_temp}.')
            wcu.set_temperature(bb_temp=bb_temp)
            logging.info(f'wcu.bb_temp: {wcu.bb_temp}')

            #########################################################
            # Wait for BB source to reach temperature.

            # placeholder in lieu of a thermal model
            logging.info('Waiting for the WCU BB source to reach temperature.')
            time.sleep(0.5)

            #########################################################
            # While BB reaches temperature, take background exposure
            logging.info('Taking background exposure.')
            pipe_2_log(lambda m=metis: m.effects.pprint_all(), msg="Optical train effects (for background)")
            # see current observing params
            logging.info("All OBS parameters:")
            for key, value in cmd['OBS'].items():
                logging.info(f"  {key}: {value}")

            # compile the observation
            ipdb.set_trace()
            logging.info('Compiling the observation.')
            metis.observe()

            # do readout with observation params
            logging.info('Getting readout.')
            if use_exp_time_only:
                outhdul_off = metis.readout(exptime=exptime, reset=False)[0]
            else:
                outhdul_off = metis.readout(ndit=ndit, dit=dit, reset=False)[0]
            outhdul_off.info()

            # save to FITS file (for debugging)
            '''
            outhdul_off.writeto('junk.fits', overwrite=True)
            file_name1 = 'test1.png'
            plt.clf()
            plt.imshow(outhdul[1].data, origin='lower')
            plt.title(config_params)
            plt.show()
            ipdb.set_trace()
            #plt.savefig('/podman-share/' + file_name1)
            #logging.info('Saved ' + file_name1)
            '''

            '''
            file_name2 = 'test2.png'
            plt.clf()
            plt.hist(outhdul_off[1].data.ravel(), bins=200)
            plt.title('Counts in background exposure\n' + config_params)
            plt.show()
            #plt.savefig('/podman-share/' + file_name2)
            #logging.info('Saved ' + file_name2)
            '''

            #########################################################
            # Set the WCU Flux Controlling Mask to "OPEN".
            logging.info('Setting the wcu bb aperture to OPEN')
            wcu.set_bb_aperture(value = 1.)

            #########################################################
            # Take science exposure with same params as background
            logging.info('Taking science exposure with same params as background.')
            pipe_2_log(lambda m=metis: m.effects.pprint_all(), msg="Optical train effects (for science exposure)")
            # recompile
            metis.observe()
            # get the readout
            if use_exp_time_only:
                outhdul_on = metis.readout(exptime=exptime, reset=False)[0]
            else:
                outhdul_on = metis.readout(ndit=ndit, dit=dit, reset=False)[0]
            logging.info('Science exposure readout.')

            #########################################################
            # Background-subtract

            background = outhdul_off[1].data
            # background-subtract
            raw_sci_readout = outhdul_on[1].data
            bckgd_subted = raw_sci_readout - background
            
            # write
            abs_file_name_write = file_name
            os.makedirs(os.path.dirname(abs_file_name_write) or '.', exist_ok=True)


            # Copy the primary header
            primary_hdu = fits.PrimaryHDU(header=outhdul_on[0].header)
            # Add background-subtracted readout as first extension
            hdu_bckgd_subted = fits.ImageHDU(data=bckgd_subted, name='BCKGD_SUBTED')
            # Add raw science readout as second extension
            hdu_raw_readout = fits.ImageHDU(data=raw_sci_readout, name='RAW_READOUT')
            # Add background as third extension
            hdu_background = fits.ImageHDU(data=background, name='BACKGROUND')
            hdul_new = fits.HDUList([primary_hdu, hdu_bckgd_subted, hdu_raw_readout, hdu_background])

            # add some stuff to the header, some of which may be redundant
            hdul_new[0].header['FILTER'] = (obs_filter, 'Observing filter')
            hdul_new[0].header['WCU_FP'] = (fp_mask, 'WCU focal plane mask')
            hdul_new[0].header['WCU_PP'] = (pp_mask, 'WCU pupil plane mask')
            hdul_new[0].header['BB_TEMP'] = (bb_temp.value, 'BB temperature')
            if ndit is not None:
                hdul_new[0].header['NDIT'] = (ndit, 'Number of dithered exposures')
                hdul_new[0].header['DIT'] = (dit, 'Det integration time')
            else:
                hdul_new[0].header['EXPTIME'] = (exptime, 'Exposure time')
            
            hdul_new.writeto(abs_file_name_write, overwrite=True)
            logging.info('Saved background-subtracted readout without aberrations to ' + abs_file_name_write)

            ipdb.set_trace()

            # display (for debugging)
            '''
            plt.clf()
            plt.hist(outhdul[1].data.ravel(), bins=200)
            plt.title('Counts in science exposure\n' + config_params)
            plt.show()
            ipdb.set_trace()
            '''


if __name__ == "__main__":
    main()