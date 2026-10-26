"""
Make simulated data for the IMG-OPT-02 image distorition test

Reqs.:
- Ref. Overleaf doc IMG_OPT_02_Test_Geometric_Distortion

1. METIS-1097: The Imager shall provide a pixel scale of 5.47 +0.26/-0.26 mas/pix for
   the LM-band and
              6.79+0.25/-0.50 mas/pix for the N-band.
METIS-3502: After calibration, the distortions introduced by METIS shall be removed to
better than
              0.5 mas (ca. 1/10 px for the L band imager) over the full field of view.
METIS-8222: The center of the H2RG chip within IMG-LM-DET shall be offset from the METIS
optical axis
              by 175 mas ± 25 mas on-sky PtV in the "across H2RG stripe" direction
              (i.e., perpendicularly
              to the orientation of the 32 stripes in the H2RG detector).
METIS-9920: Image scale and distortion of METIS shall be constant (for each optical
configuration, even after
              change of observing modes) to an accuracy of 10-3 (goal: 10-4) at L/M-band
              and 2×10-3 (goal: 2×10-4)
              at N-band with respect to the full field of view.

No requirement cases are simulated for this test (yet): one plain frame per observing
configuration, with the OpticalTrain and readouts from modules/sim_common.py.
"""

import datetime
import logging
import os

import numpy as np
from astropy.io import fits

from modules import sim_common
from pipeline_registry import (  # pylint: disable=import-error,wrong-import-order
    CLUSTER_DISTORTION,
    COLOR_DISTORTION,
    pipeline_stage,
)

# LM filters
# dict_keys(['open', 'Lp', 'short-L', 'L_spec', 'Mp', 'M_spec', 'Br_alpha',
# 'Br_alpha_ref', 'PAH_3.3', 'PAH_3.3_ref', 'CO_1-0_ice', 'CO_ref', 'H2O-ice',
# 'IB_4.05', 'HCI_L_short', 'HCI_L_long', 'HCI_M'])
# (filter, ND filter, DIT, NDIT); all with grid_lm, PPS-CFO2, wcu_img_lm
LM_OBS_CONFIGS = [
    ("Br_alpha", None, 0.065, 2),
    ("Br_alpha_ref", "ND_OD1", 0.4, 5),
    ("Lp", "ND_OD2", 3 / 8, 3),
    ("H2O-ice", "ND_OD1", 0.06, 1),
    ("short-L", "ND_OD2", 0.375, 4),
    ("PAH_3.3", "ND_OD1", 0.1875, 7),
    ("PAH_3.3_ref", "ND_OD1", 0.1875, 8),
    ("IB_4.05", "ND_OD1", 1 / 6, 12),
    ("HCI_L_short", "ND_OD2", 5 / 6, 3),
    ("HCI_L_long", "ND_OD1", 0.06, 25),
    ("Mp", "ND_OD2", 0.5, 5),
    ("CO_1-0_ice", "ND_OD1", 2 / 21, 21),
    ("CO_ref", "ND_OD1", 2 / 21, 21),
    ("HCI_M", "ND_OD1", 1 / 11, 22),
    ("L_spec", "ND_OD2", 0.12, 11),
    ("M_spec", "ND_OD2", 2 / 7, 7),
]


@pipeline_stage(
    name="generate_distortion_sim",
    cluster=CLUSTER_DISTORTION,
    cluster_color=COLOR_DISTORTION,
    label="Generate distortion sims",
)
def generate_distortion_data(instrument, out_dir):
    """
    Generate simulated data for the IMG-OPT-02 image distortion test: a background and
    an illuminated frame for one WCU set-up (see sim_common.build_optical_train).

    Writes IMG_OPT_02_wcu_focal_mask_bckgrnd_subted_<fp_mask>_pupil_mask_<pp_mask>
    _filter_<filter>.fits with BCKGD_SUBTED, RAW_READOUT and BACKGROUND (the layout the
    IMG-OPT-02 ANALYSIS was written for).

    OUTPUTS:
    - str: path of the written file
    """
    metis = sim_common.build_optical_train(instrument)

    logging.info("Closing WCU BB aperture first to get a background ...")
    background, _ = sim_common.readout_frame(metis, 0.0, instrument)
    logging.info("Re-opening WCU BB aperture to get a PSF ...")
    raw_sci_readout, header = sim_common.readout_frame(metis, 1.0, instrument)
    bckgd_subted = raw_sci_readout - background

    hdul_new = fits.HDUList(
        [
            fits.PrimaryHDU(header=header),
            fits.ImageHDU(data=bckgd_subted, name="BCKGD_SUBTED"),
            fits.ImageHDU(data=raw_sci_readout, name="RAW_READOUT"),
            fits.ImageHDU(data=background, name="BACKGROUND"),
        ]
    )
    hdr = hdul_new[0].header
    hdr["FILTER"] = (instrument["obs_filter"], "Observing filter")
    hdr["WCU_FP"] = (instrument["fp_mask"], "WCU focal plane mask")
    hdr["WCU_PP"] = (instrument["pp_mask"], "WCU pupil plane mask")
    hdr["PSF_KERN"] = (sim_common.psf_kernel_of(instrument), "ScopeSim PSF kernel")
    hdr["ND_FILT"] = (str(instrument.get("nd_filter")), "ND filter")
    hdr["BB_TEMP"] = (sim_common.BB_TEMP.value, "BB temperature")
    hdr["NDIT"] = (instrument["ndit"], "Number of DITs")
    hdr["DIT"] = (instrument["dit"], "Det integration time")

    abs_file_name_write = os.path.join(
        out_dir,
        f"IMG_OPT_02_wcu_focal_mask_bckgrnd_subted_{instrument['fp_mask']}"
        f"_pupil_mask_{instrument['pp_mask']}_filter_{instrument['obs_filter']}.fits",
    )
    hdul_new.writeto(abs_file_name_write, overwrite=True)
    logging.info("Saved %s", abs_file_name_write)
    logging.info(
        "Median of raw science readout / background / background-subtracted: "
        "%.4f / %.4f / %.4f",
        np.median(raw_sci_readout),
        np.median(background),
        np.median(bckgd_subted),
    )
    return abs_file_name_write


def main():
    """Simulate the IMG-OPT-02 distortion data for each observing configuration."""

    stem = "/podman-share/metis_work/METIS_MAIT_code_eckhart/"
    # write simulated data here, where the ANALYSIS reads it
    # (config/config_file_IMG_02_strehl_runs.yaml)
    out_dir = stem + "data/IMG_OPT_02_distortion_simmed_data/"
    os.makedirs(out_dir, exist_ok=True)

    now = datetime.datetime.now()
    sim_common.start_sim_logging(
        stem, "IMG_02_logs", "IMG_02_simulation_psf_image_quality", now
    )
    logging.info("Simmed file output directory: %s", out_dir)
    sim_common.setup_scopesim()

    for obs_filter, nd_filter, dit, ndit in LM_OBS_CONFIGS:
        generate_distortion_data(
            {
                "fp_mask": "grid_lm",
                "pp_mask": "PPS-CFO2",
                "obs_filter": obs_filter,
                "nd_filter": nd_filter,
                "obs_mode": "wcu_img_lm",
                "dit": float(dit),
                "ndit": ndit,
            },
            out_dir,
        )


if __name__ == "__main__":
    main()
