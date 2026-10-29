"""
Make simulated data for the IMG-OPT-03 PSF image quality test

Reqs.:
- Ref. Overleaf doc IMG_OPT_03_Test_Description_PSF_Image_Quality

1. METIS-1408: Quality and alignment of the optical components within Mid-infrared ELT
Imager and Spectrograph (METIS) shall provide diffraction limited performance
(Strehl >= 80 %) at lambda > 3um in all modes over the entire FOV.
2. METIS-1409: The Instrument Wavefront Error (WFE) shall satisfy the diffraction limit
requirement (Strehl>0.8) at lambda = 3 um for IMG (both LM and NQ) and IMG. The minimum
RMS WFE below shall be satisfied over the full Field Of View (FOV) relevant to the
given optical path.
3. METIS-2864: The minimum Strehl ratio of the WCU+CFO+IMG-LM optical path shall be >80%
at 3.3um over the entire field of view.
4. METIS-3503: METIS shall be able to characterise the shape of the instrument PSF
across the entire FoV using the WCU. (Not simulated: no numeric limit.)

The procedure for measuring the image quality follows that of IMG-OPT-01, but with
additional exposures made at positions designed to obtain fully spatially sampled
imaging. A basic offset of half a pixel is used for the PSF image quality measurements
in this test in horizontal, vertical, and diagonal directions. We also need to achieve
a significantly higher SNR for accurate measurement of the FWHM, and better control of
calibration and flat fielding errors for measurement of the Encircled Energy. The flat
field will be derived from IMG-RAD-(TBD).

Structure: one function per requirement, driven by the scenario in
config/config_file_IMG_OPT_03_SIM.yaml and run by modules.sim_common.run_scenario. Each
writes a case that clearly passes and one that clearly fails (and one near the limit,
if configured), named IMG_OPT_03_<requirement>_<instrument>_<case>.fits, with the
requirement and the injected truth in the primary header (REQ_ID, CASE, EXPECTED,
REQ_QTY, REQ_LIM, PASS_IF, INJ_VAL, REQ_PLH, SIMVER; see modules/sim_common.py). The
scenario entry TABLE-4-1 also writes the unmodified frames of the E-REP-MPIA-1203
Table 4-1 rows (IMG_OPT_03_TABLE-4-1_<instrument>_reference.fits).

The Strehl ratio is injected by moving part of the PSF flux into a halo (see
inject_strehl), applied to the signal (illuminated frame - dark_1); BACKGROUND is an
independent dark.
"""

import datetime
import logging

import numpy as np
import scopesim as sim
from scipy.ndimage import gaussian_filter

from modules import sim_common
from modules.config_common import load_observing_config
from modules.helpers import load_config_and_pipe
from pipeline_registry import (  # pylint: disable=import-error,wrong-import-order
    CLUSTER_PSF,
    COLOR_PSF,
    pipeline_stage,
)

TEST_PREFIX = "IMG_OPT_03"


def inject_strehl(signal, strehl_factor, halo_sigma_pix):
    """
    Lower the Strehl ratio of the PSFs in signal by strehl_factor (<= 1), conserving
    the flux: signal' = s * signal + (1 - s) * halo, with halo = signal convolved with
    a normalised Gaussian of sigma halo_sigma_pix, and s chosen so that the brightest
    PSF's peak is strehl_factor times what it was (the others' approximately so).

    OUTPUTS:
    - ndarray: signal'
    - float: s
    """
    if not 0.0 < strehl_factor <= 1.0:
        raise ValueError(f"strehl_factor must be in (0, 1], got {strehl_factor}")
    halo = gaussian_filter(signal, sigma=halo_sigma_pix, mode="constant")
    peak_yx = np.unravel_index(np.argmax(signal), signal.shape)
    peak, halo_at_peak = float(signal[peak_yx]), float(halo[peak_yx])
    s = (strehl_factor * peak - halo_at_peak) / (peak - halo_at_peak)
    if s < 0.0:
        raise ValueError(
            f"Cannot lower the peak by {strehl_factor}: the halo alone is "
            f"{halo_at_peak / peak:.3f} of the peak; use a wider halo"
        )
    return s * signal + (1.0 - s) * halo, s


@pipeline_stage(
    name="sim_img_03_strehl_cases",
    cluster=CLUSTER_PSF,
    cluster_color=COLOR_PSF,
    label="SIM METIS-1408/-1409/-2864: Strehl cases",
)
def simulate_strehl(ctx, req_key, req_cfg):
    """
    Simulated data for a Strehl requirement (METIS-1408/-1409, METIS-2864: Strehl >=
    0.8).

    Per case, the illuminated ScopeSim frame with the Strehl ratio lowered by
    injected strehl / reference_strehl (see inject_strehl and the scenario config).

    OUTPUTS:
    - list[str]: written files
    """
    expected = sim_common.check_cases(req_key, req_cfg)
    dark_1, _ = ctx.darks()
    flat, _ = ctx.lit_frame()
    signal = flat - dark_1
    cfg = req_cfg["strehl_injection"]
    reference_strehl = float(cfg["reference_strehl"][ctx.det_key])
    band = ctx.det_key.split("_")[-1]  # 'lm' or 'n'
    halo_sigma_pix = float(cfg["halo_sigma_frac_cutout"]) * float(
        ctx.observing["cutout_edge_native"][band]
    )
    logging.info(
        "%s: reference Strehl %.3f; halo sigma %.2f pix",
        req_key,
        reference_strehl,
        halo_sigma_pix,
    )

    written = []
    for case, case_cfg in req_cfg["cases"].items():
        inject = case_cfg["inject"]
        factor = float(inject["strehl"]) / reference_strehl
        signal_new, s = inject_strehl(signal, factor, halo_sigma_pix)
        logging.info(
            "%s case %s: Strehl %.3f -> peak factor %.3f, s = %.4f",
            req_key,
            case,
            float(inject["strehl"]),
            factor,
            s,
        )
        written.append(
            ctx.write_case(
                req_key,
                case,
                "",
                dark_1 + signal_new,
                sim_common.requirement_cards(
                    req_key, req_cfg, case, inject, expected[case], ctx.simver
                )
                + [
                    ("STREHREF", reference_strehl, "Reference Strehl of ScopeSim PSFs"),
                    ("HALO_SIG", halo_sigma_pix, "Sigma of injected halo (pix)"),
                    ("HALO_S", s, "Fraction of signal left in the PSF core"),
                ],
            )
        )
    return written


@pipeline_stage(
    name="sim_img_03_reference_frames",
    cluster=CLUSTER_PSF,
    cluster_color=COLOR_PSF,
    label="SIM Table 4-1 reference frames (no injection)",
)
def write_reference_frame(ctx, req_key, req_cfg):
    """
    The unmodified illuminated ScopeSim frame (no requirement, nothing injected), as
    <prefix>_<req_key>_<instrument>_reference.fits.

    OUTPUTS:
    - list[str]: written files
    """
    del req_cfg  # no limit or cases
    flat, _ = ctx.lit_frame()
    return [
        ctx.write_case(
            req_key,
            "reference",
            "",
            flat,
            [
                ("REQ_ID", "none", "No requirement simulated"),
                ("CASE", "reference", "Unmodified ScopeSim frame"),
                ("SIMVER", ctx.simver, "ScopeSim, IRDB, code versions"),
            ],
        )
    ]


# requirement key (in the scenario config) -> function writing its simulated data
SIMULATORS = {
    "METIS-1408-1409": simulate_strehl,
    "METIS-2864": simulate_strehl,
    "TABLE-4-1": write_reference_frame,
}


@pipeline_stage(
    name="generate_psf_quality_sim",
    depends_on=("sim_img_03_strehl_cases", "sim_img_03_reference_frames"),
    cluster=CLUSTER_PSF,
    cluster_color=COLOR_PSF,
    label="Generate PSF-quality sims",
)
def main():
    """
    Generate IMG-OPT-03 PSF image quality simulations, one set per requirement, as set
    in the scenario config.
    """
    stem = "/podman-share/metis_work/METIS_MAIT_code_eckhart/"
    out_dir = stem + "data/IMG_OPT_03_psf_quality_simmed_data/"

    now = datetime.datetime.now()
    sim_common.start_sim_logging(
        stem, "IMG_03_logs", "IMG_03_simulation_psf_image_quality", now
    )
    logging.info("Simmed file output directory: %s", out_dir)
    # uncomment to (re-)download the IRDB packages (overwrites them in the IRDB)
    # sim.download_packages(
    #     ["METIS", "ELT", "Armazones"], save_dir=sim_common.IRDB_REL_PATH
    # )
    sim_common.setup_scopesim()

    observing = load_observing_config(stem)
    scenario = load_config_and_pipe(
        config_file_choice=stem + "config/config_file_IMG_OPT_03_SIM.yaml",
        print_one_line=False,
    )
    sim_common.run_scenario(scenario, SIMULATORS, observing, out_dir, TEST_PREFIX)


if __name__ == "__main__":
    main()
