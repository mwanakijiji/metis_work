"""
Make simulated data for the IMG-OPT-01 FOV measurement

Reqs.:
- Ref. Overleaf doc IMG_OPT_01_Test_Description_Field_of_View_and_Ghost_Measurement.pdf

1. METIS-1095: The FoV shall be 10.0 +1.0/-0.0 arcsec for the LM-arm and 13.47
+0.50/-0.99 arcsec for the N-arm of the IMG.

!!! TBD MORE DISTORTION INFO FROM DD NEEDED FIRST (warned in the log and on the plots)

Structure: one function per requirement, driven by the scenario in
config/config_file_IMG_OPT_01_METIS_AIT_img_cal_fov_SIM.yaml and run by
modules.sim_common.run_scenario. Each case is four dithers of the WCU pinhole, named
IMG_OPT_01_<requirement>_<instrument>_<case>_dither<N>.fits, with the requirement and
the injected truth in the primary header (REQ_ID, CASE, EXPECTED, REQ_QTY, REQ_LIM,
PASS_IF, INJ_VAL, REQ_PLH, SIMVER; see modules/sim_common.py) and the dither position
(DITHX, DITHY). There is a case that clearly passes, and one that clearly fails on
either side of the FoV range.

The FoV is injected through the dither offsets (see simulate_metis_1095). Each dither is
a ScopeSim readout; BACKGROUND is an independent dark.

FYI plots (results/IMG_OPT_01_fov_analysis_results/fyi_sim_*.png): per case, the four
dithers co-added with the measured pinhole positions, cutouts around each pinhole, and
the FoV the ANALYSIS would measure from those positions vs. the injected one; per
instrument, the dither geometry of all cases. The measured pinhole positions are also
logged as guesses for config_file_IMG_OPT_01_METIS_AIT_img_cal_fov_ANALYSIS.yaml.
"""

import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import scopesim as sim
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter
from skimage.measure import block_reduce

from modules import sim_common
from modules.config_common import load_observing_config
from modules.helpers import load_config_and_pipe
from modules.paths import project_root
from modules.tbd_warnings import add_tbd_banner, log_tbd_distortion
from pipeline_registry import (  # pylint: disable=import-error,wrong-import-order
    CLUSTER_FOV,
    COLOR_FOV,
    pipeline_stage,
)

TEST_PREFIX = "IMG_OPT_01"

# corners of the dither square, in units of dither_offset_arcsec; consecutive dithers
# are adjacent corners, as the ANALYSIS assumes
DITHER_CORNERS = np.array([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])

# FYI plots: binning of the co-added frame, and side of the pinhole cutouts (pixels)
FYI_BIN = 8
FYI_CUTOUT = 41


def dither_positions(dither_offset_arcsec, fov_true, fov_injected):
    """
    Dither positions (x, y) in arcsec, scaled by fov_true / fov_injected so that an
    analysis assuming the nominal offsets measures fov_injected.
    """
    return DITHER_CORNERS * dither_offset_arcsec * (fov_true / fov_injected)


def measure_peak(image, sigma=2.0):
    """(y, x) of the brightest pixel of a lightly smoothed copy of image."""
    smoothed = gaussian_filter(np.nan_to_num(np.asarray(image, dtype=float)), sigma)
    return np.unravel_index(int(np.argmax(smoothed)), smoothed.shape)


def fov_from_peaks(peaks_yx, dither_offset_arcsec, illuminated_pixels):
    """
    FoV (arcsec) as the ANALYSIS (backbone_img_opt_01_fov.fov_calc) derives it from the
    pinhole positions: pixel scale from the separations of consecutive dithers, taken
    to be 2 * dither_offset_arcsec, times the number of illuminated pixels.
    """
    peaks = np.asarray(peaks_yx, dtype=float)
    sides_pix = np.linalg.norm(peaks - np.roll(peaks, -1, axis=0), axis=1)
    return float(np.mean(2.0 * dither_offset_arcsec / sides_pix * illuminated_pixels))


def cutout(image, center_yx, size):
    """size x size cutout centred on center_yx, zero-padded at the frame edges."""
    half = size // 2
    padded = np.pad(np.asarray(image, dtype=float), half, mode="constant")
    y, x = int(center_yx[0]) + half, int(center_yx[1]) + half
    return padded[y - half : y + half + 1, x - half : x + half + 1]


def fyi_plot_case(  # pylint: disable=too-many-arguments,too-many-locals
    results_dir,
    label,
    case_info,
    coadded_binned,
    cutouts,
    illuminated_pixels,
):
    """
    FYI plot of one case: the co-added (binned) dithers with the measured pinhole
    positions and the nominal illuminated area, and a cutout around each pinhole;
    stamped with the TBD distortion warning.

    INPUTS:
    - results_dir (str): where to write fyi_sim_<label>.png
    - label (str): e.g. 'METIS-1095_Br_alpha_pass'
    - case_info (dict): fov_injected, fov_measured, expected, positions (arcsec, (4, 2)),
      peaks_yx (pix, (4, 2)), shape (full-frame (ny, nx))
    - coadded_binned (ndarray): sum of the binned background-subtracted dithers
    - cutouts (list[ndarray]): one cutout per dither, centred on its pinhole
    - illuminated_pixels (int): side of the illuminated area (pixels)

    OUTPUTS:
    - str: path of the written plot
    """
    peaks = np.asarray(case_info["peaks_yx"], dtype=float)
    positions = np.asarray(case_info["positions"], dtype=float)
    ny, nx = case_info["shape"]

    fig = plt.figure(figsize=(16, 8.5))
    grid = fig.add_gridspec(2, 4, width_ratios=[2, 2, 1, 1])
    ax_main = fig.add_subplot(grid[:, :2])
    positive = np.clip(coadded_binned, 0.0, None)
    vmax = float(np.max(positive)) if np.max(positive) > 0 else 1.0
    ax_main.imshow(
        positive + vmax * 1e-5,
        origin="lower",
        cmap="magma",
        norm=LogNorm(vmin=vmax * 1e-4, vmax=vmax),
        extent=(-0.5, nx - 0.5, -0.5, ny - 0.5),
    )
    # nominal illuminated area, centred on the frame
    half = illuminated_pixels / 2.0
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    ax_main.add_patch(
        plt.Rectangle(
            (cx - half, cy - half),
            illuminated_pixels,
            illuminated_pixels,
            fill=False,
            color="cyan",
            linestyle="--",
            label=f"Nominal illuminated area ({illuminated_pixels} pix)",
        )
    )
    closed = np.vstack([peaks, peaks[:1]])
    ax_main.plot(closed[:, 1], closed[:, 0], "w-", linewidth=0.8)
    for i, (peak, pos) in enumerate(zip(peaks, positions)):
        ax_main.plot(peak[1], peak[0], "+", color="lime", markersize=14, mew=2)
        ax_main.annotate(
            f'{i}: ({pos[0]:+.2f}, {pos[1]:+.2f})"\n'
            f"pix ({peak[1]:.0f}, {peak[0]:.0f})",
            (peak[1], peak[0]),
            textcoords="offset points",
            xytext=(8, 8),
            color="white",
            fontsize=8,
        )
    ax_main.set_xlabel("x (pixels)")
    ax_main.set_ylabel("y (pixels)")
    ax_main.legend(loc="lower right", fontsize=8)
    ax_main.set_title(
        f"Dithers co-added (binned {FYI_BIN}x{FYI_BIN}, log stretch)", fontsize=10
    )

    for i, image in enumerate(cutouts):
        ax = fig.add_subplot(grid[i // 2, 2 + i % 2])
        ax.imshow(image, origin="lower", cmap="magma")
        ax.plot(FYI_CUTOUT // 2, FYI_CUTOUT // 2, "+", color="lime", markersize=10)
        ax.set_title(
            f"Dither {i}: pinhole at ({peaks[i][1]:.0f}, {peaks[i][0]:.0f})",
            fontsize=8,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"IMG-OPT-01 SIM {label}: injected FoV {case_info['fov_injected']:.2f}\" "
        f"(expected {case_info['expected']}); FoV from pinhole positions, as the "
        f"ANALYSIS measures it: {case_info['fov_measured']:.2f}\"",
        fontsize=11,
        y=0.95,
    )
    # leave room at the top for the warning banner
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    add_tbd_banner(fig)
    os.makedirs(results_dir, exist_ok=True)
    file_name = os.path.join(results_dir, f"fyi_sim_{label}.png")
    fig.savefig(file_name)
    plt.close(fig)
    logging.info("Saved FYI plot %s", file_name)
    return file_name


def fyi_plot_dither_geometry(
    results_dir, label, dither_offset_arcsec, cases_info, limit_spec
):
    """
    FYI plot of the dither positions (arcsec) of every case of one instrument, with the
    nominal square the ANALYSIS assumes; stamped with the TBD distortion warning.

    INPUTS:
    - results_dir (str): where to write fyi_sim_<label>_dither_geometry.png
    - label (str): e.g. 'METIS-1095_Br_alpha'
    - dither_offset_arcsec (float): nominal dither offset
    - cases_info (dict): case -> {positions, fov_injected, fov_measured, expected}
    - limit_spec (dict): the requirement's FoV limit {value: [lo, hi], ...}

    OUTPUTS:
    - str: path of the written plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8.6))
    nominal = np.vstack([DITHER_CORNERS, DITHER_CORNERS[:1]]) * dither_offset_arcsec
    ax.plot(
        nominal[:, 0],
        nominal[:, 1],
        "k--",
        label=f'Nominal (+/-{dither_offset_arcsec:g}", as the ANALYSIS assumes)',
    )
    for i, (case, info) in enumerate(cases_info.items()):
        pos = np.asarray(info["positions"], dtype=float)
        closed = np.vstack([pos, pos[:1]])
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            "o-",
            color=f"C{i}",
            label=(
                f"{case}: FoV inj. {info['fov_injected']:.2f}\", from pinholes "
                f"{info['fov_measured']:.2f}\" -> {info['expected']}"
            ),
        )
    ax.set_aspect("equal")
    ax.set_xlabel("WCU FP mask shift x (arcsec)")
    ax.set_ylabel("WCU FP mask shift y (arcsec)")
    ax.legend(fontsize=8, loc="center")
    ax.set_title(
        f"IMG-OPT-01 SIM {label}: dither positions per case\n"
        f"(FoV limit {limit_spec['value']} arcsec)",
        fontsize=10,
    )
    # leave room at the top for the warning banner
    fig.subplots_adjust(top=0.88)
    add_tbd_banner(fig)
    os.makedirs(results_dir, exist_ok=True)
    file_name = os.path.join(results_dir, f"fyi_sim_{label}_dither_geometry.png")
    fig.savefig(file_name)
    plt.close(fig)
    logging.info("Saved FYI plot %s", file_name)
    return file_name


@pipeline_stage(
    name="sim_img_01_metis_1095",
    cluster=CLUSTER_FOV,
    cluster_color=COLOR_FOV,
    label="SIM METIS-1095: FoV",
)
def simulate_metis_1095(ctx, req_key, req_cfg):  # pylint: disable=too-many-locals
    """
    Simulated data for METIS-1095: the FoV shall be 10.0 +1.0/-0.0 arcsec (LM) and
    13.47 +0.50/-0.99 arcsec (N).

    The ANALYSIS derives the pixel scale from the pinhole positions in four dithers it
    takes to be 2 * dither_offset_arcsec apart, and the FoV as that pixel scale times
    the number of illuminated pixels. Per case, the dithers are scaled by FoV_true /
    FoV_injected (FoV_true = pixel scale * illuminated pixels), so the FoV it measures
    is the injected one. If ctx.results_dir is set, also writes FYI plots there.

    OUTPUTS:
    - list[str]: written files
    """
    expected = sim_common.check_cases(req_key, req_cfg)
    _, dark_2 = ctx.darks()
    illuminated_pixels = int(req_cfg["illuminated_pixels"])
    fov_true = (
        float(ctx.observing["pixel_scales"][ctx.det_key]) / 1000.0 * illuminated_pixels
    )
    offset = float(req_cfg["dither_offset_arcsec"])
    tag = sim_common.instrument_tag(ctx.instrument)
    logging.info(
        "%s: FoV_true %.3f arcsec; nominal dither offset %.2f arcsec",
        req_key,
        fov_true,
        offset,
    )

    written = []
    cases_info = {}
    for case, case_cfg in req_cfg["cases"].items():
        inject = case_cfg["inject"]
        cards = sim_common.requirement_cards(
            req_key, req_cfg, case, inject, expected[case], ctx.simver
        )
        positions = dither_positions(offset, fov_true, float(inject["fov"]))
        peaks_yx, cutouts, coadded = [], [], None
        for i, pos in enumerate(positions):
            sim_common.set_fpmask_shift(ctx.metis, pos)
            raw, header = sim_common.readout_frame(ctx.metis, 1.0, ctx.instrument)
            written.append(
                sim_common.write_case_file(
                    ctx.out_dir,
                    ctx.test_prefix,
                    req_key,
                    ctx.instrument,
                    case,
                    f"_dither{i}",
                    raw,
                    dark_2,
                    header,
                    cards
                    + [
                        ("DITHER", i, "Dither index"),
                        ("DITHX", float(pos[0]), "WCU FP mask shift x (arcsec)"),
                        ("DITHY", float(pos[1]), "WCU FP mask shift y (arcsec)"),
                        ("DITHNOM", offset, "Nominal dither offset (arcsec)"),
                        ("FOV_TRUE", fov_true, "Pixel scale * illum. pixels (arcsec)"),
                    ],
                    saturation_adu=ctx.saturation_adu,
                )
            )
            # for the FYI plots: pinhole position, cutout, binned co-added frame
            bckgd_subted = raw - dark_2
            peak = measure_peak(bckgd_subted)
            peaks_yx.append(peak)
            cutouts.append(cutout(bckgd_subted, peak, FYI_CUTOUT))
            binned = block_reduce(bckgd_subted, (FYI_BIN, FYI_BIN), func=np.mean)
            coadded = binned if coadded is None else coadded + binned

        fov_measured = fov_from_peaks(peaks_yx, offset, illuminated_pixels)
        logging.info(
            "%s case %s: injected FoV %.3f arcsec; from the pinhole positions (as the "
            "ANALYSIS measures it) %.3f arcsec",
            req_key,
            case,
            float(inject["fov"]),
            fov_measured,
        )
        log_tbd_distortion(f"IMG-OPT-01 SIM, {req_key} {tag} case {case}")
        logging.info(
            "%s case %s: pinhole positions (guesses for the ANALYSIS config):\n%s",
            req_key,
            case,
            "\n".join(
                f"  - {{guess_x: {int(x)}, guess_y: {int(y)}}}  # dither{i}"
                for i, (y, x) in enumerate(peaks_yx)
            ),
        )
        cases_info[case] = {
            "positions": positions,
            "peaks_yx": peaks_yx,
            "fov_injected": float(inject["fov"]),
            "fov_measured": fov_measured,
            "expected": expected[case],
            "shape": dark_2.shape,
        }
        if ctx.results_dir:
            fyi_plot_case(
                ctx.results_dir,
                f"{req_key}_{tag}_{case}",
                cases_info[case],
                coadded,
                cutouts,
                illuminated_pixels,
            )

    if ctx.results_dir:
        fyi_plot_dither_geometry(
            ctx.results_dir,
            f"{req_key}_{tag}",
            offset,
            cases_info,
            req_cfg["limit"]["fov"],
        )
    sim_common.set_fpmask_shift(ctx.metis, (0.0, 0.0))
    return written


# requirement key (in the scenario config) -> function writing its simulated data
SIMULATORS = {
    "METIS-1095": simulate_metis_1095,
}


@pipeline_stage(
    name="generate_fov_sim",
    depends_on=("sim_img_01_metis_1095",),
    cluster=CLUSTER_FOV,
    cluster_color=COLOR_FOV,
    label="Generate FOV sims",
)
def main():
    """
    Generate IMG-OPT-01 FoV simulations, one set per requirement, as set in the
    scenario config.
    """
    stem = project_root()
    out_dir = stem + "data/IMG_OPT_01_fov_simmed_data/"
    # FYI plots of the simulated data go with the test's results
    results_dir = stem + "results/IMG_OPT_01_fov_analysis_results/"

    now = datetime.datetime.now()
    sim_common.start_sim_logging(
        stem, "IMG_01_fov_sim_logs", "IMG_OPT_01_METIS_AIT_img_cal_fov_SIM", now
    )
    log_tbd_distortion("IMG-OPT-01 SIM")
    logging.info("Simmed file output directory: %s", out_dir)
    logging.info("FYI plot directory: %s", results_dir)
    # uncomment to (re-)download the IRDB packages (overwrites them in the IRDB)
    # sim.download_packages(
    #   ["METIS", "ELT", "Armazones"], save_dir=sim_common.IRDB_DIR
    # )
    sim_common.setup_scopesim()

    observing = load_observing_config(stem)
    scenario = load_config_and_pipe(
        config_file_choice=stem
        + "config/config_file_IMG_OPT_01_METIS_AIT_img_cal_fov_SIM.yaml",
        print_one_line=False,
    )
    sim_common.run_scenario(
        scenario,
        SIMULATORS,
        observing,
        out_dir,
        TEST_PREFIX,
        results_dir=results_dir,
    )
    log_tbd_distortion("IMG-OPT-01 SIM")


if __name__ == "__main__":
    main()
