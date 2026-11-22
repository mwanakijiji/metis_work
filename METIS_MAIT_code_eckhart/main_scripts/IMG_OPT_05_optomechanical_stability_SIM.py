"""
Make simulated data for the IMG-OPT-05 optomechanical stability test

Reqs.:
- Ref. IMG-OPT-05 Optomechanical stability, Table 1-1 (requirements verified by
  IMG-OPT-05)

1. METIS-1230 (verified by test at System level): METIS shall require an on-sky flux
calibration not more often than once per hour per instrument mode, so long as the
instrument is not powered down or disturbed in a manner outside its nominal operations
concept during this period.

2. METIS-5422 (verified by test at System level, applicable to IMG): The position of
the Vortex Phase Mask (VPM) located in CFO-FP2 shall be stable with respect to the IMG
detectors, with the following accuracy:
On timescales between 10 sec and 10 min (goal: up to 1h)
- 0.2 mas RMS on sky for the L- and M-band VPMs with respect to the IMG-LM detector
- 0.5 mas RMS on sky for the N-band VPMs with respect to the IMG-N detector
On timescales below 10 sec:
- 1 mas RMS on sky for the L- and M-band VPMs with respect to the IMG-LM detector
- 2.5 mas RMS on sky for the N-band VPMs with respect to the IMG-N detector

METIS-5422 is tested with pupil imaging: the position of a sharp pupil image on the
detector, over sequences of frames spanning each timescale. ScopeSim can't do pupil
imaging (and the CFO-FP2 mask isn't in ScopeSim yet), so simulate_metis_5422_stability
adds a sharp uniform disk, the pupil image (pupil_image), to independent ScopeSim dark
readouts (WCU background and detector noise), with photon noise, in the L, M and N
bands (filters Lp and Mp on the LM detector, N2 on the N detector). For each
timescale in the scenario config (timescales_s, in seconds), each case (pass/fail) and
each instrument it writes a sequence of n_frames (10) frames,
evenly spread over the timescale (frame_times), into the subdirectory
<timescale>s_<instrument>/ of the output directory:
  IMG_OPT_05_METIS-5422_<instrument>_<case>_<timescale>s_frameNN.fits
From frame to frame the pupil image jitters by independent random offsets
(jitter_track), scaled so that their 2D RMS about the mean position is exactly the
injected position_rms, in mas on sky: detector pixels x the pixel scale of the
instrument's detector (IMG-LM 5.47, IMG-N 6.79 mas/px; pixel_scales in the common
observing config), so the same RMS is a smaller pixel shift on the N detector. The
limit and the cases are per detector (by_detector: 0.2 mas LM, 0.5 mas N). Besides
the requirement cards (REQ_ID, CASE, EXPECTED, REQ_LIM, INJ_VAL = position_rms, ...),
each primary header has the timestamp and the truth of its frame:
  TSCALE    timescale tested (s); LIMTYPE 'requirement' (<= 10 min) or 'goal'
  FRAME, NFRAMES   frame number in the sequence, sequence length
  DATE-OBS, MJD-OBS, T_REL   time of the frame (UTC), seconds since the sequence start
  PUP_X, PUP_Y, SHIFT_X, SHIFT_Y, SHIFTMAS   true pupil centre (px), offset from the
            sequence's mean position (px, and mas on sky)
  PUPDIAM, PIXSCALE, PUPLEVEL, SEQ_RMS   pupil diameter (px), pixel scale (mas/px),
            pupil signal level (ADU), RMS of the sequence (mas)
The files have the usual layout (BCKGD_SUBTED, RAW_READOUT, BACKGROUND; BACKGROUND is
an independent dark). FYI plots of the tracks go to
sim_plots/IMG_OPT_05_optomechanical_stability_SIM/.

METIS-1230 is a stub (simulation of the flux-calibration stability TBD).
"""

import datetime
import logging
import os
import zlib

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time, TimeDelta

from modules import sim_common
from modules.config_common import load_observing_config
from modules.helpers import load_config_and_pipe
from modules.paths import project_root
from pipeline_registry import (  # pylint: disable=import-error,wrong-import-order
    CLUSTER_OPTOMECH,
    COLOR_OPTOMECH,
    pipeline_stage,
)

TEST_PREFIX = "IMG_OPT_05"


def pupil_image(shape, centre_xy, diameter_px, oversample=32):
    """
    Sharp pupil image: a uniform disk of diameter diameter_px centred at centre_xy
    (x, y; pixel centres at integer coordinates), each pixel weighted by the fraction of
    its area inside the disk. Pixels within 0.75 px of the edge get that fraction from
    oversample x oversample sub-pixels; the others are 0 or 1. The image's centroid then
    follows sub-pixel shifts to ~0.1% (a linear ramp in the distance to the edge moves
    it only ~0.987x the shift).

    OUTPUTS:
    - ndarray: values in [0, 1]
    """
    yy, xx = np.indices(shape, dtype=float)
    radius = 0.5 * diameter_px
    dist = np.hypot(xx - centre_xy[0], yy - centre_xy[1])
    image = (dist <= radius).astype(float)
    # pixel corners are up to sqrt(2)/2 px from the pixel centre
    edge_y, edge_x = np.nonzero(np.abs(dist - radius) < 0.75)
    sub = (np.arange(oversample) + 0.5) / oversample - 0.5
    sub_x, sub_y = (grid.ravel() for grid in np.meshgrid(sub, sub))
    inside = (
        np.hypot(
            edge_x[:, None] + sub_x[None, :] - centre_xy[0],
            edge_y[:, None] + sub_y[None, :] - centre_xy[1],
        )
        <= radius
    )
    image[edge_y, edge_x] = inside.mean(axis=1)
    return image


def jitter_track(n_frames, rms_px, rng):
    """
    Random jitter: independent Gaussian offsets (dx, dy) per frame, with their mean
    removed and scaled so that their 2D RMS sqrt(mean(dx^2 + dy^2)) is exactly rms_px.

    OUTPUTS:
    - ndarray: dx (n_frames,)
    - ndarray: dy (n_frames,)
    """
    offsets = rng.standard_normal((2, n_frames))
    offsets -= offsets.mean(axis=1, keepdims=True)
    offsets *= rms_px / np.sqrt(np.mean(np.sum(offsets**2, axis=0)))
    return offsets[0], offsets[1]


def track_rms(dx, dy):
    """2D RMS of offsets about their mean: sqrt(mean((dx - <dx>)^2 + (dy - <dy>)^2))."""
    dx, dy = np.asarray(dx, float), np.asarray(dy, float)
    return float(np.sqrt(np.mean((dx - dx.mean()) ** 2 + (dy - dy.mean()) ** 2)))


def frame_times(timescale_s, n_frames):
    """Frame times [s] from the sequence start, evenly spanning the timescale."""
    return np.linspace(0.0, float(timescale_s), int(n_frames))


def timescale_label(seconds):
    """Timescale label in file and directory names, e.g. '10s', '3600s'."""
    return f"{float(seconds):g}s"


def dark_readout(metis, instrument):
    """
    One more dark readout (detector noise drawn anew) of the last observation, which
    must have been taken with the WCU black-body aperture closed.
    """
    hdul = metis.readout(ndit=instrument["ndit"], dit=instrument["dit"], reset=False)[0]
    return hdul[1].data.astype(float)


def fyi_plot_tracks(plot_dir, label, tracks, pix_scale, limit_mas):
    """
    FYI plot of the injected jitter tracks of one case: offsets (mas on sky) vs. time,
    per timescale, with the RMS of each sequence.

    INPUTS:
    - plot_dir (str): where to write fyi_sim_<label>.png
    - label (str): e.g. 'METIS-5422_Lp_pass'
    - tracks (dict): timescale [s] -> (times [s], dx [px], dy [px])
    - pix_scale (float): mas / px
    - limit_mas (float): requirement limit on the RMS (mas)

    OUTPUTS:
    - str: path of the written plot
    """
    fig, axes = plt.subplots(
        1, len(tracks), figsize=(5 * len(tracks), 4), squeeze=False
    )
    for ax, (timescale, (times, dx, dy)) in zip(axes[0], tracks.items()):
        ax.plot(times, dx * pix_scale, "o-", label="x")
        ax.plot(times, dy * pix_scale, "s-", label="y")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xlabel("time since sequence start [s]")
        ax.set_ylabel("offset from mean position [mas on sky]")
        ax.set_title(
            f"{timescale_label(timescale)}: RMS "
            f"{track_rms(dx, dy) * pix_scale:.3f} mas (limit {limit_mas:g})",
            fontsize=9,
        )
        ax.legend(fontsize=8)
    fig.suptitle(f"IMG-OPT-05 SIM {label}: injected pupil-image jitter", fontsize=10)
    fig.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    file_name = os.path.join(plot_dir, f"fyi_sim_{label}.png")
    fig.savefig(file_name)
    plt.close(fig)
    logging.info("Saved FYI plot %s", file_name)
    return file_name


def _stub(req_key, req_cfg):
    """Log that a requirement's simulation is a stub; nothing is written."""
    logging.info(
        "%s (%s): stub, no files written", req_key, req_cfg.get("req_id", req_key)
    )
    return []


@pipeline_stage(
    name="simulate_metis_1230_flux_calibration",
    cluster=CLUSTER_OPTOMECH,
    cluster_color=COLOR_OPTOMECH,
    planned=True,  # stub: drawn dashed in the pipeline flow diagram
    label="SIM METIS-1230: flux calibration interval (stub)",
)
def simulate_metis_1230_flux_calibration(ctx, req_key, req_cfg):
    """
    Stub for METIS-1230 (on-sky flux calibration not more often than once per hour per
    instrument mode): writes nothing.

    OUTPUTS:
    - list[str]: written files (empty)
    """
    del ctx  # not used yet
    return _stub(req_key, req_cfg)


@pipeline_stage(
    name="simulate_metis_5422_stability",
    cluster=CLUSTER_OPTOMECH,
    cluster_color=COLOR_OPTOMECH,
    label="SIM METIS-5422: pupil-image position stability per timescale",
)
def simulate_metis_5422_stability(  # pylint: disable=too-many-locals
    ctx, req_key, req_cfg
):
    """
    Simulated data for METIS-5422 (position of the mask in CFO-FP2 stable w.r.t. the
    IMG detectors, as RMS on sky per timescale), tested with pupil imaging: for each
    timescale (timescales_s), case and the instrument, n_frames frames evenly spread
    over the timescale, each an independent ScopeSim dark readout plus a sharp pupil
    image (pupil_image) with photon noise, jittered by offsets (jitter_track) whose 2D
    RMS is the case's injected position_rms (mas on sky). Written into
    <out_dir>/<timescale>s_<instrument>/, with timestamps and the per-frame truth in
    the header (see the module docstring).

    OUTPUTS:
    - list[str]: written files
    """
    expected = sim_common.check_cases(req_key, req_cfg)
    limit = req_cfg["limit"]["position_rms"]
    pupil_cfg = req_cfg["pupil"]
    n_frames = int(req_cfg["n_frames"])
    pix_scale = float(ctx.observing["pixel_scales"][ctx.det_key])  # mas / px
    gain = float(ctx.metis.cmds["!DET.gain"])  # e- / ADU
    tag = sim_common.instrument_tag(ctx.instrument)

    dark_1, _ = ctx.darks()  # dark_2: BACKGROUND of every file
    ctx.lit_frame()  # its header goes into every file
    shape = dark_1.shape
    level = float(pupil_cfg["level_frac_sat"]) * ctx.saturation_adu - float(
        np.median(dark_1)
    )
    centre = (
        np.asarray(pupil_cfg["centre_px"], float)
        if pupil_cfg.get("centre_px") is not None
        else (np.array(shape[::-1], float) - 1.0) / 2.0
    )
    diameter = float(pupil_cfg["diameter_px"])
    logging.info(
        "%s (%s): pupil image %.0f px across at (%.1f, %.1f), level %.4g ADU (%.2f "
        "sat. with the dark); %s frames per sequence; pixel scale %.3f mas/px",
        req_key,
        tag,
        diameter,
        centre[0],
        centre[1],
        level,
        float(pupil_cfg["level_frac_sat"]),
        n_frames,
        pix_scale,
    )

    # dark readouts: observe once with the BB aperture closed, then read out anew
    ctx.metis["wcu_source"].set_bb_aperture(value=0.0)
    ctx.metis.observe()

    # independent tracks per instrument: seed from the config and the filter tag
    rng = np.random.default_rng([int(pupil_cfg["seed"]), zlib.crc32(tag.encode())])
    start = Time(datetime.datetime.now(datetime.timezone.utc))
    written = []
    tracks = {case: {} for case in req_cfg["cases"]}
    for timescale in req_cfg["timescales_s"]:
        label = timescale_label(timescale)
        limtype = (
            "requirement"
            if float(timescale) <= float(req_cfg["requirement_max_s"])
            else "goal"
        )
        times = frame_times(timescale, n_frames)
        for case, case_cfg in req_cfg["cases"].items():
            inject = case_cfg["inject"]
            rms_mas = float(inject["position_rms"])
            dx, dy = jitter_track(n_frames, rms_mas / pix_scale, rng)
            tracks[case][timescale] = (times, dx, dy)
            logging.info(
                "%s %s %s (%s): jitter RMS %.4f px = %.4f mas (injected %g mas, limit "
                "%g mas)",
                req_key,
                case,
                label,
                limtype,
                track_rms(dx, dy),
                track_rms(dx, dy) * pix_scale,
                rms_mas,
                float(limit["value"]),
            )
            cards = sim_common.requirement_cards(
                req_key, req_cfg, case, inject, expected[case], ctx.simver
            )
            for k in range(n_frames):
                pos = centre + np.array([dx[k], dy[k]])
                signal_e = level * gain * pupil_image(shape, pos, diameter)
                signal = rng.poisson(signal_e).astype(float) / gain
                raw = dark_readout(ctx.metis, ctx.instrument) + signal
                t_obs = start + TimeDelta(times[k], format="sec")
                frame_cards = cards + [
                    ("TSCALE", float(timescale), "Timescale tested (s)"),
                    ("LIMTYPE", limtype, "Limit at this timescale: requirement/goal"),
                    ("FRAME", k + 1, "Frame number in the sequence"),
                    ("NFRAMES", n_frames, "Frames in the sequence"),
                    ("DATE-OBS", t_obs.isot, "Time of the frame (UTC)"),
                    ("MJD-OBS", float(t_obs.mjd), "Time of the frame (MJD)"),
                    ("T_REL", float(times[k]), "Seconds since sequence start"),
                    ("PUP_X", float(pos[0]), "True pupil centre x (px, 0-based)"),
                    ("PUP_Y", float(pos[1]), "True pupil centre y (px, 0-based)"),
                    ("SHIFT_X", float(dx[k]), "Offset from mean position x (px)"),
                    ("SHIFT_Y", float(dy[k]), "Offset from mean position y (px)"),
                    (
                        "SHIFTMAS",
                        float(np.hypot(dx[k], dy[k]) * pix_scale),
                        "Offset from mean position (mas on sky)",
                    ),
                    ("PUPDIAM", diameter, "Pupil image diameter (px)"),
                    ("PIXSCALE", pix_scale, "Pixel scale (mas/px)"),
                    ("PUPLEVEL", level, "Pupil signal level (ADU)"),
                    (
                        "SEQ_RMS",
                        track_rms(dx, dy) * pix_scale,
                        "RMS of the sequence (mas on sky)",
                    ),
                ]
                written.append(
                    ctx.write_case(
                        req_key,
                        case,
                        f"_{label}_frame{k + 1:02d}",
                        raw,
                        frame_cards,
                        subdir=f"{label}_{tag}",
                    )
                )

    if ctx.plot_dir:
        for case, case_tracks in tracks.items():
            fyi_plot_tracks(
                ctx.plot_dir,
                f"{req_key}_{tag}_{case}",
                case_tracks,
                pix_scale,
                float(limit["value"]),
            )
    return written


# requirement key (in the scenario config) -> function writing its simulated data
SIMULATORS = {
    "METIS-1230": simulate_metis_1230_flux_calibration,
    "METIS-5422": simulate_metis_5422_stability,
}


@pipeline_stage(
    name="generate_optomechanical_stability_sim",
    depends_on=(
        "simulate_metis_1230_flux_calibration",
        "simulate_metis_5422_stability",
    ),
    cluster=CLUSTER_OPTOMECH,
    cluster_color=COLOR_OPTOMECH,
    label="Generate optomechanical-stability sims",
)
def main():
    """
    IMG-OPT-05 optomechanical stability simulations, one set per requirement, as set
    in the scenario config (METIS-5422; METIS-1230 is a stub).
    """
    stem = project_root()
    out_dir = stem + "data/IMG_OPT_05_optomechanical_stability_simmed_data/"
    plot_dir = sim_common.sim_plot_dir(stem, __file__)

    now = datetime.datetime.now()
    sim_common.start_sim_logging(
        stem,
        "IMG_OPT_05_optomechanical_stability_logs",
        "IMG_OPT_05_optomechanical_stability",
        now,
    )
    logging.info("Simmed file output directory: %s", out_dir)
    logging.info("FYI plot directory: %s", plot_dir)
    sim_common.setup_scopesim()

    observing = load_observing_config(stem)
    scenario = load_config_and_pipe(
        config_file_choice=stem
        + "config/config_file_IMG_OPT_05_optomechanical_stability_SIM.yaml",
        print_one_line=False,
    )
    n_tbd = sim_common.log_stub_inputs(scenario)
    logging.info("%s instrument set-up(s) with masks TBD (not simulated)", n_tbd)

    sim_common.run_scenario(
        scenario, SIMULATORS, observing, out_dir, TEST_PREFIX, plot_dir=plot_dir
    )


if __name__ == "__main__":
    main()
