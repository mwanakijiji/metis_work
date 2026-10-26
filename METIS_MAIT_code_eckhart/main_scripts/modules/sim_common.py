"""
Machinery shared by the SIM scripts: ScopeSim set-up, the WCU OpticalTrain, darks and
readouts, and the requirement scenarios (limits, cases on either side of them, and the
FITS files with the requirement and the injected truth in the header).

A scenario config (config/config_file_<test>_SIM.yaml) has
- instruments: name -> WCU / instrument set-up (obs_filter, fp_mask, pp_mask,
  psf_kernel, nd_filter, obs_mode, dit, ndit, intrapixel_capacitance); or a single
  'instrument:' block (as for IMG-RAD-07)
- requirements: requirement key -> {simulate, description, limit, cases, ...}, where
  - limit: quantity -> {value, pass_if ('<=', '>=' or 'in' with value [lo, hi]), unit}
  - cases: case name -> {inject: {quantity: injected value}}; a 'pass' and a 'fail'
    case are required, and must be on the right side of the limit
  - by_detector (optional): detector key ('img_lm', 'img_n') -> keys (e.g. limit,
    cases) overriding the ones above for instruments on that detector
  - instruments (optional): names of the instruments to simulate this requirement for
    (default: all)
  - detectors (optional): detector keys to simulate this requirement for (default: all)
Each SIM script maps requirement keys to simulator functions and calls run_scenario.

Each case file is named <test prefix>_<requirement>_<instrument>_<case>[_...].fits,
where <instrument> is the instrument name (or the filter, for a single 'instrument:'),
and has a header-only primary HDU with the requirement cards (see requirement_cards),
then RAW_READOUT, BCKGD_SUBTED (= RAW_READOUT - BACKGROUND) and BACKGROUND, as float32,
plus any extra extensions of the test.
"""

import copy
import logging
import os
import subprocess
import sys

import numpy as np
import scopesim as sim
import yaml
from astropy import units as u
from astropy.io import fits

from .helpers import pipe_2_log, setup_logging

BB_TEMP = 1000 * u.K

# IRDB (METIS/, ELT/, Armazones/ packages) location relative to main_scripts/, i.e.
# ~/Desktop = /podman-share; run the SIM scripts from main_scripts/
IRDB_REL_PATH = "../../../"

# obs_mode -> detector key used in the observing config (pixel_scales, saturation_adu)
DETECTOR_KEYS = {"wcu_img_lm": "img_lm", "wcu_img_n": "img_n"}


def start_sim_logging(stem, log_subdir, log_stub, now):
    """Log file <stem>logs/<log_subdir>/log_<log_stub>_<timestamp>.txt."""
    log_dir = os.path.join(stem, "logs", log_subdir)
    log_file_name = os.path.join(
        log_dir, f"log_{log_stub}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    )
    setup_logging(log_dir=log_dir, log_file_name=log_file_name, now=now)


def setup_scopesim():
    """Point ScopeSim at the local IRDB and log the versions of things."""
    sim.link_irdb(IRDB_REL_PATH)
    # keep commented out: it re-extracts the packages over the IRDB on every run
    # sim.download_packages(["METIS", "ELT", "Armazones"])
    pipe_2_log(sim.bug_report, msg="ScopeSim bug report")


def detector_key(obs_mode):
    """Detector key ('img_lm' or 'img_n') of a WCU imaging obs_mode."""
    if obs_mode not in DETECTOR_KEYS:
        raise ValueError(f"No detector key for obs_mode {obs_mode!r}")
    return DETECTOR_KEYS[obs_mode]


def sanity_check_user_inputs(metis, obs_filter, fp_mask, pp_mask):
    """Exit if the filter/masks the OpticalTrain uses do not match the user inputs."""
    checks = (
        ("OBS filter", "!OBS.filter_name", obs_filter),
        ("WCU FP mask", "!WCU.current_fpmask", fp_mask),
        ("OBS PP mask", "!OBS.pupil_mask", pp_mask),
    )
    for label, key, user_value in checks:
        if metis.cmds.get(key) != user_value:
            logging.error(
                "! ------- %s: %s does not match user input: %s",
                label,
                metis.cmds.get(key),
                user_value,
            )
            sys.exit(1)
    logging.info("User filter inputs match instrument inputs")


def sim_version():
    """Short string of the ScopeSim, IRDB METIS and code versions (for SIMVER)."""
    parts = [f"ss{sim.__version__}"]
    try:
        pkgs_path = sim.rc.__config__["!SIM.file.local_packages_path"]
        with open(
            os.path.join(pkgs_path, "METIS", "version.yaml"), encoding="utf-8"
        ) as file:
            parts.append(f"irdb{yaml.safe_load(file)['version']}")
    except Exception:  # pylint: disable=broad-except
        parts.append("irdb?")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).stdout.strip()
        parts.append(f"git{commit}")
    except Exception:  # pylint: disable=broad-except
        parts.append("git?")
    return " ".join(parts)[:68]


def passes_limit(value, spec):
    """True if value is on the passing side of one limit spec {value, pass_if}."""
    if spec["pass_if"] == "<=":
        return value <= float(spec["value"])
    if spec["pass_if"] == ">=":
        return value >= float(spec["value"])
    if spec["pass_if"] == "in":
        lo, hi = (float(v) for v in spec["value"])
        return lo <= value <= hi
    raise ValueError(f"Unknown pass_if: {spec['pass_if']!r}")


def expected_outcome(limit, inject):
    """
    'PASS' if every injected quantity is on the passing side of its limit, else 'FAIL'.

    INPUTS:
    - limit (dict): quantity -> {value, pass_if ('<=', '>=' or 'in'), ...}
    - inject (dict): quantity -> injected value
    """
    for quantity, spec in limit.items():
        if not passes_limit(float(inject[quantity]), spec):
            return "FAIL"
    return "PASS"


def check_cases(req_key, req_cfg):
    """
    Check the cases of a requirement: a 'pass' and a 'fail' case exist, every case
    injects every limited quantity, and the 'pass'/'fail' cases are on the right side
    of the limit. Exits otherwise.

    OUTPUTS:
    - dict: case name -> EXPECTED ('PASS' or 'FAIL')
    """
    limit = req_cfg["limit"]
    cases = req_cfg["cases"]
    if "pass" not in cases or "fail" not in cases:
        logging.error("%s: needs both a 'pass' and a 'fail' case", req_key)
        sys.exit(1)
    expected = {}
    for case, case_cfg in cases.items():
        missing = set(limit) - set(case_cfg["inject"])
        if missing:
            logging.error("%s case %s: no injected %s", req_key, case, sorted(missing))
            sys.exit(1)
        expected[case] = expected_outcome(limit, case_cfg["inject"])
        if case in ("pass", "fail") and expected[case] != case.upper():
            logging.error(
                "%s case %s: injected %s is %s, not %s, w.r.t. the limit %s",
                req_key,
                case,
                case_cfg["inject"],
                expected[case],
                case.upper(),
                limit,
            )
            sys.exit(1)
        logging.info(
            "%s case %s: inject %s -> expected %s",
            req_key,
            case,
            case_cfg["inject"],
            expected[case],
        )
    return expected


def resolve_for_detector(req_cfg, det_key):
    """
    Requirement config for one detector: a copy of req_cfg with the keys in
    req_cfg['by_detector'][det_key] (if any) overriding the top-level ones.
    """
    resolved = copy.deepcopy(req_cfg)
    overrides = resolved.pop("by_detector", {}) or {}
    resolved.update(overrides.get(det_key, {}))
    return resolved


def requirement_cards(req_key, req_cfg, case, inject, expected, simver):
    """FITS header cards with the requirement and the injected truth."""
    cards = [
        ("REQ_ID", req_cfg.get("req_id", req_key), "Requirement simulated"),
        ("CASE", case, "Case of the scenario"),
        ("EXPECTED", expected, "Expected outcome vs. requirement"),
        (
            "REQ_PLH",
            bool(req_cfg.get("limit_is_placeholder", False)),
            "Requirement limit is a placeholder",
        ),
        ("SIMVER", simver, "ScopeSim, IRDB, code versions"),
    ]
    limit = req_cfg["limit"]
    numbered = len(limit) > 1
    for i, (quantity, spec) in enumerate(limit.items(), start=1):
        n = str(i) if numbered else ""
        if isinstance(spec["value"], (list, tuple)):
            lim = ",".join(f"{float(v):g}" for v in spec["value"])
        else:
            lim = float(spec["value"])
        cards += [
            (f"REQ_QTY{n}", quantity, f"Quantity ({spec.get('unit', '')})"),
            (f"REQ_LIM{n}", lim, "Requirement limit"),
            (f"PASS_IF{n}", spec["pass_if"], "Pass if measured <op> REQ_LIM"),
            (f"INJ_VAL{n}", float(inject[quantity]), "Injected value"),
        ]
    return cards


def psf_kernel_of(instrument):
    """
    ScopeSim PSF kernel (METIS_psfs/PSF_<kernel>.fits): the configured psf_kernel, else
    <pp_mask>_WCU (only the PPS masks have _WCU kernels, without the ELT spiders).
    """
    return instrument.get("psf_kernel") or instrument["pp_mask"] + "_WCU"


def build_optical_train(instrument):
    """
    ScopeSim OpticalTrain for one WCU set-up (pupil mask and matching PSF kernel set,
    IPC on or off).
    """
    properties = {
        "!OBS.filter_name": instrument["obs_filter"],
        "!WCU.current_fpmask": instrument["fp_mask"],
        "!OBS.pupil_mask": instrument["pp_mask"],
        "!SIM.psf.interp_order": 1,
    }
    if instrument.get("nd_filter") is not None:
        properties["!OBS.nd_filter_name"] = instrument["nd_filter"]
    cmd = sim.UserCommands(
        use_instrument="METIS",
        set_modes=[instrument["obs_mode"]],
        properties=properties,
    )
    metis = sim.OpticalTrain(cmd)

    if not instrument.get("intrapixel_capacitance", True):
        metis["ipc"].include = False
        logging.info("Turning off inter-pixel capacitance")
    else:
        logging.info("Using default inter-pixel capacitance")

    # Set the pupil mask (transmission) and the matching PSF kernel (ScopeSim doesn't
    # link the two)
    psf_kernel = psf_kernel_of(instrument)
    metis["pupil_masks"].change_mask(instrument["pp_mask"])
    metis["psf"].update(pupil_mask=psf_kernel)
    logging.info("WCU PP mask: %s; PSF kernel: %s", instrument["pp_mask"], psf_kernel)

    pipe_2_log(
        lambda m=metis: m.effects.pprint_all(),
        msg="Optical train effects (initial)",
    )
    sanity_check_user_inputs(
        metis,
        obs_filter=instrument["obs_filter"],
        fp_mask=instrument["fp_mask"],
        pp_mask=instrument["pp_mask"],
    )
    return metis


def set_fpmask_shift(metis, shift_xy_arcsec):
    """Shift the WCU focal-plane mask to (x, y) arcsec (absolute position)."""
    metis["wcu_source"].set_fpmask(shift=np.asarray(shift_xy_arcsec), angle=0.0)
    logging.info("WCU FP mask shifted to (x, y) = %s arcsec", shift_xy_arcsec)


def readout_frame(metis, bb_aperture, instrument):
    """Observe with the WCU black-body aperture at bb_aperture and read out one frame."""
    metis["wcu_source"].set_bb_aperture(value=float(bb_aperture))
    metis.observe()
    hdul = metis.readout(ndit=instrument["ndit"], dit=instrument["dit"], reset=False)[0]
    logging.info(
        "Read out frame: BB aperture %.4f, DIT %s s, NDIT %s, median %.4g, max %.4g "
        "counts",
        bb_aperture,
        instrument["dit"],
        instrument["ndit"],
        float(np.median(hdul[1].data)),
        float(np.max(hdul[1].data)),
    )
    return hdul[1].data.astype(float), hdul[0].header


def take_darks(metis, instrument):
    """
    Two independent darks (black-body aperture closed): dark_1 to extract the signal
    from the illuminated frames, dark_2 to write out as BACKGROUND.
    """
    dark_1, _ = readout_frame(metis, 0.0, instrument)
    dark_2, _ = readout_frame(metis, 0.0, instrument)
    logging.info(
        "Std of dark_2 - dark_1: %.4g counts%s",
        float(np.std(dark_2 - dark_1)),
        "" if np.std(dark_2 - dark_1) > 0 else " -- NOT INDEPENDENT READOUTS!",
    )
    return dark_1, dark_2


def smooth_pattern(shape, kind="tilt_and_radial"):
    """
    Smooth illumination pattern over the frame with mean 0 and max |P| = 1, so that
    flat * (1 + a * P) deviates from its mean by at most a.
    """
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    x = (xx - (nx - 1) / 2) / (nx / 2)
    y = (yy - (ny - 1) / 2) / (ny / 2)
    if kind == "tilt_and_radial":
        pattern = 0.6 * x + 0.3 * y + 0.4 * (x**2 + y**2)
    elif kind == "saddle":  # a different pattern, for changes between flux steps
        pattern = x * y + 0.3 * x
    else:
        raise ValueError(f"Unknown pattern kind: {kind}")
    pattern = pattern - pattern.mean()
    return pattern / np.max(np.abs(pattern))


def instrument_tag(instrument):
    """Instrument label in file names: its scenario name, else the filter."""
    return instrument.get("name") or instrument["obs_filter"]


def write_case_file(
    out_dir,
    test_prefix,
    req_key,
    instrument,
    case,
    suffix,
    raw,
    background,
    primary_header,
    cards,
    extra_hdus=(),
):
    """
    Write one case: primary header (ScopeSim's + set-up + requirement cards), then
    RAW_READOUT, BCKGD_SUBTED (raw - background) and BACKGROUND, as float32, then
    extra_hdus (astropy HDUs, written as given).

    OUTPUTS:
    - str: path of the written file
    """
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(header=primary_header),
            fits.ImageHDU(data=raw.astype(np.float32), name="RAW_READOUT"),
            fits.ImageHDU(
                data=(raw - background).astype(np.float32), name="BCKGD_SUBTED"
            ),
            fits.ImageHDU(data=background.astype(np.float32), name="BACKGROUND"),
        ]
        + list(extra_hdus)
    )
    hdr = hdul[0].header
    hdr["FILTER"] = (instrument["obs_filter"], "Observing filter")
    hdr["WCU_FP"] = (instrument["fp_mask"], "WCU focal plane mask")
    hdr["WCU_PP"] = (instrument["pp_mask"], "WCU pupil plane mask")
    hdr["PSF_KERN"] = (psf_kernel_of(instrument), "ScopeSim PSF kernel")
    hdr["ND_FILT"] = (str(instrument.get("nd_filter")), "ND filter")
    hdr["BB_TEMP"] = (BB_TEMP.value, "BB temperature")
    hdr["NDIT"] = (instrument["ndit"], "Number of DITs")
    hdr["DIT"] = (instrument["dit"], "Det integration time")
    for key, value, comment in cards:
        hdr[key] = (value, comment)

    os.makedirs(out_dir, exist_ok=True)
    file_name = os.path.join(
        out_dir,
        f"{test_prefix}_{req_key}_{instrument_tag(instrument)}_{case}{suffix}.fits",
    )
    hdul.writeto(file_name, overwrite=True)
    logging.info("Saved %s", file_name)
    return file_name


class SimContext:  # pylint: disable=too-many-instance-attributes
    """
    What a requirement simulator needs for one instrument set-up: the OpticalTrain,
    the instrument config, the merged observing config, where and how to write (incl.
    an optional directory for FYI plots), and (cached, so requirements simulated for
    the same instrument share them) two independent darks and a full-aperture
    illuminated frame.
    """

    def __init__(
        self,
        metis,
        instrument,
        observing,
        out_dir,
        test_prefix,
        simver,
        results_dir=None,
    ):
        self.metis = metis
        self.instrument = instrument
        self.observing = observing
        self.out_dir = out_dir
        self.test_prefix = test_prefix
        self.simver = simver
        self.results_dir = results_dir
        self.det_key = detector_key(instrument["obs_mode"])
        self._darks = None
        self._lit = None

    def darks(self):
        """(dark_1, dark_2): see take_darks."""
        if self._darks is None:
            self._darks = take_darks(self.metis, self.instrument)
        return self._darks

    def lit_frame(self):
        """(frame, header): readout with the WCU black-body aperture fully open."""
        if self._lit is None:
            self._lit = readout_frame(self.metis, 1.0, self.instrument)
        return self._lit

    def write_case(self, req_key, case, suffix, raw, cards, extra_hdus=()):
        """write_case_file with dark_2 as BACKGROUND and the lit frame's header."""
        _, dark_2 = self.darks()
        _, header = self.lit_frame()
        return write_case_file(
            self.out_dir,
            self.test_prefix,
            req_key,
            self.instrument,
            case,
            suffix,
            raw,
            dark_2,
            header,
            cards,
            extra_hdus=extra_hdus,
        )


def scenario_instruments(scenario):
    """name -> instrument config, from 'instruments:' or a single 'instrument:'."""
    if "instruments" in scenario:
        return {
            name: {**cfg, "name": name} for name, cfg in scenario["instruments"].items()
        }
    return {None: dict(scenario["instrument"])}


def requirement_applies(req_cfg, name, instrument):
    """True if a requirement is to be simulated for this instrument."""
    names = req_cfg.get("instruments")
    if names is not None and name not in names:
        return False
    detectors = req_cfg.get("detectors")
    if detectors is not None and detector_key(instrument["obs_mode"]) not in detectors:
        return False
    return True


def run_scenario(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    scenario, simulators, observing, out_dir, test_prefix, results_dir=None
):
    """
    Run the requirement simulators of a scenario config: for each instrument, one
    OpticalTrain; for each requirement with simulate: true that applies to it, the
    requirement's simulator, simulator(ctx, req_key, req_cfg) -> list of written files,
    with req_cfg resolved for the instrument's detector (see resolve_for_detector).

    INPUTS:
    - scenario (dict): the scenario config
    - simulators (dict): requirement key -> simulator function
    - observing (dict): merged observing config (common + test-specific)
    - out_dir (str): directory to write to
    - test_prefix (str): file name prefix, e.g. 'IMG_OPT_04'
    - results_dir (str or None): directory for FYI plots (ctx.results_dir), if any

    OUTPUTS:
    - dict: (requirement key, instrument name) -> list of written files
    """
    to_simulate = {}
    for req_key, req_cfg in scenario["requirements"].items():
        if not req_cfg.get("simulate", False):
            logging.info(
                "%s: not simulated (%s)", req_key, req_cfg.get("reason", "no reason")
            )
            continue
        if req_key not in simulators:
            logging.error("%s: simulate is true, but there is no simulator", req_key)
            sys.exit(1)
        to_simulate[req_key] = req_cfg

    simver = sim_version()
    logging.info("SIMVER: %s", simver)
    logging.info(
        "! -- ScopeSim assumes blackbody injection efficiency of 1., but true "
        "value is expected to be between ~1/6 and 1."
    )

    written = {}
    for name, instrument in scenario_instruments(scenario).items():
        applicable = {
            req_key: req_cfg
            for req_key, req_cfg in to_simulate.items()
            if requirement_applies(req_cfg, name, instrument)
        }
        if not applicable:
            logging.info("Instrument %s: no requirements to simulate", name)
            continue
        logging.info("##################################################")
        logging.info("Instrument %s: %s", name, instrument)
        ctx = SimContext(
            build_optical_train(instrument),
            instrument,
            observing,
            out_dir,
            test_prefix,
            simver,
            results_dir=results_dir,
        )
        for req_key, req_cfg in applicable.items():
            logging.info("==================================================")
            logging.info("%s (%s): %s", req_key, name, req_cfg["description"])
            written[(req_key, name)] = simulators[req_key](
                ctx, req_key, resolve_for_detector(req_cfg, ctx.det_key)
            )

    logging.info("==================================================")
    for (req_key, name), files in written.items():
        logging.info("%s (%s): %s files written", req_key, name, len(files))
    return written
