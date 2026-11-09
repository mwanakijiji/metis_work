"""
Machinery shared by the ANALYSIS scripts: data states from a defaults + runs config,
logging the injected truth of SIM files, the raw-readout saturation check, noise-aware
verdicts and the comparison of a verdict with the injected truth (EXPECTED).

(Used by IMG-RAD-07; the same pieces are still inline in the IMG-OPT-03 ANALYSIS.)
"""

import logging
import os

import numpy as np
from astropy.io import fits
from scipy import ndimage

from modules.helpers import load_config_and_pipe


def load_data_states(stem, config_file, path_keys=("file_name_abs",)):
    """
    Data states from a config with 'defaults' and 'runs': each run is the defaults
    overridden by the run's entries, with the relative paths named in path_keys (if
    present) prefixed with stem.

    INPUTS:
    - stem (str): project root, ending in '/'
    - config_file (str): data-states config, relative to stem or absolute
    - path_keys (iterable of str): keys holding paths relative to stem

    OUTPUTS:
    - list[dict]: one merged dict per run
    """
    path = config_file if os.path.isabs(config_file) else stem + config_file
    config = load_config_and_pipe(config_file_choice=path, print_one_line=False)
    defaults = config.get("defaults") or {}
    states = []
    for entry in config.get("runs") or []:
        merged = {**defaults, **entry}
        for key in path_keys:
            value = merged.get(key)
            if isinstance(value, str) and not value.startswith("/"):
                merged[key] = stem + value
        states.append(merged)
    return states


def log_injected_truth(header):
    """
    Log the requirement and injected truth from a SIM primary header (REQ_ID, CASE,
    REQ_QTY / INJ_VAL, or numbered REQ_QTYn / INJ_VALn, EXPECTED), if there.
    """
    if "REQ_ID" not in header:
        return
    if "REQ_QTY" in header:
        injected = f"{header['REQ_QTY']} = {header.get('INJ_VAL')}"
    else:
        parts = []
        num = 1
        while f"REQ_QTY{num}" in header:
            parts.append(f"{header[f'REQ_QTY{num}']} = {header.get(f'INJ_VAL{num}')}")
            num += 1
        injected = ", ".join(parts) or "(none)"
    logging.info(
        "Simulated data: %s case %s, injected %s -> expected %s",
        header["REQ_ID"],
        header.get("CASE"),
        injected,
        header.get("EXPECTED"),
    )


def injected_limits(header):
    """quantity -> limit from a SIM primary header (REQ_QTY/REQ_LIM, or numbered)."""
    if "REQ_QTY" in header:
        return {header["REQ_QTY"]: header.get("REQ_LIM")}
    limits = {}
    num = 1
    while f"REQ_QTY{num}" in header:
        limits[header[f"REQ_QTY{num}"]] = header.get(f"REQ_LIM{num}")
        num += 1
    return limits


def check_saturation(file_name_abs, saturation_adu, hdu_index="RAW_READOUT", frac=0.8):
    """
    Log an error if the max counts in image extension hdu_index (the raw readout,
    before background subtraction) exceed frac of the saturation level. Does not stop
    the analysis.

    OUTPUTS:
    - bool: True if the check passed
    """
    with fits.open(file_name_abs) as hdul:
        max_counts = float(np.nanmax(hdul[hdu_index].data))
    name = os.path.basename(file_name_abs)
    if max_counts > frac * saturation_adu:
        logging.error(
            "! ------- %s: max counts in extension %s = %.0f ADU > %s x saturation "
            "(%.0f of %.0f ADU)",
            name,
            hdu_index,
            max_counts,
            frac,
            frac * saturation_adu,
            saturation_adu,
        )
        return False
    logging.info(
        "Saturation check OK for %s: max counts in extension %s = %.0f ADU <= %s x "
        "saturation (%.0f ADU)",
        name,
        hdu_index,
        max_counts,
        frac,
        frac * saturation_adu,
    )
    return True


def verdict_with_noise(measured, sigma, limit, n_sigma=3.0):
    """
    Verdict for a quantity that must not exceed limit, honest about the noise sigma of
    the measurement: PASS if measured + n_sigma * sigma <= limit, FAIL if
    measured - n_sigma * sigma > limit, else INCONCLUSIVE (also if either is not
    finite).
    """
    if not (np.isfinite(measured) and np.isfinite(sigma)):
        return "INCONCLUSIVE"
    if measured + n_sigma * sigma <= limit:
        return "PASS"
    if measured - n_sigma * sigma > limit:
        return "FAIL"
    return "INCONCLUSIVE"


def combine_verdicts(verdicts):
    """FAIL if any verdict is FAIL, PASS if all are PASS, else INCONCLUSIVE."""
    verdicts = list(verdicts)
    if "FAIL" in verdicts:
        return "FAIL"
    if verdicts and all(v == "PASS" for v in verdicts):
        return "PASS"
    return "INCONCLUSIVE"


def compare_with_expected(req_id, verdict, header):
    """
    Log whether a verdict agrees with the injected truth (EXPECTED in a SIM primary
    header).

    OUTPUTS:
    - bool or None: True/False if the header has EXPECTED, else None
    """
    expected = header.get("EXPECTED")
    if expected is None:
        logging.info("%s: no EXPECTED in the header (not simulated data?)", req_id)
        return None
    consistent = expected == verdict
    logging.info(
        "%s simulated data: expected %s; measured %s -> %s",
        req_id,
        expected,
        verdict,
        "consistent" if consistent else "INCONSISTENT",
    )
    return consistent


def field_mask(frame, frac=0.5, erosion_px=4):
    """
    Illuminated field: pixels above frac x the median of the frame (drops the
    reference-pixel border and anything unilluminated), eroded by erosion_px to avoid
    the edge (inter-pixel capacitance leaks into the border).
    """
    mask = frame > frac * np.median(frame)
    if erosion_px > 0:
        mask = ndimage.binary_erosion(mask, iterations=int(erosion_px))
    return mask
