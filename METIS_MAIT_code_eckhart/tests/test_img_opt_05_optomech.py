"""
Unit tests for the IMG-OPT-05 analysis (modules/backbone_img_opt_05_optomech_stability.py),
on synthetic pupil-image sequences made with the SIM's pupil_image and jitter_track.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time, TimeDelta

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "main_scripts", _ROOT / "misc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import IMG_OPT_05_optomechanical_stability_SIM as sim05  # noqa: E402
from modules import backbone_img_opt_05_optomech_stability as stab  # noqa: E402

SHAPE = (400, 400)
DIAMETER = 240.0
LEVEL = 10000.0
NOISE = 30.0  # ADU per pixel
CENTRE = np.array([199.5, 199.5])
PIX_SCALE = 5.47
CFG = {
    "pupil": {"margin_px": 20, "inner_frac": 0.8, "n_iter": 3},
    "span_tolerance": 0.1,
    "requirement_max_s": 600,
    "n_sigma": 3,
    "saturation_frac": 0.8,
    "limits": {"METIS-5422": {"img_lm": {"position_rms": 0.2}}},
}


def synthetic_frame(shift, rng, noise=NOISE):
    frame = LEVEL * sim05.pupil_image(SHAPE, CENTRE + shift, DIAMETER)
    return frame + rng.normal(0.0, noise, SHAPE)


def test_pupil_centre_recovers_subpixel_shift_and_sigma_is_honest():
    rng = np.random.default_rng(0)
    errors, sigmas = [], []
    for _ in range(20):
        shift = rng.uniform(-0.1, 0.1, 2)
        centre, sigma, radius, level = stab.pupil_centre(synthetic_frame(shift, rng))
        errors.append(centre - (CENTRE + shift))
        sigmas.append(sigma)
        assert radius == pytest.approx(DIAMETER / 2, abs=1.0)
        assert level == pytest.approx(LEVEL, rel=0.01)
    errors, sigmas = np.array(errors), np.array(sigmas)
    assert np.max(np.abs(errors)) < 0.005
    # the propagated sigma is conservative (>= the actual scatter), but not wildly
    scatter = errors.std(axis=0)
    assert np.all(sigmas.mean(axis=0) >= 0.7 * scatter)
    assert np.all(sigmas.mean(axis=0) < 10 * scatter + 1e-4)


def test_rms_motion_debiases_noise():
    rng = np.random.default_rng(1)
    dx, dy = sim05.jitter_track(10, 0.05, rng)
    true = np.column_stack([dx, dy])
    sig = np.full((10, 2), 0.002)
    noisy = true + rng.normal(0.0, 0.002, true.shape)
    rms, sigma, raw, offsets = stab.rms_motion(noisy, sig)
    assert rms == pytest.approx(0.05, abs=3 * sigma + 1e-3)
    assert rms <= raw
    assert offsets.shape == (10, 2)
    # pure noise: debiased RMS near zero, sigma at the noise floor
    rms0, sigma0, _, _ = stab.rms_motion(rng.normal(0.0, 0.002, (10, 2)), sig)
    assert rms0 < 3 * sigma0


def test_detector_of_filter():
    observing = {
        "polychromatic_observing_filters_lm_leaf_name": {"Lp": "a"},
        "polychromatic_observing_filters_n_leaf_name": {"N2": "b"},
    }
    assert stab.detector_of_filter(observing, "Lp") == "img_lm"
    assert stab.detector_of_filter(observing, "N2") == "img_n"
    with pytest.raises(ValueError):
        stab.detector_of_filter(observing, "nope")


@pytest.mark.parametrize("rms_mas, expected", [(0.1, "PASS"), (0.4, "FAIL")])
def test_sequence_end_to_end(tmp_path, rms_mas, expected):
    rng = np.random.default_rng(2)
    n_frames, timescale = 10, 60.0
    dx, dy = sim05.jitter_track(n_frames, rms_mas / PIX_SCALE, rng)
    start = Time("2026-01-01T00:00:00", scale="utc")
    for k, t_rel in enumerate(sim05.frame_times(timescale, n_frames)):
        frame = synthetic_frame(np.array([dx[k], dy[k]]), rng)
        primary = fits.PrimaryHDU()
        hdr = primary.header
        hdr["FILTER"] = "Lp"
        hdr["REQ_ID"], hdr["CASE"], hdr["EXPECTED"] = "METIS-5422", "x", expected
        hdr["REQ_QTY"], hdr["REQ_LIM"], hdr["INJ_VAL"] = "position_rms", 0.2, rms_mas
        hdr["DATE-OBS"] = (start + TimeDelta(t_rel, format="sec")).isot
        hdr["SHIFT_X"], hdr["SHIFT_Y"] = float(dx[k]), float(dy[k])
        fits.HDUList(
            [primary, fits.ImageHDU(frame.astype(np.float32), name="BCKGD_SUBTED")]
        ).writeto(tmp_path / f"seq_frame{k + 1:02d}.fits")

    observing = {
        "img_opt_05": CFG,
        "pixel_scales": {"img_lm": PIX_SCALE},
        "polychromatic_observing_filters_lm_leaf_name": {"Lp": "x"},
    }
    state = {
        "req_id": "METIS-5422",
        "instrument": "Lp",
        "case": "x",
        "timescale_s": timescale,
        "file_name_pattern": str(tmp_path / "seq_frame*.fits"),
        "results_write_dir": str(tmp_path / "results"),
    }
    result = stab.populate_run(state, observing)
    result = stab.measure_positions(result)
    assert result.times[-1] == pytest.approx(timescale)
    result = stab.metis_5422_check(result)
    assert result.measured["position_rms"] == pytest.approx(rms_mas, abs=0.02)
    assert result.verdicts["METIS-5422"] == expected
    assert result.measured["offset_error_max_px"] < 0.005
    assert Path(stab.plot_run(result)[0]).exists()
    assert stab.log_test_outcome(result).consistent is True
