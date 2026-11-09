"""
Unit tests for the requirement scenarios of the SIM scripts (modules/sim_common.py) and
the per-test injections (that what is injected is what the ANALYSIS measures).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

# the SIM scripts and modules import pipeline_registry from misc/
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "main_scripts", _ROOT / "misc"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import IMG_OPT_01_METIS_AIT_img_cal_fov_SIM as sim01  # noqa: E402
import IMG_OPT_03_METIS_AIT_img_cal_psf_quality_SIM as sim03  # noqa: E402
import IMG_OPT_04_METIS_AIT_img_stray_light_SIM as sim04  # noqa: E402
import IMG_OPT_06_METIS_AIT_out_of_field_SIM as sim06  # noqa: E402
import IMG_OPT_07_pupil_imaging_SIM as sim_opt07  # noqa: E402
import IMG_RAD_07_photometric_performance_SIM as sim07  # noqa: E402
import IMG_RAD_09_nominal_background_behavior_SIM as sim09  # noqa: E402
from modules import sim_common  # noqa: E402
from modules.config_common import load_observing_config  # noqa: E402
from modules import backbone_img_rad_09_nom as nom  # noqa: E402

LIMIT_LE = {"x": {"value": 1.0, "pass_if": "<="}}
LIMIT_GE = {"x": {"value": 1.0, "pass_if": ">="}}
LIMIT_IN = {"x": {"value": [10.0, 11.0], "pass_if": "in"}}


@pytest.mark.parametrize(
    "limit, value, expected",
    [
        (LIMIT_LE, 0.5, "PASS"),
        (LIMIT_LE, 1.0, "PASS"),
        (LIMIT_LE, 2.0, "FAIL"),
        (LIMIT_GE, 2.0, "PASS"),
        (LIMIT_GE, 0.5, "FAIL"),
        (LIMIT_IN, 10.5, "PASS"),
        (LIMIT_IN, 9.5, "FAIL"),
        (LIMIT_IN, 11.5, "FAIL"),
    ],
)
def test_expected_outcome(limit, value, expected):
    assert sim_common.expected_outcome(limit, {"x": value}) == expected


def test_expected_outcome_fails_if_any_quantity_fails():
    limit = {**LIMIT_LE, "y": {"value": 1.0, "pass_if": ">="}}
    assert sim_common.expected_outcome(limit, {"x": 0.5, "y": 2.0}) == "PASS"
    assert sim_common.expected_outcome(limit, {"x": 0.5, "y": 0.5}) == "FAIL"


def test_check_cases():
    req_cfg = {
        "limit": LIMIT_IN,
        "cases": {
            "pass": {"inject": {"x": 10.5}},
            "fail": {"inject": {"x": 11.5}},
            "fail_low": {"inject": {"x": 9.5}},
        },
    }
    assert sim_common.check_cases("R", req_cfg) == {
        "pass": "PASS",
        "fail": "FAIL",
        "fail_low": "FAIL",
    }


@pytest.mark.parametrize(
    "cases",
    [
        {"pass": {"inject": {"x": 0.5}}},  # no 'fail' case
        {"pass": {"inject": {"x": 2.0}}, "fail": {"inject": {"x": 3.0}}},  # wrong side
        {"pass": {"inject": {"x": 0.5}}, "fail": {"inject": {}}},  # nothing injected
    ],
)
def test_check_cases_exits_on_bad_cases(cases):
    with pytest.raises(SystemExit):
        sim_common.check_cases("R", {"limit": LIMIT_LE, "cases": cases})


def test_resolve_for_detector():
    req_cfg = {
        "simulate": True,
        "limit": LIMIT_LE,
        "by_detector": {"img_n": {"limit": LIMIT_GE}},
    }
    assert sim_common.resolve_for_detector(req_cfg, "img_lm")["limit"] == LIMIT_LE
    resolved_n = sim_common.resolve_for_detector(req_cfg, "img_n")
    assert resolved_n["limit"] == LIMIT_GE and "by_detector" not in resolved_n
    assert req_cfg["limit"] == LIMIT_LE  # not modified


def test_requirement_cards():
    req_cfg = {
        "req_id": "METIS-1",
        "limit": {**LIMIT_IN, "y": {"value": 2.0, "pass_if": "<="}},
    }
    cards = {
        key: value
        for key, value, _ in sim_common.requirement_cards(
            "R", req_cfg, "pass", {"x": 10.5, "y": 1.0}, "PASS", "simver"
        )
    }
    assert cards["REQ_ID"] == "METIS-1" and cards["EXPECTED"] == "PASS"
    assert cards["REQ_LIM1"] == "10,11" and cards["PASS_IF1"] == "in"
    assert cards["REQ_LIM2"] == 2.0 and cards["INJ_VAL2"] == 1.0
    assert cards["REQ_PLH"] is False


def test_requirement_applies():
    lm = {"obs_mode": "wcu_img_lm"}
    assert sim_common.requirement_applies({}, "a", lm)
    assert not sim_common.requirement_applies({"instruments": ["b"]}, "a", lm)
    assert not sim_common.requirement_applies({"detectors": ["img_n"]}, "a", lm)


def test_scenario_configs_are_consistent():
    # every simulated requirement of every SIM scenario has valid cases (for each
    # detector it is resolved for) and a simulator
    scenarios = {
        "config_file_IMG_OPT_01_METIS_AIT_img_cal_fov_SIM.yaml": sim01.SIMULATORS,
        "config_file_IMG_OPT_03_SIM.yaml": sim03.SIMULATORS,
        "config_file_IMG_OPT_04_stray_light_SIM.yaml": sim04.SIMULATORS,
        "config_file_IMG_OPT_06_outoffield_stray_light_SIM.yaml": sim06.SIMULATORS,
        "config_file_IMG_OPT_07_pupil_imaging_SIM.yaml": sim_opt07.SIMULATORS,
        "config_file_IMG_RAD_07_phot_performance_SIM.yaml": sim07.SIMULATORS,
        "config_file_IMG_RAD_09_nominal_background_SIM.yaml": sim09.SIMULATORS,
    }
    for file_name, simulators in scenarios.items():
        with open(_ROOT / "config" / file_name, encoding="utf-8") as file:
            scenario = yaml.safe_load(file)
        instruments = sim_common.scenario_instruments(scenario)
        for req_key, req_cfg in scenario["requirements"].items():
            if not req_cfg.get("simulate") or req_key == "TABLE-4-1":
                continue
            if simulators is not None:
                assert req_key in simulators, (file_name, req_key)
            applies_to = [
                name
                for name, inst in instruments.items()
                if sim_common.requirement_applies(req_cfg, name, inst)
            ]
            assert applies_to, (file_name, req_key)
            for name in applies_to:
                det_key = sim_common.detector_key(instruments[name]["obs_mode"])
                sim_common.check_cases(
                    req_key, sim_common.resolve_for_detector(req_cfg, det_key)
                )


def test_load_observing_config_merge_order(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "a.yaml").write_text("pixel_scales: {img_lm: 1.0}\nk: a\n")
    (tmp_path / "config" / "b.yaml").write_text("k: b\n")
    common = Path(_ROOT / "config" / "config_file_common_observing_params.yaml")
    (tmp_path / "config" / common.name).write_text(common.read_text())
    merged = load_observing_config(
        str(tmp_path) + "/", "config/a.yaml", "config/b.yaml"
    )
    assert merged["k"] == "b"  # later files win
    assert merged["pixel_scales"] == {"img_lm": 1.0}  # test file over the common one
    assert merged["saturation_adu"]["img_lm"] == 25000  # from the common one


def _gaussian_psf(shape, center, sigma, amplitude=1.0):
    yy, xx = np.indices(shape)
    return amplitude * np.exp(
        -((yy - center[0]) ** 2 + (xx - center[1]) ** 2) / (2 * sigma**2)
    )


@pytest.mark.parametrize("factor", [0.92, 0.6])
def test_inject_strehl_lowers_peak_and_conserves_flux(factor):
    signal = _gaussian_psf((121, 121), (60, 60), sigma=1.5, amplitude=1000.0)
    signal_new, s = sim03.inject_strehl(signal, factor, halo_sigma_pix=4.0)
    assert 0.0 < s < 1.0
    assert np.max(signal_new) == pytest.approx(factor * np.max(signal), rel=1e-6)
    assert np.sum(signal_new) == pytest.approx(np.sum(signal), rel=1e-6)


def test_inject_strehl_rejects_factor_above_one():
    with pytest.raises(ValueError):
        sim03.inject_strehl(np.ones((5, 5)), 1.2, halo_sigma_pix=1.0)


@pytest.mark.parametrize("direction", ["+x", "-y"])
def test_scaled_out_of_field_scatter(direction):
    scatter = sim06.scaled_out_of_field_scatter((64, 80), 1e5, 1e4, direction)
    assert 1e5 / np.max(scatter) == pytest.approx(1e4)


@pytest.mark.parametrize("pixel_scale_mas, fov_injected", [(5.47, 11.5), (6.79, 12.0)])
def test_dither_positions_give_injected_fov(pixel_scale_mas, fov_injected):
    # as the ANALYSIS (backbone_img_opt_01_fov.fov_calc): pixel scale from the corner
    # separations taken to be 2 * the nominal dither offset, times illuminated pixels
    offset, n_pix = 3.0, 1920
    fov_true = pixel_scale_mas / 1000.0 * n_pix
    positions_pix = (
        sim01.dither_positions(offset, fov_true, fov_injected)
        / (pixel_scale_mas / 1000.0)
    )
    sides_pix = np.linalg.norm(positions_pix - np.roll(positions_pix, -1, axis=0), axis=1)
    fov_measured = np.mean(2 * offset / sides_pix * n_pix)
    assert fov_measured == pytest.approx(fov_injected)


def test_artefact_labels_avoid_psf():
    cfg = {"seed": 4, "n_shapes": 5, "pixels_per_shape": [30, 60]}
    labels = sim04.artefact_labels((200, 200), cfg, psf_yx=(100, 100), exclusion_pix=40)
    yy, xx = np.indices(labels.shape)
    assert not np.any(labels[(yy - 100) ** 2 + (xx - 100) ** 2 < 40**2])
    assert sorted(np.unique(labels)) == list(range(int(labels.max()) + 1))


def test_artefact_image_amplitudes_and_flux():
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[10:20, 10:20] = 1
    labels[30:34, 30:40] = 2
    image = sim04.artefact_image(labels, [2.0, 5.0])
    assert image[15, 15] == pytest.approx(2.0)  # interior keeps the amplitude
    assert np.sum(image) == pytest.approx(2.0 * 100 + 5.0 * 40)  # flux conserved


# IMG-RAD-07 METIS-3300: flux steps from ND filters x DIT (fixed BB aperture)
ND_TRANS = {"ND_OD3": 1e-3, "ND_OD4": 1e-4, "ND_OD5": 1e-5}
MINDIT = 0.04


def test_exposure_steps_are_linear():
    steps = sim07.exposure_steps(3.0, 15)
    assert len(steps) == 15 and steps[-1] == pytest.approx(3.0)
    assert np.allclose(np.diff(steps), 3.0 / 15)


def test_max_exposure_puts_peak_at_limit():
    # probe: peak 2305 ADU at T 1e-3, DIT 0.04 s
    peak_per_exposure = 2305.0 / (1e-3 * 0.04)
    e_max = sim07.max_exposure(2305.0, 1e-3, 0.04, 20000.0, 0.03)
    assert peak_per_exposure * e_max == pytest.approx(20000.0 / 1.03)
    with pytest.raises(ValueError):
        sim07.max_exposure(0.0, 1e-3, 0.04, 20000.0, 0.03)


def test_choose_nd_and_dit():
    # long enough with the least attenuating ND
    assert sim07.choose_nd_and_dit(1e-4, ND_TRANS, MINDIT) == (
        "ND_OD3",
        pytest.approx(0.1),
    )
    # OD3 would need DIT 0.02 s < mindit -> OD4 at 0.2 s
    assert sim07.choose_nd_and_dit(2e-5, ND_TRANS, MINDIT) == (
        "ND_OD4",
        pytest.approx(0.2),
    )
    # too short even with OD5
    with pytest.raises(ValueError):
        sim07.choose_nd_and_dit(1e-7, ND_TRANS, MINDIT)


def test_metis_3300_steps_legal_and_unsaturated():
    # measured: BB open, ND_OD3, DIT 0.04 s -> raw peak ~2305 ADU; saturation 25000
    saturation = 25000.0
    peak_probe, t_probe, dit_probe = 2305.0, 1e-3, 0.04
    max_raw = sim_common.MAX_RAW_FRAC * saturation
    e_max = sim07.max_exposure(peak_probe, t_probe, dit_probe, max_raw, 0.03)
    peak_per_exposure = peak_probe / (t_probe * dit_probe)
    peaks = []
    for exposure in sim07.exposure_steps(e_max, 15):
        name, dit = sim07.choose_nd_and_dit(exposure, ND_TRANS, MINDIT)
        assert dit >= MINDIT
        assert ND_TRANS[name] * dit == pytest.approx(exposure)
        peaks.append(peak_per_exposure * exposure * 1.01 * 1.01)  # fail-case injection
    assert max(peaks) <= max_raw
    assert np.allclose(np.diff(peaks), peaks[0])  # smooth (linear) progression


# IMG-RAD-09: point-source noise of the on-sky background (aperture sums)


def test_aperture_radius_px():
    # lambda/D for 3.82 um on 39.146 m is ~20.1 mas, i.e. ~3.68 px at 5.47 mas/px
    radius = nom.aperture_radius_px(3.82e-6, 39.146, 5.47, 1.0)
    assert radius == pytest.approx(3.68, abs=0.01)


def test_aperture_noise_of_white_noise():
    # the sum of N independent pixels of sigma s has noise s * sqrt(N)
    rng = np.random.default_rng(1)
    frame = rng.normal(0.0, 2.0, (600, 600))
    mask = np.ones(frame.shape, bool)
    _, n_pix = nom.aperture_sums(frame, mask, 4.0, 12.0)
    noise, n_ap = nom.aperture_noise(frame, mask, 4.0, 12.0)
    assert n_ap > 1000
    assert noise == pytest.approx(2.0 * np.sqrt(n_pix), rel=0.05)


def test_rad_09_injection_scales_noise_to_injected_level():
    # as the SIM: raw = B + factor x (A - B) with factor = injected / natural, so the
    # aperture noise of raw - B (in flux units) is the injected level
    rng = np.random.default_rng(2)
    a = 1000.0 + rng.normal(0.0, 3.0, (400, 400))
    b = 1000.0 + rng.normal(0.0, 3.0, (400, 400))
    mask = np.ones(a.shape, bool)
    dit, calib = 0.3, 2.0
    natural = nom.noise_to_flux(nom.aperture_noise(a - b, mask, 3.7, 11.0)[0], dit, calib)
    injected = 1.7 * natural
    raw = b + injected / natural * (a - b)
    measured = nom.noise_to_flux(nom.aperture_noise(raw - b, mask, 3.7, 11.0)[0], dit, calib)
    assert measured == pytest.approx(injected, rel=1e-9)
