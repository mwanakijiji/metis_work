# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Simulation and analysis code for METIS (ELT mid-IR instrument) AIT (assembly, integration
and test) of the imager (IMG). The maintained code is in `METIS_MAIT_code_eckhart/`; most
other top-level directories (`playing_with_scopesim/`, `metis-drs-setup/`, `met-resource/`,
notebooks, `misc/old_*`) are scratch, reference material or third-party setup.

## Environment

- Everything runs inside the podman container `edps-jupyter` (Python + ScopeSim 0.11.2,
  photutils, scikit-image, OpenCV; see `METIS_MAIT_code_eckhart/requirements.txt`). The
  host's `~/Desktop` is mounted at `/podman-share`, and scripts hard-code paths such as
  `/podman-share/metis_work/METIS_MAIT_code_eckhart/`. The host Python lacks these
  packages, so IDE "import could not be resolved" warnings are expected.
- The ScopeSim IRDB (`METIS/`, `ELT/`, `Armazones/` packages) lives in `~/Desktop`
  (= `/podman-share`); SIM scripts call `sim.link_irdb("../../../")` relative to
  `main_scripts/`, so run them from that directory. Keep `sim.download_packages(...)`
  commented out: it re-extracts the packages over the IRDB on every run.
- The share is on iCloud: `OSError: [Errno 35] Resource deadlock avoided` means an
  evicted (dataless) file; read it on the host (e.g. `cat file > /dev/null`) to fetch it.
- A bare `Killed` with no traceback is the container OOM killer; full-frame images are
  2048x2048, so avoid one full-frame array per detected object.

## Commands (run in the container)

```bash
cd /podman-share/metis_work/METIS_MAIT_code_eckhart

# tests (each test file puts main_scripts/ and misc/ on sys.path itself)
pytest tests/
pytest tests/test_out_of_field.py::test_not_detected_too_shallow

# a test's simulation, then its analysis (run from main_scripts/)
cd main_scripts
python IMG_OPT_06_METIS_AIT_out_of_field_SIM.py
python IMG_OPT_06_METIS_AIT_out_of_field_ANALYSIS.py
```

Linting is done by the git pre-commit hook (`.git/hooks/pre-commit`, run from the repo
root): `black -l 88` on `main_scripts/`, then `flake8` and `pylint` with
`--max-line-length=88` (excluding tests, misc, dev_notebooks). The same hook regenerates
`docs/uml/classes.{dot,svg}` (pyreverse on `main_scripts/modules`) and
`docs/uml/pipeline_flow.{dot,svg}` (`docs/uml/render_pipeline_flow.py`) and `git add`s
them.

## Architecture

Each AIT test case `IMG_OPT_NN` / `IMG_RAD_NN` (named after the test descriptions, whose
requirement texts are quoted in the script docstrings) has a pair of scripts in
`main_scripts/`:

- `*_SIM.py`: builds a ScopeSim `OpticalTrain` in a WCU (warm calibration unit) mode
  (`wcu_img_lm` / `wcu_img_n`), with a WCU focal-plane mask (e.g. `pinhole_lm`), a pupil
  mask (e.g. `PPS-CFO2`, using its `_WCU` PSF kernel) and a filter; reads out a
  background (black-body aperture closed) and a science frame (aperture open), may add
  synthetic artefacts, and writes FITS files. FITS layout convention: header-only
  primary, then `RAW_READOUT`, `BCKGD_SUBTED`, `BACKGROUND` (+ extras such as
  `BACKGROUND_2`), with test metadata in the primary header.
- `*_ANALYSIS.py`: thin `main()` that loads YAML configs, loops over "data states" (one
  per FITS file) and calls backbone functions in order, logging pass/fail per
  requirement to a timestamped log file.
- `modules/backbone_img_*.py`: the actual analysis, one module per test. Shared helpers:
  `modules/helpers.py` (`setup_logging`, `load_config_and_pipe`, `pipe_2_log`) and
  `modules/psf_grid_prep.py` (`load_fits_data`, which works around the Errno 35 issue;
  2-pass oversampled centroiding).

Data flow and conventions:
- `config/config_file_IMG_<NN>_*_{observing,data_states,coords_guesses}.yaml`:
  observing parameters (filter wavelengths, pixel scales, D_aperture, sensitivities,
  thresholds), the list of FITS files with per-file analysis options, and PSF position
  guesses. Data-state `file_name`s are relative to `METIS_MAIT_code_eckhart/`; simulated
  data for analysis go in `data/IMG_<NN>_simmed_*_data/`. IMG-06 merges its observing
  config on top of IMG-04's.
- Analysis results (plots, FITS) go to `results/IMG_<NN>_*_results/`.
- Results are carried through the pipeline as dataclasses mutated by each stage, e.g.
  `StrayLightResult` in `backbone_img_04_stray_light.py`, subclassed by
  `OutOfFieldResult` in `backbone_img_06_out_of_field.py`. IMG-06 reuses the IMG-04
  centroiding, PSF masking, segmentation and noise machinery.
- Stray-light segmentation (IMG-04/06) runs several methods (threshold, hough_circle,
  active_contour, watershed, kmeans, meanshift) and combines them by a weighted vote set
  per data state. Regions are kept as one int32 label map (`segment_map`), iterated with
  `iter_regions()` per bounding box.
- Pass/fail against noise must be statistically honest: METIS-9522 compares a region's
  total flux with 3-sigma of the noise of a sum over the same area (`area_noise_sigma`);
  IMG-06 METIS-5838 requires a significant detection (noise from a null frame,
  `BACKGROUND_2 - BACKGROUND`) and otherwise reports a lower limit / INCONCLUSIVE.

### Pipeline registry

`misc/pipeline_registry.py` provides the `@pipeline_stage(name=..., depends_on=...,
cluster=..., cluster_color=..., label=...)` decorator used on SIM and backbone functions.
The docs tooling does not import the scripts; it parses decorators with `ast`, so
decorator arguments must be literals or names bound to module-level literals in the
same file (e.g. `CLUSTER_OOF = "..."` defined in the script itself), or the `CLUSTER_*` /
`COLOR_*` constants in the registry. Every `depends_on` id must be an existing stage
name, or rendering the flow diagram fails.

### Constitution (non-negotiable)

### Lessons learned
