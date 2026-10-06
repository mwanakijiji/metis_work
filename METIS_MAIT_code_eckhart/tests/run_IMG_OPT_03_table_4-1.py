# Reproduces E-REP-MPIA-1203 0-1, Table 4-1 ("Pinhole vs. true point source PSF analysis")
# with our own IMG-OPT-03 fitting pipeline.
#
# The pinhole (PH) images are the ones IMG_OPT_03_METIS_AIT_img_cal_psf_quality_ANALYSIS.py reads
# in, i.e. the 'runs' in the data-states config (--data-states). Each run is matched to a manifest
# row by (filter, pp_mask); the manifest supplies the report label, d_ph_um and any PS image.
# Runs with no matching manifest row are added as extra rows (d_ph_um from the observing config),
# and manifest rows with no matching run are listed as having no PH data.
#
# For each (filter, cold stop) row, and for each of the PH and true point source (PS) images, this:
#   1. cuts out the PSF around the brightest (smoothed) pixel,
#   2. runs the production per-PSF analysis (process_one_psf, free annular-aperture fit),
#      writing free_ann_ap_best_fit_num_coord_0_fpmask_*_ppmask_*_filter_*.png into <out>/PH or <out>/PS,
#   3. collects the Gaussian-fit FWHM and the best-fit D_aper, D_obsc.
# It then writes our version of the table (same columns as the report) as CSV and Markdown.
#
# PS images are fitted with a delta-function source (no pinhole in the model); PH images with the
# pinhole diameter d_ph_um from the manifest. D_aper and D_obsc in the table come from the PS fit,
# as in the report, or from the PH fit if there is no PS image (column 'D fit from').
#
# Not collected by pytest (no test_ prefix). Usage:
#   python tests/run_IMG_OPT_03_table_4-1.py [--manifest ...] [--data-states ...] [--out ...]

import argparse
import copy
import datetime
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]  # METIS_MAIT_code_eckhart/
sys.path.insert(0, str(ROOT / "main_scripts"))
sys.path.insert(0, str(ROOT / "misc"))  # pipeline_registry

from modules.backbone_img_03_psf_quality import (  # noqa: E402
    process_one_psf,
    resolve_config_for_masks,
)
from modules.helpers import load_config_and_pipe, setup_logging  # noqa: E402
from modules.psf_grid_prep import load_fits_data  # noqa: E402
from modules.strehl_fcns import imaging_band_from_fp_mask  # noqa: E402

OVERSAMPLE_FACTOR = 3  # same as strehl_psfs
COLUMNS = [
    "filter",
    "cold stop",
    "d_ph [um]",
    "FWHM PH [mas]",
    "FWHM PS [mas]",
    "FWHM ratio PH / PS",
    "D_aper [m]",
    "D_obsc [m]",
]


def _abs_path(path):
    """Paths in the manifest may be absolute or relative to METIS_MAIT_code_eckhart/."""
    return path if os.path.isabs(path) else str(ROOT / path)


def rows_from_data_states(manifest_rows, data_states_file, config_observing):
    """
    Build the table rows from the images the IMG-OPT-03 ANALYSIS script reads in.

    INPUTS
    ----------
    manifest_rows : list of dict
        Rows of the Table 4-1 inputs manifest.
    data_states_file : str
        Data-states config (defaults + runs) read by the ANALYSIS script.
    config_observing : dict
        Observing configuration (for pinhole diameters of runs not in the manifest).

    OUTPUTS
    -------
    list of dict
        Manifest rows with ph_file / fp_mask taken from the matching run (or ph_file=None if
        there is none), followed by extra rows for runs that match no manifest row.
    """
    with open(data_states_file) as f:
        data_states_config = yaml.safe_load(f)
    defaults = data_states_config.get("defaults", {})
    runs = [{**defaults, **entry} for entry in data_states_config.get("runs", [])]
    logging.info(f"Read {len(runs)} runs from {data_states_file}")

    rows = []
    matched = set()
    for manifest_row in manifest_rows:
        row = dict(manifest_row)
        hits = [
            i
            for i, run in enumerate(runs)
            if run["filter_name"] == row["filter"] and run["pp_mask"] == row["pp_mask"]
        ]
        if row.get("skip") or not hits:
            row["ph_file"] = None
        else:
            if len(hits) > 1:
                logging.warning(
                    f"{len(hits)} runs match filter={row['filter']}, pp_mask={row['pp_mask']}; "
                    f"using the first"
                )
            run = runs[hits[0]]
            row["ph_file"] = run["file_name_abs"]
            row["fp_mask"] = run["fp_mask"]
        matched.update(hits)
        rows.append(row)

    # runs that are not rows of Table 4-1
    for i, run in enumerate(runs):
        if i in matched:
            continue
        band = imaging_band_from_fp_mask(run["fp_mask"])
        d_ph_um = config_observing["pinhole_diam_um"][band]
        row = {
            "label": run["filter_name"],
            "filter": run["filter_name"],
            "pp_mask": run["pp_mask"],
            "fp_mask": run["fp_mask"],
            "d_ph_um": d_ph_um,
            "ph_file": run["file_name_abs"],
            "ps_file": None,
        }
        if d_ph_um is None:
            row["d_ph_um"] = float("nan")
            row["skip"] = f"no pinhole_diam_um for band {band} in observing config"
        logging.info(
            f"Run not in Table 4-1 manifest, added as extra row: "
            f"filter={run['filter_name']}, pp_mask={run['pp_mask']}"
        )
        rows.append(row)
    return rows


def _local_filter_paths(leaf_names):
    """
    Filter-curve paths in the observing config ('/METIS/filters/...') resolve inside the
    podman container; outside it, fall back to the copy under inst_pkgs/.
    """
    resolved = {}
    for name, leaf in leaf_names.items():
        local = ROOT / "inst_pkgs" / leaf.lstrip("/")
        resolved[name] = leaf if os.path.exists(leaf) or not local.exists() else str(local)
    return resolved


def cutout_around_peak(image, edge):
    """
    Square cutout of side ``edge`` (native pixels) around the brightest pixel of a lightly
    smoothed copy of ``image`` (smoothing guards against single hot pixels).
    """
    image = np.asarray(image, dtype=float)
    smoothed = gaussian_filter(np.nan_to_num(image), sigma=1.0)
    y_peak, x_peak = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    y1, x1 = y_peak - edge // 2, x_peak - edge // 2
    y2, x2 = y1 + edge, x1 + edge
    if y1 < 0 or x1 < 0 or y2 > image.shape[0] or x2 > image.shape[1]:
        raise ValueError(
            f"PSF peak at (y, x)=({y_peak}, {x_peak}) is too close to the edge of a "
            f"{image.shape} image for a {edge}-pixel cutout"
        )
    return image[y1:y2, x1:x2]


def analyze_one_image(file_name, row, source_type, config_observing, out_dir):
    """
    Run the production per-PSF analysis on one PH or PS image.

    INPUTS
    ----------
    file_name : str
        FITS file with the science image in extension 1.
    row : dict
        Manifest entry (filter, pp_mask, fp_mask, d_ph_um, ...).
    source_type : str
        'PH' (finite pinhole in the model) or 'PS' (delta-function source).
    config_observing : dict
        Observing configuration as read from the YAML file.
    out_dir : str
        Top-level output directory; plots go to <out_dir>/<source_type>/.

    OUTPUTS
    -------
    dict
        FWHM [mas] and best-fit D_aper, D_obsc (with 1-sigma errors).
    """
    config_row = resolve_config_for_masks(
        config_observing, fp_mask=row["fp_mask"], pp_mask=row["pp_mask"]
    )
    band = config_row["band"]
    config_row["pinhole_diam_um"] = copy.deepcopy(config_row["pinhole_diam_um"])
    config_row["pinhole_diam_um"][band] = (
        float(row["d_ph_um"]) if source_type == "PH" else None
    )
    config_row["polychromatic_observing_filters_leaf_name"] = _local_filter_paths(
        config_row["polychromatic_observing_filters_leaf_name"]
    )

    image, _ = load_fits_data(file_name, hdu_index=1)
    cookie = cutout_around_peak(image, config_row["cutout_edge_native"])

    results_write_dir = os.path.join(out_dir, source_type)
    os.makedirs(results_write_dir, exist_ok=True)
    result = process_one_psf(
        0,
        1,
        cookie_cutout_original_this_psf=cookie,
        oversample_factor=OVERSAMPLE_FACTOR,
        filter_name=row["filter"],
        fp_mask=row["fp_mask"],
        pp_mask=row["pp_mask"],
        config_observing=config_row,
        results_write_dir=results_write_dir,
        fit_method="curve_fit",
        fit_simmed_psf=False,
        fit_annular_aperture_fixed=False,
        fit_annular_aperture_free=True,
    )

    fwhm_mas = (
        0.5
        * (result.fwhm_x_normsamp + result.fwhm_y_normsamp)
        * config_row["pixel_scale_mas"]
    )
    updates = result.strehl_updates
    logging.info(
        f"{row['label']} / {row['pp_mask']} / {source_type}: FWHM={fwhm_mas:.2f} mas, "
        f"D_aper={updates['D_aperture_fit']:.3f}, D_obsc={updates['D_obscuration_fit']:.3f}"
    )
    return {
        "fwhm_mas": fwhm_mas,
        "D_aper": updates["D_aperture_fit"],
        "D_aper_err": updates["D_aperture_err"],
        "D_obsc": updates["D_obscuration_fit"],
        "D_obsc_err": updates["D_obscuration_err"],
        "strehl_eso": updates["strehl_free_ann_ap_eso"],
    }


def _fmt(value, fmt):
    return "—" if value is None or not np.isfinite(value) else format(value, fmt)


def _fmt_pm(value, err):
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.2f} ± {err:.2f}" if np.isfinite(err) else f"{value:.2f}"


def build_table(manifest_rows, config_observing, out_dir):
    """Analyze every available PH/PS image and return the table as a DataFrame."""
    table_rows = []
    for row in manifest_rows:
        fits_out = {"PH": None, "PS": None}
        note = row.get("skip")
        if not note:
            for source_type, key in (("PH", "ph_file"), ("PS", "ps_file")):
                file_name = row.get(key)
                if not file_name:
                    continue
                file_name = _abs_path(file_name)
                if not os.path.exists(file_name):
                    logging.warning(f"{source_type} file not found: {file_name}")
                    continue
                fits_out[source_type] = analyze_one_image(
                    file_name, row, source_type, config_observing, out_dir
                )
            missing = [s for s in ("PH", "PS") if fits_out[s] is None]
            note = f"no {' / '.join(missing)} data" if missing else ""

        ph, ps = fits_out["PH"], fits_out["PS"]
        fwhm_ph = ph["fwhm_mas"] if ph else None
        fwhm_ps = ps["fwhm_mas"] if ps else None
        ratio = fwhm_ph / fwhm_ps if (ph and ps) else None
        # D_aper, D_obsc from the PS fit (as in the report) if there is one, else from the PH fit
        # (which models the finite pinhole, as the ANALYSIS script does)
        d_fit, d_source = (ps, "PS") if ps else ((ph, "PH") if ph else (None, "—"))
        table_rows.append(
            {
                "filter": row["label"],
                "cold stop": row["pp_mask"],
                "d_ph [um]": f"{float(row['d_ph_um']):.1f}",
                "FWHM PH [mas]": _fmt(fwhm_ph, ".2f"),
                "FWHM PS [mas]": _fmt(fwhm_ps, ".2f"),
                "FWHM ratio PH / PS": _fmt(ratio, ".4f"),
                "D_aper [m]": _fmt_pm(d_fit["D_aper"], d_fit["D_aper_err"]) if d_fit else "—",
                "D_obsc [m]": _fmt_pm(d_fit["D_obsc"], d_fit["D_obsc_err"]) if d_fit else "—",
                "Strehl ESO": _fmt(d_fit["strehl_eso"], ".3f") if d_fit else "—",
                "D fit from": d_source,
                "note": note,
            }
        )
    return pd.DataFrame(table_rows, columns=COLUMNS + ["Strehl ESO", "D fit from", "note"])


def write_table(df, out_dir):
    csv_path = os.path.join(out_dir, "table_4-1_ours.csv")
    md_path = os.path.join(out_dir, "table_4-1_ours.md")
    df.to_csv(csv_path, index=False)

    header = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    footer = (
        "\nOur analogue of E-REP-MPIA-1203 0-1, Table 4-1. FWHM is the mean of the x and y "
        "FWHM of a 2D Gaussian fit to the 3x-oversampled cutout (not necessarily the method "
        "used in the report). D_aper and D_obsc are from the free annular-aperture fit, with "
        "1-sigma curve_fit errors: to the true point source (PS) image if there is one (as in the "
        "report), otherwise to the pinhole (PH) image with the finite pinhole in the model "
        "(column 'D fit from'). 'Strehl ESO' is the ESO-definition Strehl (peak / total flux of the "
        "data over that of the best-fit model) from the same fit.\n"
    )
    with open(md_path, "w") as f:
        f.write("\n".join([header, rule] + body) + "\n" + footer)
    logging.info(f"Saved {csv_path}")
    logging.info(f"Saved {md_path}")
    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "config/config_file_IMG_OPT_03_table_4-1_inputs.yaml"),
    )
    parser.add_argument(
        "--data-states",
        default=str(ROOT / "config/config_file_IMG_OPT_03_psf_quality_data_states.yaml"),
        help="data-states config read by the ANALYSIS script; its runs are the PH images",
    )
    parser.add_argument(
        "--observing-config",
        default=str(ROOT / "config/config_file_IMG_OPT_03_psf_quality_observing_params.yaml"),
    )
    parser.add_argument(
        "--out", default=str(ROOT / "results/IMG_OPT_03_table_4-1/")
    )
    args = parser.parse_args()

    now = datetime.datetime.now()
    os.makedirs(args.out, exist_ok=True)
    setup_logging(
        log_dir=args.out,
        log_file_name=os.path.join(
            args.out, f"log_table_4-1_{now.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
        ),
        now=now,
    )

    config_observing = load_config_and_pipe(
        config_file_choice=args.observing_config, print_one_line=False
    )
    with open(args.manifest) as f:
        manifest_rows = yaml.safe_load(f)["rows"]
    rows = rows_from_data_states(manifest_rows, args.data_states, config_observing)

    df = build_table(rows, config_observing, args.out)
    write_table(df, args.out)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
