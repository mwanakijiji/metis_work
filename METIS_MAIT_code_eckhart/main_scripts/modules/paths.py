"""
Machine-specific paths (project root, ScopeSim IRDB), set in config/paths.yaml so that
switching machines (host <-> podman container) means changing one line there.
"""

import functools
import os

import yaml

PATHS_FILE = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "paths.yaml"
    )
)

@functools.lru_cache(maxsize=None)
def _load_paths():
    """Load (once) the paths config from PATHS_FILE."""
    try:
        with open(PATHS_FILE, "r", encoding="utf-8") as paths_file:
            paths = yaml.safe_load(paths_file) or {}
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"Paths config {PATHS_FILE} not found; create it with a "
            "'project_root:' line"
        ) from err
    if not paths.get("project_root"):
        raise KeyError(f"No 'project_root' set in {PATHS_FILE}")
    return paths


def project_root():
    """
    The project root (METIS_MAIT_code_eckhart/), as set in config/paths.yaml.

    OUTPUTS
    -------
    str
        Absolute path, ending in '/' (scripts append to it, e.g. stem + "data/").
    """
    return os.path.join(_load_paths()["project_root"], "")


def irdb_dir():
    """
    The ScopeSim IRDB directory, as set in config/paths.yaml (relative to the project
    root, or absolute).

    OUTPUTS
    -------
    str
        Absolute path, ending in '/'.
    """
    irdb = _load_paths().get("irdb_dir", "irdb/")
    return os.path.join(project_root(), irdb, "")
