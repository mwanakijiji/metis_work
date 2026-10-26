"""
The observing config common to all IMG AIT tests, with each test's own observing config
merged on top.
"""

import logging
import os

from .helpers import load_config_and_pipe

COMMON_OBSERVING_CONFIG = "config/config_file_common_observing_params.yaml"


def load_observing_config(stem, *test_config_files):
    """
    Load the common observing config, then merge each test-specific config on top, in
    order (top-level keys of later files override earlier ones).

    INPUTS
    ----------
    stem : str
        Project root (METIS_MAIT_code_eckhart/), prepended to relative paths.
    *test_config_files : str
        Test-specific observing configs, relative to stem or absolute.

    OUTPUTS
    -------
    dict
        The merged observing config.
    """
    merged = {}
    for file_name in (COMMON_OBSERVING_CONFIG,) + test_config_files:
        path = file_name if os.path.isabs(file_name) else os.path.join(stem, file_name)
        config = load_config_and_pipe(config_file_choice=path, print_one_line=False)
        overridden = sorted(set(merged) & set(config or {}))
        if overridden:
            logging.info("Keys overridden by %s: %s", path, overridden)
        merged.update(config or {})
    return merged
