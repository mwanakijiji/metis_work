# Modules package for PSF/Strehl analysis.
# Put METIS_MAIT_code_eckhart/misc on sys.path so
# ``from pipeline_registry import pipeline_stage`` works.
import sys
from pathlib import Path

# __file__ is main_scripts/modules/__init__.py → parents[2] is
# METIS_MAIT_code_eckhart/
# this directory allows import of pipeline_registry
_MISC = Path(__file__).resolve().parents[2] / "misc"
if str(_MISC) not in sys.path:
    sys.path.insert(0, str(_MISC))
