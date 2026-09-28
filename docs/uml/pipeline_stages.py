"""Pipeline stage graph for docs/uml flow diagrams.

Built from ``@pipeline_stage`` annotations on the real METIS AIT callables
(see ``METIS_MAIT_code_eckhart.misc.pipeline_registry``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_MISC = Path(__file__).resolve().parents[2] / "METIS_MAIT_code_eckhart" / "misc"
if str(_MISC) not in sys.path:
    sys.path.insert(0, str(_MISC))

from pipeline_registry import graph_from_registry  # noqa: E402

NODES, EDGES, CLUSTERS, PLANNED_STAGES, STAGES = graph_from_registry()

__all__ = [
    "CLUSTERS",
    "EDGES",
    "NODES",
    "PLANNED_STAGES",
    "STAGES",
]
