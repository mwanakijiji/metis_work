"""
Lightweight pipeline-stage registry (Option 2).

Decorate the real callables with ``@pipeline_stage(...)``; docs tooling walks
the registry to build the flowchart. Script ``main()`` functions keep their
own control flow — this module does not orchestrate execution.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Cluster metadata shared by decorators and fallback registrations.
CLUSTER_FOV = "IMG-OPT-01 FOV"
CLUSTER_DISTORTION = "IMG-OPT-02 Distortion"
CLUSTER_PSF = "IMG-OPT-03 PSF quality"
CLUSTER_STRAY = "IMG-OPT-04 Stray light"
CLUSTER_OOF = "IMG-OPT-06 Out-of-field stray light"
CLUSTER_PHOT = "IMG-RAD-07 Photometric performance"
COLOR_FOV = "#1f77b4"
COLOR_DISTORTION = "#ff7f0e"
COLOR_PSF = "#2ca02c"
COLOR_STRAY = "#d62728"
COLOR_OOF = "#8c564b"
COLOR_PHOT = "#9467bd"


@dataclass(frozen=True)
class StageSpec:
    """Metadata for one METIS AIT pipeline stage."""

    id: str
    label: str
    depends_on: Tuple[str, ...] = ()
    cluster: Optional[str] = None
    cluster_color: Optional[str] = None
    planned: bool = False
    qualname: Optional[str] = None


_REGISTRY: Dict[str, StageSpec] = {}
_IMPORTS_DONE = False


def pipeline_stage(
    *,
    name: Optional[str] = None,
    depends_on: Sequence[str] = (),
    cluster: Optional[str] = None,
    cluster_color: Optional[str] = None,
    planned: bool = False,
    label: Optional[str] = None,
):
    """
    Register a callable as a pipeline stage for docs / discovery.

    ``depends_on`` lists immediate predecessor stage ids (DAG parents).
    """

    def decorator(fn):
        stage_id = name or fn.__name__
        register_stage(
            stage_id,
            depends_on=depends_on,
            cluster=cluster,
            cluster_color=cluster_color,
            planned=planned,
            label=label or stage_id,
            qualname=getattr(fn, "__qualname__", stage_id),
        )
        return fn

    return decorator


def register_stage(
    stage_id: str,
    *,
    depends_on: Sequence[str] = (),
    cluster: Optional[str] = None,
    cluster_color: Optional[str] = None,
    label: Optional[str] = None,
    planned: bool = False,
    qualname: Optional[str] = None,
) -> None:
    """Register a stage by id (implemented or planned)."""
    _REGISTRY[stage_id] = StageSpec(
        id=stage_id,
        label=label or stage_id,
        depends_on=tuple(depends_on),
        cluster=cluster,
        cluster_color=cluster_color,
        planned=planned,
        qualname=qualname,
    )


def register_planned_stage(
    stage_id: str,
    *,
    depends_on: Sequence[str] = (),
    cluster: Optional[str] = None,
    cluster_color: Optional[str] = None,
    label: Optional[str] = None,
) -> None:
    """Register a stage that is not yet implemented as a real callable."""
    register_stage(
        stage_id,
        depends_on=depends_on,
        cluster=cluster,
        cluster_color=cluster_color,
        label=label,
        planned=True,
        qualname=None,
    )


def get_registry() -> Dict[str, StageSpec]:
    return dict(_REGISTRY)


def clear_registry() -> None:
    """Test helper."""
    global _IMPORTS_DONE
    _REGISTRY.clear()
    _IMPORTS_DONE = False


def _register_planned_defaults() -> None:
    """Stages that are planned but not yet implemented (drawn dashed)."""
    # none at the moment; add with register_planned_stage(...) if needed


_DECORATOR_KWARGS = ("name", "depends_on", "cluster", "cluster_color", "planned", "label")


def _literal(node: ast.expr, constants: Dict[str, object]):
    """Evaluate a decorator argument: a literal, or a name bound to a literal."""
    if isinstance(node, ast.Name):
        if node.id not in constants:
            raise ValueError(f"cannot resolve name {node.id!r}")
        return constants[node.id]
    return ast.literal_eval(node)


def _stages_in_file(path: Path) -> List[Dict]:
    """Read the ``@pipeline_stage(...)`` decorators in one source file, without
    importing it (the SIM scripts and backbones need ScopeSim, cv2, ... at import)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # names usable in decorator arguments: this registry's cluster/colour constants,
    # plus module-level literal assignments in the file itself (e.g. CLUSTER_PHOT)
    constants: Dict[str, object] = {
        k: v for k, v in globals().items() if k.startswith(("CLUSTER_", "COLOR_"))
    }
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            try:
                constants[stmt.targets[0].id] = ast.literal_eval(stmt.value)
            except ValueError:
                pass

    stages = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if not (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Name)
                and dec.func.id == "pipeline_stage"
            ):
                continue
            kwargs = {
                kw.arg: _literal(kw.value, constants)
                for kw in dec.keywords
                if kw.arg in _DECORATOR_KWARGS
            }
            kwargs["qualname"] = node.name
            stages.append(kwargs)
    return stages


def _register_from_source() -> None:
    """Register every ``@pipeline_stage`` in main_scripts/ by parsing the source."""
    main_scripts = Path(__file__).resolve().parent.parent / "main_scripts"
    paths = sorted(main_scripts.glob("*.py")) + sorted(
        (main_scripts / "modules").glob("*.py")
    )
    for path in paths:
        for kw in _stages_in_file(path):
            stage_id = kw.get("name") or kw["qualname"]
            register_stage(
                stage_id,
                depends_on=kw.get("depends_on", ()),
                cluster=kw.get("cluster"),
                cluster_color=kw.get("cluster_color"),
                label=kw.get("label") or stage_id,
                planned=kw.get("planned", False),
                qualname=kw["qualname"],
            )


def ensure_stages_registered() -> Dict[str, StageSpec]:
    """Populate the registry from the ``@pipeline_stage`` decorators in the source."""
    global _IMPORTS_DONE
    if not _IMPORTS_DONE:
        _register_from_source()
        _register_planned_defaults()
        _IMPORTS_DONE = True
    return get_registry()


def graph_from_registry() -> Tuple[
    Dict[str, str],
    List[Tuple[str, str]],
    Dict[str, Dict],
    frozenset,
    List[str],
]:
    """
    Build NODES, EDGES, CLUSTERS, PLANNED_STAGES, STAGES from the registry.

    Edges are ``(predecessor, stage)`` for each ``depends_on`` entry.
    """
    registry = ensure_stages_registered()
    nodes = {stage_id: spec.label for stage_id, spec in registry.items()}
    edges: List[Tuple[str, str]] = []
    for stage_id, spec in registry.items():
        for parent in spec.depends_on:
            if parent not in registry:
                raise KeyError(
                    f"Stage {stage_id!r} depends on unknown stage {parent!r}"
                )
            edges.append((parent, stage_id))

    clusters: Dict[str, Dict] = {}
    for stage_id, spec in registry.items():
        if not spec.cluster:
            continue
        # Stable cluster key from label (slug)
        key = (
            spec.cluster.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
        )
        if key not in clusters:
            clusters[key] = {
                "label": spec.cluster,
                "color": spec.cluster_color or "#666666",
                "nodes": [],
            }
        clusters[key]["nodes"].append(stage_id)
        if spec.cluster_color:
            clusters[key]["color"] = spec.cluster_color

    planned = frozenset(s.id for s in registry.values() if s.planned)
    stages = list(nodes.values())
    return nodes, edges, clusters, planned, stages
