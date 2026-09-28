#!/usr/bin/env python3
"""Render the generate-sims pipeline graph as a Graphviz flowchart (DOT + SVG).

The graph is collected from ``@pipeline_stage`` decorators via
``pipeline_stages`` / ``METIS_MAIT_code_eckhart.misc.pipeline_registry``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pipeline_stages import CLUSTERS, EDGES, NODES, PLANNED_STAGES

HERE = Path(__file__).resolve().parent


def build_dot() -> str:
    lines = [
        "digraph pipeline {",
        "  rankdir=LR;",
        '  node [shape=box, style="rounded,filled", fillcolor="white"];',
        '  edge [color="#444444"];',
        "",
    ]
    for cluster_id, meta in CLUSTERS.items():
        lines.append(f"  subgraph cluster_{cluster_id} {{")
        lines.append(f'    label="{meta["label"]}";')
        lines.append('    style="dashed";')
        lines.append(f'    color="{meta["color"]}";')
        for node_id in meta["nodes"]:
            lines.append(f"    {node_id};")
        lines.append("  }")
        lines.append("")

    for node_id, label in NODES.items():
        if node_id in PLANNED_STAGES:
            lines.append(
                f'  {node_id} [label="{label}", style="rounded,dashed,filled", '
                f'fillcolor="#f0f0f0", fontcolor="#666666"];'
            )
        else:
            lines.append(f'  {node_id} [label="{label}"];')
    lines.append("")
    for src, dst in EDGES:
        lines.append(f"  {src} -> {dst};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    dot_path = HERE / "pipeline_flow.dot"
    svg_path = HERE / "pipeline_flow.svg"
    dot_path.write_text(build_dot(), encoding="utf-8")
    print(f"Wrote {dot_path}")

    try:
        subprocess.run(
            ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
            check=True,
        )
    except FileNotFoundError:
        print(
            "graphviz `dot` not found; DOT written but SVG not generated",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"`dot` failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
