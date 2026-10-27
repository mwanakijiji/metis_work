#!/usr/bin/env python3
"""Render the generate-sims pipeline graph as a Graphviz flowchart (DOT + SVG).

The graph is collected from ``@pipeline_stage`` decorators via
``pipeline_stages`` / ``METIS_MAIT_code_eckhart.misc.pipeline_registry``.

Layout (rankdir=LR, so a rank is a column):
- columns: each stage's column is its depth in the graph (longest path from a
  source), and stages of the same depth share a column across all tests
  (``rank=same`` with ``newrank``), e.g. all SIMs in column 0;
- rows: one row per test cluster, ordered by test number (IMG-OPT-01, -02, ...),
  enforced by invisible edges between the stages of consecutive rows in each column;
  a shared cluster (label not 'IMG-OPT-NN ...' / 'IMG-RAD-NN ...', e.g. the
  stray-light common stages) gets its own row right after the first test feeding it;
- arrows within a test are heavy (straight rows); arrows between clusters (e.g. into
  and out of shared machinery) are dashed, grey and do not constrain the layout.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

from pipeline_stages import CLUSTERS, EDGES, NODES, PLANNED_STAGES

HERE = Path(__file__).resolve().parent

# characters per line in box labels
LABEL_WRAP = 24
# 'IMG-OPT-04 Stray light' -> 4
_TEST_NUMBER = re.compile(r"^IMG-(?:OPT|RAD)-(\d+)\b")
# pairs of test numbers whose rows are swapped after sorting (by hand: Graphviz does
# not stack the rows strictly in list order, so this is tuned to how the chart renders)
ROW_SWAPS = ((4, 6),)


def _wrap(label: str) -> str:
    """Wrap a box label onto several lines (DOT line breaks)."""
    return "\\n".join(textwrap.wrap(label, LABEL_WRAP)) or label


def _node_cluster() -> dict[str, str]:
    """Cluster id of each node (nodes without a cluster are left out)."""
    return {
        node_id: cluster_id
        for cluster_id, meta in CLUSTERS.items()
        for node_id in meta["nodes"]
    }


def _columns() -> dict[str, int]:
    """Depth of each node: longest path from a source node (0 for sources)."""
    parents: dict[str, list[str]] = {node_id: [] for node_id in NODES}
    for src, dst in EDGES:
        parents[dst].append(src)

    depth: dict[str, int] = {}

    def _depth(node_id: str, visiting: frozenset = frozenset()) -> int:
        if node_id in depth:
            return depth[node_id]
        if node_id in visiting:  # cycle guard (should not happen in a pipeline)
            return 0
        value = max(
            (_depth(p, visiting | {node_id}) + 1 for p in parents[node_id]),
            default=0,
        )
        depth[node_id] = value
        return value

    for node_id in NODES:
        _depth(node_id)
    return depth


def _row_order(node_cluster: dict[str, str]) -> list[str]:
    """
    Cluster ids, top to bottom: tests by number; each shared cluster right after the
    lowest-numbered test cluster feeding into it; anything else last; then the rows of
    each pair of tests in ROW_SWAPS swapped.
    """

    def _test_number(cluster_id: str) -> int | None:
        match = _TEST_NUMBER.match(CLUSTERS[cluster_id]["label"])
        return int(match.group(1)) if match else None

    keys = {}
    for cluster_id, meta in CLUSTERS.items():
        number = _test_number(cluster_id)
        if number is not None:
            keys[cluster_id] = (number, 0, meta["label"])
            continue
        # shared cluster: after the lowest-numbered test feeding into it
        feeders = [
            _test_number(node_cluster[src])
            for src, dst in EDGES
            if node_cluster.get(dst) == cluster_id
            and src in node_cluster
            and node_cluster[src] != cluster_id
        ]
        feeders = [n for n in feeders if n is not None]
        keys[cluster_id] = (min(feeders) if feeders else 10**6, 1, meta["label"])
    rows = sorted(CLUSTERS, key=lambda cluster_id: keys[cluster_id])

    row_of_test = {_test_number(c): i for i, c in enumerate(rows) if _test_number(c)}
    for num_a, num_b in ROW_SWAPS:
        if num_a in row_of_test and num_b in row_of_test:
            i, j = row_of_test[num_a], row_of_test[num_b]
            rows[i], rows[j] = rows[j], rows[i]
    return rows


def build_dot() -> str:
    node_cluster = _node_cluster()
    columns = _columns()
    rows = _row_order(node_cluster)
    row_index = {cluster_id: i for i, cluster_id in enumerate(rows)}
    no_row = len(rows)  # nodes without a cluster go below all rows

    lines = [
        "digraph pipeline {",
        "  rankdir=LR;",
        "  newrank=true;",
        "  splines=ortho;",
        "  nodesep=0.35;",
        "  ranksep=0.7;",
        '  node [shape=box, style="rounded,filled", fillcolor="white", width=2.2, '
        "fontsize=11];",
        '  edge [color="#444444"];',
        "",
    ]

    # clusters, one per row, in row order
    for cluster_id in rows:
        meta = CLUSTERS[cluster_id]
        lines.append(f"  subgraph cluster_{cluster_id} {{")
        lines.append(f'    label="{meta["label"]}";')
        lines.append("    labeljust=l;")
        lines.append('    style="dashed";')
        lines.append(f'    color="{meta["color"]}";')
        for node_id in meta["nodes"]:
            lines.append(f"    {node_id};")
        lines.append("  }")
        lines.append("")

    # nodes
    for node_id, label in NODES.items():
        if node_id in PLANNED_STAGES:
            lines.append(
                f'  {node_id} [label="{_wrap(label)}", style="rounded,dashed,filled", '
                f'fillcolor="#f0f0f0", fontcolor="#666666"];'
            )
        else:
            lines.append(f'  {node_id} [label="{_wrap(label)}"];')
    lines.append("")

    # columns: same depth -> same rank; within a column, invisible edges between
    # consecutive rows fix the top-to-bottom order of the rows
    by_column: dict[int, list[str]] = {}
    for node_id, column in columns.items():
        by_column.setdefault(column, []).append(node_id)
    for column in sorted(by_column):
        nodes_here = sorted(
            by_column[column],
            key=lambda n: (row_index.get(node_cluster.get(n), no_row), n),
        )
        lines.append(f"  {{ rank=same; {'; '.join(nodes_here)}; }}")
        for upper, lower in zip(nodes_here, nodes_here[1:]):
            lines.append(f"  {upper} -> {lower} [style=invis];")
    lines.append("")

    # real edges: heavy within a cluster (straight rows); between clusters dashed,
    # grey and not constraining the layout (avoids spaghetti)
    for src, dst in EDGES:
        if node_cluster.get(src) == node_cluster.get(dst):
            lines.append(f"  {src} -> {dst} [weight=10];")
        else:
            lines.append(
                f'  {src} -> {dst} [style=dashed, color="#999999", constraint=false];'
            )
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
