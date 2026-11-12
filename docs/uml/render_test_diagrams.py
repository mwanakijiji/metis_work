#!/usr/bin/env python3
"""Render per-test diagrams of the SIM and ANALYSIS scripts (DOT + SVG, + Markdown).

For one AIT test (e.g. ``IMG_OPT_06``) this writes to ``docs/uml/tests/``:

- ``<test>_callgraph_sim`` / ``<test>_callgraph_analysis``: the calls from the
  script's ``main()`` into ``main_scripts/`` (stages = boxes, helpers = ovals,
  with the first sentence of each docstring; the full docstring is the SVG tooltip);
- ``<test>_dataflow``: which ANALYSIS stage writes (solid) / reads (dashed) which field
  of the result dataclasses (fields grouped by class, inherited ones separately);
- ``<test>_io``: config YAMLs -> scripts, the FITS layout the SIM writes (extensions,
  primary-header cards), which ANALYSIS stage reads which extension / card, and the
  output directories;
- ``<test>_overview``: the test's slice of the pipeline-stage graph above a table of
  the stages (module, purpose, fields written), plus ``<test>_overview.md`` with the
  same table and links to the source lines.

Like ``render_pipeline_flow.py``, nothing is imported: the source is parsed with
``ast`` (the scripts need ScopeSim, cv2, ... at import). Calls are resolved through
the module's imports (``import modules.x as y``, ``from modules import x``, ``from
modules.x import f``), method calls through the (unique) method names of the project's
classes, and functions passed as values (e.g. the SIMULATORS dict given to
``sim_common.run_scenario``) are drawn as dashed "passed as callback" edges. Static
analysis is a heuristic: dynamic dispatch (getattr, dict lookups at run time) is not
followed.

Usage (from anywhere)::

    python docs/uml/render_test_diagrams.py IMG_OPT_06 [IMG_OPT_03 ...]
    python docs/uml/render_test_diagrams.py --all
"""

from __future__ import annotations

import argparse
import ast
import html
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CODE = REPO / "METIS_MAIT_code_eckhart"
MAIN_SCRIPTS = CODE / "main_scripts"
MODULES = MAIN_SCRIPTS / "modules"
OUT_DIR = HERE / "tests"

sys.path.insert(0, str(CODE / "misc"))
from pipeline_registry import (  # noqa: E402  pylint: disable=wrong-import-position
    ensure_stages_registered,
)

# levels of calls shown below main() in the call graphs (functions defined in the
# script itself do not count, so local wrappers do not eat the budget)
CALL_DEPTH = 2
WRAP = 34  # characters per line of a purpose text in a node
SHARED_MODULES = {"stray_light_common", "strehl_fcns", "analysis_common"}
INFRA_MODULES = {
    "helpers",
    "config_common",
    "paths",
    "psf_grid_prep",
    "sim_common",
    "tbd_warnings",
    "amoeba",
}
FILL_SCRIPT = "#f2f2f2"
FILL_INFRA = "#e8f0fb"
FILL_SHARED = "#fde3e3"
FILL_FALLBACK = "#f5f0e6"
UPPER_KEY = re.compile(r"^[A-Z][A-Z0-9_<>]{1,15}$")
OUT_PATH = re.compile(
    r"(?:^|/)((?:results|logs|data)/[^\"']*|[\w]+_(?:results|logs))/?$"
)


# --------------------------------------------------------------------------------------
# source index
# --------------------------------------------------------------------------------------


@dataclass
class Module:
    """One parsed source file."""

    key: str  # 'stray_light_common' for modules/, the file stem for scripts
    path: Path
    tree: ast.Module
    functions: dict[str, ast.FunctionDef] = field(default_factory=dict)
    classes: dict[str, ast.ClassDef] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)  # local name -> module key
    imported: dict[str, tuple[str, str]] = field(
        default_factory=dict
    )  # name -> (mod, name)
    collections: dict[str, list[str]] = field(
        default_factory=dict
    )  # NAME -> [func names]
    stages: dict[str, str] = field(default_factory=dict)  # function name -> stage id

    @property
    def is_script(self) -> bool:
        return self.path.parent == MAIN_SCRIPTS


@dataclass
class ClassInfo:
    """A (data)class: its own fields, base class names, properties, methods."""

    module: str
    name: str
    fields: list[str]
    bases: list[str]
    properties: list[str]
    methods: dict[str, ast.FunctionDef]


Func = tuple[str, str]  # (module key, function name)


class Index:
    """All of main_scripts/ (scripts and modules/), parsed once."""

    def __init__(self) -> None:
        self.modules: dict[str, Module] = {}
        for path in sorted(MAIN_SCRIPTS.glob("*.py")) + sorted(MODULES.glob("*.py")):
            key = path.stem
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            self.modules[key] = self._index_module(key, path, tree)
        self.classes: dict[str, ClassInfo] = {}
        for mod in self.modules.values():
            for name, node in mod.classes.items():
                self.classes.setdefault(name, self._class_info(mod.key, node))
        # method name -> owner, only for names defined by exactly one project class
        owners: dict[str, list[Func]] = {}
        for info in self.classes.values():
            for meth in info.methods:
                if not meth.startswith("__"):
                    owners.setdefault(meth, []).append(
                        (info.module, f"{info.name}.{meth}")
                    )
        self.methods = {m: o[0] for m, o in owners.items() if len(o) == 1}
        self.stage_spec = ensure_stages_registered()

    @staticmethod
    def _index_module(key: str, path: Path, tree: ast.Module) -> Module:
        mod = Module(key=key, path=path, tree=tree)
        for stmt in tree.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                mod.functions[stmt.name] = stmt
                for dec in stmt.decorator_list:
                    if (
                        isinstance(dec, ast.Call)
                        and isinstance(dec.func, ast.Name)
                        and dec.func.id == "pipeline_stage"
                    ):
                        name = next(
                            (
                                kw.value.value
                                for kw in dec.keywords
                                if kw.arg == "name"
                                and isinstance(kw.value, ast.Constant)
                            ),
                            stmt.name,
                        )
                        mod.stages[stmt.name] = name
            elif isinstance(stmt, ast.ClassDef):
                mod.classes[stmt.name] = stmt
            elif isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.name.startswith("modules."):
                        target = alias.name.split(".", 1)[1]
                        mod.aliases[alias.asname or target] = target
            elif isinstance(stmt, ast.ImportFrom):
                if stmt.module == "modules":
                    for alias in stmt.names:
                        mod.aliases[alias.asname or alias.name] = alias.name
                elif stmt.module and stmt.module.startswith("modules."):
                    target = stmt.module.split(".", 1)[1]
                    for alias in stmt.names:
                        mod.imported[alias.asname or alias.name] = (target, alias.name)
            elif (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, (ast.Dict, ast.List, ast.Tuple))
            ):
                values = (
                    stmt.value.values
                    if isinstance(stmt.value, ast.Dict)
                    else stmt.value.elts
                )
                names = [v.id for v in values if isinstance(v, ast.Name)]
                if names:
                    mod.collections[stmt.targets[0].id] = names
        return mod

    @staticmethod
    def _class_info(module: str, node: ast.ClassDef) -> ClassInfo:
        fields, properties, methods = [], [], {}
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                fields.append(stmt.target.id)
            elif isinstance(stmt, ast.FunctionDef):
                is_prop = any(
                    isinstance(d, ast.Name) and d.id in ("property", "cached_property")
                    for d in stmt.decorator_list
                )
                if is_prop:
                    properties.append(stmt.name)
                else:
                    methods[stmt.name] = stmt
        bases = [
            b.attr if isinstance(b, ast.Attribute) else getattr(b, "id", "")
            for b in node.bases
        ]
        return ClassInfo(module, node.name, fields, bases, properties, methods)

    # ---- lookups ---------------------------------------------------------------

    def func_node(self, func: Func) -> ast.FunctionDef | None:
        mod_key, name = func
        mod = self.modules.get(mod_key)
        if mod is None:
            return None
        if "." in name:  # Class.method
            cls, meth = name.split(".", 1)
            info = self.classes.get(cls)
            return info.methods.get(meth) if info else None
        return mod.functions.get(name)

    def stage_of(self, func: Func) -> str | None:
        mod = self.modules.get(func[0])
        return mod.stages.get(func[1]) if mod else None

    def resolve_name(self, mod: Module, name: str) -> tuple[str, Func] | None:
        """('func'|'class', (module, name)) for a bare name in module ``mod``."""
        if name in mod.functions:
            return "func", (mod.key, name)
        if name in mod.classes:
            return "class", (mod.key, name)
        if name in mod.imported:
            target, real = mod.imported[name]
            tmod = self.modules.get(target)
            if tmod and real in tmod.functions:
                return "func", (target, real)
            if tmod and real in tmod.classes:
                return "class", (target, real)
        return None

    def resolve_call(self, mod: Module, func: ast.expr) -> tuple[str, Func] | None:
        if isinstance(func, ast.Name):
            return self.resolve_name(mod, func.id)
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id in mod.aliases:
                tmod = self.modules.get(mod.aliases[func.value.id])
                if tmod is None:
                    return None
                if func.attr in tmod.functions:
                    return "func", (tmod.key, func.attr)
                if func.attr in tmod.classes:
                    return "class", (tmod.key, func.attr)
                return None
            if func.attr in self.methods:
                return "func", self.methods[func.attr]
        return None

    def fields_of(self, cls: str) -> list[tuple[str, str]]:
        """(owning class, field) for a class and its bases, base-class fields first."""
        info = self.classes.get(cls)
        if info is None:
            return []
        out: list[tuple[str, str]] = []
        for base in info.bases:
            out += self.fields_of(base)
        out += [(cls, f) for f in info.fields + info.properties]
        return out


def _ordered(nodes) -> list:
    return sorted(
        nodes, key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))
    )


def _body_nodes(fn: ast.FunctionDef) -> list[ast.AST]:
    """Nodes of a function body, skipping decorators and the docstring."""
    out = []
    for stmt in fn.body:
        out.extend(ast.walk(stmt))
    return _ordered(out)


def docstring(fn: ast.AST | None) -> str:
    return (ast.get_docstring(fn) or "").strip() if fn is not None else ""


def purpose(fn: ast.AST | None, limit: int = 110) -> str:
    """First sentence of a docstring (first paragraph, joined)."""
    first_par = docstring(fn).split("\n\n")[0]
    text = " ".join(first_par.split())
    match = re.search(r"(?<=[a-z0-9)\]])\.(\s|$)", text)
    if match:
        text = text[: match.start()]
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------------------
# call graph
# --------------------------------------------------------------------------------------


@dataclass
class CallGraph:
    root: Func
    nodes: dict[Func, int] = field(default_factory=dict)  # func -> depth
    classes: list[Func] = field(default_factory=list)  # instantiated classes
    edges: list[tuple[Func, Func, str]] = field(default_factory=list)  # src, dst, kind
    order: dict[Func, int] = field(default_factory=dict)  # call order of main's callees


def callees(index: Index, func: Func) -> list[tuple[str, str, Func]]:
    """(kind, 'call'|'callback', target) in source order, unique per target."""
    fn = index.func_node(func)
    if fn is None:
        return []
    mod = index.modules[func[0]]
    out, seen = [], set()
    call_funcs = set()
    for node in _body_nodes(fn):
        if isinstance(node, ast.Call):
            call_funcs.add(id(node.func))
            resolved = index.resolve_call(mod, node.func)
            if resolved and resolved[1] != func and resolved[1] not in seen:
                seen.add(resolved[1])
                out.append((resolved[0], "call", resolved[1]))
        elif (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and id(node) not in call_funcs
        ):
            names = mod.collections.get(node.id, [node.id])
            for name in names:
                resolved = index.resolve_name(mod, name)
                if (
                    resolved
                    and resolved[0] == "func"
                    and resolved[1] != func
                    and resolved[1] not in seen
                ):
                    seen.add(resolved[1])
                    out.append(("func", "callback", resolved[1]))
    return out


def build_callgraph(
    index: Index, root: Func, depth_limit: int = CALL_DEPTH
) -> CallGraph:
    graph = CallGraph(root=root)
    graph.nodes[root] = 0
    frontier = [root]
    while frontier:
        nxt = []
        for func in frontier:
            depth = graph.nodes[func]
            for i, (kind, how, target) in enumerate(callees(index, func)):
                if func == root:
                    graph.order.setdefault(target, i + 1)
                if kind == "class":
                    if target not in graph.classes:
                        graph.classes.append(target)
                    graph.edges.append((func, target, how))
                    continue
                local = index.modules[target[0]].is_script
                tdepth = depth if local and depth > 0 else depth + 1
                if func == root:
                    tdepth = 1
                if tdepth > depth_limit and not local:
                    continue
                graph.edges.append((func, target, how))
                if target not in graph.nodes:
                    graph.nodes[target] = tdepth
                    nxt.append(target)
        frontier = nxt
    return graph


def reachable(index: Index, root: Func, stop: set[Func] = frozenset()) -> list[Func]:
    """All functions reachable from root (no depth limit), not entering ``stop``."""
    seen = [root]
    todo = [root]
    while todo:
        func = todo.pop()
        for kind, _, target in callees(index, func):
            if kind == "func" and target not in seen and target not in stop:
                seen.append(target)
                todo.append(target)
    return seen


# --------------------------------------------------------------------------------------
# helpers for DOT
# --------------------------------------------------------------------------------------


def nid(func: Func | str) -> str:
    text = func if isinstance(func, str) else f"{func[0]}__{func[1]}"
    return re.sub(r"\W", "_", text)


def esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def wrap(text: str, width: int = WRAP) -> str:
    return "\n".join(textwrap.wrap(text, width)) if text else ""


def module_fill(index: Index, mod_key: str, test_color: str | None) -> str:
    mod = index.modules.get(mod_key)
    if mod is not None and mod.is_script:
        return FILL_SCRIPT
    if mod_key in INFRA_MODULES:
        return FILL_INFRA
    if mod_key in SHARED_MODULES:
        return FILL_SHARED
    return lighten(test_color) if test_color else FILL_FALLBACK


def lighten(color: str, amount: float = 0.82) -> str:
    rgb = [int(color[i : i + 2], 16) for i in (1, 3, 5)]
    rgb = [round(c + (255 - c) * amount) for c in rgb]
    return "#" + "".join(f"{c:02x}" for c in rgb)


def module_label(index: Index, mod_key: str) -> str:
    mod = index.modules[mod_key]
    return mod.path.name if mod.is_script else f"modules/{mod.path.name}"


def header_lines(title: str, legend: str) -> list[str]:
    return [
        f'  label="{esc(title)}\\n{esc(legend)}";',
        "  labelloc=t;",
        "  fontsize=14;",
        '  fontname="Helvetica";',
        '  node [fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9, color="#555555"];',
    ]


def render(dot_text: str, stem: str) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dot_path = OUT_DIR / f"{stem}.dot"
    svg_path = OUT_DIR / f"{stem}.svg"
    dot_path.write_text(dot_text, encoding="utf-8")
    subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
    return [dot_path, svg_path]


# --------------------------------------------------------------------------------------
# test discovery
# --------------------------------------------------------------------------------------


@dataclass
class Test:
    test_id: str  # 'IMG_OPT_06'
    sim: str | None  # module key of the SIM script
    analysis: str | None
    title: str = ""
    color: str | None = None


def find_tests(index: Index) -> dict[str, Test]:
    tests: dict[str, Test] = {}
    for key, mod in index.modules.items():
        match = re.match(r"^(IMG_(?:OPT|RAD)_\d+)_.*_(SIM|ANALYSIS)$", key)
        if not match or not mod.is_script:
            continue
        test = tests.setdefault(match.group(1), Test(match.group(1), None, None))
        if match.group(2) == "SIM":
            test.sim = key
        else:
            test.analysis = key
    for test in tests.values():
        # the test's cluster (title and colour) from its SIM main() stage
        for key in (test.sim, test.analysis):
            mod = index.modules.get(key) if key else None
            for stage_id in (mod.stages.values() if mod else ()):
                spec = index.stage_spec.get(stage_id)
                if spec and spec.cluster:
                    test.title, test.color = spec.cluster, spec.cluster_color
                    break
            if test.title:
                break
        test.title = test.title or test.test_id.replace("_", "-")
    return dict(sorted(tests.items()))


# --------------------------------------------------------------------------------------
# A: call graphs
# --------------------------------------------------------------------------------------


def callgraph_dot(index: Index, test: Test, script: str, which: str) -> str:
    root = (script, "main")
    graph = build_callgraph(index, root)
    lines = [
        "digraph callgraph {",
        "  rankdir=LR;",
        "  ranksep=0.6;",
        "  nodesep=0.15;",
    ]
    lines += header_lines(
        f"{test.title}: {which} call graph ({index.modules[script].path.name})",
        "box = @pipeline_stage, oval = helper, dashed box = class instantiated, "
        "dashed arrow = passed as callback; numbers = call order in main(); "
        "hover for the full docstring",
    )
    by_module: dict[str, list[Func]] = {}
    for func in list(graph.nodes) + graph.classes:
        by_module.setdefault(func[0], []).append(func)
    for mod_key, funcs in by_module.items():
        fill = module_fill(index, mod_key, test.color)
        lines.append(f"  subgraph cluster_{nid(mod_key)} {{")
        lines.append(f'    label="{esc(module_label(index, mod_key))}";')
        lines.append(f'    style=filled; fillcolor="{fill}"; color="#999999";')
        for func in funcs:
            lines.append("    " + callgraph_node(index, graph, func))
        lines.append("  }")
    for src, dst, how in graph.edges:
        attrs = []
        if src == root and dst in graph.order:
            attrs.append(f'label="{graph.order[dst]}"')
        if how == "callback":
            attrs.append("style=dashed")
        lines.append(f"  {nid(src)} -> {nid(dst)} [{', '.join(attrs)}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def callgraph_node(index: Index, graph: CallGraph, func: Func) -> str:
    fn = (
        index.func_node(func)
        if func not in graph.classes
        else index.modules[func[0]].classes.get(func[1])
    )
    tooltip = esc(docstring(fn) or func[1])
    name = func[1]
    text = purpose(fn)
    if func in graph.classes:
        label = f"{name}(...)"
        style = 'shape=box, style="rounded,dashed,filled", fillcolor="white"'
    elif index.stage_of(func) is not None:
        label = f"{name}\n{wrap(text)}" if text else name
        bold = ",bold" if func == graph.root else ""
        style = f'shape=box, style="filled{bold}", fillcolor="white"'
    elif func == graph.root:
        label = f"{name}()"
        style = 'shape=box, style="filled,bold", fillcolor="white"'
    else:
        label = f"{name}\n{wrap(text)}" if text else name
        style = "shape=ellipse"
    return f'{nid(func)} [label="{esc(label)}", tooltip="{tooltip}", {style}];'


# --------------------------------------------------------------------------------------
# B: data flow of result-dataclass fields
# --------------------------------------------------------------------------------------


@dataclass
class FieldFlow:
    classes: list[str]  # result classes, constructed in the ANALYSIS
    field_owner: dict[str, str]  # field -> owning class
    actors: list[Func]  # main() and its callees, in call order
    ctor_actors: dict[str, Func]  # pseudo actor 'Class(...)' -> the function calling it
    writes: dict[str, list[str]]  # actor id -> fields
    reads: dict[str, list[str]]


def actor_key(func: Func) -> str:
    return f"{func[0]}:{func[1]}"


def field_flow(index: Index, analysis: str) -> FieldFlow:
    root = (analysis, "main")
    top = [root] + [
        target for kind, _, target in callees(index, root) if kind == "func"
    ]
    # each top-level actor also owns the accesses of the helpers it reaches
    # (unless another top-level actor is in between)
    stop = set(top)
    owned = {func: reachable(index, func, stop - {func}) for func in top}

    # result classes: classes instantiated anywhere in the ANALYSIS, and their bases
    classes: list[str] = []
    for func in reachable(index, root):
        for kind, _, target in callees(index, func):
            if kind == "class" and target[1] in index.classes:
                for owner, _ in index.fields_of(target[1]):
                    if owner not in classes:
                        classes.append(owner)
                if target[1] not in classes:
                    classes.append(target[1])
    field_owner: dict[str, str] = {}
    for cls in classes:
        for owner, fld in index.fields_of(cls):
            field_owner.setdefault(fld, owner)

    writes: dict[str, list[str]] = {}
    reads: dict[str, list[str]] = {}
    ctor_actors: dict[str, Func] = {}

    def add(store: dict, key: str, fld: str) -> None:
        if fld in field_owner and fld not in store.setdefault(key, []):
            store[key].append(fld)

    for actor in top:
        for func in owned[actor]:
            fn = index.func_node(func)
            if fn is None:
                continue
            mod = index.modules[func[0]]
            for node in _body_nodes(fn):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id not in mod.aliases
                ):
                    target = writes if isinstance(node.ctx, ast.Store) else reads
                    add(target, actor_key(actor), node.attr)
                elif isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Name)
                        and node.func.id == "setattr"
                        and len(node.args) >= 2
                        and isinstance(node.args[1], ast.Constant)
                    ):
                        add(writes, actor_key(actor), node.args[1].value)
                        continue
                    resolved = index.resolve_call(mod, node.func)
                    if (
                        resolved
                        and resolved[0] == "class"
                        and resolved[1][1] in classes
                    ):
                        cls = resolved[1][1]
                        key = f"ctor:{cls}:{actor_key(actor)}"
                        ctor_actors[key] = actor
                        own = [f for _, f in index.fields_of(cls)]
                        for i, _ in enumerate(node.args):
                            if i < len(own):
                                add(writes, key, own[i])
                        for kw in node.keywords:
                            if kw.arg:
                                add(writes, key, kw.arg)
    # drop a field read in the same actor that wrote it before (keep both edges,
    # but a pure self-loop field adds nothing) -- keep as is; only drop empty actors
    return FieldFlow(classes, field_owner, top, ctor_actors, writes, reads)


def dataflow_dot(index: Index, test: Test, flow: FieldFlow) -> str:
    lines = ["digraph dataflow {", "  rankdir=LR;", "  ranksep=0.9;", "  nodesep=0.12;"]
    lines += header_lines(
        f"{test.title}: data flow of the result fields (ANALYSIS)",
        "solid blue = stage writes the field, dashed grey = stage reads it; "
        "italic = property (derived)",
    )
    used = {
        f
        for flds in list(flow.writes.values()) + list(flow.reads.values())
        for f in flds
    }
    # actors
    lines.append(
        '  node [shape=box, style="rounded,filled", fillcolor="white", penwidth=1.5];'
    )
    actor_ids = []
    for key, actor in flow.ctor_actors.items():
        cls = key.split(":")[1]
        lines.append(f'  {nid(key)} [label="{esc(cls)}(...)\\n(in {esc(actor[1])})"];')
        actor_ids.append(key)
    for actor in flow.actors:
        key = actor_key(actor)
        if key in flow.writes or key in flow.reads:
            fn = index.func_node(actor)
            lines.append(
                f'  {nid(key)} [label="{esc(actor[1])}", '
                f'tooltip="{esc(docstring(fn) or actor[1])}"];'
            )
            actor_ids.append(key)
    # fields, one cluster per class
    for cls in flow.classes:
        info = index.classes[cls]
        flds = [
            f
            for f in info.fields + info.properties
            if f in used and flow.field_owner.get(f) == cls
        ]
        if not flds:
            continue
        lines.append(f"  subgraph cluster_{nid(cls)} {{")
        lines.append(f'    label="{esc(cls)} ({esc(info.module)})";')
        fill = module_fill(index, info.module, test.color)
        lines.append(f'    style=filled; fillcolor="{fill}"; color="#999999";')
        for fld in flds:
            italic = ', fontname="Helvetica-Oblique"' if fld in info.properties else ""
            lines.append(
                f'    {nid("f_" + fld)} [label="{esc(fld)}", shape=note, '
                f'style=filled, fillcolor="white", penwidth=1{italic}];'
            )
        lines.append("  }")
    for key in actor_ids:
        for fld in flow.writes.get(key, []):
            lines.append(f'  {nid(key)} -> {nid("f_" + fld)} [color="#1f77b4"];')
        for fld in flow.reads.get(key, []):
            if fld in flow.writes.get(key, []):
                continue  # written and read by the same stage: the write says it
            lines.append(
                f'  {nid("f_" + fld)} -> {nid(key)} [color="#888888", style=dashed];'
            )
    lines.append("}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# C: SIM -> ANALYSIS I/O map
# --------------------------------------------------------------------------------------


def const_text(node: ast.expr) -> str | None:
    """A string constant, or an f-string with its fields shown as <...>."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts = []
        for val in node.values:
            if isinstance(val, ast.Constant):
                parts.append(str(val.value))
            else:
                inner = val.value
                parts.append(f"<{getattr(inner, 'id', getattr(inner, 'attr', 'x'))}>")
        return "".join(parts)
    return None


def _is_header_expr(node: ast.expr) -> bool:
    name = getattr(node, "id", None) or getattr(node, "attr", None) or ""
    return "header" in name.lower() or name.lower() in ("hdr", "hdr0", "phdr")


def _is_hdul_expr(node: ast.expr) -> bool:
    name = getattr(node, "id", None) or getattr(node, "attr", None) or ""
    return "hdul" in name.lower() or name.lower() in ("hdus", "hdu_list")


@dataclass
class FitsIO:
    extensions: list[str] = field(default_factory=list)
    cards: list[str] = field(default_factory=list)


def fits_writes(index: Index, funcs: list[Func]) -> FitsIO:
    io = FitsIO()
    for func in funcs:
        fn = index.func_node(func)
        for node in _body_nodes(fn) if fn else ():
            if isinstance(node, ast.Call):
                fname = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if fname.endswith("HDU"):
                    for kw in node.keywords:
                        text = const_text(kw.value) if kw.arg == "name" else None
                        if text and text not in io.extensions:
                            io.extensions.append(text)
            elif isinstance(node, ast.Tuple) and len(node.elts) == 3:
                text = const_text(node.elts[0])
                if text and UPPER_KEY.match(text) and text not in io.cards:
                    io.cards.append(text)
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Store)
                and _is_header_expr(node.value)
            ):
                text = const_text(node.slice)
                if text and UPPER_KEY.match(text) and text not in io.cards:
                    io.cards.append(text)
    return io


def fits_reads(index: Index, funcs: list[Func]) -> FitsIO:
    io = FitsIO()

    def add(lst: list[str], text: str | None) -> None:
        if text and UPPER_KEY.match(text) and text != "EXTNAME" and text not in lst:
            lst.append(text)

    for func in funcs:
        fn = index.func_node(func)
        for node in _body_nodes(fn) if fn else ():
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if kw.arg in ("hdu_index", "ext", "extname"):
                        add(io.extensions, const_text(kw.value))
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and _is_header_expr(node.func.value)
                    and node.args
                ):
                    add(io.cards, const_text(node.args[0]))
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                if _is_header_expr(node.value):
                    add(io.cards, const_text(node.slice))
                elif _is_hdul_expr(node.value):
                    add(io.extensions, const_text(node.slice))
    return io


def string_constants(index: Index, mod_key: str) -> list[str]:
    mod = index.modules.get(mod_key)
    if mod is None:
        return []
    out = []
    for node in ast.walk(mod.tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
    return out


def yaml_files(index: Index, mod_key: str) -> list[str]:
    out = []
    for text in string_constants(index, mod_key):
        if text.endswith(".yaml") and "\n" not in text:
            name = text.rsplit("/", 1)[-1]
            if name not in out:
                out.append(name)
    return out


def output_dirs(index: Index, mod_keys: list[str]) -> list[str]:
    out = []
    for key in mod_keys:
        for text in string_constants(index, key):
            if "\n" in text or " " in text:
                continue
            match = OUT_PATH.search(text)
            found = match.group(1).rstrip("/") if match else None
            if found and found not in out:
                out.append(found)
    return out


def test_backbones(index: Index, test: Test) -> list[str]:
    keys = []
    for script in (test.sim, test.analysis):
        mod = index.modules.get(script) if script else None
        if mod is None:
            continue
        for target in list(mod.aliases.values()) + [
            t for t, _ in mod.imported.values()
        ]:
            if target.startswith("backbone") and target not in keys:
                keys.append(target)
    return keys


def io_dot(index: Index, test: Test, flow: FieldFlow | None) -> str:
    lines = ["digraph io {", "  rankdir=TB;", "  ranksep=0.55;", "  nodesep=0.2;"]
    lines += header_lines(
        f"{test.title}: SIM → ANALYSIS I/O map",
        "gold = config read, black = FITS written / read (edge labels: header cards), "
        "green = outputs",
    )
    sims = [test.sim] if test.sim else []
    ans = [test.analysis] if test.analysis else []
    # config YAMLs
    yamls: dict[str, list[str]] = {}
    for script in sims + ans:
        names = yaml_files(index, script)
        # YAMLs a reached module names in a module-level constant (e.g. the common
        # observing config, loaded inside config_common.load_observing_config)
        for mod_key in dict.fromkeys(f[0] for f in reachable(index, (script, "main"))):
            for stmt in index.modules[mod_key].tree.body:
                if (
                    mod_key != script
                    and isinstance(stmt, ast.Assign)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                    and stmt.value.value.endswith(".yaml")
                ):
                    names.append(stmt.value.value.rsplit("/", 1)[-1])
        for name in dict.fromkeys(names):
            yamls.setdefault(name, []).append(script)
    lines.append("  subgraph cluster_cfg {")
    lines.append(
        '    label="config/"; style=filled; fillcolor="#fff8dc"; color="#bbbbbb";'
    )
    for name in yamls:
        lines.append(
            f'    {nid("y_" + name)} [label="{esc(name)}", shape=note, style=filled, '
            'fillcolor="white"];'
        )
    lines.append("  }")

    # SIM side: main + the SIM-file functions that (transitively) write FITS
    written = FitsIO()
    sim_writers: list[Func] = []
    for script in sims:
        root = (script, "main")
        fill = module_fill(index, script, test.color)
        lines.append(f"  subgraph cluster_{nid(script)} {{")
        lines.append(f'    label="{esc(index.modules[script].path.name)}";')
        lines.append(
            f'    style=filled; fillcolor="{lighten(test.color or "#999999")}"; '
            'color="#999999";'
        )
        lines.append(
            f'    {nid(root)} [label="main()", shape=box, style="rounded,filled,bold", '
            f'fillcolor="white"];'
        )
        for _, _, target in callees(index, root):
            if target[0] != script:
                continue
            io = fits_writes(index, reachable(index, target))
            if io.extensions or io.cards:
                sim_writers.append(target)
                text = purpose(index.func_node(target))
                lines.append(
                    f'    {nid(target)} [label="{esc(target[1])}\\n{esc(wrap(text))}", '
                    f'shape=box, style="rounded,filled", fillcolor="white"];'
                )
                lines.append(f"    {nid(root)} -> {nid(target)} [style=dashed];")
        lines.append("  }")
        _ = fill
        io = fits_writes(index, reachable(index, root))
        written.extensions += [e for e in io.extensions if e not in written.extensions]
        written.cards += [c for c in io.cards if c not in written.cards]
    data_dirs = [d for d in output_dirs(index, sims) if d.startswith("data/")]

    # the FITS file
    if written.extensions or written.cards:
        cards = "\\n".join(textwrap.wrap(" ".join(written.cards), 60))
        exts = "|".join(f"<{nid('x_' + e)}> {esc(e)}" for e in written.extensions)
        title = esc(data_dirs[0] if data_dirs else "simulated data") + "\\n*.fits"
        lines.append(
            f'  fits [shape=record, style=filled, fillcolor="#e8f0fb", '
            f'label="{{{title}|<primary> PRIMARY header:\\n{cards}|{{{exts}}}}}"];'
        )
        for writer in sim_writers:
            lines.append(f"  {nid(writer)} -> fits;")

    # ANALYSIS side
    reads_by_actor: dict[Func, FitsIO] = {}
    for script in ans:
        root = (script, "main")
        top = [root] + [t for k, _, t in callees(index, root) if k == "func"]
        stop = set(top)
        lines.append(f"  subgraph cluster_{nid(script)} {{")
        lines.append(f'    label="{esc(index.modules[script].path.name)} + modules";')
        lines.append(f'    style=filled; fillcolor="{FILL_SCRIPT}"; color="#999999";')
        lines.append(
            f'    {nid(root)} [label="main()", shape=box, style="rounded,filled,bold", '
            f'fillcolor="white"];'
        )
        for actor in top:
            io = fits_reads(index, reachable(index, actor, stop - {actor}))
            if not (io.extensions or io.cards):
                continue
            reads_by_actor[actor] = io
            if actor != root:
                lines.append(
                    f'    {nid(actor)} [label="{esc(actor[1])}", shape=box, '
                    f'style="rounded,filled", fillcolor="white"];'
                )
        # result fields set from header cards in main (constructor keywords)
        lines.append("  }")
        for actor, io in reads_by_actor.items():
            for ext in io.extensions:
                port = f":{nid('x_' + ext)}" if ext in written.extensions else ""
                lines.append(f"  fits{port} -> {nid(actor)};")
            if io.cards:
                label = esc(wrap(", ".join(io.cards), 40))
                lines.append(f'  fits:primary -> {nid(actor)} [label="{label}"];')

    # config edges
    for name, scripts in yamls.items():
        for script in scripts:
            lines.append(
                f'  {nid("y_" + name)} -> {nid((script, "main"))} [color="#b8860b"];'
            )

    # outputs
    outs = [
        d
        for d in output_dirs(index, ans + test_backbones(index, test))
        if not d.startswith("data/")
    ]
    if outs and ans:
        lines.append("  subgraph cluster_out {")
        lines.append(
            '    label="outputs"; style=filled; fillcolor="#e6f4e6"; color="#bbbbbb";'
        )
        for out in outs:
            lines.append(
                f'    {nid("o_" + out)} [label="{esc(out)}/", shape=folder, '
                f'style=filled, fillcolor="white"];'
            )
        lines.append("  }")
        for out in outs:
            lines.append(
                f'  {nid((ans[0], "main"))} -> {nid("o_" + out)} [color="#2ca02c"];'
            )
    _ = flow
    lines.append("}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# D: overview (stage graph + table) and its Markdown twin
# --------------------------------------------------------------------------------------


@dataclass
class StageRow:
    stage: str
    func: Func
    module: str
    purpose: str
    writes: list[str]
    line: int


def stage_rows(index: Index, test: Test, flow: FieldFlow | None) -> list[StageRow]:
    funcs: list[Func] = []
    for script in (test.sim, test.analysis):
        if script:
            for func in reachable(index, (script, "main")):
                if index.stage_of(func) and func not in funcs:
                    funcs.append(func)
    # plus any stage in the test's own cluster (e.g. FYI boxes are not functions)
    rows = []
    for func in funcs:
        fn = index.func_node(func)
        writes: list[str] = []
        if flow is not None:
            writes = flow.writes.get(actor_key(func), [])
        if not writes and func[0] == test.sim:
            # the cards / extensions this SIM adds itself (not the common ones that
            # sim_common writes for every test)
            own = [f for f in reachable(index, func) if f[0] == test.sim]
            io = fits_writes(index, own)
            writes = io.cards + io.extensions or ["FITS case files (via sim_common)"]
        rows.append(
            StageRow(
                stage=index.stage_of(func),
                func=func,
                module=module_label(index, func[0]),
                purpose=purpose(fn, 140),
                writes=writes,
                line=fn.lineno if fn else 1,
            )
        )
    return rows


def overview_dot(index: Index, test: Test, rows: list[StageRow]) -> str:
    stage_ids = {row.stage for row in rows}
    cluster_stages = {
        sid for sid, spec in index.stage_spec.items() if spec.cluster == test.title
    }
    stage_ids |= cluster_stages
    by_stage = {row.stage: row for row in rows}
    lines = [
        "digraph overview {",
        "  rankdir=TB;",
        "  ranksep=0.35;",
        "  newrank=true;",
    ]
    lines += header_lines(
        f"{test.title}: overview",
        "pipeline stages (this test's + the shared ones it uses), then each stage's "
        "purpose and the result fields it writes",
    )
    lines.append("  subgraph cluster_flow {")
    lines.append('    label="pipeline stages"; style=rounded; color="#999999";')
    lines.append('    node [shape=box, style="rounded,filled"];')
    for sid in sorted(stage_ids):
        spec = index.stage_spec.get(sid)
        row = by_stage.get(sid)
        fill = (
            module_fill(index, row.func[0], test.color)
            if row
            else lighten(test.color or "#999999")
        )
        style = '"rounded,filled"'
        if spec is not None and spec.planned:
            style = '"rounded,dashed,filled"'
        elif spec is not None and spec.fyi:
            style = '"rounded,dashed,filled"'
        bold = ", penwidth=2" if row and row.func[1] == "log_test_outcome" else ""
        label = spec.label if spec else sid
        lines.append(
            f'    {nid(sid)} [label="{esc(label)}", fillcolor="{fill}", '
            f"style={style}{bold}];"
        )
    last = []
    for sid in sorted(stage_ids):
        spec = index.stage_spec.get(sid)
        for parent in spec.depends_on if spec else ():
            if parent in stage_ids:
                lines.append(f"    {nid(parent)} -> {nid(sid)};")
        if spec and not any(
            sid in (index.stage_spec[o].depends_on)
            for o in stage_ids
            if o in index.stage_spec
        ):
            last.append(sid)
    lines.append("  }")

    cells = [
        '<TR><TD BGCOLOR="#dddddd"><B>stage</B></TD>'
        '<TD BGCOLOR="#dddddd"><B>module</B></TD>'
        '<TD BGCOLOR="#dddddd"><B>purpose (docstring)</B></TD>'
        '<TD BGCOLOR="#dddddd"><B>writes</B></TD></TR>'
    ]
    for row in rows:
        writes = ", ".join(row.writes)
        cells.append(
            "<TR>"
            f'<TD ALIGN="LEFT">{html.escape(row.stage)}</TD>'
            f'<TD ALIGN="LEFT">{html.escape(row.module)}</TD>'
            f'<TD ALIGN="LEFT">{html_wrap(row.purpose, 60)}</TD>'
            f'<TD ALIGN="LEFT">{html_wrap(writes, 40)}</TD>'
            "</TR>"
        )
    table = (
        '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">'
        + "".join(cells)
        + "</TABLE>"
    )
    lines.append(f"  stage_table [shape=plaintext, label=<{table}>];")
    for sid in last:
        lines.append(f"  {nid(sid)} -> stage_table [style=invis];")
    lines.append("}")
    return "\n".join(lines) + "\n"


def html_wrap(text: str, width: int) -> str:
    return '<BR ALIGN="LEFT"/>'.join(
        html.escape(t) for t in textwrap.wrap(text, width)
    ) + ('<BR ALIGN="LEFT"/>' if text else "")


def overview_md(index: Index, test: Test, rows: list[StageRow]) -> str:
    out = [
        f"# {test.title}",
        "",
        "Generated by `docs/uml/render_test_diagrams.py`; do not edit by hand.",
        "",
        f"- Overview: [{test.test_id}_overview.svg]({test.test_id}_overview.svg)",
    ]
    for kind in ("callgraph_sim", "callgraph_analysis", "dataflow", "io"):
        if (OUT_DIR / f"{test.test_id}_{kind}.svg").exists():
            out.append(
                f"- {kind.replace('_', ' ')}: "
                f"[{test.test_id}_{kind}.svg]({test.test_id}_{kind}.svg)"
            )
    out += [
        "",
        "| stage | function | purpose (docstring) | writes |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        rel = Path("../../..") / index.modules[row.func[0]].path.relative_to(REPO)
        link = f"[{row.module}:{row.line}]({rel.as_posix()}#L{row.line})"
        writes = ", ".join(f"`{w}`" for w in row.writes)
        purpose_text = row.purpose.replace("|", "\\|")
        out.append(f"| `{row.stage}` | {link} | {purpose_text} | {writes} |")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def render_test(index: Index, test: Test) -> list[Path]:
    written: list[Path] = []
    stem = test.test_id
    if test.sim and "main" in index.modules[test.sim].functions:
        written += render(
            callgraph_dot(index, test, test.sim, "SIM"), f"{stem}_callgraph_sim"
        )
    flow = None
    if test.analysis and "main" in index.modules[test.analysis].functions:
        written += render(
            callgraph_dot(index, test, test.analysis, "ANALYSIS"),
            f"{stem}_callgraph_analysis",
        )
        flow = field_flow(index, test.analysis)
        if flow.writes or flow.reads:
            written += render(dataflow_dot(index, test, flow), f"{stem}_dataflow")
    written += render(io_dot(index, test, flow), f"{stem}_io")
    rows = stage_rows(index, test, flow)
    written += render(overview_dot(index, test, rows), f"{stem}_overview")
    md_path = OUT_DIR / f"{stem}_overview.md"
    md_path.write_text(overview_md(index, test, rows), encoding="utf-8")
    written.append(md_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tests", nargs="*", help="test ids, e.g. IMG_OPT_06 IMG_RAD_07")
    parser.add_argument(
        "--all", action="store_true", help="every test with a SIM/ANALYSIS"
    )
    args = parser.parse_args(argv)

    index = Index()
    tests = find_tests(index)
    wanted = (
        list(tests) if args.all else [t.upper().replace("-", "_") for t in args.tests]
    )
    if not wanted:
        parser.error(f"give test ids or --all; known: {', '.join(tests)}")
    unknown = [t for t in wanted if t not in tests]
    if unknown:
        parser.error(f"unknown test(s) {unknown}; known: {', '.join(tests)}")
    try:
        for test_id in wanted:
            paths = render_test(index, tests[test_id])
            print(f"{test_id}: wrote {len(paths)} files to {OUT_DIR}")
    except FileNotFoundError:
        print("graphviz `dot` not found; put it on PATH", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"`dot` failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
