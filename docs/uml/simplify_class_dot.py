#!/usr/bin/env python3
"""Simplify pyreverse class DOT labels so graph-easy can render them as ASCII."""

from __future__ import annotations

import re
import sys
from pathlib import Path


def _html_br_to_newlines(text: str) -> str:
    text = re.sub(r"<br\s*ALIGN=\"LEFT\"\s*/>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    return text


def _flatten_record_label(raw: str) -> str:
    """Turn pyreverse HTML record label into a plain multiline box label."""
    # Strip outer <{ ... }>
    body = raw.strip()
    if body.startswith("<{") and body.endswith("}>"):
        body = body[2:-2]
    # Split record sections on bare |; keep \| (escaped union types) intact
    parts = re.split(r"(?<!\\)\|", body)
    cleaned = []
    for part in parts:
        part = part.replace("\\|", "|")
        part = _html_br_to_newlines(part).strip("\n")
        if part.strip():
            cleaned.append(part)
    flat = "\n\n".join(cleaned)
    # Escape for DOT double-quoted string
    flat = flat.replace("\\", "\\\\").replace('"', '\\"')
    flat = flat.replace("\n", "\\n")
    return flat


def simplify(dot_text: str) -> str:
    pattern = re.compile(
        r'^(?P<indent>\s*)"(?P<node>[^"]+)"\s*\[(?P<attrs>.*)\]\s*;\s*$',
        re.MULTILINE,
    )

    def repl(match: re.Match[str]) -> str:
        indent = match.group("indent")
        node = match.group("node")
        attrs = match.group("attrs")
        label_match = re.search(r"label=<(?P<label>\{.*?\})>", attrs)
        if not label_match:
            # Already a simple label, or unexpected format — keep as-is
            return match.group(0)

        short_name = node.rsplit(".", 1)[-1]
        flat = _flatten_record_label("<" + label_match.group("label") + ">")
        # Prefer short class name on the first line if the flattened text starts with it
        return f'{indent}"{node}" ' f'[label="{flat}", shape="box", style="solid"];'

    # Also shorten edge endpoints visually? Keep full IDs for uniqueness.
    out = pattern.sub(repl, dot_text)
    # Drop unused HTML/fontcolor attrs that confuse some parsers on edges — edges are fine.
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.dot output.dot", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.write_text(simplify(src.read_text()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
