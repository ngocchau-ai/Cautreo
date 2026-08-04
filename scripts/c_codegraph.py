#!/usr/bin/env python3
"""C codegraph indexer for CAUTREO (C11 project).

Scans .c/.h files, extracts functions, structs, typedefs, enums, and #include
dependencies. Writes codegraph.json + markdown reports (summary, dependencies, impact),
mirroring the NPS Core codegraph schema (schema_version 0.1).

Usage:
    python scripts/c_codegraph.py --root <CAUTREO> [--output <json>] [--outdir <dir>]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Sequence

DEFAULT_EXCLUDED = frozenset({".git", "build", "__pycache__", ".venv"})

_FUNC_RE = re.compile(
    r"^(?P<ret>[A-Za-z_][\w\s\*]*?)\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*\("
)
_STRUCT_RE = re.compile(r"^\s*(?:typedef\s+)?struct\s+(?P<name>[A-Za-z_]\w*)\s*[;{]")
_TYPEDEF_RE = re.compile(r"^\s*typedef\s+.*?\b(?P<name>[A-Za-z_]\w*)\s*;")
_ENUM_RE = re.compile(r"^\s*(?:typedef\s+)?enum\s+(?P<name>[A-Za-z_]\w*)\s*[;{]")
_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"](?P<inc>[^>"]+)[>"]')


def _git_head(root: Path) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
        if r.returncode == 0 and re.fullmatch(r"[0-9a-fA-F]{40,64}", r.stdout.strip()):
            return r.stdout.strip()
    except OSError:
        pass
    return None


def _index_file(root: Path, path: Path) -> tuple[list[dict], list[dict], list[dict]]:
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    nodes: list[dict] = []
    edges: list[dict] = []
    errors: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return [], [], [{"path": rel, "line": 0, "message": str(e)}]

    module = rel.replace("/", ".").rsplit(".", 1)[0]
    nodes.append({"id": f"file:{rel}", "kind": "file", "name": path.name,
                 "qualified_name": rel, "path": rel, "line": 1})

    for lineno, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith("//") or s.startswith("/*") or s.startswith("*"):
            continue
        m = _FUNC_RE.match(line)
        if m and m.group("name") not in ("if", "for", "while", "switch", "return"):
            nodes.append({"id": f"func:{rel}:{m.group('name')}", "kind": "function",
                       "name": m.group("name"), "qualified_name": f"{module}.{m.group('name')}",
                       "path": rel, "line": lineno})
            edges.append({"source": f"file:{rel}", "target": f"func:{rel}:{m.group('name')}",
                       "kind": "contains"})
            continue
        m = _STRUCT_RE.match(line)
        if m:
            nodes.append({"id": f"struct:{rel}:{m.group('name')}", "kind": "struct",
                       "name": m.group("name"), "qualified_name": f"{module}.{m.group('name')}",
                       "path": rel, "line": lineno})
            edges.append({"source": f"file:{rel}", "target": f"struct:{rel}:{m.group('name')}",
                       "kind": "contains"})
            continue
        m = _TYPEDEF_RE.match(line)
        if m:
            nodes.append({"id": f"type:{rel}:{m.group('name')}", "kind": "typedef",
                       "name": m.group("name"), "qualified_name": f"{module}.{m.group('name')}",
                       "path": rel, "line": lineno})
            edges.append({"source": f"file:{rel}", "target": f"type:{rel}:{m.group('name')}",
                       "kind": "contains"})
            continue
        m = _INCLUDE_RE.match(line)
        if m:
            edges.append({"source": f"file:{rel}", "target": f"include:{m.group('inc')}",
                       "kind": "includes"})
    return nodes, edges, errors


def build(root: Path) -> dict:
    root = root.resolve()
    nodes, edges, errors = [], [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDED]
        for fn in sorted(filenames):
            if fn.endswith((".c", ".h")):
                n, e, er = _index_file(root, Path(dirpath) / fn)
                nodes.extend(n); edges.extend(e); errors.extend(er)
    nodes = sorted({n["id"]: n for n in nodes}.values(), key=lambda n: n["id"])
    edges = sorted({(e["source"], e["target"], e["kind"]): e for e in edges}.values(),
                 key=lambda e: (e["kind"], e["source"], e["target"]))
    errors = sorted(errors, key=lambda e: (e["path"], e["line"]))
    head = _git_head(root)
    return {"schema_version": "0.1",
            "status": "FRESH" if head else "UNVERSIONED",
            "repository_head": head, "codegraph_commit": head,
            "nodes": nodes, "edges": edges, "errors": errors}


def write_reports(root: Path, graph: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    head = graph["repository_head"] or "UNVERSIONED"
    (outdir / "codegraph.json").write_text(
        json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (outdir / "codegraph-summary.md").write_text(
        f"# CAUTREO Codegraph Summary\n\n"
        f"Status: {graph['status']}\nRepository HEAD: {head}\n"
        f"Nodes: {len(graph['nodes'])}\nEdges: {len(graph['edges'])}\n"
        f"Errors: {len(graph['errors'])}\n", encoding="utf-8")
    deps = [f"- {e['source']} -> {e['target']}" for e in graph["edges"] if e["kind"] == "includes"]
    (outdir / "module-dependencies.md").write_text(
        "# CAUTREO Include Dependencies\n\n" + ("\n".join(deps) if deps else "- None.\n") + "\n",
        encoding="utf-8")
    paths = sorted({n["path"] for n in graph["nodes"] if n["kind"] == "file"})
    (outdir / "change-impact.md").write_text(
        "# CAUTREO Change Impact\n\nGenerated baseline for indexed files.\n\n"
        "## Indexed File Paths\n" + "\n".join(f"- {p}" for p in paths) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--outdir", type=Path, default=Path("memory/03-codegraph"))
    args = p.parse_args(argv)
    graph = build(args.root)
    write_reports(args.root, graph, args.outdir)
    print(f"CAUTREO codegraph: {len(graph['nodes'])} nodes, "
          f"{len(graph['edges'])} edges, {len(graph['errors'])} errors -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())