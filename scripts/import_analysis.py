"""Analyze imports across biomni.tool modules to determine external dependencies.

Usage:
    python scripts/import_analysis.py --print-missing [--show-files]

The script:
1. Walks biomni/tool (excluding non-runtime data dirs) collecting .py files.
2. Parses AST to extract top-level import module names.
3. Normalizes to top-level package (e.g. pandas.core -> pandas).
4. Filters out (heuristically) standard library modules.
5. Prints unique third-party packages and (optionally) which files import them.

You can pipe the final list into pip install (after reviewing names).
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections import defaultdict
from pathlib import Path

# Heuristic stdlib identifiers (not exhaustive, but reduces noise).
STDLIB_BASE = set(sys.builtin_module_names) | {
    "abc",
    "argparse",
    "asyncio",
    "base64",
    "collections",
    "concurrent",
    "contextlib",
    "copy",
    "csv",
    "ctypes",
    "dataclasses",
    "datetime",
    "enum",
    "functools",
    "fractions",
    "glob",
    "gzip",
    "hashlib",
    "heapq",
    "hmac",
    "html",
    "http",
    "importlib",
    "inspect",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "logging",
    "math",
    "multiprocessing",
    "numbers",
    "operator",
    "os",
    "pathlib",
    "pickle",
    "platform",
    "plistlib",
    "pprint",
    "queue",
    "random",
    "re",
    "resource",
    "sched",
    "secrets",
    "shlex",
    "shutil",
    "signal",
    "site",
    "sqlite3",
    "ssl",
    "stat",
    "statistics",
    "string",
    "subprocess",
    "sys",
    "tempfile",
    "textwrap",
    "threading",
    "time",
    "timeit",
    "types",
    "typing",
    "typing_extensions",
    "unittest",
    "urllib",
    "uuid",
    "warnings",
    "weakref",
    "xml",
    "xmlrpc",
    "zipfile",
    "zoneinfo",
}

# Known / expected third-party top-level packages we don't want to misclassify.
KNOWN_THIRD_PARTY = {
    "numpy",
    "pandas",
    "scipy",
    "matplotlib",
    "seaborn",
    "networkx",
    "requests",
    "bs4",
    "tqdm",
    "PyPDF2",
    "googlesearch",
    "langchain",
    "langchain_openai",
    "langchain_anthropic",
    "openai",
    "transformers",
    "sentencepiece",
    "dotenv",
    "python_dotenv",
    "hypha_rpc",
    "anyio",
    "pydantic",
    "yaml",
    "pyyaml",
    "beautifulsoup4",
    "lxml",
    "sklearn",
    "statsmodels",
}

EXCLUDE_DIRS = {"tool_description", "example_mcp_tools", "schema_db", "__pycache__"}


def top_level(name: str) -> str:
    return name.split(".")[0].replace("-", "_")


def is_std(name: str) -> bool:
    if name in KNOWN_THIRD_PARTY:
        return False
    return name in STDLIB_BASE


def collect_py_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for item in root.iterdir():
        if item.is_dir():
            if item.name in EXCLUDE_DIRS:
                continue
            files.extend(collect_py_files(item))
        elif item.suffix == ".py" and item.name != "__init__.py":
            files.append(item)
    return files


def parse_imports(path: Path) -> set[str]:
    mods: set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception:
        return mods
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(top_level(alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(top_level(node.module))
    return mods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-missing",
        action="store_true",
        help="Show only non-stdlib modules",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="List which files import each module",
    )
    args = parser.parse_args()

    tool_root = Path(__file__).resolve().parent.parent / "biomni" / "tool"
    if not tool_root.exists():
        print(f"Tool directory not found: {tool_root}", file=sys.stderr)
        sys.exit(1)

    files = collect_py_files(tool_root)
    module_to_files: dict[str, set[Path]] = defaultdict(set)
    for f in files:
        for mod in parse_imports(f):
            module_to_files[mod].add(f.relative_to(tool_root))

    third_party = sorted(
        m for m in module_to_files if not is_std(m) and not m.startswith("biomni")
    )

    if args.print_missing:
        print("Third-party modules (top-level heuristic list):")
        for m in third_party:
            if args.show_files:
                print(f"- {m}: {', '.join(sorted(str(p) for p in module_to_files[m]))}")
            else:
                print(f"- {m}")
        print("\nReview & install (adjust names as needed):")
        print("pip install " + " ".join(third_party))
    else:
        print("All discovered modules (3rd-party flagged):")
        for m in sorted(module_to_files):
            tag = " (3rd)" if m in third_party else ""
            print(f"- {m}{tag}")


if __name__ == "__main__":  # pragma: no cover
    main()
