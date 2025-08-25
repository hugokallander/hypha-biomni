"""Validate importability of Biomni tool implementation modules.

This script attempts to import each module under biomni.tool.* that defines
APIs via the description registry. It reports missing third-party dependency
errors without raising, summarizing results at the end.

Run:
    python scripts/validate_imports.py

Exit code is 0 even if some failures occur; inspect summary.
"""

from __future__ import annotations

import importlib
import traceback

from biomni.utils import read_module2api


def collect_tool_modules() -> list[str]:
    modules = set()
    module2api = read_module2api()
    for module_path in module2api.keys():
        modules.add(module_path)
    return sorted(modules)


def main() -> None:
    failures: dict[str, str] = {}
    successes: list[str] = []
    for mod in collect_tool_modules():
        try:
            importlib.import_module(mod)
            successes.append(mod)
        except Exception as exc:  # broad by design to surface any failure
            failures[mod] = f"{exc}\n{traceback.format_exc().splitlines()[-1]}"
    print("Import validation summary")
    print("-------------------------")
    print(f"Successful: {len(successes)}")
    for s in successes[:25]:
        print(f"  + {s}")
    if len(successes) > 25:
        print("  ... (truncated)")
    print(f"Failures: {len(failures)}")
    for mod, err in failures.items():
        print(f"  - {mod}: {err}")
    if failures:
        print("\nSome modules failed to import. Consider installing missing packages.")


if __name__ == "__main__":  # pragma: no cover
    main()
