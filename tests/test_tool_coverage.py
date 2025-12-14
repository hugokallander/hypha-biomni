"""Coverage tests for Hypha-exposed Biomni tools.

These are intentionally lightweight integration checks:
- Ensure every tool described in `biomni.tool.tool_description.*` is present on the remote service.
- Provide visibility into which tools are not yet exercised by higher-level tests.

The goal is to prevent silent regressions where new tools are added but never exposed via Hypha.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from biomni.utils import read_module2api

if TYPE_CHECKING:
    from hypha_rpc.rpc import RemoteService


def _all_tool_names() -> list[str]:
    module2api = read_module2api()
    names = [spec["name"] for api in module2api.values() for spec in api]
    return sorted(set(names))


def _tools_called_in_tests() -> set[str]:
    tests_dir = Path(__file__).resolve().parent
    used: set[str] = set()

    for path in tests_dir.glob("test_*.py"):
        if path.name == Path(__file__).name:
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "hypha_service"
                ):
                    used.add(node.func.attr)

    return used


@pytest.mark.asyncio
async def test_all_tools_are_exposed_on_remote_service(
    hypha_service: RemoteService,
) -> None:
    """Every described tool should be exposed as a Hypha RPC method.

    Single test (not parametrized) to avoid opening a new Hypha connection per tool.
    """
    missing: list[str] = []
    for tool_name in _all_tool_names():
        try:
            fn = getattr(hypha_service, tool_name)
        except AttributeError:
            missing.append(tool_name)
            continue
        if not callable(fn):
            missing.append(tool_name)

    assert not missing, f"Missing tools on remote service: {', '.join(missing)}"


def test_report_untested_tools() -> None:
    """Non-failing report of which tools lack direct call tests.

    This gives a stable place to see what's left without forcing
    potentially flaky network/data-heavy executions in CI.
    """
    all_tools = set(_all_tool_names())
    called = _tools_called_in_tests()
    untested = sorted(all_tools - called)

    # Use a no-op assertion so the message is visible in output when desired.
    assert True, (
        f"Untested tools ({len(untested)}): "
        f"{', '.join(untested[:50])}{' ...' if len(untested) > 50 else ''}"
    )
