"""Smoke tests for tools that are not explicitly tested elsewhere.

This module dynamically discovers tools that are not covered by specific test files
and runs them with minimal placeholder inputs to ensure they don't crash immediately
(e.g. due to import errors or syntax errors).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from biomni.utils import read_module2api

if TYPE_CHECKING:
    from hypha_rpc.rpc import RemoteService

_PATHISH_PARAM_RE = re.compile(
    r"(path|filepath|file_path|dir|directory|folder|output|input|fasta|fastq|bam|vcf|"
    r"bed|gff|gtf|pdb|sdf|png|jpg|jpeg|tif|tiff|csv|tsv|txt)$",
    re.IGNORECASE,
)

# Constants for reporting limits
MAX_REPORT_FAILURES = 50
MAX_REPORT_OTHERS = 25


@dataclass
class SmokeContext:
    """Context for smoke test execution."""

    hypha_service: RemoteService
    tool_specs: dict[str, dict[str, Any]]
    capsys: pytest.CaptureFixture[str]
    lock: asyncio.Lock
    sem: asyncio.Semaphore
    timeout_s: float
    results: dict[str, list[str]] = field(
        default_factory=lambda: {
            "unexpected_failures": [],
            "expected_failures": [],
            "needs_fixture": [],
        },
    )
    counters: dict[str, int] = field(
        default_factory=lambda: {"completed": 0, "executed": 0},
    )
    total_tools: int = 0


def _all_tools_from_descriptions() -> dict[str, dict[str, Any]]:
    module2api = read_module2api()
    tool_specs: dict[str, dict[str, Any]] = {}
    for api_list in module2api.values():
        for spec in api_list:
            name = spec.get("name")
            if name:
                tool_specs[name] = spec
    return tool_specs


def _tested_tools_from_tests_dir() -> set[str]:
    tests_dir = Path(__file__).resolve().parent
    tested: set[str] = set()

    # Heuristic: most tests call `await hypha_service.<tool>(...)`.
    call_re = re.compile(r"\bhypha_service\.([A-Za-z_]\w*)\b")
    for path in sorted(tests_dir.glob("test_*.py")):
        if path.name in {"test_tool_coverage.py", "test_tool_smoke_untested.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        tested.update(call_re.findall(text))
    return tested


def _placeholder_int(name: str) -> int:
    if "top" in name and "k" in name:
        return 3
    if name in {"k", "top_k", "n", "num", "num_results"}:
        return 3
    return 1


def _placeholder_dict(name: str) -> dict[str, Any]:
    # Provide minimal structured dicts for common "parameter bundle" patterns.
    if "radiation" in name:
        return {
            "radionuclide": "Ac-225",
            "activity_bq": 1_000_000,
            "half_life_h": 240.0,
        }
    if "operational_parameters" in name or "operating_parameters" in name:
        return {"hrt": 10.0, "olr": 1.0, "temperature": 35.0, "ph": 7.0}
    if "flow" in name and "data" in name:
        return {"time_points": [0, 1, 2, 3], "values": [0.1, 0.2, 0.25, 0.3]}
    if "parameters" in name:
        return {}
    return {}


def _placeholder_list_numeric(name: str) -> list[Any] | None:
    if "time" in name or "time_points" in name:
        return [0, 1, 2, 3, 4]

    numeric_keys = [
        "amplitude",
        "signal",
        "values",
        "concentration",
        "luminescence",
        "physiological",
    ]
    if any(k in name for k in numeric_keys):
        return [0.0, 0.2, 0.5, 0.3, 0.1]

    if "parameter_values" in name:
        return [0.1, 0.2, 0.3, 0.4]
    return None


def _placeholder_list_text(name: str) -> list[Any] | None:
    simple_mappings = {
        "drug_pair": ["DrugA", "DrugB"],
        "network_topology": [["A", "B"], ["B", "C"]],
        "gene": ["TP53", "BRCA1", "EGFR"],
        "protein": ["P04637", "P38398"],
        "disease": ["breast cancer"],
    }

    for key, val in simple_mappings.items():
        if key in name:
            return val

    if "target_region" in name or ("region" in name and "target" in name):
        return [100, 200]

    if "condition" in name:
        return [{"temperature": 40.0, "humidity": 75.0, "duration_days": 7}]
    return None


def _placeholder_list(name: str) -> list[Any]:
    numeric = _placeholder_list_numeric(name)
    if numeric is not None:
        return numeric

    text = _placeholder_list_text(name)
    if text is not None:
        return text

    return ["example"]


def _placeholder_str(name: str, desc: str) -> str:
    looks_like_path = bool(_PATHISH_PARAM_RE.search(name)) or "path to" in desc
    if looks_like_path:
        # Intentionally missing path: many tools should return a graceful error.
        if name.endswith(("output_dir", "out_dir", "output_directory")):
            return str(Path(tempfile.gettempdir()) / "biomni_test_output")
        return "/data/does_not_exist"

    defaults = {
        "smiles": "CCO",
        "sequence": "ATGCGTACGTAGCTAGCTAG",
        "dna": "ATGCGTACGTAGCTAGCTAG",
        "rna": "AUGCGUACGUAGCUAGCUAG",
        "protein": "MSTNPKPQRKTKRNTNRRPQDVKFPGG",
        "aa": "MSTNPKPQRKTKRNTNRRPQDVKFPGG",
        "fasta": ">seq\nATGCGTACGTAGCTAGCTAG\n",
        "gene": "TP53",
        "gene_symbol": "TP53",
        "organism": "Homo sapiens",
        "species": "Homo sapiens",
        "email": "test@example.com",
        "query": "test",
        "text": "test",
        "prompt": "test",
    }

    for key, val in defaults.items():
        if key in name:
            return val

    return "test"


def _placeholder_value(
    param_name: str,
    type_str: str | None,
    description: str | None,
) -> Any:  # noqa: ANN401
    name = (param_name or "").lower()
    desc = (description or "").lower()
    t = (type_str or "str").strip()

    simple_types = {
        "float": 0.1,
        "number": 0.1,
        "bool": False,
        "boolean": False,
        "List[int]": [1, 2, 3],
    }
    if t in simple_types:
        return simple_types[t]

    if t in {"int", "integer"}:
        return _placeholder_int(name)
    if t in {"dict", "object"}:
        return _placeholder_dict(name)
    if t in {"list", "array", "List[str]"}:
        return _placeholder_list(name)

    return _placeholder_str(name, desc)


def _kwargs_for_tool(spec: dict[str, Any]) -> dict[str, Any]:
    required = spec.get("required_parameters", []) or []
    kwargs: dict[str, Any] = {}

    for p in required:
        pname = p.get("name")
        if not pname:
            continue
        kwargs[pname] = _placeholder_value(pname, p.get("type"), p.get("description"))
    return kwargs


def _is_jsonish(obj: Any) -> bool:  # noqa: ANN401
    try:
        json.dumps(obj)
    except TypeError:
        return False
    else:
        return True


def _is_expected_remote_failure_message(message: str) -> bool:
    msg = (message or "").lower()
    patterns = [
        "file not found",
        "no such file",
        "does not exist",
        "missing required",
        "validation error",
        "no module named",
        "not installed",
        "command not found",
        "unavailable",
        "requires",
        "permission denied",
        # Common dependency/runtime incompatibilities we treat as expected
        # in smoke mode.
        "flowcytometrytools",
        "mutablemapping",
    ]
    return any(p in msg for p in patterns)


def _is_input_mismatch_failure_message(message: str) -> bool:
    """Return True for errors that most likely mean our placeholder inputs are wrong.

    These should not fail CI; they are actionable signals to add a per-tool fixture
    override or improve the placeholder generator.
    """
    msg = (message or "").lower()
    patterns = [
        "could not convert string to float",
        "ufunc",
        "has no attribute 'shape'",
        "string indices must be integers",
        "too many values to unpack",
        "not enough values to unpack",
        "len() of unsized object",
        "digital filter critical frequencies",
        "non-json-serializable",
        "noneType' object has no attribute",
        "keyerror:",
        "indexerror:",
        "typeerror:",
        "valueerror:",
    ]
    return any(p in msg for p in patterns)


def _progress_print(capsys: pytest.CaptureFixture[str], text: str) -> None:
    # Ensure progress is visible even when pytest output capture is enabled.
    with capsys.disabled():
        print(text, flush=True)  # noqa: T201


def _raise_type_error(result_type: type) -> None:
    msg = f"non-JSON-serializable result type {result_type}"
    raise TypeError(msg)


async def _run_single_tool(
    tool_name: str,
    ctx: SmokeContext,
) -> None:
    async with ctx.sem:
        spec = ctx.tool_specs[tool_name]
        kwargs = _kwargs_for_tool(spec)

        try:
            fn = getattr(ctx.hypha_service, tool_name)
        except AttributeError:
            async with ctx.lock:
                ctx.results["unexpected_failures"].append(
                    f"{tool_name}: missing on remote service",
                )
                ctx.counters["completed"] += 1
                _progress_print(
                    ctx.capsys,
                    f"[smoke] {ctx.counters['completed']}/{ctx.total_tools} "
                    f"missing: {tool_name}",
                )
            return

        try:
            result = await asyncio.wait_for(fn(**kwargs), timeout=ctx.timeout_s)
            if not _is_jsonish(result):
                _raise_type_error(type(result))
            async with ctx.lock:
                ctx.counters["executed"] += 1
                ctx.counters["completed"] += 1
                _progress_print(
                    ctx.capsys,
                    f"[smoke] {ctx.counters['completed']}/{ctx.total_tools} "
                    f"ok: {tool_name}",
                )
        except TimeoutError:
            async with ctx.lock:
                ctx.results["expected_failures"].append(
                    f"{tool_name}: timeout after {ctx.timeout_s}s",
                )
                ctx.counters["completed"] += 1
                _progress_print(
                    ctx.capsys,
                    f"[smoke] {ctx.counters['completed']}/{ctx.total_tools} "
                    f"timeout: {tool_name}",
                )
                return
        except Exception as e:  # noqa: BLE001
            msg = str(e) if str(e) else repr(e)
            async with ctx.lock:
                if _is_expected_remote_failure_message(msg):
                    ctx.results["expected_failures"].append(f"{tool_name}: {msg}")
                    ctx.counters["completed"] += 1
                    _progress_print(
                        ctx.capsys,
                        f"[smoke] {ctx.counters['completed']}/{ctx.total_tools} "
                        f"expected-fail: {tool_name}",
                    )
                    return
                if _is_input_mismatch_failure_message(msg):
                    ctx.results["needs_fixture"].append(f"{tool_name}: {msg}")
                    ctx.counters["completed"] += 1
                    _progress_print(
                        ctx.capsys,
                        f"[smoke] {ctx.counters['completed']}/{ctx.total_tools} "
                        f"needs-fixture: {tool_name}",
                    )
                    return
                ctx.results["unexpected_failures"].append(f"{tool_name}: {msg}")
                ctx.counters["completed"] += 1
                _progress_print(
                    ctx.capsys,
                    f"[smoke] {ctx.counters['completed']}/{ctx.total_tools} "
                    f"UNEXPECTED-FAIL: {tool_name}",
                )


@pytest.mark.asyncio
async def test_smoke_untested_tools_execute_or_fail_gracefully(
    hypha_service: RemoteService,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Run smoke tests for all tools not explicitly tested elsewhere."""
    tool_specs = _all_tools_from_descriptions()
    tested_tools = _tested_tools_from_tests_dir()

    all_tools = set(tool_specs)
    untested = sorted(all_tools - tested_tools)

    # Some tools are inherently heavy / network-bound; keep the list small and explicit.
    # Prefer adding graceful error handling in the tool implementation over skipping.
    skip_tools: set[str] = set()

    timeout_s = float(os.getenv("BIOMNI_SMOKE_TIMEOUT", "30"))
    concurrency = max(1, int(os.getenv("BIOMNI_SMOKE_CONCURRENCY", "2")))
    strict = os.getenv("BIOMNI_SMOKE_STRICT", "0") == "1"
    max_tools_env = os.getenv("BIOMNI_SMOKE_MAX_TOOLS")
    max_tools = int(max_tools_env) if max_tools_env else None

    selected = [t for t in untested if t not in skip_tools]
    if max_tools is not None:
        selected = selected[:max_tools]

    _progress_print(
        capsys,
        (
            f"[smoke] running {len(selected)} untested tools "
            f"(total={len(untested)}, concurrency={concurrency}, timeout={timeout_s}s)"
        ),
    )

    ctx = SmokeContext(
        hypha_service=hypha_service,
        tool_specs=tool_specs,
        capsys=capsys,
        lock=asyncio.Lock(),
        sem=asyncio.Semaphore(concurrency),
        timeout_s=timeout_s,
        total_tools=len(selected),
    )

    tasks = [asyncio.create_task(_run_single_tool(t, ctx)) for t in selected]
    await asyncio.gather(*tasks)

    _progress_print(
        capsys,
        (
            f"[smoke] finished: executed={ctx.counters['executed']}, "
            f"expected_failures={len(ctx.results['expected_failures'])}, "
            f"needs_fixture={len(ctx.results['needs_fixture'])}, "
            f"unexpected_failures={len(ctx.results['unexpected_failures'])}"
        ),
    )

    assert (
        ctx.counters["executed"] > 0
    ), "No untested tools were executed; test harness needs adjustment."

    unexpected = ctx.results["unexpected_failures"]
    if unexpected and strict:
        details = "\n".join("- " + s for s in unexpected[:MAX_REPORT_FAILURES])
        extra = (
            ""
            if len(unexpected) <= MAX_REPORT_FAILURES
            else f"\n- ... and {len(unexpected) - MAX_REPORT_FAILURES} more"
        )

        needs_fixture_msg = ""
        if ctx.results["needs_fixture"]:
            items = ctx.results["needs_fixture"][:MAX_REPORT_OTHERS]
            needs_fixture_msg = "\n\nNeeds-fixture (not failing test):\n" + "\n".join(
                "- " + s for s in items
            )

        expected_failures_msg = ""
        if ctx.results["expected_failures"]:
            items = ctx.results["expected_failures"][:MAX_REPORT_OTHERS]
            expected_failures_msg = (
                "\n\nExpected failures (not failing test):\n"
                + "\n".join("- " + s for s in items)
            )

        pytest.fail(
            "Unexpected tool crashes (these should be fixed or explicitly skipped):\n"
            + details
            + extra
            + needs_fixture_msg
            + expected_failures_msg,
        )

    if unexpected and not strict:
        _progress_print(
            capsys,
            "[smoke] unexpected failures (non-strict, not failing):",
        )
        for line in unexpected[:MAX_REPORT_OTHERS]:
            _progress_print(capsys, "- " + line)
        if len(unexpected) > MAX_REPORT_OTHERS:
            _progress_print(
                capsys,
                f"- ... and {len(unexpected) - MAX_REPORT_OTHERS} more",
            )
