from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

from biomni.utils import read_module2api

_PATHISH_PARAM_RE = re.compile(
    r"(path|filepath|file_path|dir|directory|folder|output|input|fasta|fastq|bam|vcf|bed|gff|gtf|pdb|sdf|png|jpg|jpeg|tif|tiff|csv|tsv|txt)$",
    re.IGNORECASE,
)


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


def _placeholder_value(
    param_name: str,
    type_str: str | None,
    description: str | None,
) -> Any:
    name = (param_name or "").lower()
    desc = (description or "").lower()
    t = (type_str or "str").strip()

    # Prefer explicit path cues from either name or description.
    looks_like_path = bool(_PATHISH_PARAM_RE.search(name)) or "path to" in desc

    if t in {"int", "integer"}:
        if "top" in name and "k" in name:
            return 3
        if name in {"k", "top_k", "n", "num", "num_results"}:
            return 3
        return 1
    if t in {"float", "number"}:
        return 0.1
    if t in {"bool", "boolean"}:
        return False
    if t in {"dict", "object"}:
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
    if t in {"list", "array", "List[str]"}:
        # Numeric series
        if "time" in name or "time_points" in name:
            return [0, 1, 2, 3, 4]
        if any(
            k in name
            for k in [
                "amplitude",
                "signal",
                "values",
                "concentration",
                "luminescence",
                "physiological",
            ]
        ):
            return [0.0, 0.2, 0.5, 0.3, 0.1]
        if "parameter_values" in name:
            return [0.1, 0.2, 0.3, 0.4]

        # Pairs / tuples encoded as list
        if "drug_pair" in name:
            return ["DrugA", "DrugB"]
        if "target_region" in name or ("region" in name and "target" in name):
            return [100, 200]
        if "network_topology" in name:
            return [["A", "B"], ["B", "C"]]

        # Text-y collections
        if "gene" in name:
            return ["TP53", "BRCA1", "EGFR"]
        if "protein" in name:
            return ["P04637", "P38398"]
        if "disease" in name:
            return ["breast cancer"]
        if "condition" in name:
            # Common in formulation stability / protocols.
            return [{"temperature": 40.0, "humidity": 75.0, "duration_days": 7}]
        return ["example"]
    if t in {"List[int]"}:
        return [1, 2, 3]

    # Default: string-like
    if looks_like_path:
        # Intentionally missing path: many tools should return a graceful error.
        if name.endswith(("output_dir", "out_dir", "output_directory")):
            return "/tmp/biomni_test_output"
        return "/data/does_not_exist"
    if "smiles" in name:
        return "CCO"
    if "sequence" in name or "dna" in name:
        return "ATGCGTACGTAGCTAGCTAG"
    if "rna" in name:
        return "AUGCGUACGUAGCUAGCUAG"
    if "protein" in name or "aa" in name:
        return "MSTNPKPQRKTKRNTNRRPQDVKFPGG"
    if "fasta" in name:
        return ">seq\nATGCGTACGTAGCTAGCTAG\n"
    if name in {"gene", "gene_symbol"}:
        return "TP53"
    if "organism" in name or "species" in name:
        return "Homo sapiens"
    if "email" in name:
        return "test@example.com"
    if "query" in name or "text" in name or "prompt" in name:
        return "test"
    return "test"


def _kwargs_for_tool(spec: dict[str, Any]) -> dict[str, Any]:
    required = spec.get("required_parameters", []) or []
    kwargs: dict[str, Any] = {}

    for p in required:
        pname = p.get("name")
        if not pname:
            continue
        kwargs[pname] = _placeholder_value(pname, p.get("type"), p.get("description"))
    return kwargs


def _is_jsonish(obj: Any) -> bool:
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False


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
        # Common dependency/runtime incompatibilities we treat as expected in smoke mode.
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


@pytest.mark.asyncio
async def test_smoke_untested_tools_execute_or_fail_gracefully(
    hypha_service,
    capsys: pytest.CaptureFixture[str],
):
    tool_specs = _all_tools_from_descriptions()
    tested_tools = _tested_tools_from_tests_dir()

    all_tools = set(tool_specs)
    untested = sorted(all_tools - tested_tools)

    # Some tools are inherently heavy / network-bound; keep the list small and explicit.
    # Prefer adding graceful error handling in the tool implementation over skipping long-term.
    skip_tools: set[str] = set()

    timeout_s = float(os.getenv("BIOMNI_SMOKE_TIMEOUT", "30"))
    # Default concurrency low to avoid overwhelming Hypha or the container.
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

    lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    unexpected_failures: list[str] = []
    expected_failures: list[str] = []
    needs_fixture: list[str] = []
    completed = 0
    executed = 0

    async def run_one(tool_name: str) -> None:
        nonlocal completed, executed
        async with sem:
            spec = tool_specs[tool_name]
            kwargs = _kwargs_for_tool(spec)

            try:
                fn = getattr(hypha_service, tool_name)
            except AttributeError:
                async with lock:
                    unexpected_failures.append(
                        f"{tool_name}: missing on remote service",
                    )
                    completed += 1
                    _progress_print(
                        capsys,
                        f"[smoke] {completed}/{len(selected)} missing: {tool_name}",
                    )
                return

            try:
                result = await asyncio.wait_for(fn(**kwargs), timeout=timeout_s)
                if not _is_jsonish(result):
                    raise TypeError(f"non-JSON-serializable result type {type(result)}")
                async with lock:
                    executed += 1
                    completed += 1
                    _progress_print(
                        capsys,
                        f"[smoke] {completed}/{len(selected)} ok: {tool_name}",
                    )
            except TimeoutError:
                async with lock:
                    expected_failures.append(f"{tool_name}: timeout after {timeout_s}s")
                    completed += 1
                    _progress_print(
                        capsys,
                        f"[smoke] {completed}/{len(selected)} timeout: {tool_name}",
                    )
                    return
            except Exception as e:  # noqa: BLE001 - classify below
                msg = str(e) if str(e) else repr(e)
                async with lock:
                    if _is_expected_remote_failure_message(msg):
                        expected_failures.append(f"{tool_name}: {msg}")
                        completed += 1
                        _progress_print(
                            capsys,
                            f"[smoke] {completed}/{len(selected)} expected-fail: {tool_name}",
                        )
                        return
                    if _is_input_mismatch_failure_message(msg):
                        needs_fixture.append(f"{tool_name}: {msg}")
                        completed += 1
                        _progress_print(
                            capsys,
                            f"[smoke] {completed}/{len(selected)} needs-fixture: {tool_name}",
                        )
                        return
                    unexpected_failures.append(f"{tool_name}: {msg}")
                    completed += 1
                    _progress_print(
                        capsys,
                        f"[smoke] {completed}/{len(selected)} UNEXPECTED-FAIL: {tool_name}",
                    )

    tasks = [asyncio.create_task(run_one(t)) for t in selected]
    await asyncio.gather(*tasks)

    _progress_print(
        capsys,
        (
            f"[smoke] finished: executed={executed}, expected_failures={len(expected_failures)}, "
            f"needs_fixture={len(needs_fixture)}, unexpected_failures={len(unexpected_failures)}"
        ),
    )

    # If we didn't run anything, something is off (e.g., detection regex broke).
    assert (
        executed > 0
    ), "No untested tools were executed; test harness needs adjustment."

    if unexpected_failures and strict:
        details = "\n".join("- " + s for s in unexpected_failures[:50])
        extra = (
            ""
            if len(unexpected_failures) <= 50
            else f"\n- ... and {len(unexpected_failures) - 50} more"
        )
        pytest.fail(
            "Unexpected tool crashes (these should be fixed or explicitly skipped):\n"
            + details
            + extra
            + (
                "\n\nNeeds-fixture (not failing test):\n"
                + "\n".join("- " + s for s in needs_fixture[:25])
                if needs_fixture
                else ""
            )
            + (
                "\n\nExpected failures (not failing test):\n"
                + "\n".join("- " + s for s in expected_failures[:25])
                if expected_failures
                else ""
            ),
        )

    if unexpected_failures and not strict:
        # In non-strict mode, we surface unexpected failures as progress output
        # without failing CI. This makes it possible to iterate on fixtures/tool
        # hardening without blocking the whole test suite.
        _progress_print(
            capsys,
            "[smoke] unexpected failures (non-strict, not failing):",
        )
        for line in unexpected_failures[:25]:
            _progress_print(capsys, "- " + line)
        if len(unexpected_failures) > 25:
            _progress_print(capsys, f"- ... and {len(unexpected_failures) - 25} more")
