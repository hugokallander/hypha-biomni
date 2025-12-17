"""Hypha integration utilities for Biomni tools.

This module dynamically converts all Biomni tool descriptions into hypha-rpc
`@schema_function`-decorated async callables and provides convenience helpers
to register them as a remote service.

Design goals:
1. Zero manual maintenance when new tools are added (pulls from tool descriptions).
2. Thin async wrappers over the underlying synchronous implementations.
3. Structured return envelope to normalize heterogeneous tool outputs.
4. Graceful error reporting without raising through the RPC boundary unless desired.

Usage example:

    from hypha_rpc import connect_to_server
    from biomni.hypha_service import build_schema_functions, register_all_tools
    import asyncio

    async def main():
        server = await connect_to_server(
            {"server_url": "http://localhost:9000", "token": "YOUR_TOKEN"}
        )
        tool_funcs = build_schema_functions()
        await register_all_tools(
            server,
            service_id="biomni-tools",
            service_name="Biomni Tool Service",
            functions=tool_funcs,
            visibility="public",
        )

    asyncio.run(main())

"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import logging
import os
import sys
import uuid
from functools import lru_cache
from typing import TYPE_CHECKING, Any

# Fix for Matplotlib crash on macOS when running in background threads
import matplotlib

matplotlib.use("Agg")

from dotenv import load_dotenv
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function

from biomni import affinity_capture_rna_tools, mirdb_tools
from biomni.config import default_config
from biomni.tool.tool_registry import ToolRegistry
from biomni.utils import DatasetTuple, download_files, read_module2api

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_REMOTE_URL = "https://hypha.aicell.io"
DEFAULT_SERVICE_ID = "biomni-test"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_registry() -> ToolRegistry:
    """Return a populated ToolRegistry from description modules."""
    return ToolRegistry(read_module2api())


@lru_cache(maxsize=1024)
def _resolve_tool_callable(tool_name: str) -> Callable[..., Any]:
    """Import the Python function implementing a tool.

    The implementation lives in a module whose name matches the domain in the
    tool description path (e.g., biomni.tool.biochemistry). We search the
    description maps for which module listed the name.
    """
    module2api = read_module2api()  # { module_path: [api_dict, ...] }
    for module_path, api_list in module2api.items():
        for api in api_list:
            if api["name"] == tool_name:
                impl_module_name = module_path.replace("biomni.tool.", "biomni.tool.")
                try:
                    impl_module = importlib.import_module(impl_module_name)
                except (
                    Exception
                ) as e:  # Capture import errors (missing third-party deps)
                    error_msg = (
                        f"Failed importing module {impl_module_name} for "
                        f"{tool_name}: {e}"
                    )
                    raise ImportError(error_msg) from e
                fn = getattr(impl_module, tool_name, None)
                if fn is None:
                    error_msg = f"Function {tool_name} not found in {impl_module_name}"
                    raise AttributeError(error_msg)
                return fn
    error_msg = f"Tool '{tool_name}' not found in any registered module."
    raise ValueError(error_msg)


def _test_mode_stub_result(tool_name: str, kwargs: dict[str, object]) -> object:
    """Return a fast, JSON-serializable placeholder result for test mode.

    This is used to keep the remote service responsive under the test harness'
    strict per-test timeout, and to avoid importing heavy optional deps.
    """
    if tool_name in {
        "analyze_mitochondrial_morphology_and_potential",
        "analyze_protein_colocalization",
        "segment_and_analyze_microbial_cells",
    }:
        return {
            "log": f"[test mode] stubbed {tool_name}",
            "inputs": {k: str(v)[:200] for k, v in kwargs.items()},
        }

    if tool_name == "analyze_flow_cytometry_immunophenotyping":
        return {
            "log": "[test mode] placeholder flow cytometry analysis (no real FCS parsing)",
            "populations": {
                "CD4+ T cells": {"count": 0, "percent": 0.0},
                "CD8+ T cells": {"count": 0, "percent": 0.0},
            },
        }

    if tool_name == "docking_autodock_vina":
        smiles_list = kwargs.get("smiles_list")
        if not isinstance(smiles_list, list):
            smiles_list = []
        return {
            "log": "[test mode] docking skipped",
            "results": [
                {
                    "smiles": str(s),
                    "best_affinity_kcal_mol": -7.0,
                }
                for s in smiles_list
            ],
        }

    if tool_name == "analyze_xenograft_tumor_growth_inhibition":
        return {
            "log": "[test mode] xenograft analysis skipped",
            "summary": {"tgi_percent": 0.0},
        }

    # Default: quick string (always JSON-serializable)
    return f"[test mode] stubbed {tool_name}"


# ---------------------------------------------------------------------------
# Public API: building schema functions
# ---------------------------------------------------------------------------

_ALLOWED_SIMPLE_TYPES = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "List[str]": list,
    "List[int]": list,
}


def _sanitize_param(p: dict) -> dict:
    """Normalize parameter while preserving original type in its description.

    We add the original (possibly composite) type string to the description and
    map the exposed simple "type" to a core JSON-compatible primitive so that
    Hypha's schema extraction keeps the parameter present.
    """
    new_p = dict(p)
    raw_type = (new_p.get("type") or "").strip()
    desc = (new_p.get("description") or "").strip()
    if raw_type:
        marker = f"(type: {raw_type})"
        if marker not in desc:
            if desc and not desc.endswith("."):
                desc += "."
            desc = f"{desc} {marker}".strip()
    composite = " or " in raw_type
    has_numpy = "numpy" in raw_type.lower()
    unsupported = (
        composite or has_numpy or (raw_type and raw_type not in _ALLOWED_SIMPLE_TYPES)
    )
    if unsupported and raw_type:
        if "list" in raw_type.lower():
            fallback = "list"
        else:
            for t in ["int", "float", "str", "bool"]:
                if t in raw_type.lower():
                    fallback = t
                    break
            else:  # pragma: no cover
                fallback = "str"
        new_p["type"] = fallback
    elif raw_type == "":
        new_p["type"] = "str"
    new_p["description"] = desc
    new_p["original_type"] = raw_type or new_p["type"]
    return new_p


def _sanitize_params(
    required: list[dict],
    optional: list[dict],
) -> tuple[list[dict], list[dict]]:
    return [*_iter_sanitize(required)], [*_iter_sanitize(optional)]


def _iter_sanitize(params: list[dict]) -> Iterator[dict]:  # generator helper
    for p in params:
        yield _sanitize_param(p)


def _build_docstring(tool_name: str, spec: dict) -> str:
    return (spec.get("description") or tool_name).strip()


def _json_type(py_type: str) -> tuple[str, dict | None]:
    mapping: dict[str, tuple[str, dict | None]] = {
        "int": ("integer", None),
        "float": ("number", None),
        "str": ("string", None),
        "bool": ("boolean", None),
        "dict": ("object", None),
        "List[int]": ("array", {"type": "integer"}),
        "List[str]": ("array", {"type": "string"}),
        "list": ("array", None),
    }
    return mapping.get(py_type, ("string", None))


def _build_parameter_schema(required: list[dict], optional: list[dict]) -> dict:
    props: dict[str, dict] = {}
    required_names: list[str] = []
    for group, is_required in ((required, True), (optional, False)):
        for p in group:
            name = p["name"]
            py_type = p.get("type", "str")
            json_type, items = _json_type(py_type)
            prop: dict[str, Any] = {
                "type": json_type,
                "description": p.get("description", name),
            }
            if items:
                prop["items"] = items
            if not is_required and "default" in p:
                prop["default"] = p.get("default")
            if is_required:
                required_names.append(name)
            props[name] = prop
    return {"type": "object", "properties": props, "required": required_names}


def _create_async_function(  # noqa: PLR0913 accepts several parts for clarity
    tool_name: str,
    impl: Callable[..., Any] | None,
    required: list[dict],
    optional: list[dict],
    func_description: str,
    param_schema: dict,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    async def _wrapper(
        _impl: Callable[..., Any] | None = impl,
        _tool: str = tool_name,
        **kwargs: object,
    ) -> dict[str, Any]:
        mode = os.getenv("BIOMNI_TEST_MODE", "").strip()
        if mode == "1":
            print(f"DEBUG: Tool {_tool} called in TEST MODE (1)")
            return _test_mode_stub_result(_tool, kwargs)

        if _impl is None:
            _impl = _resolve_tool_callable(_tool)

        impl_params = inspect.signature(_impl).parameters
        call_kwargs = {k: v for k, v in kwargs.items() if k in impl_params}
        timeout_raw = os.getenv("BIOMNI_TOOL_TIMEOUT_SECONDS", "")
        timeout_s: float | None
        if timeout_raw.strip() == "":
            timeout_s = None
        else:
            try:
                timeout_s = float(timeout_raw)
            except ValueError:
                timeout_s = None

        async def _run() -> Any:
            if inspect.iscoroutinefunction(_impl):
                return await _impl(**call_kwargs)
            return await asyncio.to_thread(_impl, **call_kwargs)

        try:
            if timeout_s is not None and timeout_s > 0:
                return await asyncio.wait_for(_run(), timeout=timeout_s)
            return await _run()
        except TimeoutError:
            raise TimeoutError(
                f"Tool '{_tool}' timed out after {timeout_s:.1f}s",
            )
        except Exception as exc:
            raise exc

    arg_parts = [p["name"] for p in required]
    for p in optional:
        default_repr = repr(p.get("default"))
        arg_parts.append(f"{p['name']}={default_repr}")
    arg_sig = ", ".join(arg_parts)
    src = [
        f"async def generated_func({arg_sig}):",
        "    _locals = locals()",
        "    return await _base(**_locals)",
    ]
    namespace = {"_base": _wrapper}
    exec("\n".join(src), namespace)  # nosec  # noqa: S102
    gf = namespace["generated_func"]
    gf.__name__ = tool_name
    gf.__doc__ = func_description
    annotations = {}
    for p in required + optional:
        t = p.get("type", "str")
        annotations[p["name"]] = _ALLOWED_SIMPLE_TYPES.get(t, str)
    gf.__annotations__ = annotations  # type: ignore[attr-defined]
    return schema_function(
        gf,
        name=tool_name,
        description=func_description,
        parameters=param_schema,
    )


def _unavailable_function(
    tool_name: str,
    error: str,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    async def _unavailable() -> dict[str, Any]:  # no user parameters
        raise FileNotFoundError(error)

    return schema_function(
        _unavailable,
        name=tool_name,
        description=f"Unavailable tool placeholder for {tool_name} (import error).",
        parameters={"type": "object", "properties": {}, "required": []},
    )


def build_schema_functions() -> dict[str, Callable[..., Awaitable[dict[str, Any]]]]:
    """Build async schema-wrapped functions for every Biomni tool.

    Composite / unsupported parameter types are collapsed into descriptions so
    Hypha schema extraction sees only simple core Python types.
    """
    registry = _get_registry()
    module2api = read_module2api()
    tool_specs: dict[str, dict] = {
        spec["name"]: spec for api in module2api.values() for spec in api
    }
    out: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {}
    for meta in registry.list_tools():
        name = meta["name"]
        spec = tool_specs.get(name)
        if not spec:
            continue
        req_raw = spec.get("required_parameters", [])
        opt_raw = spec.get("optional_parameters", [])
        req, opt = _sanitize_params(req_raw, opt_raw)
        func_desc = _build_docstring(name, spec)
        param_schema = _build_parameter_schema(req, opt)
        out[name] = _create_async_function(
            name,
            None,
            req,
            opt,
            func_desc,
            param_schema,
        )
    return out


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------
async def register_all_tools(  # noqa: PLR0913 public API needs explicit args
    *,
    workspace: str,
    client_id: str | None = None,
    service_id: str = DEFAULT_SERVICE_ID,
    server_url: str = DEFAULT_REMOTE_URL,
    service_name: str = "Biomni Tool Service",
    functions: dict[str, Callable[..., Awaitable[dict[str, Any]]]] | None = None,
    visibility: str = "public",
    extra_config: dict[str, Any] | None = None,
) -> None:
    """Register all tool schema functions as one Hypha service."""
    load_dotenv(override=True)

    # The test harness expects fast, bounded tool execution. Only apply these
    # defaults in the remote-service process (Docker) and allow users to
    # override via env vars.
    if service_id == DEFAULT_SERVICE_ID:
        os.environ.setdefault("BIOMNI_TEST_MODE", "0")
        os.environ.setdefault("BIOMNI_TOOL_TIMEOUT_SECONDS", "15")

    print(f"DEBUG: BIOMNI_TEST_MODE is set to: {os.environ.get('BIOMNI_TEST_MODE')}")
    logger.info(f"BIOMNI_TEST_MODE is set to: {os.environ.get('BIOMNI_TEST_MODE')}")

    datasets = [
        DatasetTuple(
            artifact_alias="affinity_capture-ms",
            file_path="affinity_capture-ms.parquet",
        ),
        DatasetTuple(
            artifact_alias="affinity_capture-rna",
            file_path="affinity_capture-rna.parquet",
        ),
        DatasetTuple(
            artifact_alias="mirdb",
            file_path="miRDB_v6.0_results.parquet",
        ),
    ]

    # Don't block service registration on dataset downloads; this can take
    # longer than the harness' startup delay.
    download_task = asyncio.create_task(download_files(datasets))

    def _log_download_result(task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dataset prefetch failed: %s", exc)

    download_task.add_done_callback(_log_download_result)

    server_config = {
        "server_url": server_url,
        "workspace": workspace,
        "token": os.getenv("HYPHA_TOKEN"),
    }

    resolved_client_id = (
        client_id
        or os.getenv("HYPHA_CLIENT_ID")
        or f"{service_id}-{uuid.uuid4().hex[:8]}"
    )
    server_config["client_id"] = resolved_client_id

    server = await connect_to_server(server_config)

    if functions is None:
        functions = build_schema_functions()

    # Merge in miRDB custom functions (outside dynamic tool registry)
    try:  # pragma: no cover - defensive path
        functions.update(mirdb_tools.get_mirdb_schema_functions())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed loading miRDB functions: %s", exc)

    # Merge in affinity capture RNA dataset functions
    try:  # pragma: no cover - defensive path
        functions.update(
            affinity_capture_rna_tools.get_affinity_capture_rna_schema_functions(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed loading affinity capture RNA functions: %s",
            exc,
        )

    config = {"visibility": visibility}
    if extra_config:
        config.update(extra_config)

    payload = {"id": service_id, "name": service_name, "config": config}
    payload.update(functions)

    await server.register_service(payload)  # type: ignore[attr-defined]

    server_workspace = server.config.workspace

    log_msg = (
        f"Service '{service_name}' registered with ID"
        f" '{service_id}' in workspace '{server_workspace}'."
    )
    logger.info(log_msg)

    log_msg2 = f"Access it at {server_url}/{server_workspace}/services/{service_id}"
    logger.info(log_msg2)

    await server.serve()


# ---------------------------------------------------------------------------
# CLI for quick inspection
# ---------------------------------------------------------------------------


def _cli_list() -> None:
    funcs = build_schema_functions()
    logger.info("Total tools wrapped: %d", len(funcs))
    num_tools_display = 25
    for i, name in enumerate(sorted(funcs.keys())):
        if i < num_tools_display:
            logger.info(" - %s", name)
        elif i == num_tools_display:
            logger.info(" ... (truncated)")
            break


if __name__ == "__main__":  # pragma: no cover
    _cli_list()
    parser = argparse.ArgumentParser(description="Aria tools launch commands.")

    subparsers = parser.add_subparsers()

    parser_schema = subparsers.add_parser("schema", help="Dump a tool's schema JSON")
    parser_schema.add_argument("tool", type=str, help="Tool name to inspect")
    parser_schema.set_defaults(func="_dump_schema")

    parser_remote = subparsers.add_parser("remote")
    parser_remote.add_argument("--server-url", type=str, default=DEFAULT_REMOTE_URL)
    parser_remote.add_argument(
        "--service-id",
        type=str,
        default=DEFAULT_SERVICE_ID,
    )
    parser_remote.add_argument(
        "--client-id",
        type=str,
        default=None,
        help="Client ID for the remote connection",
    )
    parser_remote.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace to register the service under",
    )
    parser_remote.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="LLM model to set in the default config",
    )
    parser_remote.set_defaults(func=register_all_tools)

    args = parser.parse_args()

    default_config.llm = args.model

    if getattr(args, "func", None) == "_dump_schema":
        # Synchronous path: just dump schema and exit
        all_funcs = build_schema_functions()
        tool_name = args.tool
        fn = all_funcs.get(tool_name)
        if not fn:
            logger.error("Tool '%s' not found.", tool_name)
            sys.exit(1)
        schema = getattr(fn, "__schema__", None)
        if not schema:
            logger.error("Schema not available for '%s'.", tool_name)
            sys.exit(1)
        import json as _json

        print(_json.dumps(schema, indent=2))  # noqa: T201 (intentional CLI output)
        sys.exit(0)

    if not hasattr(args, "func"):
        logger.warning("No subcommand provided; append 'remote' or 'schema'.")
        sys.exit(1)

    asyncio.run(
        args.func(
            workspace=args.workspace,
            client_id=args.client_id,
            service_id=args.service_id,
            server_url=args.server_url,
        ),
    )
