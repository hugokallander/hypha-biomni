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
import traceback
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function

from biomni.tool.tool_registry import ToolRegistry
from biomni.utils import read_module2api

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
    impl: Callable[..., Any],
    required: list[dict],
    optional: list[dict],
    func_description: str,
    param_schema: dict,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    async def _wrapper(
        _impl: Callable[..., Any] = impl,
        _tool: str = tool_name,
        **kwargs: object,
    ) -> dict[str, Any]:
        try:
            impl_params = inspect.signature(_impl).parameters
            call_kwargs = {k: v for k, v in kwargs.items() if k in impl_params}
            result = _impl(**call_kwargs)
        except Exception as err:  # noqa: BLE001 pragma: no cover
            logger.debug(
                "Exception in tool '%s': %s\n%s",
                _tool,
                err,
                traceback.format_exc(),
            )
            return {"tool": _tool, "ok": False, "error": str(err)}
        return {"tool": _tool, "ok": True, "result": result}

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
        parameters={
            "name": tool_name,
            "description": func_description,
            "parameters": param_schema,
        },
    )


def _unavailable_function(
    tool_name: str,
    error: str,
) -> Callable[..., Awaitable[dict[str, Any]]]:
    async def _unavailable(
        err: str = error,
        tool: str = tool_name,
        **_: object,
    ) -> dict[str, Any]:
        return {"tool": tool, "ok": False, "error": err}

    return schema_function(_unavailable)


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
        try:
            impl = _resolve_tool_callable(name)
        except Exception as exc:  # noqa: BLE001 import or attribute error
            logger.warning("Skipping tool '%s' due to import error: %s", name, exc)
            out[name] = _unavailable_function(name, f"unavailable: {exc}")
            continue
        req_raw = spec.get("required_parameters", [])
        opt_raw = spec.get("optional_parameters", [])
        req, opt = _sanitize_params(req_raw, opt_raw)
        func_desc = _build_docstring(name, spec)
        param_schema = _build_parameter_schema(req, opt)
        out[name] = _create_async_function(
            name,
            impl,
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
    load_dotenv()

    server_config = {
        "server_url": server_url,
        "workspace": workspace,
        "token": os.getenv("HYPHA_TOKEN"),
    }

    if client_id:
        server_config["client_id"] = client_id

    server = await connect_to_server(server_config)

    if functions is None:
        functions = build_schema_functions()

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
    parser_remote.set_defaults(func=register_all_tools)

    args = parser.parse_args()

    loop = asyncio.get_event_loop()

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

    try:
        task = loop.create_task(
            args.func(
                workspace=args.workspace,
                client_id=args.client_id,
                service_id=args.service_id,
                server_url=args.server_url,
            ),
        )
    except AttributeError:
        logger.warning("No subcommand provided; append 'remote' or 'schema'.")
        sys.exit(1)

    tasks = set()
    tasks.add(task)
    task.add_done_callback(tasks.discard)

    loop.run_forever()
