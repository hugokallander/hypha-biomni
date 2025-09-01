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
import traceback
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from hypha_rpc import connect_to_server
from hypha_rpc.utils.schema import schema_function

from biomni.tool.tool_registry import ToolRegistry
from biomni.utils import read_module2api

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_REMOTE_URL = "https://hypha.aicell.io"
DEFAULT_SERVICE_ID = "biomni-tools"

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
                    error_msg = f"Failed importing module {impl_module_name} for {tool_name}: {e}"
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


def build_schema_functions() -> dict[str, Callable[..., Awaitable[dict[str, Any]]]]:
    """Build async schema-wrapped functions for every Biomni tool.

    Returns:
        Dict mapping tool name -> async callable decorated with @schema_function.

    """
    registry = _get_registry()
    module2api = read_module2api()
    tool_specs: dict[str, dict] = {}
    for api_list in module2api.values():
        for spec in api_list:
            tool_specs[spec["name"]] = spec

    name2func: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {}

    for meta in registry.list_tools():
        tool_name = meta["name"]
        spec = tool_specs.get(tool_name)
        if not spec:
            continue
        try:
            impl = _resolve_tool_callable(tool_name)
        except Exception as exc:  # capture as exc for closure stability
            err_msg = f"unavailable: {exc}"
            logger.warning("Skipping tool '%s' due to import error: %s", tool_name, exc)

            async def _unavailable(
                err: str = err_msg,
                tool: str = tool_name,
                **kw: Any,
            ) -> dict[str, Any]:
                return {"tool": tool, "ok": False, "error": err}

            name2func[tool_name] = schema_function(_unavailable)
            continue

        required = spec.get("required_parameters", [])
        optional = spec.get("optional_parameters", [])

        async def _wrapper(
            _impl: Callable[..., Any] = impl,
            _tool: str = tool_name,
            **kwargs: Any,
        ) -> dict[str, Any]:
            try:
                impl_params = inspect.signature(_impl).parameters
                call_kwargs = {k: v for k, v in kwargs.items() if k in impl_params}
                result = _impl(**call_kwargs)
            except Exception as err:  # pragma: no cover
                logger.debug(
                    "Exception in tool '%s': %s\n%s",
                    _tool,
                    err,
                    traceback.format_exc(),
                )
                return {"tool": _tool, "ok": False, "error": str(err)}
            else:
                return {"tool": _tool, "ok": True, "result": result}

        arg_defs = [p["name"] for p in required]

        for p in optional:
            default_repr = repr(p.get("default", None))
            arg_defs.append(f"{p['name']}={default_repr}")
        arg_sig = ", ".join(arg_defs)
        src = ["async def generated_func(" + arg_sig + "):"]
        src.append("    _locals = locals()")
        src.append("    return await _base(**_locals)")
        namespace = {"_base": _wrapper}
        code = "\n".join(src)
        exec(code, namespace)  # nosec
        gf = namespace["generated_func"]
        gf.__name__ = tool_name
        gf.__doc__ = spec.get("description", tool_name)
        schema_func = schema_function(gf)
        name2func[tool_name] = schema_func

    return name2func


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------
async def register_all_tools(
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

    server = await connect_to_server(
        {
            "server_url": server_url,
            "workspace": workspace,
            "token": os.getenv("HYPHA_TOKEN"),
        },
    )

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

    task = loop.create_task(
        args.func(
            workspace=args.workspace,
            client_id=args.client_id,
            service_id=args.service_id,
            server_url=args.server_url,
        ),
    )

    tasks = set()
    tasks.add(task)
    task.add_done_callback(tasks.discard)

    loop.run_forever()
