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

import importlib
import inspect
import logging
import traceback
from typing import TYPE_CHECKING, Any

from hypha_rpc.utils.schema import schema_function

from biomni.tool.tool_registry import ToolRegistry
from biomni.utils import read_module2api

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

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
                    raise ImportError(
                        f"Failed importing module {impl_module_name} for {tool_name}: {e}",
                    ) from e
                fn = getattr(impl_module, tool_name, None)
                if fn is None:
                    raise AttributeError(
                        f"Function {tool_name} not found in {impl_module_name}",
                    )
                return fn
    raise ValueError(f"Could not locate implementation for tool '{tool_name}'")


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

            async def _unavailable(_err=err_msg, _tool=tool_name, **_kw):  # type: ignore[no-untyped-def]
                return {"tool": _tool, "ok": False, "error": _err}

            name2func[tool_name] = schema_function(_unavailable)  # type: ignore
            continue

        required = spec.get("required_parameters", [])
        optional = spec.get("optional_parameters", [])

        async def _wrapper(_impl=impl, _tool=tool_name, **kwargs):  # type: ignore[no-untyped-def]
            try:
                impl_params = inspect.signature(_impl).parameters
                call_kwargs = {k: v for k, v in kwargs.items() if k in impl_params}
                result = _impl(**call_kwargs)
                return {"tool": _tool, "ok": True, "result": result}
            except Exception as err:  # pragma: no cover
                logger.debug(
                    "Exception in tool '%s': %s\n%s",
                    _tool,
                    err,
                    traceback.format_exc(),
                )
                return {"tool": _tool, "ok": False, "error": str(err)}

        arg_defs = []
        for p in required:
            arg_defs.append(p["name"])
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
        schema_func = schema_function(gf)  # type: ignore
        name2func[tool_name] = schema_func

    return name2func


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------
async def register_all_tools(
    server: Any,
    service_id: str = "biomni-tools",
    service_name: str = "Biomni Tool Service",
    functions: dict[str, Callable[..., Awaitable[dict[str, Any]]]] | None = None,
    visibility: str = "public",
    extra_config: dict[str, Any] | None = None,
) -> None:
    """Register all tool schema functions as one Hypha service."""
    if functions is None:
        functions = build_schema_functions()

    config = {"visibility": visibility}
    if extra_config:
        config.update(extra_config)

    payload = {"id": service_id, "name": service_name, "config": config}
    payload.update(functions)

    await server.register_service(payload)  # type: ignore[attr-defined]


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
